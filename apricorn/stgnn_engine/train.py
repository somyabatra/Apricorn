# FILE: apricorn/stgnn_engine/train.py  (REWRITTEN)
# Member 2 - Training pipeline: imitation + congestion-aware loss, with validation.
#
#   python -m apricorn.stgnn_engine.train                      # generate data (if needed) and train
#   python -m apricorn.stgnn_engine.train --epochs 40 --device cuda
#   python -m apricorn.stgnn_engine.train --data-dir data/train --generate 0
#
# What the model learns (see labels.py):
#   IMITATION  - edges on Member 1's ground-truth routes get LOWER weight than other edges.
#   CONGESTION - edges that the real router later conflicts on get HIGHER weight;
#                nets that the router had to rip up most often get HIGHER priority.
# The GRU is trained on short sequences in which router feedback is fed back into the
# graph exactly as at inference time (data_graph.update_overflow + feedback.conflict_signal).

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from apricorn.contracts import GraphDataContract
from apricorn.stgnn_engine.model import ApricornSTGNN
from apricorn.stgnn_engine.heads import ApricornHeads
from apricorn.stgnn_engine.labels import (
    collect_rollout,
    graph_sequence,
    gt_edge_mask,
    priority_targets,
)
from apricorn.stgnn_engine.dataset import generate_mixed_dataset, list_boards, load_spec

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_to_tensors(graph: GraphDataContract, device: torch.device):
    x = torch.from_numpy(graph.x_nodes).to(device)
    edge_index = torch.from_numpy(graph.edge_index).to(device)
    edge_attr = torch.from_numpy(graph.edge_attr).to(device)
    return x, edge_index, edge_attr


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

@dataclass
class BoardSample:
    name: str
    graphs: List[GraphDataContract]   # what the model sees at step t (feedback from epochs < t)
    conflicts: List[np.ndarray]       # (E,) router conflicts produced by routing AFTER seeing graphs[t]
    gt_mask: np.ndarray               # (E,) bool - edges of the ground-truth routes
    prio: Dict[int, float]            # priority targets ({} = no information)


@dataclass
class Config:
    epochs: int = 25
    lr: float = 3e-4
    T: int = 6
    warmup_epochs: int = 5      # imitation only, before congestion terms kick in
    accum: int = 4              # boards per optimizer step
    pairs: int = 2048           # sampled (low, high) edge pairs per step
    margin: float = 0.5
    w_imit: float = 1.0
    w_cong: float = 1.0
    w_prio: float = 1.0
    w_reg: float = 1e-3
    val_frac: float = 0.1
    seed: int = GLOBAL_SEED


def build_samples(paths: List[str], T: int, cache_file: Optional[str], verbose: bool = True) -> List[BoardSample]:
    """Parse boards, build graphs, run the real router for T epochs (cached on disk)."""
    from apricorn.data_graph.graph_builder import build_graph

    cache: Dict[str, object] = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

    samples: List[BoardSample] = []
    t0 = time.time()
    dirty = False
    for i, p in enumerate(paths):
        spec = load_spec(p)
        graph = build_graph(spec)
        key = f"{os.path.basename(p)}|T{T}"
        ro = cache.get(key)
        if ro is None:
            ro = collect_rollout(graph, T=T)   # NOTE: neutral costs, fallback DRC (training signal only)
            cache[key] = ro
            dirty = True
        samples.append(
            BoardSample(
                name=os.path.basename(p),
                graphs=graph_sequence(graph, ro),
                conflicts=ro.conflict_seq,
                gt_mask=gt_edge_mask(graph, spec),
                prio=priority_targets(graph, ro),
            )
        )
        if verbose and (i + 1) % 50 == 0:
            print(f"[train] prepared {i + 1}/{len(paths)} boards ({time.time() - t0:.0f}s)")
    if cache_file and dirty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    return samples


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def pair_hinge(w: torch.Tensor, low_mask: torch.Tensor, high_mask: torch.Tensor, k: int, margin: float):
    """Want w[low] + margin <= w[high] for random (low, high) pairs.
    Returns (loss, pair_accuracy) or (None, None) when either set is empty."""
    low_idx = low_mask.nonzero(as_tuple=True)[0]
    high_idx = high_mask.nonzero(as_tuple=True)[0]
    if low_idx.numel() == 0 or high_idx.numel() == 0:
        return None, None
    lo = low_idx[torch.randint(low_idx.numel(), (k,), device=w.device)]
    hi = high_idx[torch.randint(high_idx.numel(), (k,), device=w.device)]
    diff = w[hi] - w[lo]
    return F.relu(margin - diff).mean(), (diff > 0).float().mean().detach()


def run_board(model, heads, s: BoardSample, device, cfg: Config, use_congestion: bool):
    """Forward one board's whole sequence. Returns (loss, metrics dict)."""
    N = s.graphs[0].x_nodes.shape[0]
    h = model.init_hidden(N, device)
    gt = torch.from_numpy(s.gt_mask).to(device)
    total = torch.zeros((), device=device)
    acc_gt, acc_cf, prio_corr, prio_mse = [], [], [], []

    for t, g in enumerate(s.graphs):
        x, ei, ea = graph_to_tensors(g, device)
        h = model(x, ei, ea, h)                       # GRU state carried through the sequence
        w = heads.edge_cost_head(h, ei, ea)           # (E,) >= 1.0

        step = cfg.w_reg * ((w - 1.0) ** 2).mean()

        l_imit, a_imit = pair_hinge(w, gt, ~gt, cfg.pairs, cfg.margin)
        if l_imit is not None:
            step = step + cfg.w_imit * l_imit
            acc_gt.append(a_imit.item())

        if use_congestion:
            conflict = torch.from_numpy(s.conflicts[t] > 0).to(device)
            l_c, a_c = pair_hinge(w, ~conflict, conflict, cfg.pairs, cfg.margin)
            if l_c is not None:
                step = step + cfg.w_cong * l_c
                acc_cf.append(a_c.item())

            if s.prio:
                ids, scores = heads.net_priority_head.forward_tensor(h, g.net_demands)
                if ids:
                    tgt = torch.tensor([s.prio[nid] for nid in ids], dtype=torch.float32, device=device)
                    mse = F.mse_loss(scores, tgt)
                    step = step + cfg.w_prio * mse
                    prio_mse.append(mse.item())
                    if len(ids) > 2 and scores.std() > 1e-6 and tgt.std() > 1e-6:
                        c = torch.corrcoef(torch.stack([scores.detach(), tgt]))[0, 1].item()
                        if c == c:   # not NaN
                            prio_corr.append(c)
        total = total + step

    total = total / len(s.graphs)
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return total, {
        "gt_pair_acc": mean(acc_gt),
        "conflict_pair_acc": mean(acc_cf),
        "prio_corr": mean(prio_corr),
        "prio_mse": mean(prio_mse),
    }


def _nanmean(values) -> float:
    vals = [v for v in values if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def validate(model, heads, samples: List[BoardSample], device, cfg: Config, use_congestion: bool):
    model.eval()
    heads.eval()
    losses, ms = [], []
    # Fixed sampled pairs every epoch (comparable numbers) WITHOUT disturbing the training RNG stream.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(1234)
        for s in samples:
            loss, m = run_board(model, heads, s, device, cfg, use_congestion)
            losses.append(loss.item())
            ms.append(m)
    agg = {k: _nanmean([m[k] for m in ms]) for k in ms[0]} if ms else {}
    return _nanmean(losses), agg


def _json_safe(obj):
    """NaN/Inf are not valid JSON (the browser's JSON.parse rejects them) -> None."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "data/train",
    generate: int = 300,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "auto",
    cfg: Optional[Config] = None,
    cache_file: Optional[str] = "data/rollout_cache.pkl",
    verbose: bool = True,
) -> str:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    if verbose:
        print(f"[train] device={device}")

    paths = list_boards(data_dir)
    if not paths and generate > 0:
        print(f"[train] no boards in '{data_dir}', generating {generate} mixed-difficulty boards...")
        paths = generate_mixed_dataset(data_dir, generate)
    if not paths:
        raise SystemExit(f"[train] no boards found in '{data_dir}' (use --generate N)")

    samples = build_samples(paths, cfg.T, cache_file, verbose)
    rng = random.Random(cfg.seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_frac))
    val, tr = samples[:n_val], samples[n_val:]
    if verbose:
        print(f"[train] {len(tr)} train boards, {len(val)} validation boards (never trained on)")

    model = ApricornSTGNN().to(device)
    heads = ApricornHeads().to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = optim.AdamW(params, lr=cfg.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "stgnn_v1.pt")
    log_path = os.path.join(checkpoint_dir, "training_log.json")
    history = []
    best_val, best_epoch = float("inf"), 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        use_cong = epoch > cfg.warmup_epochs
        model.train()
        heads.train()
        rng.shuffle(tr)
        run_loss, n = 0.0, 0
        opt.zero_grad()
        for i, s in enumerate(tr, 1):
            loss, _ = run_board(model, heads, s, device, cfg, use_cong)
            (loss / cfg.accum).backward()
            run_loss += loss.item()
            n += 1
            if i % cfg.accum == 0 or i == len(tr):
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        sched.step()

        val_loss, vm = validate(model, heads, val, device, cfg, use_cong)
        rec = {
            "epoch": epoch,
            "phase": "imitation+congestion" if use_cong else "imitation-warmup",
            "train_loss": run_loss / max(n, 1),
            "val_loss": val_loss,
            "val_gt_pair_acc": vm.get("gt_pair_acc"),
            "val_conflict_pair_acc": vm.get("conflict_pair_acc"),
            "val_prio_corr": vm.get("prio_corr"),
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)

        # Best checkpoint. Loss terms change when the congestion phase starts, so only compare
        # validation losses from the full objective (or from warmup if there is no later phase).
        eligible = use_cong or cfg.warmup_epochs >= cfg.epochs
        if eligible and val_loss == val_loss and val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "heads_state_dict": heads.state_dict(),
                    "loss": float(val_loss),
                    "seed": cfg.seed,
                },
                ckpt_path,
            )

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                _json_safe({"config": asdict(cfg), "device": str(device), "n_train": len(tr), "n_val": len(val),
                            "best_epoch": best_epoch, "best_val_loss": best_val, "history": history}),
                f, indent=2,
            )
        if verbose:
            fmt = lambda v: "  n/a" if v is None or v != v else f"{v:.3f}"
            print(f"[train] ep {epoch:3d}/{cfg.epochs} [{rec['phase']}] train={rec['train_loss']:.4f} val={val_loss:.4f} "
                  f"gt_acc={fmt(rec['val_gt_pair_acc'])} conf_acc={fmt(rec['val_conflict_pair_acc'])} "
                  f"prio_r={fmt(rec['val_prio_corr'])} ({rec['seconds']}s)")

    if not os.path.exists(ckpt_path):          # safety net: always leave a checkpoint behind
        torch.save({"epoch": cfg.epochs, "model_state_dict": model.state_dict(),
                    "heads_state_dict": heads.state_dict(), "loss": float(val_loss), "seed": cfg.seed}, ckpt_path)
    if verbose:
        print(f"[train] done. best epoch={best_epoch} val_loss={best_val:.4f}  ->  {ckpt_path}")
        print("[train] next: python -m apricorn.stgnn_engine.evaluate   (proves whether it beats the no-AI baseline)")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Apricorn STGNN")
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--generate", type=int, default=300, help="boards to generate if data-dir is empty")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=6, help="router epochs per training sequence")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--cache", default="data/rollout_cache.pkl")
    a = ap.parse_args()
    cfg = Config(epochs=a.epochs, lr=a.lr, T=a.T, warmup_epochs=min(a.warmup_epochs, a.epochs))
    train(a.data_dir, a.generate, a.checkpoint_dir, a.device, cfg, a.cache)


if __name__ == "__main__":
    main()
