# FILE: apricorn/stgnn_engine/dataset.py  (numpy only)
"""
Board dataset helpers.

Member 1's default generator makes 32x32 boards with 15 nets, which the router
solves in a single epoch with ANY costs - there is nothing for an AI to improve.
`generate_mixed_dataset` therefore mixes in denser boards where negotiation is
actually needed (conflicts, rip-ups), which is where STGNN guidance can matter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

# (layers, width/height, nets requested)
DEFAULT_MIX: Sequence[Tuple[int, int, int]] = (
    (2, 16, 25),
    (2, 20, 30),
    (2, 20, 40),
    (2, 24, 45),
    (4, 24, 40),
)


def list_boards(data_dir: str) -> List[str]:
    return sorted(str(p) for p in Path(data_dir).glob("*.json"))


def generate_mixed_dataset(
    out_dir: str,
    n: int,
    seed_offset: int = 1000,
    mix: Sequence[Tuple[int, int, int]] = DEFAULT_MIX,
) -> List[str]:
    from apricorn.data_graph.parser_gen import generate_synthetic_board

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        layers, size, nets = mix[i % len(mix)]
        spec = generate_synthetic_board(
            layers=layers, width=size, height=size, num_nets=nets, seed=seed_offset + i
        )
        p = os.path.join(out_dir, f"board_{i:04d}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        paths.append(p)
    return paths


def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
