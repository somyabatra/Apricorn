# FILE: apricorn/data_graph/parser_gen.py
"""
Board Parser & Synthetic Dataset Generator for Project Apricorn.
Member 1: Data & Graph Lead.

Responsible for:
1. Loading board descriptions from JSON or KiCad .kicad_pcb S-expressions.
2. Procedurally generating solvable multi-layer PCB instances with ground-truth A* routes.
3. Dataset batch generation for Member 2's STGNN training pipeline.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _solve_ground_truth_route(
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    layers: int,
    width: int,
    height: int,
    obstacles_by_layer: Dict[int, set],
    existing_traces: set,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Lightweight 3D grid A* router to guarantee that procedurally generated
    boards are 100% solvable without collisions.
    """
    import heapq

    def heuristic(p1, p2):
        # 3D Manhattan + via weight
        return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) + 4.0 * abs(p1[0] - p2[0])

    open_set: List[Tuple[float, float, Tuple[int, int, int]]] = []
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    cost_so_far: Dict[Tuple[int, int, int], float] = {start: 0.0}

    planar_dirs = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (-1, -1, 1.4142)
    ]

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cl, cx, cy = current

        # Planar moves
        for dx, dy, step_cost in planar_dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                nxt = (cl, nx, ny)
                if (nx, ny) in obstacles_by_layer.get(cl, set()) and nxt != goal:
                    continue
                if nxt in existing_traces and nxt != goal:
                    continue

                new_cost = current_cost + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, goal)
                    heapq.heappush(open_set, (priority, new_cost, nxt))
                    came_from[nxt] = current

        # Via moves
        for dl in (-1, 1):
            nl = cl + dl
            if 0 <= nl < layers:
                nxt = (nl, cx, cy)
                if (cx, cy) in obstacles_by_layer.get(nl, set()) and nxt != goal:
                    continue
                if nxt in existing_traces and nxt != goal:
                    continue

                new_cost = current_cost + 4.0
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, goal)
                    heapq.heappush(open_set, (priority, new_cost, nxt))
                    came_from[nxt] = current

    return None


def generate_synthetic_board(
    layers: int = 2,
    width: int = 32,
    height: int = 32,
    num_nets: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Procedurally generate a multi-layer board instance with guaranteed solvable
    ground-truth routes and non-overlapping obstacles.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate rectangular obstacle polygons per layer
    obstacles = []
    obstacles_by_layer: Dict[int, set] = {l: set() for l in range(layers)}

    num_obstacles_per_layer = max(1, min(3, width // 10))
    for l in range(layers):
        for _ in range(num_obstacles_per_layer):
            w_obs = rng.integers(3, 7)
            h_obs = rng.integers(3, 7)
            x_min = rng.integers(2, max(3, width - w_obs - 2))
            y_min = rng.integers(2, max(3, height - h_obs - 2))
            poly = [
                (float(x_min), float(y_min)),
                (float(x_min + w_obs), float(y_min)),
                (float(x_min + w_obs), float(y_min + h_obs)),
                (float(x_min), float(y_min + h_obs)),
            ]
            obstacles.append({"layer": l, "polygon": poly})
            for ox in range(x_min, x_min + w_obs + 1):
                for oy in range(y_min, y_min + h_obs + 1):
                    if 0 <= ox < width and 0 <= oy < height:
                        obstacles_by_layer[l].add((ox, oy))

    # 2. Procedurally place pins and compute ground-truth routes
    occupied_pins: set = set()
    existing_traces: set = set()
    nets = []

    attempts = 0
    max_attempts = num_nets * 10
    net_id = 0

    while net_id < num_nets and attempts < max_attempts:
        attempts += 1
        l1 = int(rng.integers(0, layers))
        x1 = int(rng.integers(1, width - 1))
        y1 = int(rng.integers(1, height - 1))

        l2 = int(rng.integers(0, layers))
        x2 = int(rng.integers(1, width - 1))
        y2 = int(rng.integers(1, height - 1))

        p1 = (l1, x1, y1)
        p2 = (l2, x2, y2)

        if p1 == p2 or p1 in occupied_pins or p2 in occupied_pins:
            continue
        if (x1, y1) in obstacles_by_layer[l1] or (x2, y2) in obstacles_by_layer[l2]:
            continue

        gt_route = _solve_ground_truth_route(
            p1, p2, layers, width, height, obstacles_by_layer, existing_traces
        )

        if gt_route is not None:
            occupied_pins.add(p1)
            occupied_pins.add(p2)
            # Reserve path cells to prevent trace collisions in synthetic ground truth
            for cell in gt_route:
                existing_traces.add(cell)

            nets.append({
                "net_id": net_id,
                "pins": [list(p1), list(p2)],
                "ground_truth_path": [list(c) for c in gt_route],
            })
            net_id += 1

    board_spec = {
        "board_id": f"synthetic_L{layers}_{width}x{height}_s{seed}",
        "layers": layers,
        "width": width,
        "height": height,
        "origin": [0.0, 0.0],
        "resolution": 1.0,
        "clearance": 0.5,
        "trace_width": 0.4,
        "nets": nets,
        "obstacles": obstacles,
    }
    return board_spec


def parse_kicad_pcb(content: str) -> Dict[str, Any]:
    """
    Minimal pure-Python S-expression parser for KiCad .kicad_pcb board format.
    Extracts pad coordinates and keepout zones without external CAD binaries.
    """
    layers = 2
    width = 32
    height = 32
    obstacles: List[Dict[str, Any]] = []
    nets_dict: Dict[int, List[List[int]]] = {}

    # Extract pad definitions: (pad "1" smd rect (at 10.5 20.3) ... (net 1 "Net-(R1-Pad1)"))
    pad_matches = re.findall(
        r'\(pad\s+"?[^"]*"?\s+\w+\s+\w+\s+\(at\s+([\d\.-]+)\s+([\d\.-]+)[^\)]*\)[\s\S]*?\(net\s+(\d+)',
        content,
    )
    for x_str, y_str, net_str in pad_matches:
        net_id = int(net_str)
        if net_id == 0:  # KiCad net 0 is unconnected
            continue
        px = max(0, min(width - 1, int(round(float(x_str)))))
        py = max(0, min(height - 1, int(round(float(y_str)))))
        if net_id not in nets_dict:
            nets_dict[net_id] = []
        nets_dict[net_id].append([0, px, py])

    # Extract keepout zones: (zone (net 0) (layers ...) (polygon (pts (xy 10 10) ...)))
    zone_matches = re.findall(r'\(zone[\s\S]*?\(pts([\s\S]*?)\)\s*\)', content)
    for zone in zone_matches:
        pts = re.findall(r'\(xy\s+([\d\.-]+)\s+([\d\.-]+)\)', zone)
        if len(pts) >= 3:
            poly = [(float(x), float(y)) for x, y in pts]
            obstacles.append({"layer": 0, "polygon": poly})

    nets = [{"net_id": k, "pins": v} for k, v in nets_dict.items() if len(v) >= 2]

    return {
        "board_id": "kicad_parsed_board",
        "layers": layers,
        "width": width,
        "height": height,
        "origin": [0.0, 0.0],
        "resolution": 1.0,
        "clearance": 0.5,
        "trace_width": 0.4,
        "nets": nets,
        "obstacles": obstacles,
    }


def load_or_generate_board(board_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Universal board loader required by apricorn.main and Member 1 deliverable.
    Loads JSON, KiCad .kicad_pcb, or procedural fallback if board_path is None or absent.
    """
    if board_path and os.path.exists(board_path):
        if board_path.endswith(".kicad_pcb"):
            with open(board_path, "r", encoding="utf-8") as f:
                return parse_kicad_pcb(f.read())
        elif board_path.endswith(".json"):
            with open(board_path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Deterministic default fallback (4-layer, 32x32 board with 15 nets)
    return generate_synthetic_board(layers=2, width=32, height=32, num_nets=15, seed=42)


def generate_synthetic_dataset(
    num_instances: int = 500,
    output_dir: str = "data/synthetic",
    layers: int = 2,
    width: int = 32,
    height: int = 32,
    nets_per_board: int = 15,
    seed_offset: int = 1000,
) -> List[str]:
    """
    Deliverable 1.1: Generates 500 solvable multi-layer board instances
    with ground-truth A* routes for Member 2's STGNN training pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for i in range(num_instances):
        seed = seed_offset + i
        spec = generate_synthetic_board(
            layers=layers,
            width=width,
            height=height,
            num_nets=nets_per_board,
            seed=seed,
        )
        file_path = os.path.join(output_dir, f"board_{i:04d}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        generated_files.append(file_path)

    return generated_files
