# FILE: apricorn/data_graph/drc_engine.py
"""
Pure-Software Geometric DRC Engine for Project Apricorn.
Member 1: Data & Graph Lead.

Enforces pure-software geometric Design Rule Checking (DRC) using Shapely
and R-Tree spatial indexing to detect trace-to-trace shorts, pad collisions,
and keepout violations in <5ms. Zero external CAD CLI or hardware dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

try:
    import rtree.index
    HAS_RTREE_PACKAGE = True
except ImportError:
    HAS_RTREE_PACKAGE = False

from apricorn.contracts import GraphDataContract


def _node_to_coord(node_idx: int, grid_shape: Tuple[int, int, int]) -> Tuple[int, float, float]:
    """Converts a 1D node index to (layer, x, y) coordinates."""
    _, X, Y = grid_shape
    layer = node_idx // (X * Y)
    rem = node_idx % (X * Y)
    x = float(rem // Y)
    y = float(rem % Y)
    return layer, x, y


def check_drc(
    graph: GraphDataContract,
    paths: Dict[int, List[int]],
    clearance: float = 0.5,
    trace_width: float = 0.4,
    return_violators: bool = False,
) -> Union[int, Tuple[int, Set[int]]]:
    """
    Evaluates DRC compliance for a set of routed net paths across all board layers.

    Args:
        graph: Canonical GraphDataContract containing board geometry & obstacles.
        paths: Mapping from net_id to ordered list of node indices forming routed traces.
        clearance: Minimum clearance in mm (default 0.5mm).
        trace_width: Trace width in mm (default 0.4mm).
        return_violators: If True, returns (total_violations, set_of_violating_net_ids).

    Returns:
        drc_violations: Total count of clearance breaches, shorts, and keepout violations.
                        0 indicates a clean, DRC-converged board.
    """
    if not paths:
        return (0, set()) if return_violators else 0

    grid_shape = graph.grid_shape
    layers = grid_shape[0]
    total_violations = 0
    violators: Set[int] = set()
    buffer_dist = (trace_width / 2.0) + (clearance / 2.0)

    # 1. Parse Obstacles per Layer
    obstacles_by_layer: Dict[int, List[Polygon]] = {l: [] for l in range(layers)}
    for obs_layer, poly_pts in graph.obstacle_polygons:
        if 0 <= obs_layer < layers and len(poly_pts) >= 3:
            obstacles_by_layer[obs_layer].append(Polygon(poly_pts))

    # 2. Extract Geometry Segments per Layer for each Net
    # layer -> list of (net_id, geometry_obj)
    layer_geoms: Dict[int, List[Tuple[int, Any]]] = {l: [] for l in range(layers)}

    # Track pins per net for pad-collision checks
    pin_locations: Dict[int, Set[Tuple[int, float, float]]] = {}
    for nid, terminals in graph.net_demands.items():
        pin_locations[nid] = set()
        for t in terminals:
            pin_locations[nid].add(_node_to_coord(t, grid_shape))

    for net_id, node_list in paths.items():
        if len(node_list) < 2:
            continue

        for i in range(len(node_list) - 1):
            u = node_list[i]
            v = node_list[i + 1]

            l1, x1, y1 = _node_to_coord(u, grid_shape)
            l2, x2, y2 = _node_to_coord(v, grid_shape)

            if l1 == l2:
                # Planar segment
                if x1 == x2 and y1 == y2:
                    geom = Point(x1, y1).buffer(buffer_dist)
                else:
                    geom = LineString([(x1, y1), (x2, y2)]).buffer(buffer_dist)
                layer_geoms[l1].append((net_id, geom))
            else:
                # Interlayer Via transition: add pad circle on both involved layers
                min_l, max_l = min(l1, l2), max(l1, l2)
                for vl in range(min_l, max_l + 1):
                    geom = Point(x1, y1).buffer(buffer_dist)
                    layer_geoms[vl].append((net_id, geom))

    # 3. Fast Spatial Index Verification per Layer using R-Tree (STRtree)
    for l in range(layers):
        geoms_with_meta = layer_geoms[l]
        if not geoms_with_meta:
            continue

        geoms = [g for _, g in geoms_with_meta]
        net_ids = [nid for nid, _ in geoms_with_meta]

        tree = STRtree(geoms)

        # Check Trace-to-Trace & Via collisions between different nets
        for i, (nid_i, geom_i) in enumerate(geoms_with_meta):
            # Query candidate intersections from spatial index
            candidate_indices = tree.query(geom_i)
            for j in candidate_indices:
                if j > i:  # Avoid double-counting and self-comparison
                    nid_j = net_ids[j]
                    if nid_i != nid_j:
                        # Collision between different nets!
                        if geom_i.intersects(geoms[j]):
                            total_violations += 1
                            violators.add(nid_i)
                            violators.add(nid_j)

        # Check Keepout / Obstacle Violations
        obs_polys = obstacles_by_layer[l]
        for obs in obs_polys:
            cand_indices = tree.query(obs)
            for c_idx in cand_indices:
                if obs.intersects(geoms[c_idx]):
                    total_violations += 1
                    violators.add(net_ids[c_idx])

    # 4. Check Trace-to-Foreign-Pin Shorts
    for l in range(layers):
        geoms_with_meta = layer_geoms[l]
        if not geoms_with_meta:
            continue
        geoms = [g for _, g in geoms_with_meta]
        tree = STRtree(geoms)

        for foreign_nid, pins in pin_locations.items():
            for pl, px, py in pins:
                if pl == l:
                    pad_pt = Point(px, py)
                    hits = tree.query(pad_pt)
                    for h in hits:
                        if geoms_with_meta[h][0] != foreign_nid:
                            if geoms[h].intersects(pad_pt):
                                total_violations += 1
                                violators.add(geoms_with_meta[h][0])

    if return_violators:
        return total_violations, violators
    return total_violations
