"""
Emulated and WebGPU cross-validation for the flat adaptive FMM kernel.

This validates that the flat layout (adaptive_gpu_metadata.py) and kernel
logic (adaptive_fmm.wgsl) evaluate the exact same analytical mathematics
as core.adaptive_fmm.TreeFreeElasticAdaptiveFMM across orders p=0..4.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from core.adaptive_fmm import (
    exact_direct_nbody_2d,
    exact_direct_nbody_forces_2d,
    p2m as adaptivefmm_p2m,
    m2m as adaptivefmm_m2m,
    m2l as adaptivefmm_m2l,
    l2l as adaptivefmm_l2l,
    p2l as adaptivefmm_p2l,
    m2p as adaptivefmm_m2p,
    l2p as adaptivefmm_l2p,
    l2p_force as adaptivefmm_l2p_force,
    p2p_potential_and_force,
)
from core.adaptive_gpu_metadata import build_flat_adaptive_metadata, FlatAdaptiveMetadata, INVALID


def _node_center(metadata: FlatAdaptiveMetadata, node: int) -> complex:
    cs = metadata.node_center_size[node]
    return complex(float(cs[0]), float(cs[1]))


def _node_depth(metadata: FlatAdaptiveMetadata, node: int) -> int:
    return int(metadata.node_center_size[node, 3])


def _is_terminal(metadata: FlatAdaptiveMetadata, node: int) -> bool:
    return bool(metadata.node_flags[node] & 1)


def evaluate_flat_adaptive_emulated(
    positions: np.ndarray,
    metadata: FlatAdaptiveMetadata,
    *,
    expansion_order: int = 2,
    charges: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the flat adaptive GPU schedule in NumPy to verify exactness.

    Mirrors the full adaptive FMM pipeline:
      1. P2M at terminal leaves
      2. M2M upward pass (bottom-up)
      3. Downward pass: L2L + List 2 M2L + List 4 P2L (top-down, level 1..max)
      4. Particle evaluation: L2P + List 3 M2P + List 1 P2P
    """
    n_nodes = metadata.node_count
    n_particles = len(positions)
    order = expansion_order
    if charges is None:
        charges = np.ones(n_particles, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)

    multipoles = np.zeros((n_nodes, order + 1), dtype=np.complex128)
    locals_ = np.zeros((n_nodes, order + 1), dtype=np.complex128)

    # ------------------------------------------------------------------
    # 1. P2M at terminal leaves
    # ------------------------------------------------------------------
    for node in range(n_nodes):
        start, count = metadata.node_particle_range[node]
        if count == 0:
            continue
        center = _node_center(metadata, node)
        idxs = metadata.particle_indices[start:start + count]
        pts = positions[idxs]
        q = charges[idxs]
        multipoles[node] = adaptivefmm_p2m(pts, q, center, order)

    # ------------------------------------------------------------------
    # 2. M2M upward pass (deepest level first)
    # ------------------------------------------------------------------
    max_depth = 0
    for node in range(n_nodes):
        d = _node_depth(metadata, node)
        if d > max_depth:
            max_depth = d

    # Build level -> node list mapping
    level_nodes: dict[int, list[int]] = {}
    for node in range(n_nodes):
        d = _node_depth(metadata, node)
        level_nodes.setdefault(d, []).append(node)

    for lvl in range(max_depth, -1, -1):
        for node in level_nodes.get(lvl, []):
            if _is_terminal(metadata, node):
                continue
            center = _node_center(metadata, node)
            acc = np.zeros(order + 1, dtype=np.complex128)
            for slot in range(4):
                ch = int(metadata.node_children[node, slot])
                if ch == INVALID:
                    continue
                ch_center = _node_center(metadata, ch)
                acc += adaptivefmm_m2m(multipoles[ch], ch_center, center, order)
            multipoles[node] = acc

    # ------------------------------------------------------------------
    # 3. Downward pass: L2L + List 2 M2L + List 4 P2L (level 1..max)
    # ------------------------------------------------------------------
    for lvl in range(1, max_depth + 1):
        for node in level_nodes.get(lvl, []):
            center = _node_center(metadata, node)
            parent = int(metadata.node_parent[node])
            if parent != INVALID:
                p_center = _node_center(metadata, parent)
                locals_[node] += adaptivefmm_l2l(
                    locals_[parent], p_center, center, order
                )

            # List 2: M2L
            for src in metadata.list_for(node, 1):
                src = int(src)
                src_center = _node_center(metadata, src)
                locals_[node] += adaptivefmm_m2l(
                    multipoles[src], src_center, center, order
                )

            # List 4: P2L (particles in distant large leaves)
            for src in metadata.list_for(node, 3):
                src = int(src)
                s_start, s_count = metadata.node_particle_range[src]
                if s_count == 0:
                    continue
                idxs = metadata.particle_indices[s_start:s_start + s_count]
                pts = positions[idxs]
                q = charges[idxs]
                locals_[node] += adaptivefmm_p2l(pts, q, center, order)

    # ------------------------------------------------------------------
    # 4. Particle evaluation: L2P + List 3 M2P + List 1 P2P
    # ------------------------------------------------------------------
    potentials = np.zeros(n_particles, dtype=np.float64)
    fx = np.zeros(n_particles, dtype=np.float64)
    fy = np.zeros(n_particles, dtype=np.float64)

    for i in range(n_particles):
        pos_c = complex(positions[i, 0], positions[i, 1])
        target = int(metadata.leaf_node_for_particle[i])
        tc = _node_center(metadata, target)

        # L2P: local expansion evaluation
        pot = adaptivefmm_l2p(locals_[target], pos_c, tc, order)
        potentials[i] += pot
        lfx, lfy = adaptivefmm_l2p_force(locals_[target], pos_c, tc, order)
        fx[i] += lfx
        fy[i] += lfy

        # List 3: M2P (distant small descendants of colleagues)
        for src in metadata.list_for(target, 2):
            src = int(src)
            sc = _node_center(metadata, src)
            pot_d, deriv_d = adaptivefmm_m2p(multipoles[src], sc, pos_c, order)
            potentials[i] += pot_d
            fx[i] -= deriv_d.real
            fy[i] += deriv_d.imag

        # List 1: direct P2P
        for source_node in metadata.list_for(target, 0):
            source_node = int(source_node)
            s_start, s_count = metadata.node_particle_range[source_node]
            if s_count == 0:
                continue
            for j in metadata.particle_indices[s_start:s_start + s_count]:
                j = int(j)
                if j == i:
                    continue
                diff = positions[i] - positions[j]
                r2 = max(float(diff[0] * diff[0] + diff[1] * diff[1]), 1e-12)
                qj = charges[j]
                potentials[i] += qj * 0.5 * math.log(r2)
                fx[i] -= qj * diff[0] / r2
                fy[i] -= qj * diff[1] / r2

    return potentials, fx, fy
