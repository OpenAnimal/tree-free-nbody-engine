"""Standardized variant benchmark for the physics contact broadphase.

Variants:
  standard     — brute-force O(N^2) tet-tet AABB overlap count (the exact
                 colliding-pair set)
  +elastichash — broadphase candidate-pair set via the existing CellIndex
                 ring-1 neighborhood code (AABB centers indexed in the
                 elastic hash; ring-1 cell neighborhood generates the
                 candidate pairs)

Accuracy semantics: a broadphase is a FILTER, not an exact solver. The
correctness check is "every brute-force colliding pair appears in the
broadphase candidate set" (`no missed collisions: True`). The broadphase
candidate set need NOT equal the exact set — false positives are allowed
and pruned by the downstream narrow phase.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.benchmark_kit import VariantBenchmark
from core.spatial_index import CellIndex


def _demo_tet_mesh(res: int = 12):
    """Volumetric tet grid matching the geometry of run_surgical_demo in
    tetrahedral_surgical_soft_robotics.py, but with REAL tet connectivity
    (a 6-tet-per-cube decomposition) so the AABBs actually overlap and the
    broadphase has something to filter. The demo's `tets` array is a
    placeholder of zeros; here we populate it so the benchmark measures a
    non-trivial overlap set."""
    x = np.linspace(0.2, 0.8, res)
    y = np.linspace(0.2, 0.8, res)
    z = np.linspace(0.2, 0.8, res)
    X, Y, Z = np.meshgrid(x, y, z)
    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    def vid(i, j, k):
        return i + res * j + res * res * k

    tets = np.zeros(((res - 1) ** 3 * 6, 4), dtype=np.int64)
    t = 0
    for k in range(res - 1):
        for j in range(res - 1):
            for i in range(res - 1):
                c000 = vid(i, j, k)
                c100 = vid(i + 1, j, k)
                c010 = vid(i, j + 1, k)
                c110 = vid(i + 1, j + 1, k)
                c001 = vid(i, j, k + 1)
                c101 = vid(i + 1, j, k + 1)
                c011 = vid(i, j + 1, k + 1)
                c111 = vid(i + 1, j + 1, k + 1)
                # 6-tet decomposition of the cube (all share the c000-c111 diagonal).
                tets[t] = (c000, c100, c110, c111); t += 1
                tets[t] = (c000, c110, c010, c111); t += 1
                tets[t] = (c000, c010, c011, c111); t += 1
                tets[t] = (c000, c011, c001, c111); t += 1
                tets[t] = (c000, c001, c101, c111); t += 1
                tets[t] = (c000, c101, c100, c111); t += 1
    return vertices, tets


def _tet_aabbs(vertices: np.ndarray, tets: np.ndarray):
    """Per-tet AABB (min, max) from its 4 vertex indices. Tets with zero
    connectivity (the demo's placeholder rows) fall back to a degenerate
    AABB around a single vertex so they participate in the broadphase
    without falsely widening the overlap set."""
    n = len(tets)
    valid = np.zeros(n, dtype=bool)
    aabb_min = np.empty((n, 3), dtype=np.float64)
    aabb_max = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        idx = tets[i]
        if np.all(idx >= 0) and np.all(idx < len(vertices)) and len(np.unique(idx)) == 4:
            v = vertices[idx]
            aabb_min[i] = v.min(axis=0)
            aabb_max[i] = v.max(axis=0)
            valid[i] = True
        else:
            # Degenerate AABB: a tiny box around vertex 0 so the placeholder
            # tets do not introduce spurious overlaps with real tets.
            p = vertices[0]
            aabb_min[i] = p
            aabb_max[i] = p
    return aabb_min, aabb_max, valid


def _brute_force_overlap_pairs(aabb_min, aabb_max, valid):
    """Exact O(N^2) AABB overlap pair set as a sorted 1D array of
    pair-encodings (i * N + j with i < j)."""
    n = len(valid)
    pairs = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            if (aabb_min[i, 0] <= aabb_max[j, 0] and aabb_max[i, 0] >= aabb_min[j, 0] and
                aabb_min[i, 1] <= aabb_max[j, 1] and aabb_max[i, 1] >= aabb_min[j, 1] and
                aabb_min[i, 2] <= aabb_max[j, 2] and aabb_max[i, 2] >= aabb_min[j, 2]):
                pairs.append(i * n + j)
    return np.array(pairs, dtype=np.int64), n


def _broadphase_candidate_pairs(aabb_min, aabb_max, valid, cell_size, ring=1):
    """CellIndex ring-`ring` broadphase candidate-pair set (a SUPERSET of
    the exact colliding pairs). AABB centers are indexed in world mode;
    for each occupied cell, every pair of tets whose centers fall inside
    the ring-`ring` neighborhood is emitted as a candidate. With
    cell_size >= max AABB extent and ring=1, every pair of overlapping
    AABBs has centers within one cell and is therefore emitted."""
    centers = 0.5 * (aabb_min + aabb_max)
    # Only index valid tets; map bucket item ids back to tet ids.
    valid_ids = np.where(valid)[0]
    idx = CellIndex(dims=3, cell_size=cell_size)
    idx.build(centers[valid_ids])
    # Build a per-cell list of tet ids (positions in valid_ids).
    tet_by_cell = {}
    for k, items in idx.items():
        tet_by_cell[k] = valid_ids[items]
    n = len(valid)
    cand = set()
    for k, tets_in in tet_by_cell.items():
        neigh = idx.neighbor_keys(k, ring=ring)
        for nk in neigh:
            tets_n = tet_by_cell[nk]
            for a in tets_in:
                for b in tets_n:
                    if a < b:
                        cand.add(int(a) * n + int(b))
                    elif b < a:
                        cand.add(int(b) * n + int(a))
    return np.array(sorted(cand), dtype=np.int64), n


def run_contact_broadphase_variants(res: int = 8):
    vertices, tets = _demo_tet_mesh(res=res)
    aabb_min, aabb_max, valid = _tet_aabbs(vertices, tets)
    # cell_size >= max AABB extent along any axis => overlapping AABBs have
    # centers within one cell => ring=1 captures every colliding pair.
    extents = (aabb_max - aabb_min)[valid]
    max_extent = float(extents.max()) if len(extents) else 1.0
    cell_size = max_extent  # ring=1 suffices (center distance <= 1 cell)

    # One-off correctness check (broadphase is a filter: every exact
    # colliding pair must appear in the candidate set).
    brute_pairs, n = _brute_force_overlap_pairs(aabb_min, aabb_max, valid)
    cand_pairs, _ = _broadphase_candidate_pairs(
        aabb_min, aabb_max, valid, cell_size, ring=1
    )
    brute_set = set(int(p) for p in brute_pairs)
    cand_set = set(int(p) for p in cand_pairs)
    missed = brute_set - cand_set
    no_missed = len(missed) == 0
    note = (f"no missed collisions: {no_missed}; "
            f"{len(brute_set)} exact pairs / {len(cand_set)} broadphase candidates "
            f"(filter superset, narrow-phase prunes false positives)")

    bench = VariantBenchmark(
        f"Tetrahedral contact broadphase (demo mesh res={res}, {len(tets)} tets)"
    )
    bench.add(
        "standard (brute O(N^2))",
        lambda: _brute_force_overlap_pairs(aabb_min, aabb_max, valid)[0],
        note="exact AABB overlap pair set",
    )
    bench.add(
        "+elastichash (CellIndex ring-1)",
        lambda: _broadphase_candidate_pairs(
            aabb_min, aabb_max, valid, cell_size, ring=1
        )[0],
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_contact_broadphase_variants()
