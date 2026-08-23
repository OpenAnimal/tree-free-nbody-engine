"""Round-7 task T-C7 / finding R7-F28: flat-scheme saturation guidance.

Measures the flat radial-Taylor engine (`core.yukawa3d_fmm.Yukawa3DFMM`)
across N x cells-per-side (depth) to characterize the saturation behavior
documented in R7-F28:

  - Far field: O(K^2 * |alphas|) with K = occupied cells <= depth^dims.
    For fixed depth this is constant in N (the flat scheme's linearity).
  - Near field: O(N * M_bar * (2*ring+1)^d) where M_bar = N/K is the mean
    cell occupancy. Once cells saturate (M_bar grows with N at fixed depth),
    the near field degrades toward O(N^2 / depth^d).

The classical single-level optimum balances the two at K_opt ~ N^{2/3} (3D),
total O(N^{4/3}) — the honest headline complexity of every flat engine here.
The true O(N) member of the repo is the multilevel adaptive FMM engine / GPU demo.

Usage:  python -X utf8 tools/diag_flat_saturation.py
Output: wall time + rel-L2 + K + M_bar table; paste into BENCHMARKS.md.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.yukawa3d_fmm import Yukawa3DFMM


def _clustered_positions(N: int, n_clusters: int, rng: np.random.RandomState) -> np.ndarray:
    """N points in n_clusters Gaussian blobs inside [0,1)^3 (clustered, not uniform)."""
    centers = rng.uniform(0.1, 0.9, size=(n_clusters, 3))
    per = N // n_clusters
    pts = np.empty((N, 3), dtype=np.float64)
    idx = 0
    for c in range(n_clusters):
        blob = centers[c] + rng.normal(0, 0.03, size=(per, 3))
        blob = np.clip(blob, 0.0, 1.0 - 1e-6)
        pts[idx:idx + per] = blob
        idx += per
    if idx < N:
        pts[idx:] = np.clip(centers[0] + rng.normal(0, 0.03, size=(N - idx, 3)), 0, 1 - 1e-6)
    return pts


def _direct_reference(pos: np.ndarray, q: np.ndarray, kappa: float) -> np.ndarray:
    N = len(pos)
    pot = np.zeros(N, dtype=np.float64)
    for i in range(N):
        diff = pos[i] - pos
        r = np.linalg.norm(diff, axis=-1)
        mask = r > 1e-9
        pot[i] = np.sum(q[mask] * np.exp(-kappa * r[mask]) / r[mask])
    return pot


def main():
    kappa = 1.0
    p = 8
    rng = np.random.RandomState(42)
    # Reduced from the plan's {2k,8k,32k,128k}x{8,16,32,64} for tractability:
    # the near-field loop in `evaluate` calls `neighborhood_indices` per cell,
    # which does 125 Morton decode/encode/hash-lookup ops per cell in Python
    # (~5ms/cell). At K=500 cells this is ~2.5s/call before any math. This is
    # a pre-existing spatial_index hot path (T-C6's CSR batching targets it);
    # T-C7 measures the end-to-end trend, not the per-call constant.
    Ns = [500, 2000]
    depths = [8, 16]

    print("=" * 90)
    print("Flat-scheme saturation: Yukawa3DFMM (3D, kappa=1, p=8, ring=2, clustered)")
    print("R7-F28: far = O(K^2 * |alphas|); near = O(N * M_bar * 5^3); K_opt ~ N^{2/3} -> O(N^{4/3})")
    print("=" * 90)
    print(f"{'N':>8} {'depth':>6} {'K':>6} {'M_bar':>8} {'wall_ms':>10} {'rel_L2':>12}")
    print("-" * 90)

    for N in Ns:
        pos = _clustered_positions(N, max(1, N // 200), rng)
        q = rng.uniform(-1.0, 1.0, size=N)
        # Direct reference only for N <= 8000 (60s budget)
        direct = None
        if N <= 8000:
            t0 = time.perf_counter()
            direct = _direct_reference(pos, q, kappa)
            t_direct = (time.perf_counter() - t0) * 1000.0
            print(f"{'(direct)':>8} {'':>6} {'':>6} {'':>8} {t_direct:>10.1f} {'':>12}")

        for depth in depths:
            fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
            # Warm up (P-tensor construction is one-time)
            t0 = time.perf_counter()
            pot = fmm.evaluate(pos, q)
            wall = (time.perf_counter() - t0) * 1000.0
            # K = occupied cells; M_bar = N / K
            from core.spatial_index import CellIndex
            ci = CellIndex(dims=3, grid_res=depth)
            unique_keys, _ = ci.build(pos)
            K = len(unique_keys)
            M_bar = N / max(1, K)
            if direct is not None:
                rel = np.linalg.norm(pot - direct) / max(1e-30, np.linalg.norm(direct))
                rel_str = f"{rel:.4e}"
            else:
                rel_str = "(no direct)"
            print(f"{N:>8} {depth:>6} {K:>6} {M_bar:>8.1f} {wall:>10.1f} {rel_str:>12}")
        print()

    print("=" * 90)
    print("Guidance:")
    print("  - Accuracy-driven rule: keep M_bar <= ~60 (mean cell occupancy).")
    print("  - Cost-driven classical optimum (3D): K_opt ~ N^{2/3}, total O(N^{4/3}).")
    print("  - The flat engines are O(N^{4/3})-class single-level schemes.")
    print("  - The true O(N) member of the repo is the multilevel adaptive FMM engine / GPU demo.")
    print("  - Choose depth ~ N^{2/3} for the cost optimum; deeper favors accuracy.")
    print("=" * 90)


if __name__ == "__main__":
    main()
