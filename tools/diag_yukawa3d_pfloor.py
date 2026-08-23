"""Round-5 task 5.2 diagnostic: Yukawa3D p-floor root-cause analysis.

Runs the three experiments specified by round-6 task 5.2 (historical plan; retained in git history):
  a. Derivative-tensor audit (P_{alpha,n} nonzero counts per order;
     D_{alpha+beta} bound check).
  b. Single-pair test: worst-converging far cell pair, Taylor contribution
     at p=4..12 vs exact per-particle sum for those two cells.
  c. Sweep ring_direct=3 at p=6,8,10.

Standalone:  python -X utf8 tools/diag_yukawa3d_pfloor.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.yukawa3d_fmm import (
    Yukawa3DFMM, _build_P_tensors, _multi_indices,
)


def _protein(n_atoms=2000, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10 * np.pi, n_atoms)
    r_helix, r_super = 0.25, 0.4
    x = (r_super + r_helix * np.cos(5 * t)) * np.cos(t)
    y = (r_super + r_helix * np.cos(5 * t)) * np.sin(t)
    z = 0.08 * t + r_helix * np.sin(5 * t)
    coords = np.stack([x, y, z], axis=1)
    coords = (coords - np.min(coords, axis=0)) / (np.ptp(coords, axis=0) + 1e-6) * 0.8 + 0.1
    charges = (np.sin(3 * t) * 1.0 + rng.normal(0, 0.2, size=n_atoms)).astype(np.float64)
    return coords.astype(np.float64), charges


def experiment_a(p=8):
    print(f"\n=== Experiment (a): P-tensor audit at p={p} ===")
    P = _build_P_tensors(max_order=2 * p)
    # Count nonzero P_{alpha,n} per |alpha| order.
    print(f"{'|alpha|':>8} {'#alphas':>8} {'#nonzero P_{a,n}':>18} "
          f"{'orders n present':>20}")
    for total in range(0, 2 * p + 1):
        nz_count = 0
        ns_seen = set()
        n_alphas = 0
        for a in _multi_indices(total):
            if a not in P:
                continue
            n_alphas += 1
            for n, poly in P[a].items():
                if poly:
                    nz_count += 1
                    ns_seen.add(n)
        print(f"{total:>8} {n_alphas:>8} {nz_count:>18} "
              f"{sorted(ns_seen)!r:>20}")

    # Check that for |alpha|=2p, there ARE nonzero entries.
    top = 2 * p
    nz_top = 0
    for a in _multi_indices(top):
        for n, poly in P.get(a, {}).items():
            if poly:
                nz_top += 1
    print(f"Nonzero P_{{alpha,n}} at |alpha|=2p={top}: {nz_top}")
    if nz_top == 0:
        print("  *** OFF-BY-ONE SUSPECTED: top order is empty ***")

    # Check the D_{alpha+beta} bound: every gamma used in M2L must have
    # |gamma| <= 2p. The decompositions method enforces this; verify.
    fmm = Yukawa3DFMM(depth=6, p=p, kappa=2.0)
    decomps = fmm._decompositions()
    max_gamma = max((g[0] + g[1] + g[2] for g in decomps), default=0)
    print(f"Max |gamma| in M2L decompositions: {max_gamma} (limit 2p={2*p})")
    # Also check the highest |gamma|=2p entries actually have nonzero P.
    top_gammas = [g for g in decomps if g[0] + g[1] + g[2] == 2 * p]
    missing = [g for g in top_gammas if not any(P.get(g, {}).get(n) for n in range(2 * p + 1))]
    print(f"Top-order gammas with decompositions: {len(top_gammas)}; "
          f"of those with ALL-EMPTY P tensor: {len(missing)}")
    if missing:
        print(f"  *** MISSING P at top order, e.g. {missing[:3]} ***")


def experiment_b(p_max=12):
    print(f"\n=== Experiment (b): single-pair Taylor convergence ===")
    coords, charges = _protein(n_atoms=2000, seed=42)
    depth = 6
    h = 1.0 / depth
    from core.spatial_index import CellIndex
    ci = CellIndex(dims=3, grid_res=depth)
    unique_keys, inverse = ci.build(coords)
    inverse = np.asarray(inverse, dtype=np.int64)
    cell_ints = np.array([ci.key_ints(int(k)) for k in unique_keys], dtype=np.int64)
    centers = (cell_ints.astype(np.float64) + 0.5) * h
    K = len(unique_keys)
    # Far mask at ring_direct=2.
    dci = cell_ints[:, None, :] - cell_ints[None, :, :]
    cheb = np.max(np.abs(dci), axis=-1)
    far_mask = cheb > 2
    r_ts = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
    # Smallest |d_ts| among far pairs -> worst-converging.
    far_r = np.where(far_mask, r_ts, np.inf)
    flat_idx = int(np.argmin(far_r))
    t_id, s_id = divmod(flat_idx, K)
    print(f"Worst far pair: target cell {t_id} (ints {cell_ints[t_id]}), "
          f"source cell {s_id} (ints {cell_ints[s_id]}), "
          f"|d_ts|={r_ts[t_id, s_id]:.6f}, h={h:.6f}, "
          f"ratio={r_ts[t_id, s_id]/h:.4f}")

    # Particles in those two cells.
    idx_t = ci.bucket(int(unique_keys[t_id]))
    idx_s = ci.bucket(int(unique_keys[s_id]))
    pts_t = coords[idx_t]
    pts_s = coords[idx_s]
    q_t = charges[idx_t]
    q_s = charges[idx_s]
    ct = centers[t_id]
    cs = centers[s_id]
    d_ts = ct - cs

    # Exact per-particle sum: u(x_i) = sum_j q_j G(|x_i - x_j|), i in t, j in s.
    diff = pts_t[:, None, :] - pts_s[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    kappa = 2.0
    exact = np.sum(q_s[None, :] * np.exp(-kappa * r) / r, axis=1)
    exact_norm = np.linalg.norm(exact)
    if exact_norm < 1e-300:
        exact_norm = 1e-300

    print(f"{'p':>4} {'rel-L2 (single pair)':>22}")
    for p in range(4, p_max + 1, 2):
        fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
        # Build moments for source cell only.
        from core.yukawa3d_fmm import _factorial
        alphas_p = fmm._alphas_p
        disp_s = pts_s - cs
        n_mom = len(alphas_p)
        M = np.zeros(n_mom)
        for bi, beta in enumerate(alphas_p):
            a, b, c = beta
            M[bi] = np.sum(q_s * (disp_s[:, 0] ** a) * (disp_s[:, 1] ** b) * (disp_s[:, 2] ** c)) / _factorial(beta)
        # Evaluate local expansion at target particles.
        r_d = np.array([[np.linalg.norm(d_ts)]])  # (1,1)
        Gn = np.stack([fmm._G_n(r_d, n) for n in range(2 * p + 1)], axis=-1)  # (1,1,2p+1)
        # L_alpha = sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta
        decomps = fmm._decompositions()
        dx = np.array([[d_ts[0]]])
        dy = np.array([[d_ts[1]]])
        dz = np.array([[d_ts[2]]])
        L = np.zeros(n_mom)
        for gamma in fmm._alphas_2p:
            d_list = decomps.get(gamma)
            if not d_list:
                continue
            Dg = fmm._eval_D_tensor(gamma, dx, dy, dz, Gn)[0, 0]
            for (ai, bi, sgn) in d_list:
                L[ai] += sgn * Dg * M[bi]
        # u(x) = sum_alpha (1/alpha!) L_alpha (x - c_t)^alpha
        disp_t = pts_t - ct
        one_over_fact = 1.0 / fmm._alpha_fact
        est = np.zeros(len(pts_t))
        for ai, alpha in enumerate(alphas_p):
            a, b, c = alpha
            est += one_over_fact[ai] * L[ai] * (disp_t[:, 0] ** a) * (disp_t[:, 1] ** b) * (disp_t[:, 2] ** c)
        rel = np.linalg.norm(est - exact) / exact_norm
        print(f"{p:>4} {rel:>22.4e}")


def experiment_c():
    print(f"\n=== Experiment (c): ring_direct=3 sweep at p=6,8,10 ===")
    coords, charges = _protein(n_atoms=2000, seed=42)
    from apps.app5_benchmark_variants import _direct_debye_huckel
    ref = _direct_debye_huckel(coords, charges, kappa=2.0)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-300:
        ref_norm = 1e-300
    from core import Yukawa3DFMM
    print(f"{'ring':>6} {'p':>4} {'rel-L2':>14}")
    for ring in (2, 3):
        for p in (6, 8, 10):
            fmm = Yukawa3DFMM(depth=6, p=p, kappa=2.0, ring_direct=ring)
            est = fmm.evaluate(coords, charges)
            rel = float(np.linalg.norm(est - ref) / ref_norm)
            print(f"{ring:>6} {p:>4} {rel:>14.4e}")


if __name__ == "__main__":
    experiment_a(p=8)
    experiment_b(p_max=12)
    experiment_c()
