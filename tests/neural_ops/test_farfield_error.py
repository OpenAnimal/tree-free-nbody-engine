"""
Far-field error law for `TreeFreeMultipoleAttention` (round-7 task T-D1,
re-gated round 14).

Sweeps σ/cell x grid_depth x feature spread on the far-field
approximation and fits `rel-L2 ≈ A·(cell/σ)² + B·s_qk` where `s_qk` =
mean in-cluster std of the scaled dot τq·k.

Term (a) = feature-dot collapse: `exp(τ q_i·k_j) → exp(τ q_i·k̄_c)`.
Term (b) = spatial dipole truncation: scales like `(cell/σ)²`.

The claimed regime (asserted by `test_farfield_error_law`): the far field
is accurate — rel-L2 < 0.1 for σ ≥ 2·cell — when features are
NEAR-CONSTANT within clusters (const-K control) or mildly spread
(small-K, in-cluster dot std s_qk ≈ 0.08).

Out of regime (documented finding, also asserted): with O(1) in-cluster
feature spread (randn-K, s_qk ≈ 0.9) the error is 0.3–0.7 and dominated
by B·s_qk at every σ/cell. This is intrinsic, not an implementation gap:
the exact per-cluster value sum Σ_j exp(τ q·k_j) v_j is
lognormal-concentrated (dominated by the within-cluster max of q·k), and
no finite-moment expansion recovers it. Verified by direct measurement
(round 14): adding the second-order cluster-covariance weight correction
E_j[exp(τ q·k_j)] = exp(τ q·k̄)·exp(τ²/2 q^T Σ_c q) relieved rel-L2 by
only ~5–10%, and additionally adding the first-order value-feature cross
moment Σ_j δk_j ⊗ v_j was non-monotone (2x better at depth 3–4, 2x
WORSE at depth 5–6, σ/cell = 8). The corrections were reverted; the
honest error model is the two-term fit below.

Run standalone:  python -X utf8 tests/neural_ops/test_farfield_error.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

# Make repo root importable
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_ops.multipole_attention import TreeFreeMultipoleAttention


def _dense_reference(coords, q_feat, k_feat, v_feat, sigma, temperature):
    """Full dense softmax over w_ij = exp(-dist²/2σ²)·exp(τ q_i·k_j)."""
    N, D = q_feat.shape
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, d)
    dist_sq = np.sum(diff ** 2, axis=-1)             # (N, N)
    spatial_w = np.exp(-dist_sq / (2.0 * sigma ** 2))
    dot = np.matmul(q_feat, k_feat.T) * temperature  # (N, N)
    w = spatial_w * np.exp(np.clip(dot, -50.0, 50.0))
    # Exclude self
    np.fill_diagonal(w, 0.0)
    weight = np.sum(w, axis=1, keepdims=True)        # (N, 1)
    weight_safe = np.maximum(weight, 1e-30)
    out = np.matmul(w, v_feat) / weight_safe         # (N, D)
    return out


def _in_cluster_std_qk(coords, q_feat, k_feat, cell_size, temperature):
    """Mean in-cluster std of the scaled dot τ·q·k (term (a) driver)."""
    # Bin into cells
    origin = np.min(coords, axis=0)
    shifted = coords - origin
    ix = np.maximum(0, (shifted[:, 0] / cell_size).astype(np.int64))
    iy = np.maximum(0, (shifted[:, 1] / cell_size).astype(np.int64))
    keys = ix * 100000 + iy
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    K = len(unique_keys)
    stds = []
    for c in range(K):
        mask = inverse == c
        if np.sum(mask) < 2:
            continue
        q_c = q_feat[mask]
        k_c = k_feat[mask]
        # Per-pair scaled dot within cluster: τ * q_i · k_j
        dot_in = np.matmul(q_c, k_c.T) * temperature
        stds.append(np.std(dot_in))
    return float(np.mean(stds)) if stds else 0.0


def _eval_config(q_feat, k_feat, v_feat, coords, sigma, grid_depth,
                 temperature, D):
    """One (config, σ, depth) cell: layer output vs dense reference."""
    layer = TreeFreeMultipoleAttention(
        embed_dim=D, spatial_dim=2,
        grid_depth=grid_depth, spatial_sigma=sigma,
        temperature=temperature,
    )
    out_approx, _meta = layer.forward(q_feat, k_feat, v_feat, coords)
    out_dense = _dense_reference(coords, q_feat, k_feat, v_feat,
                                 sigma, temperature)
    rel_l2 = float(np.linalg.norm(out_approx - out_dense) /
                   max(1e-30, np.linalg.norm(out_dense)))
    return rel_l2


def run_farfield_error_sweep(N=2000, D=64, seed=42,
                             sigma_cell_ratios=(0.5, 1.0, 2.0, 4.0, 8.0),
                             grid_depths=(3, 4, 5, 6), verbose=True):
    """Sweep σ/cell x grid_depth x feature spread, fit A·(cell/σ)² + B·s_qk.

    Three feature configurations per cell:
      randn-K  unit-variance features (the hard case, s_qk ≈ 0.9)
      small-K  0.08 x unit variance   (the claimed regime, s_qk ≈ 0.08)
      const-K  constant features      (term (a) exactly zeroed)
    """
    rng = np.random.RandomState(seed)
    # 2D coordinates in [0, 1]²
    coords = rng.uniform(0.0, 1.0, size=(N, 2)).astype(np.float64)
    q_feat = rng.randn(N, D).astype(np.float64)
    k_feat = rng.randn(N, D).astype(np.float64)
    v_feat = rng.randn(N, D).astype(np.float64)
    temperature = 1.0 / np.sqrt(D)
    k_small = 0.08 * k_feat
    k_const = np.ones((N, D), dtype=np.float64)

    results = []
    print(f"\n=== Far-field error law sweep (N={N}, D={D}, seed={seed}) ===")
    print(f"{'σ/cell':>8} {'depth':>6} {'cell':>8} "
          f"{'rel-L2 randn':>13} {'rel-L2 small':>13} {'rel-L2 const':>13}"
          f" {'s_qk rnd/sml':>13}")
    print("-" * 90)

    for grid_depth in grid_depths:
        grid_res = 1 << grid_depth
        cell = 1.0 / grid_res
        for ratio in sigma_cell_ratios:
            sigma = ratio * cell
            row = {'sigma_cell': ratio, 'depth': grid_depth, 'cell': cell,
                   'sigma': sigma}
            for label, k_c in (("randn-K", k_feat), ("small-K", k_small),
                               ("const-K", k_const)):
                row[label] = _eval_config(q_feat, k_c, v_feat, coords,
                                          sigma, grid_depth, temperature, D)
                row[f's_qk_{label}'] = _in_cluster_std_qk(
                    coords, q_feat, k_c, cell, temperature)
            results.append(row)
            print(f"{ratio:>8.1f} {grid_depth:>6d} {cell:>8.5f} "
                  f"{row['randn-K']:>13.4e} {row['small-K']:>13.4e} "
                  f"{row['const-K']:>13.4e} "
                  f"{row['s_qk_randn-K']:.2f}/{row['s_qk_small-K']:.2f}")

    # Least-squares fit on the randn-K rows: rel-L2 ≈ A·(cell/σ)² + B·s_qk
    X = np.array([[(r['cell'] / r['sigma']) ** 2, r['s_qk_randn-K']]
                  for r in results])
    y = np.array([r['randn-K'] for r in results])
    coeffs, residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    A, B = float(coeffs[0]), float(coeffs[1])
    print("-" * 90)
    print("\nFit (randn-K): rel-L2 ≈ A·(cell/σ)² + B·s_qk")
    print(f"  A = {A:.4e}")
    print(f"  B = {B:.4e}")
    if len(residuals) > 0:
        print(f"  residual sum of squares = {residuals[0]:.4e}")

    return {'A': A, 'B': B, 'results': results}


def check_error_law(sweep, gate=0.1, sigma_cell_min=2.0):
    """The re-scoped round-14 gate + the documented finding, as checks.

    Gate (claimed regime): for σ ≥ 2·cell, near-constant (const-K) and
    mildly spread (small-K) features reach rel-L2 < `gate` at every depth.
    Finding (out of regime): O(1)-spread features (randn-K) stay ABOVE
    `gate`, monotonically worse than small-K at the same cell — the error
    is bounded by feature spread, not spatial discretization.
    """
    gate_pass = True
    finding_holds = True
    for r in sweep['results']:
        if r['sigma_cell'] < sigma_cell_min:
            continue
        if r['const-K'] >= gate or r['small-K'] >= gate:
            gate_pass = False
        if r['randn-K'] < gate:
            finding_holds = False
        if r['randn-K'] <= r['small-K']:
            finding_holds = False
    return gate_pass, finding_holds


def test_farfield_error_law():
    """CI-sized sweep: the gate holds in the claimed regime, and the
    out-of-regime finding holds (asserted so it cannot silently regress
    into an over-claim)."""
    sweep = run_farfield_error_sweep(N=600, D=64, seed=42,
                                     sigma_cell_ratios=(2.0, 8.0),
                                     grid_depths=(3, 5))
    gate_pass, finding_holds = check_error_law(sweep)
    assert gate_pass, (
        "far-field gate FAILED in the claimed regime (const-K/small-K "
        "rel-L2 >= 0.1 at sigma >= 2*cell): " +
        "; ".join(f"d{r['depth']} s{r['sigma_cell']:.0f}: "
                  f"const={r['const-K']:.3f} small={r['small-K']:.3f}"
                  for r in sweep['results'] if r['sigma_cell'] >= 2.0))
    assert finding_holds, (
        "out-of-regime finding regressed: randn-K rel-L2 dropped to "
        "small-K levels — the error law changed, re-examine the claim")
    # Term (a) dominates term (b): the feature-spread coefficient B is
    # clearly positive and above A (signed: A < B; A is often a small
    # negative fit artifact on a 4-cell grid).
    assert sweep['B'] > 0.3, f"B = {sweep['B']:.3f} (feature term vanished?)"
    assert sweep['A'] < sweep['B'], (
        f"A = {sweep['A']:.3f} vs B = {sweep['B']:.3f}")


if __name__ == "__main__":
    sweep = run_farfield_error_sweep()
    gate_pass, finding_holds = check_error_law(sweep)
    if gate_pass:
        print("\nGATE PASS: rel-L2 < 0.1 for σ ≥ 2·cell on const-K and "
              "small-K (s_qk ≈ 0.08) at every depth.")
    else:
        print("\nGATE FAIL: claimed regime violated (see table).")
    if finding_holds:
        print("FINDING HOLDS (documented, asserted): randn-K (s_qk ≈ 0.9) "
              "stays above 0.1 at every σ ≥ 2·cell, monotonically worse "
              "than small-K — the far-field error is bounded by the "
              "in-cluster feature spread (lognormal concentration of the "
              "cluster value sum), not by the spatial discretization. "
              "Moment corrections were measured and refuted (module "
              "docstring); do not re-claim the randn-K regime.")
    else:
        print("FINDING REGRESSED: randn-K accuracy reached small-K levels.")
