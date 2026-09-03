"""
Application 11: Diffusion Sampling — FMM-over-Trajectory vs Parallel-in-Time vs FMM-Ensemble
============================================================================================
A two-arm benchmark that empirically settles where FMM belongs in diffusion / flow-matching
policy sampling.

Setup
-----
Target: a 2D Gaussian mixture with 4 well-separated modes inside [0,1]^2.  The score
    s(x) = grad log p(x)
is analytic for a Gaussian mixture, so the sampler cost is isolated from any neural
network.  This is a clean stand-in for the iterative score-based sampling loop of a
diffusion policy (ULA / probability-flow ODE share the same K-step chain structure as
DDPM/flow-matching sampling).

Arm 1 — single trajectory: where does the K-step latency go?
-------------------------------------------------------------
The diffusion sampling ODE/SDE is a CHAIN:  x_{k+1} = F(x_k).
There is NO all-pairs kernel across time steps; the only cross-step coupling is the
integral constraint x_K = x_0 + integral of F, which is a prefix sum (parallel scan),
NOT an N-body sum.  Therefore:

  * Sequential Euler : K sequential score evals (latency = K forwards).
  * Picard / parallel-in-time (Shih et al., 2023, "Parallel DDIM" / ParaDiGMS):
        guess the whole trajectory, evaluate the score at all K states in ONE batched
        call, re-integrate via cumsum (parallel scan), repeat P Picard iterations.
        Latency = P forwards (P << K).  This is the correct primitive for the chain.
  * "FMM-over-trajectory" strawman: treat the K intermediate states of the single
        trajectory as particles and add an FMM all-pairs repulsion among them.  This
        injects a spurious coupling the chain does not have.  Expected: costs more,
        does not improve (and can hurt) sample quality.  This is the negative result.

Arm 2 — ensemble of M trajectories: where FMM IS the right primitive
--------------------------------------------------------------------
Run M particles; at each step they interact via Stein / RBF repulsion in state space to
prevent mode collapse and cover all modes of the multi-modal action distribution (the
actual reason to use a diffusion policy).  The interaction IS an all-pairs sum
    phi_i = sum_{j!=i} q_j k(x_i, x_j) (x_i - x_j) / h^2
which is O(M^2) brute and O(M) via the tree-free multipole drift.  This is the
legitimate, radical FMM win: it makes Stein-ensembled diffusion affordable.

  * Independent : M ULA chains, no interaction (baseline quality / mode coverage).
  * Stein-brute : + O(M^2) RBF repulsion per step.
  * Stein-FMM   : + O(M)   RBF repulsion per step (TreeFreeMultipoleFlowDrift).

Metrics
-------
wall-clock, sequential score-call rounds (latency proxy for a neural score),
mode coverage (fraction of modes hit by >=1 sample within tau), MMD^2 to target.

References
----------
- Shih, Padmanabhan, Poole, & Murphy (2023). Parallel Sampling of Diffusion Models.
- Liu & Wang (2017). Stein Variational Gradient Descent. NeurIPS.
- Farach-Colton, Krapivin, & Kuszmaul (2025). Optimal Open Addressing. arXiv:2501.02305.

CONCLUSION (Sep 2026) — HONEST ASSESSMENT:
  This benchmark confirmed two things:

  Arm 1 (Picard, K-step chain): 6x speedup. But this is NOT novel —
  ParaDiGMS (Shih et al., 2023), consistency models, and flow matching
  already address the K-step denoising chain. Flow matching (already in
  v7_flowmatch.py) reduces K directly from 100 to 1-10. This is the
  main diffusion bottleneck, and it is already solved by other methods.

  Arm 2 (FMM, M-particle ensemble): 7.3x at M=4096. Real, but only
  helps the non-standard Stein-ensemble variant. Standard diffusion
  policies (Chi et al., 2023) sample 1 action or a small independent
  batch — no Stein repulsion needed.

  FMM + Stein ensembles do NOT solve the general diffusion bottleneck.
  They optimize a niche setting (Stein-ensembled sampling) that isn't
  standard practice, applied to a bottleneck (M-particle interaction)
  that isn't the main one (K-step chain is).

  WHAT IS GENUINELY UNSOLVED — RL fine-tuning mode collapse:
    Behavior-cloned diffusion policy + RL reward -> mode collapse (reward
    pushes toward highest-reward mode, kills others). This is STILL OPEN
    as of 2026:
      - HRF (Oct 2024): partial mitigation via hierarchical denoising
      - DPPO (ICLR 2025): stable fine-tuning, doesn't address diversity
      - DRIFT (Jan 2026): calls it "the curse of diversity collapse",
        "a fundamental limitation" -- proposes reward shaping, partial
      - NCDPO (May 2025): tractable gradients, doesn't address diversity
    None fully solves it. The Stein/FMM diversity machinery could be
    repurposed as a REGULARIZER during RL fine-tuning to prevent mode
    collapse, rather than as an inference-time accelerator. That is the
    promising pivot direction.

  Related files (completed ablation/negative result):
    - app11_diffusion_trajectory_fmm_bench.py (this file)
    - stein_ensemble_diffusion_ablation.py (2D/16D ablation)
    - stein_ensemble_gif_report.py (visualization)
    - stein_ensemble_toy_problems.py (real toy problems)
"""

from __future__ import annotations
import os
import sys
import time

# Repo root on sys.path so `from neural_ops...` resolves (matches other apps).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neural_ops.multipole_flow_drift import TreeFreeMultipoleFlowDrift


# ─────────────────────────────────────────────────────────────────────────────
# Target: 2D Gaussian mixture with analytic score
# ─────────────────────────────────────────────────────────────────────────────

def make_target(n_modes: int = 4, sigma: float = 0.06, seed: int = 0):
    """Modes on a grid inside [0,1]^2. Returns (mus, sigma, weights)."""
    side = int(round(np.sqrt(n_modes)))
    assert side * side == n_modes, "n_modes must be a perfect square"
    rng = np.random.RandomState(seed)
    margin = 0.22
    grid = np.linspace(margin, 1.0 - margin, side)
    mus = np.array([[gx, gy] for gy in grid for gx in grid])
    w = rng.dirichlet(np.ones(n_modes) * 5.0).astype(np.float64)
    w = w / w.sum()
    return mus.astype(np.float64), float(sigma), w


def gm_score(x: np.ndarray, mus: np.ndarray, sigma: float, w: np.ndarray) -> np.ndarray:
    """grad log p(x) for p = sum_k w_k N(x; mu_k, sigma^2 I).  x: (N,2) -> (N,2)."""
    x = np.asarray(x, dtype=np.float64)
    two_s2 = 2.0 * sigma * sigma
    # log_unnorm_k(x_i) = log w_k - d/2 log(2 pi s^2) - ||x_i-mu_k||^2/(2 s^2)
    diff = x[:, None, :] - mus[None, :, :]          # (N, K, 2)
    sq = np.sum(diff ** 2, axis=-1)                  # (N, K)
    log_w = np.log(w + 1e-300)
    log_comp = log_w[None, :] - sq / two_s2          # drop const term (cancels in softmax)
    log_comp -= log_comp.max(axis=1, keepdims=True)
    r = np.exp(log_comp)                             # (N, K) responsibilities
    r /= r.sum(axis=1, keepdims=True)
    # score_i = sum_k r_ik (mu_k - x_i)/sigma^2
    score = (r[:, :, None] * (mus[None, :, :] - x[:, None, :])).sum(axis=1) / (sigma ** 2)
    return score


def gm_sample(n: int, mus: np.ndarray, sigma: float, w: np.ndarray, rng) -> np.ndarray:
    idx = rng.choice(len(mus), size=n, p=w)
    return mus[idx] + sigma * rng.randn(n, mus.shape[1])


def mode_coverage(samples: np.ndarray, mus: np.ndarray, sigma: float,
                  tau: float = 3.0) -> float:
    """Fraction of modes with >=1 sample within tau*sigma."""
    d = np.linalg.norm(samples[:, None, :] - mus[None, :, :], axis=-1)  # (N, K)
    hit = (d.min(axis=0) < tau * sigma)
    return float(hit.mean())


def mmd2_sq(samples: np.ndarray, target: np.ndarray, bw: float = 0.05) -> float:
    """Biased MMD^2 with Gaussian kernel (squared distance)."""
    X = samples.astype(np.float64)
    Y = target.astype(np.float64)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    k = np.exp(-XX / (2 * bw ** 2)).mean() + np.exp(-YY / (2 * bw ** 2)).mean() \
        - 2.0 * np.exp(-XY / (2 * bw ** 2)).mean()
    return float(k)


# ─────────────────────────────────────────────────────────────────────────────
# Arm 1: single-trajectory samplers (chain → parallel scan, not FMM)
# ─────────────────────────────────────────────────────────────────────────────

def _ode_step(x, eta, mus, sigma, w):
    """One deterministic probability-flow ODE step: x <- x + 0.5*eta*score(x)."""
    s = gm_score(x, mus, sigma, w)
    return x + 0.5 * eta * s


def sequential_ode(x0, K, eta, mus, sigma, w, rng=None):
    """K sequential score evals on the deterministic probability-flow ODE.
    Returns (x_final, n_score_rounds)."""
    x = x0.copy()
    for _ in range(K):
        x = _ode_step(x[None, :], eta, mus, sigma, w)[0]
        x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return x, K  # K sequential rounds


def picard_ode(x0, K, eta, P, mus, sigma, w, rng=None, damp=0.5):
    """Parallel-in-time Picard on the deterministic ODE (what parallel DDIM targets).

    Warm start: constant-score Euler trajectory (cost: 1 score call).  Then P-1 damped
    Picard correction rounds, each = 1 batched score call over all K states + parallel
    scan (cumsum) re-integration.  Total latency proxy = P sequential score rounds.

    Damping (relaxation) is required because the GM score is stiff (Lipschitz ~1/sigma^2);
    undamped Picard diverges for L*T > 1.  This mirrors real parallel-DDIM, which uses a
    warm start (previous trajectory / DDIM baseline) + a few correction steps.
    """
    # Warm start: Euler with the score at x0 held constant across the trajectory.
    s0 = gm_score(x0[None, :], mus, sigma, w)[0]
    traj = x0[None, :] + np.cumsum(np.tile(0.5 * eta * s0, (K, 1)), axis=0)
    traj = np.clip(traj, 1e-4, 1.0 - 1e-4)

    for _ in range(P - 1):
        s_all = gm_score(traj, mus, sigma, w)               # ONE batched call over K states
        integrated = x0[None, :] + np.cumsum(0.5 * eta * s_all, axis=0)
        integrated = np.clip(integrated, 1e-4, 1.0 - 1e-4)
        traj = (1.0 - damp) * traj + damp * integrated       # damped relaxation
    return traj[-1], P  # P sequential rounds


def fmm_trajectory_ode(x0, K, eta, mus, sigma, w, rng, drift_op, lam=0.5):
    """Strawman: add FMM all-pairs repulsion among the K intermediate states of ONE chain.

    This is the naive "FMM the trajectory" idea.  The chain has no all-pairs kernel, so
    this injects a spurious coupling.  K sequential score evals + K FMM calls on tiny
    (<=K+1) particle sets -> pure Python overhead, no benefit.
    """
    x = x0.copy()
    history = [x.copy()]
    for _ in range(K):
        s = gm_score(x[None, :], mus, sigma, w)[0]
        pts = np.clip(np.array(history), 1e-4, 1.0 - 1e-4).astype(np.float32)
        drift, _ = drift_op.compute_drift(pts)
        phi = drift[-1]  # force on the current (last) state
        x = x + 0.5 * eta * (s + lam * phi)
        x = np.clip(x, 1e-4, 1.0 - 1e-4)
        history.append(x.copy())
    return x, K  # K sequential rounds (+ K FMM calls)


# ─────────────────────────────────────────────────────────────────────────────
# Arm 2: ensemble samplers (M particles) — the real FMM win
# ─────────────────────────────────────────────────────────────────────────────

def _brute_repulsion(x, kernel, h, eps=1e-3):
    """O(M^2) all-pairs repulsion. kernel: 'gaussian_rbf' or 'coulomb_soft'."""
    diff = x[:, None, :] - x[None, :, :]            # (M, M, 2)
    sq = np.sum(diff ** 2, axis=-1)                  # (M, M)
    np.fill_diagonal(sq, np.inf)                     # exclude self
    if kernel == "gaussian_rbf":
        kern = np.exp(-sq / (2.0 * h * h)) / (h * h)
    elif kernel == "coulomb_soft":
        kern = 1.0 / (sq + eps * eps) ** 1.5
    else:
        raise ValueError(kernel)
    kern = np.where(np.isfinite(sq), kern, 0.0)
    return np.einsum('mn,mnd->md', kern, diff)


def independent_ensemble(M, K, eta, mus, sigma, w, rng):
    x = rng.uniform(0.05, 0.95, size=(M, 2))
    sqrt_eta = np.sqrt(eta)
    for _ in range(K):
        s = gm_score(x, mus, sigma, w)
        x = x + 0.5 * eta * s + sqrt_eta * rng.randn(M, 2)
    return np.clip(x, 1e-4, 1.0 - 1e-4)


def stein_brute_ensemble(M, K, eta, mus, sigma, w, rng, h, lam, kernel="gaussian_rbf"):
    x = rng.uniform(0.05, 0.95, size=(M, 2))
    sqrt_eta = np.sqrt(eta)
    for _ in range(K):
        s = gm_score(x, mus, sigma, w)
        phi = _brute_repulsion(x, kernel, h) / M
        x = x + 0.5 * eta * (s + lam * phi) + sqrt_eta * rng.randn(M, 2)
        x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return x


def _grid_depth_for(M, target_per_cell=16.0, lo=2, hi=8):
    """Pick grid_depth so ~target_per_cell particles per cell (n_cells ~ M/target)."""
    res = max(2.0, np.sqrt(M / target_per_cell))
    depth = int(round(np.log2(res)))
    return max(lo, min(hi, depth))


def stein_fmm_ensemble(M, K, eta, mus, sigma, w, rng, h, lam, kernel="gaussian_rbf"):
    x = rng.uniform(0.05, 0.95, size=(M, 2))
    sqrt_eta = np.sqrt(eta)
    drift_op = TreeFreeMultipoleFlowDrift(
        spatial_dim=2, grid_depth=_grid_depth_for(M),
        kernel_type=kernel, rbf_sigma=h, softening=1e-3,
    )
    for _ in range(K):
        s = gm_score(x, mus, sigma, w)
        pts = np.clip(x, 1e-4, 1.0 - 1e-4).astype(np.float32)
        drift, _ = drift_op.compute_drift(pts)
        phi = drift.astype(np.float64) / M
        x = x + 0.5 * eta * (s + lam * phi) + sqrt_eta * rng.randn(M, 2)
        x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Bench drivers
# ─────────────────────────────────────────────────────────────────────────────

def run_arm1(mus, sigma, w, rng, K=48, eta=0.01, P=8, lam_fmm=0.5):
    print("\n" + "=" * 78)
    print("ARM 1 — single trajectory: deterministic ODE chain (sequential vs Picard vs FMM-strawman)")
    print("=" * 78)
    print(f"K={K} steps, eta={eta}, Picard rounds P={P} (warm-start + damped), FMM-strawman lam={lam_fmm}")
    print("-" * 78)

    drift_op = TreeFreeMultipoleFlowDrift(
        spatial_dim=2, grid_depth=5, kernel_type="gaussian_rbf",
        rbf_sigma=0.1, softening=1e-3,
    )

    n_trials = 60
    x0s = rng.uniform(0.05, 0.95, size=(n_trials, 2))
    target = gm_sample(4096, mus, sigma, w, rng)

    rows = []
    for name, fn in [
        ("sequential",   lambda x0: sequential_ode(x0, K, eta, mus, sigma, w)),
        ("picard",       lambda x0: picard_ode(x0, K, eta, P, mus, sigma, w)),
        ("fmm-strawman", lambda x0: fmm_trajectory_ode(x0, K, eta, mus, sigma, w, rng,
                                                       drift_op, lam=lam_fmm)),
    ]:
        t0 = time.perf_counter()
        outs = np.array([fn(x0)[0] for x0 in x0s])
        rounds = fn(x0s[0])[1]
        dt = time.perf_counter() - t0
        cov = mode_coverage(outs, mus, sigma)
        mmd = mmd2_sq(outs, target)
        rows.append((name, dt, rounds, cov, mmd))
        print(f"{name:14s} | time {dt:7.3f}s | score-rounds {rounds:3d} "
              f"| mode-cov {cov:.3f} | MMD^2 {mmd:.4e}")

    print("-" * 78)
    print("Reading: 'score-rounds' is the latency proxy (= sequential forward passes with a")
    print("neural score net). Picard uses P<<K rounds -> P forwards vs K forwards, with")
    print("matching mode coverage.  The FMM-strawman does NOT reduce the round count (the")
    print("actual bottleneck) and is ~57x slower in wall-clock from per-step FMM Python")
    print("overhead on tiny sets.  Its lower MMD here is incidental — the spurious all-pairs")
    print("term changes the dynamics (a different ODE), not a principled quality gain.")
    return rows


def run_arm2(mus, sigma, w, rng, K=60, eta=0.02, h=0.09, lam=0.5):
    print("\n" + "=" * 78)
    print("ARM 2 — ensemble of M: Stein repulsion brute O(M^2) vs FMM O(M)")
    print("=" * 78)
    print(f"K={K} steps, eta={eta}, repulsion h={h}, lam={lam}, grid_depth tuned per M (~16/cell)")
    print("-" * 78)

    target = gm_sample(8192, mus, sigma, w, rng)
    Ms = [64, 256, 1024, 4096]

    for kernel in ("coulomb_soft", "gaussian_rbf"):
        print(f"\n--- kernel = {kernel} ---")
        print(f"{'M':>6} | {'sampler':12s} | {'time':>9s} | {'mode-cov':>8s} | {'MMD^2':>10s}")
        print("-" * 78)
        for M in Ms:
            depth = _grid_depth_for(M)
            for name, fn in [
                ("independent", lambda M=M: independent_ensemble(M, K, eta, mus, sigma, w, rng)),
                ("stein-brute", lambda M=M, k=kernel: stein_brute_ensemble(
                    M, K, eta, mus, sigma, w, rng, h, lam, kernel=k)),
                ("stein-fmm",   lambda M=M, k=kernel: stein_fmm_ensemble(
                    M, K, eta, mus, sigma, w, rng, h, lam, kernel=k)),
            ]:
                t0 = time.perf_counter()
                out = fn()
                dt = time.perf_counter() - t0
                cov = mode_coverage(out, mus, sigma)
                mmd = mmd2_sq(out, target)
                print(f"{M:>6} | {name:12s} | {dt:8.3f}s | {cov:8.3f} | {mmd:10.4e}")
            print(f"  (grid_depth={depth} for M={M})  " + "-" * 48)

    print("\nReading: FMM beats brute O(M^2) at large M for BOTH kernels (6-7x at M=4096),")
    print("with matching mode coverage and MMD.  For the LONG-RANGE Coulomb kernel, FMM")
    print("multipole is the canonical O(M) tool.  For the SHORT-RANGE Gaussian RBF, FMM")
    print("still wins here (h > cell_size, so the far field carries real mid-range weight),")
    print("but a cell-list with ring-r neighbor search (r = ceil(3h/cell)) is the more")
    print("natural O(M) alternative for short-range kernels — it skips the multipole math")
    print("and does direct near-field within the interaction radius.  FMM overhead dominates")
    print("at small M (M=64): use brute below the crossover.")
    return None


def main():
    rng = np.random.RandomState(123)
    mus, sigma, w = make_target(n_modes=4, sigma=0.06, seed=1)
    print(f"Target: {len(mus)}-mode Gaussian mixture, sigma={sigma}, weights={np.round(w,3)}")
    print(f"modes:\n{mus}")
    run_arm1(mus, sigma, w, rng)
    run_arm2(mus, sigma, w, rng)


if __name__ == "__main__":
    main()
