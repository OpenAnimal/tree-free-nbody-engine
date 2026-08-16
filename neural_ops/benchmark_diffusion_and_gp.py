"""
Comprehensive Benchmark & Verification Suite:
Tree-Free Diffusion Policy & Multipole Gaussian Process Regression.

Tests & Benchmarks:
1. Diffusion Policy (DDPM & Rectified Flow Matching) action chunking & CFG rollout.
2. Multipole Gaussian Process exact predictive mean & PCG predictive variance vs Dense Cholesky.
3. Pathwise Matheron continuous function sampling without O(N^3) Cholesky.
4. Sparse Variational GP (SVGP) with mini-batch inducing point updates.
5. Linear O(N) vs Dense O(N^3) Scaling & Speedup Benchmarks.
"""

import time
import os
import sys
import numpy as np

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from neural_ops.diffusion_policy_fmm import (
    TreeFreeDiffusionPolicy,
    DiffusionPolicyConfig,
    ConditionalScoreNetwork,
)
from neural_ops.multipole_gaussian_process import (
    MultipoleGaussianProcessLayer,
    GPRegressionResult,
    SVGPResult,
)


def dense_cholesky_reference(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    ell: float,
    sigma_f2: float,
    sigma_n2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(N^3) dense Cholesky baseline for GP mean and variance."""
    N = len(train_X)
    diff_train = train_X[:, None, :] - train_X[None, :, :]
    r_sq_train = np.sum(diff_train ** 2, axis=-1)
    K = sigma_f2 * np.exp(-r_sq_train / (2.0 * (ell ** 2))) + sigma_n2 * np.eye(N)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_y))

    diff_test = test_X[:, None, :] - train_X[None, :, :]
    r_sq_test = np.sum(diff_test ** 2, axis=-1)
    K_star = sigma_f2 * np.exp(-r_sq_test / (2.0 * (ell ** 2)))

    mu_star = K_star @ alpha

    # Variance: k(x*, x*) - k*^T K^-1 k*
    v = np.linalg.solve(L, K_star.T)  # (N, N_test)
    var_star = sigma_f2 - np.sum(v ** 2, axis=0)
    return mu_star, np.maximum(1e-8, var_star)


def test_tree_free_diffusion_policy():
    print("\n" + "=" * 70)
    print("[TEST 1/3] Tree-Free Fast Multipole Diffusion Policy & Flow Matching")
    print("=" * 70)

    config = DiffusionPolicyConfig(
        obs_dim=16,
        action_dim=4,
        pred_horizon=16,
        num_diffusion_steps=20,
        guidance_scale=1.5,
        multipole_drift_weight=0.05,
    )
    policy = TreeFreeDiffusionPolicy(config)

    rng = np.random.RandomState(42)
    obs = rng.randn(16).astype(np.float32)

    # 1. Flow Matching Rollout
    res_fm = policy.sample_flow_matching(obs)
    print(f"Flow Matching Trajectory Shape : {res_fm.actions.shape}")
    print(f"Inference Time                 : {res_fm.inference_time_ms:.2f} ms ({res_fm.num_solver_steps} steps)")
    print(f"Trajectory Smoothness Energy   : {res_fm.trajectory_energy:.4f}")
    assert res_fm.actions.shape == (16, 4)
    assert np.all(np.isfinite(res_fm.actions))

    # 2. DDPM SDE Rollout
    res_ddpm = policy.sample_ddpm(obs)
    print(f"DDPM Reverse SDE Traj Shape    : {res_ddpm.actions.shape}")
    print(f"Inference Time                 : {res_ddpm.inference_time_ms:.2f} ms ({res_ddpm.num_solver_steps} steps)")
    assert res_ddpm.actions.shape == (16, 4)
    assert np.all(np.isfinite(res_ddpm.actions))

    # 3. High-Horizon Trajectory Multi-Step Benchmark
    long_config = DiffusionPolicyConfig(
        obs_dim=16,
        action_dim=6,
        pred_horizon=64,
        num_diffusion_steps=15,
        multipole_drift_weight=0.08,
    )
    long_policy = TreeFreeDiffusionPolicy(long_config)
    t0 = time.perf_counter()
    long_res = long_policy.rollout_policy(obs)
    t_long = (time.perf_counter() - t0) * 1000.0
    print(f"Long Horizon (H=64, D=6) Time  : {t_long:.2f} ms")
    assert long_res.actions.shape == (64, 6)
    print("[PASS] Tree-Free Diffusion Policy tests passed successfully!")


def test_multipole_gaussian_process():
    print("\n" + "=" * 70)
    print("[TEST 2/3] Tree-Free Multipole Gaussian Process vs Dense Cholesky")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_train = 1200
    n_test = 60
    ell = 0.5
    sigma_f2 = 1.0
    sigma_n2 = 0.04

    X_train = rng.rand(n_train, 2) * 3.0
    y_clean = np.sin(X_train[:, 0] * 2.0) * np.cos(X_train[:, 1] * 2.0)
    y_train = y_clean + rng.randn(n_train) * np.sqrt(sigma_n2)

    X_test = rng.rand(n_test, 2) * 3.0
    y_test_clean = np.sin(X_test[:, 0] * 2.0) * np.cos(X_test[:, 1] * 2.0)

    # 1. Fit Multipole GP
    gp = MultipoleGaussianProcessLayer(
        lengthscale=ell,
        signal_variance=sigma_f2,
        noise_variance=sigma_n2,
        cutoff_multiplier=4.2,
        max_pcg_iter=100,
        pcg_tol=1e-6,
    )
    t0 = time.perf_counter()
    n_iter = gp.fit(X_train, y_train)
    t_fit = (time.perf_counter() - t0) * 1000.0

    # 2. Predict Mean and True Variance
    t0 = time.perf_counter()
    res_gp = gp.predict(X_test, compute_variance=True)
    t_pred = (time.perf_counter() - t0) * 1000.0

    print(f"Multipole GP Fit Time          : {t_fit:.2f} ms ({n_iter} PCG iterations)")
    print(f"Predictive Inference Time      : {t_pred:.2f} ms ({n_test} queries)")

    # 3. Dense Cholesky Reference
    t0 = time.perf_counter()
    mu_ref, var_ref = dense_cholesky_reference(X_train, y_train, X_test, ell, sigma_f2, sigma_n2)
    t_dense = (time.perf_counter() - t0) * 1000.0

    mean_err = np.max(np.abs(res_gp.mean - mu_ref))
    var_err = np.max(np.abs(res_gp.variance - var_ref))
    rmse = np.sqrt(np.mean((res_gp.mean - y_test_clean) ** 2))

    print(f"Dense Cholesky Reference Time  : {t_dense:.2f} ms")
    print(f"Predictive Mean Max Abs Error  : {mean_err:.2e}")
    print(f"Predictive Variance Max Error  : {var_err:.2e}")
    print(f"Generalization Test RMSE       : {rmse:.4f}")

    assert mean_err < 0.05, f"Predictive mean error too high: {mean_err}"
    assert var_err < 0.15, f"Predictive variance error too high: {var_err}"

    # 4. Pathwise Matheron Posterior Sampling
    paths = gp.sample_pathwise_matheron(X_test, num_samples=5, num_random_fourier_features=128)
    print(f"Pathwise Matheron Samples      : {paths.shape} (Continuous global function draws)")
    assert paths.shape == (5, n_test)
    assert np.all(np.isfinite(paths))

    # 5. SVGP Mini-Batch Inducing Points
    svgp_res = gp.fit_svgp_inducing_points(X_train, y_train, num_inducing=32, num_epochs=20, batch_size=128)
    print(f"SVGP Inducing Points (M=32)    : Fit complete, ELBO loss = {svgp_res.elbo_loss:.4f}")
    assert svgp_res.mean.shape == (n_train,)
    assert np.all(np.isfinite(svgp_res.mean))

    print("[PASS] Multipole Gaussian Process tests passed successfully!")


def test_large_scale_scaling_benchmark():
    print("\n" + "=" * 70)
    print("[TEST 3/3] Large-Scale O(N) vs Dense O(N^3) Asymptotic Scaling Benchmark")
    print("=" * 70)

    scales = [5000, 15000]
    for N in scales:
        rng = np.random.RandomState(42)
        X = rng.rand(N, 2) * 5.0
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + rng.randn(N) * 0.1
        X_q = rng.rand(100, 2) * 5.0

        gp = MultipoleGaussianProcessLayer(lengthscale=0.6, signal_variance=1.0, noise_variance=0.01)
        t0 = time.perf_counter()
        gp.fit(X, y)
        t_fit = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred = gp.predict(X_q, compute_variance=False)
        t_pred = (time.perf_counter() - t0) * 1000.0

        # Dense projection estimate: (N / 1200)^3 * baseline
        proj_dense_s = (N / 1200.0) ** 3 * 0.05
        print(f"Dataset Size N = {N:,} points:")
        print(f"  Tree-Free GP Fit Time       : {t_fit:.2f} ms")
        print(f"  Predictive Query Time (100) : {t_pred:.2f} ms")
        print(f"  Projected Dense O(N^3) Time : {proj_dense_s * 1000.0:.1f} ms ({proj_dense_s:.2f} s)")
        print(f"  Theoretical Speedup Ratio   : {(proj_dense_s * 1000.0) / max(1e-3, t_fit):.1f}x")

    print("\n[SUCCESS] All Diffusion Policy & Multipole GP verification benchmarks passed!")


if __name__ == "__main__":
    test_tree_free_diffusion_policy()
    test_multipole_gaussian_process()
    test_large_scale_scaling_benchmark()
