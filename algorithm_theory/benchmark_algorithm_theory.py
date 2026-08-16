"""
Comprehensive Scalability & Verification Benchmark Suite for algorithm_theory.

Benchmarks:
1. Frontier-Clustered SSSP vs Dijkstra Baseline on 3D Manifolds (Duan et al. STOC 2025).
2. Asymmetric Low-Rank Tensor M2L vs Naive Dense Contraction (Alman Laser MM Exponents).
3. Non-Uniform FFT Type 1 vs Direct O(N*M) Exponential Sum (Barnett / Greengard NUFFT).
4. Screened Yukawa / Debye-Hückel Electrolyte FMM vs Exact Coulomb (Greengard & Huang).
5. Fast Entropic Optimal Transport (Matrix-Free Sinkhorn) vs Dense Reference.
6. Sublinear Metric Distance Oracle & Effective Resistance Query Throughput.

Outputs:
- Generates publication-ready figure: algorithm_theory/algorithm_theory_benchmark.png
"""

import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm_theory.tree_free_geodesic_fmm import MeshfreeGeodesicSolver
from algorithm_theory.algebraic_multipole_tensor import LowRankFarFieldContraction
from algorithm_theory.non_uniform_fourier_hash import NonUniformFourierHash, direct_nufft_type1_baseline
from algorithm_theory.screened_yukawa_fmm import ScreenedYukawaFMM
from algorithm_theory.optimal_transport_fmm import FastEntropicOptimalTransport, direct_sinkhorn_baseline
from algorithm_theory.sublinear_distance_oracle import SublinearDistanceOracle
from algorithm_theory.network_power_centrality import NetworkPowerCentrality


def run_comprehensive_benchmark():
    print("=" * 80)
    print("STARTING ALGORITHM THEORY & DOMAIN-EXTENDED SCALING BENCHMARK SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Frontier-Clustered SSSP vs Dijkstra
    # -------------------------------------------------------------
    print("\n[1/6] Benchmarking Frontier-Clustered SSSP vs Dijkstra on 3D Surface...")
    n_nodes_list = [1000, 2000, 4000, 8000]
    t_dijkstra = []
    t_fc_sssp = []

    for n in n_nodes_list:
        np.random.seed(42)
        theta = np.random.uniform(0, 2 * np.pi, n)
        u = np.random.uniform(-1, 1, n)
        r = np.sqrt(1 - u**2)
        pts = np.stack([r * np.cos(theta) * 10, r * np.sin(theta) * 10, u * 10], axis=-1).astype(np.float32)

        solver = MeshfreeGeodesicSolver(pts, k_neighbors=12)

        t0 = time.perf_counter()
        _ = solver.solve_geodesic(0, method="dijkstra")
        t_dijkstra.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = solver.solve_geodesic(0, method="frontier_clustered")
        t_fc_sssp.append((time.perf_counter() - t0) * 1000.0)

        print(f"  N={n:5d} | Dijkstra: {t_dijkstra[-1]:6.2f} ms | Frontier-Clustered: {t_fc_sssp[-1]:6.2f} ms")

    # -------------------------------------------------------------
    # 2. Asymmetric Low-Rank Tensor vs Dense Contraction
    # -------------------------------------------------------------
    print("\n[2/6] Benchmarking Low-Rank Multipole Tensor vs Dense Contraction...")
    orders = [2, 3, 4, 5, 6]
    coeffs_count = [(p + 1)**3 for p in orders]
    t_dense_m2l = []
    t_lowrank_m2l = []

    n_src, n_tgt = 80, 80
    for p in orders:
        engine = LowRankFarFieldContraction(order=p, dim=3)
        p_dim = engine.tensor_engine.n_coeffs
        rng = np.random.RandomState(42)
        src_m = rng.randn(n_src, p_dim)
        src_c = rng.uniform(-10, -2, (n_src, 3))
        tgt_c = rng.uniform(2, 10, (n_tgt, 3))

        _, td = engine.contract_clusters(src_m, src_c, tgt_c, method="dense")
        _, tlr = engine.contract_clusters(src_m, src_c, tgt_c, method="low_rank")

        t_dense_m2l.append(td * 1000.0)
        t_lowrank_m2l.append(tlr * 1000.0)
        speedup = td / max(tlr, 1e-9)
        print(f"  Order p={p} (P={p_dim:3d}) | Dense M2L: {td*1000.0:7.2f} ms | Low-Rank M2L: {tlr*1000.0:6.2f} ms | Speedup: {speedup:5.1f}x")

    # -------------------------------------------------------------
    # 3. Non-Uniform FFT (NUFFT) vs Direct O(N*M)
    # -------------------------------------------------------------
    print("\n[3/6] Benchmarking Non-Uniform Fast Fourier Transform (NUFFT)...")
    nufft_sizes = [5000, 10000, 20000, 40000]
    t_nufft_fast = []
    t_nufft_direct = []
    grid_dim = (64, 64)

    for n_pts in nufft_sizes:
        np.random.seed(42)
        pts_2d = (np.random.rand(n_pts, 2) * 2.0 - 1.0) * np.pi
        w_2d = np.random.randn(n_pts) + 1j * np.random.randn(n_pts)

        nufft_mod = NonUniformFourierHash(grid_shape=grid_dim, dim=2, window_width=8)
        t0 = time.perf_counter()
        _ = nufft_mod.type1_nonuniform_to_uniform(pts_2d, w_2d)
        t_f = (time.perf_counter() - t0) * 1000.0
        t_nufft_fast.append(t_f)

        # Baseline on subset
        n_sub = min(1500, n_pts)
        t0 = time.perf_counter()
        _ = direct_nufft_type1_baseline(pts_2d[:n_sub], w_2d[:n_sub], grid_dim)
        t_dir_proj = (time.perf_counter() - t0) * 1000.0 * (n_pts / n_sub)
        t_nufft_direct.append(t_dir_proj)

        print(f"  N={n_pts:5d} | Direct O(N*M): {t_dir_proj:7.2f} ms | Tree-Free NUFFT: {t_f:6.2f} ms | Speedup: {t_dir_proj/max(t_f, 1e-6):5.1f}x")

    # -------------------------------------------------------------
    # 4. Screened Yukawa / Debye-Hückel Electrolyte FMM
    # -------------------------------------------------------------
    print("\n[4/6] Benchmarking Screened Yukawa Electrolyte FMM...")
    yukawa_sizes = [4000, 8000, 16000, 32000]
    t_yukawa_fmm = []
    t_yukawa_direct = []

    for n_ions in yukawa_sizes:
        np.random.seed(42)
        pos = np.random.rand(n_ions, 3) * 6.0
        q = np.random.randn(n_ions) + 1.0

        fmm = ScreenedYukawaFMM(kappa=2.0, cell_size=1.0)
        t0 = time.perf_counter()
        _ = fmm.compute_screened_potential_field(pos, q)
        t_f = (time.perf_counter() - t0) * 1000.0
        t_yukawa_fmm.append(t_f)

        n_sub = min(1500, n_ions)
        t0 = time.perf_counter()
        _ = fmm.direct_evaluate(pos[:n_sub], pos, q)
        t_dir_proj = (time.perf_counter() - t0) * 1000.0 * (n_ions / n_sub)
        t_yukawa_direct.append(t_dir_proj)

        print(f"  N={n_ions:5d} | Direct O(N^2): {t_dir_proj:7.2f} ms | Screened FMM: {t_f:6.2f} ms | Speedup: {t_dir_proj/max(t_f, 1e-6):5.1f}x")

    # -------------------------------------------------------------
    # 5. Fast Entropic Optimal Transport (Sinkhorn)
    # -------------------------------------------------------------
    print("\n[5/6] Benchmarking Fast Entropic Optimal Transport...")
    ot_sizes = [2000, 4000, 8000, 16000]
    t_ot_fast = []
    t_ot_dense = []

    for n_pts in ot_sizes:
        np.random.seed(42)
        s_p = np.random.randn(n_pts, 2) * 0.4 + np.array([-1.0, 0.0])
        t_p = np.random.randn(n_pts, 2) * 0.5 + np.array([+1.0, 0.0])
        s_m = np.random.rand(n_pts) + 0.5
        t_m = np.random.rand(n_pts) + 0.5

        ot_sol = FastEntropicOptimalTransport(regularization_gamma=0.15, max_iterations=20)
        t0 = time.perf_counter()
        _, _, _, iters = ot_sol.solve_transport_plan(s_p, s_m, t_p, t_m)
        t_f = (time.perf_counter() - t0) * 1000.0
        t_ot_fast.append(t_f)

        n_sub = min(1000, n_pts)
        t0 = time.perf_counter()
        _ = direct_sinkhorn_baseline(s_p[:n_sub], s_m[:n_sub], t_p[:n_sub], t_m[:n_sub], gamma=0.15, max_iter=20)
        t_dir_proj = (time.perf_counter() - t0) * 1000.0 * ((n_pts * n_pts) / (n_sub * n_sub))
        t_ot_dense.append(t_dir_proj)

        print(f"  N={n_pts:5d} | Dense Sinkhorn: {t_dir_proj:7.2f} ms | Fast Sinkhorn: {t_f:6.2f} ms | Speedup: {t_dir_proj/max(t_f, 1e-6):5.1f}x")

    # -------------------------------------------------------------
    # 6. Sublinear Metric Distance Oracle & Effective Resistance
    # -------------------------------------------------------------
    print("\n[6/6] Benchmarking Sublinear Metric Oracle & Effective Resistance...")
    np.random.seed(42)
    n_oracle_pts = 8000
    pts_surface = np.random.randn(n_oracle_pts, 3) * 5.0
    oracle = SublinearDistanceOracle(pts_surface, eps=0.15)
    
    n_query_pairs = [1000, 5000, 20000, 50000]
    oracle_latencies = []
    
    for q_count in n_query_pairs:
        u_idx = np.random.randint(0, n_oracle_pts, q_count)
        v_idx = np.random.randint(0, n_oracle_pts, q_count)
        pairs = np.stack([u_idx, v_idx], axis=-1)

        _, t_ms = oracle.query_batch(pairs, method="embedding")
        oracle_latencies.append(t_ms)
        qps = q_count / (t_ms / 1000.0)
        print(f"  Queries={q_count:6d} | Latency: {t_ms:6.2f} ms | Throughput: {qps:10,.0f} queries/sec")

    # -------------------------------------------------------------
    # Visualization: 6-Panel Benchmark Figure
    # -------------------------------------------------------------
    print("\nGenerating 6-Panel Benchmark Visualization Figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    # Panel 1: SSSP
    ax = axes[0, 0]
    ax.plot(n_nodes_list, t_dijkstra, 'o--', color='#d9534f', linewidth=2, markersize=7, label="Dijkstra O(m + n log n)")
    ax.plot(n_nodes_list, t_fc_sssp, 's-', color='#0275d8', linewidth=2.5, markersize=7, label="Frontier-Clustered (O(n))")
    ax.set_title("A. Breaking the SSSP Sorting Barrier\n(Duan et al. STOC 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Manifold Nodes (N)", fontsize=10)
    ax.set_ylabel("Execution Time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Tensor M2L
    ax = axes[0, 1]
    ax.plot(coeffs_count, t_dense_m2l, 'o--', color='#d9534f', linewidth=2, markersize=7, label="Dense M2L O(P²)")
    ax.plot(coeffs_count, t_lowrank_m2l, 's-', color='#5cb85c', linewidth=2.5, markersize=7, label="Low-Rank Tensor M2L")
    ax.set_title("B. Asymmetric Tensor M2L Contraction\n(Alman Laser Exponent Methods)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Expansion Coefficients P = (p + 1)³", fontsize=10)
    ax.set_ylabel("Contraction Time (ms)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: NUFFT
    ax = axes[0, 2]
    ax.plot(nufft_sizes, t_nufft_direct, 'o--', color='#d9534f', linewidth=2, markersize=7, label="Direct O(N*M) Sum")
    ax.plot(nufft_sizes, t_nufft_fast, 's-', color='#6f42c1', linewidth=2.5, markersize=7, label="Tree-Free NUFFT")
    ax.set_title("C. Non-Uniform Fast Fourier Transform\n(Elastic Hash Gridding)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Non-Uniform Points", fontsize=10)
    ax.set_ylabel("Execution Time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Screened Yukawa
    ax = axes[1, 0]
    ax.plot(yukawa_sizes, t_yukawa_direct, 'o--', color='#d9534f', linewidth=2, markersize=7, label="Direct O(N²) Coulomb")
    ax.plot(yukawa_sizes, t_yukawa_fmm, 's-', color='#f0ad4e', linewidth=2.5, markersize=7, label="Screened Yukawa FMM (O(N))")
    ax.set_title("D. Screened Electrolyte Electrostatics\n(Debye-Hückel Screened FMM)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Electrolyte Ions (N)", fontsize=10)
    ax.set_ylabel("Evaluation Time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 5: Optimal Transport
    ax = axes[1, 1]
    ax.plot(ot_sizes, t_ot_dense, 'o--', color='#d9534f', linewidth=2, markersize=7, label="Dense Sinkhorn O(N*M)")
    ax.plot(ot_sizes, t_ot_fast, 's-', color='#20c997', linewidth=2.5, markersize=7, label="Tree-Free Sinkhorn O(N+M)")
    ax.set_title("E. Fast Entropic Optimal Transport\n(Spatial Hash Gaussian Convolution)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Particles (N = Source = Target)", fontsize=10)
    ax.set_ylabel("Transport Solve Time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 6: Oracle Query Throughput
    ax = axes[1, 2]
    qps_rates = [q / (t / 1000.0) / 1e6 for q, t in zip(n_query_pairs, oracle_latencies)]
    bars = ax.bar([f"{q:,}" for q in n_query_pairs], qps_rates, color='#17a2b8', width=0.55, edgecolor='black', alpha=0.85)
    ax.set_title("F. Sublinear Metric Distance Oracle\n(Thorup-Zwick / Elastic Landmark Routing)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch Query Size", fontsize=10)
    ax.set_ylabel("Throughput (Million Queries / sec)", fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}M", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6, axis='y')

    out_path = os.path.join(os.path.dirname(__file__), "algorithm_theory_benchmark.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Comprehensive Benchmark Figure to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
