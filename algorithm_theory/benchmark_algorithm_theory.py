"""
Comprehensive Scalability & Verification Benchmark Suite for algorithm_theory.

Benchmarks:
1. Frontier-Clustered SSSP vs Dijkstra Baseline on 3D Manifolds (Duan et al. STOC 2025).
2. Asymmetric Low-Rank Tensor M2L vs Naive Dense Contraction (Laser MM Exponent techniques).
3. Matrix-Free Spectral Meshfree PCG vs Standard CG Residual Convergence.
4. Sublinear Approximate Distance Oracle vs Exact Dijkstra Online Query Latency.

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
from algorithm_theory.spectral_meshfree_laplacian import solve_meshfree_poisson
from algorithm_theory.sublinear_distance_oracle import SublinearDistanceOracle


def run_comprehensive_benchmark():
    print("=" * 80)
    print("STARTING ALGORITHM THEORY & BREAKTHROUGH SCALING BENCHMARK SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Frontier-Clustered SSSP vs Dijkstra
    # -------------------------------------------------------------
    print("\n[1/4] Benchmarking Frontier-Clustered SSSP vs Dijkstra on 3D Surface...")
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
    print("\n[2/4] Benchmarking Low-Rank Multipole Tensor vs Dense Contraction...")
    orders = [2, 3, 4, 5, 6]
    coeffs_count = [(p + 1)**3 for p in orders]
    t_dense_m2l = []
    t_lowrank_m2l = []
    speedups_m2l = []

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
        
        t_dense_m2l.append(td)
        t_lowrank_m2l.append(tlr)
        speedup = td / max(1e-6, tlr)
        speedups_m2l.append(speedup)
        print(f"  Order p={p} (P={p_dim:3d}, Rank={engine.tensor_engine.rank:2d}) | Dense: {td:7.2f} ms | Low-Rank: {tlr:5.2f} ms | Speedup: {speedup:6.1f}x")

    # -------------------------------------------------------------
    # 3. Spectral Meshfree Laplacian Residual Convergence
    # -------------------------------------------------------------
    print("\n[3/4] Benchmarking Meshfree Laplacian Solver Convergence (CG vs PCG)...")
    rng = np.random.RandomState(42)
    n_poisson = 4000
    pts_poisson = rng.uniform(-4, 4, (n_poisson, 3))
    r2 = np.sum(pts_poisson**2, axis=1)
    rhs_poisson = np.exp(-r2 / 3.0) - np.mean(np.exp(-r2 / 3.0))

    _, iters_std, res_hist_std, t_std = solve_meshfree_poisson(
        pts_poisson, rhs_poisson, support_radius=1.2, kappa=0.05, tol=1e-5, max_iters=100, use_preconditioner=False
    )
    _, iters_pcg, res_hist_pcg, t_pcg = solve_meshfree_poisson(
        pts_poisson, rhs_poisson, support_radius=1.2, kappa=0.05, tol=1e-5, max_iters=100, use_preconditioner=True
    )
    print(f"  Standard CG:     {iters_std} iters ({t_std:.1f} ms) | Final Res: {res_hist_std[-1]:.2e}")
    print(f"  Multi-Scale PCG: {iters_pcg} iters ({t_pcg:.1f} ms) | Final Res: {res_hist_pcg[-1]:.2e}")

    # -------------------------------------------------------------
    # 4. Sublinear Distance Oracle Query Latency
    # -------------------------------------------------------------
    print("\n[4/4] Benchmarking Sublinear Approximate Distance Oracle Queries...")
    n_ado_pts = 3000
    ado_pts = rng.uniform(-10, 10, (n_ado_pts, 3))
    oracle = SublinearDistanceOracle(ado_pts, eps=0.15)
    
    n_query_batch = 5000
    u_idx = rng.randint(0, n_ado_pts, n_query_batch)
    v_idx = rng.randint(0, n_ado_pts, n_query_batch)
    query_pairs = np.stack([u_idx, v_idx], axis=-1)

    _, t_ado_query = oracle.query_batch(query_pairs, method="landmark_oracle")
    _, t_emb_query = oracle.query_batch(query_pairs, method="embedding")
    
    ado_qps = n_query_batch / (t_ado_query / 1000.0)
    emb_qps = n_query_batch / (t_emb_query / 1000.0)
    print(f"  Landmark Oracle: {t_ado_query:.2f} ms ({ado_qps:.0f} queries/sec)")
    print(f"  Embedding Norm:  {t_emb_query:.2f} ms ({emb_qps:.0f} queries/sec)")

    # -------------------------------------------------------------
    # Plotting Benchmark Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#58a6ff')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    # Panel 1: SSSP Scaling
    ax1 = axes[0, 0]
    ax1.plot(n_nodes_list, t_dijkstra, 'o--', color='#f85149', label='Classic Dijkstra O(m + n log n)', linewidth=2)
    ax1.plot(n_nodes_list, t_fc_sssp, 's-', color='#2ea043', label='Frontier-Clustered SSSP (Duan STOC 25)', linewidth=2.5)
    ax1.set_xlabel('Number of Manifold Vertices (N)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Frontier-Clustered SSSP vs Dijkstra Barrier')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Panel 2: Low-Rank Tensor Speedup
    ax2 = axes[0, 1]
    bar_width = 0.35
    x_pos = np.arange(len(orders))
    ax2.bar(x_pos - bar_width/2, t_dense_m2l, width=bar_width, color='#ff7b72', label='Dense M2L O(P²)')
    ax2.bar(x_pos + bar_width/2, t_lowrank_m2l, width=bar_width, color='#388bfd', label='Asymmetric Low-Rank M2L O(PR)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"p={p}\n(P={c})" for p, c in zip(orders, coeffs_count)])
    ax2.set_xlabel('Multipole Expansion Order p (Terms P)')
    ax2.set_ylabel('Execution Time (ms, log scale)')
    ax2.set_yscale('log')
    ax2.set_title('Far-Field Tensor M2L Contraction Speedup')
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Panel 3: Spectral Poisson Convergence
    ax3 = axes[1, 0]
    ax3.plot(res_hist_std, 'r--', label=f'Standard CG ({iters_std} iters)', linewidth=1.8)
    ax3.plot(res_hist_pcg, 'g-', label=f'Multi-Scale PCG ({iters_pcg} iters)', linewidth=2.5)
    ax3.set_yscale('log')
    ax3.set_xlabel('Iteration Count')
    ax3.set_ylabel('Relative Residual ||r|| / ||b||')
    ax3.set_title('Spectral Meshfree Laplacian Convergence')
    ax3.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Panel 4: Sublinear Distance Oracle
    ax4 = axes[1, 1]
    methods = ['Classic SSSP\n(Dijkstra)', 'Landmark Oracle\n(O(log 1/ε))', 'Metric Embedding\n(O(1) Vectorized)']
    # Estimated single query latencies in microseconds
    dijkstra_single_us = (t_dijkstra[1] / 1.0) * 1000.0  # Approx single query
    landmark_single_us = (t_ado_query / n_query_batch) * 1000.0
    embedding_single_us = (t_emb_query / n_query_batch) * 1000.0
    
    times_us = [dijkstra_single_us, landmark_single_us, embedding_single_us]
    colors = ['#d29922', '#1f6feb', '#a371f7']
    bars = ax4.bar(methods, times_us, color=colors, width=0.55)
    ax4.set_ylabel('Query Latency (μs, log scale)')
    ax4.set_yscale('log')
    ax4.set_title('Sublinear Distance Oracle Online Query Latency')
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{yval:.2f} μs", ha='center', va='bottom', color='#c9d1d9', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "algorithm_theory_benchmark.png")
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"\n[+] Benchmark visualization saved successfully: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
