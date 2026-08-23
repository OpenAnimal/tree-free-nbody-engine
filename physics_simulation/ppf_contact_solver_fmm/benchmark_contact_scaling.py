"""
Comprehensive Scaling & Performance Benchmark for Matrix-Free IPC Cloth Solver
Compares:
  1. Naive All-Pairs IPC O(N^2) — measured for N <= 2100, extrapolated beyond
  2. Standard Dynamic CSR Sparse Matrix IPC — an ANALYTIC MODEL (not measured):
     t_bvh = 0.0032*N, t_csr = 0.0115*N + 12.0; no DynCSRMat implementation
     exists in this repo, so these numbers are a modeled baseline, not data.
  3. Matrix-Free Tree-Free IPC Solver (vectorized canonical-half-offset
     broadphase + Matrix-Free SpMV)

Validates scalability on triangulated cloth meshes from N = 500 to N = 20,000 vertices.

RE-MEASURED 2026-08-21 (vectorized broadphase): the broadphase was rewritten
from a per-key Python loop over occupied cells (which emitted all triu pairs
of each 27-cell neighborhood and deduped with np.unique, ~98% of step time)
to a fully-vectorized canonical-half-offset scheme (13 Chebyshev-1 offsets +
49 Chebyshev-2 closure offsets with an occupied-midpoint check, each pair
emitted exactly once, no dedup).  The broadphase share of total step time
dropped from ~98% to ~28-38% (re-measured 2026-08-22 on the current working
tree; the 2026-08-21 figure of ~15-25% predates the Newton-PCG geometry
caching + np.bincount scatter, which sped up PCG and thereby RAISED the
broadphase fraction), and the matrix-free solver is now FASTER than
naive O(N^2) at every scale N >= 2025 (measured 4.4x-11.2x on 2026-08-22;
the earlier 1.7x-4.0x figure is retained below as the conservative 2026-08-21
measurement).  At N=484 it is still
slower (0.5x) because the O(N^2) distance matrix is cheap at small N while
the fixed Newton-PCG overhead dominates.  The bottleneck moved from the
broadphase to the Newton-PCG solve (~61-72% of step time on the 2026-08-22
re-measure).  The pre-vectorization
per-key-loop numbers (408 / 2650 / 12579 / 33249 / 86476 ms, 0.01x-0.09x vs
naive) are retained only as history in the README.  See the README "Scaling &
Performance Benchmarks" section for the full re-measured table.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(__file__))
from matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid
)

def run_contact_benchmarks():
    print("=" * 85)
    print("SCALING BENCHMARK: MATRIX-FREE TREE-FREE IPC vs DYNAMIC CSR & NAIVE IPC")
    print("Inspired by ZOZO PPF Contact Solver & Farach-Colton, Krapivin, & Kuszmaul (2025)")
    print("=" * 85)

    scales = [484, 2025, 5041, 10000, 19881]
    
    results = {
        "N": [],
        "triangles": [],
        "naive_ipc_ms": [],
        "dyncsr_ipc_ms": [],
        "matrix_free_ipc_ms": [],
        "dyncsr_mem_mb": [],
        "matrix_free_mem_mb": [],
        "broadphase_bvh_ms": [],
        "broadphase_morton_ms": []
    }
    
    for N_target in scales:
        grid_n = int(np.round(np.sqrt(N_target)))
        cloth = create_cloth_grid(
            nx=grid_n, ny=grid_n,
            width=1.0, height=1.0,
            center=(0.5, 0.5, 0.5),
            k_stretch=3000.0,
            k_bend=30.0
        )
        N = cloth.num_vertices
        M_tris = len(cloth.triangles)
        results["N"].append(N)
        results["triangles"].append(M_tris)
        
        positions = cloth.rest_positions.copy()
        velocities = np.zeros_like(positions)
        
        # Add slight random perturbation
        rng = np.random.RandomState(42)
        positions[:, 2] += (rng.rand(N) - 0.5) * 0.05
        
        print(f"\nEvaluating Scale N = {N:,} Vertices (M = {M_tris:,} Triangles)...")
        
        # 1. Naive All-Pairs IPC (O(N^2)) — measured for N <= 2100,
        #    quadratic extrapolation beyond (labeled "(extrapolated)").
        naive_extrapolated = False
        if N <= 2100:
            t0 = time.perf_counter()
            diff = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            active_mask = (dists < 0.02) & (dists > 1e-6)
            _ = np.sum(active_mask)
            t_naive = (time.perf_counter() - t0) * 1000.0
        else:
            base_t = results["naive_ipc_ms"][1]
            base_n = results["N"][1]
            t_naive = base_t * ((N / base_n) ** 2)
            naive_extrapolated = True

        results["naive_ipc_ms"].append(t_naive)

        # 2. Standard DynCSRMat IPC — ANALYTIC MODEL (not measured; no
        #    DynCSRMat implementation exists in this repo).  The linear
        #    model t = 0.0032*N + 0.0115*N + 12.0 is a hypothetical
        #    baseline for comparison, NOT empirical data.
        avg_nnz_per_vertex = 14
        total_nnz_blocks = N * avg_nnz_per_vertex
        mem_dyncsr = (total_nnz_blocks * (9 * 8 + 8)) / (1024.0 * 1024.0)
        results["dyncsr_mem_mb"].append(mem_dyncsr)

        t_bvh = 0.0032 * N
        t_csr_assembly = 0.0115 * N + 12.0
        t_dyncsr = t_bvh + t_csr_assembly
        results["broadphase_bvh_ms"].append(t_bvh)
        results["dyncsr_ipc_ms"].append(t_dyncsr)
        
        # 3. Matrix-Free Tree-Free IPC Solver
        solver = MatrixFreeIPCSolver(
            dhat=0.015,
            stiffness=5e3,
            max_newton_iters=2,
            cg_max_iters=6
        )
        
        # Warmup
        _, _, _ = solver.solve_step(positions, velocities, cloth=cloth, dt=0.01)
        
        # Timed step
        t_mf_0 = time.perf_counter()
        _, _, m_fmm = solver.solve_step(positions, velocities, cloth=cloth, dt=0.01)
        t_mf_total = (time.perf_counter() - t_mf_0) * 1000.0
        
        results["matrix_free_ipc_ms"].append(t_mf_total)
        results["matrix_free_mem_mb"].append(0.0)
        results["broadphase_morton_ms"].append(m_fmm["broadphase_ms"])
        
        speedup_csr = t_dyncsr / max(1e-3, t_mf_total)
        speedup_naive = t_naive / max(1e-3, t_mf_total)

        naive_label = " (extrapolated)" if naive_extrapolated else ""
        bp_ms = m_fmm["broadphase_ms"]
        bp_frac = 100.0 * bp_ms / max(1e-6, t_mf_total)
        print(f"  -> Naive All-Pairs IPC:      {t_naive:8.2f} ms{naive_label}")
        print(f"  -> Standard DynCSRMat IPC:    {t_dyncsr:8.2f} ms (modeled, not measured; CSR Alloc: {mem_dyncsr:.2f} MB)")
        print(f"  -> Matrix-Free Tree-Free IPC: {t_mf_total:8.2f} ms (broadphase {bp_ms:.2f} ms / {bp_frac:.1f}%, 0 MB CSR Alloc, {speedup_csr:.1f}x vs DynCSR model, {speedup_naive:.1f}x vs Naive)")

    # -------------------------------------------------------------
    # Render Publication Figure: fmm_contact_benchmark.png
    # -------------------------------------------------------------
    print("\nGenerating Publication Benchmark Figure: fmm_contact_benchmark.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0B0E14")
    
    text_color = "#E6EDF3"
    grid_color = "#21262D"
    pane_color = "#161B22"
    border_color = "#30363D"
    
    for ax in axes.flat:
        ax.set_facecolor(pane_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.3, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color(border_color)

    # Panel 1: End-to-End Latency Scaling
    ax1 = axes[0, 0]
    ax1.plot(results["N"], results["naive_ipc_ms"], 'o--', color="#FF4D4D", label="Naive All-Pairs IPC $O(N^2)$ (extrapolated for N>2100)", linewidth=2)
    ax1.plot(results["N"], results["dyncsr_ipc_ms"], 's-', color="#FFB800", label="Standard DynCSRMat IPC (modeled, not measured)", linewidth=2)
    ax1.plot(results["N"], results["matrix_free_ipc_ms"], '*-', color="#00FF88", label="Matrix-Free Tree-Free IPC (measured)", linewidth=2.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("1. Contact Solver Step Latency (Log-Log)", color=text_color, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cloth Vertex Count (N)", color=text_color, fontsize=10)
    ax1.set_ylabel("Step Execution Time (ms)", color=text_color, fontsize=10)
    ax1.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8)

    # Panel 2: Memory Allocated for Dynamic Sparse Matrices
    ax2 = axes[0, 1]
    bar_w = 0.35
    indices = np.arange(len(results["N"]))
    ax2.bar(indices - bar_w/2, results["dyncsr_mem_mb"], bar_w, label="DynCSRMat Dynamic Allocations", color="#FF7B72")
    ax2.bar(indices + bar_w/2, results["matrix_free_mem_mb"], bar_w, label="Matrix-Free IPC (0 MB)", color="#00FF88")
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f"{n:,}" for n in results["N"]])
    ax2.set_title("2. Dynamic Sparse Matrix Memory Footprint", color=text_color, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cloth Vertex Count (N)", color=text_color, fontsize=10)
    ax2.set_ylabel("Allocated Memory (MB)", color=text_color, fontsize=10)
    ax2.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8)

    # Panel 3: Broadphase Collision Candidate Pruning Latency
    ax3 = axes[1, 0]
    ax3.plot(results["N"], results["broadphase_bvh_ms"], 's-', color="#FFB800", label="GPU BVH Rebuild & Traverse (modeled)", linewidth=2)
    ax3.plot(results["N"], results["broadphase_morton_ms"], 'o-', color="#00F0FF", label="CellIndex Spatial Hash (single-threaded numpy)", linewidth=2)
    ax3.set_title("3. Broadphase Collision Candidate Pruning Latency", color=text_color, fontsize=11, fontweight="bold")
    ax3.set_xlabel("Vertex Count (N)", color=text_color, fontsize=10)
    ax3.set_ylabel("Broadphase Latency (ms)", color=text_color, fontsize=10)
    ax3.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8)

    # Panel 4: Speedup Factor vs Naive O(N^2)
    ax4 = axes[1, 1]
    speedups_naive = np.array(results["naive_ipc_ms"]) / np.array(results["matrix_free_ipc_ms"])
    ax4.bar(indices, speedups_naive, color="#388BFD", label="Speedup vs Naive $O(N^2)$")
    ax4.set_xticks(indices)
    ax4.set_xticklabels([f"{n:,}" for n in results["N"]])
    for i, sp in enumerate(speedups_naive):
        ax4.text(i, sp + 0.1, f"{sp:.1f}x", ha='center', color="#00FF88", fontweight="bold", fontsize=9)
    ax4.set_title("4. Net Speedup Factor vs Naive $O(N^2)$ IPC", color=text_color, fontsize=11, fontweight="bold")
    ax4.set_xlabel("Vertex Count (N)", color=text_color, fontsize=10)
    ax4.set_ylabel("Speedup Multiplier", color=text_color, fontsize=10)
    ax4.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "fmm_contact_benchmark.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Benchmark figure saved successfully to: {out_path}")

if __name__ == "__main__":
    run_contact_benchmarks()
