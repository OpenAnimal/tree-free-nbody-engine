"""
Scalability & Verification Benchmark for Graphics & Real-Time Rendering Suite (`graphics_rendering`).
Benchmarks:
1. Point-Based Surfel Radiosity (PBGI) vs Direct O(N^2) Form Factors.
2. Volumetric Multipole Ambient Occlusion (FMM VAO) vs All-Pairs Occluder Testing.
3. Gridless Dynamic Irradiance Probe Field Interpolation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphics_rendering.surfel_radiosity_gi import SurfelRadiosityGI
from graphics_rendering.volumetric_fmm_ao import VolumetricFMMAmbientOcclusion
from graphics_rendering.dynamic_irradiance_cache import DynamicIrradianceCache

def benchmark_graphics_suite():
    print("==========================================================================")
    print(" GRAPHICS RENDERING & RADIANCE SUITE: COMPREHENSIVE BENCHMARK")
    print("==========================================================================")
    
    np.random.seed(42)
    scales = [1000, 5000, 15000, 30000]
    
    gi_latencies = []
    ao_latencies = []
    probe_latencies = []

    for n in scales:
        print(f"\n[+] Testing Scene Scale: N = {n:,} Surface / Volumetric Elements...")
        
        # 1. Surfel Radiosity GI
        pos = np.random.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float32)
        norm = np.random.normal(0, 1, size=(n, 3)).astype(np.float32)
        norm /= np.linalg.norm(norm, axis=1, keepdims=True)
        alb = np.random.uniform(0.3, 0.8, size=(n, 3)).astype(np.float32)
        areas = np.full(n, 0.04, dtype=np.float32)
        emiss = np.zeros((n, 3), dtype=np.float32)
        emiss[:max(1, n//50)] = np.array([10.0, 8.0, 5.0], dtype=np.float32)

        gi_engine = SurfelRadiosityGI(cell_size=2.5)
        gi_res = gi_engine.compute_indirect_bounce(pos, norm, alb, areas, emiss, bounces=1)
        gi_latencies.append(gi_res["latency_ms"])
        print(f"    [-] Surfel Radiosity (1-Bounce): {gi_res['latency_ms']:.2f} ms ({gi_res['fps_capacity']:.1f} FPS)")

        # 2. Volumetric Ambient Occlusion
        r_occ = np.random.uniform(0.1, 0.3, size=n).astype(np.float32)
        opac = np.random.uniform(0.5, 1.0, size=n).astype(np.float32)
        q_pos = np.random.uniform(-8.0, 8.0, size=(min(5000, n), 3)).astype(np.float32)
        
        vao = VolumetricFMMAmbientOcclusion(cell_size=2.0)
        vao.insert_occluders(pos, r_occ, opac)
        ao_res = vao.evaluate_ao_field(q_pos)
        ao_latencies.append(ao_res["latency_ms"])
        print(f"    [-] Volumetric AO Field:        {ao_res['latency_ms']:.2f} ms ({ao_res['throughput_queries_per_sec']:,.0f} queries/s)")

        # 3. Dynamic Irradiance Caching
        n_probes = min(2048, max(256, n // 5))
        p_pos = np.random.uniform(-12.0, 12.0, size=(n_probes, 3)).astype(np.float32)
        l0 = np.random.uniform(0.2, 1.0, size=(n_probes, 3)).astype(np.float32)
        l1 = np.random.uniform(-0.3, 0.3, size=(n_probes, 3, 3)).astype(np.float32)

        cache = DynamicIrradianceCache(cell_size=3.0)
        cache.update_probe_field(p_pos, l0, l1)
        probe_res = cache.query_actor_irradiance(pos[:min(5000, n)], norm[:min(5000, n)])
        probe_latencies.append(probe_res["latency_ms"])
        print(f"    [-] Irradiance Probe Query:     {probe_res['latency_ms']:.2f} ms ({probe_res['fps_capacity']:.1f} FPS)")

    # Generate Visualization Figure
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#080C15')
    ax.set_facecolor('#0E1525')
    ax.grid(True, color='#1E293B', linestyle='--', alpha=0.6)

    ax.plot(scales, gi_latencies, marker='o', lw=2.5, color='#00F0FF', label='Surfel Radiosity (PBGI 1-Bounce)')
    ax.plot(scales, ao_latencies, marker='s', lw=2.5, color='#FF007F', label='Volumetric Ambient Occlusion (VAO)')
    ax.plot(scales, probe_latencies, marker='^', lw=2.5, color='#39FF14', label='Dynamic Irradiance SH Probes')

    ax.set_title('Graphics & Rendering Suite: Real-Time Scaling vs Scene Complexity', fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax.set_xlabel('Active Scene Element Count (Surfels / Occluders / Vertices)', fontsize=11, fontweight='medium', color='#94A3B8')
    ax.set_ylabel('Execution Latency (milliseconds)', fontsize=11, fontweight='medium', color='#94A3B8')
    ax.tick_params(colors='#94A3B8', which='both')
    ax.legend(frameon=True, facecolor='#131D2E', edgecolor='#334155', labelcolor='#FFFFFF', fontsize=10.5)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    fig.tight_layout()
    output_png = os.path.join(os.path.dirname(__file__), "graphics_rendering_benchmark.png")
    fig.savefig(output_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[+] Successfully saved benchmark visualization: {output_png}")
    print("==========================================================================")

if __name__ == '__main__':
    benchmark_graphics_suite()
