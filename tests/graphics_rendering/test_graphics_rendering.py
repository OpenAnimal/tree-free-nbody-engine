"""
Unit & Integration Test Suite for Graphics & Real-Time Rendering Suite (`graphics_rendering`).
Tests:
1. Point-Based Global Illumination (Surfel Radiosity GI)
2. Volumetric Ambient Occlusion & Continuous Raymarching (FMM VAO)
3. Gridless Dynamic Irradiance Probe Field (SH L0+L1)
4. Asynchronous Multi-GPU Zero-Copy Streaming Graphics Pipeline
5. Hardware GPU Interop & Zero-Copy Staging (CUDA / Vulkan / DirectX 12)
"""

import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphics_rendering.surfel_radiosity_gi import SurfelRadiosityGI
from graphics_rendering.volumetric_fmm_ao import (
    VolumetricFMMAmbientOcclusion,
    VolumetricSamplingMode,
    SparseVolumetricVoxelGrid,
)
from graphics_rendering.dynamic_irradiance_cache import DynamicIrradianceCache
from graphics_rendering.async_zerocopy_streaming import AsyncZeroCopyGraphicsPipeline
from graphics_rendering.gpu_hardware_interop import (
    GPUBackendAPI,
    HardwareZeroCopyBuffer,
    HardwareVolumetricFieldBuffer,
    HardwareSHProbeBuffer,
    Hardware3DVoxelTextureBuffer,
    HardwareGraphicsBridge,
    pack_volumetric_clusters_gpu_layout,
    pack_sh_probes_gpu_layout,
    unpack_sh_probes_gpu_layout
)

def test_surfel_radiosity():
    print("[+] Testing SurfelRadiosityGI...")
    n_surfels = 500
    rng = np.random.RandomState(42)
    pos = rng.uniform(-5.0, 5.0, (n_surfels, 3)).astype(np.float32)
    norm = rng.normal(0, 1, (n_surfels, 3)).astype(np.float32)
    norm /= np.linalg.norm(norm, axis=-1, keepdims=True)
    alb = rng.uniform(0.3, 0.8, (n_surfels, 3)).astype(np.float32)
    areas = np.full(n_surfels, 0.05, dtype=np.float32)
    emiss = np.zeros((n_surfels, 3), dtype=np.float32)
    emiss[:10] = np.array([5.0, 4.0, 3.0], dtype=np.float32)

    gi = SurfelRadiosityGI(cell_size=2.0)
    res = gi.compute_indirect_bounce(pos, norm, alb, areas, emiss, bounces=2)
    
    assert res["num_surfels"] == n_surfels
    assert res["indirect_radiance"].shape == (n_surfels, 3)
    assert np.all(res["total_radiance"] >= 0.0)
    print(f"    [PASS] 2-Bounce Radiosity evaluated in {res['latency_ms']:.2f} ms")


def test_volumetric_fmm_ao_and_raymarching():
    print("[+] Testing VolumetricFMMAmbientOcclusion & Continuous Raymarching...")
    n_occ = 1000
    n_queries = 200
    n_rays = 100
    rng = np.random.RandomState(42)
    
    p_occ = rng.uniform(-5.0, 5.0, (n_occ, 3)).astype(np.float32)
    r_occ = rng.uniform(0.1, 0.3, n_occ).astype(np.float32)
    opac = rng.uniform(0.5, 1.0, n_occ).astype(np.float32)

    vao = VolumetricFMMAmbientOcclusion(cell_size=1.5)
    vao.insert_occluders(p_occ, r_occ, opac)

    # 1. Point AO Field evaluation
    q_pos = rng.uniform(-4.0, 4.0, (n_queries, 3)).astype(np.float32)
    ao_res = vao.evaluate_ao_field(q_pos)
    assert len(ao_res["ao_values"]) == n_queries
    assert np.all(ao_res["ao_values"] >= 0.0) and np.all(ao_res["ao_values"] <= 1.0)
    print(f"    [PASS] Point AO field evaluated in {ao_res['latency_ms']:.2f} ms")

    # 2. Continuous Volumetric Raymarching (Test all 3 modes: FMM, VOXEL, HYBRID)
    r_orig = rng.uniform(-3.0, 3.0, (n_rays, 3)).astype(np.float32)
    r_dir = rng.normal(0, 1, (n_rays, 3)).astype(np.float32)
    
    # 2a. 3D Voxel Grid rasterization and raymarch
    v_grid = vao.build_voxel_grid(grid_resolution=32)
    assert v_grid.density_grid.shape == (32, 32, 32)
    tex3d = v_grid.export_texture3d_layout()
    assert tex3d.shape == (32, 32, 32, 4)

    # Voxel raymarch
    v_ray_res = vao.sample_volumetric_ray_transmittance(
        r_orig, r_dir,
        step_size=0.3,
        max_steps=8,
        extinction_coeff=1.0,
        light_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        mode=VolumetricSamplingMode.VOXEL_ONLY
    )
    assert len(v_ray_res["transmittance"]) == n_rays
    assert v_ray_res["mode"] == "VOXEL_ONLY"

    # Hybrid raymarch
    h_ray_res = vao.sample_volumetric_ray_transmittance(
        r_orig, r_dir,
        step_size=0.3,
        max_steps=8,
        extinction_coeff=1.0,
        light_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        mode=VolumetricSamplingMode.HYBRID
    )
    assert len(h_ray_res["transmittance"]) == n_rays
    assert h_ray_res["mode"] == "HYBRID"

    # FMM raymarch
    ray_res = vao.sample_volumetric_ray_transmittance(
        r_orig, r_dir,
        step_size=0.3,
        max_steps=8,
        extinction_coeff=1.0,
        light_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        mode=VolumetricSamplingMode.FMM_ONLY
    )
    assert len(ray_res["transmittance"]) == n_rays
    assert ray_res["inscattered_radiance"].shape == (n_rays, 3)
    assert np.all(ray_res["transmittance"] >= 0.0) and np.all(ray_res["transmittance"] <= 1.0)
    print(f"    [PASS] Volumetric raymarching in 3 modes (Voxel: {v_ray_res['latency_ms']:.2f}ms, Hybrid: {h_ray_res['latency_ms']:.2f}ms, FMM: {ray_res['latency_ms']:.2f}ms)")

    # 3. GPU Cluster Buffer Export
    gpu_buf = vao.export_gpu_cluster_buffer()
    assert gpu_buf.shape == (len(vao.macro_clusters), 2, 4)
    print(f"    [PASS] Exported GPU cluster layout ({gpu_buf.shape})")


def test_volumetric_ao_fmm_field():
    """Round-7 task X-G1: FMM-accelerated AO far field.

    Validates that ``evaluate_ao_field_fmm`` (vectorized Barnes-Hut near/far
    split with the actual AO kernel) matches ``evaluate_ao_exact``
    (per-particle O(Q*N) reference) within the plan's <= 5e-2 rel-L2
    acceptance gate on a 5k-point cloud, and reports the accuracy vs the
    all-cluster baseline.
    """
    print("[+] Testing X-G1: FMM-accelerated AO field (Barnes-Hut near/far)...")
    rng = np.random.RandomState(42)
    n_occ = 5000
    n_queries = 5000
    p_occ = rng.uniform(-8.0, 8.0, (n_occ, 3)).astype(np.float32)
    r_occ = rng.uniform(0.1, 0.4, n_occ).astype(np.float32)
    opac = rng.uniform(0.5, 1.0, n_occ).astype(np.float32)

    vao = VolumetricFMMAmbientOcclusion(cell_size=1.5, occlusion_radius=10.0)
    vao.insert_occluders(p_occ, r_occ, opac)

    q_pos = rng.uniform(-6.0, 6.0, (n_queries, 3)).astype(np.float32)

    # Exact reference (O(Q*N) — slow but ground truth).
    exact = vao.evaluate_ao_exact(q_pos)

    # FMM-accelerated field.
    fmm_res = vao.evaluate_ao_field_fmm(q_pos, depth=8, p=6, near_ring=1)
    fmm_ao = fmm_res["ao_values"].astype(np.float64)

    # All-cluster monopole baseline (the current production path).
    cluster_keys = list(vao.macro_clusters.keys())
    centers = np.stack([vao.macro_clusters[k]["center"] for k in cluster_keys], axis=0)
    masses = np.array([vao.macro_clusters[k]["mass"] for k in cluster_keys], dtype=np.float32)
    radii = np.array([vao.macro_clusters[k]["eff_radius"] for k in cluster_keys], dtype=np.float32)
    all_cluster_ao = vao._evaluate_ao_cpu(q_pos, centers, masses, radii, chunk_size=4096).astype(np.float64)

    # rel-L2 vs exact.
    exact_norm = np.linalg.norm(exact)
    rel_l2_fmm = np.linalg.norm(fmm_ao - exact) / max(exact_norm, 1e-12)
    rel_l2_all = np.linalg.norm(all_cluster_ao - exact) / max(exact_norm, 1e-12)

    print(f"    FMM rel-L2 vs exact: {rel_l2_fmm:.4e} (gate <= 5e-2)")
    print(f"    All-cluster rel-L2 vs exact: {rel_l2_all:.4e}")
    print(f"    FMM latency: {fmm_res['latency_ms']:.2f} ms, backend={fmm_res['backend_used']}")

    # The Barnes-Hut near/far path is more accurate than the all-cluster
    # monopole baseline because the near field is exact per-particle (not
    # monopole).  The gate is 5e-2 rel-L2 (the plan's acceptance criterion;
    # visual metric — AO is a heuristic).  The method's value is ACCURACY
    # (exact near field), not asymptotic speedup — see the complexity
    # caveat in the method docstring.
    assert rel_l2_fmm < 5e-2, \
        f"FMM AO rel-L2 {rel_l2_fmm:.4e} too high (gate 5e-2)"
    # The near/far split should also be more accurate than the all-cluster
    # baseline (exact near field vs monopole near field).
    assert rel_l2_fmm <= rel_l2_all, \
        f"near/far rel-L2 {rel_l2_fmm:.4e} worse than all-cluster {rel_l2_all:.4e}"
    print(f"    [PASS] X-G1 Barnes-Hut AO field within tolerance "
          f"(rel-L2={rel_l2_fmm:.4e} <= 5e-2, all-cluster={rel_l2_all:.4e})")


def test_dynamic_irradiance_cache():
    print("[+] Testing DynamicIrradianceCache...")
    n_probes = 256
    n_verts = 500
    rng = np.random.RandomState(42)

    probe_pos = rng.uniform(-10.0, 10.0, (n_probes, 3)).astype(np.float32)
    probe_l0 = rng.uniform(0.1, 1.0, (n_probes, 3)).astype(np.float32)
    probe_l1 = rng.uniform(-0.4, 0.4, (n_probes, 3, 3)).astype(np.float32)

    cache = DynamicIrradianceCache(cell_size=3.0)
    cache.update_probe_field(probe_pos, probe_l0, probe_l1)

    v_pos = rng.uniform(-8.0, 8.0, (n_verts, 3)).astype(np.float32)
    v_norm = rng.normal(0, 1, (n_verts, 3)).astype(np.float32)
    v_norm /= np.linalg.norm(v_norm, axis=1, keepdims=True)

    irr_res = cache.query_actor_irradiance(v_pos, v_norm)
    assert irr_res["irradiance"].shape == (n_verts, 3)
    assert np.all(irr_res["irradiance"] >= 0.0)

    # Test GPU packing & unpacking
    gpu_probes = cache.export_gpu_probe_buffer()
    assert gpu_probes.shape == (n_probes, 5, 4)

    up_pos, up_l0, up_l1, up_rad = unpack_sh_probes_gpu_layout(gpu_probes)
    np.testing.assert_allclose(up_pos, probe_pos, atol=1e-5)
    np.testing.assert_allclose(up_l0, probe_l0, atol=1e-5)
    np.testing.assert_allclose(up_l1, probe_l1, atol=1e-5)
    print(f"    [PASS] Dynamic Irradiance Cache queried and packed (5x float4) in {irr_res['latency_ms']:.2f} ms")


def test_dynamic_irradiance_near_far_accuracy():
    """Round-8 audit: pin the documented failure modes of
    ``query_actor_irradiance_near_far`` (the hash-driven near-field path).

    1. Empty-neighborhood fallback equivalence: a vertex whose 27-cell
       neighborhood contains no probe must fall back to the brute
       all-probe Gaussian query and match ``query_actor_irradiance``
       EXACTLY (the method is a cache, not a crop -- it must not return
       black where brute returns a nonzero value).

    2. Ring-truncation error ceiling: on a dense-probe scene the
       per-vertex L2 relative error vs the brute all-probe query must
       stay below a stated, measured threshold.  Measured 2026-08-20 on
       2048 probes / 40^3 volume / cell_size=3.0 / 10000 query vertices:
         ring=1 -> max per-vertex L2 rel err = 0.301  (threshold 0.65, ~2.16x)
         ring=2 -> max per-vertex L2 rel err = 0.036  (threshold 0.10, ~2.8x)
       The thresholds give ~2x headroom above the measured maxima while
       still catching a real regression (a broken neighborhood scan or a
       reintroduced directional-axis bug would blow past them by orders
       of magnitude).
    """
    print("[+] Testing DynamicIrradianceCache near_far accuracy + fallback...")
    rng = np.random.RandomState(42)
    n_probes = 2048
    n_verts = 10000
    probe_pos = rng.uniform(-20.0, 20.0, (n_probes, 3)).astype(np.float32)
    probe_l0 = rng.uniform(0.2, 1.5, (n_probes, 3)).astype(np.float32)
    probe_l1 = rng.uniform(-0.5, 0.5, (n_probes, 3, 3)).astype(np.float32)
    cache = DynamicIrradianceCache(cell_size=3.0)
    cache.update_probe_field(probe_pos, probe_l0, probe_l1)
    v_pos = rng.uniform(-15.0, 15.0, (n_verts, 3)).astype(np.float32)
    v_norm = rng.normal(0, 1, (n_verts, 3)).astype(np.float32)
    v_norm /= np.linalg.norm(v_norm, axis=1, keepdims=True)

    brute = cache.query_actor_irradiance(v_pos, v_norm)["irradiance"]
    bnorm = np.linalg.norm(brute, axis=1)

    # --- (1) Empty-neighborhood fallback equivalence ---
    # Sparse-probe scene so some query vertices have an empty 27-cell
    # neighborhood but nearby probes still contribute a nonzero brute value.
    rng2 = np.random.RandomState(123)
    sp_pos = rng2.uniform(-20.0, 20.0, (300, 3)).astype(np.float32)
    sp_l0 = rng2.uniform(0.2, 1.5, (300, 3)).astype(np.float32)
    sp_l1 = rng2.uniform(-0.5, 0.5, (300, 3, 3)).astype(np.float32)
    sparse = DynamicIrradianceCache(cell_size=3.0)
    sparse.update_probe_field(sp_pos, sp_l0, sp_l1)
    qv_pos = rng2.uniform(-15.0, 15.0, (2000, 3)).astype(np.float32)
    qv_norm = rng2.normal(0, 1, (2000, 3)).astype(np.float32)
    qv_norm /= np.linalg.norm(qv_norm, axis=1, keepdims=True)
    empty_idx = []
    for i in range(len(qv_pos)):
        qk = sparse.index.key_of(qv_pos[i])
        nk = sparse.index.neighbor_keys(qk, ring=1)
        if not any(int(k) in sparse.cell_probe_map for k in nk):
            empty_idx.append(i)
    assert len(empty_idx) > 0, "test scene must contain at least one empty-neighborhood vertex"
    ei = np.array(empty_idx, dtype=np.int64)
    brute_empty = sparse.query_actor_irradiance(qv_pos[ei], qv_norm[ei])["irradiance"]
    nf_empty = sparse.query_actor_irradiance_near_far(qv_pos[ei], qv_norm[ei], ring=1)
    # Fallback must match brute exactly (same kernel over all probes).
    np.testing.assert_allclose(nf_empty, brute_empty, atol=1e-6, rtol=0,
                               err_msg="near_far empty-neighborhood fallback != brute all-probe query")
    # And the brute value must be nonzero for at least one (else the test
    # would trivially pass with a black fallback).
    assert np.any(np.linalg.norm(brute_empty, axis=1) > 1e-4), \
        "empty-neighborhood brute values all ~0; fallback equivalence is vacuous"
    print(f"    [PASS] empty-neighborhood fallback matches brute exactly "
          f"({len(empty_idx)} empty-neighborhood vertices)")

    # --- (2) Ring-truncation error ceiling ---
    for ring, thresh in ((1, 0.65), (2, 0.10)):
        nf = cache.query_actor_irradiance_near_far(v_pos, v_norm, ring=ring)
        rel = np.linalg.norm(nf - brute, axis=1) / np.maximum(bnorm, 1e-6)
        measured_max = float(rel.max())
        assert measured_max < thresh, \
            f"ring={ring} max per-vertex L2 rel err {measured_max:.4f} >= threshold {thresh}"
        print(f"    [PASS] ring={ring} max per-vertex L2 rel err = {measured_max:.4f} "
              f"(threshold {thresh})")


def test_hardware_gpu_interop():
    print("[+] Testing Hardware GPU Interop & Zero-Copy Staging...")
    bridge = HardwareGraphicsBridge(max_elements=1000, backend=GPUBackendAPI.HOST_SHARED)
    rng = np.random.RandomState(42)
    pos = rng.randn(500, 3).astype(np.float32)
    norm = pos / (np.linalg.norm(pos, axis=-1, keepdims=True) + 1e-12)

    staged = bridge.stage_geometry_for_gpu(pos, norm)
    assert staged["elements_staged"] == 500
    bridge.complete_gpu_frame(staged["active_slot"])

    # Test dedicated volumetric, probe, and 3D voxel texture hardware buffers
    vol_buf = HardwareVolumetricFieldBuffer(max_clusters=128, backend=GPUBackendAPI.CUDA)
    dummy_cl = {
        0: {"center": np.array([1.0, 2.0, 3.0], dtype=np.float32), "mass": 5.0, "eff_radius": 0.5}
    }
    staged_cl = vol_buf.stage_clusters(dummy_cl, cell_size=1.5)
    assert staged_cl == 1
    assert vol_buf.get_pointer() > 0

    sh_buf = HardwareSHProbeBuffer(max_probes=128, backend=GPUBackendAPI.VULKAN)
    staged_pr = sh_buf.stage_probes(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        np.zeros((1, 3, 3), dtype=np.float32),
        probe_radius=2.0
    )
    assert staged_pr == 1
    assert sh_buf.get_pointer() > 0

    vox_tex_buf = Hardware3DVoxelTextureBuffer(depth=16, height=16, width=16, backend=GPUBackendAPI.DIRECTX12)
    dummy_grid = np.ones((16, 16, 16), dtype=np.float32) * 0.5
    staged_vox = vox_tex_buf.stage_voxel_grid(dummy_grid)
    assert staged_vox == 16 * 16 * 16
    assert vox_tex_buf.get_pointer() > 0
    print("    [PASS] Zero-copy hardware staging (Buffers, SH Probes, and 3D Voxel Texture) verified.")


def test_async_zerocopy_streaming():
    print("[+] Testing AsyncZeroCopyGraphicsPipeline...")
    pipeline = AsyncZeroCopyGraphicsPipeline(max_elements=5000, tile_depth=2)
    rng = np.random.RandomState(42)
    N = 1000
    pos = rng.uniform(0.1, 0.9, (N, 3)).astype(np.float32)
    norm = rng.normal(0, 1, (N, 3)).astype(np.float32)
    norm /= np.linalg.norm(norm, axis=-1, keepdims=True)
    alb = rng.uniform(0.3, 0.8, (N, 3)).astype(np.float32)

    dirty = pipeline.update_dynamic_geometry_async(pos, norm, alb)
    assert dirty > 0

    radiance, stats = pipeline.render_frame_radiance(N)
    assert radiance.shape == (N, 3)
    assert stats.fps_estimate > 0.0
    print(f"    [PASS] Streaming pipeline evaluated {N} surfels at {stats.fps_estimate:.1f} FPS")


if __name__ == "__main__":
    print("==================================================================")
    print(" GRAPHICS RENDERING SUITE: UNIT & INTEGRATION TEST HARNESS")
    print("==================================================================")
    t0 = time.perf_counter()
    test_surfel_radiosity()
    test_volumetric_fmm_ao_and_raymarching()
    test_volumetric_ao_fmm_field()
    test_dynamic_irradiance_cache()
    test_dynamic_irradiance_near_far_accuracy()
    test_hardware_gpu_interop()
    test_async_zerocopy_streaming()
    t_total = (time.perf_counter() - t0) * 1000.0
    print("==================================================================")
    print(f"ALL GRAPHICS RENDERING TESTS PASSED in {t_total:.2f} ms")
    print("==================================================================")
