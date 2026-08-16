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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    test_dynamic_irradiance_cache()
    test_hardware_gpu_interop()
    test_async_zerocopy_streaming()
    t_total = (time.perf_counter() - t0) * 1000.0
    print("==================================================================")
    print(f"ALL GRAPHICS RENDERING TESTS PASSED in {t_total:.2f} ms")
    print("==================================================================")
