"""
Comprehensive AMD / ATI Radeon & Cross-Platform Compliance Test Suite
======================================================================
Verifies hardware accelerator detection, AMD ROCm / HIP kernel headers,
OpenCL C compute kernels, WebGPU WGSL shaders, graphics zero-copy interop,
and AMD AMF video codec bridges.
"""

import unittest
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.device_runtime import (
    DeviceRuntime,
    AcceleratorVendor,
    ComputeBackend,
    DeviceDescriptor,
)
from core.hip_kernels import (
    HIP_KERNEL_SOURCE_PATH,
    get_hip_kernel_source,
)
from core.opencl_kernels import (
    OPENCL_CL_SOURCE_PATH,
    OpenCLFMMContext,
    is_opencl_available,
)
from core.webgpu_kernels import (
    WGSL_SOURCE_PATH,
    get_wgsl_source,
    is_webgpu_available,
)
from graphics_rendering.gpu_hardware_interop import (
    GPUBackendAPI,
    HardwareZeroCopyBuffer,
    HardwareVolumetricFieldBuffer,
    HardwareSHProbeBuffer,
    HardwareGraphicsBridge,
)
from video_streaming_codecs.ffmpeg_interop_bridge import (
    FFmpegInteropBridge,
)


class TestAMDRadeonCompliance(unittest.TestCase):
    def test_device_runtime_discovery(self):
        """Test device runtime accelerator discovery and ranking."""
        devices = DeviceRuntime.get_available_devices(force_refresh=True)
        self.assertGreater(len(devices), 0, "At least one compute device (e.g. CPU or GPU) should be discovered.")
        
        optimal = DeviceRuntime.get_optimal_device()
        self.assertIsNotNone(optimal)
        self.assertIsInstance(optimal, DeviceDescriptor)
        self.assertIn(optimal.backend, list(ComputeBackend))

    def test_rocm_hip_kernel_integrity(self):
        """Test that the native ROCm/HIP kernel source exists and contains AMD wavefront / LDS instructions."""
        self.assertTrue(os.path.exists(HIP_KERNEL_SOURCE_PATH), "tree_free_fmm_kernel.hip must exist.")
        src = get_hip_kernel_source()
        self.assertIn("hip_morton_encode_3d", src)
        self.assertIn("hip_evaluate_near_field_p2p_kernel", src)
        self.assertIn("hip_compute_cluster_multipoles_kernel", src)
        self.assertIn("AMD_WAVE_SIZE", src)
        self.assertIn("lds_particles", src)
        self.assertIn("atomicCAS", src)

    def test_opencl_kernel_integrity(self):
        """Test that the OpenCL C compute kernels exist and define required entrypoints."""
        self.assertTrue(os.path.exists(OPENCL_CL_SOURCE_PATH), "tree_free_fmm_opencl.cl must exist.")
        with open(OPENCL_CL_SOURCE_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("opencl_morton_encode_3d", src)
        self.assertIn("opencl_p2p_coulomb_nbody", src)
        self.assertIn("opencl_volumetric_ao_sample", src)
        self.assertIn("local_particles", src)

        ctx = OpenCLFMMContext.get_instance(prefer_amd=True)
        info = ctx.get_device_info()
        self.assertIn(info["status"], ["READY", "UNAVAILABLE"])

    def test_webgpu_wgsl_integrity(self):
        """Test WebGPU WGSL compute shader source structure (T-E1 CSR form:
        counting-sort cell lists replaced the old shared-memory tile scan)."""
        self.assertTrue(os.path.exists(WGSL_SOURCE_PATH), "tree_free_fmm.wgsl must exist.")
        wgsl = get_wgsl_source()
        self.assertIn("@compute", wgsl)
        self.assertIn("@workgroup_size", wgsl)
        self.assertIn("scan_cells", wgsl)          # CSR prefix-sum pass
        self.assertIn("cellStart", wgsl)           # CSR cell ranges
        self.assertIn("sortedIndex", wgsl)         # CSR particle order
        self.assertIn("atomicAdd", wgsl)           # counting-sort scatter
        self.assertIn("inverseSqrt", wgsl)

    def test_graphics_hardware_interop_backends(self):
        """Test that AMD ROCm, DirectML, OpenCL, AMF, and Metal are first-class GPUBackendAPI members."""
        expected_backends = ["CUDA", "ROCM_HIP", "DIRECTML", "OPENCL", "VULKAN", "DIRECTX12", "AMF", "METAL", "HOST_SHARED"]
        for b in expected_backends:
            self.assertIn(b, [member.value for member in GPUBackendAPI])

        # Test zero-copy buffer creation with ROCm and DirectML backend tags
        rocm_buf = HardwareZeroCopyBuffer(1024, np.float32, channels=3, backend=GPUBackendAPI.ROCM_HIP)
        self.assertEqual(rocm_buf.descriptor.backend, GPUBackendAPI.ROCM_HIP)
        self.assertGreater(rocm_buf.get_pointer(), 0)

        dml_buf = HardwareZeroCopyBuffer(1024, np.float32, channels=3, backend=GPUBackendAPI.DIRECTML)
        self.assertEqual(dml_buf.descriptor.backend, GPUBackendAPI.DIRECTML)

        vol_buf = HardwareVolumetricFieldBuffer(max_clusters=512, backend=GPUBackendAPI.OPENCL)
        self.assertEqual(vol_buf.buffer.descriptor.backend, GPUBackendAPI.OPENCL)

    def test_ffmpeg_amd_amf_encoders(self):
        """Test AMD AMF hardware encoder registration and optimal CLI argument
        generation. AMD-hardware-dependent: the bridge only registers AMF /
        VAAPI profiles when the local ffmpeg build exposes them, so skip
        (don't fail) on CPU-only hosts."""
        probe = FFmpegInteropBridge()
        has_amd = any(
            name.endswith(("_amf", "_vaapi"))
            for name in probe.available_encoders
        )
        if not has_amd:
            self.skipTest(
                "no AMD AMF/VAAPI encoders in the local ffmpeg build "
                f"(available: {sorted(probe.available_encoders)})"
            )
        bridge = FFmpegInteropBridge()
        
        # Verify AMD AMF profiles are registered
        self.assertIn("av1_amf", bridge.available_encoders)
        self.assertIn("hevc_amf", bridge.available_encoders)
        self.assertIn("h264_amf", bridge.available_encoders)
        self.assertIn("av1_vaapi", bridge.available_encoders)
        self.assertIn("hevc_vaapi", bridge.available_encoders)
        self.assertIn("h264_vaapi", bridge.available_encoders)

        # Generate AMD AMF AV1 plan
        plan_amf = bridge.generate_encoding_plan(
            input_spec="pipe:0",
            output_path="test_amd.mp4",
            codec="av1",
            prefer_hardware=True,
            prefer_amd=True
        )
        self.assertEqual(plan_amf.encoder_flag, "av1_amf")
        self.assertTrue(plan_amf.is_hardware)
        cmd_str = " ".join(plan_amf.generated_cli_command)
        self.assertIn("-c:v av1_amf", cmd_str)
        self.assertIn("-quality quality", cmd_str)
        self.assertIn("-rc cqp", cmd_str)

        # Generate AMD AMF HEVC plan
        plan_hevc_amf = bridge.generate_encoding_plan(
            input_spec="pipe:0",
            output_path="test_hevc_amd.mp4",
            codec="hevc",
            prefer_hardware=True,
            prefer_amd=True
        )
        self.assertEqual(plan_hevc_amf.encoder_flag, "hevc_amf")
        self.assertIn("-c:v hevc_amf", " ".join(plan_hevc_amf.generated_cli_command))

    def test_opencl_morton_encode_lazy_program_no_attribute_error(self):
        """Regression (audit finding #2): `opencl_morton_encode_3d` must not
        raise `AttributeError: 'NoneType' object has no attribute
        'opencl_morton_encode_3d'` when called on a fresh OpenCL context
        that has not yet run `opencl_tree_free_nbody`.

        ROOT CAUSE: the Round-7 lazy-program refactor stopped building
        `self.program` in `_initialize_context`; programs are now built /
        cached per workgroup_size via `ctx.get_program(wg)`. The morton
        encode entrypoint still referenced `ctx.program` directly, so the
        first call on a fresh context hit `ctx.program is None`. The fix
        fetches the workgroup-256 program explicitly via `get_program(256)`.

        This test is skipped silently when OpenCL is unavailable (the
        function-under-test raises `RuntimeError("OpenCL is not
        available.")`, which we interpret as a skip, not a failure, so the
        suite stays green on hosts without an OpenCL ICD)."""
        if not is_opencl_available():
            self.skipTest("OpenCL is not available on this host.")
        # Import lazily so the module load itself does not require OpenCL.
        from core.opencl_kernels.opencl_fmm_backend import opencl_morton_encode_3d
        # Reset the singleton so we exercise the fresh-context path
        # (the regression only manifests when `ctx.program` was never built).
        try:
            OpenCLFMMContext.reset_instance()
        except AttributeError:
            # Older revisions may not expose reset_instance; the fresh
            # get_instance() below still exercises the lazy path.
            pass
        coords = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float32)
        try:
            keys = opencl_morton_encode_3d(coords, depth=6)
        except RuntimeError as e:
            if "not available" in str(e):
                self.skipTest("OpenCL context unavailable at runtime.")
            raise
        # If we got here, no AttributeError was raised -- the lazy-program
        # fix works. Validate the output shape and dtype.
        self.assertEqual(keys.shape, (3,))
        self.assertTrue(np.issubdtype(keys.dtype, np.integer))
        # Morton keys for distinct coords at depth=6 must be distinct.
        self.assertEqual(len(np.unique(keys)), 3)


if __name__ == "__main__":
    unittest.main()
