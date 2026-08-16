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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        """Test WebGPU WGSL compute shader source structure."""
        self.assertTrue(os.path.exists(WGSL_SOURCE_PATH), "tree_free_fmm.wgsl must exist.")
        wgsl = get_wgsl_source()
        self.assertIn("@compute", wgsl)
        self.assertIn("@workgroup_size", wgsl)
        self.assertIn("tile_particles", wgsl)
        self.assertIn("workgroupBarrier()", wgsl)
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
        """Test AMD AMF hardware encoder registration and optimal CLI argument generation."""
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


if __name__ == "__main__":
    unittest.main()
