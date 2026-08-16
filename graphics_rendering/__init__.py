"""
Graphics & Real-Time Rendering Suite (`graphics_rendering`)
Point-Based Global Illumination, Surfel Radiosity, Volumetric AO, Continuous Raymarching,
Gridless Irradiance Caching, and Asynchronous Zero-Copy Multi-GPU Streaming Pipelines.
Powered by Tree-Free Fast Multipole Method (FMM) and Elastic Spatial Hashing.
"""

from .surfel_radiosity_gi import SurfelRadiosityGI, Surfel
from .volumetric_fmm_ao import (
    VolumetricFMMAmbientOcclusion,
    VolumetricSamplingMode,
    SparseVolumetricVoxelGrid,
)
from .dynamic_irradiance_cache import DynamicIrradianceCache
from .async_zerocopy_streaming import AsyncZeroCopyGraphicsPipeline, StreamingTile, FrameRenderStats
from .gpu_hardware_interop import (
    GPUBackendAPI,
    BufferAccessFlag,
    GPUBufferDescriptor,
    GPURenderSyncFence,
    GPURaymarchConfig,
    HardwareZeroCopyBuffer,
    HardwareVolumetricFieldBuffer,
    HardwareSHProbeBuffer,
    Hardware3DVoxelTextureBuffer,
    HardwareGraphicsBridge,
    pack_volumetric_clusters_gpu_layout,
    pack_sh_probes_gpu_layout,
    unpack_sh_probes_gpu_layout,
)

__all__ = [
    "SurfelRadiosityGI",
    "Surfel",
    "VolumetricFMMAmbientOcclusion",
    "VolumetricSamplingMode",
    "SparseVolumetricVoxelGrid",
    "DynamicIrradianceCache",
    "AsyncZeroCopyGraphicsPipeline",
    "StreamingTile",
    "FrameRenderStats",
    "GPUBackendAPI",
    "BufferAccessFlag",
    "GPUBufferDescriptor",
    "GPURenderSyncFence",
    "GPURaymarchConfig",
    "HardwareZeroCopyBuffer",
    "HardwareVolumetricFieldBuffer",
    "HardwareSHProbeBuffer",
    "Hardware3DVoxelTextureBuffer",
    "HardwareGraphicsBridge",
    "pack_volumetric_clusters_gpu_layout",
    "pack_sh_probes_gpu_layout",
    "unpack_sh_probes_gpu_layout",
]
