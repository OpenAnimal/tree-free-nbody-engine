"""
Graphics & Real-Time Rendering Suite (`graphics_rendering`)
Point-Based Global Illumination, Surfel Radiosity, Volumetric AO, Continuous Raymarching,
Gridless Irradiance Caching, and Asynchronous Zero-Copy Multi-GPU Streaming Pipelines.
Spatial hashing and clustering primitives; see each module's docstring for the
method it actually implements (most modules are Barnes-Hut-style tree codes or
voxel/SH interpolants, not translation-based FMMs).
"""

from .surfel_radiosity_gi import SurfelRadiosityGI, Surfel
from .volumetric_fmm_ao import (
    VolumetricFMMAmbientOcclusion,
    VolumetricSamplingMode,
    SparseVolumetricVoxelGrid,
    VolumetricMonopoleAO,
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
    "VolumetricMonopoleAO",
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
