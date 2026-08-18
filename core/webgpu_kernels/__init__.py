"""
WebGPU WGSL Kernels and Cross-Platform Browser/Edge Acceleration
================================================================
"""

from .webgpu_fmm_runner import (
    WGSL_SOURCE_PATH,
    ADAPTIVE_CGR88_WGSL_SOURCE_PATH,
    get_wgsl_source,
    get_adaptive_cgr88_wgsl_source,
    is_webgpu_available,
    get_webgpu_adapter_info,
)

__all__ = [
    "WGSL_SOURCE_PATH",
    "ADAPTIVE_CGR88_WGSL_SOURCE_PATH",
    "get_wgsl_source",
    "get_adaptive_cgr88_wgsl_source",
    "is_webgpu_available",
    "get_webgpu_adapter_info",
]
