"""
OpenCL Acceleration Module for ATI / AMD Radeon, Intel, and NVIDIA Hardware
===========================================================================
"""

from .opencl_fmm_backend import (
    OPENCL_CL_SOURCE_PATH,
    OpenCLFMMContext,
    is_opencl_available,
    opencl_tree_free_nbody,
    opencl_morton_encode_3d,
)

__all__ = [
    "OPENCL_CL_SOURCE_PATH",
    "OpenCLFMMContext",
    "is_opencl_available",
    "opencl_tree_free_nbody",
    "opencl_morton_encode_3d",
]
