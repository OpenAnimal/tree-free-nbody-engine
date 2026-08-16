"""
AMD ROCm / HIP Hardware Acceleration Kernels for Tree-Free FMM
==============================================================
Provides native HIP C++ kernels for AMD Radeon RX 6000/7000/8000 (RDNA 2/3/4)
and AMD Instinct MI100/MI200/MI300 (CDNA 1/2/3) accelerators.
"""

import os

HIP_KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "tree_free_fmm_kernel.hip")

def get_hip_kernel_source() -> str:
    """Returns raw source code of the ROCm/HIP Tree-Free FMM kernel."""
    with open(HIP_KERNEL_SOURCE_PATH, "r", encoding="utf-8") as f:
        return f.read()

__all__ = ["HIP_KERNEL_SOURCE_PATH", "get_hip_kernel_source"]
