"""
CUDA & Triton Hardware Acceleration Kernels for Tree-Free FMM
"""

from .triton_tree_free_fmm import (
    HAS_TRITON,
)

__all__ = [
    "HAS_TRITON",
]
