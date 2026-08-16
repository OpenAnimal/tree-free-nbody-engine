"""
Quantized & Bitpacked Tree-Free Fast Multipole Method (FMM) Module
Inspired by Vercidium (2024) and Farach-Colton, Krapivin, Kuszmaul (2025).

Provides optional high-throughput fixed-point quantization, contiguous bit-packing,
run-length greedy multipole aggregation, 64-bit Morton bitboards, and zero-probe
neighbor striding.
"""

from .packed_particle_types import (
    pack_particles_64bit_3d,
    unpack_particles_64bit_3d,
    pack_particles_32bit_2d,
    unpack_particles_32bit_2d,
)
from .bitboard_occupancy import (
    MortonBitboard2D,
    MortonBitboard3D,
)
from .greedy_multipole_mesh import (
    GreedyMultipoleAggregator2D,
)
from .direct_morton_stride import (
    FastMortonNeighborTable2D,
    morton_inc_x_2d,
    morton_dec_x_2d,
    morton_inc_y_2d,
    morton_dec_y_2d,
)
from .packed_vectorized_fmm import (
    VoxelPackedTreeFreeFMM,
)

__all__ = [
    "VoxelPackedTreeFreeFMM",
    "pack_particles_64bit_3d",
    "unpack_particles_64bit_3d",
    "pack_particles_32bit_2d",
    "unpack_particles_32bit_2d",
    "MortonBitboard2D",
    "MortonBitboard3D",
    "GreedyMultipoleAggregator2D",
    "FastMortonNeighborTable2D",
    "morton_inc_x_2d",
    "morton_dec_x_2d",
    "morton_inc_y_2d",
    "morton_dec_y_2d",
]
