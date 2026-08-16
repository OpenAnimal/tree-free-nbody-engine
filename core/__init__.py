"""
Core Engine for Tree-Free N-Body & Spatial Field Computing
==========================================================
Non-Reordering Open Addressing Hash (Farach-Colton et al. 2025)
+ Fast Multipole Method (Greengard & Rokhlin, 1987)
"""

from .elastic_hash import (
    ElasticHashTable,
    jax_hash_probe,
)
from .tree_free_fmm import (
    morton_encode_2d,
    decode_morton_2d,
    get_box_center_2d,
    TreeFreeFMM,
    p2m,
    m2l,
    eval_local,
    exact_direct_nbody,
)
from .fast_vectorized_fmm import (
    FastVectorizedFMM,
)
from .bitboard_morton_avx import (
    BitboardMorton3D,
    morton_encode_3d_64bit,
)
from .device_runtime import (
    DeviceRuntime,
    AcceleratorVendor,
    ComputeBackend,
    DeviceDescriptor,
)
from .opencl_kernels import (
    is_opencl_available,
    opencl_tree_free_nbody,
    opencl_morton_encode_3d,
)
from .hip_kernels import (
    get_hip_kernel_source,
)
from .webgpu_kernels import (
    get_wgsl_source,
    is_webgpu_available,
)
from .zig_backend import (
    is_zig_available,
    get_zig_version,
    zig_simd_p2p_potentials,
    zig_simd_p2p_forces,
    zig_encode_morton3d,
    zig_build_bitboard64,
    zig_contact_forces,
    zig_2d_p2m,
    zig_2d_m2l,
    zig_2d_l2p,
    zig_3d_p2m,
    zig_3d_m2p,
    zig_3d_m2l,
    zig_3d_l2p,
)
from .jax_tree_free_fmm import (
    HAS_JAX,
)

if HAS_JAX:
    from .jax_tree_free_fmm import (
        jax_morton_encode_2d,
        jax_multi_level_probe_lookup,
        jax_elastic_probe_lookup,
        jax_p2m_expansion,
        jax_m2l_translation,
        jax_l2p_evaluation,
        jax_p2p_near_field,
        jax_direct_nbody_reference,
        jax_direct_nbody,
        compute_nbody_forces_jax,
    )
else:
    jax_morton_encode_2d = None
    jax_multi_level_probe_lookup = None
    jax_elastic_probe_lookup = None
    jax_p2m_expansion = None
    jax_m2l_translation = None
    jax_l2p_evaluation = None
    jax_p2p_near_field = None
    jax_direct_nbody_reference = None
    jax_direct_nbody = None
    compute_nbody_forces_jax = None

__all__ = [
    "ElasticHashTable",
    "jax_hash_probe",
    "morton_encode_2d",
    "decode_morton_2d",
    "get_box_center_2d",
    "TreeFreeFMM",
    "FastVectorizedFMM",
    "BitboardMorton3D",
    "morton_encode_3d_64bit",
    "is_zig_available",
    "get_zig_version",
    "zig_simd_p2p_potentials",
    "zig_simd_p2p_forces",
    "zig_encode_morton3d",
    "zig_build_bitboard64",
    "zig_contact_forces",
    "zig_2d_p2m",
    "zig_2d_m2l",
    "zig_2d_l2p",
    "zig_3d_p2m",
    "zig_3d_m2p",
    "zig_3d_m2l",
    "zig_3d_l2p",
    "DeviceRuntime",
    "AcceleratorVendor",
    "ComputeBackend",
    "DeviceDescriptor",
    "is_opencl_available",
    "opencl_tree_free_nbody",
    "opencl_morton_encode_3d",
    "get_hip_kernel_source",
    "get_wgsl_source",
    "is_webgpu_available",
    "p2m",
    "m2l",
    "eval_local",
    "exact_direct_nbody",
    "HAS_JAX",
    "jax_morton_encode_2d",
    "jax_multi_level_probe_lookup",
    "jax_elastic_probe_lookup",
    "jax_p2m_expansion",
    "jax_m2l_translation",
    "jax_l2p_evaluation",
    "jax_p2p_near_field",
    "jax_direct_nbody_reference",
    "jax_direct_nbody",
    "compute_nbody_forces_jax",
]
