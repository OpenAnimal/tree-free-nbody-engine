"""
Core Engine for Tree-Free N-Body & Spatial Field Computing
==========================================================
Non-Reordering Open Addressing Hash (Farach-Colton et al. 2025)
+ Carrier, Greengard, & Rokhlin (1988) 2D Adaptive Fast Multipole Method
+ Greengard & Rokhlin (1987) Regular Fast Multipole Method
"""

from .elastic_hash import (
    ElasticHashTable,
    ElasticBatchingHashTable,
    funnel_probe,
    jax_hash_probe,
)
from .spatial_index import CellIndex, morton_2d_key, morton_3d_key
from .validation import cross_validate, fmt_validation, assert_accuracy
from .benchmark_kit import VariantBenchmark
from .adaptive_gpu_metadata import (
    FlatAdaptiveMetadata,
    build_flat_adaptive_metadata,
    MAX_INTERACTIONS_PER_NODE,
)
from .cgr88_adaptive_fmm import (
    CGR88AdaptiveFMM,
    TreeFreeElasticAdaptiveFMM,
    GreengardRokhlin87RegularFMM,
    AdaptiveQuadTree,
    QuadBox,
    morton_encode_box,
    decode_morton_box,
    exact_direct_nbody_2d,
    exact_direct_nbody_forces_2d,
    p2m as cgr88_p2m,
    m2m as cgr88_m2m,
    m2l as cgr88_m2l,
    l2l as cgr88_l2l,
    l2p as cgr88_l2p,
    l2p_force as cgr88_l2p_force,
    p2l as cgr88_p2l,
    m2p as cgr88_m2p,
    p2p_potential_and_force,
)
from .tree_free_fmm import (
    morton_encode_2d,
    decode_morton_2d,
    get_box_center_2d,
    morton_key_from_indices,
    TreeFreeFMM,
    p2m,
    m2m,
    m2l,
    l2l,
    eval_local,
    eval_local_force,
    exact_direct_nbody,
    exact_direct_forces,
)
from .fast_vectorized_fmm import (
    FastVectorizedFMM,
)
from .yukawa3d_fmm import (
    Yukawa3DFMM,
    derivative_fd_guard as yukawa3d_derivative_fd_guard,
    toy_2cell_check as yukawa3d_toy_2cell_check,
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
    get_adaptive_cgr88_wgsl_source,
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
        jax_m2m_translation,
        jax_m2l_translation,
        jax_l2l_translation,
        jax_l2p_evaluation,
        jax_l2p_force_evaluation,
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
    jax_m2m_translation = None
    jax_m2l_translation = None
    jax_l2l_translation = None
    jax_l2p_evaluation = None
    jax_l2p_force_evaluation = None
    jax_p2p_near_field = None
    jax_direct_nbody_reference = None
    jax_direct_nbody = None
    compute_nbody_forces_jax = None

__all__ = [
    "CellIndex",
    "morton_2d_key",
    "morton_3d_key",
    "cross_validate",
    "fmt_validation",
    "assert_accuracy",
    "VariantBenchmark",
    "FlatAdaptiveMetadata",
    "build_flat_adaptive_metadata",
    "MAX_INTERACTIONS_PER_NODE",
    "CGR88AdaptiveFMM",
    "TreeFreeElasticAdaptiveFMM",
    "GreengardRokhlin87RegularFMM",
    "AdaptiveQuadTree",
    "QuadBox",
    "morton_encode_box",
    "decode_morton_box",
    "exact_direct_nbody_2d",
    "exact_direct_nbody_forces_2d",
    "cgr88_p2m",
    "cgr88_m2m",
    "cgr88_m2l",
    "cgr88_l2l",
    "cgr88_l2p",
    "cgr88_l2p_force",
    "cgr88_p2l",
    "cgr88_m2p",
    "p2p_potential_and_force",
    "ElasticHashTable",
    "ElasticBatchingHashTable",
    "funnel_probe",
    "jax_hash_probe",
    "morton_encode_2d",
    "decode_morton_2d",
    "get_box_center_2d",
    "morton_key_from_indices",
    "TreeFreeFMM",
    "FastVectorizedFMM",
    "Yukawa3DFMM",
    "yukawa3d_derivative_fd_guard",
    "yukawa3d_toy_2cell_check",
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
    "get_adaptive_cgr88_wgsl_source",
    "is_webgpu_available",
    "p2m",
    "m2m",
    "m2l",
    "l2l",
    "eval_local",
    "eval_local_force",
    "exact_direct_nbody",
    "exact_direct_forces",
    "HAS_JAX",
    "jax_morton_encode_2d",
    "jax_multi_level_probe_lookup",
    "jax_elastic_probe_lookup",
    "jax_p2m_expansion",
    "jax_m2m_translation",
    "jax_m2l_translation",
    "jax_l2l_translation",
    "jax_l2p_evaluation",
    "jax_l2p_force_evaluation",
    "jax_p2p_near_field",
    "jax_direct_nbody_reference",
    "jax_direct_nbody",
    "compute_nbody_forces_jax",
]
