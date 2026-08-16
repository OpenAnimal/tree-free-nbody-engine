"""
Theoretical & Algorithmic Foundations Suite (algorithm_theory).

Explores and implements modern algorithmic breakthroughs integrated with the Tree-Free Fast Multipole Architecture:
1. Frontier-Clustered Shortest Paths (Duan et al. STOC 2025 - Breaking the Sorting Barrier for SSSP)
2. Asymmetric Low-Rank Tensor Multipole Contractions (Alman et al. 2024/2025 - Matrix Multiplication Laser Exponents)
3. Spectral Meshfree Multi-Scale Laplacian Solvers (Nearly-linear time Spielman-Teng / Cohen style preconditioners)
4. Sublinear-Time Approximate Distance Oracles (Elastic-Hash multi-resolution metric embeddings)
"""

from .tree_free_geodesic_fmm import (
    FrontierClusteredSSSP,
    MeshfreeGeodesicSolver,
    DijkstraBaselineSSSP,
    compute_meshfree_geodesic_field
)

from .algebraic_multipole_tensor import (
    AsymmetricMultipoleTensor,
    LowRankFarFieldContraction,
    FastTensorM2L,
    benchmark_tensor_vs_dense
)

from .spectral_meshfree_laplacian import (
    SpectralMeshfreeLaplacian,
    MultiScalePreconditionedSolver,
    solve_meshfree_poisson
)

from .sublinear_distance_oracle import (
    SublinearDistanceOracle,
    ElasticMetricEmbedding,
    MultiScaleLandmarkOracle
)

__all__ = [
    "FrontierClusteredSSSP",
    "MeshfreeGeodesicSolver",
    "DijkstraBaselineSSSP",
    "compute_meshfree_geodesic_field",
    "AsymmetricMultipoleTensor",
    "LowRankFarFieldContraction",
    "FastTensorM2L",
    "benchmark_tensor_vs_dense",
    "SpectralMeshfreeLaplacian",
    "MultiScalePreconditionedSolver",
    "solve_meshfree_poisson",
    "SublinearDistanceOracle",
    "ElasticMetricEmbedding",
    "MultiScaleLandmarkOracle",
]
