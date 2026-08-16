"""
Theoretical & Algorithmic Foundations Suite (algorithm_theory).

Translates frontier algorithmic breakthroughs into tree-free numerical kernels:
1. Frontier-Clustered Shortest Paths (Duan et al. STOC 2025 - Breaking the Sorting Barrier for SSSP)
2. Asymmetric Low-Rank Tensor Multipole Contractions (Alman et al. 2024/2025 - Matrix Multiplication Laser Exponents)
3. Spectral Meshfree Multi-Scale Laplacian Solvers (Nearly-linear time Spielman-Teng / Cohen SDDM preconditioners)
4. Sublinear-Time Approximate Distance Oracles (Elastic-Hash multi-resolution metric embeddings)
5. Non-Uniform Fast Fourier Transform (Barnett / Greengard NUFFT Type 1, 2, 3 via Elastic Gridding)
6. Matrix-Free Talbot Contour Numerical Laplace Inversion (Talbot / Weideman complex contour resolvents)
7. Directional Butterfly Oscillatory Wavefield Factorizations (Engquist & Ying high-frequency Helmholtz)
8. Continuous Meshfree Wavelets & Multi-Resolution Filterbanks (Antoine et al. CWT on point manifolds)
9. Screened Yukawa / Debye-Hückel Electrolyte FMM (Greengard & Huang screened Poisson electrostatics)
10. Constant-Potential Metal Electrode Boundary Element Method (Matrix-free GMRES capacitance BEM)
11. Fast Entropic Optimal Transport (Sinkhorn-Knopp Gaussian kernel scaling in O(N + M))
12. Matrix-Free Effective Resistance & Network Power Centrality (Spielman-Srivastava randomized projections)
13. Neutral Spatial Graph Partitioning & Space Decomposition (ReCom MCMC Spanning Tree Partitions)
14. Continuous Fast Multipole Fock Exchange (CFMM 2-Electron Coulomb & Exchange Integrals in O(N))
15. Continuous Non-Local Opinion Dynamics (Hegselmann-Krause multi-agent bounded confidence & polarization)
16. Continuous Spatial Voting Equilibrium (Downsian continuous multi-party policy Nash gradient flows)
17. Co-Optimal Transport & Dual-Manifold Biclustering (COOT / Gromov-Wasserstein alternating Sinkhorn)
18. Nearly-Linear Spectral Bipartite Biclustering (Dhillon normalized bipartite spectral partitioning)
19. Sublinear Multi-Scale FastDTW Alignment (Salvador & Chan multi-resolution time series warping)
20. Continuous Koopman Spectral Operator (Williams & Mezic EDMD non-linear dynamical linearization)
21. Fast Multipole Fractional Volterra Memory (Schadle & Lubich non-local Caputo memory convolution in O(T))
22. Takens' Delay Embedding Phase Space Attractor & Anomaly Search (Chaotic attractor recurrence in O(T))
23. Localized Ensemble Kalman Filter (Gaspari-Cohn spatial covariance tapering EnKF in O(N*M^2))
24. Spielman-Srivastava Spectral Graph Sparsifier (Effective resistance leverage score edge sampling)
25. Matrix-Free Personalized PageRank (PPR SDDM normalized random walk with restart resolvent)
26. Fast Hilbert-Schmidt Independence & Causal Direction Discovery (RFF non-linear HSIC & ANM causal arrows)
27. Matrix-Free Gaussian Process Regression & Uncertainty (Exact GP in O(N) time and memory)
28. Functional ANOVA & Global Sobol Sensitivity Decomposition (Saltelli-Sobol non-linear variance attribution)
29. Flat Multipole Range Tree (Pointerless multidimensional orthogonal range sums & prefix moments)
30. Zero-Displacement Elastic Quotient Filter (Non-reordering lock-free approximate membership & frequency sketch)
31. Sublinear Approximate Edit Distance & Pattern Search (Metric q-gram embeddings & banded alignment)
32. Spatial Disjoint Set & Dynamic Connectivity (Linear O(N) neighborhood percolation & spanning forest)
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

from .non_uniform_fourier_hash import (
    NonUniformFourierHash,
    direct_nufft_type1_baseline
)

from .fractional_laplace_contour import (
    MatrixFreeTalbotLaplaceInverter
)

from .oscillatory_butterfly_kernel import (
    OscillatoryButterflyKernel
)

from .continuous_meshfree_wavelet import (
    ContinuousMeshfreeWavelet
)

from .screened_yukawa_fmm import (
    ScreenedYukawaFMM
)

from .capacitance_boundary_bem import (
    CapacitanceBoundaryBEM
)

from .optimal_transport_fmm import (
    FastEntropicOptimalTransport,
    direct_sinkhorn_baseline
)

from .network_power_centrality import (
    NetworkPowerCentrality,
    dense_exact_effective_resistance
)

from .spatial_graph_partitioning import (
    SpatialGraphPartitioning
)

from .quantum_fock_exchange_fmm import (
    ContinuousFockExchangeFMM
)

from .opinion_dynamics_fmm import (
    ContinuousOpinionDynamicsFMM,
    direct_opinion_drift_reference
)

from .spatial_voting_equilibrium import (
    SpatialVotingEquilibriumEngine
)

from .co_optimal_transport import (
    CoOptimalTransport,
    FastCoOptimalTransport,
    FastGromovWasserstein,
    compute_pairwise_distances
)

from .spectral_biclustering_fmm import (
    SpectralBiclusteringFMM
)

from .sublinear_fast_dtw import (
    SublinearFastDTW
)

from .koopman_spectral_operator import (
    ContinuousKoopmanOperator
)

from .fractional_volterra_memory import (
    FractionalVolterraMemoryFMM
)

from .phase_space_attractor_fmm import (
    PhaseSpaceAttractorFMM
)

from .localized_ensemble_kalman_fmm import (
    LocalizedEnsembleKalmanFilter
)

from .spectral_graph_sparsifier import (
    SpectralGraphSparsifier
)

from .personalized_pagerank_fmm import (
    PersonalizedPageRankFMM
)

from .kernel_causal_discovery import (
    FastKernelCausalDiscovery,
    direct_dense_hsic
)

from .matrix_free_gaussian_process import (
    MatrixFreeGaussianProcess,
    dense_cholesky_gp_baseline
)

from .functional_sobol_anova import (
    FunctionalSobolANOVA,
    ishigami_benchmark_function
)

from .multipole_range_tree import (
    FlatMultipoleRangeTree,
    morton_encode_nd,
    direct_range_query_baseline
)

from .elastic_quotient_filter import (
    ElasticQuotientFilter,
    ClassicBloomFilterBaseline
)

from .sublinear_edit_distance import (
    SublinearEditDistance,
    exact_wagner_fischer_edit_distance,
    BKTree,
    BKTreeNode,
    ElasticFuzzyDictionary
)

from .spatial_disjoint_set_fmm import (
    SpatialDisjointSetFMM
)

from .spatial_point_cloud_compression import (
    SpatialPointCloudCompressor,
    morton_encode_3d_uint64,
    morton_decode_3d_uint64,
    compute_point_cloud_psnr
)

__all__ = [
    # 1. Pure TCS Foundations
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
    # 2. Continuous Transforms & Wavefields
    "NonUniformFourierHash",
    "direct_nufft_type1_baseline",
    "MatrixFreeTalbotLaplaceInverter",
    "OscillatoryButterflyKernel",
    "ContinuousMeshfreeWavelet",
    # 3. Energy, Batteries & Electrochemistry
    "ScreenedYukawaFMM",
    "CapacitanceBoundaryBEM",
    # 4. Structural Networks & Spatial Combinatorics
    "FastEntropicOptimalTransport",
    "direct_sinkhorn_baseline",
    "NetworkPowerCentrality",
    "dense_exact_effective_resistance",
    "SpatialGraphPartitioning",
    # 5. Quantum Chemistry & Electronic Structure
    "ContinuousFockExchangeFMM",
    # 6. Continuous Opinion Dynamics & Spatial Voting
    "ContinuousOpinionDynamicsFMM",
    "direct_opinion_drift_reference",
    "SpatialVotingEquilibriumEngine",
    # 7. Co-Clustering & Dual-Manifold Geometry
    "CoOptimalTransport",
    "FastCoOptimalTransport",
    "FastGromovWasserstein",
    "compute_pairwise_distances",
    "SpectralBiclusteringFMM",
    # 8. Modern Time-Series & Continuous Dynamics
    "SublinearFastDTW",
    "ContinuousKoopmanOperator",
    "FractionalVolterraMemoryFMM",
    "PhaseSpaceAttractorFMM",
    # 9. Advanced State Estimation & Non-Linear Filtering
    "LocalizedEnsembleKalmanFilter",
    # 10. Graph Search, Random Walks & Sparsification
    "SpectralGraphSparsifier",
    "PersonalizedPageRankFMM",
    # 11. Causal Mathematics & Inference
    "FastKernelCausalDiscovery",
    "direct_dense_hsic",
    # 12. Modern Bayesian & Classical Statistics
    "MatrixFreeGaussianProcess",
    "dense_cholesky_gp_baseline",
    "FunctionalSobolANOVA",
    "ishigami_benchmark_function",
    # 13. Basic Datatypes & Fundamental Data Structures
    "FlatMultipoleRangeTree",
    "morton_encode_nd",
    "direct_range_query_baseline",
    "ElasticQuotientFilter",
    "ClassicBloomFilterBaseline",
    "SublinearEditDistance",
    "exact_wagner_fischer_edit_distance",
    "BKTree",
    "BKTreeNode",
    "ElasticFuzzyDictionary",
    "SpatialDisjointSetFMM",
    # 14. Spatial Coordinate & Point Cloud Compression
    "SpatialPointCloudCompressor",
    "morton_encode_3d_uint64",
    "morton_decode_3d_uint64",
    "compute_point_cloud_psnr",
]
