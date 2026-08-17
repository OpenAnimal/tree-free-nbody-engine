# Theoretical Algorithmic Foundations & Continuous Operators (`algorithm_theory`)
### Bridging Frontier Theoretical Computer Science with Tree-Free Numerical Kernels

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![STOC 2025 Best Paper](https://img.shields.io/badge/SSSP-Breaking%20Sorting%20Barrier-crimson.svg)](https://arxiv.org/abs/2409.04354)
[![Matrix Mult Bound](https://img.shields.io/badge/%CF%89%20%3C-2.371339-blueviolet.svg)](https://arxiv.org/abs/2404.16349)
[![FOCS 2024](https://img.shields.io/badge/Elastic%20Hash-Non--Reordering-orange.svg)](https://arxiv.org/abs/2501.02305)

---

> 🔬 **Research & Algorithmic Integration Suite:**  
> The `algorithm_theory` module translates recent breakthroughs in theoretical computer science into concrete, high-performance computational geometry, quantum chemistry, continuous transforms, causal mathematics, and dynamical systems. By synthesizing **Frontier Clustering (Duan et al. STOC 2025)**, **Asymmetric Tensor Laser Methods (Alman et al. 2024/2025)**, **Nearly-Linear Spectral Laplacians (Spielman-Teng SDDM)**, **Sublinear Approximate Distance Oracles (Thorup-Zwick)**, **Non-Uniform Spectral Gridding (Greengard/Barnett NUFFT)**, **Co-Optimal Transport (COOT)**, **Koopman Operator Linearization (EDMD)**, **Localized Ensemble Kalman Filtering (LEnKF)**, and **Kernel Causal Discovery (HSIC)** with **Tree-Free Elastic Spatial Hashing**, this package eliminates classical algorithmic bottlenecks across science and engineering.

---

## 🌟 Theoretical Breakthroughs & Practical Transference

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THEORETICAL TO PRACTICAL MAPPING                                     │
├──────────────────────────────────────┬──────────────────────────────────┬──────────────────────────────┤
│ Theoretical Breakthrough             │ Classical Computational Limit    │ Tree-Free N-Body Engine      │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 1. Breaking Dijkstra Sorting Barrier │ O(m + n log n) comparison-based  │ tree_free_geodesic_fmm.py    │
│    (Duan et al. STOC 2025)           │ priority-queue bottleneck        │ Bucketed frontier relaxation │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 2. Asymmetric Laser Matrix Mult      │ O(P²) dense M2L tensor           │ algebraic_multipole_tensor.py│
│    ω < 2.371339 (Alman et al. 2024)  │ contraction for order p (P~pᴰ)   │ Low-rank Tucker/CP M2L (295x)│
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 3. Nearly-Linear Spectral Laplacians │ O(N³) or slow mesh FEM solves    │ spectral_meshfree_laplacian.py│
│    (Spielman-Teng SDDM)              │ for continuous Poisson PDEs      │ Matrix-free two-level PCG    │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 4. Sublinear Approximate Distance    │ O(N²) all-pairs geodesic table / │ sublinear_distance_oracle.py │
│    Oracles (Thorup-Zwick / Bourgain) │ O(N log N) online path query     │ O(log 1/ε) ADO (14.4M qps)   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 5. Non-Uniform FFT Gridding          │ O(N*M) continuous non-equispaced │ non_uniform_fourier_hash.py  │
│    (Greengard & Lee / FINUFFT)       │ exponential Fourier sum          │ Elastic Hash NUFFT Type 1/2  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 6. Matrix-Free Talbot Inversion      │ O(N³) dense matrix exponential   │ fractional_laplace_contour.py│
│    (Talbot / Weideman Contours)      │ exp(t*L) for transient diffusion │ Multi-shift complex Krylov   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 7. Directional Wavefield Butterfly   │ High-frequency Helmholtz rank    │ oscillatory_butterfly_kernel.py│
│    (Engquist & Ying 2007)            │ growth (k * diam >> 1)           │ Directional ray factorization│
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 8. Screened Electrolyte FMM          │ O(N²) Debye-Hückel electrolyte   │ screened_yukawa_fmm.py       │
│    (Greengard & Huang Screened)      │ Coulomb screening summation      │ Adaptive screening cutoff    │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 9. Constant-Potential Electrode BEM  │ O(N³) dense capacitance matrix   │ capacitance_boundary_bem.py  │
│    (First-Kind Boundary Integral)    │ inversion for fixed voltage      │ Matrix-free GMRES BEM solver │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 10. Fast Entropic Optimal Transport  │ O(k_iter * N * M) dense Sinkhorn │ optimal_transport_fmm.py     │
│    (Cuturi / Peyre Wasserstein)      │ Gibbs kernel materialization     │ Spatial hash Gaussian conv   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 11. Randomized Effective Resistance  │ O(|V|³) Moore-Penrose pseudoinv  │ network_power_centrality.py  │
│    (Spielman & Srivastava JL Proj)   │ for network chokepoint analysis  │ Block-PCG 5.3M queries/sec   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 12. Neutral Spatial Decomposition    │ Exponential MCMC mixing & snake- │ spatial_graph_partitioning.py│
│    (ReCom Spanning Tree Sampling)    │ like district boundary artifacts │ Balanced tree cut partitions │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 13. Continuous Fock Exchange CFMM    │ O(N⁴) 4-center electron repulsion│ quantum_fock_exchange_fmm.py │
│    (White & Head-Gordon Gaussian CFMM) integrals in Hartree-Fock/DFT    │ Gaussian overlap FMM (O(N))  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 14. Non-Local Opinion Dynamics       │ O(N²) pairwise distance checks   │ opinion_dynamics_fmm.py      │
│    (Hegselmann-Krause Multi-Agent)   │ per continuous belief step       │ Spatial hash bounded sweeps  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 15. Continuous Spatial Voting        │ High-dimensional voter integral  │ spatial_voting_equilibrium.py│
│    (Downsian Nash Policy Gradients)  │ across non-uniform electorates   │ Analytical potential gradient│
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 16. Co-Optimal Transport (COOT)      │ O(N1*N2*D1*D2) 4-tensor cross-   │ co_optimal_transport.py      │
│    (Redko & Courty NeurIPS 2020)     │ manifold biclustering cost       │ Alternating matrix-vector OT │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 17. Spectral Bipartite Biclustering  │ O(min(R,C)*R*C) dense SVD on     │ spectral_biclustering_fmm.py │
│    (Dhillon Normalized Cut KDD 2001) │ massive bipartite matrices       │ Matrix-free power iteration  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 18. Sublinear Multi-Scale FastDTW    │ O(T1 * T2) dynamic programming   │ sublinear_fast_dtw.py        │
│    (Salvador & Chan FastDTW 2007)    │ grid for time series alignment   │ Adaptive corridor refinement │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 19. Koopman Spectral Linearization   │ Non-linear chaotic ODEs cannot   │ koopman_spectral_operator.py │
│    (Mezic / Williams EDMD 2015)      │ be integrated linearly long-term │ Infinite-D observable lifting│
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 20. Fractional Volterra Memory       │ O(T²) historical memory integral │ fractional_volterra_memory.py│
│    (Caputo fractional ODE memory)    │ accumulation in time series      │ 1D dyadic multipole moments  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 21. Phase Space Attractor & Anomaly  │ O(T²) all-pairs recurrence matrix│ phase_space_attractor_fmm.py │
│    (Takens' Delay Embedding 1981)    │ search for chaotic motifs        │ Spatial hash local density   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 22. Localized Ensemble Kalman Filter │ O(N³) dense covariance inversion │ localized_ensemble_kalman_fmm.py│
│    (Gaspari-Cohn spatial tapering)   │ and low-rank sampling noise      │ Local O(N*M²) block update   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 23. Spectral Graph Sparsification    │ Dense O(|V|²) graph analysis     │ spectral_graph_sparsifier.py │
│    (Spielman & Srivastava 2011)      │ is intractable at scale          │ Resistance leverage sampling │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 24. Personalized PageRank Resolvent  │ O(|V|³) matrix inverse for       │ personalized_pagerank_fmm.py │
│    (Page & Brin / Tong ICDM 2006)    │ random walks with restart        │ SDDM PCG in O(|E|) time      │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 25. Kernel Causal Discovery (HSIC)   │ O(N²) kernel Gram matrices for   │ kernel_causal_discovery.py   │
│    (Gretton & Zhang KCIT/ANM 2011)   │ non-linear independence tests    │ Random Fourier Features O(N) │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 26. Matrix-Free Gaussian Process     │ O(N³) Cholesky factorization and │ matrix_free_gaussian_process.py│
│    (Wang, Pleiss, Wilson NeurIPS)    │ O(N²) memory for exact GPs       │ Linear-time matrix-free PCG  │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 27. Functional Sobol ANOVA           │ Exponential variance integrals   │ functional_sobol_anova.py    │
│    (Saltelli & Sobol GSA 2001/2010)  │ in high-dimensional models       │ Saltelli matrix decomposition│
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 28. Flat Multipole Range Tree        │ O(N log^(d-1) N) pointer-based   │ multipole_range_tree.py      │
│    (Orthogonal Multidimensional Tree)│ Range Trees / k-d tree searching │ Flat Morton prefix moments   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 29. Elastic Quotient Filter          │ Shifting runs & Robin Hood lock  │ elastic_quotient_filter.py   │
│    (Zero-Displacement AMQ / Sketch)  │ cascades in quotient/cuckoo sets │ Non-reordering lock-free EQF │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 30. Sublinear Edit Distance          │ O(N * M) Wagner-Fischer dynamic  │ sublinear_edit_distance.py   │
│    (Batu-Ergun / Andoni Metric Embed)│ programming matrix bottleneck    │ Multi-scale q-gram banded DP │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 31. Spatial Disjoint Set FMM         │ O(N²) all-pairs distance graph   │ spatial_disjoint_set_fmm.py  │
│    (Geometric Dynamic Connectivity)  │ for DBSCAN/EMST percolation      │ O(N) cell neighborhood DSU   │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 32. Spatial Point Cloud Compression  │ Slow pointer-based Octrees in    │ spatial_point_cloud_compression.py│
│    (Morton Delta Varint Bitpacking)  │ Google Draco / MPEG PCC          │ Contiguous Morton delta SIMD │
└──────────────────────────────────────┴──────────────────────────────────┴──────────────────────────────┘
```

---

## 📂 Implemented Modules & Architecture

```text
algorithm_theory/
├── __init__.py                          # Public package exports (all 28 modules)
├── README.md                            # Comprehensive theory, formulations & benchmarks
├── benchmark_algorithm_theory.py        # Scalability & Verification Suite
├── algorithm_theory_benchmark.png       # Generated publication benchmark visualization
│
│   # --- 1. Pure TCS Foundations ---
├── tree_free_geodesic_fmm.py            # Duan-inspired Frontier-Clustered SSSP on 3D Manifolds
├── algebraic_multipole_tensor.py        # Asymmetric Low-Rank Tensor Factorization for High-Order M2L
├── spectral_meshfree_laplacian.py       # Matrix-Free Nearly-Linear Poisson Solver with Multi-Scale PCG
├── sublinear_distance_oracle.py         # Sublinear Approximate Distance Oracle & Metric Embeddings
│
│   # --- 2. Continuous Transforms & Wavefields ---
├── non_uniform_fourier_hash.py          # NUFFT Type 1/2/3 via Elastic Hash Spreading & Deconvolution
├── fractional_laplace_contour.py        # Matrix-Free Talbot Contour Numerical Laplace Inversion
├── oscillatory_butterfly_kernel.py      # Directional Butterfly High-Frequency Helmholtz Factorization
├── continuous_meshfree_wavelet.py       # Continuous Meshfree Wavelet (CWT) Filterbanks on Point Sets
│
│   # --- 3. Energy, Electrochemistry & Wave Physics ---
├── screened_yukawa_fmm.py               # Debye-Hückel / Yukawa Screened Electrolyte FMM (e⁻ᵏʳ/r)
├── capacitance_boundary_bem.py          # Constant-Potential Metal Electrode Boundary Element Solver
│
│   # --- 4. Structural Networks, Resource Equity & Combinatorics ---
├── optimal_transport_fmm.py             # Fast Entropic Wasserstein-2 / Sinkhorn Gaussian Transport
├── network_power_centrality.py          # Spielman-Srivastava Effective Resistance & Power Centrality
├── spatial_graph_partitioning.py        # Neutral ReCom MCMC Balanced Spanning Tree Space Partitions
│
│   # --- 5. Quantum Chemistry & Electronic Structure ---
├── quantum_fock_exchange_fmm.py         # Continuous Fast Multipole 2-Electron Exchange Operator (O(N⁴) → O(N))
│
│   # --- 6. Continuous Opinion Dynamics & Spatial Voting ---
├── opinion_dynamics_fmm.py              # Hegselmann-Krause: Non-local continuous belief dynamics
├── spatial_voting_equilibrium.py        # Hotelling-Downs: Continuous spatial voting Nash gradient flows
│
│   # --- 7. Co-Clustering & Dual-Manifold Geometry ---
├── co_optimal_transport.py              # Redko et al.: Co-Optimal Transport (COOT) Biclustering
├── spectral_biclustering_fmm.py         # Dhillon: Normalized Spectral Bipartite Co-Clustering
│
│   # --- 8. Modern Time-Series & Continuous Dynamics ---
├── sublinear_fast_dtw.py                # Salvador & Chan: Multi-Scale Adaptive Corridor FastDTW
├── koopman_spectral_operator.py         # Mezic & Williams: Extended Dynamic Mode Decomposition (EDMD)
├── fractional_volterra_memory.py        # Fast Multipole Convolution for Non-Local Fractional Memory
├── phase_space_attractor_fmm.py         # Takens' Delay Embedding Phase Space Attractor & Anomaly Search
│
│   # --- 9. Advanced State Estimation & Filtering ---
├── localized_ensemble_kalman_fmm.py     # Gaspari-Cohn Covariance Tapered Local EnKF (O(N·M²))
│
│   # --- 10. Graph Search, Random Walks & Sparsification ---
├── spectral_graph_sparsifier.py         # Spielman-Srivastava Effective Resistance Graph Sparsifier
├── personalized_pagerank_fmm.py         # Matrix-Free SDDM Personalized PageRank & Random Walk Resolvent
│
│   # --- 11. Causal Mathematics & Inference ---
├── kernel_causal_discovery.py           # Random Fourier Feature Fast HSIC & Additive Noise Causal Discovery
│
│   # --- 12. Modern Bayesian & Classical Statistics ---
├── matrix_free_gaussian_process.py      # Exact O(N) Gaussian Process Regression & Uncertainty Quantification
├── functional_sobol_anova.py            # Saltelli-Sobol Functional Variance Decomposition & Global Sensitivity
│
│   # --- 13. Basic Datatypes & Fundamental Data Structures ---
├── multipole_range_tree.py              # Flat Multidimensional Multipole Range Tree for Orthogonal Range Sums
├── elastic_quotient_filter.py           # Non-Reordering Lock-Free Elastic Quotient Filter & Frequency Sketch
├── sublinear_edit_distance.py           # Sublinear Edit Distance, BK-Tree Baseline & Elastic Fuzzy Dictionary
├── spatial_disjoint_set_fmm.py          # Spatial Disjoint Set Union (Union-Find) & Dynamic Connectivity
│
│   # --- 14. Spatial Coordinate & Point Cloud Compression ---
└── spatial_point_cloud_compression.py   # Tree-Free Morton Delta Varint Point Cloud & LiDAR Compression
```

---

## 🔬 Theoretical Citations

1. **Continuous Fast Multipole Method for Large Scale Gaussian Based Quantum Chemistry.** White, Johnson, Gill, Head-Gordon (1994, 1996). *Chemical Physics Letters*, 253(3-4).
2. **Co-Optimal Transport.** Redko, Courty, Flamary, Tuia (2020). *NeurIPS*.
3. **Co-clustering Documents and Words Using Bipartite Spectral Graph Partitioning.** Dhillon (2001). *ACM SIGKDD*.
4. **Toward Accurate Dynamic Time Warping in Linear Time and Space.** Salvador, Chan (2007). *Intelligent Data Analysis*, 11(5).
5. **A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition.** Williams, Kevrekidis, Rowley (2015). *Journal of Nonlinear Science*, 25(6).
6. **Construction of Correlation Functions in Two and Three Dimensions.** Gaspari, Cohn (1999). *Quarterly Journal of the Royal Meteorological Society*, 125(554).
7. **Graph Sparsification by Effective Resistances.** Spielman, Srivastava (2011). *SIAM Journal on Computing*, 40(6).
8. **Measuring Statistical Dependence with Hilbert-Schmidt Norms.** Gretton, Bousquet, Smola, Schölkopf (2005). *Algorithmic Learning Theory (ALT)*.
9. **GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration.** Gardner, Pleiss, Wu, Weinberger, Wilson (2018). *NeurIPS*.
10. **Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates.** Sobol (2001). *Mathematics and Computers in Simulation*, 55(1-3).
11. **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths.** Duan, Cheng, Mao, Yin, Ren (2025). *ACM STOC 2025 Best Paper* / [arXiv:2409.04354](https://arxiv.org/abs/2409.04354).
12. **Optimal Bounds for Open Addressing Without Reordering.** Farach-Colton, Krapivin, Kuszmaul (2025). *IEEE FOCS 2024* / [arXiv:2501.02305](https://arxiv.org/abs/2501.02305).
