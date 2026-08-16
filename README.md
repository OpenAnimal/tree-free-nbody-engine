<p align="center">
  <img src="assets/banner.png" alt="Tree-Free FMM Banner" width="100%">
</p>

# Tree-Free N-Body Engine (`tree-free-nbody-engine`)

### Octree-Free, Lock-Free $O(N)$ Spatial Computing & Fast Multipole Method (FMM)

**Pointerless Spatial Computing & Fast Multipole Method (FMM) via Optimal Non-Reordering Open Addressing**  
*(Farach-Colton, Krapivin, Kuszmaul 2025)*

<p align="left">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html"><img src="https://img.shields.io/badge/Live_Demo-Launch_Simulation-blueviolet?style=for-the-badge&logo=webgl" alt="Launch Live WebGL Demo"></a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Renderer-WebGL%202.0%20%2F%20WebGPU-orange.svg" alt="Renderer: WebGL 2.0 / WebGPU">
  <img src="https://img.shields.io/badge/Scale-250%2C000%2B%20Particles-purple.svg" alt="Scale: 250k+ Particles">
</p>

---

## What Is This? (Plain-English & Domain Concepts)

If you are coming from game development, robotics, artificial intelligence, biophysics, or graphics, **FMM (Fast Multipole Method)** combined with **Non-Reordering Elastic Hashing** provides a universal foundation for spatial computing without tree hierarchies:

* **Simulation & Physics:**
  * **Linear-Time $N$-Body Solver** ($O(N)$ gravity, electrostatics, vortex dynamics)
  * **Matrix-Free Potential Field & Collision Engine**
  * **Distant-Cluster Multipole Approximation** (P2M, M2M, M2L, L2L, L2P expansions)
* **AI, Transformers & Neural Operators:**
  * **Linear-Time $O(N)$ All-Pairs Spatial Attention Engine** (Flash Multipole Attention)
  * **Continuous Meshfree Graph Neural Networks** without explicit edge lists or adjacency matrices
  * **Matrix-Free Gaussian Process Regression & Continuous Action Diffusion Policies**
* **Biophysics, Genomics & Neurotechnology:**
  * **Linear-Time Macromolecular Electrostatics, Allosteric Pocket Discovery & Cryo-EM Fitting**
  * **Non-Periodic Continuum Molecular Dynamics & Condensate Phase Dynamics**
  * **Real-Time EEG/MEG Leadfield Solvers & sLORETA 3D Cortical Source Imaging**
* **Game Development & Robotics (Unreal Engine / MuJoCo):**
  * **Pointerless Spatial Physics & Proximity Solver** for millions of rigid/soft bodies
  * **100% Lock-Free Contiguous Spatial Hash Table** (CAS-insertable across all CPU/GPU threads)
  * **Real-Time Flocking, Continuum Swarm Pathfinding, Wave Function Collapse & Surface LOD Decimation**
* **Graphics, Rendering & Video Codecs:**
  * **Point-Based Global Illumination (PBGI), Surfel Radiosity & Hybrid 3D Voxel + FMM Volumetric Raymarching**
  * **Perceptual Rate Control (Delta-QP), Stochastic Noise Field Synthesis & Lock-Free Motion Estimation**

> **The Core Breakthrough:** Instead of computing interactions between all $N 	imes N$ pairs ($O(N^2)$, which crashes performance at scale), this engine computes localized near-field interactions directly ($O(1)$) and translates distant particles into hierarchical cluster multipoles ($O(N)$) &mdash; **strictly without using pointer-chasing Octrees, $k$-d trees, or BVHs, and without requiring multi-pass Radix Sorts**.

---

## Architectural Synthesis: Eliminating the Tree & Sorting Bottleneck

In classical GPU/CPU $N$-body algorithms (e.g., Barnes-Hut or tree-based FMM), spatial partitioning required:
1. Computing Morton/Z-order codes for all $N$ particles.
2. **Sorting all $N$ particles by their Morton code** using multi-pass GPU Radix Sort ($O(N \log N)$) to enforce spatial locality.
3. Building pointer-based Octrees/BVHs over the sorted keys with dynamic pointer allocations and warp divergence.

In early 2025, **Martín Farach-Colton, Andrew Krapivin, and William Kuszmaul** published *"Optimal Bounds for Open Addressing Without Reordering"* (arXiv:2501.02305 / IEEE FOCS 2024), breaking a **40-year-old theoretical barrier** (dating back to Andrew Yao's 1985 conjecture) by demonstrating that an open-addressed hash table achieves:
1. **$O(1)$ amortized probe complexity**
2. **$O(\log \delta^{-1})$ expected worst-case search complexity**
3. **Strictly zero reordering / element displacement**, even at high load factors ($\ge 95\%$).

By synthesizing **Optimal Non-Reordering Elastic Open Addressing** with the **Fast Multipole Method**, this engine **completely replaces sorting and tree construction**:
* **Zero Sorting Passes:** Particles are binned into multi-level geometric hash buckets in $O(1)$ time with no Radix Sort.
* **Pointerless Spatial Indexing:** Replaces pointer-based octrees with flat spatial Morton arrays indexed via non-reordering open addressing.
* **Lock-Free Concurrency:** Because insertions never displace existing keys, parallel threads insert via single atomic compare-and-swap (`atomicCAS`) primitives without cascading eviction locks.
* **Contiguous SIMD Streaming:** Near-field direct evaluations (P2P) and far-field multipole translations (M2L) stream from contiguous memory blocks.

```text
+-------------------------------------------------------------------------+
|                  Dynamic Particle / Coordinate Stream                   |
+-------------------------------------------------------------------------+
                                    |
                                    |
                                    v
+-------------------------------------------------------------------------+
|    TREE-FREE MULTIPOLE HASH TABLE (Farach-Colton / Kuszmaul Layout)     |
|   - Multi-Level Geometrically Sized Sub-Arrays                          |
|   - Strict Zero-Reordering (Lock-Free / CAS-Compatible)                 |
|   - Bounded O(log 1/delta) Search Latency at High Load Factors          |
+-------------------------------------------------------------------------+
            |                                               |
            v (Near-Field: P2P Direct)                      v (Far-Field: M2L Multipole)
  O(1) 3x3 Neighborhood Hash Probes               Vectorized Cluster Matrix Broadcast
  (SIMD Continuous Streaming)                     (Linear O(N) Far-Field Potential)
```

---

## Repository Architecture & Core Modules

The repository is organized into focused, modular packages covering fundamental mathematics, neural operators, biophysics, graphics, robotics, and theoretical computer science:

```text
tree-free-nbody-engine/
├── core/                                # Core Tree-Free FMM & Elastic Hash backends
│   ├── elastic_hash.py                  # Optimal non-reordering open addressing hash table
│   ├── fast_vectorized_fmm.py           # CPU SIMD vectorized FMM engine
│   ├── jax_tree_free_fmm.py             # JAX JIT-compiled GPU/CPU execution backend
│   ├── bitboard_morton_avx.py           # AVX bitboard Morton spatial encoding
│   ├── zig_backend.py                   # High-performance compiled Zig C-ABI bindings
│   └── cuda_kernels/                    # Native CUDA (.cu) & Triton JIT FMM kernels
│
├── neural_ops/                          # Linear O(N) Neural Network & Spatial AI Layers
│   ├── multipole_attention.py           # Linear-time Tree-Free Multipole Attention (TFMA)
│   ├── flash_multipole_kernel.py        # Fused memory-efficient Flash Multipole Attention
│   ├── visual_transformer_ops.py        # Multi-scale & cross-multipole attention for Vision
│   ├── diffusion_policy_fmm.py          # Continuous diffusion policy (DDPM/Flow Matching) for robotics
│   ├── multipole_gaussian_process.py    # Matrix-free GP regression & sparse variational GP (SVGP)
│   ├── continuous_meshfree_gnn.py       # Continuous message passing without adjacency graphs
│   ├── equivariant_field_layer.py       # E(3)/SO(3) equivariant physical neural fields
│   ├── multipole_mamba_ssm.py           # Multipole state-space long-range sequence modeling
│   └── hyperbolic_multipole_attention.py# Poincaré ball & Lorentz geometric manifold attention
│
├── bioinformatics/                      # Structural Biology, Pan-Genomics & Neurotechnology
│   ├── allosteric_druggability_engine.py# Mode-perturbation allosteric pocket discovery
│   ├── binding_pocket_detector.py       # O(N) surface probe binding pocket detector
│   ├── macromolecular_nma_engine.py     # Coarse-grained Normal Mode Analysis (ANM/GNM)
│   ├── non_periodic_md_engine.py        # Non-periodic continuum solvent molecular dynamics
│   ├── solvation_free_energy.py         # Screened Poisson-Boltzmann / Generalized Born solvation
│   ├── biomolecular_condensate_engine.py# Phase-separation & condensate spatial dynamics
│   ├── cryo_em_flexible_fitting.py      # Deformable cryo-EM density map fitting
│   ├── rna_tertiary_folding_engine.py   # RNA 3D backbone folding & riboswitch electrostatics
│   ├── smart_biologics_designer.py      # Antibody pH-switch recycling & developability profiler
│   ├── kmer_elastic_hash.py             # Lock-free k-mer & minimizer sequence search
│   ├── causal_perturb_seq_grn.py        # Single-cell CRISPR perturbation gene regulatory networks
│   ├── biosignal_lsl_stream_engine.py   # Real-time multi-channel (64-512ch) EEG/fMRI LSL stream pipeline
│   ├── eeg_source_localization_fmm.py   # 3-shell forward leadfield solver & sLORETA 3D cortical imaging
│   └── whole_cell_viral_simulation.py   # Multi-million atom whole-virion envelope simulation
│
├── algorithm_theory/                    # Frontier Theoretical Computer Science Modules
│   ├── tree_free_geodesic_fmm.py        # Frontier SSSP breaking the Dijkstra barrier (STOC 2025)
│   ├── algebraic_multipole_tensor.py    # Asymmetric laser tensor compression for high-order M2L
│   ├── spectral_meshfree_laplacian.py   # Nearly-linear SDD Galerkin Poisson solver (Spielman-Teng)
│   ├── sublinear_distance_oracle.py     # (1+eps)-approximate Thorup-Zwick distance oracle
│   ├── matrix_free_gaussian_process.py  # Matrix-free block PCG GP solver with predictive variance
│   ├── multipole_range_tree.py          # Flat multidimensional multipole orthogonal range query tree
│   ├── elastic_quotient_filter.py       # Zero-displacement lock-free elastic quotient filter sketch
│   ├── sublinear_edit_distance.py       # Sublinear banded q-gram edit distance & BK-tree dictionary
│   ├── spatial_disjoint_set_fmm.py      # Spatial Disjoint Set Union (Union-Find) dynamic connectivity
│   ├── spatial_point_cloud_compression.py# Tree-free Morton delta varint LiDAR/point cloud compression
│   ├── optimal_transport_fmm.py         # Multilevel Sinkhorn Wasserstein distance
│   ├── co_optimal_transport.py          # Dual feature-sample Co-Optimal Transport (COOT)
│   ├── koopman_spectral_operator.py     # Extended Dynamic Mode Decomposition (EDMD)
│   ├── localized_ensemble_kalman_fmm.py # Scalable spatial state estimation (LEnKF)
│   ├── quantum_fock_exchange_fmm.py     # Hartree-Fock exact exchange potential evaluation
│   └── sublinear_fast_dtw.py            # Sublinear dynamic time warping for massive trajectories
│
├── physics_simulation/                  # Matrix-Free Contact & Shell Mechanics
│   └── ppf_contact_solver_fmm/          # Incremental Potential Contact (IPC) & barrier solver
│       ├── matrix_free_ipc.py           # Matrix-free IPC without dynamic sparse matrices
│       ├── tetrahedral_surgical_soft_robotics.py # 3D hyperelastic surgical soft body simulation
│       └── cloth_shell_simulation.py    # Thin-shell & cloth large-deformation dynamics
│
├── graphics_rendering/                  # Real-Time Global Illumination & Radiance Suite
│   ├── dynamic_irradiance_cache.py      # Gridless spherical harmonic irradiance probe fields
│   ├── surfel_radiosity_gi.py           # Multi-bounce surfel radiosity global illumination
│   ├── volumetric_fmm_ao.py             # Hybrid 3D Voxel + FMM Volumetric Raymarching & Deep Shadowing
│   ├── async_zerocopy_streaming.py      # Non-blocking double-buffered GPU tile ring streaming
│   └── gpu_hardware_interop.py          # 16-byte float4 structured layouts & zero-copy compute buffers
│
├── game_mechanics_spatial/              # Spatial Computing & Interactive Mechanics
│   ├── massive_crowd_flocking.py        # O(N) boid steering with 1€ adaptive anti-jitter filter
│   ├── harmonic_flow_field_pathfinding.py # Continuous multipole & screened Yukawa continuum swarm navigation
│   ├── wave_function_collapse_pcg.py    # Bitset AC-4 constraint wave function collapse procedural engine
│   ├── procedural_dungeon_network.py    # Poisson-disc & MST procedural dungeon room graph synthesizer
│   ├── line_of_sight_fog_of_war.py      # Continuous line-of-sight & visibility occlusion queries
│   ├── fast_mesh_lod_decimator.py       # Quadric error metric mesh decimation via Morton hashing
│   └── smart_brush_lasso_selector.py    # Sub-millisecond polygon lasso spatial selection
│
├── quantized_bitpacked_optimization/    # Systems Optimizations & Cache-Line Saturation
│   ├── packed_vectorized_fmm.py         # Integer bitfield coordinate packing & SIMD streaming
│   ├── bitboard_occupancy.py            # Bitboard 64-bit spatial occupancy masks
│   └── greedy_multipole_mesh.py         # Run-length Morton cluster merging for M2L pruning
│
├── video_streaming_codecs/              # Video & Sensor Motion Compression Intelligence
│   ├── perceptual_rate_controller.py    # Spatial-temporal contrast sensitivity & Delta-QP optimizer
│   ├── scenecut_gop_analyzer.py         # Temporal discontinuity & adaptive GOP keyframe planner
│   ├── parametric_noise_field_codec.py  # Parametric stochastic noise field decomposition & reconstruction
│   ├── adaptive_hls_dash_segmenter.py   # Low-latency ABR slicer & HLS/DASH manifest generator
│   ├── spatial_dct_entropy_codec.py     # Reference 2D DCT & run-length entropy codec testbed
│   ├── ffmpeg_interop_bridge.py         # Non-blocking FFmpeg & GPU hardware encoder (NVENC/QSV) bridge
│   ├── lockfree_motion_estimation.py    # Lock-free spatial hash motion vector estimation
│   ├── greedy_macroblock_merger.py      # Greedy run-length DCT operation pruning
│   ├── one_euro_video_stabilizer.py     # Real-time adaptive camera gyro stabilization
│   └── neuromorphic_event_streaming.py  # Asynchronous spike event stream filtering
│
└── apps/                                # 10 Interactive Domain Case Studies & Benchmarks
```

---

## Performance & Scaling Benchmarks

Empirical scaling benchmarks comparing naive direct evaluation, dense vectorized NumPy matrix kernels, JAX JIT compilation, and the Tree-Free FMM pipeline from $N = 100$ to $N = 100,000$ particles:

<p align="center">
  <img src="assets/benchmark_scaling_analysis.png" alt="Scaling Benchmark Analysis" width="95%">
</p>

### Execution Latency Benchmark Table

| Particle Count ($N$) | Naive Python CPU $O(N^2)$ | Vectorized NumPy $O(N^2)$ | JAX JIT Compiled $O(N^2)$ | **Vectorized Tree-Free FMM $O(N)$** | Speedup vs. NumPy Direct |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 100$** | 0.91 ms | 0.35 ms | 46.8 ms *(dispatch)* | **3.17 ms** | $0.11\times$ |
| **$N = 500$** | 8.84 ms | 13.5 ms | 49.8 ms | **13.97 ms** | $0.97\times$ *(crossover)* |
| **$N = 2,000$** | 244.5 ms | 157.9 ms | 50.3 ms | **16.69 ms** | **$9.5\times$** |
| **$N = 4,000$** | 978.1 ms | 593.5 ms | 51.6 ms | **166.55 ms** | **$3.6\times$** |
| **$N = 8,000$** | 3,912.4 ms | 2,361.0 ms | 74.2 ms | **188.67 ms** | **$12.5\times$** |
| **$N = 16,000$** | 15,649.5 ms | 9,444.0 ms | 156.8 ms | **246.35 ms** | **$38.3\times$** |
| **$N = 64,000$** | 250,391.7 ms (250s) | 151,104.7 ms (151s) | 2,509.2 ms | **2,741.17 ms** | **$55.1\times$** |
| **$N = 100,000$** | 611,307.8 ms (611s) | 368,908.0 ms (369s) | 6,125.9 ms | **2,884.09 ms** | **$127.9\times$** |

### Hash Table Stress Test at 92% Load Factor

* **10,000 capacity:** 9,200 keys inserted in 34.3 ms ($268,128\text{ ops/s}$), avg **5.51 probes/key**.
* **50,000 capacity:** 46,000 keys inserted in 176.7 ms ($260,390\text{ ops/s}$), avg **5.48 probes/key**.
* **200,000 capacity:** 184,000 keys inserted in 786.9 ms ($233,843\text{ ops/s}$), avg **5.50 probes/key**.
* **Reordering count:** Strictly **0** (zero element relocation or eviction cascading).

---

## Interactive Real-Time Simulation (WebGL 2.0 / WebGPU)

<p align="center">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html">
    <img src="assets/simulation_demo.gif" alt="Tree-Free N-Body Real-Time GPU Animation" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html">
    <img src="https://img.shields.io/badge/Launch_Live_Interactive_Simulation-blueviolet?style=for-the-badge&logo=webgl" alt="Launch Live Simulation">
  </a>
</p>

To run the interactive visualization locally, open [`index.html`](index.html) in any modern web browser or start a local server:
```bash
python -m http.server 8090
# Open http://localhost:8090 in your browser
```

---

## Application Suite & Domain Case Studies

The `apps/` directory provides ten complete, runnable domain demonstrations:

| App | Domain | Script | Key Highlight |
| :--- | :--- | :--- | :--- |
| **1. Galaxy Collision** | Astrophysics | [`apps/app1_galaxy_collision.py`](apps/app1_galaxy_collision.py) | Dynamic rotating spiral galaxy merger with continuous spatial re-binning. |
| **2. Hydrodynamics** | Fluid Mechanics | [`apps/app2_hydrodynamics.py`](apps/app2_hydrodynamics.py) | Biot-Savart vortex streamfunction for Kelvin-Helmholtz instability. |
| **3. Spatial Attention** | Deep Learning | [`apps/app3_spatial_attention.py`](apps/app3_spatial_attention.py) | Linear $O(N)$ spatial multipole attention over non-uniform point clouds. |
| **4. 1€ Boid Flocking** | Swarm Robotics | [`apps/app4_fmm_boids_1euro.py`](apps/app4_fmm_boids_1euro.py) | Multilevel boid swarms with 1€ adaptive filtering to suppress jitter. |
| **5. Molecular Electrostatics** | Biophysics | [`apps/app5_bioinformatics.py`](apps/app5_bioinformatics.py) | Screened Debye-Hückel & Coulomb solvation over folded protein structures. |
| **6. MuJoCo Proximity** | Robotics | [`apps/app6_mujoco_proximity.py`](apps/app6_mujoco_proximity.py) | Soft-contact normal force fields & ground proximity for bipedal locomotion. |
| **7. High-Dim Memory** | Vector Search | [`apps/app7_highdim_memory.py`](apps/app7_highdim_memory.py) | Hyperplane LSH partitioning 64D embeddings at $3.8\times 10^6\text{ vectors/s}$. |
| **8. Manifold Unfolding** | Dimensionality | [`apps/app8_dimension_reduction_knn.py`](apps/app8_dimension_reduction_knn.py) | Non-linear 8D-to-2D Laplacian Eigenmaps using $O(N)$ hash $k$-NN graphs. |
| **9. Streaming Vector DB** | Information Retrieval | [`apps/app9_streaming_vector_db.py`](apps/app9_streaming_vector_db.py) | Lock-free ingestion of 128D embeddings at $\sim 170,000\text{ vectors/s}$. |
| **10. Continuous GNN** | Graph AI | [`apps/app10_continuous_gnn_fmm.py`](apps/app10_continuous_gnn_fmm.py) | Matrix-free spatial graph convolutions without allocating adjacency matrices. |

<p align="center">
  <img src="assets/app1_galaxy_collision.png" width="32%" alt="Galaxy Collision">
  <img src="assets/app2_hydrodynamic_vortex.png" width="32%" alt="Hydrodynamic Vortex">
  <img src="assets/app3_spatial_attention.png" width="32%" alt="Spatial Attention">
</p>
<p align="center">
  <img src="assets/app4_fmm_boids_1euro.png" width="32%" alt="1 Euro Boids">
  <img src="assets/app5_protein_electrostatics.png" width="32%" alt="Protein Electrostatics">
  <img src="assets/app6_mujoco_proximity.png" width="32%" alt="MuJoCo Proximity">
</p>
<p align="center">
  <img src="assets/app7_highdim_embeddings.png" width="32%" alt="High-Dim Embeddings">
  <img src="assets/app8_dimension_reduction_knn.png" width="32%" alt="Manifold Unfolding">
  <img src="assets/app9_streaming_vector_db.png" width="32%" alt="Streaming Vector DB">
</p>

---

## Quickstart & Installation

### Prerequisites
* Python 3.10+
* `numpy`, `matplotlib`
* *Optional:* `jax`, `torch`, `triton` (for GPU / JIT acceleration), `scipy`

### Installation

```bash
# Clone the repository
git clone https://github.com/OpenAnimal/tree-free-nbody-engine.git
cd tree-free-nbody-engine

# Install base dependencies
pip install -e .

# Or install with optional acceleration packages
pip install -e ".[all]"
```

### Running Benchmarks & Applications

```bash
# Run the core scaling benchmark suite
python apps/benchmark_suite.py

# Run individual domain demonstrations
python apps/app1_galaxy_collision.py
python apps/app4_fmm_boids_1euro.py
python apps/app5_bioinformatics.py
python apps/app9_streaming_vector_db.py
python apps/app10_continuous_gnn_fmm.py

# Run sub-package test suites & benchmarks
python neural_ops/test_fmm_neural_ops.py
python neural_ops/test_neural_ops_advanced.py
python bioinformatics/test_sota_modules.py
python algorithm_theory/benchmark_algorithm_theory.py
python graphics_rendering/test_graphics_rendering.py
python video_streaming_codecs/test_video_streaming.py
python physics_simulation/ppf_contact_solver_fmm/cloth_shell_simulation.py
```

---

## Citation

If you use `tree-free-nbody-engine` in your academic research, engineering systems, or software projects, please cite this repository:

```bibtex
@software{tree_free_nbody_engine2026,
  author       = {OpenAnimal},
  title        = {Tree-Free N-Body Engine: Octree-Free, Lock-Free O(N) Spatial Computing & Multipole Mechanics},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/OpenAnimal/tree-free-nbody-engine}},
  url          = {https://github.com/OpenAnimal/tree-free-nbody-engine}
}
```

---

## Theoretical References

1. **Optimal Bounds for Open Addressing Without Reordering**  
   Martín Farach-Colton, Andrew Krapivin, William Kuszmaul (2025).  
   *IEEE FOCS 2024* / [arXiv:2501.02305](https://arxiv.org/abs/2501.02305).
2. **A Fast Algorithm for Particle Simulations**  
   Leslie Greengard, Vladimir Rokhlin (1987).  
   *Journal of Computational Physics*, 73(2), 325–348.
3. **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths**  
   Ran Duan, Jiayan Cheng, Xiao Mao, Longhui Yin, Hanrui Ren (2024/2025).  
   *ACM STOC 2025 Best Paper* / [arXiv:2409.04354](https://arxiv.org/abs/2409.04354).
4. **More Asymmetry Yields Faster Matrix Multiplication**  
   Josh Alman, Ran Duan, Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, Renfei Zhou (2024/2025).  
   [arXiv:2404.16349](https://arxiv.org/abs/2404.16349).
5. **Nearly-Linear Time Algorithms for Graph Laplacians**  
   Daniel A. Spielman, Shang-Hua Teng (2004, 2011).  
   *SIAM J. Comput.* 40(4), 981–1025.
6. **Approximate Distance Oracles**  
   Mikkel Thorup, Uri Zwick (2005).  
   *Journal of the ACM*, 52(1), 1–24.
7. **1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in Interactive Systems**  
   Géry Casiez, Nicolas Roussel, Daniel Vogel (2012).  
   *ACM CHI Conference on Human Factors in Computing Systems*.
8. **Incremental Potential Contact: Intersection- and Inversion-Free, Large-Deformation Dynamics**  
   Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis Zorin, Daniele Panozzo (2020).  
   *ACM Transactions on Graphics (SIGGRAPH 2020)*.

---

> **Project Status & Support Notice:** Released under the MIT License. Take the code and do whatever you want with it — no roadmap, no active maintenance, and no unpaid support. Please do not open issues asking for personal assistance or feature requests.
