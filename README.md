<p align="center">
  <img src="assets/banner.png" alt="Tree-Free FMM Banner" width="100%">
</p>

# Tree-Free N-Body Engine (`tree-free-nbody-engine`)

### Octree-Free, Lock-Free $O(N)$ $N$-Body & Spatial Field Engine

**Pointerless Spatial Computing & Fast Multipole Method (FMM)** (Greengard & Rokhlin, 1987) via **Optimal Non-Reordering Open Addressing** (Farach-Colton et al., 2025)

<p align="left">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html"><img src="https://img.shields.io/badge/Live_Demo-Launch_Simulation-blueviolet?style=for-the-badge&logo=webgl" alt="Launch Live WebGL Demo"></a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Dependencies-Near_Zero-success.svg" alt="Near Dependency Free">
  <img src="https://img.shields.io/badge/Renderer-WebGL%202.0%20%2F%20WebGPU-orange.svg" alt="Renderer: WebGL 2.0 / WebGPU">
  <img src="https://img.shields.io/badge/Scale-250%2C000%2B%20Particles-purple.svg" alt="Scale: 250k+ Particles">
</p>

---

> 🔬 **Research Prototype & Exploratory Notice:**
> While core algorithms include automated verification tests and empirical scaling benchmarks, these implementations are exploratory research prototypes and WILL contain bugs that have to be fixed over time.

> ⚠️ **Project Status & Support Notice:** Released under MIT. Take the code and do whatever you want with it — no roadmap, no active maintenance, and no unpaid support. (I am poor as a church mouse and don't have time or resources for anything.)

---

## 💡 What Is This? (Synonyms & Plain-English Concepts)

**The Core Idea:** Instead of computing interactions between all $N \times N$ pairs ($O(N^2)$, this engine computes nearby interactions directly and groups distant particles into cluster multipoles ($O(N)$) — all without using slow pointer-chasing Octrees or BVHs.
Here is what this engine actually can do across some domains:

* **Simulation & Physics:**
  * **Linear-Time $N$-Body Solver** ($O(N)$ $N$-Body Engine)
  * **Fast Particle Interaction & Potential Field Engine**
  * **Hierarchical Cluster Physics Engine / Distant-Cluster Approximation**
* **Game Development & Robotics (Unreal / MuJoCo):**
  * **Pointerless Spatial Physics & Proximity Solver**
  * **Massive-Scale Particle / Collision Field Engine**
  * **Lock-Free Spatial Hash $N$-Body Simulator**
  * **Contiguous Memory / Flat-Grid Multipole Force Solver**
* **AI, Graphics & High-Performance Computing:**
  * **Linear-Time All-Pairs Spatial Attention Engine**
  * **Fast Distance / Potential Matrix Accelerator**
  * **Hierarchical Spatial Kernel Evaluator**
  * **Pointerless Octree-Free Spatial Indexer**

---

## Background

The **Fast Multipole Method (FMM)** (Greengard & Rokhlin, 1987) — widely recognized as one of the top ten algorithms of the century — reduced $N$-body and potential field evaluations from quadratic $O(N^2)$ to linear $O(N)$. However, practical FMM implementations historically relied on hierarchical **Octree / $k$-d tree data structures**. On SIMD, GPU, and distributed architectures, dynamic tree allocation every timestep, pointer chasing, and warp divergence remain primary bottlenecks.

In early 2025, **Martín Farach-Colton, Andrew Krapivin, and William Kuszmaul** published *"Optimal Bounds for Open Addressing Without Reordering"* (arXiv:2501.02305 / IEEE FOCS 2024), breaking a **40-year-old theoretical barrier** (dating back to Andrew Yao's 1985 conjecture) by demonstrating that an open-addressed hash table achieves:

1. **$O(1)$ amortized probe complexity**
2. **$O(\log \delta^{-1})$ expected worst-case search complexity**
3. **Strictly zero reordering / element displacement**, even at high load factors ($\ge 95\%$).

By synthesizing **Optimal Non-Reordering Elastic Open Addressing** with the **Fast Multipole Method**, this engine **completely replaces sorting and tree construction**:

* **Pointerless Spatial Indexing:** Replaces pointer-based octrees with flat spatial Morton arrays indexed via non-reordering open addressing.
* **Lock-Free Concurrency:** Because insertions never displace existing keys, parallel threads insert via single atomic compare-and-swap (`atomicCAS`) primitives without cascading eviction locks.
* **Contiguous SIMD Streaming:** Near-field direct evaluations (P2P) and far-field multipole translations (M2L) stream from contiguous memory blocks.
* **Near-Dependency Free:** Designed as self-contained mathematical building blocks.

```text
+-------------------------------------------------------------------------+
|                  Dynamic Particle / Coordinate Stream                   |
+-------------------------------------------------------------------------+
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

## Performance & Scaling Benchmarks

Empirical scaling benchmarks comparing naive direct evaluation, dense vectorized NumPy matrix kernels, JAX JIT compilation, and the Tree-Free FMM pipeline from $N = 100$ to $N = 100,000$ particles:

<p align="center">
  <img src="assets/benchmark_scaling_analysis.png" width="95%" alt="Scaling Benchmark Analysis">
</p>

### Execution Latency Benchmark Table


| Particle Count ($N$) | Naive Python CPU$O(N^2)$ | Vectorized NumPy$O(N^2)$ | JAX JIT Compiled$O(N^2)$ | **Vectorized Tree-Free FMM** **$O(N)$** |  Speedup vs. NumPy Direct  |
| :--------------------: | :------------------------: | :------------------------: | :------------------------: | :---------------------------------------: | :--------------------------: |
|    **$N = 100$**    |         0.91 ms         |         0.35 ms         |   46.8 ms*(dispatch)*   |               **3.17 ms**               |        $0.11\times$        |
|    **$N = 500$**    |         8.84 ms         |         13.5 ms         |         49.8 ms         |              **13.97 ms**              | $0.97\times$ *(crossover)* |
|   **$N = 2,000$**   |         244.5 ms         |         157.9 ms         |         50.3 ms         |              **16.69 ms**              |      **$9.5\times$**      |
|   **$N = 4,000$**   |         978.1 ms         |         593.5 ms         |         51.6 ms         |              **166.55 ms**              |      **$3.6\times$**      |
|   **$N = 8,000$**   |        3,912.4 ms        |        2,361.0 ms        |         74.2 ms         |              **188.67 ms**              |      **$12.5\times$**      |
|   **$N = 16,000$**   |       15,649.5 ms       |        9,444.0 ms        |         156.8 ms         |              **246.35 ms**              |      **$38.3\times$**      |
|   **$N = 64,000$**   |   250,391.7 ms (250s)   |   151,104.7 ms (151s)   |        2,509.2 ms        |             **2,741.17 ms**             |      **$55.1\times$**      |
|  **$N = 100,000$**  |   611,307.8 ms (611s)   |   368,908.0 ms (369s)   |        6,125.9 ms        |             **2,884.09 ms**             |     **$127.9\times$**     |

---

### Implementation Backends

* **CPU (Vectorized NumPy):** SIMD matrix broadcasts for P2M/M2L expansions; zero compiler dependencies (`core/fast_vectorized_fmm.py`).
* **Quantized & Bitpacked Engine:** $5.0\times$ cache compression via fixed-point integer bitfields, 64-bit Morton bitboards, and run-length cluster merging (`quantized_bitpacked_optimization/`) to trade even more performance for some precision.
* **JAX JIT (CPU/GPU):** Differentiable end-to-end pipeline with `@jax.jit` complex multipoles and reverse-mode AD (`core/jax_tree_free_fmm.py`).
* **NVIDIA CUDA:** Native kernel with `atomicCAS` lock-free insertions and fused `__shared__` memory tiles (`core/cuda_kernels/tree_free_fmm_kernel.cu`).
* **AMD ROCm / HIP:** Native AMD Radeon kernel with lock-free atomics and warp shuffle reductions (`core/hip_kernels/tree_free_fmm_kernel.hip`).
* **OpenAI Triton:** Block-tiled GPU kernel for PyTorch with fused SRAM potential evaluations (`core/cuda_kernels/triton_tree_free_fmm.py`).
* **OpenCL (Cross-Platform):** Vendor-neutral compute backend for AMD, Intel, and Apple GPUs (`core/opencl_kernels/tree_free_fmm_opencl.cl`).
* **WebGPU / WebGL 2.0:** In-browser WGSL compute shaders and GPU-instanced rendering up to 250k+ particles (`index.html`, `core/webgpu_kernels/`).
* **Zig Native:** Crossplatform, dependency-free near-instant-compilation C-ABI library for high-throughput CPU execution (`core/zig_backend.py`, `native/zig/`).

---

## Interactive Real-Time Simulation

<p align="center">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html">
    <img src="assets/simulation_demo.gif" alt="Tree-Free N-Body Real-Time GPU Animation" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html">
    <img src="https://img.shields.io/badge/▶_Launch_Live_Interactive_Simulation-blueviolet?style=for-the-badge&logo=webgl" alt="Launch Live Simulation">
  </a>
</p>

---

## Application Suite & Domain Case Studies

The `apps/` directory provides ten complete, runnable domain demonstrations:

### Application 1: Dynamic N-Body Galaxy Collision

* **Script:** [`apps/app1_galaxy_collision.py`](apps/app1_galaxy_collision.py)
* **Description:** Simulates the gravitational collision of two rotating spiral galaxies with continuous spatial re-binning in the elastic hash table without tree reallocations.

<p align="center">
  <img src="assets/app1_galaxy_collision.png" width="100%" alt="Galaxy Collision Time Evolution">
</p>

---

### Application 2: Continuous Hydrodynamic Vortex Field (Biot-Savart Law)

* **Script:** [`apps/app2_hydrodynamics.py`](apps/app2_hydrodynamics.py)
* **Description:** Evaluates the streamfunction and velocity streamlines for a Kelvin-Helmholtz vortex sheet instability across an Eulerian fluid grid.

<p align="center">
  <img src="assets/app2_hydrodynamic_vortex.png" width="70%" alt="Hydrodynamic Vortex">
</p>

---

### Application 3: Linear $O(N)$ Spatial Multipole Attention

* **Script:** [`apps/app3_spatial_attention.py`](apps/app3_spatial_attention.py)
* **Description:** Evaluates localized near-field attention alongside far-field cluster multipole moments on non-uniform spatial point clouds.

<p align="center">
  <img src="assets/app3_spatial_attention.png" width="95%" alt="Spatial Attention">
</p>

---

### Application 4: Multilevel Boids with 1€ Adaptive Anti-Jitter Filter

* **Script:** [`apps/app4_fmm_boids_1euro.py`](apps/app4_fmm_boids_1euro.py)
* **Description:** Combines multilevel FMM boid swarms (near-field separation + far-field cohesion) with a 1€ adaptive filter to suppress micro-collision jitter while maintaining responsive steering.

<p align="center">
  <img src="assets/app4_fmm_boids_1euro.png" width="95%" alt="1 Euro Boids">
</p>

---

### Application 5: 3D Protein Molecular Electrostatics (Bioinformatics)

* **Script:** [`apps/app5_bioinformatics.py`](apps/app5_bioinformatics.py)
* **Description:** Evaluates 3D screened Debye-Hückel and Coulomb solvation potentials over folded protein structures ($N = 3,000$ atoms) in 14.4 ms.

<p align="center">
  <img src="assets/app5_protein_electrostatics.png" width="70%" alt="Protein Electrostatics">
</p>

---

### Application 6: MuJoCo-Style Terrain Proximity & Ground Contacts

* **Script:** [`apps/app6_mujoco_proximity.py`](apps/app6_mujoco_proximity.py)
* **Description:** Computes soft-contact normal force vectors and ground-effect proximity distance fields for bipedal footpads traversing uneven terrain in 13.3 ms.

<p align="center">
  <img src="assets/app6_mujoco_proximity.png" width="95%" alt="MuJoCo Proximity">
</p>

---

### Application 7: High-Dimensional Continuous Graph & Memory Partitioning

* **Script:** [`apps/app7_highdim_memory.py`](apps/app7_highdim_memory.py)
* **Description:** Partitions 64-dimensional dense embedding vectors using Hyperplane Locality-Sensitive Hashing (LSH) into the non-reordering table at $3.8\times 10^6\text{ vectors/sec}$ with 0.029 ms query latency.

<p align="center">
  <img src="assets/app7_highdim_embeddings.png" width="95%" alt="High-Dim Embeddings">
</p>

---

### Application 8: High-to-Low Dimensional Manifold Unfolding (8D to 2D via Hash k-NN)

* **Script:** [`apps/app8_dimension_reduction_knn.py`](apps/app8_dimension_reduction_knn.py)
* **Description:** Demonstrates non-linear manifold unfolding (Laplacian Eigenmaps) from an 8D curved space to 2D by constructing the $k$-NN graph in $O(N)$ using the non-reordering hash without quadratic pairwise distance matrices.

<p align="center">
  <img src="assets/app8_dimension_reduction_knn.png" width="95%" alt="Dimension Reduction">
</p>

---

### Application 9: Lock-Free High-Dimensional Streaming Vector Database

* **Script:** [`apps/app9_streaming_vector_db.py`](apps/app9_streaming_vector_db.py)
* **Description:** Ingests $d = 128$ dimensional embeddings dynamically at $\sim 170,000\text{ vectors/sec}$ with zero reordering operations and provides sub-millisecond multi-probe approximate nearest-neighbor retrieval.

<p align="center">
  <img src="assets/app9_streaming_vector_db.png" width="95%" alt="Streaming Vector DB">
</p>

---

### Application 10: Matrix-Free Continuous Graph Neural Network (FMM-GNN)

* **Script:** [`apps/app10_continuous_gnn_fmm.py`](apps/app10_continuous_gnn_fmm.py)
* **Description:** Implements continuous spatial graph convolutions across unstructured node clouds in linear time without allocating adjacency matrices or storing explicit edge lists.

<p align="center">
  <img src="assets/app10_continuous_gnn_fmm.png" width="95%" alt="Continuous GNN FMM">
</p>

---

## Modular Architecture & Building Blocks

All packages are organized into standalone, decoupled directories. For complete file trees, individual module descriptions, and test suites, see **[`OVERVIEW.md`](OVERVIEW.md)**.


| Package                                                                      | Purpose & Focus Areas                           | Standalone Key Modules                                                                      |
| :----------------------------------------------------------------------------- | :------------------------------------------------ | :-------------------------------------------------------------------------------------------- |
| **[`core/`](core/)**                                                         | Core Tree-Free FMM & Elastic Hash Backends      | `elastic_hash.py`, `fast_vectorized_fmm.py`, `jax_tree_free_fmm.py`                         |
| **[`native/`](native/)**                                                     | Compiled Zig & C-ABI Systems Acceleration       | `native/include/tree_free_fmm.h`, `native/zig/src/simd_p2p.zig`                             |
| **[`quantized_bitpacked_optimization/`](quantized_bitpacked_optimization/)** | Systems & Cache-Line Optimizations              | `packed_vectorized_fmm.py`, `bitboard_occupancy.py`, `greedy_multipole_mesh.py`             |
| **[`neural_ops/`](neural_ops/)**                                             | Linear$O(N)$ Spatial AI & Attention Layers      | `multipole_attention.py`, `continuous_meshfree_gnn.py`, `diffusion_policy_fmm.py`           |
| **[`bioinformatics/`](bioinformatics/)**                                     | Structural Biology, Solvation & Omics           | `allosteric_druggability_engine.py`, `solvation_free_energy.py`, `kmer_elastic_hash.py`     |
| **[`algorithm_theory/`](algorithm_theory/)**                                 | Frontier TCS & Fast Graph Solvers               | `tree_free_geodesic_fmm.py`, `optimal_transport_fmm.py`, `spectral_meshfree_laplacian.py`   |
| **[`physics_simulation/`](physics_simulation/)**                             | Matrix-Free IPC Contact & Shell Mechanics       | `cloth_shell_simulation.py`, `tetrahedral_surgical_soft_robotics.py`                        |
| **[`graphics_rendering/`](graphics_rendering/)**                             | Real-Time GI & Radiance Fields                  | `dynamic_irradiance_cache.py`, `surfel_radiosity_gi.py`, `volumetric_fmm_ao.py`             |
| **[`game_mechanics_spatial/`](game_mechanics_spatial/)**                     | Spatial Computing & Swarm Mechanics             | `massive_crowd_flocking.py`, `harmonic_flow_field_pathfinding.py`                           |
| **[`video_streaming_codecs/`](video_streaming_codecs/)**                     | Video & Motion Compression Intelligence         | `perceptual_rate_controller.py`, `one_euro_video_stabilizer.py`, `ffmpeg_interop_bridge.py` |
| **[`apps/`](apps/)**                                                         | 10 Interactive Domain Case Studies & Benchmarks | `benchmark_suite.py`, `app1_galaxy_collision.py` ... `app10_continuous_gnn_fmm.py`          |

---

## Quickstart & Installation

### Prerequisites

* **Core Runtime:** Python 3.10+, `numpy`, `matplotlib`
* **GPU & Acceleration Packages (Optional):**
  * `jax`, `jaxlib` (Differentiable JAX JIT execution on CPU/GPU/TPU)
  * `torch`, `triton` (PyTorch neural operators & OpenAI Triton GPU kernels)
  * `pyopencl` (Cross-platform OpenCL compute for AMD Radeon, Intel, Apple Silicon)
  * `wgpu` (Python WebGPU WGSL compute shader runner)
* **Scientific & Media Packages (Optional):**
  * `scipy` (High-dimensional manifold unfolding & sparse solvers)
  * `Pillow` (Demo GIF generation & media processing)
* **Native C-ABI Toolchain (Optional):**
  * Zig Compiler 0.11+ / 0.13+ (for compiling bare-metal binaries in `native/zig/`)

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

---

## Theoretical References

1. **Optimal Bounds for Open Addressing Without Reordering.** Farach-Colton, Krapivin, Kuszmaul (2025). *IEEE FOCS 2024* / [arXiv:2501.02305](https://arxiv.org/abs/2501.02305).
2. **A Fast Algorithm for Particle Simulations.** Greengard, Rokhlin (1987). *Journal of Computational Physics*, 73(2), 325–348.
3. **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths.** Duan, Cheng, Mao, Yin, Ren (2025). *ACM STOC 2025 Best Paper* / [arXiv:2409.04354](https://arxiv.org/abs/2409.04354).
4. **More Asymmetry Yields Faster Matrix Multiplication.** Alman, Duan, Williams, Xu, Xu, Zhou (2024). [arXiv:2404.16349](https://arxiv.org/abs/2404.16349).
5. **Nearly-Linear Time Algorithms for Graph Laplacians.** Spielman, Teng (2011). *SIAM J. Comput.* 40(4), 981–1025.
6. **Approximate Distance Oracles.** Thorup, Zwick (2005). *Journal of the ACM*, 52(1), 1–24.
7. **1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in Interactive Systems.** Casiez, Roussel, Vogel (2012). *ACM CHI Conference on Human Factors in Computing Systems*.
8. **Incremental Potential Contact: Intersection- and Inversion-Free, Large-Deformation Dynamics.** Li, Ferguson, Schneider, Langlois, Zorin, Panozzo (2020). *ACM Transactions on Graphics (SIGGRAPH 2020)*.

---

## Citation

If you want to reference `tree-free-nbody-engine`, you can use this citation:

```bibtex
@software{tree_free_nbody_engine2026,
  author = {OpenAnimal},
  title  = {Tree-Free N-Body Engine: Octree-Free, Lock-Free O(N) Spatial Computing \& Multipole Mechanics},
  year   = {2026},
  url    = {https://github.com/OpenAnimal/tree-free-nbody-engine}
}
```
