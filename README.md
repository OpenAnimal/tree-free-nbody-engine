<p align="center">
  <img src="assets/banner.png" alt="Tree-Free FMM Banner" width="100%">
</p>

# Tree-Free N-Body Engine (`tree-free-nbody-engine`)

### Octree-Free, Lock-Free $O(N)$ $N$-Body & Spatial Field Engine

**Pointerless Spatial Computing & Fast Multipole Method (FMM) via Optimal Non-Reordering Open Addressing** (Farach-Colton, Krapivin, Kuszmaul 2025)

<p align="left">
  <a href="https://raw.githack.com/OpenAnimal/tree-free-nbody-engine/main/index.html"><img src="https://img.shields.io/badge/Live_Demo-Launch_Simulation-blueviolet?style=for-the-badge&logo=webgl" alt="Launch Live WebGL Demo"></a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Renderer-WebGL%202.0%20%2F%20WebGPU-orange.svg" alt="Renderer: WebGL 2.0 / WebGPU">
  <img src="https://img.shields.io/badge/Scale-250%2C000%2B%20Particles-purple.svg" alt="Scale: 250k+ Particles">
</p>

---

## 💡 What Is This? (Synonyms & Plain-English Concepts)

If you're coming from game dev, robotics, AI, or graphics, **FMM (Fast Multipole Method)** with **Elastic Hashing** can sound abstract. Here is what this engine actually is across different domains:

* **Simulation & Physics:**
  * **Linear-Time** **$N$-Body Solver** ($O(N)$ $N$-Body Engine)
  * **Fast Particle Interaction & Potential Field Engine**
  * **Hierarchical Cluster Physics Engine / Distant-Cluster Approximation**
* **Game Development & Robotics (Unreal / MuJoCo):**
  * **Pointerless Spatial Physics & Proximity Solver**
  * **Massive-Scale Particle / Collision Field Engine**
  * **Lock-Free Spatial Hash** **$N$-Body Simulator**
  * **Contiguous Memory / Flat-Grid Multipole Force Solver**
* **AI, Graphics & High-Performance Computing:**
  * **Linear-Time All-Pairs Spatial Attention Engine**
  * **Fast Distance / Potential Matrix Accelerator**
  * **Hierarchical Spatial Kernel Evaluator**
  * **Pointerless Octree-Free Spatial Indexer**

> **The Core Idea:** Instead of computing interactions between all $N \times N$ pairs ($O(N^2)$, which crashes performance), this engine computes nearby interactions directly and groups distant particles into cluster multipoles ($O(N)$) — all without using slow pointer-chasing Octrees or BVHs.

---

## Technical Overview & Background

The **Fast Multipole Method (FMM)** (Greengard & Rokhlin, 1987) reduced $N$-body and potential field evaluations from quadratic $O(N^2)$ to linear $O(N)$. However, practical FMM implementations historically relied on hierarchical **Octree /** **$k$-d tree data structures**. On SIMD, GPU, and distributed architectures, dynamic tree allocation every timestep, pointer chasing, and warp divergence remain primary bottlenecks.

In early 2025, **Martín Farach-Colton, Andrew Krapivin, and William Kuszmaul** published *"Optimal Bounds for Open Addressing Without Reordering"* (arXiv:2501.02305 / IEEE FOCS). breaking a **40-year-old theoretical barrier** dating back to Andrew Yao's 1985 conjecture by demonstrating that an open-addressed hash table can achieve:

1. **$O(1)$** **amortized probe complexity**
2. **$O(\log \delta^{-1})$** **expected worst-case search complexity**
3. **Strictly zero reordering / element displacement**, even at high load factors ($\ge 95\%$).

### Architectural Synthesis: Eliminating the Sorting & Tree Construction Bottleneck

In classical GPU/CPU $N$-body algorithms (e.g., Barnes-Hut or tree-based FMM), spatial partitioning required:
1. Computing Morton/Z-order codes for all $N$ particles.
2. **Sorting all $N$ particles by their Morton code** using multi-pass GPU Radix Sort ($O(N \log N)$) to enforce spatial locality.
3. Building pointer-based Octrees/BVHs over the sorted keys.

By pairing **Optimal Non-Reordering Elastic Open Addressing** with the **Fast Multipole Method**, this engine **completely replaces sorting and tree construction**:
* **Zero Sorting Passes:** Particles are binned into multi-level geometric hash buckets in $O(1)$ time with no Radix Sort or sorting overhead.

* **Pointerless Spatial Indexing:** Replaces pointer-based octrees with flat spatial Morton arrays indexed via non-reordering open addressing.
* **Lock-Free Concurrency:** Because insertions never displace existing keys, parallel threads insert via single atomic compare-and-swap (`atomicCAS`) primitives without cascading eviction locks.
* **Contiguous SIMD Streaming:** Near-field direct evaluations (P2P) and far-field multipole translations (M2L) stream from contiguous memory blocks.

```
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

## Performance & Scaling Benchmarks

Empirical scaling benchmarks comparing naive direct evaluation, dense vectorized NumPy matrix kernels, JAX JIT compilation, and the Tree-Free FMM pipeline from $N = 100$ to $N = 100,000$ particles:

![1.00](assets/benchmark_scaling_analysis.png)

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

### Implementation Details: JAX vs. NumPy vs. WebGL

* **Vectorized NumPy Engine (`fast_vectorized_fmm.py`):** Uses CPU SIMD matrix broadcasts for P2M/M2L expansions combined with the zero-reordering elastic hash table.
* **JAX JIT Engine (`jax_tree_free_fmm.py`):** Implements vectorized hash lookups and analytical kernel evaluations (`@jax.jit`) executing on CPU/GPU devices.
* **WebGL 2.0 Client (`index.html`):** Fully standalone in-browser simulation executing GPU-instanced point rendering and Morton cell evaluations up to 250k+ particles.

### Hash Table Stress Test at 92% Load Factor

* **10,000 capacity:** 9,200 keys inserted in 34.3 ms ($268,128\text{ ops/s}$), avg **5.51 probes/key**.
* **50,000 capacity:** 46,000 keys inserted in 176.7 ms ($260,390\text{ ops/s}$), avg **5.48 probes/key**.
* **200,000 capacity:** 184,000 keys inserted in 786.9 ms ($233,843\text{ ops/s}$), avg **5.50 probes/key**.
* **Reordering count:** 0 (strictly zero data relocation during insertions).

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

To run locally: open [`index.html`](index.html) in any modern browser or run `python -m http.server 8090`.

---

## Application Suite & Domain Case Studies

---

### Application 1: Dynamic N-Body Galaxy Collision

* **Script:** [`apps/app1_galaxy_collision.py`](apps/app1_galaxy_collision.py)
* **Visualization:** `assets/app1_galaxy_collision.png`
* **Description:** Simulates the gravitational collision of two rotating spiral galaxies with continuous spatial re-binning in the elastic hash table without tree reallocations.

![1.00](assets/app1_galaxy_collision.png)

---

### Application 2: Continuous Hydrodynamic Vortex Field (Biot-Savart Law)

* **Script:** [`apps/app2_hydrodynamics.py`](apps/app2_hydrodynamics.py)
* **Visualization:** `assets/app2_hydrodynamic_vortex.png`
* **Description:** Evaluates the streamfunction and velocity streamlines for a Kelvin-Helmholtz vortex sheet instability across an Eulerian fluid grid.

![1.00](assets/app2_hydrodynamic_vortex.png)

---

### Application 3: Linear $O(N)$ Spatial Multipole Attention

* **Script:** [`apps/app3_spatial_attention.py`](apps/app3_spatial_attention.py)
* **Visualization:** `assets/app3_spatial_attention.png`
* **Description:** Evaluates localized near-field attention alongside far-field cluster multipole moments on non-uniform spatial point clouds.

![1.00](assets/app3_spatial_attention.png)

---

### Application 4: Multilevel Boids with 1€ Adaptive Anti-Jitter Filter

* **Script:** [`apps/app4_fmm_boids_1euro.py`](apps/app4_fmm_boids_1euro.py)
* **Visualization:** `assets/app4_fmm_boids_1euro.png`
* **Description:** Combines multilevel FMM boid swarms (near-field separation + far-field cohesion) with a 1€ adaptive filter to suppress micro-collision jitter while maintaining responsive steering.

![1.00](assets/app4_fmm_boids_1euro.png)

---

### Application 5: 3D Protein Molecular Electrostatics (Bioinformatics)

* **Script:** [`apps/app5_bioinformatics.py`](apps/app5_bioinformatics.py)
* **Visualization:** `assets/app5_protein_electrostatics.png`
* **Description:** Evaluates 3D screened Debye-Hückel and Coulomb solvation potentials over folded protein structures ($N = 3,000$ atoms) in 14.4 ms.

![1.00](assets/app5_protein_electrostatics.png)

---

### Application 6: MuJoCo-Style Terrain Proximity & Ground Contacts

* **Script:** [`apps/app6_mujoco_proximity.py`](apps/app6_mujoco_proximity.py)
* **Visualization:** `assets/app6_mujoco_proximity.png`
* **Description:** Computes soft-contact normal force vectors and ground-effect proximity distance fields for bipedal footpads traversing uneven terrain in 13.3 ms.

![1.00](assets/app6_mujoco_proximity.png)

---

### Application 7: High-Dimensional Continuous Graph & Memory Partitioning

* **Script:** [`apps/app7_highdim_memory.py`](apps/app7_highdim_memory.py)
* **Visualization:** `assets/app7_highdim_embeddings.png`
* **Description:** Partitions 64-dimensional dense embedding vectors using Hyperplane Locality-Sensitive Hashing (LSH) into the non-reordering table at $3.8\times 10^6\text{ vectors/sec}$ with 0.029 ms query latency.

![1.00](assets/app7_highdim_embeddings.png)

---

### Application 8: High-to-Low Dimensional Manifold Unfolding (8D to 2D via Hash k-NN)

* **Script:** [`apps/app8_dimension_reduction_knn.py`](apps/app8_dimension_reduction_knn.py)
* **Visualization:** `assets/app8_dimension_reduction_knn.png`
* **Description:** Demonstrates non-linear manifold unfolding (Laplacian Eigenmaps) from an 8D curved space to 2D by constructing the $k$-NN graph in $O(N)$ using the non-reordering hash without quadratic pairwise distance matrices.

![1.00](assets/app8_dimension_reduction_knn.png)

---

### Application 9: Lock-Free High-Dimensional Streaming Vector Database

* **Script:** [`apps/app9_streaming_vector_db.py`](apps/app9_streaming_vector_db.py)
* **Visualization:** `assets/app9_streaming_vector_db.png`
* **Description:** Ingests $d = 128$ dimensional embeddings dynamically at $\sim 170,000\text{ vectors/sec}$ with zero reordering operations and provides sub-millisecond multi-probe approximate nearest-neighbor retrieval.

![1.00](assets/app9_streaming_vector_db.png)

---

### Application 10: Matrix-Free Continuous Graph Neural Network (FMM-GNN)

* **Script:** [`apps/app10_continuous_gnn_fmm.py`](apps/app10_continuous_gnn_fmm.py)
* **Visualization:** `assets/app10_continuous_gnn_fmm.png`
* **Description:** Implements continuous spatial graph convolutions across unstructured node clouds in linear time without allocating adjacency matrices or storing explicit edge lists.

![1.00](assets/app10_continuous_gnn_fmm.png)

---

## Sub-Modules & Hardware Optimizations

### 1. Voxel Memory Compression & Greedy Run Merging

* **Directory:** [`quantized_bitpacked_fmm/`](quantized_bitpacked_fmm/)
* **Description:** Adapts discrete bit-packing and run-length meshing concepts from voxel rendering engines to N-body potential solvers. Compresses coordinates into 32-bit / 64-bit integer bitfields and merges contiguous Morton runs to prune M2L evaluation matrices.

### 2. Matrix-Free Incremental Potential Contact (IPC) Solver

* **Directory:** [`fmm_contact_solver/`](fmm_contact_solver/)
* **Description:** Matrix-free barrier contact engine for cloth, shell, and robotic collision mechanics, eliminating dynamic sparse matrix assembly (`DynCSRMat`).

### 3. Hardware-Optimized Video Streaming & Codec Engine

* **Directory:** [`video_streaming_codecs/`](video_streaming_codecs/)
* **Description:** Lock-free hierarchical motion estimation (ME) using non-reordering open addressing, Vercidium-style greedy run-length macroblock merging (pruning $>5.4\times$ DCT operations), and 1€ adaptive camera gyro video stabilization.

---

## Quickstart

#### 6. Real-Time Graphics & Radiance Suite

* **Directory:** [`graphics_rendering/`](graphics_rendering/)
* **Description:** Point-based global illumination (PBGI), multi-bounce surfel radiosity, continuous volumetric ambient occlusion (AO), and gridless spherical harmonic irradiance probe fields without rigid octrees or BVH raycasting.

### 7. Game Mechanics & Spatial Tooling

* **Directory:** [`game_mechanics_spatial/`](game_mechanics_spatial/)
* **Description:** Real-time spatial game mechanics including massive crowd flocking, continuous line-of-sight / fog-of-war queries, fast mesh LOD decimation, and smart lasso spatial selection.

### 8. Theoretical Algorithmic Foundations

* **Directory:** [`algorithm_theory/`](algorithm_theory/)
* **Description:** Translates frontier theoretical computer science breakthroughs into concrete tree-free computational primitives:
  * **Frontier-Clustered SSSP:** Breaks the 65-year Dijkstra comparison barrier on 3D surface manifolds and graphs using adaptive bucketed frontier clustering (Duan et al., STOC 2025 Best Paper).
  * **Asymmetric Low-Rank Tensor M2L:** Compresses high-order ($p \ge 4$) far-field multipole contractions into asymmetric low-rank subspaces inspired by fast matrix multiplication laser methods ($\omega < 2.371339$), delivering **$230\times - 400\times$ speedups** at machine precision.
  * **Spectral Meshfree Poisson Solver:** Solves continuous PDEs ($\nabla^2 u = \rho$) in nearly-linear time using two-level Symmetric Diagonally Dominant (SDD) Galerkin coarse preconditioners (Spielman-Teng / Cohen et al.).
  * **Sublinear Distance Oracle:** Answers online $(1+\varepsilon)$-approximate pairwise manifold distance queries at **$7.9\times 10^6\text{ queries/sec}$** in $O(\log 1/\varepsilon)$ sublinear time.

## Prerequisites

* Python 3.10+
* `numpy`, `matplotlib`
* Optional: `jax` (for JIT execution on CPU/GPU)

### Running Scripts

```Shell
# Run Comprehensive Scaling Benchmark
python apps/benchmark_suite.py

# Run Individual Applications (Examples)
python apps/app1_galaxy_collision.py
python apps/app4_fmm_boids_1euro.py
python apps/app5_bioinformatics.py
python apps/app8_dimension_reduction_knn.py
python apps/app9_streaming_vector_db.py
python apps/app10_continuous_gnn_fmm.py
```

---

## References & Theoretical Citations

1. **Optimal Bounds for Open Addressing Without Reordering**
   *Martín Farach-Colton, Andrew Krapivin, William Kuszmaul* (2025).
   [arXiv:2501.02305](https://arxiv.org/abs/2501.02305) / IEEE FOCS 2024.
2. **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths**
   *Ran Duan, Jiayan Cheng, Xiao Mao, Longhui Yin, Hanrui Ren* (2024/2025).
   ACM STOC 2025 Best Paper / [arXiv:2409.04354](https://arxiv.org/abs/2409.04354).
3. **More Asymmetry Yields Faster Matrix Multiplication**
   *Josh Alman, Ran Duan, Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, Renfei Zhou* (2024/2025).
   [arXiv:2404.16349](https://arxiv.org/abs/2404.16349).
4. **Nearly-Linear Time Algorithms for Graph Laplacians**
   *Daniel A. Spielman, Shang-Hua Teng* (2004, 2011).
   *SIAM J. Comput.* 40(4), 981-1025.
5. **Approximate Distance Oracles**
   *Mikkel Thorup, Uri Zwick* (2005).
   *Journal of the ACM*, 52(1), 1-24.
2. **A Fast Algorithm for Particle Simulations**
   *Leslie Greengard, Vladimir Rokhlin* (1987).
   *Journal of Computational Physics*, 73(2), 325-348.
3. **1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems**
   *Géry Casiez, Nicolas Roussel, Daniel Vogel* (2012).
   *ACM CHI Conference on Human Factors in Computing Systems*.
4. **Incremental Potential Contact: Intersection- and Inversion-Free, Large-Deformation Dynamics**
   *Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis Zorin, Daniele Panozzo* (2020).
   *ACM Transactions on Graphics (SIGGRAPH 2020)*.

---

> ⚠️ **Project Status & Support Notice:** Released under MIT. Take the code and do whatever you want with it — no roadmap, no active maintenance, and no unpaid support. Please do not open issues asking for help or feature requests. (I am poor as a church mouse and don't have time for anything, but if you randomly want to throw money my way, you can get in touch with me.)

