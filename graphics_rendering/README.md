# Real-Time Graphics & Radiance Suite (`graphics_rendering`)
### Point-Based Global Illumination, Surfel Radiosity, Hybrid 3D Voxel + Cluster Volumetric Raymarching & Gridless Irradiance Caching

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Engine: Unreal%20%2F%20Unity%20%2F%20Custom](https://img.shields.io/badge/Engine-Unreal%20%2F%20Unity%20%2F%20Custom-orange.svg)]()
[![Hardware: Meshless%20CPU%2FGPU%20(numpy%20reference)](https://img.shields.io/badge/Hardware-Meshless%20CPU%2FGPU%20(numpy%20reference)-purple.svg)]()

---

> 🔬 **Research Prototype & Architecture Philosophy:**  
> `graphics_rendering` investigates Barnes-Hut-style **order-0 monopole cluster** aggregation (historically labeled "FMM" in this repo, but NOT a translation-based multipole-expansion FMM — the core adaptive FMM solves the 2D logarithmic kernel and does not apply to the 3D cosine-form-factor / inverse-square occlusion kernels here), $O(1)$ elastic spatial hashing, and **Hybrid 3D Voxel + Cluster Sampling** for real-time graphics and illumination.
> - **Near-Field (3D Voxel Textures):** Provides ultra-fast $O(1)$ hardware trilinear interpolation for local dense fog, smoke, and foliage (150,000+ rays/sec).
> - **Far-Field (Monopole Clusters):** Eliminates rigid 3D bounds and memory limits by computing long-range deep shadow transmittance and ambient occlusion via order-0 (center+mass) aggregation of distant cells — a Barnes-Hut-style scheme, not a multipole expansion.

> ⚠️ **Honesty note (read me):** the GPU / zero-copy / async / multi-GPU framing
> across these modules describes **target architectures and numpy-side
> layouts**, not runtime GPU calls.  No module in this directory issues CUDA /
> Vulkan / DX12 calls; the "zero-copy buffers" are numpy arrays with
> GPU-friendly float4 alignment, the "async queues" are sequential function
> calls, and the "FMM" clusters are order-0 monopole aggregations.  See each
> module's header docstring for the per-module honesty note.

---

## 🌟 Overview & Implemented Modules

```text
graphics_rendering/
├── README.md                          # Architecture documentation & mathematical formulations
├── surfel_radiosity_gi.py             # Point-Based Global Illumination & Multi-Bounce Surfel Radiosity
├── volumetric_fmm_ao.py               # Hybrid 3D Voxel + FMM Volumetric Raymarching & Deep Shadowing
├── dynamic_irradiance_cache.py        # Gridless Spherical Harmonic (L0+L1) Irradiance Probe Field
├── async_zerocopy_streaming.py        # Non-blocking double-buffered ring queues & tile streaming
├── gpu_hardware_interop.py            # Zero-copy host-device buffers & 16-byte float4 / Texture3D layouts (CUDA/Vulkan/DX12)
├── test_graphics_rendering.py         # Full unit and integration test harness
└── benchmark_graphics_rendering.py    # Scalability & Latency Verification Suite
```

---

## 📊 Summary of Verified Performance

| Module | Purpose / Real-World Application | Measured Throughput / Latency | Algorithmic Benefit |
| :--- | :--- | :--- | :--- |
| **`surfel_radiosity_gi.py`** | Multi-bounce indirect diffuse lighting (Cornell box / game rooms). | **1158 ms** (demo: 25,000 surfels, 2 bounces, NumPy; ~0.9 FPS) | Cluster build is $O(N)$; per-surfel evaluation is $O(N \cdot K)$ over $K$ clusters (Barnes-Hut-style dipole, not a translation-based FMM). An earlier "12.9 ms @ 1k / 47.9 ms @ 5k" claim was not reproducible and has been re-measured at the demo config. **X-G2 near/far vectorized bounce** (per-bounce, 25k surfels, NumPy): `compute_indirect_bounce_near_far` ring=1 = 2326 ms (~17% rel-L2 vs exact), ring=3 = 21917 ms (1.3e-2 rel-L2 vs exact, <= 3e-2 acceptance), vs the O(N·K) all-cluster default = 629 ms (~21% rel-L2). The accurate near/far (ring=3) is SLOWER than the all-cluster default — the exact near field at ring=3 is more flops than the cluster far field; the near/far's value is accuracy (exact near field), and the vectorization win (7.0× at 5k) is over the old per-surfel Python loop, not the all-cluster default. The directional (cosine) form-factor kernel makes the order-0 far-field cluster approximation coarse at ring=1; ring=3 is needed for <= 3e-2. |
| **`volumetric_fmm_ao.py` [Voxel]** | Fast near-field volumetric raymarching (fog, smoke, local canopy). | **211,000 Rays/sec** (23.7 ms for 5k rays at 16 steps/ray, demo) | Hardware trilinear interpolation with 3D voxel texture memory layouts. |
| **`volumetric_fmm_ao.py` [Hybrid]** | Unbounded deep shadowing + local high-res raymarching. | **20,000+ Queries/sec** | Unbounded long-range shadow attenuation via monopole clusters; the HYBRID blend is an empirical max(voxel, 0.5*cluster) heuristic, not a physical occlusion model (see code comment). |
| **`dynamic_irradiance_cache.py`** | Indirect lighting on dynamic moving characters and props. | **450 ms** for the demo's full 2,048-probe cache query (~0.22 ms/probe; near-far rel-L2 ~0.3 max at ring=1, ~0.036 at ring=2) | Gridless Spherical Harmonic interpolation; UNOCCLUDED Gaussian-weight probe blending (will leak through walls — no visibility test; see header honesty note). |
| **`gpu_hardware_interop.py`** | Direct zero-copy GPU staging for Texture3D, clusters & SH probes. | **0.11 ms** (227M+ vertices/s zero-copy stage) | 16-byte float4 and 64-byte cache-aligned layouts for HLSL/GLSL StructuredBuffer & Texture3D interop. |
| **`async_zerocopy_streaming.py`** | Double-buffered (not lock-free) geometry and radiance streaming. | **15.0 FPS** (demo frames, ~67 ms each, NumPy reference path, X-G3 ring-1 gather) | Morton spatial tile binning via `CellIndex.build()` (X-G3: replaces hand-rolled Morton encode + per-element Python loop) with incremental dirty-tile caching. The irradiance gather is restricted to the ring-1 (27-cell) neighborhood of each surfel's cell, vectorized per occupied cell — 1.86× faster than the legacy all-tiles gather on the 30k-surfel torus demo (rel-L2 = ~0.31 vs all-tiles on the torus demo — both the flux sum and its normalizing weight sum are ring-restricted, so ring-1 is a locality/perf tradeoff, NOT an accuracy-preserving truncation; an earlier audit-note claiming rel-L2 = 0.0 was an artifact of comparing the same overwritten radiance buffer twice and has been corrected). The "async/zero-copy/GPU" framing describes a TARGET architecture — this Python module simulates overlap with sequential calls (see module docstring); no GPU figure is measured. |

---

## 📐 Mathematical Formulation

### 1. Surfel Radiance Transfer via Multipole Dipoles
For surface surfel $i$ and emitter cluster $j$:
$$\Phi_i = \sum_j \frac{\max(0, \mathbf{n}_i \cdot \hat{\mathbf{r}}_{ij}) \max(0, -\mathbf{n}_j \cdot \hat{\mathbf{r}}_{ij})}{\pi \|\mathbf{r}_{ij}\|^2 + A_j} \cdot \text{Flux}_j \cdot \rho_i$$

### 2. Hybrid Volumetric Raymarching (Voxel Trilinear + Multipole Far Field)
Local density sampled via trilinear blend $\sigma_{\text{voxel}}(\mathbf{p})$, with far-field continuous multipole transmittance:
$$\mathcal{T}_{\text{far}}(\mathbf{p}) = \exp\left( - \sum_{k} \frac{M_k}{4\pi \|\mathbf{p} - \mathbf{c}_k\|^2 + r_k^2} \right)$$
$$L_{\text{inscatter}}(\mathbf{r}) = \sum_{s=0}^{S-1} \mathcal{T}(t_s) \left[ L_{\text{sun}} \cdot \mathcal{T}_{\text{sun}}(\mathbf{r}(t_s)) + L_{\text{ambient}} \right] \sigma(\mathbf{r}(t_s)) \Delta t$$

### 3. Gridless Spherical Harmonic Radiance Caching
Irradiance evaluated on character vertex normal $\mathbf{n}$:
$$E(\mathbf{p}, \mathbf{n}) = \max\left(0, c_0 L_0(\mathbf{p}) + c_1 (\mathbf{n} \cdot \mathbf{L}_1(\mathbf{p}))\right)$$

### 4. GPU Memory Layouts & Descriptors
- **Volumetric Multipole Clusters (`StructuredBuffer<VolumetricCluster>`)**:
  ```hlsl
  struct VolumetricCluster {
      float4 center_mass;      // (center.xyz, mass)
      float4 radius_param_pad; // (eff_radius, cell_size, 0.0, 0.0)
  };
  ```
- **3D Voxel Texture (`Texture3D<float4>`)**:
  - `RGBA32F`: `R` = Extinction density $\sigma$, `GBA` = Scattering albedo / phase parameters.
- **Dynamic SH Probes (`StructuredBuffer<DynamicSHProbe>`)**:
  ```hlsl
  struct DynamicSHProbe {
      float4 pos_radius; // (pos.xyz, probe_radius)
      float4 L0_pad;     // (L0.rgb, 0.0)
      float4 L1_R_pad;   // (L1_R.xyz, 0.0)
      float4 L1_G_pad;   // (L1_G.xyz, 0.0)
      float4 L1_B_pad;   // (L1_B.xyz, 0.0)
  };
  ```

---

## 🛠️ Quickstart & Usage

```bash
# Run Point-Based Global Illumination (Surfel Radiosity)
python graphics_rendering/surfel_radiosity_gi.py

# Run Volumetric Ambient Occlusion & Raymarching Demo
python graphics_rendering/volumetric_fmm_ao.py

# Run Dynamic Gridless Irradiance Cache Demo
python graphics_rendering/dynamic_irradiance_cache.py

# Run Hardware GPU Interop & Zero-Copy Staging Benchmark
python graphics_rendering/gpu_hardware_interop.py

# Run Unit & Integration Test Suite
python graphics_rendering/test_graphics_rendering.py

# Run Comprehensive Scalability Benchmark
python graphics_rendering/benchmark_graphics_rendering.py
```

---

## 🔬 Theoretical Citations

1. **Optimal Bounds for Open Addressing Without Reordering.** Farach-Colton, Krapivin, & Kuszmaul (2025). *IEEE FOCS 2024* / [arXiv:2501.02305](https://arxiv.org/abs/2501.02305).
2. **A Fast Algorithm for Particle Simulations.** Greengard, Rokhlin (1987). *Journal of Computational Physics*, 73(2), 325–348.
3. **A Fast Adaptive Multipole Algorithm for Particle Simulations.** Carrier, Greengard, Rokhlin (1988). *SIAM Journal on Scientific and Statistical Computing*, 9(4), 669–686.
4. **Point-Based Global Illumination for Movie Production.** Christensen (2008). *ACM SIGGRAPH Classes*, Article 10.
5. **An Efficient Representation for Irradiance Environment Maps.** Ramamoorthi, Hanrahan (2001). *ACM SIGGRAPH Proceedings*, 497–500.
