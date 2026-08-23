# Tree-Free Fast Multipole Machine Learning Engine (`neural_ops`)
### Sub-Quadratic Neural Building Blocks via Non-Reordering Open Addressing & Multipole Expansions — $O(N)$ at fixed grid depth, $N^{4/3}$-class for the multilevel flash engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Complexity: sub-quadratic](https://img.shields.io/badge/Complexity-O(N)%20fixed%20depth%20%C2%B7%20N%5E4%2F3%20multilevel-brightgreen.svg)]()
[![Memory: 0 MB NxN Matrix](https://img.shields.io/badge/VRAM-0%20MB%20N%C3%97N%20Matrix-purple.svg)]()
[![Hardware: SIMD / GPU Friendly](https://img.shields.io/badge/Hardware-Lock--Free%20%2F%20Tree--Free-orange.svg)]()

---

> 🔬 **Research Prototype & Exploratory Notice:**  
> `neural_ops` is an experimental research exploration investigating $O(N)$ tree-free multipole attention, lock-free spatial hashing, higher-order spherical harmonics, and geometric neural operators. While all modules include automated unit tests and scaling benchmarks, this code represents an exploratory prototype. Community audits, feedback, and pull requests are welcomed.

## 📦 Drop-in usage

Copy the `neural_ops/` folder into your project — it is **self-contained** (numpy
core; torch/jax acceleration optional and auto-detected). Inside the source
repository the canonical `core/` engines are used; standalone, dependency-free
fallbacks with identical outputs take over (see `neural_ops/_core_deps.py`;
parity pinned by `tests/neural_ops/test_dropin_standalone.py`).

```python
import numpy as np
from neural_ops import TreeFreeMultipoleAttention

coords = ...  # your (N, spatial_dim) positions, NORMALIZED to [0, 1)^d:
coords = (coords - coords.min(axis=0)) / (np.ptp(coords, axis=0) + 1e-9)

Q, K, V = ...  # float32 (N, d_model) projections
att = TreeFreeMultipoleAttention(embed_dim=Q.shape[-1], spatial_dim=coords.shape[-1],
                                 backend="numpy")  # or "torch" / "jax" if installed
out, meta = att.forward(Q, K, V, coords)
```

**Coordinate contract:** the spatial operators quantize onto a unit-grid
(`[0, 1)^dims`). Out-of-range coordinates are clipped — and since round 9 they
trigger a `RuntimeWarning` instead of failing silently (see
`neural_ops/_coord_contract.py`). Always min-max normalize as above.

Two advanced paths additionally want the full repository:
`EquivariantMultipoleLayer(kernel="tayloryukawa")` and the
`infinite_multipole_memory_network` example (both raise an informative
`ImportError` without `core/`); every other module is standalone.

---

## 💡 Executive Overview & Motivation

Modern Deep Learning (Transformers, Vision-Language Models, Equivariant GNNs, Diffusion, and World Models) is fundamentally bottlenecked by **all-pairs interaction complexity**:

1. **The Attention Bottleneck ($O(N^2)$):** Softmax Multi-Head Attention evaluates all $N 	imes N$ token affinities. In long-context LLMs ($100	ext{k}	ext{–}1	ext{M}$ tokens) and high-resolution vision ($4	ext{K}/8	ext{K}$ images, 3D point clouds), storing and computing dense attention matrices crashes GPU VRAM.
2. **The Truncation Dilemma in Physical AI:** Equivariant Graph Neural Networks (AlphaFold, MACE, NequIP, TorchMD-Net) artificially truncate interactions at local cutoff spheres ($r_{	ext{cut}} pprox 5	ext{–}10	ext{ \AA}$) because calculating all-pairs fields is $O(N^2)$, missing global allosteric and long-range electrostatic polarization.
3. **The Dynamic Tree Bottleneck on Accelerators:** Traditional hierarchical acceleration algorithms (Octrees, $k$-d trees, BVHs) require dynamic pointer allocations, recursion, and warp-divergent memory lookups every step, which severely serialize modern GPU Tensor Cores.

### 🌟 What `neural_ops` Solves
By marrying the **Fast Multipole Method (FMM)** (Greengard & Rokhlin, 1987; Carrier, Greengard, & Rokhlin, 1988) with **Optimal Non-Reordering Open Addressing** (Farach-Colton, Krapivin, & Kuszmaul, 2025), `neural_ops` provides **drop-in neural network building blocks** that achieve:

* **Linear-complexity spatial aggregation:** Bucketed near field + far cluster moments (Taylor / spherical-harmonic / proxy). The flash variant is $O(N \cdot K)$ far + $O(N \cdot w \cdot B_c)$ near (flat single-level, $N^{4/3}$-class) after T-D2's cluster-level far restructure; a true $O(N)$ multilevel far hierarchy is still future work. The layer modules are $O(N)$ at fixed grid depth.
* **$0	ext{ MB } N 	imes N$ Attention Matrices:** Never allocates or materializes quadratic pairwise matrices.
* **Lock-Free Concurrency (`atomicCAS`):** Non-reordering open addressing guarantees zero element displacement even at high load factors ($\ge 95\%$), enabling lock-free streaming on SIMD and GPU architectures.
* **True Infinite Global Receptive Field:** Unlike windowed attention (which truncates distant context), multipole attention retains continuous long-range communication.

---

## 🏛️ Comprehensive Architecture & Module Matrix

```
+-----------------------------------------------------------------------------------------------------------------------------------+
|                                                 NEURAL OPS (`neural_ops`) MATRIX                                                  |
+-----------------------------------------------------------------------------------------------------------------------------------+
|  1. Attention & Vision        |  2. Physical & Geometric AI     |  3. Generative & Probabilistic |  4. Memory & Autograd          |
|-------------------------------|---------------------------------|--------------------------------|--------------------------------|
| - TreeFreeMultipoleAttention  | - SphericalMultipoleAttention   | - TreeFreeDiffusionPolicy      | - HierarchicalElasticKVCache   |
| - MultiHeadMultipoleAttention | - EquivariantMultipoleLayer     | - MultipoleGaussianProcess     | - ElasticMultipoleKVCache      |
| - FlashMultipoleKernel        | - EquivariantTransformer        | - TreeFreeMultipoleFlowDrift   | - MultipoleAdjointEngine (VJP) |
| - VisualTransformerOps (ViT)  | - KernelIndependentNeuralOp     | - NeuralPME (Particle-Mesh)    |                                |
| - HyperbolicMultipoleAttn     | - ContinuousMeshfreeGNNLayer    | - MultipoleSpatialSSM (Mamba)  |                                |
|                               | - NeuralSPHIPCLayer             |                                |                                |
+-----------------------------------------------------------------------------------------------------------------------------------+
```

### Module Breakdown

| Module Class | Domain & Purpose | Asymptotic Complexity | Accuracy / Verification Status |
| :--- | :--- | :--- | :--- |
| **`TreeFreeMultipoleAttention`** | Linear spatial/sequence attention for 2D/3D tokens (ViTs, LiDAR, Point Clouds) | **$O(N)$ Compute / $O(N)$ Memory** | Far-field approx; rel-L2 ~0.5 for randn features (see `test_farfield_error.py`). Shape/NaN/Inf checked in `test_fmm_neural_ops.py`. |
| **`MultiHeadMultipoleAttention`** | Multi-head projection wrapper for Transformer architectures | **$O(N \cdot D)$** | Inherits `TreeFreeMultipoleAttention` accuracy. |
| **`FlashMultipoleAttentionEngine`** | Fused chunked near-field/far-field execution engine | **$O(N \cdot K)$ far + $O(N \cdot w \cdot B_c)$ near (flat single-level, $N^{4/3}$-class)** — true $O(N)$ far hierarchy is Round-7 task T-D2 | Far-field approx; no accuracy test yet (latency-only benchmark). |
| **`MultiScaleVisualMultipoleAttention`** | Multi-scale pyramid visual multipole attention & hybrid Conv-Multipole for Vision | **$O(N)$ Multi-Scale ViT** | Shape/NaN/Inf checked; no dense accuracy comparison. |
| **`TreeFreeDiffusionPolicy`** | Continuous action diffusion policy (DDPM/Flow Matching) with multipole spatial drift | **$O(H \cdot D)$ Action Chunks** | Shape/finite/finiteness checked in `benchmark_diffusion_and_gp.py`. |
| **`MultipoleGaussianProcessLayer`** | Matrix-free GP regression via Preconditioned CG & Sparse Variational GP (SVGP) | **$O(N \cdot \text{iters} \cdot \text{nnz})$ PCG / $O(N \cdot M^2)$ SVGP** | Mean max-abs error < 0.05, variance max error < 0.15 vs dense Cholesky (N=1200). NOT FMM-accelerated; cutoff-truncated. |
| **`SphericalMultipoleAttention`** | Directional spherical-harmonic cluster correlation (degree $L$ $Y_l^m$ of unit directions) | **$O(N \cdot (L+1)^2)$** | Shape/NaN/Inf checked; NOT a full solid-harmonic multipole expansion (no $r^l$ radial factor). |
| **`KernelIndependentNeuralOperator`** | KI-FMM operator with SVD skeletonization for arbitrary learned/neural kernels | **$O(N \cdot N_{\text{proxy}})$** | Shape/NaN/Inf checked; no dense accuracy comparison. |
| **`HyperbolicMultipoleAttention`** | Non-Euclidean attention in Poincaré ball with Fréchet centroids & Möbius addition | **$O(N)$ Hyperbolic** | Shape/NaN/Inf checked; no dense accuracy comparison. |
| **`TreeFreeMultipoleFlowDrift`** | Stein score & repulsive drift for continuous flow matching & score diffusion ODEs | **$O(N)$ Drift Field** | Dipole sign verified vs exact 2-charge field (rel-L2 = 9.9e-5). |
| **`NeuralPME`** | Linear-spectral Particle-Mesh Ewald solver (Short-range hash + Reciprocal NUFFT) | **$O(N + M \log M)$** | 2-particle periodic Ewald verified: force rel-L2 = 7.6e-7, Newton III = 4.3e-16, FD-of-potential rel-L2 = 4.3e-5. |
| **`MultipoleSpatialSSM`** | Multi-dimensional selective state space model combining 1D scan with FMM spatial mixing | **$O(N)$ State Space** | Shape/NaN/Inf checked; no dense accuracy comparison. |
| **`EquivariantMultipoleTransformerLayer`** | $\text{SE}(3)$-equivariant dual scalar-vector self-attention with long-range multipoles | **$O(N)$ $\text{SE}(3)$ Attention** | SE(3) equivariance verified: scalar_sim=0.995, vec_sim=0.997. |
| **`MultipoleAdjointEngine`** | Exact analytical Vector-Jacobian Product (VJP) — dense O(N²) reference, FD-verified ~3.6e-10; use as ground truth for approximate backward passes | **$O(N^2)$ exact reference** | FD-verified max rel-error = 3.57e-10. |
| **`HierarchicalElasticKVCache`** | 3-tier streaming KV-cache (Sliding window + Semantic LSH + Coarse Pyramid) | **$O(1)$ Append / $O(K)$ Decode** | **Experimental; recall varies with hp** (Elastic cache: recall 0.59→1.00 as hp 4→32; Hierarchical: 0.57–0.70; see `test_kv_cache_recall.py`). Tier-1/tier-2 dedup verified (rel-L2 = 3.6e-8 vs exact full attention). |
| **`NeuralSPHIPCLayer`** | Continuum mechanics layer (SPH fluid Navier-Stokes + IPC contact barrier) | **$O(N)$ Mesh-Free Continuum** | SPH density matches exact all-pairs sum (rel-L2 = 5.3e-8). |
| **`ContinuousMeshfreeGNNLayer`** | Continuous spatial graph convolution without edge lists or adjacency matrices | **$O(N)$ Continuous Conv** | Shape/NaN/Inf/ReLU checked. |
| **`EquivariantMultipoleLayer`** | $\text{SE}(3)$-equivariant vector field and scalar potential injection for Physical AI | **$O(N)$ Physical Field** | SE(3) equivariance verified: vec cosine = 0.9999. |

---

## 📐 Mathematical Formulations

### 1. Higher-Order Spherical Harmonic Multipole Attention
For particle coordinates $\mathbf{x}_i \in \mathbb{R}^3$, cluster moments expand in real spherical harmonics $Y_l^m$:
$$M_C^{lm} = \sum_{j \in C} \mathbf{v}_j \, \|\mathbf{x}_j - \mathbf{c}_C\|^l Y_l^m\left(rac{\mathbf{x}_j - \mathbf{c}_C}{\|\mathbf{x}_j - \mathbf{c}_C\|}
ight)$$
Far-field potential evaluated at query $\mathbf{x}_i$:
$$\mathbf{y}_i^{	ext{far}} = \sum_{C \in 	ext{Far}(i)} w_{iC} \sum_{l=0}^L \sum_{m=-l}^l M_C^{lm} \, Y_l^m\left(rac{\mathbf{x}_i - \mathbf{c}_C}{\|\mathbf{x}_i - \mathbf{c}_C\|}
ight)$$

### 2. Matrix-Free Gaussian Process Regression & Uncertainty
For $N$ observation points, the predictive mean and variance solve:
$$oldsymbol{\mu}_* = \mathbf{K}_* oldsymbol{lpha}, \quad oldsymbol{lpha} = (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_*^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$
evaluated in $O(N)$ time via sparse block Preconditioned Conjugate Gradient (PCG) without allocating full $N 	imes N$ Gram matrices.

### 3. Tree-Free Continuous Diffusion Policy & Flow Matching
Generates smooth robot action chunk trajectories $A \in \mathbb{R}^{H 	imes D}$ via conditioned velocity matching augmented with multipole spatial potential guidance:
$$rac{dA_t}{dt} = v_	heta(A_t, O, t) + \lambda 
abla_A \Phi_{	ext{multipole}}(A_t)$$

### 4. Kernel-Independent FMM via SVD Skeletonization
For arbitrary non-linear activation or learned neural kernel $K(\mathbf{x}, \mathbf{y})$, equivalent surface proxy charges $\mathbf{q}_{	ext{proxy}} \in \mathbb{R}^{N_{	ext{proxy}} 	imes D}$ satisfy:
$$\mathbf{q}_{	ext{proxy}} = (\mathbf{G}^T \mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{G}^T \mathbf{X}_C, \quad G_{j, p} = K(\mathbf{x}_j, \mathbf{y}_p^{	ext{proxy}})$$
Evaluating far-field target points requires zero manual derivative derivations: $\mathbf{y}_i = \sum_p K(\mathbf{x}_i, \mathbf{y}_p^{	ext{proxy}}) \mathbf{q}_{	ext{proxy}, p}$.

### 5. Hyperbolic Poincaré Attention
In Poincaré ball $\mathbb{B}_c^d$ with negative curvature $c > 0$, geodesic distance is evaluated via Möbius addition:
$$d_c(\mathbf{u}, \mathbf{v}) = rac{2}{\sqrt{c}} \operatorname{artanh}\left(\sqrt{c} \| -\mathbf{u} \oplus_c \mathbf{v} \|
ight)$$
Far-field clusters summarize representations around Riemannian Fréchet centroids $\mathbf{c}_C = \exp_0^c\left( rac{1}{|C|} \sum_{j \in C} \log_0^c(\mathbf{x}_j) 
ight)$.

### 6. Exact Dense-Adjoint VJP Backpropagation (Reference)
Computes exact analytical gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}, \frac{\partial \mathcal{L}}{\partial \mathbf{K}}, \frac{\partial \mathcal{L}}{\partial \mathbf{V}}, \frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ for the dense $O(N^2)$ attention operator. The implementation materializes five $N \times N$ arrays (`A`, `S`, `grad_A`, `grad_dot`, and an $(N, N, d)$ coords-diff chain) and is therefore $O(N^2)$ in time and memory — it is a finite-difference-verified (~3.6e-10) ground-truth reference, not an $O(N)$ transposed-FMM adjoint. A future true $O(N)$ adjoint via transposed FMM passes is the Round-7 plan's open task.

---

## 📊 Verification & Test Suites

Run the complete verification test suites:
```bash
# 1. Run the full neural_ops test suite (all operator tests)
python -m pytest tests/neural_ops/ -q

# 2. Run a single suite standalone, e.g.
python -m tests.neural_ops.test_fmm_neural_ops
python -m tests.neural_ops.test_neural_ops_advanced

# 3. Run diffusion policy and Gaussian process benchmarks
python neural_ops/benchmark_diffusion_and_gp.py
```

---

## 📜 Academic References & Citations

```bibtex
@article{farachcolton2025optimal,
  title={Optimal Bounds for Open Addressing Without Reordering},
  author={Farach-Colton, Mart{'\i}n and Krapivin, Andrew and Kuszmaul, William},
  journal={arXiv preprint arXiv:2501.02305},
  year={2025}
}

@article{greengard1987fast,
  title={A Fast Algorithm for Particle Simulations},
  author={Greengard, Leslie and Rokhlin, Vladimir},
  journal={Journal of Computational Physics},
  volume={73},
  number={2},
  pages={325--348},
  year={1987}
}

@article{carrier1988fast,
  title={A Fast Adaptive Multipole Algorithm for Particle Simulations},
  author={Carrier, J. and Greengard, Leslie and Rokhlin, Vladimir},
  journal={SIAM Journal on Scientific and Statistical Computing},
  volume={9},
  number={4},
  pages={669--686},
  year={1988}
}
```

---
*Maintained by the Tree-Free N-Body Engine open-source project.*
