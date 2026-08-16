# Tree-Free Fast Multipole Machine Learning Engine (`neural_ops`)
### Linear-Time $O(N)$ Neural Network Building Blocks via Non-Reordering Open Addressing & Multipole Expansions

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Complexity: O(N) Linear](https://img.shields.io/badge/Attention-O(N)%20Linear-brightgreen.svg)]()
[![Memory: 0 MB NxN Matrix](https://img.shields.io/badge/VRAM-0%20MB%20N%C3%97N%20Matrix-purple.svg)]()
[![Hardware: SIMD / GPU Friendly](https://img.shields.io/badge/Hardware-Lock--Free%20%2F%20Tree--Free-orange.svg)]()

---

> 🔬 **Research Prototype & Exploratory Notice:**  
> `neural_ops` is an experimental research exploration investigating $O(N)$ tree-free multipole attention, lock-free spatial hashing, higher-order spherical harmonics, and geometric neural operators. While all modules include automated unit tests and scaling benchmarks, this code represents an exploratory prototype. Community audits, feedback, and pull requests are welcomed.

---

## 💡 Executive Overview & Motivation

Modern Deep Learning (Transformers, Vision-Language Models, Equivariant GNNs, Diffusion, and World Models) is fundamentally bottlenecked by **all-pairs interaction complexity**:

1. **The Attention Bottleneck ($O(N^2)$):** Softmax Multi-Head Attention evaluates all $N 	imes N$ token affinities. In long-context LLMs ($100	ext{k}	ext{–}1	ext{M}$ tokens) and high-resolution vision ($4	ext{K}/8	ext{K}$ images, 3D point clouds), storing and computing dense attention matrices crashes GPU VRAM.
2. **The Truncation Dilemma in Physical AI:** Equivariant Graph Neural Networks (AlphaFold, MACE, NequIP, TorchMD-Net) artificially truncate interactions at local cutoff spheres ($r_{	ext{cut}} pprox 5	ext{–}10	ext{ \AA}$) because calculating all-pairs fields is $O(N^2)$, missing global allosteric and long-range electrostatic polarization.
3. **The Dynamic Tree Bottleneck on Accelerators:** Traditional hierarchical acceleration algorithms (Octrees, $k$-d trees, BVHs) require dynamic pointer allocations, recursion, and warp-divergent memory lookups every step, which severely serialize modern GPU Tensor Cores.

### 🌟 What `neural_ops` Solves
By marrying the **Fast Multipole Method (FMM)** (Greengard & Rokhlin, 1987) with **Optimal Non-Reordering Open Addressing** (Farach-Colton, Krapivin, Kuszmaul, 2025), `neural_ops` provides **drop-in neural network building blocks** that achieve:

* **Strict Linear $O(N)$ Complexity:** Computes exact near-field details while summarizing distant clusters via Taylor, spherical harmonic, and proxy multipole moments.
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

| Module Class | Domain & Purpose | Asymptotic Complexity |
| :--- | :--- | :--- |
| **`TreeFreeMultipoleAttention`** | Linear spatial/sequence attention for 2D/3D tokens (ViTs, LiDAR, Point Clouds) | **$O(N)$ Compute / $O(N)$ Memory** |
| **`MultiHeadMultipoleAttention`** | Multi-head projection wrapper for Transformer architectures | **$O(N \cdot D)$** |
| **`FlashMultipoleAttentionEngine`** | Fused chunked near-field/far-field execution engine | **$O(N)$ Fused** |
| **`MultiScaleVisualMultipoleAttention`** | Multi-scale pyramid visual multipole attention & hybrid Conv-Multipole for Vision | **$O(N)$ Multi-Scale ViT** |
| **`TreeFreeDiffusionPolicy`** | Continuous action diffusion policy (DDPM/Flow Matching) with multipole spatial drift | **$O(H \cdot D)$ Action Chunks** |
| **`MultipoleGaussianProcessLayer`** | Exact $O(N)$ GP regression via Preconditioned CG & Sparse Variational GP (SVGP) | **$O(N)$ PCG / $O(M^2)$ SVGP** |
| **`SphericalMultipoleAttention`** | Arbitrary degree $L$ spherical harmonic ($Y_l^m$) & solid tensor far-field attention | **$O(N \cdot (L+1)^2)$** |
| **`KernelIndependentNeuralOperator`** | KI-FMM operator with SVD skeletonization for arbitrary learned/neural kernels | **$O(N \cdot N_{	ext{proxy}})$** |
| **`HyperbolicMultipoleAttention`** | Non-Euclidean attention in Poincaré ball with Fréchet centroids & Möbius addition | **$O(N)$ Hyperbolic** |
| **`TreeFreeMultipoleFlowDrift`** | Stein score & repulsive drift for continuous flow matching & score diffusion ODEs | **$O(N)$ Drift Field** |
| **`NeuralPME`** | Linear-spectral Particle-Mesh Ewald solver (Short-range hash + Reciprocal NUFFT) | **$O(N + M \log M)$** |
| **`MultipoleSpatialSSM`** | Multi-dimensional selective state space model combining 1D scan with FMM spatial mixing | **$O(N)$ State Space** |
| **`EquivariantMultipoleTransformerLayer`** | $	ext{SE}(3)$-equivariant dual scalar-vector self-attention with long-range multipoles | **$O(N)$ $	ext{SE}(3)$ Attention** |
| **`MultipoleAdjointEngine`** | Exact analytical Vector-Jacobian Product (VJP) and transposed adjoint backprop | **$O(N)$ Training Memory** |
| **`HierarchicalElasticKVCache`** | 3-tier streaming KV-cache (Sliding window + Semantic LSH + Coarse Pyramid) | **$O(1)$ Append / $O(K)$ Decode** |
| **`NeuralSPHIPCLayer`** | Continuum mechanics layer (SPH fluid Navier-Stokes + IPC contact barrier) | **$O(N)$ Mesh-Free Continuum** |
| **`ContinuousMeshfreeGNNLayer`** | Continuous spatial graph convolution without edge lists or adjacency matrices | **$O(N)$ Continuous Conv** |
| **`EquivariantMultipoleLayer`** | $	ext{SE}(3)$-equivariant vector field and scalar potential injection for Physical AI | **$O(N)$ Physical Field** |

---

## 📐 Mathematical Formulations

### 1. Higher-Order Spherical Harmonic Multipole Attention
For particle coordinates $\mathbf{x}_i \in \mathbb{R}^3$, cluster moments expand in real spherical harmonics $Y_l^m$:
$$M_C^{lm} = \sum_{j \in C} \mathbf{v}_j \, \|\mathbf{x}_j - \mathbf{c}_C\|^l Y_l^m\left(rac{\mathbf{x}_j - \mathbf{c}_C}{\|\mathbf{x}_j - \mathbf{c}_C\|}ight)$$
Far-field potential evaluated at query $\mathbf{x}_i$:
$$\mathbf{y}_i^{	ext{far}} = \sum_{C \in 	ext{Far}(i)} w_{iC} \sum_{l=0}^L \sum_{m=-l}^l M_C^{lm} \, Y_l^m\left(rac{\mathbf{x}_i - \mathbf{c}_C}{\|\mathbf{x}_i - \mathbf{c}_C\|}ight)$$

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
$$d_c(\mathbf{u}, \mathbf{v}) = rac{2}{\sqrt{c}} \operatorname{artanh}\left(\sqrt{c} \| -\mathbf{u} \oplus_c \mathbf{v} \|ight)$$
Far-field clusters summarize representations around Riemannian Fréchet centroids $\mathbf{c}_C = \exp_0^c\left( rac{1}{|C|} \sum_{j \in C} \log_0^c(\mathbf{x}_j) ight)$.

### 6. Transposed Adjoint State VJP Backpropagation
Computes exact analytical gradients $rac{\partial \mathcal{L}}{\partial \mathbf{Q}}, rac{\partial \mathcal{L}}{\partial \mathbf{K}}, rac{\partial \mathcal{L}}{\partial \mathbf{V}}, rac{\partial \mathcal{L}}{\partial \mathbf{x}}$ via transposed FMM operations in $O(N)$ active memory, eliminating intermediate $N 	imes N$ autograd graph storage.

---

## 📊 Verification & Test Suites

Run the complete verification test suites:
```bash
# 1. Run all advanced neural operators unit tests
python neural_ops/test_neural_ops_advanced.py

# 2. Run core foundation tests
python neural_ops/test_fmm_neural_ops.py

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
```

---
*Maintained by the Tree-Free N-Body Engine open-source project.*
