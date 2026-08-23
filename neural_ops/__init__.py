"""
Neural Ops (`neural_ops`)
=========================
Neural Network Building Blocks powered by Tree-Free Spatial Hashing &
Multipole-Style Far-Field Aggregation (Farach-Colton, Krapivin, & Kuszmaul, 2025 Non-Reordering
Open Addressing).

Scope caveat (read before citing): "Multipole" in this package's layer names
denotes spatially bucketed aggregation with low-order per-cell moments (mean /
weighted centroid / dipole), used as a far-field approximation inside neural
operators. None of these implement Greengard & Rokhlin FMM operator hierarchies.
The elastic-hash-style spatial indexing is the load-bearing idea being
demonstrated. See `STATUS.md` for the honest accuracy/complexity table.

Core Modules:
- TreeFreeMultipoleAttention, MultiHeadMultipoleAttention: O(N) Spatial & Sequence Attention.
- SphericalMultipoleAttention: Directional Spherical-Harmonic Cluster Correlation.
- KernelIndependentNeuralOperator: Kernel-Independent FMM with SVD Skeletonization.
- HyperbolicMultipoleAttention: Non-Euclidean Poincaré/Lorentz Hyperbolic Attention.
- TreeFreeMultipoleFlowDrift: O(N) Drift Field for Continuous Flow Matching & Diffusion ODEs.
- NeuralPME: Linear-Spectral O(N + M log M) Particle-Mesh Ewald Neural Operator.
- MultipoleSpatialSSM: Multi-Dimensional Selective State Space Model with FMM Mixing.
- EquivariantMultipoleTransformerLayer: SE(3)-Equivariant Dual Scalar-Vector Self-Attention.
- MultipoleAdjointEngine: Exact Analytical VJP & Transposed Adjoint Backpropagation Engine.
- HierarchicalElasticKVCache: Multi-Resolution 3-Tier Streaming KV-Cache (experimental; recall-limited).
- NeuralSPHIPCLayer: All-Pairs SPH Hydrodynamics & IPC Contact Barrier Mechanics.
- ElasticMultipoleKVCache: Lock-Free, Contiguous O(1) Streaming KV-Cache.
- ContinuousMeshfreeGNNLayer: Continuous Geometric Graph Convolution without Adjacency Matrices.
- EquivariantMultipoleLayer: E(3)/SE(3) Equivariant Physical Field Injection.
"""

from .multipole_attention import TreeFreeMultipoleAttention, MultiHeadMultipoleAttention
from .spherical_multipole_attention import SphericalMultipoleAttention, compute_real_spherical_harmonics
from .kernel_independent_fmm import KernelIndependentNeuralOperator
from .hyperbolic_multipole_attention import HyperbolicMultipoleAttention
from .multipole_flow_drift import TreeFreeMultipoleFlowDrift
from .spectral_neural_pme import NeuralPME
from .multipole_mamba_ssm import MultipoleSpatialSSM
from .equivariant_transformer import EquivariantMultipoleTransformerLayer
from .autograd_adjoint_fmm import MultipoleAdjointEngine
from .hierarchical_elastic_kv_cache import HierarchicalElasticKVCache
from .neural_sph_ipc import NeuralSPHIPCLayer
from .elastic_kv_cache import ElasticMultipoleKVCache
from .continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer
from .equivariant_field_layer import EquivariantMultipoleLayer
from .flash_multipole_kernel import FlashMultipoleAttentionEngine
from .diffusion_policy_fmm import (
    TreeFreeDiffusionPolicy,
    DiffusionPolicyConfig,
    ConditionalScoreNetwork,
    TrajectoryRolloutResult,
)
from .multipole_gaussian_process import (
    MultipoleGaussianProcessLayer,
    GPRegressionResult,
    SVGPResult,
)
from .visual_transformer_ops import (
    MultiScaleVisualMultipoleAttention,
    MultimodalCrossMultipoleAttention,
    ConvMultipoleHybridLayer,
)

__version__ = "2.3.0"
__all__ = [
    "TreeFreeMultipoleAttention",
    "MultiHeadMultipoleAttention",
    "FlashMultipoleAttentionEngine",
    "MultiScaleVisualMultipoleAttention",
    "MultimodalCrossMultipoleAttention",
    "ConvMultipoleHybridLayer",
    "TreeFreeDiffusionPolicy",
    "DiffusionPolicyConfig",
    "ConditionalScoreNetwork",
    "TrajectoryRolloutResult",
    "MultipoleGaussianProcessLayer",
    "GPRegressionResult",
    "SVGPResult",
    "SphericalMultipoleAttention",
    "compute_real_spherical_harmonics",
    "KernelIndependentNeuralOperator",
    "HyperbolicMultipoleAttention",
    "TreeFreeMultipoleFlowDrift",
    "NeuralPME",
    "MultipoleSpatialSSM",
    "EquivariantMultipoleTransformerLayer",
    "MultipoleAdjointEngine",
    "HierarchicalElasticKVCache",
    "NeuralSPHIPCLayer",
    "ElasticMultipoleKVCache",
    "ContinuousMeshfreeGNNLayer",
    "EquivariantMultipoleLayer",
]
