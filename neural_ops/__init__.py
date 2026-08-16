"""
Neural Ops (`neural_ops`)
=========================
Linear-Time O(N) Neural Network Building Blocks powered by
Tree-Free Fast Multipole Method (FMM) & 2025 Farach-Colton Non-Reordering Open Addressing.

Modules:
- TreeFreeMultipoleAttention: O(N) Spatial & Sequence Attention for ViTs, Point Clouds & LLMs.
- ElasticMultipoleKVCache: Lock-Free, Contiguous O(1) Streaming KV-Cache for Long-Context LLMs.
- ContinuousMeshfreeGNNLayer: Continuous Geometric Graph Convolution without Adjacency Matrices.
- EquivariantMultipoleLayer: E(3)/SE(3) Equivariant Physical Field Injection for Molecular & World Models.
"""

from .multipole_attention import TreeFreeMultipoleAttention, MultiHeadMultipoleAttention
from .elastic_kv_cache import ElasticMultipoleKVCache
from .continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer
from .equivariant_field_layer import EquivariantMultipoleLayer

__version__ = "1.0.0"
__all__ = [
    "TreeFreeMultipoleAttention",
    "MultiHeadMultipoleAttention",
    "ElasticMultipoleKVCache",
    "ContinuousMeshfreeGNNLayer",
    "EquivariantMultipoleLayer",
]
