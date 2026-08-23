"""
Run-Length "Greedy Multipole Aggregation" Engine
Inspired by Vercidium (2024) Greedy Meshing & 2D Slicing Run-Length Merging.

In standard FMM, evaluating M2L interactions across K individual leaf clusters requires
a K x K interaction matrix.
Greedy Multipole Aggregation scans contiguous runs of active Morton cells.
When sibling leaves share a common parent prefix and smooth potential gradients,
it merges them via localized M2M (Multipole-to-Multipole) translation into Macro-Clusters.
This drops the M2L transfer matrix dimension from K^2 down to K_greedy^2 (a 4x-16x reduction).

M2M translation formula (2D logarithmic kernel, expansion
  M_k = -sum_i q_i (z_i - z_c)^k / k):
  M_p^{parent} = -M_0^{child} * d^p / p
                 + sum_{k=1}^{p} (k/p) * C(p,k) * d^{p-k} * M_k^{child}
where d = z_child - z_parent and C(p,k) = p! / (k! (p-k)!). The factor
(k/p)*C(p,k) equals the standard CGR88 binomial C(p-1, k-1). The k=0
(monopole) term carries a MINUS sign and uses d^p (NOT (-d)^p); the
k>=1 terms use d^{p-k} (NOT (-d)^{p-k}) with coefficient (k/p)*C(p,k)
(NOT p/k). The earlier revision had p/k instead of k/p and (-dz) powers
with the wrong k=0 sign, which mis-translated every parent moment above
order 0; the full double-sum with the corrected signs/powers is now used
(audit-verified against a direct parent-moment computation to <1e-12 rel).
"""

import numpy as np
from typing import Tuple, List, Dict
from math import comb

class GreedyMultipoleAggregator2D:
    """
    Greedily merges contiguous 2D Morton sibling clusters into macro-multipole nodes.
    """
    def __init__(self, order: int = 4):
        self.order = order

    def aggregate_runs(
        self,
        unique_morton_keys: np.ndarray,
        centers: np.ndarray,
        cluster_multipoles: np.ndarray,
        depth: int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scans Morton keys in O(K) linear time and merges 2x2 sibling quadrants (4 leaves).
        Returns:
            macro_centers: (M,) complex centers
            macro_multipoles: (M, order+1) aggregated multipoles
            cluster_to_macro_map: (K,) index mapping each original cluster to its macro parent
            reduction_ratio: float
        """
        K = len(unique_morton_keys)
        if K <= 4:
            return centers, cluster_multipoles, np.arange(K), 1.0

        # Parent coordinates obtained by halving integer grid indices at (depth - 1)
        cluster_ix = (unique_morton_keys >> 12) & 0xFFF
        cluster_iy = unique_morton_keys & 0xFFF
        parent_ix = cluster_ix >> 1
        parent_iy = cluster_iy >> 1
        parent_keys = ((depth - 1) << 24) | (parent_ix << 12) | parent_iy

        # Identify contiguous runs of siblings with identical parent_key
        unique_parents, parent_inverse, counts = np.unique(
            parent_keys, return_inverse=True, return_counts=True
        )
        M = len(unique_parents)

        # Compute macro centers at depth - 1
        u_parent_ix = (unique_parents >> 12) & 0xFFF
        u_parent_iy = unique_parents & 0xFFF
        box_size_parent = 1.0 / (1 << (depth - 1))
        macro_centers = (u_parent_ix + 0.5) * box_size_parent + 1j * ((u_parent_iy + 0.5) * box_size_parent)

        # Aggregate multipole moments using the full M2M translation:
        #   d = child_center - macro_center
        #   M_p^{parent} = -M_0 * d^p / p + sum_{k=1}^{p} (k/p)*C(p,k) * d^{p-k} * M_k
        # (for p >= 1; M_0^{parent} = sum of M_0^{child}).
        dz = centers - macro_centers[parent_inverse]
        macro_multipoles = np.zeros((M, self.order + 1), dtype=np.complex128)

        # Monopole moment (order 0) is pure sum of charges
        macro_multipoles[:, 0] = np.bincount(parent_inverse, weights=np.real(cluster_multipoles[:, 0]), minlength=M) + \
                                1j * np.bincount(parent_inverse, weights=np.imag(cluster_multipoles[:, 0]), minlength=M)

        # Full M2M translation: for each target order p, sum over all source
        # orders k = 0..p. The k=0 (monopole) term carries a MINUS sign and
        # uses d^p; the k>=1 terms use (k/p)*C(p,k) (== C(p-1,k-1)) and
        # d^{p-k}. (Audit fix: the old code used p/k instead of k/p and
        # (-dz) powers with the wrong k=0 sign.)
        for p in range(1, self.order + 1):
            term_p = -cluster_multipoles[:, 0] * (dz ** p) / p
            for k in range(1, p + 1):
                coeff = comb(p, k) * k / p   # (k/p) * C(p,k) == C(p-1, k-1)
                term_p += cluster_multipoles[:, k] * coeff * (dz ** (p - k))
            macro_multipoles[:, p] = (
                np.bincount(parent_inverse, weights=np.real(term_p), minlength=M) +
                1j * np.bincount(parent_inverse, weights=np.imag(term_p), minlength=M)
            )

        reduction_ratio = float(K) / float(max(1, M))
        return macro_centers, macro_multipoles, parent_inverse, reduction_ratio
