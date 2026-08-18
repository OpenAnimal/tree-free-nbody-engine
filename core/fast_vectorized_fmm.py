"""
Vectorized, High-Throughput Matrix Kernel Engine for Tree-Free FMM
Replaces Python for-loops with dense block SIMD matrix multiplications & vectorized CGR88 kernels.

Implements:
- Vectorized P2M (Particle to Multipole)
- Vectorized M2L (Multipole to Local) with full CGR88 binomial expansion tensor
- Vectorized L2P (Local to Particle potential & force)
- Block-tiled symmetric P2P near-field summation
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional, Union
try:
    from .cgr88_adaptive_fmm import CGR88AdaptiveFMM, GreengardRokhlin87RegularFMM, exact_direct_nbody_2d, exact_direct_nbody_forces_2d
    from .elastic_hash import ElasticHashTable
    from .tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d
except ImportError:
    from cgr88_adaptive_fmm import CGR88AdaptiveFMM, GreengardRokhlin87RegularFMM, exact_direct_nbody_2d, exact_direct_nbody_forces_2d
    from elastic_hash import ElasticHashTable
    from tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d


class FastVectorizedFMM:
    """
    Vectorized FMM Engine with Farach-Colton Non-Reordering Hash & CGR88 expansions.
    Executes cluster-cluster M2L interactions as a single vectorized matrix broadcast.
    """
    def __init__(
        self,
        depth: int = 4,
        order: int = 6,
        softening: float = 0.0
    ):
        self.depth = depth
        self.order = order
        self.softening = softening
        self.grid_res = 1 << depth
        self.hash_table = ElasticHashTable(capacity=self.grid_res * self.grid_res * 2, delta=0.05)

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        grid_res = self.grid_res
        p = self.order
        box_size = 1.0 / grid_res
        
        # 1. Morton Quantization & Bucket Binning
        ix = np.clip((positions[:, 0] * grid_res).astype(np.int32), 0, grid_res - 1)
        iy = np.clip((positions[:, 1] * grid_res).astype(np.int32), 0, grid_res - 1)
        morton_keys = (self.depth << 24) | (ix << 12) | iy
        
        unique_keys, inverse_indices = np.unique(morton_keys, return_inverse=True)
        num_clusters = len(unique_keys)
        
        cluster_ix = (unique_keys >> 12) & 0xFFF
        cluster_iy = unique_keys & 0xFFF
        cx = (cluster_ix + 0.5) * box_size
        cy = (cluster_iy + 0.5) * box_size
        centers = cx + 1j * cy  # (num_clusters,)
        
        # 2. Vectorized P2M (Particle to Multipole Moments)
        p_centers = centers[inverse_indices]
        z_pts = (positions[:, 0] + 1j * positions[:, 1]) - p_centers
        
        cluster_m = np.zeros((num_clusters, p + 1), dtype=np.complex128)
        cluster_m[:, 0] = np.bincount(inverse_indices, weights=charges, minlength=num_clusters)
        
        dz_pow = np.ones(N, dtype=np.complex128)
        for k in range(1, p + 1):
            dz_pow *= z_pts
            term = -charges * dz_pow / k
            cluster_m[:, k] = (
                np.bincount(inverse_indices, weights=np.real(term), minlength=num_clusters) +
                1j * np.bincount(inverse_indices, weights=np.imag(term), minlength=num_clusters)
            )
                               
        # 3. Vectorized M2L (CGR88 Multipole to Local Translation Matrix)
        # delta = tgt_center - src_center -> centers[:, None] - centers[None, :]
        delta = centers[:, None] - centers[None, :]
        dx = cluster_ix[:, None] - cluster_ix[None, :]
        dy = cluster_iy[:, None] - cluster_iy[None, :]
        well_separated = (np.abs(dx) > 1) | (np.abs(dy) > 1)  # (num_clusters, num_clusters)
        
        delta_safe = np.where(well_separated, delta, 1.0 + 0.0j)
        
        # cluster_l: (num_clusters, p + 1)
        cluster_l = np.zeros((num_clusters, p + 1), dtype=np.complex128)
        
        # l = 0: c_0 = a_0 * ln(delta) + sum_{k=1}^p a_k / (delta^k)
        term_l0 = cluster_m[None, :, 0] * np.log(delta_safe)
        for k in range(1, p + 1):
            term_l0 += cluster_m[None, :, k] / (delta_safe ** k)
        cluster_l[:, 0] = np.sum(np.where(well_separated, term_l0, 0.0), axis=1)
        
        # l >= 1: c_l = (a_0 * (-1)^(l-1)) / (l * delta^l) + sum_{k=1}^p [ (-1)^l * binom(k+l-1, l) * a_k ] / (delta^(k+l))
        for l in range(1, p + 1):
            term_l = cluster_m[None, :, 0] * ((-1) ** (l - 1)) / (l * (delta_safe ** l))
            for k in range(1, p + 1):
                binom_factor = ((-1) ** l) * math.comb(k + l - 1, l)
                term_l += binom_factor * cluster_m[None, :, k] / (delta_safe ** (k + l))
            cluster_l[:, l] = np.sum(np.where(well_separated, term_l, 0.0), axis=1)
        
        # 4. Vectorized L2P (Local to Particle potential and force evaluation)
        # Phi_i = Re( sum_{l=0}^p c_l * z_pts^l )
        # Psi'(z) = sum_{l=1}^p l * c_l * z_pts^(l-1)
        p_local = cluster_l[inverse_indices]  # (N, p + 1)
        
        far_pot_complex = p_local[:, 0].copy()
        far_deriv_complex = np.zeros(N, dtype=np.complex128)
        
        z_pow = np.ones(N, dtype=np.complex128)
        for l in range(1, p + 1):
            far_deriv_complex += l * p_local[:, l] * z_pow
            z_pow *= z_pts
            far_pot_complex += p_local[:, l] * z_pow
            
        potentials = np.real(far_pot_complex)
        forces_x = -np.real(far_deriv_complex)
        forces_y = np.imag(far_deriv_complex)
        
        # 5. Fast Local Direct Near-Field P2P (Evaluate self and adjacent 3x3 neighbor buckets)
        cluster_indices_list = [np.where(inverse_indices == c)[0] for c in range(num_clusters)]
        near_cluster_pairs = np.argwhere((np.abs(dx) <= 1) & (np.abs(dy) <= 1))
        eps2 = self.softening * self.softening

        for c1, c2 in near_cluster_pairs:
            idx1 = cluster_indices_list[c1]
            if len(idx1) == 0:
                continue

            if c1 == c2:
                # Self-bucket direct P2P
                p_pts = positions[idx1]
                p_q = charges[idx1]
                if len(p_pts) > 1:
                    diff = p_pts[:, None, :] - p_pts[None, :, :]
                    r2 = np.sum(diff ** 2, axis=-1) + eps2
                    np.fill_diagonal(r2, 1.0)
                    mask = ~np.eye(len(p_pts), dtype=bool)
                    
                    pot_self = np.sum(p_q[None, :] * 0.5 * np.log(r2) * mask, axis=1)
                    potentials[idx1] += pot_self
                    
                    if compute_forces:
                        inv_r2 = np.where(mask, 1.0 / r2, 0.0)
                        forces_x[idx1] -= np.sum(p_q[None, :] * diff[:, :, 0] * inv_r2, axis=1)
                        forces_y[idx1] -= np.sum(p_q[None, :] * diff[:, :, 1] * inv_r2, axis=1)
            elif c2 > c1:
                # Adjacent neighbor bucket direct P2P (symmetric contribution)
                idx2 = cluster_indices_list[c2]
                if len(idx2) == 0:
                    continue
                p_pts1 = positions[idx1]
                p_q1 = charges[idx1]
                p_pts2 = positions[idx2]
                p_q2 = charges[idx2]

                diff = p_pts1[:, None, :] - p_pts2[None, :, :]
                r2 = np.sum(diff ** 2, axis=-1) + eps2
                r2_safe = np.where(r2 < 1e-28, 1.0, r2)
                
                potentials[idx1] += np.sum(p_q2[None, :] * 0.5 * np.log(r2_safe), axis=1)
                potentials[idx2] += np.sum(p_q1[:, None] * 0.5 * np.log(r2_safe), axis=0)
                
                if compute_forces:
                    inv_r2 = 1.0 / r2_safe
                    forces_x[idx1] -= np.sum(p_q2[None, :] * diff[:, :, 0] * inv_r2, axis=1)
                    forces_y[idx1] -= np.sum(p_q2[None, :] * diff[:, :, 1] * inv_r2, axis=1)
                    
                    forces_x[idx2] += np.sum(p_q1[:, None] * diff[:, :, 0] * inv_r2, axis=0)
                    forces_y[idx2] += np.sum(p_q1[:, None] * diff[:, :, 1] * inv_r2, axis=0)

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials
