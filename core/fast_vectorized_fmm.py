"""
Vectorized, High-Throughput Matrix Kernel Engine for Tree-Free FMM
Replaces Python for-loops with dense block SIMD matrix multiplications & vectorized adaptive FMM kernels.

Implements:
- Vectorized P2M (Particle to Multipole)
- Vectorized M2L (Multipole to Local) with full adaptive FMM binomial expansion tensor
- Vectorized L2P (Local to Particle potential & force)
- Block-tiled symmetric P2P near-field summation
"""

import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Union
try:
    from .adaptive_fmm import AdaptiveFMM, GreengardRokhlin87RegularFMM, exact_direct_nbody_2d, exact_direct_nbody_forces_2d
    from .tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d
    from .elastic_hash import ElasticHashTable
    from ._csr import build_csr
except ImportError:
    from adaptive_fmm import AdaptiveFMM, GreengardRokhlin87RegularFMM, exact_direct_nbody_2d, exact_direct_nbody_forces_2d
    from tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d
    from elastic_hash import ElasticHashTable
    from _csr import build_csr


class FastVectorizedFMM:
    """
    Flat (single-level) tree-free 2D FMM with adaptive FMM expansions.

    Cell index of record: an ElasticHashTable (Farach-Colton, Krapivin, &
    Kuszmaul, 2025, elastic hashing, core.elastic_hash) maps each occupied
    Morton cell key to its dense cluster index.  Occupancy and 3x3
    adjacency are resolved through hash lookups -- no sorted arrays, no
    dicts, no pointers.  Hot numeric kernels remain vectorized NumPy.

    Complexity, stated honestly: P2M/L2P are O(N p); M2L evaluates all
    well-separated OCCUPIED cell pairs, which for a single-level scheme
    is O(K^2 p^2) with K = number of occupied cells (K <= 4^depth, a
    constant for fixed depth, so the scheme is linear in N with a
    depth-dependent constant); near-field P2P is O(N * neighbors).
    """
    def __init__(
        self,
        depth: int = 4,
        order: int = 6,
        softening: float = 0.0
    ):
        if not 1 <= depth <= 12:
            # cell keys pack (depth<<24)|(ix<<12)|iy with ix,iy < 2^depth in
            # 12 bits each; depth > 12 silently aliases distinct cells.
            raise ValueError(
                f"depth must be in [1, 12] for the 12-bit cell-key layout, got {depth}")
        self.depth = depth
        self.order = order
        self.softening = softening
        self.grid_res = 1 << depth

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
        
        # 1. Cell Quantization & Bucket Binning
        # Round-7 task T-E4 (F-03): renamed morton_keys -> cell_keys.
        # The key is a level-tagged row-major-style pair (ix in bits 12-23,
        # iy in bits 0-11) -- NOT Morton interleaving. NOTE: this is
        # TRANSPOSED from spatial_index.morton_2d_key (which puts iy in the
        # high bits); the layout is self-consistent within this module
        # (encode, decode, and the 3x3 neighbor probes below all agree),
        # but do not mix keys between the two modules.
        ix = np.clip((positions[:, 0] * grid_res).astype(np.int32), 0, grid_res - 1)
        iy = np.clip((positions[:, 1] * grid_res).astype(np.int32), 0, grid_res - 1)
        cell_keys = (self.depth << 24) | (ix << 12) | iy

        unique_keys, inverse_indices = np.unique(cell_keys, return_inverse=True)
        num_clusters = len(unique_keys)

        # Elastic hash = authoritative occupied-cell index (key -> cluster id)
        self.hash_table = ElasticHashTable(
            capacity=max(16, 2 * num_clusters), delta=0.05
        )
        for c, key in enumerate(unique_keys):
            ok, _ = self.hash_table.insert(int(key), c)
            if not ok:
                raise RuntimeError(f"elastic hash insert failed for cell key {key}")

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
                               
        # 3. Vectorized M2L (adaptive FMM Multipole to Local Translation Matrix)
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
        # Round-7 task T-E4: replaced O(N*K) list comprehension with CSR
        # (argsort + searchsorted, same pattern as T-C3's solvation engine).
        cell_start, cell_particles, _ = build_csr(inverse_indices, num_clusters)
        # Near-field pair list resolved through funnel-hash neighbor lookups:
        # for each occupied cell, probe the 3x3 neighborhood (key
        # = depth<<24 | ix<<12 | iy per the layout note above -- NOT Morton
        # interleaving) in the elastic hash; only cells the hash reports as
        # occupied take part.
        near_pairs = []
        key_depth = self.depth << 24
        for c in range(num_clusters):
            kx, ky = int(cluster_ix[c]), int(cluster_iy[c])
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nx_, ny_ = kx + ox, ky + oy
                    if 0 <= nx_ < grid_res and 0 <= ny_ < grid_res:
                        v, _ = self.hash_table.lookup(key_depth | (nx_ << 12) | ny_)
                        if v is not None:
                            near_pairs.append((c, v))
        near_cluster_pairs = np.array(near_pairs, dtype=np.int64).reshape(-1, 2)
        eps2 = self.softening * self.softening

        for c1, c2 in near_cluster_pairs:
            idx1 = cell_particles[cell_start[c1]:cell_start[c1 + 1]]
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
                idx2 = cell_particles[cell_start[c2]:cell_start[c2 + 1]]
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
