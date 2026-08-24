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
                               
        # 3. M2L as a lattice convolution (FFT).
        # For a fixed relative offset (dx, dy) the CGR88 M2L translation is
        # a FIXED (p+1, p+1) matrix M(dx, dy) (local = multipole @ M.T), so
        # the whole far-field pass c[dst] = sum_src M(dst - src) a[src] is a
        # discrete convolution of the coefficient field on the occupied-cell
        # lattice. Each coefficient order k is scattered onto its
        # grid_res x grid_res occupancy grid; the p+1 convolutions share one
        # precomputed kernel FFT bank (zeroed over the excluded 3x3 near
        # block) and are evaluated with np.fft on a zero-padded grid of side
        # 2*grid_res (>= 2R - 1, so the circular convolution equals the
        # linear one exactly). This replaces the former O(K^2)-pairs
        # p^2-loop over full (K, K) complex matrices -- the "K^2 M2L
        # dominates" row in BENCHMARKS.md -- with ~2(p+1) grid FFTs.
        R = grid_res
        S = 2 * R  # zero-padded linear-convolution size (>= 2R - 1)
        box = 1.0 / R

        # Kernel FFT bank: Kh[l, k] = FFT2(M_lk(dx, dy) on the padded grid),
        # with the 3x3 adjacent block zeroed (those pairs are near field).
        off = np.arange(S)
        dxg = np.where(off <= R - 1, off, off - S)  # signed offsets
        DX, DY = np.meshgrid(dxg, dxg, indexing="ij")
        near = (np.abs(DX) <= 1) & (np.abs(DY) <= 1)
        delta_grid = (DX * box) + 1j * (DY * box)
        dinv = np.zeros_like(delta_grid)
        okk = ~near
        dinv[okk] = 1.0 / delta_grid[okk]
        dpow = np.empty((S, S, 2 * p + 1), dtype=np.complex128)
        dpow[:, :, 0] = 1.0
        for m in range(1, 2 * p + 1):
            dpow[:, :, m] = dpow[:, :, m - 1] * dinv
        log_grid = np.zeros((S, S), dtype=np.complex128)
        log_grid[okk] = np.log(delta_grid[okk])

        def kernel_fft(l: int, k: int) -> np.ndarray:
            M = np.zeros((S, S), dtype=np.complex128)
            if l == 0 and k == 0:
                M[okk] = log_grid[okk]
            elif l == 0:
                M[okk] = dpow[:, :, k][okk]
            elif k == 0:
                M[okk] = (((-1.0) ** (l - 1)) / l) * dpow[:, :, l][okk]
            else:
                M[okk] = (((-1.0) ** l) * math.comb(k + l - 1, l)
                          * dpow[:, :, k + l])[okk]
            return np.fft.fft2(M)

        # scatter multipoles onto the padded occupancy grid, convolve
        Ah = np.empty((p + 1, S, S), dtype=np.complex128)
        for k in range(p + 1):
            grid_a = np.zeros((S, S), dtype=np.complex128)
            grid_a[cluster_ix, cluster_iy] = cluster_m[:, k]
            Ah[k] = np.fft.fft2(grid_a)
        cluster_l = np.zeros((num_clusters, p + 1), dtype=np.complex128)
        for l in range(p + 1):
            acc = np.zeros((S, S), dtype=np.complex128)
            for k in range(p + 1):
                acc += kernel_fft(l, k) * Ah[k]
            conv = np.fft.ifft2(acc)
            cluster_l[:, l] = conv[cluster_ix, cluster_iy]
        
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

        # Per-target-cell near-field blocks: group the hash-resolved cell
        # pairs by target, concatenate each target's neighbor particles in
        # one ragged gather (vectorized), and evaluate a single
        # (nt x ns) block per cell. This replaces the per-pair Python loop
        # over ~9K cell pairs with num_clusters iterations.
        order = np.argsort(near_cluster_pairs[:, 0], kind="stable")
        pairs = near_cluster_pairs[order]
        cell_ids = np.arange(num_clusters)
        p_lo = np.searchsorted(pairs[:, 0], cell_ids, side="left")
        p_hi = np.searchsorted(pairs[:, 0], cell_ids, side="right")

        for c1 in range(num_clusters):
            idx1 = cell_particles[cell_start[c1]:cell_start[c1 + 1]]
            if len(idx1) == 0:
                continue
            nbrs = pairs[p_lo[c1]:p_hi[c1], 1]
            sizes = cell_start[nbrs + 1] - cell_start[nbrs]
            total = int(sizes.sum())
            if total == 0:
                continue
            reps = np.repeat(np.arange(len(nbrs)), sizes)
            prev = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            within = np.arange(total) - np.repeat(prev, sizes)
            s_ids = cell_particles[cell_start[nbrs][reps] + within]

            p_pts1 = positions[idx1]
            q1 = charges[idx1]
            p_src = positions[s_ids]
            q_src = charges[s_ids]
            diff = p_pts1[:, None, :] - p_src[None, :, :]
            r2 = np.sum(diff ** 2, axis=-1) + eps2
            r2_safe = np.where(r2 < 1e-28, 1.0, r2)
            g = 0.5 * np.log(r2_safe)
            self_mask = idx1[:, None] == s_ids[None, :]
            g = np.where(self_mask, 0.0, g)
            potentials[idx1] += g @ q_src
            if compute_forces:
                inv_r2 = np.where(self_mask, 0.0, 1.0 / r2_safe)
                forces_x[idx1] -= (diff[:, :, 0] * inv_r2) @ q_src
                forces_y[idx1] -= (diff[:, :, 1] * inv_r2) @ q_src

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials
