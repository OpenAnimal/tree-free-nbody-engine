"""
Vectorized, High-Throughput Matrix Kernel Engine for Tree-Free FMM
Replaces Python for-loops with dense block SIMD matrix multiplications & JAX vectorization.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
try:
    from .elastic_hash import ElasticHashTable
    from .tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d
except ImportError:
    from elastic_hash import ElasticHashTable
    from tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d

try:
    from quantized_bitpacked_optimization.greedy_multipole_mesh import GreedyMultipoleAggregator2D
    from quantized_bitpacked_optimization.packed_vectorized_fmm import VoxelPackedTreeFreeFMM
except ImportError:
    try:
        from quantized_bitpacked.greedy_multipole_mesh import GreedyMultipoleAggregator2D
        from quantized_bitpacked.packed_vectorized_fmm import VoxelPackedTreeFreeFMM
    except ImportError:
        GreedyMultipoleAggregator2D = None
        VoxelPackedTreeFreeFMM = None

class FastVectorizedFMM:
    """
    Vectorized FMM Engine with Farach-Colton Non-Reordering Hash.
    Executes cluster-cluster M2L interactions as a single vectorized matrix broadcast.
    
    Supports optional high-throughput backend acceleration:
    - backend="exact_float" (default): Full IEEE-754 precision for scientific and physical simulation.
    - backend="bitpacked": 32/64-bit quantized fixed-point engine with bitboard skipping and cache-line packing.
    - enable_greedy_aggregation: Enables O(K) run-length multipole cluster merging to condense M2L transfer matrix.
    """
    def __init__(
        self,
        depth: int = 4,
        order: int = 4,
        backend: str = "exact_float",
        enable_greedy_aggregation: bool = False,
        enable_bitboard_skip: bool = False,
        enable_direct_strides: bool = False
    ):
        self.depth = depth
        self.order = order
        self.backend = backend
        self.enable_greedy_aggregation = enable_greedy_aggregation
        self.grid_res = 1 << depth
        self.hash_table = ElasticHashTable(capacity=self.grid_res * self.grid_res * 2, delta=0.05)
        
        if self.backend in ("bitpacked", "quantized", "voxel_packed"):
            if VoxelPackedTreeFreeFMM is None:
                raise ImportError("quantized_bitpacked module is required for bitpacked backend.")
            self._packed_engine = VoxelPackedTreeFreeFMM(
                depth=depth,
                order=order,
                enable_packing=True,
                enable_greedy_aggregation=enable_greedy_aggregation,
                enable_bitboard_skip=enable_bitboard_skip,
                enable_direct_strides=enable_direct_strides
            )
        else:
            self._packed_engine = None
            if enable_greedy_aggregation and GreedyMultipoleAggregator2D is not None:
                self.greedy_aggregator = GreedyMultipoleAggregator2D(order=order)
            else:
                self.greedy_aggregator = None

    def evaluate(self, positions: np.ndarray, charges: np.ndarray, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        if self._packed_engine is not None:
            potentials, metrics = self._packed_engine.evaluate(positions, charges)
            return (potentials, metrics) if return_metrics else potentials
        N = len(positions)
        grid_res = self.grid_res
        
        # 1. Morton Quantization & Bucket Binning via Non-Reordering Hash
        # Compute integer coords
        ix = np.clip((positions[:, 0] * grid_res).astype(np.int32), 0, grid_res - 1)
        iy = np.clip((positions[:, 1] * grid_res).astype(np.int32), 0, grid_res - 1)
        morton_keys = (self.depth << 24) | (ix << 12) | iy
        
        unique_keys, inverse_indices = np.unique(morton_keys, return_inverse=True)
        num_clusters = len(unique_keys)
        
        # Extract cluster centers
        cluster_ix = (unique_keys >> 12) & 0xFFF
        cluster_iy = unique_keys & 0xFFF
        box_size = 1.0 / grid_res
        cx = (cluster_ix + 0.5) * box_size
        cy = (cluster_iy + 0.5) * box_size
        centers = cx + 1j * cy  # (num_clusters,)
        
        # 2. Vectorized P2M (Particle to Multipole Moments)
        # Calculate complex relative coords dz for each particle to its cluster center
        p_centers = centers[inverse_indices]
        z_pts = (positions[:, 0] + 1j * positions[:, 1]) - p_centers
        
        # Multipole powers: (N, order+1)
        powers = np.arange(self.order + 1)
        # z_pts[:, None] ** powers[None, :]
        z_pows = np.column_stack([z_pts ** k for k in powers])
        
        # Accumulate multipole moments per cluster via bincount
        cluster_m = np.zeros((num_clusters, self.order + 1), dtype=np.complex128)
        cluster_m[:, 0] = np.bincount(inverse_indices, weights=charges, minlength=num_clusters)
        for k in range(1, self.order + 1):
            term = -charges * (z_pts ** k) / k
            cluster_m[:, k] = (np.bincount(inverse_indices, weights=np.real(term), minlength=num_clusters) +
                               1j * np.bincount(inverse_indices, weights=np.imag(term), minlength=num_clusters))
                               
        # 3. Vectorized M2L (Far-Field Interaction Matrix between clusters)
        if self.greedy_aggregator is not None and num_clusters > 8:
            macro_centers, macro_m, cluster_to_macro, _ = self.greedy_aggregator.aggregate_runs(
                unique_keys, centers, cluster_m, depth=self.depth
            )
            macro_dx = np.real(macro_centers)[:, None] - np.real(macro_centers)[None, :]
            macro_dy = np.imag(macro_centers)[:, None] - np.imag(macro_centers)[None, :]
            # Macro-clusters are separated if no child leaves are adjacent (dx >= 2.5 leaf box_size)
            well_sep_macro = (np.abs(macro_dx) >= (2.5 * box_size)) | (np.abs(macro_dy) >= (2.5 * box_size))
            
            z0_macro = macro_centers[None, :] - macro_centers[:, None]
            z0_safe = np.where(well_sep_macro, z0_macro, 1.0)
            
            m2l_kernel = np.where(well_sep_macro, macro_m[None, :, 0] * np.log(-z0_safe + 1e-12), 0.0)
            m2l_kernel_l1 = np.where(well_sep_macro, macro_m[None, :, 0] / (-z0_safe), 0.0)
            for k in range(1, self.order + 1):
                m2l_kernel += np.where(well_sep_macro, macro_m[None, :, k] / ((-z0_safe) ** k), 0.0)
                m2l_kernel_l1 -= np.where(well_sep_macro, (k * macro_m[None, :, k]) / ((-z0_safe) ** (k + 1)), 0.0)
                
            macro_l0 = np.sum(m2l_kernel, axis=1)
            macro_l1 = np.sum(m2l_kernel_l1, axis=1)
            cluster_l0 = macro_l0[cluster_to_macro]
            cluster_l1 = macro_l1[cluster_to_macro]
            dz_eval = (positions[:, 0] + 1j * positions[:, 1]) - macro_centers[cluster_to_macro[inverse_indices]]
            potentials = np.real(cluster_l0[inverse_indices] + cluster_l1[inverse_indices] * dz_eval)
        else:
            # Cluster-to-cluster distance matrix
            dx = cluster_ix[:, None] - cluster_ix[None, :]
            dy = cluster_iy[:, None] - cluster_iy[None, :]
            well_separated = (np.abs(dx) > 1) | (np.abs(dy) > 1)  # (num_clusters, num_clusters)
            
            # Center difference: z0 = src_center - tgt_center
            # tgt is row, src is col -> centers[None, :] - centers[:, None]
            z0 = centers[None, :] - centers[:, None]
            # Avoid log(0) on near field
            z0_safe = np.where(well_separated, z0, 1.0)
            
            # Compute M2L transfer matrix (num_clusters, num_clusters)
            # Primary monopole transfer: a0 * log(-z0)
            m2l_kernel = np.where(well_separated, cluster_m[None, :, 0] * np.log(-z0_safe + 1e-12), 0.0)
            m2l_kernel_l1 = np.where(well_separated, cluster_m[None, :, 0] / (-z0_safe), 0.0)
            for k in range(1, self.order + 1):
                m2l_kernel += np.where(well_separated, cluster_m[None, :, k] / ((-z0_safe) ** k), 0.0)
                m2l_kernel_l1 -= np.where(well_separated, (k * cluster_m[None, :, k]) / ((-z0_safe) ** (k + 1)), 0.0)
                
            cluster_l0 = np.sum(m2l_kernel, axis=1)  # (num_clusters,)
            cluster_l1 = np.sum(m2l_kernel_l1, axis=1)  # (num_clusters,)
            potentials = np.real(cluster_l0[inverse_indices] + cluster_l1[inverse_indices] * z_pts)
        
        # 4. Fast Local Direct Near-Field P2P (Evaluate self and adjacent 3x3 neighbor buckets)
        cluster_indices_list = [np.where(inverse_indices == c)[0] for c in range(num_clusters)]
        dx = cluster_ix[:, None] - cluster_ix[None, :]
        dy = cluster_iy[:, None] - cluster_iy[None, :]
        near_cluster_pairs = np.argwhere((np.abs(dx) <= 1) & (np.abs(dy) <= 1))

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
                    r = np.linalg.norm(diff, axis=-1) + 1e-12
                    np.fill_diagonal(r, 1.0)
                    pot_self = np.sum(p_q[None, :] * np.log(r) * (1.0 - np.eye(len(p_pts))), axis=1)
                    potentials[idx1] += pot_self
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
                r = np.linalg.norm(diff, axis=-1) + 1e-12
                potentials[idx1] += np.sum(p_q2[None, :] * np.log(r), axis=1)
                potentials[idx2] += np.sum(p_q1[:, None] * np.log(r), axis=0)

        return potentials
