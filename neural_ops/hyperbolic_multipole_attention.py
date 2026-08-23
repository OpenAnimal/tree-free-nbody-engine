"""
Hyperbolic Multipole Attention (`hyperbolic_multipole_attention.py`)
===================================================================
Linear-Time O(N) Multipole Attention in Non-Euclidean Hyperbolic Space (Poincaré Ball Model).

Optimized for:
- Hierarchical knowledge graphs and linguistic taxonomy embeddings.
- Phylogenetic trees, biological ontologies, and hierarchical memory architectures.
- Scale-free networks with continuous negative curvature ($c > 0$).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List


def mobius_add(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Möbius addition in Poincaré ball model."""
    u_norm_sq = np.sum(u ** 2, axis=-1, keepdims=True)
    v_norm_sq = np.sum(v ** 2, axis=-1, keepdims=True)
    uv_dot = np.sum(u * v, axis=-1, keepdims=True)

    num = (1.0 + 2.0 * c * uv_dot + c * v_norm_sq) * u + (1.0 - c * u_norm_sq) * v
    denom = 1.0 + 2.0 * c * uv_dot + (c ** 2) * u_norm_sq * v_norm_sq
    return num / np.maximum(denom, 1e-12)


def poincare_distance(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Computes hyperbolic geodesic distance in the Poincaré ball:
    d_c(u, v) = (2 / sqrt(c)) * artanh(sqrt(c) * || -u \oplus_c v ||)
    """
    neg_u = -u
    diff = mobius_add(neg_u, v, c=c)
    diff_norm = np.linalg.norm(diff, axis=-1)
    diff_norm_scaled = np.clip(np.sqrt(c) * diff_norm, 0.0, 1.0 - 1e-5)
    return (2.0 / np.sqrt(c)) * np.arctanh(diff_norm_scaled)


def exp_map_zero(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map from tangent space at origin to Poincaré ball."""
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    factor = np.tanh(np.sqrt(c) * v_norm) / (np.sqrt(c) * v_norm)
    return factor * v


def log_map_zero(y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map from Poincaré ball to tangent space at origin."""
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    y_norm_clipped = np.clip(np.sqrt(c) * y_norm, 0.0, 1.0 - 1e-5)
    factor = np.arctanh(y_norm_clipped) / (np.sqrt(c) * (y_norm + 1e-12))
    return factor * y


class HyperbolicMultipoleAttention:
    """
    Linear-Time Hyperbolic Multipole Attention.
    Decomposes hyperbolic attention into:
      1. Near-field exact geodesic exponential attention in local horoball clusters.
      2. Far-field tangent space Taylor multipole expansions around hyperbolic Fréchet centroids.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        spatial_dim: int = 2,
        curvature: float = 1.0,
        radial_depth: int = 4,
        angular_depth: int = 8,
        hyperbolic_sigma: float = 1.0,
        temperature: Optional[float] = None,
    ):
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.c = float(curvature)
        self.radial_depth = radial_depth
        self.angular_depth = angular_depth
        self.hyperbolic_sigma = hyperbolic_sigma
        # temperature=0.0 must survive (falsy `or` clobbers it).
        self.temperature = (1.0 / np.sqrt(embed_dim)) if temperature is None else float(temperature)

    def _hyperbolic_hash(self, point: np.ndarray) -> int:
        """
        Maps a point in Poincaré ball to a discrete polar/spherical horoball bucket.
        """
        norm = np.linalg.norm(point)
        max_norm = (1.0 / np.sqrt(self.c)) - 1e-4
        norm_ratio = np.clip(norm / max_norm, 0.0, 0.9999)

        r_bucket = int(norm_ratio * self.radial_depth)
        
        if self.spatial_dim == 2:
            angle = np.arctan2(point[1], point[0]) + np.pi # [0, 2*pi]
            ang_bucket = int((angle / (2.0 * np.pi)) * self.angular_depth) % self.angular_depth
            return (r_bucket << 16) | ang_bucket
        else:
            angle_phi = np.arctan2(point[1], point[0]) + np.pi
            angle_theta = np.arccos(np.clip(point[2] / (norm + 1e-12), -1.0, 1.0))
            phi_bucket = int((angle_phi / (2.0 * np.pi)) * self.angular_depth) % self.angular_depth
            theta_bucket = int((angle_theta / np.pi) * self.angular_depth) % self.angular_depth
            return (r_bucket << 20) | (phi_bucket << 10) | theta_bucket

    def forward(
        self,
        Q: np.ndarray,          # (N, D) Query representations
        K: np.ndarray,          # (N, D) Key representations
        V: np.ndarray,          # (N, D) Value representations
        hyper_coords: np.ndarray, # (N, spatial_dim) Points strictly inside Poincaré ball
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass of Hyperbolic Multipole Attention in O(N) linear time.
        """
        N, D = Q.shape
        max_rad = (1.0 / np.sqrt(self.c)) - 1e-4
        norms = np.linalg.norm(hyper_coords, axis=-1, keepdims=True)
        coords_clipped = np.where(norms >= max_rad, hyper_coords * (max_rad / (norms + 1e-12)), hyper_coords)

        # 1. Bucket into hyperbolic spatial hash
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            k = self._hyperbolic_hash(coords_clipped[i])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        # 2. Compute Hyperbolic Fréchet Centroids & Monopole Moments (P2M)
        all_centroids = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_mean_k = np.zeros((n_clusters, D), dtype=np.float32)
        all_sum_v = np.zeros((n_clusters, D), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]

            # Approximate Fréchet centroid via Einstein midpoint in Poincaré ball
            tangent_pts = log_map_zero(pts, c=self.c)
            mean_tangent = np.mean(tangent_pts, axis=0)
            centroid = exp_map_zero(mean_tangent, c=self.c)
            all_centroids[idx] = centroid

            all_mean_k[idx] = np.mean(K[p_ids], axis=0)
            all_sum_v[idx] = np.sum(V[p_ids], axis=0)
            all_counts[idx] = len(p_ids)
            # NOTE: tangent-space dipoles were computed here but never used in
            # the far-field evaluation (the far pass uses only monopole
            # moments: centroid, mean_k, sum_v, count). Removed as dead code.

        # 3. Vectorized Evaluation (Near exact geodesic + Far hyperbolic multipole)
        out_v = np.zeros((N, D), dtype=np.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.hyperbolic_sigma ** 2))

        total_near_evals = 0
        total_far_evals = 0

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            q_src = Q[p_src_arr]
            pts_src = coords_clipped[p_src_arr]
            c_idx_src = key_to_idx[k_src]

            # In hyperbolic space, adjacent radial and angular buckets are near neighbors
            # For simplicity and O(N) guarantees, cluster itself and immediate hash bucket are near
            near_indices_set = {c_idx_src}
            near_p_list = list(p_src)

            # Near-field exact geodesic distance
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr]
            k_near = K[near_arr]
            v_near = V[near_arr]

            # Pairwise hyperbolic geodesic distances
            geo_dists = np.zeros((M_src, len(near_arr)), dtype=np.float32)
            for m_i in range(M_src):
                geo_dists[m_i] = poincare_distance(pts_src[m_i:m_i+1], pts_near, c=self.c)

            spatial_w_near = np.exp(-(geo_dists ** 2) * inv_2_sigma_sq)
            dot_near = np.matmul(q_src, k_near.T) * self.temperature
            attn_near = spatial_w_near * np.exp(np.clip(dot_near, -30.0, 30.0))

            # Mask self-pairs (target i == source i) so each token is not
            # attended to itself.  Pass-30 fix: the original code had no
            # self-masking, adding a spurious self-attention contribution
            # (weight 1.0 at geodesic distance 0) to every near-field
            # evaluation.
            id_t = p_src_arr[:, None]
            id_s = near_arr[None, :]
            self_mask = (id_t == id_s)
            attn_near = np.where(self_mask, 0.0, attn_near)

            val_near = np.matmul(attn_near, v_near)
            weight_near = np.sum(attn_near, axis=-1, keepdims=True) + 1e-9
            total_near_evals += M_src * len(near_arr)

            # Far-field Hyperbolic Centroid Expansions
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centroids = all_centroids[far_idx_arr]
                far_k_means = all_mean_k[far_idx_arr]
                far_v_sums = all_sum_v[far_idx_arr]
                far_counts = all_counts[far_idx_arr]

                # Geodesic distances to distant centroids: (M_src, N_far)
                far_geo_dists = np.zeros((M_src, len(far_indices)), dtype=np.float32)
                for m_i in range(M_src):
                    far_geo_dists[m_i] = poincare_distance(pts_src[m_i:m_i+1], far_centroids, c=self.c)

                spatial_w_far = np.exp(-(far_geo_dists ** 2) * inv_2_sigma_sq)
                dot_far = np.matmul(q_src, far_k_means.T) * self.temperature
                w_far = spatial_w_far * np.exp(np.clip(dot_far, -30.0, 30.0)) # (M_src, N_far)

                val_far = np.matmul(w_far, far_v_sums)
                weight_far = np.matmul(w_far, far_counts[:, None])

                val_total = val_near + val_far
                weight_total = weight_near + weight_far
                total_far_evals += M_src * len(far_indices)
            else:
                val_total = val_near
                weight_total = weight_near

            out_v[p_src_arr] = val_total / weight_total

        meta = {
            "num_particles": N,
            "curvature": self.c,
            "active_clusters": n_clusters,
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
        }
        return out_v, meta
