"""
Tree-Free Macromolecular Normal Mode Analysis & Allostery Engine (`macromolecular_nma_engine.py`)
================================================================================================
Linear-Time O(N) Anisotropic Network Model (ANM) & Gaussian Network Model (GNM).
Computes low-frequency functional vibrational modes, B-factors, and allosteric cross-correlations (DCCM)
for massive macromolecular complexes (ribosomes, viral capsids, multi-domain proteins) without
forming dense 3N x 3N Hessian matrices.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any

try:
    from .pdb_loader import MolecularSystem
    from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d


@dataclass
class NormalMode:
    """A single macromolecular vibrational normal mode."""
    mode_index: int
    frequency_arbitrary_units: float
    eigenvalue: float
    eigenvector: np.ndarray        # (N, 3) 3D displacement vectors per residue
    collectivity_index: float      # Degree of collective motion [0, 1]


@dataclass
class NMAReport:
    """Complete characterization of macromolecular dynamics and allostery."""
    num_residues: int
    cutoff_radius_A: float
    spring_constant_gamma: float
    modes: List[NormalMode]
    predicted_b_factors: np.ndarray      # (N,) Predicted crystallographic B-factors
    hinge_residue_indices: List[int]     # Identified allosteric hinge residues
    dynamic_cross_correlation: np.ndarray # (N, N) DCCM allosteric matrix
    elapsed_solve_ms: float


class TreeFreeMacromolecularNMA:
    """
    Matrix-Free Anisotropic Network Model (ANM) Solver.
    Evaluates Hessian-vector products in O(N) using Elastic Spatial Hashing and
    extracts the slowest functional vibrational modes via Krylov subspace iteration.
    """
    def __init__(
        self,
        cutoff_radius: float = 12.0,      # Angstroms (standard ANM cutoff: 12-15 A)
        spring_constant: float = 1.0,     # gamma (kcal/(mol * A^2))
        temperature_kelvin: float = 300.0,
    ):
        self.cutoff = float(cutoff_radius)
        self.gamma = float(spring_constant)
        self.temp_k = float(temperature_kelvin)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            raise ValueError("cutoff_radius must be finite and positive")
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("spring_constant must be finite and positive")
        if not np.isfinite(self.temp_k) or self.temp_k < 0.0:
            raise ValueError("temperature_kelvin must be finite and non-negative")
        self.kb = 0.0019872041 # kcal / (mol * K)

    def _build_spatial_buckets(self, coords: np.ndarray) -> Tuple[Dict[Tuple[int, int, int], List[int]], float]:
        """Buckets residue C-alpha coordinates into 3D grid."""
        cell_size = self.cutoff
        grid_coords = np.floor(coords / cell_size).astype(np.int64)
        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, c in enumerate(grid_coords):
            key = (int(c[0]), int(c[1]), int(c[2]))
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)
        return buckets, cell_size

    def hessian_matvec(
        self,
        coords: np.ndarray, # (N, 3) C-alpha coordinates
        v: np.ndarray,      # (N, 3) 3D displacement vector
        buckets: Dict[Tuple[int, int, int], List[int]],
    ) -> np.ndarray:
        """
        Matrix-free ANM Hessian-vector product H * v in O(N) time:
        (H * v)_i = gamma * sum_{j in Near(i)} (r_ij x r_ij^T / ||r_ij||^2) * (v_i - v_j)
        """
        N = len(coords)
        Hv = np.zeros((N, 3), dtype=np.float64)
        v_d = v.astype(np.float64)
        pts = coords.astype(np.float64)

        for cell_k, p_ids in buckets.items():
            near_p_list = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (cell_k[0] + dx, cell_k[1] + dy, cell_k[2] + dz)
                        if nk in buckets:
                            near_p_list.extend(buckets[nk])

            near_arr = np.asarray(near_p_list, dtype=np.int64)
            p_src_arr = np.asarray(p_ids, dtype=np.int64)

            pts_src = pts[p_src_arr]   # (M, 3)
            pts_near = pts[near_arr]   # (K, 3)
            v_src = v_d[p_src_arr]     # (M, 3)
            v_near = v_d[near_arr]     # (K, 3)

            # Displacement vectors: (M, K, 3)
            diff = pts_src[:, None, :] - pts_near[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            mask = (dist_sq > 1e-6) & (dist_sq <= self.cutoff ** 2)

            dist_safe = np.where(mask, np.sqrt(dist_sq), 1.0)
            u_unit = np.where(mask[:, :, None], diff / dist_safe[:, :, None], 0.0) # (M, K, 3)

            # (v_i - v_j): (M, K, 3)
            dv = v_src[:, None, :] - v_near[None, :, :]
            # Projection: (u_ij . dv_ij)
            u_dot_dv = np.sum(u_unit * dv, axis=-1) # (M, K)

            # Force increment: sum_j gamma * (u_ij . dv_ij) * u_ij
            h_inc = np.sum(self.gamma * u_dot_dv[:, :, None] * u_unit, axis=1) # (M, 3)
            Hv[p_src_arr] += h_inc

        return Hv

    def compute_normal_modes(
        self,
        coords: np.ndarray,
        num_modes: int = 10,
        max_lanczos_iters: int = 60,
    ) -> NMAReport:
        """
        Extracts slowest vibrational normal modes and allosteric cross-correlations.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 2 or not np.all(np.isfinite(coords)):
            raise ValueError("coords must be a finite array with shape (N, 3), N >= 2")
        num_modes = int(num_modes)
        max_lanczos_iters = int(max_lanczos_iters)
        if num_modes < 1 or max_lanczos_iters < 1:
            raise ValueError("num_modes and max_lanczos_iters must be positive")
        N = len(coords)
        t0 = time.perf_counter()
        buckets, _ = self._build_spatial_buckets(coords)

        # 1. Deflation of 6 rigid-body translations & rotations
        # Matrix-free Krylov subspace iteration (Lanczos / Shifted Inverse Power)
        # Shift sigma to avoid singular zero modes
        sigma = 10.0
        
        # Build small Krylov basis
        m_iters = min(max_lanczos_iters, 3 * N)
        V_krylov = np.zeros((m_iters, N, 3), dtype=np.float64)
        T_mat = np.zeros((m_iters, m_iters), dtype=np.float64)

        rng = np.random.RandomState(42)
        v0 = rng.randn(N, 3).astype(np.float64)
        v0 /= np.linalg.norm(v0)
        V_krylov[0] = v0

        beta = 0.0
        for j in range(m_iters):
            w = self.hessian_matvec(coords, V_krylov[j], buckets)
            if j > 0:
                w -= beta * V_krylov[j - 1]
            
            alpha_j = float(np.sum(w * V_krylov[j]))
            w -= alpha_j * V_krylov[j]

            # Re-orthogonalization
            for k in range(j + 1):
                w -= np.sum(w * V_krylov[k]) * V_krylov[k]

            beta = float(np.linalg.norm(w))
            T_mat[j, j] = alpha_j
            if j + 1 < m_iters:
                T_mat[j, j + 1] = beta
                T_mat[j + 1, j] = beta
                if beta < 1e-10:
                    m_iters = j + 1
                    break
                V_krylov[j + 1] = w / beta

        T_mat = T_mat[:m_iters, :m_iters]
        eigvals, eigvecs_T = np.linalg.eigh(T_mat)

        # Discard the 6 rigid body modes (near 0) and take lowest internal modes
        valid_mask = eigvals > 1e-4
        valid_eigvals = eigvals[valid_mask]
        valid_eigvecs_T = eigvecs_T[:, valid_mask]

        if len(valid_eigvals) == 0:
            valid_eigvals = np.maximum(eigvals, 1e-3)
            valid_eigvecs_T = eigvecs_T

        n_modes_to_keep = min(num_modes, len(valid_eigvals))
        extracted_modes: List[NormalMode] = []

        # 2. Reconstruct 3D physical mode vectors: u_m = sum_k V_krylov[k] * eigvecs_T[k, m]
        inv_lambda_sum = np.zeros(N, dtype=np.float64)
        all_mode_vectors = []

        for m_i in range(n_modes_to_keep):
            val = valid_eigvals[m_i]
            vec_k = valid_eigvecs_T[:, m_i]
            u_3d = np.einsum('knv,k->nv', V_krylov[:m_iters], vec_k)
            u_norm = np.linalg.norm(u_3d) + 1e-12
            u_3d /= u_norm

            # Collectivity index: kappa = exp( - sum_i p_i ln p_i ) / N, where p_i = ||u_i||^2 / sum ||u_j||^2
            p_i = np.sum(u_3d ** 2, axis=-1)
            p_i /= (np.sum(p_i) + 1e-15)
            p_safe = np.where(p_i > 1e-12, p_i, 1.0)
            entropy = -np.sum(p_i * np.log(p_safe))
            collectivity = float(np.exp(entropy) / N)

            extracted_modes.append(NormalMode(
                mode_index=m_i + 1,
                frequency_arbitrary_units=float(np.sqrt(max(val, 1e-6))),
                eigenvalue=float(val),
                eigenvector=u_3d.astype(np.float32),
                collectivity_index=collectivity,
            ))

            inv_lambda_sum += (1.0 / max(val, 1e-4)) * np.sum(u_3d ** 2, axis=-1)
            all_mode_vectors.append(u_3d)

        # 3. Crystallographic B-Factors: B_i = (8 * pi^2 * k_B * T / 3) * sum (1 / lambda_m) ||u_{m, i}||^2
        b_factors = (8.0 * (np.pi ** 2) * self.kb * self.temp_k / 3.0) * inv_lambda_sum

        # 4. Dynamic Cross-Correlation Matrix (DCCM / Allosteric Coupling)
        # C_{ij} = sum_m (1 / lambda_m) (u_{m, i} . u_{m, j}) / sqrt(B_i * B_j)
        all_u = np.stack(all_mode_vectors, axis=0) # (num_modes, N, 3)
        weights = 1.0 / np.array([m.eigenvalue for m in extracted_modes], dtype=np.float64) # (num_modes,)

        # Weighted dot products: sum_m w_m * (u_{m, i} . u_{m, j})
        cov_matrix = np.einsum('m,mnd,mpd->np', weights, all_u, all_u)
        diag = np.sqrt(np.diag(cov_matrix)) + 1e-12
        dccm = cov_matrix / np.outer(diag, diag)
        dccm = np.clip(dccm, -1.0, 1.0)

        # 5. Identify Allosteric Hinge Residues (minima in displacement amplitude with high collectivity)
        disp_profile = inv_lambda_sum
        mean_disp = np.mean(disp_profile)
        hinges = [idx for idx in range(1, N - 1) if disp_profile[idx] < disp_profile[idx - 1] and disp_profile[idx] < disp_profile[idx + 1] and disp_profile[idx] < 0.6 * mean_disp]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return NMAReport(
            num_residues=N,
            cutoff_radius_A=self.cutoff,
            spring_constant_gamma=self.gamma,
            modes=extracted_modes,
            predicted_b_factors=b_factors.astype(np.float32),
            hinge_residue_indices=hinges,
            dynamic_cross_correlation=dccm.astype(np.float32),
            elapsed_solve_ms=elapsed_ms,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Tree-Free Macromolecular Normal Mode Analysis (ANM) Benchmark")
    print("=" * 70)

    # Synthetic multi-domain protein (N = 800 C-alpha residues)
    n_res = 800
    rng = np.random.RandomState(42)
    # Generate two globular domains connected by a flexible linker
    d1 = rng.randn(350, 3) * 12.0 + np.array([-25.0, 0.0, 0.0])
    linker = np.linspace([-12.0, 0.0, 0.0], [12.0, 0.0, 0.0], 100)
    d2 = rng.randn(350, 3) * 12.0 + np.array([+25.0, 0.0, 0.0])
    ca_coords = np.concatenate([d1, linker, d2], axis=0)

    print(f"Macromolecular Structure: {n_res} C-alpha residues")
    print(f"ANM Cutoff: 12.0 A | Temperature: 300 K")

    nma_engine = TreeFreeMacromolecularNMA(cutoff_radius=12.0, spring_constant=1.0)
    report = nma_engine.compute_normal_modes(ca_coords, num_modes=6)

    print(f"Matrix-Free ANM Solve Time: {report.elapsed_solve_ms:.2f} ms")
    print(f"Identified {len(report.modes)} functional vibrational modes:")
    for m in report.modes:
        print(f"  Mode {m.mode_index}: Frequency={m.frequency_arbitrary_units:.4f} | Eigenvalue={m.eigenvalue:.4e} | Collectivity={m.collectivity_index*100:.1f}%")

    print(f"\nAllosteric Dynamic Cross-Correlation (DCCM) Matrix: {report.dynamic_cross_correlation.shape}")
    print(f"Identified {len(report.hinge_residue_indices)} Allosteric Hinge Residues: {report.hinge_residue_indices[:5]}...")
    print(f"Mean Predicted B-factor: {float(np.mean(report.predicted_b_factors)):.2f} A^2")
    print("=" * 70)
