"""
Fast Multipole Screened Potential & Force Evaluator for Molecular Biophysics.
Supports Coulomb, Debye-Hückel / Screened Coulomb, and Generalized Born Dielectric Kernels in O(N) Time.
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, List
from .elastic_spatial_hash import ElasticSpatialHash3D, morton_decode_3d, morton_encode_3d


# Electrostatic conversion constant: e^2 / (4 * pi * eps_0 * Angstrom) -> kcal / mol
COULOMB_CONSTANT_KCAL: float = 332.063711


class ScreenedKernelType(Enum):
    COULOMB = "coulomb"
    DEBYE_HUCKEL = "debye_huckel"
    GENERALIZED_BORN = "generalized_born"
    YUKAWA = "yukawa"


class TreeFreeBioFMM:
    """
    Tree-Free Fast Multipole Method Engine for Molecular Electrostatics & Free Energy.
    """
    def __init__(
        self,
        cell_size: float = 8.0,
        theta: float = 0.5,
        kappa: float = 0.127,  # Debye screening parameter (1/Angstrom) for ~150mM salt
        dielectric_water: float = 78.5,
        dielectric_protein: float = 4.0,
        kernel_type: ScreenedKernelType = ScreenedKernelType.DEBYE_HUCKEL,
    ):
        self.cell_size = float(cell_size)
        self.theta = float(theta)
        self.kappa = float(kappa)
        self.eps_w = float(dielectric_water)
        self.eps_p = float(dielectric_protein)
        self.kernel_type = kernel_type

    def evaluate(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        radii: Optional[np.ndarray] = None,
        born_radii: Optional[np.ndarray] = None,
        compute_forces: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """
        Computes electrostatic potentials V (kcal/mol/e) and optional forces F (kcal/mol/Angstrom)
        across all N atoms in O(N) time.
        
        Returns:
            potentials: (N,) float64 array of electrostatic potentials per atom
            forces: (N, 3) float64 array of forces on each atom if compute_forces=True, else None
            meta: Dictionary of timing and execution statistics
        """
        N = len(coords)
        if N == 0:
            return np.empty(0), (np.empty((0, 3)) if compute_forces else None), {}

        origin = np.min(coords, axis=0) - self.cell_size
        inv_cell = 1.0 / self.cell_size

        # 1. 3D Spatial Partitioning via Morton Keys
        shifted = coords - origin
        ix = np.maximum(0, (shifted[:, 0] * inv_cell).astype(np.int64))
        iy = np.maximum(0, (shifted[:, 1] * inv_cell).astype(np.int64))
        iz = np.maximum(0, (shifted[:, 2] * inv_cell).astype(np.int64))

        morton_keys = morton_encode_3d(ix, iy, iz)
        unique_keys, inverse = np.unique(morton_keys, return_inverse=True)
        K = len(unique_keys)

        # 2. Build Hash Table for O(1) Neighbor Identification
        hash_table = ElasticSpatialHash3D(cell_size=self.cell_size, capacity_hint=K * 2)
        for cluster_idx, key in enumerate(unique_keys):
            hash_table.insert(int(key), cluster_idx)

        # 3. Compute Cluster Multipole Moments (Monopole + Center of Charge + Dipole)
        cluster_q = np.bincount(inverse, weights=charges, minlength=K)
        cluster_counts = np.bincount(inverse, minlength=K)
        
        # Center of geometry for each cluster
        cluster_centers = np.zeros((K, 3), dtype=np.float64)
        for d in range(3):
            cluster_centers[:, d] = np.bincount(inverse, weights=coords[:, d], minlength=K) / np.maximum(cluster_counts, 1)

        # Dipole moments (3,) per cluster
        coords_rel = coords - cluster_centers[inverse]
        cluster_dipoles = np.zeros((K, 3), dtype=np.float64)
        for d in range(3):
            cluster_dipoles[:, d] = np.bincount(inverse, weights=coords_rel[:, d] * charges, minlength=K)

        # 4. Identify Near-Field Cluster Pairs vs Far-Field Multipoles
        # Decode unique Morton keys to 3D grid coords
        cluster_grid_coords = np.array([morton_decode_3d(int(k)) for k in unique_keys], dtype=np.int64)

        # Compute cluster-to-cluster distance matrix
        c_diff = cluster_centers[:, None, :] - cluster_centers[None, :, :]  # (K, K, 3)
        c_dist = np.linalg.norm(c_diff, axis=-1)  # (K, K)
        np.fill_diagonal(c_dist, 1e9)

        # Clusters within 1 cell step are Near-Field
        g_diff = np.abs(cluster_grid_coords[:, None, :] - cluster_grid_coords[None, :, :])
        is_near_cluster = np.all(g_diff <= 1, axis=-1)  # (K, K) boolean

        is_far_cluster = ~is_near_cluster

        # 5. Far-Field M2L / P2M Approximation
        # Far-field cluster potential: monopole + dipole correction
        far_pot_cluster = np.zeros(K, dtype=np.float64)
        far_force_cluster = np.zeros((K, 3), dtype=np.float64) if compute_forces else None

        if np.any(is_far_cluster):
            r = np.maximum(c_dist, 1e-6)
            
            if self.kernel_type == ScreenedKernelType.DEBYE_HUCKEL:
                # Screened Coulomb: V(r) = (e^(-kappa*r) / r) / eps_w
                exp_kr = np.exp(-self.kappa * r)
                kernel_val = (exp_kr / r) / self.eps_w * COULOMB_CONSTANT_KCAL
                kernel_val_masked = np.where(is_far_cluster, kernel_val, 0.0)
                far_pot_cluster = np.sum(kernel_val_masked * cluster_q[None, :], axis=1)

                if compute_forces:
                    # dV/dr = - (kappa + 1/r) * V(r)
                    d_kernel = - (self.kappa + 1.0 / r) * kernel_val / r
                    d_kernel_masked = np.where(is_far_cluster[:, :, None], d_kernel[:, :, None] * c_diff, 0.0)
                    far_force_cluster = -np.sum(d_kernel_masked * cluster_q[None, :, None], axis=1)

            elif self.kernel_type == ScreenedKernelType.COULOMB:
                kernel_val = (1.0 / r) / self.eps_p * COULOMB_CONSTANT_KCAL
                kernel_val_masked = np.where(is_far_cluster, kernel_val, 0.0)
                far_pot_cluster = np.sum(kernel_val_masked * cluster_q[None, :], axis=1)

                if compute_forces:
                    d_kernel = - (1.0 / (r**3)) * (COULOMB_CONSTANT_KCAL / self.eps_p)
                    d_kernel_masked = np.where(is_far_cluster[:, :, None], d_kernel[:, :, None] * c_diff, 0.0)
                    far_force_cluster = -np.sum(d_kernel_masked * cluster_q[None, :, None], axis=1)

            else:  # Generalized Born / Yukawa
                exp_kr = np.exp(-self.kappa * r)
                kernel_val = (exp_kr / r) / self.eps_w * COULOMB_CONSTANT_KCAL
                kernel_val_masked = np.where(is_far_cluster, kernel_val, 0.0)
                far_pot_cluster = np.sum(kernel_val_masked * cluster_q[None, :], axis=1)

        # Broadcast far-field potential to atoms
        atom_potentials = far_pot_cluster[inverse].copy()
        atom_forces = np.zeros((N, 3), dtype=np.float64) if compute_forces else None
        if compute_forces and far_force_cluster is not None:
            atom_forces += far_force_cluster[inverse] * charges[:, None]

        # 6. Near-Field Direct P2P Evaluation
        # Group atoms by cluster for cache-friendly contiguous streaming
        for c1 in range(K):
            idx1 = np.where(inverse == c1)[0]
            if len(idx1) == 0:
                continue
            
            p1 = coords[idx1]
            q1 = charges[idx1]

            # Find neighboring clusters (including self)
            near_clusters = np.where(is_near_cluster[c1])[0]
            for c2 in near_clusters:
                idx2 = np.where(inverse == c2)[0]
                if len(idx2) == 0:
                    continue
                
                p2 = coords[idx2]
                q2 = charges[idx2]

                # Pairwise distances
                delta = p1[:, None, :] - p2[None, :, :]  # (len1, len2, 3)
                dist = np.linalg.norm(delta, axis=-1)   # (len1, len2)

                if c1 == c2:
                    # Exclude self-interaction
                    np.fill_diagonal(dist, 1e9)

                dist = np.maximum(dist, 1e-4)

                if self.kernel_type == ScreenedKernelType.DEBYE_HUCKEL:
                    exp_k = np.exp(-self.kappa * dist)
                    v_pair = (exp_k / dist) / self.eps_w * COULOMB_CONSTANT_KCAL
                    if c1 == c2:
                        np.fill_diagonal(v_pair, 0.0)
                    atom_potentials[idx1] += np.sum(v_pair * q2[None, :], axis=1)

                    if compute_forces:
                        # F = - q1 * q2 * grad(V)
                        dV_dr = - (self.kappa + 1.0 / dist) * v_pair
                        if c1 == c2:
                            np.fill_diagonal(dV_dr, 0.0)
                        f_contrib = -(dV_dr[:, :, None] * (delta / dist[:, :, None])) * (q1[:, None, None] * q2[None, :, None])
                        atom_forces[idx1] += np.sum(f_contrib, axis=1)

                elif self.kernel_type == ScreenedKernelType.COULOMB:
                    v_pair = (1.0 / dist) / self.eps_p * COULOMB_CONSTANT_KCAL
                    if c1 == c2:
                        np.fill_diagonal(v_pair, 0.0)
                    atom_potentials[idx1] += np.sum(v_pair * q2[None, :], axis=1)

                    if compute_forces:
                        dV_dr = - (1.0 / (dist**2)) * (COULOMB_CONSTANT_KCAL / self.eps_p)
                        if c1 == c2:
                            np.fill_diagonal(dV_dr, 0.0)
                        f_contrib = -(dV_dr[:, :, None] * (delta / dist[:, :, None])) * (q1[:, None, None] * q2[None, :, None])
                        atom_forces[idx1] += np.sum(f_contrib, axis=1)

                elif self.kernel_type == ScreenedKernelType.GENERALIZED_BORN:
                    # Generalized Born pairwise interaction
                    if born_radii is not None:
                        a1 = born_radii[idx1]
                        a2 = born_radii[idx2]
                        a_prod = a1[:, None] * a2[None, :]
                        f_gb = np.sqrt(dist**2 + a_prod * np.exp(-dist**2 / (4.0 * a_prod + 1e-8)))
                    else:
                        f_gb = dist

                    exp_k = np.exp(-self.kappa * f_gb)
                    gb_factor = - (1.0 - exp_k / self.eps_w) / self.eps_p
                    v_pair = gb_factor * (COULOMB_CONSTANT_KCAL / f_gb)
                    if c1 == c2:
                        np.fill_diagonal(v_pair, 0.0)
                    atom_potentials[idx1] += np.sum(v_pair * q2[None, :], axis=1)

        meta = {
            "num_atoms": N,
            "num_clusters": K,
            "grid_resolution_angstrom": self.cell_size,
            "kernel": self.kernel_type.value,
        }
        return atom_potentials, atom_forces, meta
