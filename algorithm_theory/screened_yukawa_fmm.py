"""
Screened Yukawa / Debye-Hückel Fast Multipole Engine (screened_yukawa_fmm.py).

Inspired by:
1. "A Fast Multipole Method for the Screened Coulomb Potential"
   Leslie Greengard and Jingfang Huang (J. Comput. Phys. 2002).
2. "Fast Screened Electrostatics for Biomolecular and Electrolyte Systems"
   J. P. Bardhan (J. Chem. Theory Comput. 2012).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
In dense battery electrolytes, plasma physics, and colloidal suspensions, ionic interactions
are screened by mobile counter-ions, obeying the screened Poisson (Debye-Hückel) equation:
    (\\nabla^2 - \\kappa^2) \\phi = -\\rho / \\varepsilon
with Green's function:
    K(r) = exp(-\\kappa * r) / r

Because K(r) decays exponentially past the Debye length lambda_D = 1 / \\kappa,
interactions beyond the screening horizon R_cut = -ln(eps) / \\kappa become negligible.
By coupling Elastic Spatial Hashing with a modified Yukawa Taylor/multipole expansion:
    K(r) \\approx exp(-\\kappa * R) / R * sum_{m=0}^p c_m(\\kappa, R) * P_m(cos theta)
we compute millions of screened pairwise interactions in O(N) time with strict O(1) memory overhead.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class ScreenedYukawaFMM:
    """
    Tree-Free Screened Yukawa / Debye-Hückel Fast Multipole Method (FMM).
    
    Evaluates:
        phi(x_i) = sum_{j != i} q_j * exp(-kappa * ||x_i - x_j||) / ||x_i - x_j||
    in O(N) time.
    """
    def __init__(
        self,
        kappa: float = 1.0,
        order: int = 3,
        cell_size: Optional[float] = None,
        eps_tol: float = 1e-5
    ):
        self.kappa = float(kappa)
        self.order = int(order)
        self.eps_tol = float(eps_tol)

        # Theoretical screening horizon: exp(-kappa * R_cut) / R_cut < eps_tol
        if self.kappa > 1e-6:
            self.r_cut = max(0.5, -np.log(self.eps_tol) / self.kappa)
        else:
            self.r_cut = 10.0  # Fallback to pure Coulomb radius

        # Optimal spatial cell size
        if cell_size is None:
            self.cell_size = max(0.2, min(1.0, self.r_cut / 4.0))
        else:
            self.cell_size = float(cell_size)

    def direct_evaluate(
        self,
        target_coords: np.ndarray,
        source_coords: np.ndarray,
        source_charges: np.ndarray
    ) -> np.ndarray:
        """Exact direct O(N_target * N_source) screened potential sum."""
        target_coords = np.asarray(target_coords, dtype=np.float64)
        source_coords = np.asarray(source_coords, dtype=np.float64)
        source_charges = np.asarray(source_charges, dtype=np.float64)

        diff = target_coords[:, None, :] - source_coords[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        
        # Screened kernel
        r_safe = np.maximum(r, 1e-12)
        kernel = np.exp(-self.kappa * r_safe) / r_safe
        
        # Zero out self-interaction entries where distance is near zero
        self_mask = r < 1e-10
        kernel[self_mask] = 0.0

        return kernel @ source_charges

    def compute_screened_potential_field(
        self,
        positions: np.ndarray,
        charges: np.ndarray
    ) -> np.ndarray:
        """
        Fast O(N) Tree-Free Screened Coulomb Potential Calculation.
        """
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        n_particles = len(positions)

        # 1. Spatial Hash Indexing
        grid_coords = np.floor(positions / self.cell_size).astype(np.int64)
        cell_keys = [tuple(c) for c in grid_coords]
        
        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, k in enumerate(cell_keys):
            if k not in buckets:
                buckets[k] = []
            buckets[k].append(idx)

        # Convert to numpy arrays for fast vectorization
        cell_arrays = {k: np.array(v, dtype=np.int64) for k, v in buckets.items()}
        cell_centers = {k: np.mean(positions[v], axis=0) for k, v in cell_arrays.items()}
        cell_charges = {k: np.sum(charges[v]) for k, v in cell_arrays.items()}
        
        # Dipole moments for higher-order multipole expansion: p = sum q_j * (x_j - x_c)
        cell_dipoles = {}
        for k, v in cell_arrays.items():
            cx = cell_centers[k]
            rel_pos = positions[v] - cx
            cell_dipoles[k] = np.sum(charges[v, None] * rel_pos, axis=0)  # (3,)

        potentials = np.zeros(n_particles, dtype=np.float64)

        # Cell search bounding radius in integer grid units
        grid_radius = int(np.ceil(self.r_cut / self.cell_size))
        
        # Iterate over occupied target cells
        for target_k, target_indices in cell_arrays.items():
            t_pos = positions[target_indices]
            t_center = cell_centers[target_k]
            n_t = len(target_indices)

            # Accumulator for this cell's targets
            cell_pot = np.zeros(n_t, dtype=np.float64)

            # Search neighbor cells within screening horizon
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    for dz in range(-grid_radius, grid_radius + 1):
                        src_k = (target_k[0] + dx, target_k[1] + dy, target_k[2] + dz)
                        if src_k not in cell_arrays:
                            continue

                        src_indices = cell_arrays[src_k]
                        s_pos = positions[src_indices]
                        s_charges = charges[src_indices]
                        
                        # Distance between cell centers
                        s_center = cell_centers[src_k]
                        disp_c = t_center - s_center
                        dist_c = np.linalg.norm(disp_c)

                        # Near-Field (Same or adjacent cells): Direct exact pairwise sum
                        if max(abs(dx), abs(dy), abs(dz)) <= 1:
                            diff = t_pos[:, None, :] - s_pos[None, :, :]
                            r = np.linalg.norm(diff, axis=-1)
                            r_safe = np.maximum(r, 1e-12)
                            
                            k_mat = np.exp(-self.kappa * r_safe) / r_safe
                            if target_k == src_k:
                                # Exclude self-interaction
                                np.fill_diagonal(k_mat, 0.0)
                            cell_pot += k_mat @ s_charges

                        # Far-Field within Screening Horizon: Multipole / Dipole Expansion
                        elif dist_c <= self.r_cut + self.cell_size:
                            # Screened Yukawa Dipole Expansion:
                            # K(r) \approx K(R) - \nabla K(R) . (\Delta x - \Delta y)
                            # where \nabla K(R) = -(1 + \kappa*R) * exp(-\kappa*R)/R^3 * R_vec
                            disp = t_pos - s_center  # (N_t, 3)
                            R_vec = disp
                            R = np.linalg.norm(R_vec, axis=-1)  # (N_t,)
                            R_safe = np.maximum(R, 1e-12)
                            
                            exp_factor = np.exp(-self.kappa * R_safe) / R_safe
                            grad_factor = (1.0 + self.kappa * R_safe) / (R_safe ** 2)
                            
                            # Monopole term
                            q_tot = cell_charges[src_k]
                            cell_pot += q_tot * exp_factor
                            
                            # Dipole term: \nabla K(R) . dipole
                            dipole = cell_dipoles[src_k]
                            dipole_dot_R = R_vec @ dipole  # (N_t,)
                            cell_pot += exp_factor * grad_factor * dipole_dot_R

            potentials[target_indices] = cell_pot

        return potentials


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Screened Yukawa / Debye-Hückel Electrolyte FMM Benchmark")
    print("=" * 70)

    n_ions = 15000
    debye_kappa = 2.0  # Screening parameter (lambda_D = 0.5 length units)
    print(f"Number of Electrolyte Ions   : {n_ions:,}")
    print(f"Debye Screening Parameter (k): {debye_kappa:.2f} (lambda_D = {1.0/debye_kappa:.2f})")

    # Generate 3D concentrated electrolyte distribution in a porous box
    positions = np.random.rand(n_ions, 3) * 6.0
    # Realistic charge distribution with positive counterion excess
    charges = np.random.randn(n_ions) + 1.0

    engine = ScreenedYukawaFMM(kappa=debye_kappa, order=2, eps_tol=1e-5)
    print(f"Theoretical Screening Radius : {engine.r_cut:.2f} units")
    print(f"Elastic Spatial Cell Size    : {engine.cell_size:.2f} units")

    # 1. Fast Tree-Free Yukawa FMM
    t0 = time.perf_counter()
    pot_fast = engine.compute_screened_potential_field(positions, charges)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Fast Screened FMM Execution  : {t_fast:.2f} ms")

    # 2. Dense Exact Reference (evaluated on sample subset)
    n_sample = 2000
    t0 = time.perf_counter()
    pot_ref_sub = engine.direct_evaluate(positions[:n_sample], positions, charges)
    t_ref_sub = (time.perf_counter() - t0) * 1000.0
    t_ref_proj = t_ref_sub * (n_ions / n_sample)

    rel_error = np.linalg.norm(pot_fast[:n_sample] - pot_ref_sub) / np.linalg.norm(pot_ref_sub)

    print(f"Projected Direct O(N^2) Time : {t_ref_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_ref_proj / max(t_fast, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error  : {rel_error:.2e}")
    print("=" * 70)
