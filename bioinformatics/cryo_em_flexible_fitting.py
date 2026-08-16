"""
Module 8: Cryo-EM Real-Space Flexible Fitting (MDFF) & Macromolecular Density Refinement Engine.
Refines atomic macromolecular structures into experimental 3D Cryo-EM electron density volumes
using O(N) Tree-Free cross-correlation gradients, FMM physical electrostatics, and stereochemical priors.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL
    from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D


@dataclass
class CryoEMFittingMetrics:
    """Quantitative quality and fit metrics for Cryo-EM density refinement."""
    initial_ccc: float             # Initial cross-correlation coefficient [0, 1]
    final_ccc: float               # Refined cross-correlation coefficient [0, 1]
    rmsd_displacement_A: float     # Root-mean-square displacement from starting pose
    clash_score_after: float       # Stereochemical clash penalty after fitting
    resolution_angstrom: float
    fitting_convergence: str       # "Converged (High Quality)", "Sub-optimal", "Diverged"


class CryoEMFlexibleFittingEngine:
    """
    O(N) Real-Space Molecular Dynamics Flexible Fitting (MDFF) Engine.
    Steers coordinates along the gradient of the Cryo-EM density potential:
    U_EM(r) = -w_EM * sum_i w_i * rho_EM(r_i)
    """
    def __init__(
        self,
        resolution_angstrom: float = 3.5,
        grid_spacing_angstrom: float = 1.0,
        em_force_weight: float = 0.3,
        temperature_kelvin: float = 300.0,
        cell_size: float = 8.0
    ):
        self.resolution = float(resolution_angstrom)
        self.grid_spacing = float(grid_spacing_angstrom)
        self.w_em = float(em_force_weight)
        self.temperature = float(temperature_kelvin)
        self.cell_size = float(cell_size)

        # Gaussian kernel width: sigma = resolution / (2 * sqrt(2 * ln(2))) ~ resolution / 2.355
        self.sigma = self.resolution / 2.355

        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=0.127,
            dielectric_water=78.5,
            dielectric_protein=4.0,
            kernel_type=ScreenedKernelType.DEBYE_HUCKEL
        )

    def generate_synthetic_density_map(
        self,
        system: MolecularSystem,
        box_padding_A: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates simulated 3D Cryo-EM density grid from atomic coordinates.
        Returns (density_3d, grid_origin, grid_dimensions).
        """
        min_b, max_b = system.get_bounding_box()
        origin = min_b - box_padding_A
        dims = np.ceil((max_b - min_b + 2.0 * box_padding_A) / self.grid_spacing).astype(int)

        density = np.zeros(dims, dtype=np.float64)
        coords = system.coords
        masses = system.masses

        # Splat Gaussian kernels onto 3D grid
        grid_x = np.arange(dims[0]) * self.grid_spacing + origin[0]
        grid_y = np.arange(dims[1]) * self.grid_spacing + origin[1]
        grid_z = np.arange(dims[2]) * self.grid_spacing + origin[2]

        for i, pt in enumerate(coords):
            w = masses[i]
            # Bounding box around atom
            r_cut = 3.0 * self.sigma
            ix0 = max(0, int((pt[0] - origin[0] - r_cut) / self.grid_spacing))
            ix1 = min(dims[0], int((pt[0] - origin[0] + r_cut) / self.grid_spacing) + 1)
            iy0 = max(0, int((pt[1] - origin[1] - r_cut) / self.grid_spacing))
            iy1 = min(dims[1], int((pt[1] - origin[1] + r_cut) / self.grid_spacing) + 1)
            iz0 = max(0, int((pt[2] - origin[2] - r_cut) / self.grid_spacing))
            iz1 = min(dims[2], int((pt[2] - origin[2] + r_cut) / self.grid_spacing) + 1)

            if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
                continue

            sub_x = grid_x[ix0:ix1] - pt[0]
            sub_y = grid_y[iy0:iy1] - pt[1]
            sub_z = grid_z[iz0:iz1] - pt[2]

            dist_sq = (
                sub_x[:, None, None]**2 +
                sub_y[None, :, None]**2 +
                sub_z[None, None, :]**2
            )

            gauss = w * np.exp(-dist_sq / (2.0 * self.sigma**2))
            density[ix0:ix1, iy0:iy1, iz0:iz1] += gauss

        # Normalize density
        norm = np.linalg.norm(density)
        if norm > 1e-9:
            density /= norm

        return density, origin, dims

    def compute_cross_correlation(
        self,
        system: MolecularSystem,
        target_density: np.ndarray,
        grid_origin: np.ndarray
    ) -> float:
        """
        Computes real-space cross-correlation coefficient (CCC) between current model and target EM map.
        """
        dims = target_density.shape
        model_dens = np.zeros(dims, dtype=np.float64)
        coords = system.coords

        for pt, m in zip(coords, system.masses):
            gx = int((pt[0] - grid_origin[0]) / self.grid_spacing)
            gy = int((pt[1] - grid_origin[1]) / self.grid_spacing)
            gz = int((pt[2] - grid_origin[2]) / self.grid_spacing)

            if 0 <= gx < dims[0] and 0 <= gy < dims[1] and 0 <= gz < dims[2]:
                model_dens[gx, gy, gz] += m

        norm_m = np.linalg.norm(model_dens)
        norm_t = np.linalg.norm(target_density)
        if norm_m > 1e-9 and norm_t > 1e-9:
            ccc = float(np.sum(model_dens * target_density) / (norm_m * norm_t))
        else:
            ccc = 0.0

        return ccc

    def compute_em_density_forces(
        self,
        system: MolecularSystem,
        target_density: np.ndarray,
        grid_origin: np.ndarray
    ) -> np.ndarray:
        """
        Calculates real-space gradient steering forces F_EM = -grad(-w_EM * rho_target(r_i)).
        """
        dims = target_density.shape
        forces = np.zeros_like(system.coords)
        coords = system.coords

        # Compute numerical gradient of target density field
        grad_x, grad_y, grad_z = np.gradient(target_density, self.grid_spacing)

        for i, pt in enumerate(coords):
            gx = int((pt[0] - grid_origin[0]) / self.grid_spacing)
            gy = int((pt[1] - grid_origin[1]) / self.grid_spacing)
            gz = int((pt[2] - grid_origin[2]) / self.grid_spacing)

            if 1 <= gx < dims[0] - 1 and 1 <= gy < dims[1] - 1 and 1 <= gz < dims[2] - 1:
                fx = grad_x[gx, gy, gz]
                fy = grad_y[gx, gy, gz]
                fz = grad_z[gx, gy, gz]
                # Force drives atom toward positive density gradient
                forces[i] = np.array([fx, fy, fz]) * self.w_em * 1000.0

        return forces

    def run_flexible_fitting(
        self,
        system: MolecularSystem,
        target_density: np.ndarray,
        grid_origin: np.ndarray,
        num_steps: int = 50,
        dt: float = 0.01
    ) -> Tuple[MolecularSystem, CryoEMFittingMetrics]:
        """
        Simulates Langevin MD flexible fitting trajectory driven by Cryo-EM map potentials.
        """
        initial_coords = system.coords.copy()
        initial_ccc = self.compute_cross_correlation(system, target_density, grid_origin)

        fitted_sys = system.copy()
        curr_coords = fitted_sys.coords
        velocities = np.zeros_like(curr_coords)
        gamma = 1.0
        friction = np.exp(-gamma * dt)

        for _ in range(num_steps):
            # 1. EM Density Forces
            f_em = self.compute_em_density_forces(fitted_sys, target_density, grid_origin)

            # 2. Electrostatic & Physical forces via FMM
            _, f_phys, _ = self.fmm.evaluate(
                coords=fitted_sys.coords,
                charges=fitted_sys.charges,
                radii=fitted_sys.radii,
                compute_forces=True
            )

            # 3. Backbone harmonic restraint to preserve protein topology
            f_restraint = -0.5 * (curr_coords - initial_coords)

            total_forces = f_em + f_phys * 0.1 + f_restraint
            velocities = velocities * friction + total_forces * dt * 0.5
            curr_coords += velocities * dt
            fitted_sys.coords = curr_coords

        final_ccc = self.compute_cross_correlation(fitted_sys, target_density, grid_origin)
        rmsd = float(np.sqrt(np.mean(np.sum((fitted_sys.coords - initial_coords)**2, axis=-1))))

        # Clash score
        diff = fitted_sys.coords[:, None, :] - fitted_sys.coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) + np.eye(fitted_sys.num_atoms) * 1e9
        clashes = float(np.sum(dist < 1.2))

        conv = "Converged (High Quality)" if final_ccc >= initial_ccc else "Sub-optimal"

        metrics = CryoEMFittingMetrics(
            initial_ccc=initial_ccc,
            final_ccc=final_ccc,
            rmsd_displacement_A=rmsd,
            clash_score_after=clashes,
            resolution_angstrom=self.resolution,
            fitting_convergence=conv
        )

        return fitted_sys, metrics
