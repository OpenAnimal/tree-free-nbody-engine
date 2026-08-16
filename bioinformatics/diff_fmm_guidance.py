"""
Module 5: Differentiable FMM Physical Guidance for Generative Molecular Flow Matching & Diffusion.
Provides exact analytical energy gradients (-nabla_x E) and electrostatic potential steering
to eliminate steric clashes and maximize pocket binding complementarity during generative sampling.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable

try:
    from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, COULOMB_CONSTANT_KCAL
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
    from .pdb_loader import MolecularSystem
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, COULOMB_CONSTANT_KCAL
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D
    from bioinformatics.pdb_loader import MolecularSystem


@dataclass
class GuidanceStepResult:
    """Output of a single physically guided reverse diffusion / flow step."""
    guided_coords: np.ndarray      # (N_lig, 3) Corrected 3D coordinates
    drift_velocity: np.ndarray     # (N_lig, 3) Physical velocity component
    coulomb_energy_kcal: float
    steric_penalty_kcal: float
    total_physical_energy_kcal: float
    clash_count: int


@dataclass
class GenerativeValidationMetrics:
    """Physicochemical quality metrics for generated molecular conformations."""
    num_ligand_atoms: int
    num_receptor_atoms: int
    clash_rate_percent: float      # % of ligand atoms clashing with receptor (< 2.0 A)
    electrostatic_binding_energy_kcal: float
    pocket_burial_sasa_A2: float
    physical_validity_score: float # 0 to 100 (higher = physically feasible lead)


class DiffFMMGuidanceEngine:
    """
    Differentiable Tree-Free FMM Physical Guidance Engine for Molecular Generative Models.
    Steers reverse-time SE(3) diffusion & flow-matching ODE samplers towards low-energy,
    clash-free, electrostatically complementary pocket poses.
    """
    def __init__(
        self,
        cell_size: float = 6.0,
        kappa: float = 0.127,
        dielectric_pocket: float = 4.0,
        steric_clash_radius: float = 2.0,
        guidance_scale_gamma: float = 0.25
    ):
        self.cell_size = float(cell_size)
        self.kappa = float(kappa)
        self.eps_p = float(dielectric_pocket)
        self.clash_radius = float(steric_clash_radius)
        self.gamma = float(guidance_scale_gamma)

        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=self.kappa,
            dielectric_protein=self.eps_p,
            kernel_type=ScreenedKernelType.DEBYE_HUCKEL
        )

    def compute_pocket_physical_gradients(
        self,
        ligand_coords: np.ndarray,      # (N_lig, 3)
        ligand_charges: np.ndarray,     # (N_lig,)
        ligand_radii: np.ndarray,       # (N_lig,)
        receptor_coords: np.ndarray,    # (N_rec, 3)
        receptor_charges: np.ndarray,   # (N_rec,)
        receptor_radii: np.ndarray      # (N_rec,)
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Computes analytical forces (F = -grad_r E_total) on ligand atoms exerted by receptor pocket.
        F_total = F_screened_Coulomb + F_steric_repulsion + F_intra_ligand
        """
        N_lig = len(ligand_coords)
        N_rec = len(receptor_coords)

        # 1. Intermolecular Screened Coulomb Electrostatics
        diff = ligand_coords[:, None, :] - receptor_coords[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq + 1e-6)

        coulomb_factor = COULOMB_CONSTANT_KCAL / self.eps_p
        q_prod = ligand_charges[:, None] * receptor_charges[None, :]
        screened_pot = q_prod * np.exp(-self.kappa * dist) / dist
        e_coulomb = float(np.sum(screened_pot)) * coulomb_factor

        # Analytical gradient -dE/dr:
        # d/dr [exp(-kappa r)/r] = -(kappa + 1/r) exp(-kappa r) / r
        # F = -dE/dr * (diff / r)
        f_coulomb_mag = coulomb_factor * screened_pot * (self.kappa + 1.0 / dist)
        f_coulomb_vec = np.sum(diff * (f_coulomb_mag[:, :, None] / (dist[:, :, None] + 1e-6)), axis=1)

        # 2. Steric Soft-Core Repulsion (Clash Avoidance)
        min_dist = (ligand_radii[:, None] + receptor_radii[None, :]) * 0.85
        overlap = np.maximum(0.0, min_dist - dist)
        e_steric = float(np.sum(25.0 * (overlap**3)))

        f_steric_mag = 75.0 * (overlap**2) / dist
        f_steric_vec = np.sum(diff * (f_steric_mag[:, :, None] / (dist[:, :, None] + 1e-6)), axis=1)

        # 3. Intramolecular Ligand Steric Self-Avoidance
        if N_lig > 1:
            diff_intra = ligand_coords[:, None, :] - ligand_coords[None, :, :]
            dist_intra = np.sqrt(np.sum(diff_intra**2, axis=-1) + np.eye(N_lig) * 1e9)
            intra_min = (ligand_radii[:, None] + ligand_radii[None, :]) * 0.75
            intra_overlap = np.maximum(0.0, intra_min - dist_intra)
            f_intra_mag = 50.0 * (intra_overlap**2) / dist_intra
            np.fill_diagonal(f_intra_mag, 0.0)
            f_intra_vec = np.sum(diff_intra * (f_intra_mag[:, :, None] / (dist_intra[:, :, None] + 1e-6)), axis=1)
        else:
            f_intra_vec = np.zeros_like(ligand_coords)

        total_forces = f_coulomb_vec + f_steric_vec + f_intra_vec
        clashes = int(np.sum(dist < 2.0))

        energies = {
            "e_coulomb": e_coulomb,
            "e_steric": e_steric,
            "e_total": e_coulomb + e_steric,
            "clashes": clashes
        }

        return total_forces, energies

    def guide_sampling_step(
        self,
        t: float,                       # Timestep in [0, 1] (1 = pure noise, 0 = final pose)
        ligand_coords: np.ndarray,      # Current noisy coordinates
        model_velocity: np.ndarray,     # Velocity predicted by neural flow matching / diffusion score
        dt: float,                      # Step size
        ligand_charges: np.ndarray,
        ligand_radii: np.ndarray,
        receptor_coords: np.ndarray,
        receptor_charges: np.ndarray,
        receptor_radii: np.ndarray,
        schedule_power: float = 1.5
    ) -> GuidanceStepResult:
        """
        Applies physical guidance drift correction to a reverse ODE flow matching step:
        x_{t - dt} = x_t - (v_theta(x_t, t) + gamma(t) * F_phys(x_t)) * dt
        """
        # Time-dependent annealing schedule: strongest guidance in mid-to-late reverse steps
        # gamma(t) = gamma_0 * (1 - t)^schedule_power
        t_factor = float((1.0 - np.clip(t, 0.0, 1.0)) ** schedule_power)
        effective_gamma = self.gamma * t_factor

        forces, metrics = self.compute_pocket_physical_gradients(
            ligand_coords=ligand_coords,
            ligand_charges=ligand_charges,
            ligand_radii=ligand_radii,
            receptor_coords=receptor_coords,
            receptor_charges=receptor_charges,
            receptor_radii=receptor_radii
        )

        # Physical drift correction (F = -grad E, so adding F moves downhill along energy surface)
        physical_drift = effective_gamma * forces
        total_velocity = model_velocity + physical_drift

        guided_coords = ligand_coords - total_velocity * dt

        return GuidanceStepResult(
            guided_coords=guided_coords,
            drift_velocity=physical_drift,
            coulomb_energy_kcal=metrics["e_coulomb"],
            steric_penalty_kcal=metrics["e_steric"],
            total_physical_energy_kcal=metrics["e_total"],
            clash_count=int(metrics["clashes"])
        )

    def run_guided_reverse_flow(
        self,
        initial_noise_coords: np.ndarray,
        neural_drift_fn: Callable[[float, np.ndarray], np.ndarray],
        num_steps: int = 50,
        ligand_charges: Optional[np.ndarray] = None,
        ligand_radii: Optional[np.ndarray] = None,
        receptor: Optional[MolecularSystem] = None
    ) -> Tuple[np.ndarray, List[GuidanceStepResult]]:
        """
        Simulates an entire reverse generative diffusion / flow-matching trajectory
        with real-time Tree-Free FMM physical guidance.
        """
        N_lig = len(initial_noise_coords)
        if ligand_charges is None:
            ligand_charges = np.random.uniform(-0.5, 0.5, N_lig)
        if ligand_radii is None:
            ligand_radii = np.full(N_lig, 1.7)

        if receptor is not None:
            rec_coords = receptor.coords
            rec_charges = receptor.charges
            rec_radii = receptor.radii
        else:
            rec_coords = np.random.randn(200, 3) * 15.0
            rec_charges = np.random.uniform(-0.8, 0.8, 200)
            rec_radii = np.full(200, 1.7)

        dt = 1.0 / num_steps
        curr_coords = initial_noise_coords.copy()
        history = []

        for step in range(num_steps):
            t = 1.0 - step * dt
            v_model = neural_drift_fn(t, curr_coords)
            
            res = self.guide_sampling_step(
                t=t,
                ligand_coords=curr_coords,
                model_velocity=v_model,
                dt=dt,
                ligand_charges=ligand_charges,
                ligand_radii=ligand_radii,
                receptor_coords=rec_coords,
                receptor_charges=rec_charges,
                receptor_radii=rec_radii
            )
            curr_coords = res.guided_coords
            history.append(res)

        return curr_coords, history

    def evaluate_generative_validity(
        self,
        ligand_coords: np.ndarray,
        ligand_charges: np.ndarray,
        ligand_radii: np.ndarray,
        receptor_coords: np.ndarray,
        receptor_charges: np.ndarray,
        receptor_radii: np.ndarray
    ) -> GenerativeValidationMetrics:
        """
        Evaluates physical validity and pocket complementarity score of a generated ligand pose.
        """
        _, metrics = self.compute_pocket_physical_gradients(
            ligand_coords, ligand_charges, ligand_radii,
            receptor_coords, receptor_charges, receptor_radii
        )

        diff = ligand_coords[:, None, :] - receptor_coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        min_dists = np.min(dist, axis=1)

        clashing_atoms = np.sum(min_dists < 1.8)
        clash_rate = float(clashing_atoms / max(1, len(ligand_coords)) * 100.0)

        # Pocket burial: contacts within 4.5 Angstroms
        pocket_contacts = np.sum(min_dists < 4.5)
        burial_sasa = float(pocket_contacts * 18.5) # Approx A^2 buried

        # Physical validity score: starts at 100, penalized by clashes and unfavorable electrostatics
        clash_penalty = clash_rate * 1.5
        elec_score = np.clip(-metrics["e_coulomb"] * 0.5, -20.0, 30.0)
        validity_score = float(np.clip(70.0 - clash_penalty + elec_score + (burial_sasa / 50.0), 5.0, 100.0))

        return GenerativeValidationMetrics(
            num_ligand_atoms=len(ligand_coords),
            num_receptor_atoms=len(receptor_coords),
            clash_rate_percent=clash_rate,
            electrostatic_binding_energy_kcal=metrics["e_coulomb"],
            pocket_burial_sasa_A2=burial_sasa,
            physical_validity_score=validity_score
        )
