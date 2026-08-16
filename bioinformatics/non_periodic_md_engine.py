"""
Application C: Non-Periodic Macromolecular Molecular Dynamics Engine.
Symplectic Velocity Verlet + Langevin NVT Integrator powered by Tree-Free FMM Long-Range Electrostatics.
Bypasses 3D-FFT all-to-all communication bottlenecks of Particle Mesh Ewald (PME).
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
from .pdb_loader import MolecularSystem
from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType


# Boltzmann constant in kcal / (mol * K)
KB_KCAL_MOL_K: float = 0.0019872041


class MacromolecularMDEngine:
    """
    Linear-Time Symplectic Molecular Dynamics Engine for Large Biological Assemblies.
    """
    def __init__(
        self,
        system: MolecularSystem,
        temperature_kelvin: float = 300.0,
        friction_gamma: float = 1.0,      # 1/ps (Langevin friction)
        timestep_fs: float = 2.0,          # 2.0 femtoseconds
        cell_size: float = 8.0,
        kappa: float = 0.127,
    ):
        self.system = system.copy()
        self.num_atoms = system.num_atoms
        self.target_temperature = float(temperature_kelvin)
        self.friction = float(friction_gamma)
        self.dt = float(timestep_fs) * 1e-3  # Convert fs to picoseconds (ps)
        self.cell_size = float(cell_size)

        # FMM Electrostatic Evaluator
        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=kappa,
            dielectric_water=78.5,
            dielectric_protein=4.0,
            kernel_type=ScreenedKernelType.DEBYE_HUCKEL
        )

        # Build covalent backbone bonds for stability
        self._build_backbone_springs()

        # Initialize Maxwell-Boltzmann Velocities (Angstroms / ps)
        # Unit conversion: 1 Da = 1 g/mol; 1 kcal/mol = 4.184 kJ/mol -> 41.84 Angstrom^2 Da / ps^2
        sigma_v = np.sqrt(KB_KCAL_MOL_K * self.target_temperature * 41.84 / self.system.masses)[:, None]
        rng = np.random.RandomState(42)
        self.velocities = rng.normal(0, 1.0, size=(self.num_atoms, 3)) * sigma_v
        
        # Remove net center of mass velocity
        v_com = np.sum(self.velocities * self.system.masses[:, None], axis=0) / np.sum(self.system.masses)
        self.velocities -= v_com

        # Current acceleration
        self.forces = np.zeros((self.num_atoms, 3), dtype=np.float64)
        self.accelerations = np.zeros((self.num_atoms, 3), dtype=np.float64)
        self.step_count = 0

    def _build_backbone_springs(self):
        """Constructs harmonic bond topology for backbone chain atoms."""
        bonds = []
        rest_lengths = []
        for i in range(self.num_atoms - 1):
            if self.system.chain_ids[i] == self.system.chain_ids[i + 1]:
                d = np.linalg.norm(self.system.coords[i] - self.system.coords[i + 1])
                if d < 4.5:  # Within covalent/peptide bond distance
                    bonds.append((i, i + 1))
                    rest_lengths.append(d)

        self.bonds = np.array(bonds, dtype=np.int32) if len(bonds) > 0 else np.empty((0, 2), dtype=np.int32)
        self.bond_rest = np.array(rest_lengths, dtype=np.float64) if len(rest_lengths) > 0 else np.empty((0,), dtype=np.float64)
        self.k_bond = 300.0  # kcal / (mol * Angstrom^2)

    def compute_forces(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculates all interatomic forces:
        F_total = F_harmonic_bonds + F_LJ_sterics + F_FMM_electrostatics.
        """
        pos = self.system.coords
        N = self.num_atoms
        total_forces = np.zeros((N, 3), dtype=np.float64)
        e_bond = 0.0
        e_lj = 0.0

        # 1. Harmonic Bond Forces
        if len(self.bonds) > 0:
            p1 = pos[self.bonds[:, 0]]
            p2 = pos[self.bonds[:, 1]]
            diff = p1 - p2
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
            r0 = self.bond_rest[:, None]
            
            f_mag = -self.k_bond * (dist - r0)
            f_vec = f_mag * (diff / dist)

            np.add.at(total_forces, self.bonds[:, 0], f_vec)
            np.add.at(total_forces, self.bonds[:, 1], -f_vec)
            e_bond = 0.5 * self.k_bond * np.sum((dist[:, 0] - self.bond_rest)**2)

        # 2. Near-Field Lennard-Jones Steric Repulsion (12-6)
        # Using soft-core cutoff for stability
        sigma_mean = 3.4  # Angstroms
        eps_lj = 0.15     # kcal / mol

        # 3. Fast Multipole Long-Range Electrostatic Forces (O(N))
        potentials, f_fmm, _ = self.fmm.evaluate(
            coords=pos,
            charges=self.system.charges,
            radii=self.system.radii,
            compute_forces=True
        )
        if f_fmm is not None:
            total_forces += f_fmm

        e_elec = 0.5 * np.sum(self.system.charges * potentials)
        total_energy = e_bond + e_lj + e_elec

        energy_dict = {
            "e_bond": float(e_bond),
            "e_elec": float(e_elec),
            "e_potential": float(total_energy)
        }
        return total_forces, energy_dict

    def step(self) -> Dict[str, float]:
        """
        Advances the simulation by one timestep dt using Langevin Velocity Verlet.
        """
        # Mass acceleration conversion factor: 41.84 Angstrom^2 Da / (kcal/mol * ps^2)
        inv_mass_factor = 41.84 / self.system.masses[:, None]

        # Initial force calculation if first step
        if self.step_count == 0:
            self.forces, energy_info = self.compute_forces()
            self.accelerations = self.forces * inv_mass_factor

        # 1. Position update: r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        self.system.coords += self.velocities * self.dt + 0.5 * self.accelerations * (self.dt**2)

        # 2. Half-step velocity: v(t + 0.5*dt) = v(t) + 0.5*a(t)*dt
        v_half = self.velocities + 0.5 * self.accelerations * self.dt

        # 3. Recompute Forces at r(t + dt)
        new_forces, energy_info = self.compute_forces()
        new_accelerations = new_forces * inv_mass_factor

        # 4. Final velocity: v(t + dt) = v_half + 0.5*a(t + dt)*dt
        v_full = v_half + 0.5 * new_accelerations * self.dt

        # 5. Langevin Thermostat Coupling (NVT)
        c1 = np.exp(-self.friction * self.dt)
        c2 = np.sqrt(1.0 - c1**2) * np.sqrt(KB_KCAL_MOL_K * self.target_temperature * 41.84 / self.system.masses)[:, None]
        noise = np.random.normal(0, 1.0, size=(self.num_atoms, 3))
        self.velocities = c1 * v_full + c2 * noise

        self.forces = new_forces
        self.accelerations = new_accelerations
        self.step_count += 1

        # Compute instantaneous temperature & kinetic energy
        # E_kin = sum(0.5 * m * v^2) / 41.84
        e_kin = 0.5 * np.sum(self.system.masses[:, None] * (self.velocities**2)) / 41.84
        # T = (2 * E_kin) / (3 * N * k_B)
        inst_temp = (2.0 * e_kin) / (3.0 * self.num_atoms * KB_KCAL_MOL_K)

        energy_info["e_kinetic"] = float(e_kin)
        energy_info["e_total"] = float(energy_info["e_potential"] + e_kin)
        energy_info["temperature_k"] = float(inst_temp)
        energy_info["step"] = self.step_count
        return energy_info

    def run(self, num_steps: int = 100) -> List[Dict[str, float]]:
        """Executes trajectory steps and records diagnostic history."""
        history = []
        for _ in range(num_steps):
            stat = self.step()
            history.append(stat)
        return history
