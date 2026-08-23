"""
Application C: Non-Periodic Macromolecular Molecular Dynamics Engine.
Symplectic Velocity Verlet + Langevin NVT Integrator powered by Tree-Free FMM Long-Range Electrostatics.
Bypasses 3D-FFT all-to-all communication bottlenecks of Particle Mesh Ewald (PME).
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
try:
    from .pdb_loader import MolecularSystem
    from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
    from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_decode_3d
    from core._csr import build_csr
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D, morton_decode_3d
    from core._csr import build_csr


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

        # Lennard-Jones steric parameters (single united-atom type).
        # R10-D4: the LJ term used to be declared in compute_forces but never
        # computed (e_lj was always 0.0 and no steric forces existed).
        self.lj_sigma = 3.4   # Angstroms
        self.lj_eps = 0.15    # kcal / mol
        self.lj_cutoff = 2.5 * self.lj_sigma  # truncated (unshifted) 12-6
        # Soft-core-like stabilization: the repulsive 12-6 force is capped at
        # lj_fmax below the cap radius r_c (root of F_12-6(r_c) = lj_fmax),
        # with the energy V(r) = V(r_c) + fmax*(r_c - r) on that branch so
        # that F = -grad V holds exactly everywhere. Unrelaxed synthetic
        # structures contain sub-sigma contacts whose uncapped r^-12 force
        # (up to ~1e6 kcal/mol/A) blows up any fixed-step integrator.
        self.lj_fmax = 200.0  # kcal / (mol * Angstrom)
        self._lj_rcap = self._solve_lj_cap_radius()

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
        # Unit conversion: 1 kcal/mol = 418.4 Da * Å^2 / ps^2
        # (1 kcal = 4184 J, 1 Da*Å^2/ps^2 = 1.66054e-23 J per molecule,
        #  so 1 kcal/mol = 4184 / (NA * 1.66054e-23) = 418.4 Da*Å^2/ps^2.
        #  The prior 41.84 was a factor-of-10 typo.)
        sigma_v = np.sqrt(KB_KCAL_MOL_K * self.target_temperature * 418.4 / self.system.masses)[:, None]
        rng = np.random.RandomState(42)
        self.velocities = rng.normal(0, 1.0, size=(self.num_atoms, 3)) * sigma_v
        
        # Remove net center of mass velocity
        v_com = np.sum(self.velocities * self.system.masses[:, None], axis=0) / np.sum(self.system.masses)
        self.velocities -= v_com

        # Current acceleration
        self.forces = np.zeros((self.num_atoms, 3), dtype=np.float64)
        self.accelerations = np.zeros((self.num_atoms, 3), dtype=np.float64)
        self.step_count = 0

    def _solve_lj_cap_radius(self) -> float:
        """Radius below which the 12-6 repulsive force reaches lj_fmax."""
        lo, hi = 0.5 * self.lj_sigma, 2.0 * self.lj_sigma
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f = self._lj_force_scalar(mid)
            if f > self.lj_fmax:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _lj_force_scalar(self, r: float) -> float:
        x6 = (self.lj_sigma / r) ** 6
        return 24.0 * self.lj_eps / r * (2.0 * x6 * x6 - x6)

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

        # 2. Near-Field Lennard-Jones Steric Repulsion (12-6, cutoff 2.5*sigma).
        #    Truncated unshifted potential via the Morton elastic spatial hash
        #    cell lists (27-cell gather), mirroring contact_map_graph.py.
        #    R10-D4: this term used to be declared but never computed.
        if N >= 2:
            sigma = self.lj_sigma
            eps = self.lj_eps
            rc = self.lj_cutoff
            lj_hash = ElasticSpatialHash3D(cell_size=rc, capacity_hint=N * 2)
            lj_origin = np.min(pos, axis=0) - rc
            _, lj_keys, lj_inv = lj_hash.build_from_coords(pos, origin=lj_origin)
            K_lj = len(lj_keys)
            lj_start, lj_particles, _ = build_csr(lj_inv, K_lj)
            lj_grid = np.array([morton_decode_3d(int(k)) for k in lj_keys],
                               dtype=np.int64)
            g_diff = np.abs(lj_grid[:, None, :] - lj_grid[None, :, :])
            lj_near = np.all(g_diff <= 1, axis=-1)

            for c1 in range(K_lj):
                idx1 = lj_particles[lj_start[c1]:lj_start[c1 + 1]]
                if len(idx1) == 0:
                    continue
                for c2 in np.where(lj_near[c1])[0]:
                    if c2 < c1:
                        continue
                    idx2 = lj_particles[lj_start[c2]:lj_start[c2 + 1]]
                    if len(idx2) == 0:
                        continue
                    p1 = pos[idx1]
                    p2 = pos[idx2]
                    delta = p1[:, None, :] - p2[None, :, :]
                    dist = np.linalg.norm(delta, axis=-1)
                    if c1 == c2:
                        i_u, j_u = np.triu_indices(len(idx1), k=1)
                        dvec = delta[i_u, j_u]
                        r_all = dist[i_u, j_u]
                        a1 = idx1[i_u]
                        a2 = idx1[j_u]
                    else:
                        dvec = delta.reshape(-1, 3)
                        r_all = dist.reshape(-1)
                        a1 = np.repeat(idx1, len(idx2))
                        a2 = np.tile(idx2, len(idx1))
                    act = r_all < rc
                    if not np.any(act):
                        continue
                    r = np.maximum(r_all[act], 1e-8)
                    d3 = dvec[act] / r[:, None]
                    sr6 = (sigma / r) ** 6
                    f_unc = (24.0 * eps / r) * (2.0 * sr6 * sr6 - sr6)
                    capped = f_unc > self.lj_fmax
                    f_pair = np.where(capped, self.lj_fmax, f_unc)
                    # energy consistent with the capped force:
                    # V = V(r_c) + fmax*(r_c - r) on the capped branch
                    v_rc = 4.0 * eps * ((sigma / self._lj_rcap) ** 12
                                        - (sigma / self._lj_rcap) ** 6)
                    e_unc = 4.0 * eps * (sr6 * sr6 - sr6)
                    e_pair = np.where(capped,
                                      v_rc + self.lj_fmax * (self._lj_rcap - r),
                                      e_unc)
                    e_lj += float(np.sum(e_pair))
                    fv = f_pair[:, None] * d3
                    np.add.at(total_forces, a1[act], fv)
                    np.add.at(total_forces, a2[act], -fv)

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
            "e_lj": float(e_lj),
            "e_elec": float(e_elec),
            "e_potential": float(total_energy)
        }
        return total_forces, energy_dict

    def step(self) -> Dict[str, float]:
        """
        Advances the simulation by one timestep dt using Langevin Velocity Verlet.
        """
        # Mass acceleration conversion factor: 418.4 Å^2 Da / (kcal/mol * ps^2)
        inv_mass_factor = 418.4 / self.system.masses[:, None]

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
        c2 = np.sqrt(1.0 - c1**2) * np.sqrt(KB_KCAL_MOL_K * self.target_temperature * 418.4 / self.system.masses)[:, None]
        noise = np.random.normal(0, 1.0, size=(self.num_atoms, 3))
        self.velocities = c1 * v_full + c2 * noise

        self.forces = new_forces
        self.accelerations = new_accelerations
        self.step_count += 1

        # Compute instantaneous temperature & kinetic energy
        # E_kin = sum(0.5 * m * v^2) / 418.4  (kcal/mol)
        e_kin = 0.5 * np.sum(self.system.masses[:, None] * (self.velocities**2)) / 418.4
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
