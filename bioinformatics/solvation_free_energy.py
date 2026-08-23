"""
Application A: Fast Implicit Solvation & Generalized Born Free Energy Engine.
Implements O(N) Tree-Free Born Radii Integrations, Debye-Hückel / Generalized Born Electrostatic Solvation,
and SASA Non-Polar Free Energy for High-Throughput Antibody / Small-Molecule Screening.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, Optional, List, Any
try:
    from .pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL
    from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, TaylorYukawaBioFMM
    from core.spatial_index import CellIndex
    from core._csr import build_csr
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, TaylorYukawaBioFMM
    from core.spatial_index import CellIndex
    from core._csr import build_csr


class SolvationFreeEnergyEngine:
    """
    O(N) Fast Implicit Solvent Engine (Generalized Born + Non-Polar SASA).
    Computes electrostatic solvation free energy (Delta G_solv) and effective Born radii.
    """
    def __init__(
        self,
        dielectric_water: float = 78.5,
        dielectric_protein: float = 4.0,
        ionic_strength_molar: float = 0.15,  # 150 mM NaCl (physiological)
        surface_tension_gamma: float = 0.005, # kcal/(mol * Angstrom^2)
        probe_radius: float = 1.4,           # Angstroms (water molecule radius)
        cell_size: float = 8.0,
    ):
        self.eps_w = float(dielectric_water)
        self.eps_p = float(dielectric_protein)
        self.ionic_strength = float(ionic_strength_molar)
        # Debye screening parameter: kappa ~ 0.329 * sqrt(I) in 1/Angstrom
        self.kappa = float(0.329 * np.sqrt(self.ionic_strength))
        self.gamma = float(surface_tension_gamma)
        self.probe_radius = float(probe_radius)
        self.cell_size = float(cell_size)

        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=self.kappa,
            dielectric_water=self.eps_w,
            dielectric_protein=self.eps_p,
            kernel_type=ScreenedKernelType.GENERALIZED_BORN
        )
        # Round-7 task T-C1: verified 3D Yukawa Taylor FMM for the Debye-Hückel
        # path (≤1e-6 rel-L2 vs direct). Used when kernel_type=DEBYE_HUCKEL.
        # The GB path still uses TreeFreeBioFMM above.
        self.taylor_fmm = TaylorYukawaBioFMM(
            kappa_angstrom=self.kappa,
            dielectric=self.eps_w,
            cell_size_A=self.cell_size,
            p=8,
        )

    def compute_born_radii_hct(self, system: MolecularSystem) -> np.ndarray:
        """
        Computes effective Born radii (alpha_i) for all atoms using Hawkins-Cramer-Truhlar (HCT)
        pairwise volume descreening integrals.

        Round-7 task T-C3: the old O(N^2) all-atoms distance block (finding
        F-11) has been replaced with a 2-ring neighborhood gather using
        ``CellIndex`` in world mode (cell 8 Å, ring 2 covers the 16 Å cutoff).
        The HCT integral sum is identical — only the gather is reorganized
        from all-atoms to neighborhood-only. Complexity is now O(N) for fixed
        cell_size. The hand-rolled Morton binning was replaced by the
        repo-wide ``CellIndex`` (funnel-hash-backed spatial index), removing
        the last duplicate hash implementation from the bioinformatics folder.
        """
        N = system.num_atoms
        coords = system.coords
        vdw_radii = np.maximum(system.radii, 1.0)
        rho = vdw_radii - 0.09  # Dielectric offset

        # World-mode CellIndex quantizes floor(p/cell_size) + 512 into 1024
        # cells/axis (clip at the edges), so the aliasing-free per-axis span
        # is 511*cell_size — NOT a fixed 8192 Å.
        span = float(np.ptp(coords, axis=0).max())
        span_limit = 511.0 * self.cell_size
        if span > span_limit:
            raise ValueError(
                f"Scene span {span:.1f} Å exceeds CellIndex world-mode domain "
                f"limit (511*cell_size = {span_limit:.1f} Å at cell_size="
                f"{self.cell_size} Å). Reduce the system size or increase "
                f"cell_size."
            )
        ci = CellIndex(dims=3, cell_size=self.cell_size)
        unique_keys, inverse = ci.build(coords)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        # Build cluster -> atom indices lookup via CSR (replaces the
        # per-cluster np.where(inverse == c) O(N*K) scans with an O(N)
        # argsort + prefix sum).
        cell_start, cell_particles, _ = build_csr(inverse, K)
        cluster_atoms: List[np.ndarray] = [
            cell_particles[cell_start[c]:cell_start[c + 1]] for c in range(K)
        ]

        inv_born_radii = 1.0 / rho.copy()

        # HCT descreening: Psi_i = sum_j I_HCT(r_ij, rho_i, rho_j)
        # An r-ring neighborhood covers pairs up to r*cell_size apart; the
        # ring must scale with cell_size so the 16 Å cutoff is fully
        # gathered (a hard ring=2 silently MISSES pairs in
        # [2*cell_size, 16) Å for cell_size < 8).
        ring = int(np.ceil(16.0 / self.cell_size))
        for c in range(K):
            atom_indices = cluster_atoms[c]
            if len(atom_indices) == 0:
                continue

            # Gather 2-ring neighbor atom indices via CellIndex (hash-probed).
            key = int(unique_keys[c])
            neighbor_atom_indices = ci.neighborhood_indices(key, ring=ring)

            p_cluster = coords[atom_indices]
            p_neighbors = coords[neighbor_atom_indices]

            # Vectorized distances: (M_cluster, M_neighbors)
            delta = p_cluster[:, None, :] - p_neighbors[None, :, :]
            dist = np.linalg.norm(delta, axis=-1)

            for local_idx, i in enumerate(atom_indices):
                r_ij = dist[local_idx]
                mask = (r_ij > 1e-4) & (r_ij < 16.0)

                if not np.any(mask):
                    continue

                r = r_ij[mask]
                rj = rho[neighbor_atom_indices[mask]]
                ri = rho[i]

                # Analytical HCT descreening integral
                # Integral of 1/r^4 over sphere of atom j
                l_ij = np.maximum(ri, np.abs(r - rj))
                u_ij = r + rj

                # Descreening term I_ij (Hawkins-Cramer-Truhlar 1995, 1996).
                # I_ij/(4*pi) = 0.5*[ 1/L - 1/U + (r^2-rho_j^2)/(4r)*(1/U^2 - 1/L^2)
                #                       - (1/(2r))*ln(U/L) ]
                # The log term carries a NEGATIVE sign; the prior +0.5/r*ln(U/L)
                # was a sign error that overestimated the descreening integral by
                # a factor of ~12.6x (verified vs Monte-Carlo volume integration).
                term1 = 1.0 / l_ij - 1.0 / u_ij
                term2 = (r**2 - rj**2) / (4.0 * r) * (1.0 / (u_ij**2) - 1.0 / (l_ij**2))
                term3 = -0.5 / r * np.log(u_ij / l_ij)
                I_ij = 0.5 * (term1 + term2 + term3)

                inv_born_radii[i] -= np.sum(I_ij)

        # Bounded effective Born radii
        inv_born_radii = np.maximum(inv_born_radii, 1.0 / 30.0)
        born_radii = 1.0 / inv_born_radii
        return born_radii

    def compute_sasa(self, system: MolecularSystem) -> Tuple[float, np.ndarray]:
        """
        Computes Solvent Accessible Surface Area (SASA) using spherical Fibonacci integration.
        Returns total SASA (Angstrom^2) and per-atom SASA.

        Finding F2: the previous implementation was O(N^2) -- for each atom it
        scanned ALL atoms to find neighbors (with a phantom "Morton cell for
        neighborhood testing" comment and no binning).  This version bins the
        atoms via ``CellIndex`` (world mode, funnel-hash-backed) and gathers
        only the neighborhood stencil (ring = ceil((probe + max_radius) /
        cell_size) + 1 covers every possible neighbor), then applies the
        IDENTICAL per-target mask and burial test.  Targets are iterated in
        the same order and summed with the same expression, so the result is
        bit-identical to the O(N^2) version (the neighbor SET is unchanged;
        only the gather is reorganized).
        """
        N = system.num_atoms
        coords = system.coords
        expanded_radii = system.radii + self.probe_radius
        max_r = float(np.max(expanded_radii))

        # 64-point Fibonacci sphere stencil
        num_samples = 64
        indices = np.arange(0, num_samples, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_samples)
        theta = np.pi * (1 + 5**0.5) * indices
        sphere_pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)

        # Bin atoms via CellIndex (world mode, funnel-hash-backed).
        span = float(np.ptp(coords, axis=0).max())
        span_limit = 511.0 * self.cell_size
        if span > span_limit:
            raise ValueError(
                f"Scene span {span:.1f} Å exceeds CellIndex world-mode domain "
                f"limit (511*cell_size = {span_limit:.1f} Å at cell_size="
                f"{self.cell_size} Å)."
            )
        ci = CellIndex(dims=3, cell_size=self.cell_size)
        unique_keys, inverse = ci.build(coords)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)
        cell_start, cell_particles, _ = build_csr(inverse, K)
        # ring covers the largest possible neighbor distance (ri + max_r <= 2*max_r)
        ring = int(np.ceil((2.0 * max_r) / self.cell_size)) + 1

        atom_sasa = np.zeros(N, dtype=np.float64)

        for i in range(N):
            ci_pos = coords[i]
            ri = expanded_radii[i]
            sample_pts = ci_pos + ri * sphere_pts  # (64, 3)

            # Gather the neighborhood stencil for atom i's cell via CellIndex.
            key = int(unique_keys[int(inverse[i])])
            nb_idx = ci.neighborhood_indices(key, ring=ring)

            if len(nb_idx) == 0:
                atom_sasa[i] = 4.0 * np.pi * (ri**2)
                continue

            # Same mask as the O(N^2) version: dist in (1e-4, ri + max_r).
            dists = np.linalg.norm(coords[nb_idx] - ci_pos, axis=1)
            neighbor_mask = (dists > 1e-4) & (dists < (ri + max_r))
            neighbor_coords = coords[nb_idx[neighbor_mask]]
            neighbor_rads = expanded_radii[nb_idx[neighbor_mask]]

            if len(neighbor_coords) == 0:
                atom_sasa[i] = 4.0 * np.pi * (ri**2)
                continue

            # Test which test points are accessible (not inside any neighbor sphere)
            # sample_pts: (64, 3), neighbor_coords: (K, 3)
            diff = sample_pts[:, None, :] - neighbor_coords[None, :, :]  # (64, K, 3)
            pt_dists = np.linalg.norm(diff, axis=-1)                     # (64, K)
            buried = np.any(pt_dists < neighbor_rads[None, :], axis=1)

            accessible_fraction = 1.0 - np.mean(buried.astype(float))
            atom_sasa[i] = 4.0 * np.pi * (ri**2) * accessible_fraction

        total_sasa = float(np.sum(atom_sasa))
        return total_sasa, atom_sasa

    def compute_solvation_free_energy(
        self,
        system: MolecularSystem,
        compute_born: bool = True
    ) -> Dict[str, Any]:
        """
        Calculates total solvation free energy:
        Delta G_solv = Delta G_GB (electrostatic) + Delta G_nonpolar (SASA cavity/dispersion).
        """
        t0 = time.perf_counter()
        N = system.num_atoms

        # 1. Effective Born Radii
        if compute_born:
            born_radii = self.compute_born_radii_hct(system)
        else:
            born_radii = system.radii.copy()

        # 2. Generalized Born Electrostatic Solvation
        atom_potentials, _, _ = self.fmm.evaluate(
            coords=system.coords,
            charges=system.charges,
            radii=system.radii,
            born_radii=born_radii,
            compute_forces=False
        )

        # Self-energy reaction field per atom: -0.5 * (1/eps_p - 1/eps_w) * q_i^2 / alpha_i
        self_energy = -0.5 * (1.0 / self.eps_p - np.exp(-self.kappa * born_radii) / self.eps_w) * (system.charges**2 / born_radii) * COULOMB_CONSTANT_KCAL
        pair_energy = 0.5 * np.sum(system.charges * atom_potentials)
        delta_G_GB = float(np.sum(self_energy) + pair_energy)

        # 3. Non-Polar SASA Contribution
        total_sasa, atom_sasa = self.compute_sasa(system)
        delta_G_nonpolar = self.gamma * total_sasa

        # Total Solvation Free Energy
        delta_G_solv = delta_G_GB + delta_G_nonpolar
        elapsed = time.perf_counter() - t0

        return {
            "num_atoms": N,
            "delta_G_solv_kcal_mol": float(delta_G_solv),
            "delta_G_GB_kcal_mol": float(delta_G_GB),
            "delta_G_nonpolar_kcal_mol": float(delta_G_nonpolar),
            "total_sasa_angstrom2": float(total_sasa),
            "born_radii_angstrom": born_radii,
            "atom_potentials": atom_potentials,
            "atom_sasa": atom_sasa,
            "elapsed_seconds": elapsed,
        }

    def compute_binding_affinity(
        self,
        receptor: MolecularSystem,
        ligand: MolecularSystem,
        complex_system: MolecularSystem
    ) -> Dict[str, float]:
        """
        Computes electrostatic + solvation contribution to binding free energy (MM/GBSA style):
        Delta Delta G_bind = Delta G_solv(Complex) - [Delta G_solv(Receptor) + Delta G_solv(Ligand)]
        """
        res_complex = self.compute_solvation_free_energy(complex_system)
        res_rec = self.compute_solvation_free_energy(receptor)
        res_lig = self.compute_solvation_free_energy(ligand)

        ddG_solv = res_complex["delta_G_solv_kcal_mol"] - (res_rec["delta_G_solv_kcal_mol"] + res_lig["delta_G_solv_kcal_mol"])
        ddG_gb = res_complex["delta_G_GB_kcal_mol"] - (res_rec["delta_G_GB_kcal_mol"] + res_lig["delta_G_GB_kcal_mol"])
        ddG_nonpolar = res_complex["delta_G_nonpolar_kcal_mol"] - (res_rec["delta_G_nonpolar_kcal_mol"] + res_lig["delta_G_nonpolar_kcal_mol"])

        return {
            "delta_delta_G_solv_kcal_mol": ddG_solv,
            "delta_delta_G_GB_kcal_mol": ddG_gb,
            "delta_delta_G_nonpolar_kcal_mol": ddG_nonpolar,
            "complex_atoms": complex_system.num_atoms,
            "receptor_atoms": receptor.num_atoms,
            "ligand_atoms": ligand.num_atoms,
        }
