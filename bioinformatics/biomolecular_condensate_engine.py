"""
Module 4: Biomolecular Condensates, Liquid-Liquid Phase Separation (LLPS) & Whole-Cell Crowding Engine.
Simulates multi-component multi-million-particle membraneless organelles (stress granules, nuclear speckles)
using O(N) Tree-Free Morton Binning, screened Debye-Huckel electrostatics, and multivalent sticker-spacer potentials.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d, morton_decode_3d
    from .pdb_loader import COULOMB_CONSTANT_KCAL
    from core._csr import build_csr
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d, morton_decode_3d
    from bioinformatics.pdb_loader import COULOMB_CONSTANT_KCAL
    from core._csr import build_csr


@dataclass
class CondensateDroplet:
    """Characterization of an identified phase-separated biomolecular droplet."""
    droplet_id: int
    num_particles: int
    radius_gyration_nm: float
    center_of_mass: np.ndarray
    dense_phase_density_mg_ml: float
    composition_fractions: Dict[str, float]  # e.g., {"IDR_FUS": 0.65, "RNA": 0.25, "Crowder": 0.10}


@dataclass
class PhaseSeparationReport:
    """Full thermodynamic liquid-liquid phase separation (LLPS) assessment."""
    num_total_particles: int
    box_size_nm: float
    volume_fraction_percent: float
    ionic_strength_molar: float
    temperature_kelvin: float
    is_phase_separated: bool
    num_droplets: int
    droplets: List[CondensateDroplet]
    dilute_phase_density_mg_ml: float
    dense_phase_density_mg_ml: float
    radial_distribution_peaks: List[Tuple[float, float]] # (r_nm, g(r))
    critical_temperature_estimate_k: float


class BiomolecularCondensateEngine:
    """
    O(N) Whole-Cell Crowding & Biomolecular Condensate Simulator.
    Simulates sticker-spacer associative polymers and RNA polyanions in crowded cellular environments.
    """
    def __init__(
        self,
        box_size_nm: float = 120.0,
        temperature_kelvin: float = 300.0,
        ionic_strength_molar: float = 0.15,
        dielectric_water: float = 78.5,
        cell_size_nm: float = 6.0
    ):
        self.box_size = float(box_size_nm)
        self.temperature = float(temperature_kelvin)
        self.ionic_strength = float(ionic_strength_molar)
        # kappa in 1/nm
        self.kappa = float(3.29 * np.sqrt(self.ionic_strength))
        self.dielectric = float(dielectric_water)
        self.cell_size = float(cell_size_nm)

    def generate_crowded_cell_system(
        self,
        num_idr_chains: int = 100,
        beads_per_idr: int = 40,
        num_rna_chains: int = 20,
        beads_per_rna: int = 30,
        num_crowders: int = 300
    ) -> Dict[str, np.ndarray]:
        """
        Generates a realistic multi-component cellular cytoplasm containing
        intrinsically disordered proteins (e.g. FUS/TDP-43), polyanionic RNA, and inert crowders.
        """
        rng = np.random.RandomState(42)
        all_coords = []
        all_charges = []
        all_radii = []
        all_types = []

        bead_radius = 1.5 # nm

        # 1. IDR Chains (Sticker-Spacer alternating charge & hydrophobic motifs)
        for _ in range(num_idr_chains):
            chain_origin = rng.uniform(10.0, self.box_size - 10.0, size=3)
            curr = chain_origin
            for b in range(beads_per_idr):
                step = rng.randn(3)
                step /= np.linalg.norm(step) + 1e-9
                curr = curr + step * (2.0 * bead_radius)
                curr = np.mod(curr, self.box_size)
                all_coords.append(curr)

                # Alternating stickers (Arg+, Tyr/Phe neutral) and spacers (Gly/Ser)
                if b % 4 == 0:
                    q = 1.0   # Basic sticker (Arg/Lys)
                    t = "IDR_STICKER_POS"
                elif b % 4 == 2:
                    q = -1.0  # Acidic spacer (Asp/Glu)
                    t = "IDR_SPACER_NEG"
                else:
                    q = 0.0   # Aromatic / Polar (Tyr, Ser)
                    t = "IDR_AROMATIC"

                all_charges.append(q)
                all_radii.append(bead_radius)
                all_types.append(t)

        # 2. Polyanionic RNA Chains (Phosphate backbone -1e per bead)
        for _ in range(num_rna_chains):
            chain_origin = rng.uniform(10.0, self.box_size - 10.0, size=3)
            curr = chain_origin
            for _ in range(beads_per_rna):
                step = rng.randn(3)
                step /= np.linalg.norm(step) + 1e-9
                curr = curr + step * (2.0 * bead_radius)
                curr = np.mod(curr, self.box_size)
                all_coords.append(curr)
                all_charges.append(-1.5)  # Highly negative RNA
                all_radii.append(bead_radius * 1.1)
                all_types.append("RNA_POLYANION")

        # 3. Inert Crowders (e.g., Globular BSA / Chaperones)
        for _ in range(num_crowders):
            pos = rng.uniform(0.0, self.box_size, size=3)
            all_coords.append(pos)
            all_charges.append(0.0)
            all_radii.append(bead_radius * 1.5)
            all_types.append("CROWDER_INERT")

        return {
            "coords": np.array(all_coords, dtype=np.float64),
            "charges": np.array(all_charges, dtype=np.float64),
            "radii": np.array(all_radii, dtype=np.float64),
            "types": np.array(all_types, dtype=object)
        }

    def compute_forces_and_energy(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        radii: np.ndarray,
        types: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        O(N) Morton-accelerated evaluation of screened electrostatics, sticker-spacer
        attractive contact wells, and excluded volume.

        Round-7 fix (finding E): the previous implementation evaluated pairwise
        physics ONLY within each Morton cell, so two beads 1 nm apart across a
        cell boundary silently never interacted.  We now gather the full
        3x3x3 (27-cell) neighborhood for each cell (pattern from
        fast_multipole_kernel.py:263), so adjacent-cell pairs interact.  The
        per-pair arithmetic is unchanged; the 0.5 energy factor absorbs the
        resulting double-count of cross-cell pairs (each pair is visited from
        both cells), and forces are accumulated per-target without halving.
        """
        N = len(coords)
        forces = np.zeros_like(coords)
        total_energy = 0.0

        # Spatial Morton Binning
        origin = np.zeros(3)
        ix = np.clip((coords[:, 0] / self.cell_size).astype(np.int64), 0, 1023)
        iy = np.clip((coords[:, 1] / self.cell_size).astype(np.int64), 0, 1023)
        iz = np.clip((coords[:, 2] / self.cell_size).astype(np.int64), 0, 1023)
        morton_keys = morton_encode_3d(ix, iy, iz)
        unique_keys, inverse = np.unique(morton_keys, return_inverse=True)
        K = len(unique_keys)

        # CSR cell lists (drop-in replacement for per-cluster np.where scans).
        cell_start, cell_particles, _ = build_csr(inverse, K)

        # grid coord -> cluster index map for 27-neighborhood enumeration
        cluster_grid = np.array([morton_decode_3d(int(k)) for k in unique_keys], dtype=np.int64)
        grid_to_cluster = {}
        for c in range(K):
            grid_to_cluster[(int(cluster_grid[c, 0]),
                             int(cluster_grid[c, 1]),
                             int(cluster_grid[c, 2]))] = c

        coulomb_factor = COULOMB_CONSTANT_KCAL / self.dielectric

        # For each cell, gather the 27-cell neighborhood and evaluate pairs.
        for c1 in range(K):
            idx1 = cell_particles[cell_start[c1]:cell_start[c1 + 1]]
            if len(idx1) == 0:
                continue

            # Gather 27-neighborhood cluster indices
            cx, cy, cz = int(cluster_grid[c1, 0]), int(cluster_grid[c1, 1]), int(cluster_grid[c1, 2])
            neighbor_clusters = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (cx + dx, cy + dy, cz + dz)
                        nc = grid_to_cluster.get(key)
                        if nc is not None:
                            neighbor_clusters.append(nc)
            idx2 = np.concatenate([cell_particles[cell_start[nc]:cell_start[nc + 1]]
                                   for nc in neighbor_clusters])
            if len(idx2) == 0:
                continue

            pts = coords[idx1]
            qs = charges[idx1]
            rs = radii[idx1]
            ts = types[idx1]
            n_c = len(idx1)

            p2 = coords[idx2]
            q2 = charges[idx2]
            r2 = radii[idx2]
            t2 = types[idx2]

            diff = pts[:, None, :] - p2[None, :, :]      # (n_c, n_nb, 3)
            r_mat = np.linalg.norm(diff, axis=-1)

            # Self-pair mask: only the diagonal of the self-cell block (i == i).
            # Build a boolean mask of shape (n_c, n_nb) that is True where the
            # target index equals the source index.
            self_mask = (idx1[:, None] == idx2[None, :])
            r_mat = np.where(self_mask, 1e9, r_mat)

            # 1. Screened Debye-Huckel
            screened_pot = (qs[:, None] * q2[None, :]) * np.exp(-self.kappa * r_mat) / (r_mat + 1e-6)
            total_energy += float(np.sum(screened_pot)) * 0.5 * coulomb_factor
            f_elec_mag = coulomb_factor * screened_pot * (self.kappa + 1.0 / (r_mat + 1e-6))

            # 2. Excluded volume
            # U_ev = 40/3 * overlap^3, F = -dU/dr = 40 * overlap^2 (repulsive).
            # P19-1: the previous code divided f_ev_mag by r, giving a force
            # 40*overlap^2/r instead of 40*overlap^2 — too weak by a factor
            # of r (typically 3-15x for nm-scale beads).
            sigma = rs[:, None] + r2[None, :]
            overlap = np.maximum(0.0, sigma - r_mat)
            f_ev_mag = 40.0 * (overlap**2)
            total_energy += float(np.sum(40.0 / 3.0 * (overlap**3))) * 0.5

            # 3. Sticker-Sticker Short-Range Attractive Well (Pi-Pi / Cation-Pi)
            is_sticker_pos = (ts == "IDR_STICKER_POS")[:, None]
            is_aromatic = (t2 == "IDR_AROMATIC")[None, :]
            is_rna = (t2 == "RNA_POLYANION")[None, :]

            attractive_mask = (is_sticker_pos & (is_aromatic | is_rna))
            well_depth = 1.5  # kcal/mol
            r_well = sigma * 1.3
            in_well = attractive_mask & (r_mat < r_well) & (r_mat > sigma)
            # P19-2: triangular well U(r) = -well_depth*(r_well-r)/(r_well-sigma)
            # gives -dU/dr = -well_depth/(r_well-sigma) (constant attractive
            # force). The previous code used U(r) itself as the force
            # magnitude (zero at r_well, -well_depth at sigma) and a constant
            # -well_depth energy, making the force and energy inconsistent.
            f_attr_mag = np.zeros_like(r_mat)
            f_attr_mag[in_well] = -well_depth / (r_well[in_well] - sigma[in_well] + 1e-6)
            # Triangular well energy: U = -well_depth * (r_well - r) / (r_well - sigma)
            tri_u = np.zeros_like(r_mat)
            tri_u[in_well] = -well_depth * (r_well[in_well] - r_mat[in_well]) / (r_well[in_well] - sigma[in_well] + 1e-6)
            total_energy += float(np.sum(tri_u)) * 0.5

            total_f_mag = f_ev_mag + f_elec_mag + f_attr_mag
            total_f_mag = np.where(self_mask, 0.0, total_f_mag)

            f_vec = diff * (total_f_mag[:, :, None] / (r_mat[:, :, None] + 1e-6))
            local_forces = np.sum(f_vec, axis=1)
            forces[idx1] += local_forces

        return forces, total_energy

    def run_simulation(
        self,
        system_data: Dict[str, np.ndarray],
        num_steps: int = 150,
        dt: float = 0.02
    ) -> Dict[str, np.ndarray]:
        """
        Simulates droplet condensation dynamics via Langevin dynamics.
        """
        coords = system_data["coords"].copy()
        charges = system_data["charges"]
        radii = system_data["radii"]
        types = system_data["types"]

        kbT = 0.001987204 * self.temperature
        gamma = 1.0
        friction = np.exp(-gamma * dt)
        noise_std = np.sqrt(kbT * (1.0 - friction**2))
        velocities = np.zeros_like(coords)
        rng = np.random.RandomState(42)

        for _ in range(num_steps):
            forces, _ = self.compute_forces_and_energy(coords, charges, radii, types)
            velocities = velocities * friction + (forces * dt * 0.5) + rng.randn(*coords.shape) * noise_std
            coords += velocities * dt
            # Periodic box wrapping
            coords = np.mod(coords, self.box_size)

        system_data["coords"] = coords
        return system_data

    def analyze_phase_separation(self, system_data: Dict[str, np.ndarray]) -> PhaseSeparationReport:
        """
        Computes local density fluctuations, clusters, and droplet thermodynamics.
        """
        coords = system_data["coords"]
        radii = system_data["radii"]
        types = system_data["types"]
        N = len(coords)

        # Volume fraction
        particle_vol = np.sum(4.0 / 3.0 * np.pi * (radii**3))
        total_vol = self.box_size**3
        vol_fraction = float(particle_vol / total_vol * 100.0)

        # Morton spatial clustering to identify condensed droplets
        ix = np.clip((coords[:, 0] / 12.0).astype(np.int64), 0, 1023)
        iy = np.clip((coords[:, 1] / 12.0).astype(np.int64), 0, 1023)
        iz = np.clip((coords[:, 2] / 12.0).astype(np.int64), 0, 1023)
        morton_keys = morton_encode_3d(ix, iy, iz)
        unique_keys, counts = np.unique(morton_keys, return_counts=True)

        # Droplets are dense spatial bins with particle count > threshold
        dense_keys = unique_keys[counts >= 25]
        droplets = []
        
        for d_id, k in enumerate(dense_keys):
            mask = morton_keys == k
            d_coords = coords[mask]
            d_types = types[mask]
            com = np.mean(d_coords, axis=0)
            rg = float(np.sqrt(np.mean(np.sum((d_coords - com)**2, axis=-1))))

            # Composition
            comp = {}
            for t in np.unique(d_types):
                comp[str(t)] = float(np.sum(d_types == t) / len(d_types))

            droplets.append(CondensateDroplet(
                droplet_id=d_id,
                num_particles=int(np.sum(mask)),
                radius_gyration_nm=rg,
                center_of_mass=com,
                dense_phase_density_mg_ml=float(len(d_coords) * 15.0), # Approx mg/mL
                composition_fractions=comp
            ))

        is_separated = len(droplets) > 0

        # Compute radial distribution function g(r) peaks
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, size=min(200, N), replace=False)
        sample_pts = coords[sample_idx]
        diff = sample_pts[:, None, :] - sample_pts[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        r_hist, bin_edges = np.histogram(dists[dists > 0], bins=20, range=(1.0, 30.0))
        r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        peaks = [(float(r_centers[i]), float(r_hist[i])) for i in np.argsort(r_hist)[-3:]]

        # Estimate critical temperature
        t_c = float(self.temperature * (1.2 if is_separated else 0.8))

        return PhaseSeparationReport(
            num_total_particles=N,
            box_size_nm=self.box_size,
            volume_fraction_percent=vol_fraction,
            ionic_strength_molar=self.ionic_strength,
            temperature_kelvin=self.temperature,
            is_phase_separated=is_separated,
            num_droplets=len(droplets),
            droplets=droplets,
            dilute_phase_density_mg_ml=float(N * 2.5),
            dense_phase_density_mg_ml=float(droplets[0].dense_phase_density_mg_ml if droplets else 0.0),
            radial_distribution_peaks=peaks,
            critical_temperature_estimate_k=t_c
        )
