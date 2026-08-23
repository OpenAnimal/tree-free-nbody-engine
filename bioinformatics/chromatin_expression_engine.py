"""
Module 2: 3D Chromatin Architecture, Polymer Dynamics & Non-Coding Variant Target Expression Engine.
Connects 1D genomic k-mer sequence motifs (via Farach-Colton Elastic Hashing) with coarse-grained
3D polyanionic chromatin polymer dynamics and Debye-Huckel electrostatics to predict in silico Hi-C
contact maps and non-coding SNP enhancer-promoter expression shifts.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .kmer_elastic_hash import KmerElasticHashTable
    from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d
    from .pdb_loader import COULOMB_CONSTANT_KCAL
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.kmer_elastic_hash import KmerElasticHashTable
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d
    from bioinformatics.pdb_loader import COULOMB_CONSTANT_KCAL


@dataclass
class GenomicLocus:
    """Represents a coarse-grained chromatin bead (e.g. 5kb - 25kb)."""
    locus_id: int
    chrom: str
    start_bp: int
    end_bp: int
    annotation: str          # "Promoter", "Enhancer", "CTCF_Boundary", "Heterochromatin", "Neutral"
    target_gene: Optional[str] = None
    epigenetic_state: str = "H3K27ac"  # e.g., "H3K27ac", "H3K4me3", "H3K27me3", "CTCF"
    net_charge: float = -20.0          # Net charge in elementary charges (e) per coarse bead


@dataclass
class ExpressionPrediction:
    """Predicted impact of a non-coding mutation on 3D chromatin looping and target gene expression."""
    variant_id: str
    chrom: str
    position_bp: int
    disrupted_motif: str
    target_gene: str
    contact_prob_wt: float
    contact_prob_mut: float
    delta_contact_prob: float
    log2_fold_change: float
    regulatory_impact: str    # "Down-Regulated (Silenced)", "Up-Regulated (De-repressed)", "Neutral"
    mechanism: str


class ChromatinPolymerModel:
    """
    3D Coarse-Grained Chromatin Polymer with Screened Debye-Huckel Electrostatics.
    """
    def __init__(
        self,
        loci: List[GenomicLocus],
        persistence_length_nm: float = 50.0,
        bead_radius_nm: float = 15.0,
        ionic_strength_molar: float = 0.15,
        temperature_kelvin: float = 310.15
    ):
        self.loci = loci
        self.num_beads = len(loci)
        self.bead_radius = float(bead_radius_nm)
        self.r0 = 2.0 * self.bead_radius # Equilibrium bond distance
        self.k_bond = 100.0              # Harmonic spring constant kcal/(mol * nm^2)
        
        # Debye screening
        self.ionic_strength = float(ionic_strength_molar)
        # kappa in 1/nm (0.329 * sqrt(I) * 10 for nm)
        self.kappa = float(3.29 * np.sqrt(self.ionic_strength))
        self.dielectric = 78.5
        self.temperature = float(temperature_kelvin)

        # Initialize beads along a self-avoiding random walk
        self.coords = self._initialize_polymer_walk()
        self.charges = np.array([locus.net_charge for locus in self.loci], dtype=np.float64)
        self.velocities = np.zeros_like(self.coords)

        # Active loop extrusion anchors (e.g. CTCF pairs)
        self.ctcf_anchors: List[Tuple[int, int]] = self._detect_ctcf_loops()

    def _initialize_polymer_walk(self) -> np.ndarray:
        """Initializes a compact self-avoiding chromatin trajectory."""
        coords = np.zeros((self.num_beads, 3), dtype=np.float64)
        rng = np.random.RandomState(42)
        for i in range(1, self.num_beads):
            step = rng.randn(3)
            step /= np.linalg.norm(step) + 1e-9
            coords[i] = coords[i - 1] + step * self.r0
        return coords

    def _detect_ctcf_loops(self) -> List[Tuple[int, int]]:
        """Identifies convergent CTCF boundary pairs to model loop extrusion."""
        ctcf_indices = [i for i, l in enumerate(self.loci) if "CTCF" in l.annotation or "CTCF" in l.epigenetic_state]
        loops = []
        for idx in range(len(ctcf_indices) - 1):
            i = ctcf_indices[idx]
            j = ctcf_indices[idx + 1]
            if 2 <= (j - i) <= 50:
                loops.append((i, j))
        return loops

    def compute_forces(self) -> Tuple[np.ndarray, float]:
        """
        Computes total polymer forces: Harmonic backbone bonds + Debye-Huckel screened
        polyanionic electrostatics + soft-core excluded volume + CTCF loop extrusion.
        """
        forces = np.zeros_like(self.coords)
        total_energy = 0.0

        # 1. Harmonic backbone bonds
        # F_i = -dU/dr_i with U = 0.5*k*(d-r0)^2, d = |r_{i+1}-r_i|.
        # dd/dr_i = -(r_{i+1}-r_i)/d, so F_i = k*(d-r0)*(r_{i+1}-r_i)/d
        # (attractive when stretched, repulsive when compressed).
        # P23-2: the previous code used -k*delta*diff_bonds/dist, which with
        # diff_bonds = r_{i+1}-r_i gives a REPULSIVE force when stretched
        # (the chain flies apart). The sign is now corrected to +k*delta.
        diff_bonds = self.coords[1:] - self.coords[:-1]
        dist_bonds = np.linalg.norm(diff_bonds, axis=-1, keepdims=True) + 1e-9
        delta = dist_bonds - self.r0
        f_bonds = self.k_bond * delta * (diff_bonds / dist_bonds)

        forces[:-1] += f_bonds
        forces[1:] -= f_bonds
        total_energy += 0.5 * self.k_bond * np.sum(delta**2)

        # 2. CTCF Loop Extrusion Harmonic Constraints
        for (i, j) in self.ctcf_anchors:
            diff_loop = self.coords[j] - self.coords[i]
            d_loop = np.linalg.norm(diff_loop) + 1e-9
            f_loop = -self.k_bond * 0.5 * (d_loop - self.r0) * (diff_loop / d_loop)
            forces[i] -= f_loop
            forces[j] += f_loop
            total_energy += 0.25 * self.k_bond * (d_loop - self.r0)**2

        # 3. All-pairs Screened Electrostatics & Excluded Volume
        # Distance matrix
        diff_all = self.coords[:, None, :] - self.coords[None, :, :]
        r_mat = np.linalg.norm(diff_all, axis=-1) + np.eye(self.num_beads) * 1e9

        # Excluded volume (Soft-core repulsion)
        # U_ev = 50/3 * overlap^3, F = -dU/dr = 50 * overlap^2 (repulsive).
        # P21-2: the previous code divided f_ev_mag by r, giving a force
        # 50*overlap^2/r instead of 50*overlap^2 — too weak by a factor of r.
        # Same bug pattern as P19-1 in biomolecular_condensate_engine.py.
        sigma = self.r0
        overlap = np.maximum(0.0, sigma - r_mat)
        f_ev_mag = 50.0 * (overlap**2)
        total_energy += float(np.sum(50.0 / 3.0 * (overlap**3))) * 0.5

        # Screened Coulomb: V(r) = q1 q2 exp(-kappa r) / (eps r)
        coulomb_factor = COULOMB_CONSTANT_KCAL / self.dielectric
        screened_pot = (self.charges[:, None] * self.charges[None, :]) * np.exp(-self.kappa * r_mat) / (r_mat + 1e-6)
        total_energy += float(np.sum(screened_pot)) * 0.5 * coulomb_factor

        f_elec_mag = coulomb_factor * screened_pot * (self.kappa + 1.0 / (r_mat + 1e-6))

        total_f_mag = f_ev_mag + f_elec_mag
        # Zero out self and adjacent bonded beads
        np.fill_diagonal(total_f_mag, 0.0)
        for i in range(self.num_beads - 1):
            total_f_mag[i, i + 1] = 0.0
            total_f_mag[i + 1, i] = 0.0

        f_vec = diff_all * (total_f_mag[:, :, None] / (r_mat[:, :, None] + 1e-6))
        forces += np.sum(f_vec, axis=1)

        return forces, total_energy

    def run_langevin_dynamics(self, num_steps: int = 200, dt: float = 0.01, gamma: float = 1.0) -> None:
        """Runs Langevin thermostat polymer relaxation steps."""
        kbT = 0.001987204 * self.temperature # kcal/mol
        friction = np.exp(-gamma * dt)
        noise_std = np.sqrt(kbT * (1.0 - friction**2))
        rng = np.random.RandomState(42)

        for _ in range(num_steps):
            forces, _ = self.compute_forces()
            self.velocities = self.velocities * friction + (forces * dt * 0.5) + rng.randn(*self.coords.shape) * noise_std
            self.coords += self.velocities * dt


class ChromatinExpressionEngine:
    """
    High-Throughput 3D Chromatin Looping and Gene Expression Predictor.
    Utilizes Farach-Colton k-mer motif hashing + 3D polymer electrostatics.
    """
    def __init__(self, kmer_size: int = 15, cell_size_nm: float = 30.0):
        self.kmer_size = kmer_size
        self.kmer_table = KmerElasticHashTable(k=kmer_size, capacity=500000)
        self.cell_size = cell_size_nm

        # Pre-seed canonical regulatory motifs
        self.known_motifs = {
            "CTCF": "CCACCAGGTGGCG",
            "TATA_BOX": "TATAAA",
            "ENHANCER_AP1": "TGACTCA",
            "ENHANCER_NFKB": "GGGACTTTCC"
        }

    def build_synthetic_chromatin_domain(
        self,
        domain_kb: int = 500,
        resolution_kb: int = 5,
        target_gene: str = "MYC"
    ) -> ChromatinPolymerModel:
        """
        Builds a coarse-grained model of an oncogenic locus (e.g. 8q24 super-enhancer / MYC).
        """
        num_beads = domain_kb // resolution_kb
        loci = []
        
        # Place promoter at 20% mark, super-enhancer at 70% mark, CTCF boundaries at ends
        promoter_idx = int(num_beads * 0.2)
        enhancer_idx = int(num_beads * 0.7)
        ctcf_1_idx = 2
        ctcf_2_idx = num_beads - 3

        for i in range(num_beads):
            start_bp = i * resolution_kb * 1000
            end_bp = (i + 1) * resolution_kb * 1000

            if i == promoter_idx:
                annot = "Promoter"
                state = "H3K4me3"
                charge = -15.0
                gene = target_gene
            elif i in [enhancer_idx, enhancer_idx + 1, enhancer_idx + 2]:
                annot = "Enhancer"
                state = "H3K27ac"
                charge = -25.0
                gene = target_gene
            elif i in [ctcf_1_idx, ctcf_2_idx]:
                annot = "CTCF_Boundary"
                state = "CTCF"
                charge = -10.0
                gene = None
            else:
                annot = "Neutral"
                state = "Heterochromatin" if (i < promoter_idx or i > enhancer_idx + 2) else "Euchromatin"
                charge = -20.0
                gene = None

            loci.append(GenomicLocus(
                locus_id=i,
                chrom="chr8",
                start_bp=start_bp,
                end_bp=end_bp,
                annotation=annot,
                target_gene=gene,
                epigenetic_state=state,
                net_charge=charge
            ))

        model = ChromatinPolymerModel(loci)
        # Equilibrate polymer
        model.run_langevin_dynamics(num_steps=150)
        return model

    def compute_in_silico_hic_map(self, model: ChromatinPolymerModel, cutoff_nm: float = 60.0) -> np.ndarray:
        """
        Computes 3D contact frequency map (in silico Hi-C) from relaxed polymer coordinates.
        Uses Gaussian contact probability: P_ij = exp(-r_ij^2 / (2 * cutoff^2)).
        """
        diff = model.coords[:, None, :] - model.coords[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        contact_matrix = np.exp(-dist_sq / (2.0 * (cutoff_nm**2)))
        return contact_matrix

    def evaluate_noncoding_variant(
        self,
        model: ChromatinPolymerModel,
        variant_bp: int,
        ref_sequence: str,
        alt_sequence: str,
        target_gene: str = "MYC"
    ) -> ExpressionPrediction:
        """
        Evaluates the functional consequence of a non-coding SNP on enhancer-promoter looping
        and target gene expression.
        """
        # Identify locus corresponding to variant position
        affected_idx = None
        for i, locus in enumerate(model.loci):
            if locus.start_bp <= variant_bp < locus.end_bp:
                affected_idx = i
                break

        if affected_idx is None:
            affected_idx = 0

        # Baseline wild-type contact matrix
        hic_wt = self.compute_in_silico_hic_map(model)

        # Locate promoter and enhancer beads for target gene
        promoter_idx = next((i for i, l in enumerate(model.loci) if l.annotation == "Promoter" and l.target_gene == target_gene), 0)
        enhancer_indices = [i for i, l in enumerate(model.loci) if l.annotation == "Enhancer" and l.target_gene == target_gene]

        if not enhancer_indices:
            enhancer_indices = [min(model.num_beads - 1, promoter_idx + 10)]

        p_wt = float(np.mean([hic_wt[promoter_idx, e_idx] for e_idx in enhancer_indices]))

        # Simulate mutation effect on chromatin model
        mut_model = ChromatinPolymerModel(list(model.loci))
        mut_model.coords = model.coords.copy()

        # Check if mutation disrupts CTCF boundary or Enhancer activity
        disrupted_motif = "None"
        if model.loci[affected_idx].annotation == "CTCF_Boundary":
            disrupted_motif = "CTCF_Insulation_Motif"
            # Remove CTCF loop anchor
            mut_model.ctcf_anchors = [loop for loop in mut_model.ctcf_anchors if loop[0] != affected_idx and loop[1] != affected_idx]
            # Relaxation steps under mutant topology
            mut_model.run_langevin_dynamics(num_steps=100)
        elif model.loci[affected_idx].annotation == "Enhancer":
            disrupted_motif = "Enhancer_Activator_Motif"
            # Decrease negative charge density (loss of transcription factor / coactivator recruitment)
            mut_model.charges[affected_idx] *= 0.3
            mut_model.run_langevin_dynamics(num_steps=100)
        else:
            mut_model.run_langevin_dynamics(num_steps=50)

        # Mutant contact matrix
        hic_mut = self.compute_in_silico_hic_map(mut_model)
        p_mut = float(np.mean([hic_mut[promoter_idx, e_idx] for e_idx in enhancer_indices]))

        delta_p = p_mut - p_wt
        # Log2 fold change in gene expression proportional to contact ratio
        ratio = max(0.01, p_mut / max(1e-5, p_wt))
        log2_fc = float(np.log2(ratio))

        if log2_fc < -0.5:
            impact = "Down-Regulated (Silenced)"
            mechanism = f"Variant disrupts {disrupted_motif}, abolishing enhancer-promoter looping to {target_gene}."
        elif log2_fc > 0.5:
            impact = "Up-Regulated (De-repressed)"
            mechanism = f"Variant disrupts insulator boundary, causing aberrant enhancer invasion and activation of {target_gene}."
        else:
            impact = "Neutral"
            mechanism = "Minimal perturbation to 3D loop topology."

        return ExpressionPrediction(
            variant_id=f"chr8:{variant_bp}_{ref_sequence}>{alt_sequence}",
            chrom="chr8",
            position_bp=variant_bp,
            disrupted_motif=disrupted_motif,
            target_gene=target_gene,
            contact_prob_wt=p_wt,
            contact_prob_mut=p_mut,
            delta_contact_prob=delta_p,
            log2_fold_change=log2_fc,
            regulatory_impact=impact,
            mechanism=mechanism
        )
