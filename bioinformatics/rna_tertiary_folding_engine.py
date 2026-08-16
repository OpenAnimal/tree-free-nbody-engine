"""
Module 6: RNA 3D Tertiary Folding, Riboswitch Conformational Switching & Polyanionic Counterion Condensation.
Models polyanionic RNA backbone (-1e/nt), divalent Mg2+ Manning counterion condensation,
base stacking / Watson-Crick-Wobble hydrogen bonding, and ligand-induced riboswitch switching.
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
class RiboswitchState:
    """Characterization of a riboswitch conformational state (Aptamer vs Expression Platform)."""
    state_name: str                # e.g., "Aptamer_Bound_Active", "Aptamer_Unbound_Apo"
    total_energy_kcal_mol: float
    electrostatic_energy_kcal: float
    base_pairing_energy_kcal: float
    stacking_energy_kcal: float
    manning_mg2_condensed_count: int
    ligand_affinity_kd_nm: float
    transcription_state: str       # "ON (Permissive)", "OFF (Terminated)"


@dataclass
class RNAFoldingResult:
    """Predicted 3D folded RNA conformation and thermodynamic stability."""
    sequence: str
    num_nucleotides: int
    secondary_structure_dotbracket: str
    num_base_pairs: int
    folding_free_energy_kcal_mol: float
    mg2_stabilization_bonus_kcal: float
    riboswitch_switch_ratio: float
    active_transcription_probability: float


class RNATertiaryFoldingEngine:
    """
    O(N) Tree-Free RNA 3D Folding & Counterion Condensation Engine.
    Combines Manning ion condensation theory, screened phosphate repulsion, and base-pair networks.
    """
    NUCLEOTIDES = ["A", "U", "G", "C"]
    
    # Base-pairing energies (kcal/mol)
    PAIR_ENERGIES = {
        ("G", "C"): -3.0, ("C", "G"): -3.0,
        ("A", "U"): -2.0, ("U", "A"): -2.0,
        ("G", "U"): -1.0, ("U", "G"): -1.0, # Wobble pair
    }

    def __init__(
        self,
        mg2_concentration_mm: float = 2.0,   # Millimolar Mg2+
        k_plus_concentration_mm: float = 100.0, # Millimolar K+
        temperature_kelvin: float = 310.15,
        cell_size: float = 6.0
    ):
        self.mg_conc = float(mg2_concentration_mm)
        self.k_conc = float(k_plus_concentration_mm)
        self.temperature = float(temperature_kelvin)
        self.cell_size = float(cell_size)

        # Ionic strength: I = 0.5 * sum(c_i * z_i^2)
        # Mg2+ (z=2, z^2=4), K+ (z=1, z^2=1), Cl- (balance)
        # I in Molar:
        self.ionic_strength = (0.5 * (self.mg_conc * 1e-3 * 4 + self.k_conc * 1e-3 * 1 + (2 * self.mg_conc + self.k_conc) * 1e-3 * 1))
        self.kappa = float(0.329 * np.sqrt(max(1e-4, self.ionic_strength)))
        self.dielectric = 78.5

        # Manning condensation parameter: xi = e^2 / (4*pi*eps*kT*b)
        # For single-stranded/duplex RNA, b ~ 0.28 nm (2.8 A) -> xi ~ 2.5 > 1.0 (strong condensation)
        self.manning_xi = 2.5
        # Net effective charge fraction after counterion condensation: 1 / (z * xi)
        self.effective_charge_fraction = 1.0 / (2.0 * self.manning_xi) # ~0.20 of bare charge

        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=self.kappa,
            dielectric_water=self.dielectric,
            dielectric_protein=4.0,
            kernel_type=ScreenedKernelType.DEBYE_HUCKEL
        )

    def build_coarse_rna_system(self, sequence: str) -> MolecularSystem:
        """
        Builds coarse-grained 3-bead-per-nucleotide RNA model:
        Phosphate (P: -1e), Ribose Sugar (C4'), Nucleobase (N1/N9).
        """
        seq = sequence.upper().replace("T", "U")
        n_nt = len(seq)
        coords = []
        charges = []
        radii = []
        masses = []
        atom_names = []
        res_names = []
        res_ids = []
        chain_ids = []

        # A-form RNA helix parameters: rise = 2.8 A, twist = 32.7 deg, radius = 9.0 A
        rise = 2.8
        twist = np.deg2rad(32.7)
        r_backbone = 9.0
        r_base = 4.5

        for i, nt in enumerate(seq):
            angle = i * twist
            z = i * rise

            # 1. Phosphate bead (-1.0e bare, condensed with Mg2+)
            px = r_backbone * np.cos(angle)
            py = r_backbone * np.sin(angle)
            pz = z
            coords.append([px, py, pz])
            charges.append(-1.0 * self.effective_charge_fraction)
            radii.append(2.0)
            masses.append(94.97) # PO4
            atom_names.append("P")
            res_names.append(f"R{nt}")
            res_ids.append(i + 1)
            chain_ids.append("R")

            # 2. Sugar bead (neutral)
            sx = (r_backbone - 2.0) * np.cos(angle + 0.1)
            sy = (r_backbone - 2.0) * np.sin(angle + 0.1)
            sz = z + 0.5
            coords.append([sx, sy, sz])
            charges.append(0.0)
            radii.append(1.8)
            masses.append(133.1) # C5H9O4
            atom_names.append("C4'")
            res_names.append(f"R{nt}")
            res_ids.append(i + 1)
            chain_ids.append("R")

            # 3. Base bead (partial dipolar charges)
            bx = r_base * np.cos(angle + 0.25)
            by = r_base * np.sin(angle + 0.25)
            bz = z + 0.8
            coords.append([bx, by, bz])
            charges.append(0.15 if nt in ["A", "C"] else -0.15)
            radii.append(2.2)
            masses.append(120.0)
            atom_names.append("BASE")
            res_names.append(f"R{nt}")
            res_ids.append(i + 1)
            chain_ids.append("R")

        return MolecularSystem(
            coords=np.array(coords, dtype=np.float64),
            charges=np.array(charges, dtype=np.float64),
            radii=np.array(radii, dtype=np.float64),
            masses=np.array(masses, dtype=np.float64),
            atom_names=atom_names,
            residue_names=res_names,
            residue_ids=np.array(res_ids, dtype=np.int32),
            chain_ids=chain_ids,
            system_name="RNA_Transcript"
        )

    def predict_secondary_structure(self, sequence: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Nussinov-style dynamic programming predicting optimal base-pairing dot-bracket notation.
        """
        seq = sequence.upper().replace("T", "U")
        n = len(seq)
        dp = np.zeros((n, n), dtype=np.float64)

        # Minimum hairpin loop length = 3
        for length in range(4, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Case 1: i is unpaired
                max_score = dp[i + 1, j]
                # Case 2: j is unpaired
                if dp[i, j - 1] > max_score:
                    max_score = dp[i, j - 1]

                # Case 3: i and j pair
                pair = (seq[i], seq[j])
                if pair in self.PAIR_ENERGIES:
                    score = dp[i + 1, j - 1] - self.PAIR_ENERGIES[pair] # Lower energy = higher score
                    if score > max_score:
                        max_score = score

                # Case 4: Bifurcation
                for k in range(i + 1, j):
                    bif = dp[i, k] + dp[k + 1, j]
                    if bif > max_score:
                        max_score = bif

                dp[i, j] = max_score

        # Traceback to extract base pairs
        pairs = []
        def traceback(i, j):
            if i >= j:
                return
            if dp[i, j] == dp[i + 1, j]:
                traceback(i + 1, j)
            elif dp[i, j] == dp[i, j - 1]:
                traceback(i, j - 1)
            elif (seq[i], seq[j]) in self.PAIR_ENERGIES and np.isclose(dp[i, j], dp[i + 1, j - 1] - self.PAIR_ENERGIES[(seq[i], seq[j])]):
                pairs.append((i, j))
                traceback(i + 1, j - 1)
            else:
                for k in range(i + 1, j):
                    if np.isclose(dp[i, j], dp[i, k] + dp[k + 1, j]):
                        traceback(i, k)
                        traceback(k + 1, j)
                        break

        traceback(0, n - 1)

        # Build dot-bracket string
        db = ["."] * n
        for (i, j) in pairs:
            db[i] = "("
            db[j] = ")"

        return "".join(db), sorted(pairs)

    def evaluate_riboswitch_switching(
        self,
        sequence: str,
        ligand_name: str = "Theophylline",
        ligand_concentration_um: float = 10.0
    ) -> Dict[str, Union[RiboswitchState, float]]:
        """
        Evaluates thermodynamic bistability of a riboswitch in the presence vs absence of target ligand.
        """
        # Apo (unbound) vs Holo (ligand-bound) conformational free energies
        dotbracket, pairs = self.predict_secondary_structure(sequence)
        
        # Base pairing free energy
        seq = sequence.upper().replace("T", "U")
        e_bp = float(sum(self.PAIR_ENERGIES.get((seq[i], seq[j]), -1.5) for i, j in pairs))
        e_stack = -1.2 * max(0, len(pairs) - 1) # Stacking bonus

        # Electrostatic screening by Mg2+
        mg_bonus = -0.8 * np.log(1.0 + self.mg_conc) * len(sequence)

        # Apo state free energy
        g_apo = e_bp + e_stack + mg_bonus

        # Holo state: ligand binds aptamer pocket with favorable electrostatic/H-bond delta
        ligand_binding_delta = -8.5 # kcal/mol (typical micromolar riboswitch binding)
        g_holo = g_apo + ligand_binding_delta

        # Switching ratio and transcription state
        kbT = 0.0019872041 * self.temperature
        exp_arg = float(np.clip(ligand_binding_delta / kbT, -30.0, 30.0))
        kd_nm = float(np.exp(exp_arg) * 1e9) # in nM
        
        # Fractional bound occupancy: [L] / ([L] + Kd)
        c_lig_nm = ligand_concentration_um * 1000.0
        bound_fraction = float(c_lig_nm / max(1e-9, c_lig_nm + kd_nm))

        apo_state = RiboswitchState(
            state_name="Apo_Unbound",
            total_energy_kcal_mol=g_apo,
            electrostatic_energy_kcal=mg_bonus,
            base_pairing_energy_kcal=e_bp,
            stacking_energy_kcal=e_stack,
            manning_mg2_condensed_count=int(len(sequence) * (1.0 - self.effective_charge_fraction) * 0.5),
            ligand_affinity_kd_nm=kd_nm,
            transcription_state="ON (Permissive)" if bound_fraction < 0.5 else "OFF"
        )

        holo_state = RiboswitchState(
            state_name="Holo_Ligand_Bound",
            total_energy_kcal_mol=g_holo,
            electrostatic_energy_kcal=mg_bonus - 2.5,
            base_pairing_energy_kcal=e_bp - 3.0,
            stacking_energy_kcal=e_stack - 1.5,
            manning_mg2_condensed_count=int(len(sequence) * (1.0 - self.effective_charge_fraction) * 0.5) + 2,
            ligand_affinity_kd_nm=kd_nm,
            transcription_state="OFF (Terminated)"
        )

        return {
            "apo_state": apo_state,
            "holo_state": holo_state,
            "ligand_bound_fraction": bound_fraction,
            "switching_ratio": float(np.exp(-ligand_binding_delta / kbT)),
            "delta_G_switching_kcal_mol": ligand_binding_delta,
            "dotbracket": dotbracket
        }
