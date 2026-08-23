"""
Module 7: TCR-pMHC Neoantigen Immunogenicity, CAR-T/TCR-T Affinity & Off-Target Cross-Reactivity Filter.
Simulates T-cell receptor (CDR3 loops) engagement with peptide-MHC complexes (HLA-A*02:01),
predicts neoantigen immunogenicity, and screens for off-target cross-reactivity against human self-peptides.

Synthetic research prototype on self-generated data; not clinical; not diagnostic;
no real patient or guideline data consumed.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, RESIDUE_PROPERTIES
    from .solvation_free_energy import SolvationFreeEnergyEngine
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, RESIDUE_PROPERTIES
    from bioinformatics.solvation_free_energy import SolvationFreeEnergyEngine
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D


@dataclass
class TCRBindingProfile:
    """Quantitative recognition metrics of a TCR engaging a target peptide-MHC complex."""
    tcr_name: str
    hla_allele: str               # e.g., "HLA-A*02:01"
    peptide_sequence: str         # e.g., "VVGADGVGK" (KRAS G12D neoantigen)
    is_neoantigen: bool
    binding_affinity_dg_kcal: float
    kd_micromolar: float
    cdr3_contact_count: int
    electrostatic_complementarity: float # -1.0 (clashing) to +1.0 (ideal)
    immunogenicity_score: float   # 0 to 100
    activation_potential: str     # "Potent Cytotoxic Activation", "Moderate", "Weak / Anergic"


@dataclass
class OffTargetSafetyReport:
    """Off-target cross-reactivity safety assessment against normal human self-peptidome."""
    tcr_name: str
    target_neoantigen: str
    num_self_peptides_screened: int
    num_cross_reactive_hits: int
    top_cross_reactive_peptides: List[Tuple[str, str, float]] # (peptide, protein_source, delta_affinity)
    cross_reactivity_risk: str    # "Safe (Low Risk)", "Moderate Caution", "High Off-Target Risk (Lethal Warning)"
    safety_confidence_score: float # 0.0 to 1.0


class TCRpMHCImmunogenicityEngine:
    """
    State-of-the-art TCR-pMHC Recognition and Autoimmune Cross-Reactivity Scanner.
    """
    def __init__(
        self,
        dielectric_water: float = 78.5,
        dielectric_protein: float = 4.0,
        ionic_strength_molar: float = 0.15,
        cell_size: float = 6.0
    ):
        self.solvation_engine = SolvationFreeEnergyEngine(
            dielectric_water=dielectric_water,
            dielectric_protein=dielectric_protein,
            ionic_strength_molar=ionic_strength_molar,
            cell_size=cell_size
        )
        self.cell_size = float(cell_size)

    def evaluate_tcr_pmhc_binding(
        self,
        tcr_cdr3_seq: str,
        peptide_seq: str,
        hla_allele: str = "HLA-A*02:01",
        is_neoantigen: bool = True
    ) -> TCRBindingProfile:
        """
        Evaluates physical binding affinity, CDR3 loop contacts, and immunogenicity score.
        """
        tcr_len = len(tcr_cdr3_seq)
        pep_len = len(peptide_seq)

        # Peptide net charges and hydrophobicity
        pep_charge = sum(RESIDUE_PROPERTIES.get(r, {}).get("net_charge", 0.0) for r in peptide_seq)
        cdr3_charge = sum(RESIDUE_PROPERTIES.get(r, {}).get("net_charge", 0.0) for r in tcr_cdr3_seq)

        # Electrostatic complementarity: opposite charges attract favorably
        charge_prod = pep_charge * cdr3_charge
        e_elec = -2.5 * charge_prod / (abs(charge_prod) + 1.0) # Favorable if charge_prod < 0

        # Sequence alignment & interface contact estimation
        contact_count = 0
        hbond_bonus = 0.0
        hydrophobic_res = {"L", "I", "V", "F", "W", "Y", "M", "P"}

        for i, p_aa in enumerate(peptide_seq):
            # Middle residues (P4 - P8) are upward-facing TCR contact positions
            if 3 <= i <= 7:
                match_tcr_idx = min(tcr_len - 1, i)
                t_aa = tcr_cdr3_seq[match_tcr_idx]
                contact_count += 1
                if p_aa in hydrophobic_res and t_aa in hydrophobic_res:
                    hbond_bonus -= 1.2
                elif p_aa == t_aa:
                    hbond_bonus -= 0.8

        # Total binding energy: baseline ~ -6.5 kcal/mol + electrostatic + hbond
        dg_bind = -6.5 + float(e_elec) + float(hbond_bonus)
        
        # Kd in micromolar at 310K
        kbT = 0.0019872041 * 310.15
        exp_arg = float(np.clip(dg_bind / kbT, -30.0, 30.0))
        kd_um = float(np.exp(exp_arg) * 1e6)

        # Electrostatic complementarity normalized [-1, 1]
        comp = float(np.clip(-charge_prod / 4.0, -1.0, 1.0))

        # Immunogenicity score: affinity + contact density
        score = float(np.clip((-dg_bind / 12.0) * 70.0 + (contact_count / 5.0) * 30.0, 5.0, 99.0))

        if dg_bind <= -8.0 and kd_um < 50.0:
            act = "Potent Cytotoxic Activation"
        elif dg_bind <= -6.0:
            act = "Moderate"
        else:
            act = "Weak / Anergic"

        return TCRBindingProfile(
            tcr_name=f"TCR_CDR3_{tcr_cdr3_seq[:6]}",
            hla_allele=hla_allele,
            peptide_sequence=peptide_seq,
            is_neoantigen=is_neoantigen,
            binding_affinity_dg_kcal=dg_bind,
            kd_micromolar=kd_um,
            cdr3_contact_count=contact_count,
            electrostatic_complementarity=comp,
            immunogenicity_score=score,
            activation_potential=act
        )

    def screen_off_target_cross_reactivity(
        self,
        tcr_cdr3_seq: str,
        target_neoantigen: str,
        self_peptidome: Optional[List[Tuple[str, str]]] = None # [(peptide_seq, source_protein)]
    ) -> OffTargetSafetyReport:
        """
        High-throughput safety screen of a candidate TCR against the human self-peptidome
        to detect life-threatening off-target cross-reactivity (e.g. cardiac Titin, neural MAGE).
        """
        # If no library supplied, build representative human self-peptide panel
        if self_peptidome is None:
            self_peptidome = [
                ("VVGAVGVGK", "KRAS_WildType"),
                ("MLWGYLQYV", "TITIN_Cardiac"),
                ("KVLEHVVRV", "MAGEA3_Melanoma"),
                ("EAAGIGILTV", "MART1_Melanoma"),
                ("LLFGYPVYV", "HTLV1_Viral"),
                ("SLYNTVATL", "HIV_Gag"),
                ("RMFPNAPYL", "WT1_Wilms_Tumor"),
                ("YLQLVFGIEV", "MAGEA1_Self"),
                ("ALWGPDPAAA", "MYOSIN_Cardiac")
            ]

        target_eval = self.evaluate_tcr_pmhc_binding(tcr_cdr3_seq, target_neoantigen, is_neoantigen=True)
        target_dg = target_eval.binding_affinity_dg_kcal

        hits = []
        for pep, prot in self_peptidome:
            eval_self = self.evaluate_tcr_pmhc_binding(tcr_cdr3_seq, pep, is_neoantigen=False)
            delta_dg = eval_self.binding_affinity_dg_kcal - target_dg # Difference in binding

            # A hit occurs if TCR binds self-peptide with affinity close to target (delta_dg < 1.5 kcal/mol)
            if eval_self.binding_affinity_dg_kcal < -6.5 and delta_dg < 1.5:
                hits.append((pep, prot, float(eval_self.binding_affinity_dg_kcal)))

        hits.sort(key=lambda x: x[2]) # Sort by strongest off-target binder

        if len(hits) == 0:
            risk = "Safe (Low Risk)"
            conf = 0.95
        elif any("TITIN" in h[1] or "MYOSIN" in h[1] for h in hits):
            risk = "High Off-Target Risk (Lethal Warning)"
            conf = 0.90
        else:
            risk = "Moderate Caution"
            conf = 0.80

        return OffTargetSafetyReport(
            tcr_name=f"TCR_{tcr_cdr3_seq[:6]}",
            target_neoantigen=target_neoantigen,
            num_self_peptides_screened=len(self_peptidome),
            num_cross_reactive_hits=len(hits),
            top_cross_reactive_peptides=hits[:5],
            cross_reactivity_risk=risk,
            safety_confidence_score=conf
        )
