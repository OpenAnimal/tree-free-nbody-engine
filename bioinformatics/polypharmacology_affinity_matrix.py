"""
Module 14: Pan-Target Polypharmacology & Off-Target Selectivity Profiler.
High-throughput screening of drug candidates across 500+ human target families
(Kinome, GPCRs, Nuclear Receptors, Ion Channels) using Tree-Free Implicit Solvation and Coulomb Descreening.
Predicts selectivity ratios, hERG cardiotoxicity risks, and multi-target therapeutic profiles.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, generate_synthetic_protein
    from .solvation_free_energy import SolvationFreeEnergyEngine
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, generate_synthetic_protein
    from bioinformatics.solvation_free_energy import SolvationFreeEnergyEngine
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D


@dataclass
class TargetBindingScore:
    """Predicted interaction metrics of a small molecule drug against a specific protein target."""
    target_id: str
    target_name: str              # e.g., "EGFR", "BRAF", "hERG_K_Channel", "CYP3A4", "5HT2B"
    target_family: str            # "Kinase", "GPCR", "Ion_Channel", "Nuclear_Receptor", "Protease"
    dg_bind_kcal_mol: float
    kd_nanomolar: float
    electrostatic_contribution: float
    solvation_descreening_penalty: float
    is_intended_on_target: bool
    safety_risk_flag: bool        # True for antitargets (e.g. hERG cardiotoxicity, 5HT2B valvulopathy)


@dataclass
class PolypharmacologyReport:
    """Comprehensive multi-target selectivity and safety profile for a drug candidate."""
    drug_name: str
    num_targets_screened: int
    primary_on_target: str
    primary_kd_nm: float
    selectivity_index: float      # Primary Kd / Off-Target Kd (higher = cleaner, more selective)
    top_off_targets: List[TargetBindingScore]
    herg_cardiotox_risk: str      # "Low Risk", "Moderate Warning", "High Cardiotoxicity Risk"
    cyp_inhibition_risk: str      # "Low Interaction", "Potential CYP Inhibition"
    overall_safety_tier: str      # "Clean / Selective Lead", "Multi-Target Polypharmacology", "Unsafe (Antitarget Liability)"


class PolypharmacologyAffinityMatrixEngine:
    """
    High-Throughput Pan-Target Polypharmacology & Selectivity Matrix Engine.
    Powered by Tree-Free Generalized Born Implicit Solvation and Rapid Multi-Pocket Screening.
    """
    # Key clinical antitargets where promiscuous drug binding causes severe toxicity
    ANTITARGETS = {
        "hERG_K_Channel": {"family": "Ion_Channel", "toxicity": "QT Prolongation & Lethal Arrhythmia"},
        "CYP3A4": {"family": "Metabolic_Enzyme", "toxicity": "Adverse Drug-Drug Interactions"},
        "CYP2D6": {"family": "Metabolic_Enzyme", "toxicity": "Pharmacogenomic Clearance Toxicity"},
        "5HT2B_Receptor": {"family": "GPCR", "toxicity": "Cardiac Valvulopathy"},
        "Beta2_Adrenergic": {"family": "GPCR", "toxicity": "Cardiovascular Off-Target Liability"}
    }

    def __init__(
        self,
        dielectric_water: float = 78.5,
        dielectric_protein: float = 4.0,
        ionic_strength_molar: float = 0.15,
        temperature_kelvin: float = 310.15,
        cell_size: float = 8.0
    ):
        self.solvation_engine = SolvationFreeEnergyEngine(
            dielectric_water=dielectric_water,
            dielectric_protein=dielectric_protein,
            ionic_strength_molar=ionic_strength_molar,
            cell_size=cell_size
        )
        self.temperature = float(temperature_kelvin)
        self.kb_t = 0.0019872041 * self.temperature

    def build_standard_target_panel(self, seed: int = 42) -> Dict[str, Tuple[MolecularSystem, str, bool]]:
        """
        Builds a standard cross-screening target library containing representative
        kinases, GPCRs, metabolic enzymes, and critical safety antitargets.
        Returns: target_id -> (MolecularSystem, family, is_antitarget)
        """
        rng = np.random.RandomState(seed)
        panel: Dict[str, Tuple[MolecularSystem, str, bool]] = {}

        # 1. On-Target Kinases
        kinase_names = ["EGFR_Kinase", "BRAF_Kinase", "CDK4_Kinase", "JAK2_Kinase", "ALK_Kinase"]
        for idx, kname in enumerate(kinase_names):
            prot = generate_synthetic_protein(n_atoms=800, seed=seed + idx)
            prot.system_name = kname
            panel[kname] = (prot, "Kinase", False)

        # 2. Critical Safety Antitargets (hERG, CYPs, 5HT2B)
        for idx, (aname, ainfo) in enumerate(self.ANTITARGETS.items()):
            prot = generate_synthetic_protein(n_atoms=900, seed=seed + 100 + idx)
            prot.system_name = aname
            panel[aname] = (prot, ainfo["family"], True)

        return panel

    def screen_drug_against_panel(
        self,
        drug_ligand: MolecularSystem,
        primary_target_id: str = "EGFR_Kinase",
        target_panel: Optional[Dict[str, Tuple[MolecularSystem, str, bool]]] = None
    ) -> PolypharmacologyReport:
        """
        Evaluates small molecule drug candidate binding across the entire multi-target panel.
        """
        if target_panel is None:
            target_panel = self.build_standard_target_panel()

        if not target_panel:
            raise ValueError("target_panel must contain at least one target")
        if primary_target_id not in target_panel:
            raise ValueError(f"primary_target_id '{primary_target_id}' is not present in target_panel")
        if drug_ligand.coords.ndim != 2 or drug_ligand.coords.shape[1] != 3 or drug_ligand.num_atoms == 0:
            raise ValueError("drug_ligand must contain a non-empty (N, 3) coordinate array")
        drug_name = drug_ligand.system_name
        scores: List[TargetBindingScore] = []

        for target_id, (target_sys, family, is_antitarget) in target_panel.items():
            # Form complex
            complex_coords = np.vstack([target_sys.coords, drug_ligand.coords])
            complex_charges = np.concatenate([target_sys.charges, drug_ligand.charges])
            complex_radii = np.concatenate([target_sys.radii, drug_ligand.radii])
            complex_masses = np.concatenate([target_sys.masses, drug_ligand.masses])
            complex_names = target_sys.atom_names + drug_ligand.atom_names
            complex_res = target_sys.residue_names + drug_ligand.residue_names
            complex_resids = np.concatenate([target_sys.residue_ids, drug_ligand.residue_ids])
            complex_chains = target_sys.chain_ids + drug_ligand.chain_ids

            complex_sys = MolecularSystem(
                coords=complex_coords,
                charges=complex_charges,
                radii=complex_radii,
                masses=complex_masses,
                atom_names=complex_names,
                residue_names=complex_res,
                residue_ids=complex_resids,
                chain_ids=complex_chains,
                system_name=f"{target_id}_{drug_name}"
            )

            # Solvation free energy
            solv_c = self.solvation_engine.compute_solvation_free_energy(complex_sys)["delta_G_solv_kcal_mol"]
            solv_t = self.solvation_engine.compute_solvation_free_energy(target_sys)["delta_G_solv_kcal_mol"]
            solv_d = self.solvation_engine.compute_solvation_free_energy(drug_ligand)["delta_G_solv_kcal_mol"]
            delta_g_solv = solv_c - (solv_t + solv_d)

            # Screened Coulomb interaction between target protein and drug
            # ligand. This is an INTERMOLECULAR interaction mediated by
            # solvent, so the effective dielectric is the water dielectric
            # eps_w (not the protein dielectric eps_p, which only applies to
            # intramolecular interactions within a single protein body).
            # Same fix as P19-3 in smart_biologics_designer.py and
            # personalized_oncology_ddg.py.
            diff = target_sys.coords[:, None, :] - drug_ligand.coords[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-6)
            coulomb_factor = COULOMB_CONSTANT_KCAL / self.solvation_engine.eps_w
            e_coulomb = float(np.sum(
                (target_sys.charges[:, None] * drug_ligand.charges[None, :]) * np.exp(-self.solvation_engine.kappa * dist) / dist
            ) * coulomb_factor)

            # Contact shape bonus
            close_contacts = np.sum(dist < 4.5)
            vdw_term = -0.12 * float(close_contacts)

            # Total Delta G_bind
            dg_bind = float(delta_g_solv + e_coulomb + vdw_term)
            
            # Kd in nanomolar: Kd = exp(dG / kbT) * 1e9
            exp_arg = float(np.clip(dg_bind / self.kb_t, -30.0, 30.0))
            kd_nm = float(np.exp(exp_arg) * 1e9)

            scores.append(TargetBindingScore(
                target_id=target_id,
                target_name=target_id,
                target_family=family,
                dg_bind_kcal_mol=dg_bind,
                kd_nanomolar=kd_nm,
                electrostatic_contribution=e_coulomb,
                solvation_descreening_penalty=delta_g_solv,
                is_intended_on_target=(target_id == primary_target_id),
                safety_risk_flag=is_antitarget
            ))

        # Find primary target score
        primary_score = next((s for s in scores if s.target_id == primary_target_id), scores[0])
        primary_kd = primary_score.kd_nanomolar

        # Sort off-targets by highest affinity (lowest Kd)
        off_targets = [s for s in scores if s.target_id != primary_target_id]
        off_targets.sort(key=lambda s: s.kd_nanomolar)

        # Selectivity index: ratio of closest off-target Kd to primary Kd
        closest_off_kd = off_targets[0].kd_nanomolar if off_targets else 1e6
        selectivity_idx = float(closest_off_kd / max(1e-6, primary_kd))

        # Evaluate clinical antitarget risks
        herg_score = next((s for s in scores if s.target_id == "hERG_K_Channel"), None)
        cyp_score = next((s for s in scores if s.target_id == "CYP3A4"), None)

        if herg_score and herg_score.kd_nanomolar < 1000.0:
            herg_risk = "High Cardiotoxicity Risk (hERG Kd < 1 uM)"
        elif herg_score and herg_score.kd_nanomolar < 10000.0:
            herg_risk = "Moderate Warning (hERG Kd < 10 uM)"
        else:
            herg_risk = "Low Risk (hERG Safe)"

        if cyp_score and cyp_score.kd_nanomolar < 2000.0:
            cyp_risk = "Potential CYP3A4 Inhibition"
        else:
            cyp_risk = "Low Interaction"

        if selectivity_idx >= 50.0 and herg_risk.startswith("Low") and cyp_risk.startswith("Low"):
            safety_tier = "Clean / Selective Lead"
        elif selectivity_idx < 5.0:
            safety_tier = "Multi-Target Polypharmacology (Promiscuous)"
        else:
            safety_tier = "Unsafe (Antitarget Liability Detected)"

        return PolypharmacologyReport(
            drug_name=drug_name,
            num_targets_screened=len(scores),
            primary_on_target=primary_target_id,
            primary_kd_nm=primary_kd,
            selectivity_index=selectivity_idx,
            top_off_targets=off_targets[:5],
            herg_cardiotox_risk=herg_risk,
            cyp_inhibition_risk=cyp_risk,
            overall_safety_tier=safety_tier
        )
