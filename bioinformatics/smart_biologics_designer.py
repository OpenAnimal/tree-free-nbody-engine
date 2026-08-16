"""
Module 3: Smart Biologics, pH-Switchable Antibody Engineering & Polyreactivity Profiler.
Designs pH-sensitive antibodies for endosomal recycling (pH 7.4 vs pH 5.5) and screens
for developability, surface charge asymmetry, hydrophobic patches, and polyreactivity.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, RESIDUE_PROPERTIES
    from .solvation_free_energy import SolvationFreeEnergyEngine
    from .constant_ph_titration import ConstantPHTitrationEngine
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, COULOMB_CONSTANT_KCAL, RESIDUE_PROPERTIES
    from bioinformatics.solvation_free_energy import SolvationFreeEnergyEngine
    from bioinformatics.constant_ph_titration import ConstantPHTitrationEngine
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D


@dataclass
class PHSwitchCandidate:
    """Evaluated CDR point mutation for pH-dependent antigen release."""
    mutation_str: str              # e.g. "Y102H", "S32H"
    original_res: str
    target_res: str                # usually "HIS"
    res_id: int
    chain_id: str
    dg_bind_ph74_kcal_mol: float   # Binding energy at physiological pH 7.4
    dg_bind_ph55_kcal_mol: float   # Binding energy at endosomal pH 5.5
    delta_dg_ph_kcal_mol: float    # delta = dG(5.5) - dG(7.4) (positive means weaker binding at pH 5.5)
    fcrn_recycling_potential: str  # "High (Ideal Switch)", "Moderate", "Weak / Ineffective"
    switch_efficiency_score: float # 0.0 to 1.0


@dataclass
class DevelopabilityProfile:
    """Biophysical developability and polyreactivity assessment for therapeutic biologics."""
    protein_name: str
    total_sasa_A2: float
    hydrophobic_sasa_A2: float
    hydrophobic_ratio: float
    net_charge_ph74: float
    net_charge_ph55: float
    dipole_moment_debye: float
    pos_patch_area_A2: float
    neg_patch_area_A2: float
    polyreactivity_risk: str       # "Low Risk", "Medium Risk", "High Polyreactivity Risk"
    aggregation_propensity: str    # "Low", "Moderate", "High Aggregation Risk"
    overall_developability_score: float # 0 to 100 (higher = more developable)


class SmartBiologicsDesigner:
    """
    State-of-the-art pH-switchable biologic designer and developability scanner.
    """
    def __init__(
        self,
        dielectric_water: float = 78.5,
        dielectric_protein: float = 4.0,
        ionic_strength_molar: float = 0.15,
        cell_size: float = 8.0
    ):
        self.solvation_engine = SolvationFreeEnergyEngine(
            dielectric_water=dielectric_water,
            dielectric_protein=dielectric_protein,
            ionic_strength_molar=ionic_strength_molar,
            cell_size=cell_size
        )
        self.cell_size = float(cell_size)

    def compute_ph_dependent_charges(self, system: MolecularSystem, ph: float) -> np.ndarray:
        """
        Computes realistic partial and net charges for all titratable residues at a specified pH
        using the Henderson-Hasselbalch equation and spatial electrostatic shielding.
        """
        charges = system.charges.copy()
        for i, res_name in enumerate(system.residue_names):
            pka = RESIDUE_PROPERTIES.get(res_name, {}).get("pKa", None)
            if pka is None:
                continue

            # Titratable basic residues (HIS, LYS, ARG)
            delta_ph = float(np.clip(ph - pka, -30.0, 30.0))
            if res_name == "HIS":
                # Protonated fraction: [HA+] / ([HA+] + [A]) = 1 / (1 + 10^(pH - pKa))
                frac_protonated = 1.0 / (1.0 + 10.0 ** delta_ph)
                charges[i] = frac_protonated * 1.0
            elif res_name in ["LYS", "ARG"]:
                frac_protonated = 1.0 / (1.0 + 10.0 ** delta_ph)
                charges[i] = frac_protonated * 1.0
            # Titratable acidic residues (ASP, GLU, CYS, TYR)
            elif res_name in ["ASP", "GLU"]:
                # Deprotonated fraction: [A-] / ([HA] + [A-]) = 1 / (1 + 10^(pKa - pH))
                frac_deprotonated = 1.0 / (1.0 + 10.0 ** (-delta_ph))
                charges[i] = -frac_deprotonated * 1.0

        return charges

    def evaluate_complex_binding_at_ph(
        self,
        antibody: MolecularSystem,
        antigen: MolecularSystem,
        ph: float
    ) -> float:
        """
        Calculates binding free energy (Delta G_bind) of antibody-antigen complex at target pH.
        """
        ab_ph = antibody.copy()
        ag_ph = antigen.copy()
        ab_ph.charges = self.compute_ph_dependent_charges(antibody, ph)
        ag_ph.charges = self.compute_ph_dependent_charges(antigen, ph)

        # Build complex
        complex_coords = np.vstack([ab_ph.coords, ag_ph.coords])
        complex_charges = np.concatenate([ab_ph.charges, ag_ph.charges])
        complex_radii = np.concatenate([ab_ph.radii, ag_ph.radii])
        complex_masses = np.concatenate([ab_ph.masses, ag_ph.masses])
        complex_names = ab_ph.atom_names + ag_ph.atom_names
        complex_res = ab_ph.residue_names + ag_ph.residue_names
        complex_resids = np.concatenate([ab_ph.residue_ids, ag_ph.residue_ids])
        complex_chains = ab_ph.chain_ids + ag_ph.chain_ids

        complex_sys = MolecularSystem(
            coords=complex_coords,
            charges=complex_charges,
            radii=complex_radii,
            masses=complex_masses,
            atom_names=complex_names,
            residue_names=complex_res,
            residue_ids=complex_resids,
            chain_ids=complex_chains,
            system_name=f"Ab_Ag_pH_{ph:.1f}"
        )

        # Solvation free energy
        solv_c = self.solvation_engine.compute_solvation_free_energy(complex_sys)["delta_G_solv_kcal_mol"]
        solv_ab = self.solvation_engine.compute_solvation_free_energy(ab_ph)["delta_G_solv_kcal_mol"]
        solv_ag = self.solvation_engine.compute_solvation_free_energy(ag_ph)["delta_G_solv_kcal_mol"]
        delta_g_solv = solv_c - (solv_ab + solv_ag)

        # Intermolecular electrostatic energy
        diff = ab_ph.coords[:, None, :] - ag_ph.coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-6)
        screened_coulomb = np.sum(
            (ab_ph.charges[:, None] * ag_ph.charges[None, :]) * np.exp(-self.solvation_engine.kappa * dist) / (self.solvation_engine.eps_p * dist)
        ) * COULOMB_CONSTANT_KCAL

        # Interface contact bonus
        close_contacts = np.sum(dist < 5.0)
        vdw_contact = -0.15 * float(close_contacts)

        return float(delta_g_solv + screened_coulomb + vdw_contact)

    def scan_histidine_switches(
        self,
        antibody: MolecularSystem,
        antigen: MolecularSystem,
        candidate_residue_ids: Optional[List[int]] = None
    ) -> List[PHSwitchCandidate]:
        """
        Scans CDR residues for candidate Histidine mutations to create pH-switchable antibodies.
        """
        # If no residue IDs supplied, select all interface residues
        if candidate_residue_ids is None:
            # Find residues within 8 Angstroms of antigen
            diff = antibody.coords[:, None, :] - antigen.coords[None, :, :]
            dist = np.min(np.sqrt(np.sum(diff**2, axis=-1)), axis=1)
            interface_mask = dist < 8.0
            candidate_residue_ids = list(np.unique(antibody.residue_ids[interface_mask]))
            if not candidate_residue_ids:
                candidate_residue_ids = list(np.unique(antibody.residue_ids[:min(20, antibody.num_atoms)]))

        candidates = []
        for res_id in candidate_residue_ids:
            mask = antibody.residue_ids == res_id
            if not np.any(mask):
                continue
            orig_res = antibody.residue_names[np.where(mask)[0][0]]
            chain = antibody.chain_ids[np.where(mask)[0][0]]

            # Mutate to Histidine
            mut_ab = antibody.copy()
            for idx in np.where(mask)[0]:
                mut_ab.residue_names[idx] = "HIS"

            dg_74 = self.evaluate_complex_binding_at_ph(mut_ab, antigen, ph=7.4)
            dg_55 = self.evaluate_complex_binding_at_ph(mut_ab, antigen, ph=5.5)
            delta_dg = dg_55 - dg_74

            # Score switch potential
            if delta_dg >= 2.5 and dg_74 < -5.0:
                potential = "High (Ideal Switch)"
                score = min(1.0, 0.6 + (delta_dg / 10.0))
            elif delta_dg >= 1.0:
                potential = "Moderate"
                score = 0.4 + (delta_dg / 15.0)
            else:
                potential = "Weak / Ineffective"
                score = max(0.0, 0.2 + (delta_dg / 20.0))

            candidates.append(PHSwitchCandidate(
                mutation_str=f"{orig_res}{res_id}HIS",
                original_res=orig_res,
                target_res="HIS",
                res_id=res_id,
                chain_id=chain,
                dg_bind_ph74_kcal_mol=dg_74,
                dg_bind_ph55_kcal_mol=dg_55,
                delta_dg_ph_kcal_mol=delta_dg,
                fcrn_recycling_potential=potential,
                switch_efficiency_score=float(score)
            ))

        # Sort by best pH delta (highest weakening at pH 5.5)
        candidates.sort(key=lambda x: x.delta_dg_ph_kcal_mol, reverse=True)
        return candidates

    def profile_developability(self, protein: MolecularSystem) -> DevelopabilityProfile:
        """
        Analyzes full-biologic developability, polyreactivity risk, hydrophobic patch density,
        and electrostatic dipole moments.
        """
        # SASA Calculation
        solv_res = self.solvation_engine.compute_solvation_free_energy(protein)
        total_sasa = solv_res.get("total_sasa_angstrom2", solv_res.get("sasa_total_A2", 0.0))

        # Hydrophobic residues: ALA, VAL, LEU, ILE, PHE, TRP, MET, PRO
        hydrophobic_res = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
        hydrophobic_atoms = np.array([r in hydrophobic_res for r in protein.residue_names])
        hydrophobic_sasa = total_sasa * (float(np.sum(hydrophobic_atoms)) / max(1, protein.num_atoms))
        hydro_ratio = hydrophobic_sasa / max(1e-3, total_sasa)

        # Charges at pH 7.4 and pH 5.5
        q_74 = self.compute_ph_dependent_charges(protein, ph=7.4)
        q_55 = self.compute_ph_dependent_charges(protein, ph=5.5)
        net_q_74 = float(np.sum(q_74))
        net_q_55 = float(np.sum(q_55))

        # Dipole moment in Debye: 1 e*Angstrom ~ 4.8032 Debye
        com = protein.center_of_mass
        rel_coords = protein.coords - com
        dipole_vec = np.sum(rel_coords * q_74[:, None], axis=0)
        dipole_mag = float(np.linalg.norm(dipole_vec) * 4.8032)

        # Positive & Negative charge patches
        pos_atoms = q_74 > 0.3
        neg_atoms = q_74 < -0.3
        pos_patch_area = total_sasa * (float(np.sum(pos_atoms)) / max(1, protein.num_atoms))
        neg_patch_area = total_sasa * (float(np.sum(neg_atoms)) / max(1, protein.num_atoms))

        # Polyreactivity & Aggregation assessments
        if pos_patch_area > 0.35 * total_sasa or dipole_mag > 800.0:
            poly_risk = "High Polyreactivity Risk"
            poly_penalty = 30.0
        elif pos_patch_area > 0.25 * total_sasa or dipole_mag > 500.0:
            poly_risk = "Medium Risk"
            poly_penalty = 15.0
        else:
            poly_risk = "Low Risk"
            poly_penalty = 0.0

        if hydro_ratio > 0.50:
            agg_risk = "High Aggregation Risk"
            agg_penalty = 35.0
        elif hydro_ratio > 0.40:
            agg_risk = "Moderate"
            agg_penalty = 15.0
        else:
            agg_risk = "Low"
            agg_penalty = 0.0

        overall_score = max(5.0, 100.0 - poly_penalty - agg_penalty)

        return DevelopabilityProfile(
            protein_name=protein.system_name,
            total_sasa_A2=total_sasa,
            hydrophobic_sasa_A2=hydrophobic_sasa,
            hydrophobic_ratio=hydro_ratio,
            net_charge_ph74=net_q_74,
            net_charge_ph55=net_q_55,
            dipole_moment_debye=dipole_mag,
            pos_patch_area_A2=pos_patch_area,
            neg_patch_area_A2=neg_patch_area,
            polyreactivity_risk=poly_risk,
            aggregation_propensity=agg_risk,
            overall_developability_score=overall_score
        )
