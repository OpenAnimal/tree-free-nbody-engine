"""
Application D: Constant-pH Monte Carlo Protonation Titration Engine (CpHMD).
Simulates pH-dependent residue protonation (His, Asp, Glu, Lys), pKa shift prediction,
and Isoelectric Point (pI) calculation accelerated by Tree-Free FMM.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
try:
    from .pdb_loader import MolecularSystem, RESIDUE_PROPERTIES
    from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, COULOMB_CONSTANT_KCAL
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem, RESIDUE_PROPERTIES
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, COULOMB_CONSTANT_KCAL


KB_T_300K: float = 0.0019872041 * 300.0  # kcal/mol at 300 K (~0.596 kcal/mol)
LN10_KB_T: float = np.log(10.0) * KB_T_300K  # ~1.372 kcal/mol per pH unit


class ConstantPHTitrationEngine:
    """
    Constant-pH Monte Carlo (CpHMC) Titration Engine.
    Predicts pKa shifts, titration curves, and protonation state ensembles.
    """
    def __init__(
        self,
        system: MolecularSystem,
        temperature_kelvin: float = 300.0,
        ionic_strength_molar: float = 0.15,
        cell_size: float = 8.0,
    ):
        self.system = system.copy()
        self.temperature = float(temperature_kelvin)
        self.kb_t = 0.0019872041 * self.temperature
        self.ln10_kb_t = np.log(10.0) * self.kb_t
        self.cell_size = float(cell_size)

        # Debye screening
        self.kappa = float(0.329 * np.sqrt(ionic_strength_molar))
        self.fmm = TreeFreeBioFMM(
            cell_size=self.cell_size,
            kappa=self.kappa,
            dielectric_water=78.5,
            dielectric_protein=4.0,
            kernel_type=ScreenedKernelType.DEBYE_HUCKEL
        )

        # Identify all titratable residues
        self._find_titratable_residues()

    def _find_titratable_residues(self):
        """Identifies titratable residues (ASP, GLU, HIS, LYS) and their representative atom indices."""
        self.titratable_sites = []
        
        # Group atoms by residue ID
        unique_resids = np.unique(self.system.residue_ids)
        for rid in unique_resids:
            idx = np.where(self.system.residue_ids == rid)[0]
            if len(idx) == 0:
                continue
            rname = self.system.residue_names[idx[0]]
            if rname in ["ASP", "GLU", "HIS", "LYS"]:
                pka_model = RESIDUE_PROPERTIES[rname]["pKa"]
                # Identify charge-bearing sidechain atoms
                sc_idx = [i for i in idx if self.system.atom_names[i] in ["NZ", "OD1", "OD2", "OE1", "OE2", "ND1", "NE2", "CB"]]
                if not sc_idx:
                    sc_idx = idx

                initial_prot = True if rname in ["HIS", "LYS"] else False
                self.titratable_sites.append({
                    "res_id": rid,
                    "res_name": rname,
                    "pka_model": pka_model,
                    "atom_indices": sc_idx,
                    "is_protonated": initial_prot,
                    "_initial_is_protonated": initial_prot,
                })

    def run_mc_titration(
        self,
        target_ph: float,
        num_mc_steps: int = 500,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes Metropolis Monte Carlo protonation trials at constant pH.
        """
        rng = np.random.RandomState(seed)
        num_sites = len(self.titratable_sites)
        if num_sites == 0:
            return {"target_ph": target_ph, "mean_protonation": {}, "net_charge": self.system.total_charge}

        # Reset protonation states to initial values so that repeated calls
        # start from a consistent state (finding P45-1: is_protonated was not
        # reset between calls, causing charge drift because current_charges
        # was reset to system.charges.copy() but is_protonated carried over
        # from the previous run, making dq applied to the wrong baseline).
        for site in self.titratable_sites:
            site["is_protonated"] = site["_initial_is_protonated"]

        # Track protonation history
        protonated_history = np.zeros((num_mc_steps, num_sites), dtype=bool)
        accepted_moves = 0

        # Current baseline potential field
        current_charges = self.system.charges.copy()
        potentials, _, _ = self.fmm.evaluate(coords=self.system.coords, charges=current_charges)

        for step in range(num_mc_steps):
            # Select random titratable site
            site_idx = rng.randint(0, num_sites)
            site = self.titratable_sites[site_idx]
            rname = site["res_name"]
            atom_idx = site["atom_indices"]

            current_state = site["is_protonated"]
            target_state = not current_state

            # Determine charge delta upon protonation/deprotonation (+1 or -1)
            # ASP/GLU: protonated = 0 (charge +1 delta vs deprotonated -1)
            # HIS/LYS: protonated = +1 (charge +1 delta vs deprotonated 0)
            dq = +1.0 if target_state else -1.0
            dq_per_atom = dq / len(atom_idx)

            # Local electrostatic potential at site atoms
            local_phi = np.mean(potentials[atom_idx])

            # Electrostatic Work: Delta G_elec = dq * Phi + self_energy_correction
            delta_g_elec = dq * local_phi
            # Reference chemical potential difference: Delta G_ref = (protonated ? + : -) * ln(10) * k_B * T * (pH - pKa_ref)
            sign = +1.0 if target_state else -1.0
            delta_g_ref = sign * self.ln10_kb_t * (target_ph - site["pka_model"])

            delta_g_total = delta_g_elec + delta_g_ref

            # Metropolis Acceptance Criterion
            # Clip the Metropolis exponent to avoid overflow for strongly
            # unfavorable protonation moves while preserving its probability.
            acceptance_prob = np.exp(np.clip(-delta_g_total / self.kb_t, -745.0, 0.0))
            if delta_g_total <= 0.0 or rng.uniform(0, 1) < acceptance_prob:
                # Accept transition
                site["is_protonated"] = target_state
                current_charges[atom_idx] += dq_per_atom
                # Rapid potential update via FMM
                potentials, _, _ = self.fmm.evaluate(coords=self.system.coords, charges=current_charges)
                accepted_moves += 1

            for s_i, s in enumerate(self.titratable_sites):
                protonated_history[step, s_i] = s["is_protonated"]

        # Calculate fractional protonation per site
        # Burn-in: discard first 20% steps
        burn_in = int(num_mc_steps * 0.2)
        mean_prot = np.mean(protonated_history[burn_in:], axis=0)

        site_results = {}
        for s_i, s in enumerate(self.titratable_sites):
            site_key = f"{s['res_name']}_{s['res_id']}"
            site_results[site_key] = {
                "fraction_protonated": float(mean_prot[s_i]),
                "pka_model": s["pka_model"],
            }

        return {
            "target_ph": float(target_ph),
            "acceptance_ratio": float(accepted_moves / num_mc_steps),
            "site_protonation": site_results,
            "mean_total_charge": float(np.sum(current_charges)),
            "num_titratable_sites": num_sites,
        }

    def compute_titration_curve(
        self,
        ph_range: Tuple[float, float] = (2.0, 12.0),
        num_ph_points: int = 11,
        steps_per_ph: int = 200
    ) -> Dict[str, Any]:
        """
        Sweeps through a pH gradient to generate complete titration curves and calculate pI.
        """
        t0 = time.perf_counter()
        ph_values = np.linspace(ph_range[0], ph_range[1], num_ph_points)
        net_charges = []
        site_curves = {f"{s['res_name']}_{s['res_id']}": [] for s in self.titratable_sites}

        for ph in ph_values:
            res = self.run_mc_titration(target_ph=ph, num_mc_steps=steps_per_ph)
            net_charges.append(res["mean_total_charge"])
            for skey in site_curves:
                if skey in res["site_protonation"]:
                    site_curves[skey].append(res["site_protonation"][skey]["fraction_protonation" if "fraction_protonation" in res["site_protonation"][skey] else "fraction_protonated"])
                else:
                    site_curves[skey].append(0.0)

        net_charges = np.array(net_charges)
        # Find Isoelectric Point (pI) where net charge crosses 0
        pI = 7.0
        if np.any(net_charges <= 0) and np.any(net_charges >= 0):
            pI = float(np.interp(0.0, net_charges[::-1], ph_values[::-1]))

        elapsed = time.perf_counter() - t0
        return {
            "ph_values": ph_values.tolist(),
            "net_charges": net_charges.tolist(),
            "isoelectric_point_pI": pI,
            "site_titration_curves": site_curves,
            "elapsed_seconds": elapsed,
        }

    def predict_pka_shifts(self) -> Dict[str, Any]:
        """
        Fast electrostatic pKa shift prediction from initial continuum potential field:
        Delta pKa = -Delta G_elec / (ln(10) * k_B * T).
        """
        if len(self.titratable_sites) == 0:
            return {"titratable_residues": [], "pka_shifts": {}}

        potentials, _, _ = self.fmm.evaluate(coords=self.system.coords, charges=self.system.charges)
        pka_shifts = {}
        for site in self.titratable_sites:
            skey = f"{site['res_name']}_{site['res_id']}"
            local_phi = float(np.mean(potentials[site["atom_indices"]]))
            # pKa is always defined for the DEPROTONATION reaction:
            #   acid:   HA -> H+ + A-   (q_prod - q_react = -1 - 0 = -1)
            #   base:   BH+ -> B + H+   (q_prod - q_react = 0 - (+1) = -1)
            # In both cases delta_G_elec = -phi, so:
            #   delta_pKa = delta_G_elec / (RT*ln10) = -phi / (RT*ln10)
            # A negative potential raises pKa for acids (favors neutral form)
            # and raises pKa for bases (stabilizes BH+). A positive potential
            # lowers pKa for both. The previous code used sign = +1 for bases,
            # which gave the OPPOSITE shift -- a positive potential raised pKa
            # for bases instead of lowering it.
            delta_pka = float(-local_phi / max(1e-3, self.ln10_kb_t))
            pka_shifts[skey] = {
                "pka_model": site["pka_model"],
                "predicted_pka": float(site["pka_model"] + delta_pka),
                "delta_pka": delta_pka,
                "local_electrostatic_potential": local_phi
            }

        return {
            "titratable_residues": [f"{s['res_name']}_{s['res_id']}" for s in self.titratable_sites],
            "pka_shifts": pka_shifts
        }

