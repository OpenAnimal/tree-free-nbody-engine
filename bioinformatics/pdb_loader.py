"""
Molecular System Representation and Structure Builders (PDB / Synthetic / Capsid).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


# Electrostatic conversion constant: e^2 / (4 * pi * eps_0 * Angstrom) -> kcal / mol
COULOMB_CONSTANT_KCAL: float = 332.063711

# Standard Bondi / AMBER van der Waals Radii (Angstroms)
ATOM_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "DEFAULT": 1.70
}

# Standard Atomic Masses (Da)
ATOM_MASSES: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.065,
    "DEFAULT": 12.011
}

# Standard Residue Net / Partial Charge Archetypes
RESIDUE_PROPERTIES: Dict[str, Dict[str, float]] = {
    "ARG": {"net_charge": 1.0, "pKa": 12.5},
    "LYS": {"net_charge": 1.0, "pKa": 10.5},
    "HIS": {"net_charge": 0.0, "pKa": 6.0},   # Titratable: neutral (HID/HIE) or +1 (HIP)
    "ASP": {"net_charge": -1.0, "pKa": 3.9},  # Titratable: -1 (deprotonated) or 0 (ASH)
    "GLU": {"net_charge": -1.0, "pKa": 4.2},  # Titratable: -1 (deprotonated) or 0 (GLH)
    "CYS": {"net_charge": 0.0, "pKa": 8.3},
    "TYR": {"net_charge": 0.0, "pKa": 10.1},
    "ALA": {"net_charge": 0.0, "pKa": 7.0},
    "VAL": {"net_charge": 0.0, "pKa": 7.0},
    "LEU": {"net_charge": 0.0, "pKa": 7.0},
    "ILE": {"net_charge": 0.0, "pKa": 7.0},
    "PHE": {"net_charge": 0.0, "pKa": 7.0},
    "TRP": {"net_charge": 0.0, "pKa": 7.0},
    "PRO": {"net_charge": 0.0, "pKa": 7.0},
    "SER": {"net_charge": 0.0, "pKa": 7.0},
    "THR": {"net_charge": 0.0, "pKa": 7.0},
    "ASN": {"net_charge": 0.0, "pKa": 7.0},
    "GLN": {"net_charge": 0.0, "pKa": 7.0},
    "MET": {"net_charge": 0.0, "pKa": 7.0},
    "GLY": {"net_charge": 0.0, "pKa": 7.0},
}


@dataclass
class MolecularSystem:
    """
    All-Atom or Coarse-Grained Macromolecular Representation.
    Coordinates are in Angstroms, Charges in elementary charge units (e),
    Energies in kcal/mol.
    """
    coords: np.ndarray          # (N, 3) float64 - Angstroms
    charges: np.ndarray         # (N,) float64 - Elementary charge (e)
    radii: np.ndarray           # (N,) float64 - Bondi radii (Angstroms)
    masses: np.ndarray          # (N,) float64 - Atomic masses (Da)
    atom_names: List[str]       # N atom names (e.g. CA, CB, N, O)
    residue_names: List[str]    # N residue names (e.g. ALA, LYS)
    residue_ids: np.ndarray     # (N,) int32
    chain_ids: List[str]        # N chain IDs
    system_name: str = "Macromolecule"
    # Provenance of the partial charges (set by the loader/generator). Real
    # force-field charges would say e.g. "AMBER ff14SB"; the parse_pdb heuristic
    # sets "heuristic element-based placeholder, not a force field".
    charges_source: str = "heuristic element-based placeholder, not a force field"

    @property
    def num_atoms(self) -> int:
        return len(self.coords)

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.charges))

    @property
    def center_of_mass(self) -> np.ndarray:
        return np.sum(self.coords * self.masses[:, None], axis=0) / np.sum(self.masses)

    def center_to_origin(self) -> None:
        self.coords -= self.center_of_mass

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        min_bound = np.min(self.coords, axis=0)
        max_bound = np.max(self.coords, axis=0)
        return min_bound, max_bound

    def copy(self) -> MolecularSystem:
        return MolecularSystem(
            coords=self.coords.copy(),
            charges=self.charges.copy(),
            radii=self.radii.copy(),
            masses=self.masses.copy(),
            atom_names=list(self.atom_names),
            residue_names=list(self.residue_names),
            residue_ids=self.residue_ids.copy(),
            chain_ids=list(self.chain_ids),
            system_name=self.system_name,
            charges_source=self.charges_source,
        )


def parse_pdb(pdb_content_or_path: str) -> MolecularSystem:
    """
    Parse a PDB file string or filepath into a MolecularSystem.
    """
    if "\n" not in pdb_content_or_path and len(pdb_content_or_path) < 512:
        with open(pdb_content_or_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = pdb_content_or_path.splitlines()

    coords = []
    atom_names = []
    res_names = []
    res_ids = []
    chain_ids = []
    charges = []
    radii = []
    masses = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21:22].strip() or "A"
                res_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                element = line[76:78].strip().upper() if len(line) >= 78 else name[0].upper()
                if not element or element.isdigit():
                    element = name[0].upper()

                r = ATOM_RADII.get(element, ATOM_RADII["DEFAULT"])
                m = ATOM_MASSES.get(element, ATOM_MASSES["DEFAULT"])

                # Estimate default partial charge if not given.
                # WARNING: this is a crude element/residue-name heuristic that
                # fires on every real PDB with no charge columns. It is NOT a
                # force field -- do not use these charges for binding/energy
                # conclusions. The provenance is recorded on
                # MolecularSystem.charges_source.
                q = 0.0
                if res_name in ["ARG", "LYS"] and name in ["NZ", "NH1", "NH2", "NE", "1HZ", "2HZ", "3HZ"]:
                    q = 0.33
                elif res_name in ["ASP", "GLU"] and name in ["OD1", "OD2", "OE1", "OE2"]:
                    q = -0.5
                elif name == "N":
                    q = -0.4
                elif name == "H" or name == "HN":
                    q = 0.3
                elif name == "C":
                    q = 0.5
                elif name == "O":
                    q = -0.5

                coords.append([x, y, z])
                atom_names.append(name)
                res_names.append(res_name)
                res_ids.append(res_id)
                chain_ids.append(chain)
                charges.append(q)
                radii.append(r)
                masses.append(m)
            except Exception:
                continue

    return MolecularSystem(
        coords=np.array(coords, dtype=np.float64),
        charges=np.array(charges, dtype=np.float64),
        radii=np.array(radii, dtype=np.float64),
        masses=np.array(masses, dtype=np.float64),
        atom_names=atom_names,
        residue_names=res_names,
        residue_ids=np.array(res_ids, dtype=np.int32),
        chain_ids=chain_ids,
        system_name="PDB_Import",
        charges_source="heuristic element-based placeholder, not a force field",
    )


def generate_synthetic_protein(n_atoms: int = 5000, seed: int = 42) -> MolecularSystem:
    """
    Generates realistic 3D protein topology with alpha-helical supercoils,
    beta-sheet motifs, backbone atoms (N, CA, C, O), and charged sidechains (Lys, Glu, Arg, Asp).
    """
    rng = np.random.RandomState(seed)
    n_residues = max(10, n_atoms // 6)
    t = np.linspace(0, 16 * np.pi, n_residues)

    # Tertiary fold trajectory
    r_super = 25.0  # Angstroms
    r_helix = 6.0   # Angstroms
    z_pitch = 0.8

    cx = (r_super + r_helix * np.cos(4 * t)) * np.cos(0.5 * t)
    cy = (r_super + r_helix * np.cos(4 * t)) * np.sin(0.5 * t)
    cz = z_pitch * t * 2.5 + r_helix * np.sin(4 * t)

    res_coords = np.stack([cx, cy, cz], axis=1)

    all_coords = []
    all_charges = []
    all_radii = []
    all_masses = []
    all_names = []
    all_resnames = []
    all_resids = []
    all_chains = []

    res_types = ["ALA", "LEU", "ARG", "GLU", "LYS", "ASP", "VAL", "PHE", "SER", "HIS"]

    for i in range(n_residues):
        rc = res_coords[i]
        res_type = res_types[i % len(res_types)]

        # Backbone N, CA, C, O
        v_fwd = res_coords[min(i + 1, n_residues - 1)] - res_coords[max(i - 1, 0)]
        v_fwd_norm = np.linalg.norm(v_fwd) + 1e-8
        fwd = v_fwd / v_fwd_norm
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(fwd, up)
        if np.linalg.norm(side) < 1e-4:
            side = np.array([1.0, 0.0, 0.0])
        else:
            side = side / np.linalg.norm(side)
        norm_up = np.cross(side, fwd)

        p_ca = rc
        p_n = rc - 1.45 * fwd - 0.4 * side
        p_c = rc + 1.52 * fwd + 0.3 * side
        p_o = p_c + 1.23 * norm_up
        p_cb = rc + 1.5 * side + 0.5 * norm_up

        atoms_res = [
            ("N", p_n, -0.47, 1.55, 14.0),
            ("CA", p_ca, 0.07, 1.70, 12.0),
            ("C", p_c, 0.51, 1.70, 12.0),
            ("O", p_o, -0.51, 1.52, 16.0),
            ("CB", p_cb, 0.00, 1.70, 12.0),
        ]

        # Add sidechain charge centers for ionic residues
        if res_type in ["ARG", "LYS"]:
            p_sc = p_cb + 2.0 * side + rng.normal(0, 0.3, 3)
            atoms_res.append(("NZ", p_sc, +1.0, 1.55, 14.0))
        elif res_type in ["ASP", "GLU"]:
            p_sc = p_cb + 2.0 * side + rng.normal(0, 0.3, 3)
            atoms_res.append(("OD1", p_sc, -1.0, 1.52, 16.0))
        elif res_type == "HIS":
            p_sc = p_cb + 1.8 * side + rng.normal(0, 0.3, 3)
            atoms_res.append(("ND1", p_sc, +0.2, 1.55, 14.0))

        for aname, acoord, achg, arad, amass in atoms_res:
            all_coords.append(acoord)
            all_charges.append(achg)
            all_radii.append(arad)
            all_masses.append(amass)
            all_names.append(aname)
            all_resnames.append(res_type)
            all_resids.append(i + 1)
            all_chains.append("A")

            if len(all_coords) >= n_atoms:
                break
        if len(all_coords) >= n_atoms:
            break

    return MolecularSystem(
        coords=np.array(all_coords[:n_atoms], dtype=np.float64),
        charges=np.array(all_charges[:n_atoms], dtype=np.float64),
        radii=np.array(all_radii[:n_atoms], dtype=np.float64),
        masses=np.array(all_masses[:n_atoms], dtype=np.float64),
        atom_names=all_names[:n_atoms],
        residue_names=all_resnames[:n_atoms],
        residue_ids=np.array(all_resids[:n_atoms], dtype=np.int32),
        chain_ids=all_chains[:n_atoms],
        system_name=f"Synthetic_Protein_{n_atoms}atoms",
        charges_source="synthetic heuristic placeholder, not a force field",
    )


def generate_viral_capsid(n_capsomers: int = 60, atoms_per_unit: int = 500, radius: float = 120.0, seed: int = 42) -> MolecularSystem:
    """
    Generates an icosahedral viral capsid assembly (e.g. Parvovirus / Adenovirus / HIV capsid model)
    spanning tens of thousands to hundreds of thousands of atoms in Angstroms.

    ``seed`` controls the per-capsomer jitter (default 42, fixed for
    reproducibility); previously this drew from the unseeded global
    ``np.random`` state.
    """
    rng = np.random.RandomState(seed)
    # Golden ratio for icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v_base = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=np.float64)
    v_base = v_base / np.linalg.norm(v_base, axis=1, keepdims=True) * radius

    all_coords = []
    all_charges = []
    all_radii = []
    all_masses = []
    all_names = []
    all_resnames = []
    all_resids = []
    all_chains = []

    for c_idx in range(n_capsomers):
        # Position on spherical shell
        center = v_base[c_idx % len(v_base)] + rng.normal(0, 5.0, 3)
        center = center / np.linalg.norm(center) * radius
        unit = generate_synthetic_protein(atoms_per_unit, seed=c_idx)
        unit_coords = unit.coords - unit.center_of_mass + center

        all_coords.append(unit_coords)
        all_charges.append(unit.charges)
        all_radii.append(unit.radii)
        all_masses.append(unit.masses)
        all_names.extend(unit.atom_names)
        all_resnames.extend(unit.residue_names)
        all_resids.append(unit.residue_ids + c_idx * 1000)
        all_chains.extend([chr(65 + (c_idx % 26))] * len(unit.coords))

    return MolecularSystem(
        coords=np.vstack(all_coords),
        charges=np.concatenate(all_charges),
        radii=np.concatenate(all_radii),
        masses=np.concatenate(all_masses),
        atom_names=all_names,
        residue_names=all_resnames,
        residue_ids=np.concatenate(all_resids),
        chain_ids=all_chains,
        system_name=f"Viral_Capsid_{len(np.vstack(all_coords))}atoms",
        charges_source="synthetic heuristic placeholder, not a force field",
    )
