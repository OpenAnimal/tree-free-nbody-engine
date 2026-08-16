"""
Example 3: SE(3)-Equivariant Long-Range Prior for Atomistic GNNs (MACE/NequIP Style)
===================================================================================
Demonstrates injecting exact long-range all-pairs electrostatic and vector fields
into an atomistic graph neural network without distance cutoff truncation (r_cut).
"""

import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import EquivariantMultipoleLayer


class EquivariantAtomisticModel:
    """
    Simulates an Equivariant Molecular Foundation Model (MACE / NequIP)
    augmented with Tree-Free all-pairs long-range physical multipoles.
    """
    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        # Global O(N) multipole field injector (Debye-Huckel screened potential)
        self.fmm_field = EquivariantMultipoleLayer(
            hidden_dim=hidden_dim,
            grid_depth=4,
            softening_radius=0.1,
            screening_kappa=0.05
        )
        # Energy readout head
        scale = 1.0 / np.sqrt(hidden_dim)
        self.w_energy = np.random.normal(0, scale, size=(hidden_dim, 1)).astype(np.float32)

    def forward(self, coords: np.ndarray, atom_features: np.ndarray, charges: np.ndarray):
        """
        coords: (N, 3) 3D atomic coordinates (Angstroms)
        atom_features: (N, hidden_dim) Latent chemical embeddings
        charges: (N,) Atomic partial charges (e)
        """
        # 1. Evaluate all-pairs SE(3)-equivariant field and invariant potentials in O(N)
        h_updated, force_vectors, potentials, meta = self.fmm_field.forward(
            coords, atom_features, charges
        )

        # 2. Total molecular potential energy prediction
        atomic_energies = np.matmul(h_updated, self.w_energy).squeeze(-1) # (N,)
        total_energy = float(np.sum(atomic_energies))

        return total_energy, force_vectors, potentials, meta


def run_mace_demo():
    print("=" * 70)
    print(">>> DEMO 3: SE(3)-Equivariant All-Pairs Physical Prior for GNNs")
    print("=" * 70)

    # Simulate a 2,500-atom protein or macromolecule
    N_atoms = 2500
    hidden_dim = 64
    np.random.seed(42)

    # Atomic coordinates (in Angstroms across a 50x50x50 box)
    coords = np.random.uniform(-25.0, 25.0, size=(N_atoms, 3)).astype(np.float32)
    atom_feats = np.random.randn(N_atoms, hidden_dim).astype(np.float32)
    charges = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=N_atoms).astype(np.float32)

    model = EquivariantAtomisticModel(hidden_dim=hidden_dim)

    t0 = time.perf_counter()
    energy, forces, potentials, meta = model.forward(coords, atom_feats, charges)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[-] Atom Count:        {N_atoms:,} atoms (Zero cutoff radius truncation)")
    print(f"[-] Forward Pass Time: {elapsed_ms:.2f} ms")
    print(f"[-] Total Energy:      {energy:.4f} a.u.")
    print(f"[-] Equivariant Forces:{forces.shape} (SE(3) vector field)")
    print(f"[-] Invariant Pot:     {potentials.shape}")
    print(f"[-] Far-Field Physics: Screened Debye-Huckel kappa={meta['kappa_screening']}")
    print("=" * 70)


if __name__ == "__main__":
    run_mace_demo()
