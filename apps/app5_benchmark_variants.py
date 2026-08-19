"""Standardized variant benchmark for Application 5 (3D protein electrostatics).

Variants:
  standard      -- exact direct O(N^2) per-atom Debye-Huckel screened Coulomb
                   potential (the natural reference for the app's kernel)
  +elastichash  -- the app's compute path: funnel-hash bucketed 3D Morton
                   clusters, then direct O(K^2) screened-Coulomb between
                   cluster centroids, broadcast back to atoms
  +fmm (Yukawa3DFMM) -- single-level flat 3D Yukawa FMM (core/yukawa3d_fmm.py,
                   depth=6, p=8) on the same kernel; the 3D analogue of the
                   2D FastVectorizedFMM, indexed by CellIndex(dims=3) + funnel
                   hash. This closes the round-3 INAPPLICABILITY.md Class C
                   gap (the 3D Yukawa kernel is no longer "right kernel, 2D-
                   only FMM").

Accuracy vs `standard` on the per-atom potential (rel L2). The cluster-mean
approximation error shows up in the table, not hidden in a note.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _protein(n_atoms: int = 3000, seed: int = 42):
    """Same backbone geometry as app5_bioinformatics.generate_synthetic_protein_backbone."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10 * np.pi, n_atoms)
    r_helix, r_super = 0.25, 0.4
    x = (r_super + r_helix * np.cos(5 * t)) * np.cos(t)
    y = (r_super + r_helix * np.cos(5 * t)) * np.sin(t)
    z = 0.08 * t + r_helix * np.sin(5 * t)
    coords = np.stack([x, y, z], axis=1)
    coords = (coords - np.min(coords, axis=0)) / (np.ptp(coords, axis=0) + 1e-6) * 0.8 + 0.1
    charges = (np.sin(3 * t) * 1.0 + rng.normal(0, 0.2, size=n_atoms)).astype(np.float64)
    return coords.astype(np.float64), charges


def _direct_debye_huckel(coords, charges, kappa=2.0):
    """Exact O(N^2) per-atom screened Coulomb: V_i = sum_j q_j exp(-k r)/r."""
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.linalg.norm(diff, axis=-1) + 1e-6
    np.fill_diagonal(r, 1e9)
    return np.sum(charges[None, :] * np.exp(-kappa * r) / r, axis=1)


def _cluster_debye_huckel(coords, charges, grid_res=16, kappa=2.0):
    """The app's path: 3D Morton bucketing via the funnel hash, then direct
    O(K^2) screened-Coulomb between cluster centroids, broadcast to atoms."""
    from core.elastic_hash import ElasticHashTable
    ix = np.clip((coords[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((coords[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    iz = np.clip((coords[:, 2] * grid_res).astype(np.int64), 0, grid_res - 1)
    morton = (ix << 20) | (iy << 10) | iz
    unique_keys, inverse = np.unique(morton, return_inverse=True)
    ht = ElasticHashTable(capacity=grid_res ** 3, delta=0.05)
    for c, k in enumerate(unique_keys):
        ht.insert(int(k), c)
    inverse = np.array([ht.lookup(int(k))[0] for k in morton], dtype=np.int64)
    num_clusters = len(unique_keys)
    centers = np.array([np.mean(coords[inverse == c], axis=0) for c in range(num_clusters)])
    cq = np.bincount(inverse, weights=charges, minlength=num_clusters)
    c_diff = centers[:, None, :] - centers[None, :, :]
    c_dist = np.linalg.norm(c_diff, axis=-1) + 1e-6
    np.fill_diagonal(c_dist, 1e9)
    cluster_pot = np.sum(cq[None, :] * np.exp(-kappa * c_dist) / c_dist, axis=1)
    return cluster_pot[inverse]


def run_app5_variants(n_atoms: int = 3000):
    from core import Yukawa3DFMM
    coords, charges = _protein(n_atoms=n_atoms)
    # The app's Debye-Huckel reference uses kappa=2.0; the +fmm row uses the
    # same kappa so the comparison is apples-to-apples on the SAME kernel.
    kappa = 2.0
    fmm = Yukawa3DFMM(depth=6, p=8, kappa=kappa)
    bench = VariantBenchmark(
        f"App 5 -- 3D protein electrostatics (N={n_atoms}, Debye-Huckel screened Coulomb, kappa={kappa})"
    )
    bench.add(
        "standard (direct O(N^2))",
        lambda: _direct_debye_huckel(coords, charges, kappa=kappa),
        note="exact per-atom screened Coulomb reference",
    )
    bench.add(
        "+elastichash (cluster O(K^2))",
        lambda: _cluster_debye_huckel(coords, charges, grid_res=16, kappa=kappa),
        accuracy_vs="standard (direct O(N^2))",
        note="funnel-hash 3D Morton clusters, direct O(K^2) between centroids; "
             "lossy cluster-mean approximation",
    )
    bench.add(
        "+fmm (Yukawa3DFMM)",
        lambda: fmm.evaluate(coords, charges),
        accuracy_vs="standard (direct O(N^2))",
        note="single-level flat 3D Yukawa FMM, depth=6 p=8; closes "
             "INAPPLICABILITY.md Class C (3D Yukawa now has a 3D FMM)",
    )
    return bench.run()


if __name__ == "__main__":
    run_app5_variants()
