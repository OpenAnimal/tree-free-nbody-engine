"""Standardized variant benchmark for Application 5 (3D protein electrostatics).

Variants:
  standard      -- exact direct O(N^2) per-atom Debye-Huckel screened Coulomb
                   potential (the natural reference for the app's kernel)
  +elastichash  -- the app's compute path: `TreeFreeBioFMM` (funnel-hash
                   bucketed 3D Morton clusters, per-atom monopole + first-order
                   dipole far-field evaluation against far-cluster centers).
                   Round-7 task T-C2 replaced the old center-broadcast with
                   per-atom evaluation; the rel-L2 dropped from ~5.7e-1 to
                   <1.5e-1 on this distribution.
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
    """Exact O(N^2) per-atom screened Coulomb: V_i = sum_j q_j exp(-k r)/r.

    Self-pairs (i==j) are excluded by setting the diagonal distance to a
    large value so exp(-k*r)/r -> 0.  Off-diagonal distances are NOT
    regularized: an earlier version added 1e-6 to every distance, which
    introduced a systematic ~6e-5 rel-L2 bias that masqueraded as a
    convergence floor in the Yukawa3D error-vs-p table (see the round-5
    root-cause analysis in core/yukawa3d_fmm.py and BENCHMARKS.md).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, 1e9)
    return np.sum(charges[None, :] * np.exp(-kappa * r) / r, axis=1)


def _cluster_debye_huckel(coords, charges, cell_size=0.05, kappa=2.0):
    """The app's path: `TreeFreeBioFMM` with funnel-hash 3D Morton bucketing
    and per-atom monopole + first-order dipole far-field (Round-7 task T-C2).

    The coords are in the unit box [0.1, 0.9]; cell_size is in the same units.
    `TreeFreeBioFMM` uses Debye-Huckel by default with kappa and dielectric_water.
    The returned potentials are in kcal/mol/e; we divide by COULOMB_CONSTANT_KCAL
    / eps_w to get back to the unit-kernel values that `standard` reports.
    """
    from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType, COULOMB_CONSTANT_KCAL
    fmm = TreeFreeBioFMM(
        cell_size=cell_size,
        kappa=kappa,
        dielectric_water=1.0,  # match _direct_debye_huckel (no eps division)
        kernel_type=ScreenedKernelType.DEBYE_HUCKEL,
    )
    pots, _, _ = fmm.evaluate(coords, charges)
    return pots / COULOMB_CONSTANT_KCAL


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
        "+elastichash (TreeFreeBioFMM per-atom dipole)",
        lambda: _cluster_debye_huckel(coords, charges, cell_size=0.05, kappa=kappa),
        accuracy_vs="standard (direct O(N^2))",
        note="funnel-hash 3D Morton clusters, per-atom monopole + dipole far "
             "field (Round-7 T-C2); replaces old center-broadcast",
    )
    bench.add(
        "+fmm (Yukawa3DFMM)",
        lambda: fmm.evaluate(coords, charges),
        accuracy_vs="standard (direct O(N^2))",
        note="single-level flat 3D Yukawa FMM, depth=6 p=8; closes "
             "INAPPLICABILITY.md Class C (3D Yukawa now has a 3D FMM)",
    )
    # Round-7 task T-C1: bio-units wrapper over Yukawa3DFMM.
    # The benchmark coords are in [0.1, 0.9] (unit-box-like). The bio wrapper
    # maps to [0.1, 0.9] with a = 0.8/span and kappa_unit = kappa_angstrom / a.
    # To match the reference (which uses kappa directly), set
    # kappa_angstrom = kappa * a = kappa * 0.8 / span. The bio wrapper returns
    # V_u * a * COULOMB / eps; with eps=1, divide by COULOMB to get V_u * a,
    # which equals V_ref since V_ref = a * V_u.
    from bioinformatics.core.fast_multipole_kernel import TaylorYukawaBioFMM, COULOMB_CONSTANT_KCAL
    _span = float(np.max(np.ptp(coords, axis=0)))
    _a = 0.8 / _span
    bio_fmm = TaylorYukawaBioFMM(
        kappa_angstrom=kappa * _a, dielectric=1.0, cell_size_A=0.167, p=8
    )
    bench.add(
        "+bio_taylor (TaylorYukawaBioFMM)",
        lambda: bio_fmm.evaluate(coords, charges)[0] / COULOMB_CONSTANT_KCAL,
        accuracy_vs="standard (direct O(N^2))",
        note="Round-7 T-C1: bio-units wrapper over Yukawa3DFMM with Å→unit "
             "box mapping; target ≤1e-6 rel-L2",
    )
    return bench.run()


def run_convergence(n_atoms: int = 2000, ps=(2, 4, 6, 8, 10, 12), depth: int = 6,
                    kappa: float = 2.0, seed: int = 42):
    """Yukawa3D error-vs-p convergence table (round-4 task 4.8).

    For each expansion order p in `ps`, run Yukawa3DFMM(depth, p, kappa) on
    the app5 protein distribution and report rel-L2 vs the exact direct
    O(N^2) screened-Coulomb reference. The table makes the convergence rate
    visible: rel-L2 should drop by ~1e-2 per +2 in p (the scheme is
    order-(p+1) in the cell radius). A previous version of this table
    floored at ~6.3e-5 for p>=6 with a note attributing it to "ring-2 near
    field + f64 round-off"; that attribution was WRONG (round-4 task 5.2
    root-cause analysis). The floor was caused by a +1e-6 distance
    regularization in the direct reference, not by the FMM. With the
    reference fixed, rel-L2 decays geometrically to ~1e-10 at p=12.

    Run standalone:  python -X utf8 apps/app5_benchmark_variants.py
    """
    from core import Yukawa3DFMM

    coords, charges = _protein(n_atoms=n_atoms, seed=seed)
    ref = _direct_debye_huckel(coords, charges, kappa=kappa)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-300:
        ref_norm = 1e-300

    print(f"\n=== Yukawa3D error-vs-p convergence "
          f"(N={n_atoms}, depth={depth}, kappa={kappa}, seed={seed}) ===")
    print(f"{'p':>4} {'rel-L2':>14} {'build+eval (s)':>16} {'note':>40}")
    print("-" * 80)
    prev_rel = None
    for p in ps:
        import time as _time
        t0 = _time.perf_counter()
        fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
        est = fmm.evaluate(coords, charges)
        dt = _time.perf_counter() - t0
        rel = float(np.linalg.norm(est - ref) / ref_norm)
        note = ""
        if prev_rel is not None and rel >= prev_rel:
            note = "no improvement (floor reached)"
        elif prev_rel is not None:
            note = f"~{rel / prev_rel:.2e}x vs prev"
        print(f"{p:>4} {rel:>14.4e} {dt:>16.4f} {note:>40}")
        prev_rel = rel
    print("-" * 80)
    print("Convergence rate: rel-L2 should drop ~1e-2 per +2 in p "
          "(order-(p+1) in cell radius).")
    return None


if __name__ == "__main__":
    run_app5_variants()
    run_convergence()
