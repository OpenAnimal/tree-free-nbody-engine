"""Round-10 Wave D regression tests: bioinformatics + environmental_modeling.

Each test pins a defect found by independent-oracle verification during
Round-10 Wave D (one-off review probes, since removed). Tests fail if the
bug is reintroduced.
"""
import os
import sys

import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bioinformatics.core.fast_multipole_kernel import (
    TreeFreeBioFMM,
    ScreenedKernelType,
    COULOMB_CONSTANT_KCAL,
)


def _direct_dh_potential(coords, charges, kappa=0.127, eps=78.5):
    """Independent O(N^2) Debye-Hueckel / Yukawa potential reference."""
    N = len(coords)
    out = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = np.linalg.norm(coords[i] - coords[j])
            out[i] += charges[j] * np.exp(-kappa * r) / r
    return out * COULOMB_CONSTANT_KCAL / eps


def _direct_dh_force(coords, charges, kappa=0.127, eps=78.5):
    """Independent O(N^2) force reference: F_i = -q_i grad_i sum_{j!=i} q_j K(r)."""
    N = len(coords)
    out = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = coords[i] - coords[j]
            r = np.linalg.norm(d)
            e = np.exp(-kappa * r)
            dK_dr = -e * (kappa / r + 1.0 / r ** 2)
            out[i] += -charges[i] * charges[j] * dK_dr * d / r
    return out * COULOMB_CONSTANT_KCAL / eps


def test_yukawa_kernel_near_field_not_dropped():
    """R10-D1: ScreenedKernelType.YUKAWA used to silently skip the entire
    near-field block (and all far-field forces), returning ~0.1% of the true
    potential and identically-zero forces. The YUKAWA kernel is
    exp(-kappa*r)/r (same math as DEBYE_HUCKEL), so its results must match the
    direct Yukawa sum."""
    rng = np.random.RandomState(20260822)
    pts = rng.uniform(0.0, 6.0, size=(12, 3))  # all inside one 8 A cell -> near field
    q = rng.uniform(-1.0, 1.0, size=12)

    eng = TreeFreeBioFMM(
        cell_size=8.0, kappa=0.127, dielectric_water=78.5,
        kernel_type=ScreenedKernelType.YUKAWA,
    )
    pot, forces, _ = eng.evaluate(pts, q, compute_forces=True)

    ref = _direct_dh_potential(pts, q)
    rel = np.linalg.norm(pot - ref) / np.linalg.norm(ref)
    assert rel < 5e-3, f"YUKAWA near-field potential rel-L2 {rel:.3e} >= 5e-3"

    refF = _direct_dh_force(pts, q)
    relF = np.linalg.norm(forces - refF) / np.linalg.norm(refF)
    assert relF < 5e-3, f"YUKAWA forces rel-L2 {relF:.3e} >= 5e-3"


def test_gb_and_yukawa_far_field_forces_present():
    """R10-D1b: the far-field `else` branch (GENERALIZED_BORN / YUKAWA) used to
    compute potentials but never forces, silently returning zero far-field
    force contributions. With two well-separated clusters, far-field forces
    must be nonzero and point along the correct separation direction (like
    charges repel across the two clusters)."""
    rng = np.random.RandomState(20260822)
    c1 = rng.uniform(-1.5, 1.5, size=(6, 3)) + np.array([0.0, 0.0, 0.0])
    c2 = rng.uniform(-1.5, 1.5, size=(6, 3)) + np.array([30.0, 0.0, 0.0])
    pts = np.vstack([c1, c2])
    q = np.ones(12)  # all like charges

    eng = TreeFreeBioFMM(
        cell_size=4.0, kappa=0.127,
        kernel_type=ScreenedKernelType.GENERALIZED_BORN,
    )
    _, forces, _ = eng.evaluate(pts, q, compute_forces=True)

    # net force on each cluster must be nonzero and repulsive (cluster 1 pushed
    # towards -x, cluster 2 towards +x)
    f1 = np.sum(forces[:6], axis=0)
    f2 = np.sum(forces[6:], axis=0)
    assert np.linalg.norm(f1) > 1e-6, (
        f"GB far-field net force is zero (|f1|={np.linalg.norm(f1):.3e})"
    )
    assert f1[0] < 0.0 < f2[0], (
        f"GB far-field force direction wrong: f1_x={f1[0]:.3e}, f2_x={f2[0]:.3e}"
    )


def test_binding_pocket_ray_stencil_symmetric():
    """R10-D2: with the default ray_directions=14 the ray stencil used to be
    the first 14 directions of a (-1,0,1)^3 enumeration ordered by dx — 9 rays
    pointing toward -x and none toward +x. The concavity/burial score was
    therefore mirror-asymmetric. A symmetric stencil must satisfy
    sum(dirs) == 0 and contain the antipode of every direction."""
    from bioinformatics.binding_pocket_detector import BindingPocketDetector

    det = BindingPocketDetector(ray_directions=14)
    rv = det.ray_vectors
    assert np.allclose(rv.sum(axis=0), 0.0, atol=1e-12), (
        f"ray stencil is anisotropic: sum={rv.sum(axis=0)}"
    )
    # every direction has its antipode in the set
    for d in rv:
        dots = rv @ (-d)
        assert dots.max() > 1.0 - 1e-9, f"direction {d} has no antipode in stencil"

    # end-to-end mirror symmetry: reflected protein must give mirrored pockets
    from bioinformatics.pdb_loader import generate_synthetic_protein
    prot = generate_synthetic_protein(n_atoms=400, seed=42)
    mirrored = prot.copy()
    mirrored.coords[:, 0] *= -1.0

    det = BindingPocketDetector(grid_spacing=1.5, min_pocket_points=8)
    r1 = det.detect_pockets(prot)
    r2 = det.detect_pockets(mirrored)
    s1 = sorted(p["druggability_score"] for p in r1["pockets"])
    s2 = sorted(p["druggability_score"] for p in r2["pockets"])
    assert len(s1) == len(s2), f"pocket count changed under mirroring: {len(s1)} vs {len(s2)}"
    assert np.allclose(s1, s2, atol=1e-6), (
        f"pocket scores not mirror-symmetric: {s1} vs {s2}"
    )
    c1 = sorted((np.round(p["center"], 6)).tolist() for p in r1["pockets"])
    c2 = sorted((np.round([-p["center"][0], p["center"][1], p["center"][2]], 6)).tolist()
                for p in r2["pockets"])
    assert c1 == c2, "pocket centers not mirrored under x-reflection"


def test_binding_pocket_probe_radius_used():
    """R10-D3: the documented water probe_radius (1.4 A) used to be stored but
    never used — the clash test compared probe points against bare vdW radii.
    A larger probe must reject strictly more candidate points."""
    from bioinformatics.binding_pocket_detector import BindingPocketDetector
    from bioinformatics.pdb_loader import generate_synthetic_protein

    prot = generate_synthetic_protein(n_atoms=400, seed=42)
    small = BindingPocketDetector(probe_radius=0.01).detect_pockets(prot)
    big = BindingPocketDetector(probe_radius=1.4).detect_pockets(prot)
    assert big["total_pocket_points"] < small["total_pocket_points"], (
        f"probe_radius has no effect: {big['total_pocket_points']} pocket points "
        f"at r=1.4 vs {small['total_pocket_points']} at r=0.01"
    )


def _lj_system():
    """3 neutral atoms, different chains (no bonds): one close pair at 4 A
    (inside the LJ cutoff 2.5*3.4 = 8.5 A) and one far atom at 100 A."""
    from bioinformatics.pdb_loader import MolecularSystem
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    return MolecularSystem(
        coords=coords, charges=np.zeros(3), radii=np.full(3, 1.7),
        masses=np.full(3, 12.0), atom_names=["C", "C", "C"],
        residue_names=["GLY", "GLY", "GLY"],
        residue_ids=np.array([1, 2, 3], dtype=np.int32),
        chain_ids=["A", "B", "C"])


def test_md_lj_forces_present_and_correct():
    """R10-D4: compute_forces advertised 'F_LJ_sterics' in the force model but
    never computed any LJ term (e_lj was always 0). The steric force between
    the close neutral pair must equal the 12-6 LJ force with the engine's
    sigma/eps and cutoff, and the far atom must feel nothing."""
    from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine

    md = MacromolecularMDEngine(_lj_system(), temperature_kelvin=10.0,
                                timestep_fs=0.1)
    forces, energy = md.compute_forces()

    sigma, eps = md.lj_sigma, md.lj_eps
    r = 4.0
    e_pair = 4.0 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    f_mag = 24.0 * eps / r * (2.0 * (sigma / r) ** 12 - (sigma / r) ** 6)

    assert energy["e_lj"] != 0.0, "e_lj reported as 0: LJ term not computed"
    assert abs(energy["e_lj"] - e_pair) < 1e-12 * max(1.0, abs(e_pair)), (
        f"e_lj {energy['e_lj']} != direct pair energy {e_pair}"
    )
    # atom 0 pushed toward -x, atom 1 toward +x, atom 2 feels nothing
    assert np.allclose(forces[0], [-f_mag, 0.0, 0.0]), forces[0]
    assert np.allclose(forces[1], [f_mag, 0.0, 0.0]), forces[1]
    assert np.allclose(forces[2], [0.0, 0.0, 0.0]), forces[2]


def test_md_lj_many_body_vs_direct_all_pairs():
    """R10-D4 oracle: cell-list LJ must reproduce the direct all-pairs 12-6
    sum (within the 2.5 sigma cutoff) on a random neutral cluster."""
    from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine
    from bioinformatics.pdb_loader import MolecularSystem

    rng = np.random.RandomState(20260822)
    n = 40
    coords = rng.uniform(0.0, 12.0, size=(n, 3))
    sysm = MolecularSystem(
        coords=coords, charges=np.zeros(n), radii=np.full(n, 1.7),
        masses=np.full(n, 12.0), atom_names=["C"] * n,
        residue_names=["GLY"] * n, residue_ids=np.arange(1, n + 1, dtype=np.int32),
        chain_ids=["A%d" % i for i in range(n)])
    md = MacromolecularMDEngine(sysm, temperature_kelvin=10.0, timestep_fs=0.1)
    forces, energy = md.compute_forces()

    sigma, eps = md.lj_sigma, md.lj_eps
    rc = 2.5 * sigma
    fmax = md.lj_fmax
    # independent bisection for the cap radius r_c with F_12-6(r_c) = fmax
    lo, hi = 0.5 * sigma, 2.0 * sigma
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        x6 = (sigma / mid) ** 6
        f = 24.0 * eps / mid * (2.0 * x6 * x6 - x6)
        if f > fmax:
            lo = mid
        else:
            hi = mid
    rcap = 0.5 * (lo + hi)
    v_rcap = 4.0 * eps * ((sigma / rcap) ** 12 - (sigma / rcap) ** 6)
    ref_e = 0.0
    ref_f = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = coords[i] - coords[j]
            r = np.linalg.norm(d)
            if r >= rc or r < 1e-12:
                continue
            sr6 = (sigma / r) ** 6
            fmag = 24.0 * eps / r * (2.0 * sr6 * sr6 - sr6)
            if fmag > fmax:
                fmag = fmax
                e = v_rcap + fmax * (rcap - r)
            else:
                e = 4.0 * eps * (sr6 * sr6 - sr6)
            ref_e += e
            ref_f[i] += fmag * d / r
            ref_f[j] -= fmag * d / r
    assert abs(energy["e_lj"] - ref_e) < 1e-10 * max(1.0, abs(ref_e)), (
        f"e_lj {energy['e_lj']} != direct {ref_e}"
    )
    assert np.linalg.norm(forces - ref_f) < 1e-10 * max(1e-12, np.linalg.norm(ref_f)), (
        f"LJ forces differ from direct: rel "
        f"{np.linalg.norm(forces - ref_f) / np.linalg.norm(ref_f):.2e}"
    )


def test_md_lj_capped_force_matches_gradient():
    """R10-D4 oracle: on the capped branch (r < r_c) the reported e_lj must be
    consistent with the capped force, i.e. F = -dV/dr = fmax toward separation.
    Two coincident-ish neutral atoms (r = 1.2 A) must repel with exactly
    lj_fmax each along the separation axis."""
    from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine

    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    from bioinformatics.pdb_loader import MolecularSystem
    sysm = MolecularSystem(
        coords=coords, charges=np.zeros(2), radii=np.full(2, 1.7),
        masses=np.full(2, 12.0), atom_names=["C", "C"],
        residue_names=["GLY", "GLY"], residue_ids=np.array([1, 2], dtype=np.int32),
        chain_ids=["A", "B"])
    md = MacromolecularMDEngine(sysm, temperature_kelvin=10.0, timestep_fs=0.1)
    forces, energy = md.compute_forces()
    fmax = md.lj_fmax
    assert np.allclose(forces[0], [-fmax, 0.0, 0.0]), forces[0]
    assert np.allclose(forces[1], [fmax, 0.0, 0.0]), forces[1]
    # FD consistency of the capped energy branch: V(r) - V(r+dr) ~ fmax*dr
    rcap = md._lj_rcap
    sigma, eps = md.lj_sigma, md.lj_eps
    v_rcap = 4.0 * eps * ((sigma / rcap) ** 12 - (sigma / rcap) ** 6)
    assert abs(energy["e_lj"] - (v_rcap + fmax * (rcap - 1.2))) < 1e-9


def test_md_stable_with_lj():
    """R10-D4 stability: with the force-capped LJ active,
    (a) a clash-free chain at 1 fs stays near the target temperature with
        <5% energy drift, and
    (b) the clashed synthetic protein (min pair distance ~1.2 A) stays
        bounded (no integrator blow-up, finite energies) while it releases
        the stored clash energy. An uncapped r^-12 on (b) reaches T ~ 1e14 K;
        the pre-fix engine reported a fake ~300 K because sterics were absent.
    """
    from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine
    from bioinformatics.pdb_loader import MolecularSystem, generate_synthetic_protein

    # (a) clean extended chain: 40 atoms, 4.0 A spacing
    n = 40
    coords = np.stack([4.0 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    chain = MolecularSystem(
        coords=coords, charges=np.zeros(n), radii=np.full(n, 1.7),
        masses=np.full(n, 12.0), atom_names=["C"] * n,
        residue_names=["GLY"] * n, residue_ids=np.arange(1, n + 1, dtype=np.int32),
        chain_ids=["A"] * n)
    md = MacromolecularMDEngine(chain, temperature_kelvin=300.0,
                                friction_gamma=0.0, timestep_fs=1.0)
    hist = md.run(num_steps=30)
    temps = [h["temperature_k"] for h in hist]
    assert all(np.isfinite(t) for t in temps)
    assert max(temps) < 600.0 and min(temps) > 100.0, (
        f"clean chain left the thermal band: {['%.0f' % t for t in temps]}"
    )
    e0 = hist[0]["e_total"]
    drift = max(abs(h["e_total"] - e0) / max(1e-9, abs(e0)) for h in hist)
    assert drift < 0.05, f"energy drift {drift:.3e} >= 5% on clean chain"

    # (b) clashed synthetic structure: bounded, finite, energy-conserving
    prot = generate_synthetic_protein(n_atoms=120, seed=5)
    md2 = MacromolecularMDEngine(prot, temperature_kelvin=300.0,
                                 friction_gamma=0.0, timestep_fs=1.0)
    hist2 = md2.run(num_steps=30)
    temps2 = [h["temperature_k"] for h in hist2]
    assert all(np.isfinite(t) for t in temps2), "non-finite temperature"
    assert max(temps2) < 1e5, f"integrator blew up: max T {max(temps2):.3e} K"
    e02 = hist2[0]["e_total"]
    drift2 = max(abs(h["e_total"] - e02) / max(1e-9, abs(e02)) for h in hist2)
    assert drift2 < 0.05, f"energy drift on clashed input {drift2:.3e} >= 5%"
