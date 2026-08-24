"""External-reference and self-consistency cross-validation for the CANONICAL
adaptive FMM engine (``core.adaptive_fmm.AdaptiveFMM``, alias
``FastAdaptiveFMM``) after the consolidation of the two adaptive modules.

External reference status (honest statement, 2026-08-24, Windows machine)
--------------------------------------------------------------------------
The right external reference for this engine is a 2D Laplace/log FMM from
the Greengard-Gimbutas lineage, i.e. FMMLIB2D (Gimbutas & Greengard, 2012).
`pyfmmlib` on PyPI (Kloeckner's Python wrappers of fmmlib2d/fmmlib3d) DOES
ship exactly that 2D log-kernel FMM with our sign convention -- its
`lfmm2dpart` computes phi(x_i) = sum_{j != i} q_j log|x_i - x_j|, which is
bit-for-bit the kernel of `exact_direct_nbody_2d` -- but the package is
source-only on PyPI and its meson build REQUIRES a Fortran compiler
(ifort/ifx/gfortran/flang), none of which exists on this Windows machine,
and no Windows wheel is published.  Alternatives checked and rejected:

  * `pyfmmlib2d` (dbstein): GitHub-only, also gfortran+f2py -- same blocker.
  * `fmm2dpy`: no longer on PyPI ("no versions found").
  * `fmm3dpy` 2.1.0: HAS Windows wheels but is strictly 3D (Laplace 1/r,
    Helmholtz, Stokes in 3D) -- wrong kernel, cannot validate a 2D log FMM.
  * `jaxfmm` 0.3.3 (pure-Python wheel, installs): its Laplace kernel is the
    3D Coulomb kernel 1/(4 pi r) on (m, 3) point arrays -- wrong kernel.

Therefore the pyfmmlib cross-check below is written against the verified
upstream API (`pyfmmlib.fmm_part("pg", iprec=..., kernel=LaplaceKernel(),
sources=(N,2), mop_charge=q)`; gradient = grad phi, our forces = -grad) and
RUNS with real gates whenever pyfmmlib is importable (e.g. Linux CI or any
machine with gfortran), but SKIPS with this exact explanation where it is
not.  The always-running fallbacks below are, per CGR88 itself:

  1. Multipole-order convergence: the error must fall geometrically in p,
     like the Carrier, Greengard, & Rokhlin (1988) truncation bound
     O((r/R)^(p+1)) -- checked directly on a controlled two-box M2L and on
     the full engine over p in [4, 16].
  2. Translation-operator round-trip identities (P2M/M2M/M2L/L2L/L2P chain
     vs direct evaluation on controlled geometry).
  3. Agreement with the retained slow classical references
     (ClassicalAdaptiveFMM, TreeFreeElasticAdaptiveFMM, and the
     GreengardRokhlin87RegularFMM uniform engine).
  4. Direct O(N^2) agreement on uniform / two-cluster / spiral adaptive
     distributions with potentials AND forces.

Run standalone from the repo root:
    python -m tests.core.test_adaptive_fmm_reference
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from core.adaptive_fmm import (  # noqa: E402
    AdaptiveFMM,
    ClassicalAdaptiveFMM,
    TreeFreeElasticAdaptiveFMM,
    GreengardRokhlin87RegularFMM,
    p2m,
    m2m,
    m2l,
    l2l,
    l2p,
    l2p_force,
    m2p,
    p2p_potential_and_force,
)


# =============================================================================
# Distributions (clustered / adaptive-friendly, deterministic)
# =============================================================================

def _uniform(n, seed=1701):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, (n, 2)), rng.uniform(-1, 1, n)


def _two_cluster(n, seed=1702):
    rng = np.random.default_rng(seed)
    half = n // 2
    c1 = rng.normal([0.25, 0.25], 0.03, (half, 2))
    c2 = rng.normal([0.75, 0.75], 0.02, (n - half, 2))
    pts = np.clip(np.vstack([c1, c2]), 0.01, 0.99)
    q = rng.uniform(-1, 1, n)
    return pts, q


def _spiral(n, seed=1703):
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    r = 0.06 + 0.42 * t
    theta = 8.0 * np.pi * t + rng.normal(0, 0.05, n)
    pts = np.column_stack([0.5 + r * np.cos(theta),
                           0.5 + r * np.sin(theta)])
    pts = np.clip(pts, 0.02, 0.98)
    q = rng.uniform(-1, 1, n)
    return pts, q


def _clustered_multiscale(n, seed=707):
    """Same generator family as the core benchmark table."""
    rng = np.random.default_rng(seed)
    n1, n2, n3 = (max(1, int(n * f)) for f in (0.20, 0.30, 0.40))
    bg = max(0, n - (n1 + n2 + n3))
    pts = np.vstack([
        rng.random((n1, 2)) * 0.10 + 0.10,
        rng.random((n2, 2)) * 0.15 + 0.70,
        rng.random((n3, 2)) * 0.30 + 0.40,
        rng.random((bg, 2)) * 0.94 + 0.03 if bg else np.empty((0, 2)),
    ]).astype(np.float64)
    pts = np.clip(pts, 0.01, 0.99)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


DISTRIBUTIONS = {
    "uniform": _uniform,
    "two-cluster": _two_cluster,
    "spiral": _spiral,
}


# =============================================================================
# Direct reference (chunked: Python-loop exact_direct_* is O(N) python calls)
# =============================================================================

def _direct_potentials_chunked(positions, charges, block=2048):
    N = len(positions)
    pot = np.zeros(N)
    px, py = positions[:, 0], positions[:, 1]
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        dx = px[lo:hi, None] - px[None, :]
        dy = py[lo:hi, None] - py[None, :]
        r2 = dx * dx + dy * dy
        rows = np.arange(hi - lo)
        cols = np.arange(lo, hi)
        r2[rows, cols] = 1.0
        pot[lo:hi] = np.sum(charges[None, :] * 0.5 * np.log(r2), axis=1)
    return pot


def _direct_forces_chunked(positions, charges, block=2048):
    N = len(positions)
    fx = np.zeros(N)
    fy = np.zeros(N)
    px, py = positions[:, 0], positions[:, 1]
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        dx = px[lo:hi, None] - px[None, :]
        dy = py[lo:hi, None] - py[None, :]
        r2 = dx * dx + dy * dy
        rows = np.arange(hi - lo)
        cols = np.arange(lo, hi)
        self_mask = np.zeros((hi - lo, N), dtype=bool)
        self_mask[rows, cols] = True
        r2s = np.where(self_mask, 1.0, r2)
        inv = np.where(self_mask, 0.0, 1.0 / r2s)
        fx[lo:hi] = -np.sum(charges[None, :] * dx * inv, axis=1)
        fy[lo:hi] = -np.sum(charges[None, :] * dy * inv, axis=1)
    return fx, fy


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-300, np.linalg.norm(b)))


def _maxrel(a, b):
    denom = max(1e-300, float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / denom)


# =============================================================================
# 1. Multipole-order convergence (the CGR88 truncation bound)
# =============================================================================

def test_m2l_geometric_convergence_two_boxes():
    """Direct check of the Carrier, Greengard, & Rokhlin (1988) truncation
    behavior: for a source box of radius r whose multipole is translated to
    a local expansion at distance R, the error must fall geometrically in p
    with ratio ~ r/R per order.  Two boxes at List-2 separation (centers 2
    box-widths apart, r = sqrt(2)/2 box widths -> ratio ~ 0.354)."""
    rng = np.random.default_rng(11)
    src_c = 0.5 + 0.5j
    dst_c = 2.5 + 0.5j          # List-2 minimum separation: 2 box widths
    pts = (src_c.real + rng.uniform(-0.5, 0.5, (60, 2)))
    pts[:, 1] = src_c.imag + rng.uniform(-0.5, 0.5, 60)
    q = rng.uniform(-1, 1, 60)
    tgt = np.array([[dst_c.real + 0.2, dst_c.imag - 0.1],
                    [dst_c.real - 0.3, dst_c.imag + 0.25]])

    exact_pot, _, _ = p2p_potential_and_force(tgt, pts, q)
    errs = []
    for p in (4, 8, 12, 16, 20):
        m_src = p2m(pts, q, src_c, p=p)
        loc = m2l(m_src, src_c, dst_c, p=p)
        fmm_pot = np.array([l2p(loc, complex(*t), dst_c, p=p) for t in tgt])
        errs.append(_maxrel(fmm_pot, exact_pot))

    print("two-box M2L max-rel error by p:", ", ".join(
        f"p={p}:{e:.2e}" for p, e in zip((4, 8, 12, 16, 20), errs)))
    # geometric decay: every +4 orders must cut the error by >= 10x
    # (theory: 0.354^4 ~ 1.6e-2 per +4 orders; 10x is a conservative gate)
    for i in range(len(errs) - 1):
        assert errs[i + 1] < errs[i] / 10.0, (
            f"M2L error not falling geometrically: {errs}")
    assert errs[-1] < 1e-8


def test_engine_order_convergence_clustered():
    """Full canonical engine: rel-L2 vs direct must fall monotonically (up
    to saturation) as p increases over [4, 16], CGR88-style."""
    pts, q = _clustered_multiscale(2000)
    pot_ref = _direct_potentials_chunked(pts, q)
    errs = []
    for p in (4, 6, 8, 10, 12, 16):
        eng = AdaptiveFMM(max_leaf_particles=24, base_depth=2,
                          max_depth=9, p=p)
        pot = eng.evaluate(pts, q, compute_forces=False)
        errs.append(_rel(pot, pot_ref))
    print("engine rel-L2 by p:", ", ".join(
        f"p={p}:{e:.2e}" for p, e in zip((4, 6, 8, 10, 12, 16), errs)))
    for i in range(len(errs) - 1):
        assert errs[i + 1] < errs[i], (
            f"engine error must decrease with p: {errs}")
    # ~0.35^2 per +2 orders -> >= 5x per step while above round-off
    for i in range(2, len(errs) - 1):
        assert errs[i + 1] < errs[i] / 5.0, (
            f"engine convergence flatter than the CGR88 bound: {errs}")
    assert errs[-1] < 1e-8


# =============================================================================
# 2. Translation-operator round-trip identities
# =============================================================================

def test_translation_round_trip_identities():
    """The exact CGR88 translation chain must be evaluation-invariant:

       P2M(leaf) -M2M-> parent -M2L-> distant child local -L2L-> grandchild
       local -L2P-> target  ==  direct P2P potential at the target.

    Both the potential and the force (via l2p_force) must match the direct
    sums to truncation level (the algebraic identities are exact; the only
    error is the p=16 truncation of the source multipole, ~1e-9 here)."""
    rng = np.random.default_rng(23)
    leaf_c = 0.5 + 0.5j
    par_c = 0.75 + 0.75j              # parent center (leaf box inside it)
    tgt_box_c = 3.0 + 1.0j          # M2L destination (>= 2 widths away)
    tgt_child_c = 3.25 + 1.25j      # L2L shift inside the M2L domain
    pts = np.column_stack([leaf_c.real + rng.uniform(-0.25, 0.25, 40),
                           leaf_c.imag + rng.uniform(-0.25, 0.25, 40)])
    q = rng.uniform(-1, 1, 40)
    targets = np.array([[tgt_child_c.real + 0.05, tgt_child_c.imag - 0.02],
                        [tgt_child_c.real - 0.04, tgt_child_c.imag + 0.06]])

    p = 16
    m_leaf = p2m(pts, q, leaf_c, p=p)
    m_par = m2m(m_leaf, leaf_c, par_c, p=p)
    loc = m2l(m_par, par_c, tgt_box_c, p=p)
    loc_child = l2l(loc, tgt_box_c, tgt_child_c, p=p)

    pot_direct, fx_direct, fy_direct = p2p_potential_and_force(
        targets, pts, q)
    for i, t in enumerate(targets):
        tz = complex(*t)
        pot_chain = l2p(loc_child, tz, tgt_child_c, p=p)
        fx_chain, fy_chain = l2p_force(loc_child, tz, tgt_child_c, p=p)
        assert abs(pot_chain - pot_direct[i]) < 1e-8, (
            f"potential round-trip mismatch at target {i}: "
            f"{pot_chain} vs {pot_direct[i]}")
        assert abs(fx_chain - fx_direct[i]) < 1e-8
        assert abs(fy_chain - fy_direct[i]) < 1e-8

    # M2M/M2L evaluation equivalence: shifting then evaluating at a distant
    # point gives the same multipole field regardless of expansion center.
    far = 5.0 + 4.0j
    pot_leaf, deriv_leaf = m2p(m_leaf, leaf_c, far, p=p)
    pot_par2, deriv_par2 = m2p(m_par, par_c, far, p=p)
    assert abs(pot_leaf - pot_par2) < 1e-10
    assert abs(deriv_leaf - deriv_par2) < 1e-10


def test_m2l_shift_invariance():
    """M2L to a center followed by L2L to a child == M2L directly to the
    child center (local expansions are Taylor series; shifting is exact)."""
    rng = np.random.default_rng(29)
    src_c = 0.0 + 0.0j
    dst_c = 3.0 + 0.0j
    child_c = 3.2 + 0.15j
    pts = rng.uniform(-0.4, 0.4, (30, 2))
    q = rng.uniform(-1, 1, 30)
    p = 14
    m_src = p2m(pts, q, src_c, p=p)
    loc_direct = m2l(m_src, src_c, child_c, p=p)
    loc_shifted = l2l(m2l(m_src, src_c, dst_c, p=p), dst_c, child_c, p=p)
    t = child_c + 0.1 - 0.07j
    a = l2p(loc_direct, t, child_c, p=p)
    b = l2p(loc_shifted, t, child_c, p=p)
    assert abs(a - b) < 1e-11, f"local shift invariance broken: {a} vs {b}"


# =============================================================================
# 3. Agreement with the retained slow classical reference engines
# =============================================================================

def test_canonical_vs_classical_engines():
    """The canonical engine must agree with BOTH retained classical
    references (per-box tree and funnel-hash variants) at truncation level,
    on an adaptive clustered scene, for potentials and forces."""
    pts, q = _clustered_multiscale(600, seed=99)
    pot_ref = _direct_potentials_chunked(pts, q)
    fx_ref, fy_ref = _direct_forces_chunked(pts, q)

    fast = AdaptiveFMM(max_leaf_particles=24, base_depth=2, max_depth=7,
                       p=10)
    pot_fast, fx_fast, fy_fast = fast.evaluate(pts, q, compute_forces=True)

    slow = ClassicalAdaptiveFMM(max_leaf_particles=24, max_depth=8, p=10)
    pot_slow = slow.evaluate(pts, q, compute_forces=False)

    hash_eng = TreeFreeElasticAdaptiveFMM(max_leaf_particles=24,
                                          base_depth=2, max_depth=7, p=10)
    pot_hash = hash_eng.evaluate(pts, q, compute_forces=False)

    r_direct = _rel(pot_fast, pot_ref)
    r_classical = _rel(pot_fast, pot_slow)
    r_hash = _rel(pot_fast, pot_hash)
    print(f"canonical vs direct {r_direct:.2e}; vs ClassicalAdaptiveFMM "
          f"{r_classical:.2e}; vs TreeFreeElasticAdaptiveFMM {r_hash:.2e}; "
          f"force vs direct {_rel(fx_fast, fx_ref):.2e}")
    assert r_direct < 5e-6
    assert r_classical < 2e-5   # both approximate direct; mutual diff ~ sum
    assert r_hash < 2e-5
    assert _rel(fx_fast, fx_ref) < 5e-4


def test_canonical_vs_regular_fmm_uniform():
    """On a uniform distribution the canonical adaptive engine must also
    agree with the independent Greengard & Rokhlin (1987) uniform-grid
    engine (different code path: fixed-depth tree, no 2:1 balancing)."""
    pts, q = _uniform(600)
    pot_ref = _direct_potentials_chunked(pts, q)
    fast = AdaptiveFMM(max_leaf_particles=24, base_depth=2, max_depth=7,
                       p=10)
    reg = GreengardRokhlin87RegularFMM(depth=4, p=10)
    pot_fast = fast.evaluate(pts, q, compute_forces=False)
    pot_reg = reg.evaluate(pts, q, compute_forces=False)
    print(f"uniform: canonical { _rel(pot_fast, pot_ref):.2e}, "
          f"GR87-regular {_rel(pot_reg, pot_ref):.2e} vs direct; "
          f"mutual {_rel(pot_fast, pot_reg):.2e}")
    assert _rel(pot_fast, pot_ref) < 5e-6
    assert _rel(pot_reg, pot_ref) < 1e-5
    assert _rel(pot_fast, pot_reg) < 2e-5


# =============================================================================
# 4. Direct O(N^2) agreement across adaptive distributions and orders
# =============================================================================

@pytest.mark.parametrize("dist", list(DISTRIBUTIONS))
@pytest.mark.parametrize("p", [8, 16])
def test_canonical_vs_direct_distributions(dist, p):
    """Uniform, two-cluster, and spiral scenes at N=2048: the canonical
    engine's potentials AND forces vs chunked exact direct O(N^2)."""
    pts, q = DISTRIBUTIONS[dist](2048)
    pot_ref = _direct_potentials_chunked(pts, q)
    fx_ref, fy_ref = _direct_forces_chunked(pts, q)
    eng = AdaptiveFMM(max_leaf_particles=24, base_depth=2, max_depth=9, p=p)
    pot, fx, fy = eng.evaluate(pts, q, compute_forces=True)
    rp, rf = _rel(pot, pot_ref), _rel(fx, fx_ref)
    print(f"{dist:12s} p={p:2d}: pot rel-L2 {rp:.2e} max-rel "
          f"{_maxrel(pot, pot_ref):.2e}; force rel-L2 {rf:.2e}")
    gate_p, gate_f = (5e-6, 5e-4) if p >= 8 else (1e-4, 1e-2)
    assert rp < gate_p, f"{dist} p={p} potential error {rp:.3e}"
    assert rf < gate_f, f"{dist} p={p} force error {rf:.3e}"


# =============================================================================
# 5. pyfmmlib external-reference cross-validation (runs when pyfmmlib is
#    installed; skips with the documented reason otherwise)
# =============================================================================

def test_pyfmmlib_external_cross_validation():
    """Compare against FMMLIB2D through Kloeckner's pyfmmlib wrappers
    (Gimbutas & Greengard, 2012): potentials AND forces on uniform /
    two-cluster / spiral distributions at N in [2048, 32000], our engine at
    multipole orders p in [8, 16], against pyfmmlib (iprec=3, tol 5e-10)
    and chunked direct O(N^2).

    SKIPS when pyfmmlib is not importable.  On this Windows machine pyfmmlib
    cannot be installed: it is an sdist-only package whose meson build needs
    a Fortran compiler (ifort/ifx/gfortran/flang), none of which is present,
    and no Windows wheel is published.  All pip-installable alternatives
    were checked and are unusable for a 2D log kernel (see module docstring
    for the itemized list)."""
    pytest.importorskip(
        "pyfmmlib",
        reason="pyfmmlib not installed: sdist-only Fortran build "
               "(needs ifort/gfortran); no Windows wheel. See module "
               "docstring for the alternatives that were checked.")
    from pyfmmlib import fmm_part, LaplaceKernel

    # API sanity against direct on a small case FIRST: the wrapper returns
    # pot = sum q log r and grad = grad phi; our forces are -grad.  If an
    # upstream API change broke the semantics, fail loudly here rather
    # than silently "passing".
    pts, q = _uniform(64, seed=5)
    pot_ref = _direct_potentials_chunked(pts, q)
    fx_ref, _ = _direct_forces_chunked(pts, q)
    pot_py, grad_py = fmm_part("pg", iprec=3, kernel=LaplaceKernel(),
                               sources=pts, mop_charge=q)
    assert _rel(np.asarray(pot_py).real, pot_ref) < 1e-10, (
        "pyfmmlib potential convention mismatch (expected sum q ln r)")
    assert _rel(-np.asarray(grad_py)[:, 0].real, fx_ref) < 1e-8, (
        "pyfmmlib gradient convention mismatch (expected grad phi; "
        "forces = -grad)")

    for dist in ("uniform", "two-cluster", "spiral"):
        for n, p in ((2048, 8), (2048, 16), (32000, 16)):
            pts, q = DISTRIBUTIONS[dist](n)
            pot_ref = _direct_potentials_chunked(pts, q)
            fx_ref, fy_ref = _direct_forces_chunked(pts, q)
            eng = AdaptiveFMM(max_leaf_particles=24, base_depth=2,
                              max_depth=9, p=p)
            pot, fx, fy = eng.evaluate(pts, q, compute_forces=True)
            pot_py, grad_py = fmm_part(
                "pg", iprec=3, kernel=LaplaceKernel(),
                sources=pts, mop_charge=q)
            pot_py = np.asarray(pot_py).real
            fpy = -np.asarray(grad_py).real  # (N, 2) grad -> force

            ours_vs_direct = _rel(pot, pot_ref)
            ours_vs_py = _rel(pot, pot_py)
            py_vs_direct = _rel(pot_py, pot_ref)
            f_ours_vs_py = max(_rel(fx, fpy[:, 0]), _rel(fy, fpy[:, 1]))
            print(f"{dist:12s} N={n:6d} p={p:2d}: "
                  f"ours-vs-direct {ours_vs_direct:.2e}, "
                  f"ours-vs-pyfmmlib {ours_vs_py:.2e} "
                  f"(max-rel {_maxrel(pot, pot_py):.2e}), "
                  f"pyfmmlib-vs-direct {py_vs_direct:.2e}, "
                  f"force ours-vs-pyfmmlib {f_ours_vs_py:.2e}")
            assert py_vs_direct < 1e-9, (
                f"pyfmmlib itself disagrees with direct at {dist} N={n}")
            assert ours_vs_direct < 5e-6
            assert ours_vs_py < 5e-6
            assert f_ours_vs_py < 5e-4


if __name__ == "__main__":
    rc = 0
    for name, fn in [
        ("m2l_geometric_convergence_two_boxes",
         test_m2l_geometric_convergence_two_boxes),
        ("engine_order_convergence_clustered",
         test_engine_order_convergence_clustered),
        ("translation_round_trip_identities",
         test_translation_round_trip_identities),
        ("m2l_shift_invariance", test_m2l_shift_invariance),
        ("canonical_vs_classical_engines",
         test_canonical_vs_classical_engines),
        ("canonical_vs_regular_fmm_uniform",
         test_canonical_vs_regular_fmm_uniform),
    ]:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"[FAIL] {name}: {e}")
    for dist in DISTRIBUTIONS:
        for p in (8, 16):
            try:
                test_canonical_vs_direct_distributions(dist, p)
                print(f"[PASS] canonical_vs_direct({dist}, p={p})")
            except Exception as e:  # noqa: BLE001
                rc = 1
                print(f"[FAIL] canonical_vs_direct({dist}, p={p}): {e}")
    try:
        test_pyfmmlib_external_cross_validation()
        print("[PASS] pyfmmlib_external_cross_validation")
    except pytest.skip.Exception:
        print("[SKIP] pyfmmlib not installed (see docstring)")
    sys.exit(rc)
