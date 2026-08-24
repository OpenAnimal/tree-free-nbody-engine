"""Cross-validation for the level-batched vectorized adaptive FMM engine
(``core.adaptive_fmm_fast.FastAdaptiveFMM``) against exact direct summation,
the classical CGR88 engines, and 2:1 balance invariants.

Run standalone from the repo root:
    python -m tests.core.test_adaptive_fmm_fast
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from core.adaptive_fmm import (  # noqa: E402
    TreeFreeElasticAdaptiveFMM,
    exact_direct_nbody_2d,
    exact_direct_nbody_forces_2d,
)
from core.adaptive_fmm_fast import FastAdaptiveFMM, _m2l_matrix  # noqa: E402


def _clustered(n, seed=707):
    """Clustered multi-scale distribution (same generator family as the
    core benchmark table)."""
    rng = np.random.default_rng(seed)
    n1 = max(1, int(n * 0.20))
    n2 = max(1, int(n * 0.30))
    n3 = max(1, int(n * 0.40))
    bg = max(0, n - (n1 + n2 + n3))
    pts = np.vstack([
        rng.random((n1, 2)) * 0.10 + 0.10,
        rng.random((n2, 2)) * 0.15 + 0.70,
        rng.random((n3, 2)) * 0.30 + 0.40,
        rng.random((bg, 2)) * 0.94 + 0.03 if bg else np.empty((0, 2)),
    ]).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-300, np.linalg.norm(b)))


def test_m2l_matrix_matches_classical_operator():
    """The per-offset M2L matrix must reproduce the scalar CGR88 m2l."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.3, 0.3, (25, 2))
    q = rng.uniform(-1, 1, 25)
    from core.adaptive_fmm import p2m
    a = p2m(pts, q, 0.0 + 0.0j, p=10)
    for delta in (complex(2.5, 1.7), complex(-1.2, 3.0), complex(4.0, -0.5)):
        c_ref = m2l_ref(a, delta)
        c_mat = a @ _m2l_matrix(delta, 10).T
        err = float(np.max(np.abs(c_ref - c_mat)))
        assert err < 1e-10, f"M2L matrix mismatch for delta={delta}: {err}"


def m2l_ref(a, delta):
    from core.adaptive_fmm import m2l
    return m2l(a, 0 + 0j, delta, p=10)


def test_potentials_and_forces_vs_direct():
    for n, leaf in ((180, 8), (600, 24)):
        pts, q = _clustered(n, seed=42 + n)
        pot_ref = exact_direct_nbody_2d(pts, q)
        fx_ref, fy_ref = exact_direct_nbody_forces_2d(pts, q)
        eng = FastAdaptiveFMM(max_leaf_particles=leaf, base_depth=2,
                              max_depth=7, p=10)
        pot, fx, fy = eng.evaluate(pts, q, compute_forces=True)
        rp = _rel(pot, pot_ref)
        rf = _rel(fx, fx_ref)
        assert rp < 5e-6, f"potential rel-L2 {rp} at N={n}"
        assert rf < 5e-4, f"force rel-L2 {rf} at N={n}"


def test_agreement_with_classical_engine():
    """The fast engine must agree with the classical funnel-hash adaptive
    engine to truncation level (both approximate the same direct sum; their
    mutual difference is bounded by ~the sum of both truncation errors)."""
    pts, q = _clustered(400, seed=99)
    classical = TreeFreeElasticAdaptiveFMM(max_leaf_particles=24,
                                           base_depth=2, max_depth=7, p=10)
    pot_classical = classical.evaluate(pts, q, compute_forces=False)
    fast = FastAdaptiveFMM(max_leaf_particles=24, base_depth=2,
                           max_depth=7, p=10)
    pot_fast = fast.evaluate(pts, q, compute_forces=False)
    pot_ref = exact_direct_nbody_2d(pts, q)
    assert _rel(pot_fast, pot_ref) < 5e-6
    assert _rel(pot_classical, pot_ref) < 5e-6
    both = _rel(pot_fast, pot_classical)
    assert both < 2e-5, f"fast vs classical divergence {both}"


def test_two_to_one_balance_invariant():
    """After a build, no two touching leaves may differ by more than one
    level (checked on the finest-grid owner map)."""
    pts, q = _clustered(2000, seed=707)
    eng = FastAdaptiveFMM(max_leaf_particles=24, base_depth=2, max_depth=9,
                          p=10)
    eng.evaluate(pts, q, compute_forces=False)
    n = eng.n_cells
    leaf = eng._leaf[:n]
    L = int(eng._lvl[:n].max())
    owner = np.full((1 << L, 1 << L), -1, dtype=np.int64)
    for c in range(n):
        if leaf[c]:
            s = L - int(eng._lvl[c])
            x0, y0 = int(eng._cx[c]) << s, int(eng._cy[c]) << s
            owner[x0:x0 + (1 << s), y0:y0 + (1 << s)] = c
    for c in range(n):
        if not leaf[c]:
            continue
        s = L - int(eng._lvl[c])
        x0, y0 = int(eng._cx[c]) << s, int(eng._cy[c]) << s
        sz = 1 << s
        ring = owner[max(0, x0 - 1):x0 + sz + 1, max(0, y0 - 1):y0 + sz + 1]
        for nbc in np.unique(ring):
            if nbc >= 0 and nbc != c:
                assert abs(int(eng._lvl[nbc]) - int(eng._lvl[c])) <= 1, (
                    f"2:1 violation: leaf {c} (lvl {eng._lvl[c]}) touches "
                    f"leaf {nbc} (lvl {eng._lvl[nbc]})")
    # no duplicate cells at the same (level, ix, iy)
    keys = eng._lvl[:n] * 100000 + eng._cx[:n] * 257 + eng._cy[:n]
    assert len(np.unique(keys)) == n, "duplicate (level, ix, iy) cells"
    # every cell is reachable from its occupancy grid
    ingrid = set()
    for g in eng._occ.values():
        ingrid.update(int(v) for v in g[g >= 0])
    assert ingrid == set(range(n)), "cells missing from occupancy grids"


def test_no_duplicate_cells_in_hash_index():
    pts, q = _clustered(500, seed=5)
    eng = FastAdaptiveFMM(max_leaf_particles=16, base_depth=2, max_depth=7,
                          p=8)
    eng.evaluate(pts, q, compute_forces=False)
    keys = eng.cell_keys
    assert len(keys) == len(set(keys)) == eng.n_cells
    # every key resolves back to a valid cell id
    for c, key in enumerate(keys):
        v, _ = eng.hash_table.lookup(key)
        assert v == c


def test_single_cell_and_tiny_inputs():
    # distinct positions only: the exact reference has no softening and
    # returns -inf for coincident particles
    pts = np.array([[0.5, 0.5], [0.52, 0.5], [0.51, 0.49]])
    q = np.array([1.0, -1.0, 0.5])
    eng = FastAdaptiveFMM(max_leaf_particles=24, base_depth=2, max_depth=6,
                          p=8)
    pot = eng.evaluate(pts, q, compute_forces=False)
    ref = exact_direct_nbody_2d(pts, q)
    assert _rel(pot, ref) < 1e-6
    # empty input (both force and potential modes)
    pot_e, fx_e, fy_e = eng.evaluate(np.empty((0, 2)), np.empty(0))
    assert len(pot_e) == len(fx_e) == len(fy_e) == 0
    assert len(eng.evaluate(np.empty((0, 2)), np.empty(0),
                            compute_forces=False)) == 0


def test_uniform_distribution():
    rng = np.random.default_rng(17)
    pts = rng.uniform(0.05, 0.95, (800, 2))
    q = rng.uniform(-1, 1, 800)
    eng = FastAdaptiveFMM(max_leaf_particles=30, base_depth=2, max_depth=6,
                          p=10)
    pot = eng.evaluate(pts, q, compute_forces=False)
    ref = exact_direct_nbody_2d(pts, q)
    assert _rel(pot, ref) < 5e-6


if __name__ == "__main__":
    test_m2l_matrix_matches_classical_operator()
    print("M2L matrix parity: PASS")
    test_potentials_and_forces_vs_direct()
    print("potentials & forces vs direct: PASS")
    test_agreement_with_classical_engine()
    print("agreement with classical engine: PASS")
    test_two_to_one_balance_invariant()
    print("2:1 balance invariant: PASS")
    test_no_duplicate_cells_in_hash_index()
    print("hash index uniqueness: PASS")
    test_single_cell_and_tiny_inputs()
    print("tiny inputs: PASS")
    test_uniform_distribution()
    print("uniform distribution: PASS")
    print("all adaptive_fmm_fast tests PASS")
