"""Round-10 Wave D regression tests: environmental_modeling edge cases and
independent-oracle checks not covered by the module-embedded anchors.
"""
import os
import sys

import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from environmental_modeling.groundwater_plume import groundwater_plume_concentration
from environmental_modeling.electrolyte_screening import (
    electrolyte_screening_potential, K_E)
from environmental_modeling.airborne_exposure import (
    airborne_exposure_room_eigen, airborne_exposure_room_images)
from environmental_modeling.radiotherapy_dose import (
    SuperpositionDoseEngine, ray_trace_lazy)


def test_groundwater_superposition_linearity_and_edges():
    """Independent superposition check: the concentration field of A+B sources
    evaluated in one call must equal the sum of the separate fields (the PDE
    is linear), and empty/zero inputs must return zeros."""
    rng = np.random.RandomState(11)
    kw = dict(flow_velocity=0.5, longitudinal_dispersivity=10.0,
              decay_rate=0.001, domain_size=100.0, depth=8, p=6,
              flow_direction=(1.0, 0.0, 0.0))
    sA = rng.uniform(20, 80, size=(6, 3)); qA = rng.uniform(0.1, 1.0, 6)
    sB = rng.uniform(20, 80, size=(5, 3)); qB = rng.uniform(0.1, 1.0, 5)
    tgt = rng.uniform(20, 80, size=(7, 3))
    cA = groundwater_plume_concentration(sA, qA, tgt, **kw)
    cB = groundwater_plume_concentration(sB, qB, tgt, **kw)
    cAB = groundwater_plume_concentration(
        np.vstack([sA, sB]), np.concatenate([qA, qB]), tgt, **kw)
    rel = np.linalg.norm(cAB - (cA + cB)) / np.linalg.norm(cA + cB)
    assert rel < 1e-6, f"superposition rel-L2 {rel:.2e} >= 1e-6"
    # non-normalized flow direction must be normalized internally
    kw2 = dict(kw); kw2["flow_direction"] = (2.0, 0.0, 0.0)
    cn = groundwater_plume_concentration(sA, qA, tgt, **kw2)
    assert np.allclose(cn, cA), "flow_direction not normalized internally"
    # empty sources -> zero concentrations
    c0 = groundwater_plume_concentration(np.empty((0, 3)), np.empty(0), tgt, **kw)
    assert c0.shape == (7,) and np.all(c0 == 0.0)


def test_electrolyte_constants_and_neutral_config():
    """Physical constants and FMM vs direct Yukawa on a net-neutral config."""
    assert abs(K_E - 14.3996) < 1e-4          # e^2/(4 pi eps0) in eV*A
    assert abs(0.329 - 1.0 / 3.04) < 0.005    # Debye length 3.04 A/sqrt(I)
    rng = np.random.RandomState(3)
    ions = rng.uniform(5, 45, size=(10, 3))
    qc = np.tile([1.0, -1.0], 5)
    els = np.zeros((4, 3)); els[:, 0] = 2.0
    els[:, 1:] = rng.uniform(5, 45, size=(4, 2))
    pot = electrolyte_screening_potential(ions, qc, els, ionic_strength=1.0,
                                          dielectric=40.0, domain_size=50.0,
                                          depth=10, p=6)
    kap = 0.329
    ref = np.zeros(4)
    for i in range(4):
        for j in range(10):
            r = np.linalg.norm(els[i] - ions[j])
            ref[i] += qc[j] * np.exp(-kap * r) / r
    ref *= K_E / 40.0
    rel = np.linalg.norm(pot - ref) / np.linalg.norm(ref)
    assert rel < 1e-5, f"electrolyte FMM vs direct rel-L2 {rel:.2e}"


def test_airborne_room_diagnostics_invariants():
    """Eigen expansion: linearity + full-problem mirror symmetry; images:
    reciprocity. These pin the Neumann-room diagnostics against sign or
    normalization regressions."""
    D_t, lam = 1.3, 0.7
    Lx, Ly, Lz = 4.0, 3.0, 2.0
    src = np.array([[1.1, 0.9, 0.7], [2.6, 2.1, 1.3]])
    Q = np.array([1.0, -0.6])
    tgt = np.array([[0.7, 1.5, 1.0], [2.2, 0.6, 1.6], [3.4, 2.4, 0.5]])
    both = airborne_exposure_room_eigen(src, Q, tgt, D_t, lam, (Lx, Ly, Lz), n_max=12)
    only1 = airborne_exposure_room_eigen(src[:1], Q[:1], tgt, D_t, lam, (Lx, Ly, Lz), n_max=12)
    only2 = airborne_exposure_room_eigen(src[1:], Q[1:], tgt, D_t, lam, (Lx, Ly, Lz), n_max=12)
    assert np.allclose(both, only1 + only2, atol=1e-14), "eigen expansion not linear"

    L = 3.0
    cs = np.array([[1.2, 1.3, 1.6]]); qs = np.array([1.0])
    tg = np.array([[1.4, 1.7, 1.3], [1.6, 1.3, 1.7], [1.5, 1.5, 1.9]])
    cs_m = cs.copy(); cs_m[:, 0] = L - cs[:, 0]
    tg_m = tg.copy(); tg_m[:, 0] = L - tg[:, 0]
    ca = airborne_exposure_room_eigen(cs, qs, tg, D_t, lam, (L, L, L), n_max=12)
    cb = airborne_exposure_room_eigen(cs_m, qs, tg_m, D_t, lam, (L, L, L), n_max=12)
    assert np.allclose(ca, cb, atol=1e-12), "eigen solution not mirror-symmetric"

    y1, y2 = np.array([[1.0, 1.5, 1.0]]), np.array([[3.0, 1.5, 1.0]])
    c12 = airborne_exposure_room_images(sources=y1, emission_rates=np.array([1.0]),
                                        targets=y2, D_t=D_t, removal_rate=lam,
                                        room_dims=(Lx, Ly, Lz))
    c21 = airborne_exposure_room_images(sources=y2, emission_rates=np.array([1.0]),
                                        targets=y1, D_t=D_t, removal_rate=lam,
                                        room_dims=(Lx, Ly, Lz))
    assert abs(c12[0] - c21[0]) < 1e-12 * max(1e-30, c12[0]), "images broke reciprocity"


def test_radiotherapy_engine_vs_direct_and_ray_trace():
    """SuperpositionDoseEngine vs direct double-Gaussian sum and ray_trace_lazy
    weight/geometry contract."""
    rng = np.random.RandomState(5)
    pts = rng.uniform(4, 26, size=(30, 3)); w = rng.uniform(0.1, 1.0, 30)
    tgt = rng.uniform(4, 26, size=(25, 3))
    eng = SuperpositionDoseEngine(s1=1.0, s2=2.0, a=0.6, b=0.4,
                                  domain_size=30.0, depth=8, p=8)
    dose = eng.evaluate(pts, w, tgt)
    ref = np.zeros(25)
    for i in range(25):
        for j in range(30):
            d2 = np.sum((tgt[i] - pts[j]) ** 2)
            ref[i] += w[j] * (0.6 * np.exp(-d2 / 2.0) + 0.4 * np.exp(-d2 / 8.0))
    rel = np.linalg.norm(dose - ref) / np.linalg.norm(ref)
    assert rel < 1e-5, f"double-Gaussian rel-L2 {rel:.2e} >= 1e-5"

    gen = ray_trace_lazy(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]),
                         length=5.0, num_points=11, total_weight=2.0, batch_size=4)
    tp, tw = [], []
    for p_, w_ in gen:
        tp.append(p_); tw.append(w_)
    tp = np.vstack(tp); tw = np.concatenate(tw)
    ts = np.linspace(0, 5, 11)
    assert len(tw) == 11
    assert np.allclose(tp[:, 2], 3.0 + ts) and np.allclose(tp[:, 0], 1.0)
    assert np.allclose(tw, 2.0 / 11 * np.exp(-0.01 * ts))
