"""Round-7 Workstream G, Task T-G1: Radiotherapy dose distribution.

Models the dose distribution from external-beam radiotherapy (photon/electron
pencil beams) using Gaussian kernels for lateral spread.

Two engines live here:

1. ``radiotherapy_dose_2d`` -- the 2D lateral-plane pencil-beam dose using the
   verified 2D Gaussian FGT.  The kernel is the standard Gaussian pencil-beam
   profile  K(r) = D0 * exp(-r^2 / (2 sigma^2));  the FGT evaluates
   exp(-|x-x_j|^2 / h^2), so the correct unit-box bandwidth is
   h = sigma * sqrt(2) / domain.  (The previous ``radiotherapy_dose_2d``
   shipped a knowingly-wrong h = sigma/domain that computed exp(-r^2/sigma^2)
   and was kept only as ``radiotherapy_dose_2d_correct``; the broken function
   has been deleted and the correct implementation promoted to the canonical
   name, with ``radiotherapy_dose_2d_correct`` retained as a back-compat
   alias.  The dead ``depth_attenuation`` parameter has been removed.)

2. ``SuperpositionDoseEngine`` -- the T-G1 3D superposition/convolution dose
   engine.  It uses an isotropic double-Gaussian kernel

       K(r) = a * exp(-r^2 / (2 s1^2)) + b * exp(-r^2 / (2 s2^2))

   computed as TWO ``Gaussian3DFGT`` evaluations (one per Gaussian width),
   with interaction points as sources and dose-grid points as targets via
   ``evaluate_targets``.  ``ray_trace_lazy`` synthesises interaction points
   along a pencil beam in a homogeneous-water phantom.

BANNER: Research prototype; homogeneous-water isotropic-kernel stand-in; NOT a
treatment-planning system; never for clinical use.

Physics-similarity model: gives O(N) screening-level dose estimates, not a
full Monte Carlo (EGSnrc/GEANT4) or collapsed-cone dose calculation.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.gaussian2d_fgt import Gaussian2DFGT
from core.gaussian2d_fgt import Gaussian3DFGT

_BANNER = ("Research prototype; homogeneous-water isotropic-kernel stand-in; "
           "NOT a treatment-planning system; never for clinical use.")


def radiotherapy_dose_2d(
    beam_entries: np.ndarray,
    beam_weights: np.ndarray,
    target_grid: np.ndarray,
    lateral_sigma: float = 0.5,
    domain_size: float = 30.0,
    depth_fgt: int = 10,
    p: int = 8,
) -> np.ndarray:
    """Compute 2D lateral dose distribution at target grid points.

    Kernel: K(r) = D0 * exp(-r^2 / (2 sigma^2)) -- the standard Gaussian
    pencil-beam profile.  The FGT evaluates exp(-|x-x_j|^2 / h^2), so the
    correct bandwidth is h = sigma * sqrt(2) (giving exp(-r^2/(2 sigma^2))).

    Parameters
    ----------
    beam_entries : (N_s, 2) -- beam entry points in the lateral plane (cm)
    beam_weights : (N_s,) -- beam weights (monitor units / dose weights)
    target_grid : (N_t, 2) -- target dose grid points in the lateral plane (cm)
    lateral_sigma : float -- pencil-beam lateral spread sigma (cm)
    domain_size : float -- domain extent (cm)
    depth_fgt : int -- FGT grid resolution
    p : int -- Hermite expansion order

    Returns
    -------
    doses : (N_t,) -- 2D lateral dose at each target point (relative units)
    """
    beam_entries = np.asarray(beam_entries, dtype=np.float64)
    beam_weights = np.asarray(beam_weights, dtype=np.float64)
    target_grid = np.asarray(target_grid, dtype=np.float64)

    src_unit = beam_entries / domain_size
    tgt_unit = target_grid / domain_size
    h = lateral_sigma * np.sqrt(2.0) / domain_size  # h = sigma*sqrt(2) in unit box

    fgt = Gaussian2DFGT(depth=depth_fgt, p=p, h=h)
    dose = fgt.evaluate_targets(src_unit, beam_weights, tgt_unit)
    return dose


# Back-compat alias for the previously-separated "correct" implementation.
radiotherapy_dose_2d_correct = radiotherapy_dose_2d


# =============================================================================
# SuperpositionDoseEngine -- T-G1 3D superposition/convolution dose engine.
# Isotropic double-Gaussian kernel via two Gaussian3DFGT evaluations.
# =============================================================================
class SuperpositionDoseEngine:
    """3D superposition dose engine with an isotropic double-Gaussian kernel.

    K(r) = a * exp(-r^2 / (2 s1^2)) + b * exp(-r^2 / (2 s2^2))

    evaluated as two ``Gaussian3DFGT`` transforms (one per width), interaction
    points as sources and dose-grid points as targets via ``evaluate_targets``.

    """ + _BANNER

    def __init__(
        self,
        s1: float = 1.0,
        s2: float = 2.0,
        a: float = 0.6,
        b: float = 0.4,
        domain_size: float = 30.0,
        depth: int = 10,
        p: int = 8,
    ):
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.a = float(a)
        self.b = float(b)
        self.domain_size = float(domain_size)
        self.depth = int(depth)
        self.p = int(p)

    def evaluate(self, interaction_points, weights, targets) -> np.ndarray:
        """Dose at ``targets`` from ``interaction_points`` with ``weights``.

        interaction_points : (N_s, 3) cm  (synthetic interaction sites)
        weights            : (N_s,)    relative dose weights
        targets            : (N_t, 3) cm  (dose grid points)
        returns            : (N_t,)    dose (relative units)
        """
        pts = np.asarray(interaction_points, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.float64)
        dom = self.domain_size
        src_unit = pts / dom
        tgt_unit = tgt / dom
        h1 = self.s1 * np.sqrt(2.0) / dom
        h2 = self.s2 * np.sqrt(2.0) / dom
        fgt1 = Gaussian3DFGT(depth=self.depth, p=self.p, h=h1)
        fgt2 = Gaussian3DFGT(depth=self.depth, p=self.p, h=h2)
        d1 = fgt1.evaluate_targets(src_unit, w, tgt_unit)
        d2 = fgt2.evaluate_targets(src_unit, w, tgt_unit)
        return self.a * d1 + self.b * d2


def ray_trace_lazy(
    origin,
    direction,
    length: float,
    num_points: int,
    total_weight: float,
    batch_size: int = 4096,
):
    """Synthesize interaction points along a pencil beam in a water phantom.

    Yields (points (M,3), weights (M,)) batches of ``num_points`` uniformly
    spaced interaction sites along the ray ``origin + t * direction``,
    ``t in [0, length]``.  Weights are uniform (``total_weight / num_points``);
    a mild exponential depth attenuation is applied so deeper sites deposit
    less (synthetic, homogeneous-water stand-in).  This is a synthetic
    interaction-point generator, not a Monte Carlo transport step.

    """ + _BANNER

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        d = np.array([0.0, 0.0, 1.0])
    else:
        d = d / nrm
    mu = 0.01  # 1/cm synthetic attenuation
    ts = np.linspace(0.0, length, num_points)
    w_each = total_weight / num_points
    start = 0
    while start < num_points:
        end = min(start + batch_size, num_points)
        tb = ts[start:end]
        pts = origin[None, :] + tb[:, None] * d[None, :]
        w = np.full(end - start, w_each, dtype=np.float64) * np.exp(-mu * tb)
        yield pts, w
        start = end


# =============================================================================
# Tests
# =============================================================================
def test_radiotherapy_dose():
    """Cross-validate the 2D lateral dose vs direct O(N^2) reference."""
    rng = np.random.RandomState(42)
    N_s = 20
    N_t = 30
    domain = 30.0
    sigma = 0.5  # cm lateral spread

    beams = rng.uniform(5, 25, size=(N_s, 2))
    weights = rng.uniform(0.5, 2.0, size=N_s)
    targets = rng.uniform(5, 25, size=(N_t, 2))

    dose_fmm = radiotherapy_dose_2d(
        beams, weights, targets,
        lateral_sigma=sigma, domain_size=domain, depth_fgt=10, p=8
    )

    dose_direct = np.zeros(N_t)
    for i in range(N_t):
        for j in range(N_s):
            r2 = np.sum((targets[i] - beams[j]) ** 2)
            dose_direct[i] += weights[j] * np.exp(-r2 / (2 * sigma ** 2))

    rel = np.linalg.norm(dose_fmm - dose_direct) / max(1e-30, np.linalg.norm(dose_direct))
    print(f"  T-G1 radiotherapy dose: N_s={N_s}, N_t={N_t}, sigma={sigma}cm, rel-L2 = {rel:.4e}")
    assert rel < 1e-5, f"T-G1 rel-L2 {rel} >= 1e-5"
    print("  T-G1 radiotherapy dose: PASS")
    return True


def _double_gaussian_1d_antideriv(x, a, b, s1, s2):
    """A(x) = integral_0^x [a exp(-u^2/(2 s1^2)) + b exp(-u^2/(2 s2^2))] du
    = a sqrt(pi/2) s1 erf(x/(s1 sqrt2)) + b sqrt(pi/2) s2 erf(x/(s2 sqrt2))."""
    from math import erf, sqrt, pi
    return (a * sqrt(pi / 2) * s1 * erf(x / (s1 * sqrt(2.0)))
            + b * sqrt(pi / 2) * s2 * erf(x / (s2 * sqrt(2.0))))


def test_superposition_dose_erf_anchor():
    """Analytic anchor (1): single pencil-beam line of interaction points,
    on-axis dose vs the closed-form 1D convolution integral of the
    double-Gaussian (erf terms).

    Beam interaction points uniformly spaced on z in [-L, L]; dose at on-axis
    target z is  D(z) = w0 * integral_{-L}^{L} K(|z - z'|) dz'
                = w0 * [A(z + L) + A(L - z)],
    where A(x) = integral_0^x K(u) du (erf terms).  K is smooth at u=0
    (Gaussian), so the trapezoidal rule over the uniform interaction grid is
    spectrally accurate and the FGT discrete sum matches the integral to
    <=1e-6.
    """
    a, b, s1, s2 = 0.6, 0.4, 1.0, 2.0
    domain = 30.0
    center = 15.0  # beam centered in the domain so unit-box coords stay in [0,1]
    L = 8.0
    N_pts = 4001
    # interaction points along z-axis through (center,center,center), on [c-L, c+L]
    zs = center + np.linspace(-L, L, N_pts)
    pts = np.full((N_pts, 3), center, dtype=np.float64)
    pts[:, 2] = zs
    # trapezoidal weight ~ 2L/(N-1) so the discrete sum approximates the integral
    dz = (2.0 * L) / (N_pts - 1)
    weights = np.full(N_pts, dz, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5  # composite trapezoid rule (endpoints halved)

    # on-axis targets, some beyond the beam ends so erf terms are exercised
    z_offsets = np.array([-10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 10.0, 12.0])
    targets = np.full((len(z_offsets), 3), center, dtype=np.float64)
    targets[:, 2] = center + z_offsets

    eng = SuperpositionDoseEngine(s1=s1, s2=s2, a=a, b=b,
                                  domain_size=domain, depth=12, p=10)
    dose = eng.evaluate(pts, weights, targets)

    # closed form: D(z) = integral_{-L}^{L} K(|z - z'|) dz'
    #               = A(z + L) + A(L - z)   (translation-invariant; uses offsets)
    exact = np.array([
        _double_gaussian_1d_antideriv(zo + L, a, b, s1, s2)
        + _double_gaussian_1d_antideriv(L - zo, a, b, s1, s2)
        for zo in z_offsets
    ])
    rel = np.linalg.norm(dose - exact) / max(1e-30, np.linalg.norm(exact))
    print(f"  T-G1 superposition erf anchor: N_pts={N_pts}, rel-L2 = {rel:.4e}")
    assert rel < 1e-6, f"superposition erf anchor rel-L2 {rel} >= 1e-6"
    print("  T-G1 superposition erf anchor: PASS")
    return True


def test_superposition_dose_linearity():
    """Analytic anchor (2): linearity D(A+B) = D(A) + D(B) to <=1e-12."""
    rng = np.random.RandomState(1)
    a, b, s1, s2 = 0.6, 0.4, 1.0, 2.0
    domain = 30.0
    eng = SuperpositionDoseEngine(s1=s1, s2=s2, a=a, b=b,
                                  domain_size=domain, depth=10, p=8)
    ptsA = rng.uniform(5, 25, size=(40, 3))
    wA = rng.uniform(0.1, 1.0, size=40)
    ptsB = rng.uniform(5, 25, size=(55, 3))
    wB = rng.uniform(0.1, 1.0, size=55)
    tgt = rng.uniform(5, 25, size=(30, 3))

    dA = eng.evaluate(ptsA, wA, tgt)
    dB = eng.evaluate(ptsB, wB, tgt)
    dAB = eng.evaluate(np.vstack([ptsA, ptsB]), np.concatenate([wA, wB]), tgt)
    rel = np.linalg.norm(dAB - (dA + dB)) / max(1e-30, np.linalg.norm(dA + dB))
    print(f"  T-G1 superposition linearity: rel-L2 = {rel:.4e}")
    assert rel < 1e-12, f"superposition linearity rel-L2 {rel} >= 1e-12"
    print("  T-G1 superposition linearity: PASS")
    return True


def test_superposition_dose_convergence():
    """Convergence (3): vs direct O(N^2) sum at N=5k, rel-L2 <= 1e-4."""
    rng = np.random.RandomState(3)
    a, b, s1, s2 = 0.6, 0.4, 1.0, 2.0
    domain = 30.0
    N_s = 5000
    N_t = 2000
    pts = rng.uniform(2, 28, size=(N_s, 3))
    w = rng.uniform(0.1, 1.0, size=N_s)
    tgt = rng.uniform(2, 28, size=(N_t, 3))

    eng = SuperpositionDoseEngine(s1=s1, s2=s2, a=a, b=b,
                                  domain_size=domain, depth=5, p=8)
    dose_fgt = eng.evaluate(pts, w, tgt)

    # direct O(N^2) reference (chunked over targets)
    dose_direct = np.zeros(N_t)
    inv2s1sq = 1.0 / (2.0 * s1 * s1)
    inv2s2sq = 1.0 / (2.0 * s2 * s2)
    chunk = 256
    for s in range(0, N_t, chunk):
        e = min(s + chunk, N_t)
        diff = tgt[s:e, None, :] - pts[None, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        dose_direct[s:e] = (a * np.exp(-r2 * inv2s1sq)
                            + b * np.exp(-r2 * inv2s2sq)) @ w

    rel = np.linalg.norm(dose_fgt - dose_direct) / max(1e-30, np.linalg.norm(dose_direct))
    print(f"  T-G1 superposition convergence: N_s={N_s}, N_t={N_t}, rel-L2 = {rel:.4e}")
    assert rel < 1e-4, f"superposition convergence rel-L2 {rel} >= 1e-4"
    print("  T-G1 superposition convergence: PASS")
    return True


if __name__ == "__main__":
    test_radiotherapy_dose()
    test_superposition_dose_erf_anchor()
    test_superposition_dose_linearity()
    test_superposition_dose_convergence()
