"""Round-7 Workstream G, Task T-G3: Groundwater contaminant plume.

Models a 3D contaminant plume from point sources (release points / leaking
tanks) to target receptors (monitoring wells) by solving the steady-state
advection-dispersion-decay equation

    (-D * grad^2 + v . grad + lam) c(x) = sum_j q_j delta(x - x_j)

exactly (point sources in an infinite domain) via a screened-Helmholtz
factorization, evaluated with the 3D Yukawa Taylor FMM.

FACTORISATION (re-derived and verified; the previous implementation omitted
the entire advection term -- see the analytic anchor tests below).  Let

    c(x) = exp((v . x) / (2D)) * u(x).

Substituting collapses the advection term and yields a pure screened
Helmholtz equation for u:

    (-D * grad^2 + (lam + |v|^2 / (4D))) u = sum_j q~_j delta(x - x_j),

with the rescaled sources

    q~_j = q_j * exp(-(v . x_j) / (2D))

and the effective screening parameter

    kappa^2 = lam / D + |v|^2 / (4 D^2).

The Green's function of the screened Helmholtz operator is the Yukawa kernel
exp(-kappa r) / (4 pi D r), so

    u(x_i) = (1 / (4 pi D)) * sum_j q~_j * exp(-kappa r_ij) / r_ij,
    c(x_i) = exp((v . x_i) / (2D)) * u(x_i).

For v = 0 this reduces to the pure-diffusion/decay form
c = q exp(-kappa r) / (4 pi D r) with kappa^2 = lam / D.

This is a physics-similarity model, not a full MODFLOW/MT3D simulation.
It gives O(N) screening-level concentration estimates at monitoring wells.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.yukawa3d_fmm import Yukawa3DFMM


def groundwater_plume_concentration(
    sources: np.ndarray,
    release_rates: np.ndarray,
    targets: np.ndarray,
    flow_velocity: float = 0.5,
    longitudinal_dispersivity: float = 10.0,
    decay_rate: float = 0.001,
    domain_size: float = 100.0,
    depth: int = 16,
    p: int = 8,
    flow_direction: np.ndarray = (1.0, 0.0, 0.0),
    molecular_diffusion: float = 1e-6,
) -> np.ndarray:
    """Compute steady-state contaminant concentration at monitoring wells.

    Solves the advection-dispersion-decay equation via the screened-Helmholtz
    factorization (see module docstring): the advection term is absorbed into
    a per-point exponential factor and a |v|^2/(4D^2) contribution to kappa,
    with rescaled sources q~_j = q_j exp(-(v.x_j)/(2D)).

    Parameters
    ----------
    sources : (N_s, 3) — contaminant release points in [0, domain_size]^3 (meters)
    release_rates : (N_s,) — mass release rates (kg/s)
    targets : (N_t, 3) — monitoring well locations (meters)
    flow_velocity : float — groundwater flow speed (m/day); magnitude of v
    longitudinal_dispersivity : float — alpha_L (meters)
    decay_rate : float — first-order decay constant lam (1/day)
    domain_size : float — domain extent (meters)
    depth : int — FMM grid resolution (linear cells per side)
    p : int — Taylor expansion order
    flow_direction : (3,) array-like — unit direction of the flow vector v.
        Default +x. Need not be normalized; it is normalized internally.
    molecular_diffusion : float — molecular diffusion contribution to D (m^2/day).

    Returns
    -------
    concentrations : (N_t,) — steady-state concentration (mass/volume units,
        proportional to the release-rate units).
    """
    sources = np.asarray(sources, dtype=np.float64)
    release_rates = np.asarray(release_rates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Dispersion tensor: D = alpha_L * |v| + D_mol (isotropic screening model).
    D = longitudinal_dispersivity * float(flow_velocity) + float(molecular_diffusion)
    D = max(D, 1e-12)

    # Flow velocity vector v (m/day).
    dvec = np.asarray(flow_direction, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(dvec))
    if nrm < 1e-12:
        dvec = np.array([1.0, 0.0, 0.0])
    else:
        dvec = dvec / nrm
    v_vec = float(flow_velocity) * dvec
    v_sq = float(np.dot(v_vec, v_vec))

    # Effective screened-Helmholtz parameter:
    #   kappa^2 = lam / D + |v|^2 / (4 D^2)
    lam = max(float(decay_rate), 0.0)
    kappa_sq = lam / D + v_sq / (4.0 * D * D)
    kappa = float(np.sqrt(kappa_sq))  # 1/meters

    # Rescaled sources: q~_j = q_j * exp(-(v . x_j) / (2D))
    v_dot_src = sources @ v_vec
    q_tilde = release_rates * np.exp(-v_dot_src / (2.0 * D))

    # Map to unit box [0, 1)^3 for the FMM engine.
    src_unit = sources / domain_size
    tgt_unit = targets / domain_size

    # The FMM evaluates sum_j q~_j exp(-kappa_unit r_unit) / r_unit in unit-box
    # coordinates. r_real = r_unit * domain_size, so
    #   exp(-kappa r_real) = exp(-kappa_unit r_unit)  with kappa_unit = kappa * span,
    #   1 / r_real = (1 / domain_size) * 1 / r_unit.
    # Hence  sum_j q~_j exp(-kappa r_real)/r_real = pot_unit / domain_size.
    kappa_unit = kappa * domain_size

    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa_unit)
    pot_unit = fmm.evaluate_targets(src_unit, q_tilde, tgt_unit)

    # u(x_i) = (1 / (4 pi D)) * pot_unit / domain_size
    u = pot_unit / (4.0 * np.pi * D * domain_size)

    # c(x_i) = exp((v . x_i) / (2D)) * u(x_i)
    v_dot_tgt = targets @ v_vec
    concentrations = np.exp(v_dot_tgt / (2.0 * D)) * u
    return concentrations


def _direct_concentration(sources, release_rates, targets, D, v_vec, kappa, domain_size):
    """Direct O(N^2) reference implementing the full factorization."""
    sources = np.asarray(sources, dtype=np.float64)
    release_rates = np.asarray(release_rates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    v_vec = np.asarray(v_vec, dtype=np.float64)
    q_tilde = release_rates * np.exp(-(sources @ v_vec) / (2.0 * D))
    out = np.zeros(len(targets))
    for i in range(len(targets)):
        for j in range(len(sources)):
            r = np.linalg.norm(targets[i] - sources[j])
            if r < 1e-10:
                continue
            out[i] += q_tilde[j] * np.exp(-kappa * r) / r
    out /= (4.0 * np.pi * D)
    out *= np.exp((targets @ v_vec) / (2.0 * D))
    return out


def test_groundwater_plume():
    """Cross-validate the full factorization vs direct O(N^2) reference."""
    rng = np.random.RandomState(42)
    N_s = 30
    N_t = 20
    domain = 100.0
    sources = rng.uniform(10, 90, size=(N_s, 3))
    release_rates = rng.uniform(0.1, 1.0, size=N_s)
    targets = rng.uniform(10, 90, size=(N_t, 3))

    v_mag = 0.5
    alpha_L = 10.0
    lam = 0.001
    D = alpha_L * v_mag + 1e-6
    v_vec = np.array([v_mag, 0.0, 0.0])
    kappa = float(np.sqrt(lam / D + np.dot(v_vec, v_vec) / (4.0 * D * D)))

    conc_fmm = groundwater_plume_concentration(
        sources, release_rates, targets,
        flow_velocity=v_mag, longitudinal_dispersivity=alpha_L,
        decay_rate=lam, domain_size=domain, depth=16, p=8,
        flow_direction=(1.0, 0.0, 0.0),
    )

    conc_direct = _direct_concentration(
        sources, release_rates, targets, D, v_vec, kappa, domain)

    rel = np.linalg.norm(conc_fmm - conc_direct) / max(1e-30, np.linalg.norm(conc_direct))
    print(f"  T-G3 groundwater plume: N_s={N_s}, N_t={N_t}, kappa={kappa:.4f} 1/m, rel-L2 = {rel:.4e}")
    assert rel < 1e-5, f"T-G3 rel-L2 {rel} >= 1e-5"
    print("  T-G3 groundwater plume: PASS")
    return True


def test_groundwater_pure_diffusion_analytic():
    """Analytic anchor (1): v=0 single source -> c = q exp(-kappa r)/(4 pi D r)."""
    domain = 100.0
    D = 1.0           # via molecular_diffusion=1.0, v=0
    lam = 0.001
    kappa = float(np.sqrt(lam / D))
    q = 2.5
    src = np.array([[50.0, 50.0, 50.0]])
    # Probe points at several radii / directions.
    probes = np.array([
        [50.0 + 10.0, 50.0, 50.0],
        [50.0, 50.0 - 15.0, 50.0],
        [50.0 + 6.0, 50.0 + 8.0, 50.0],
        [50.0 - 12.0, 50.0 + 5.0, 50.0 - 7.0],
        [50.0 + 20.0, 50.0 + 20.0, 50.0 + 20.0],
    ])
    rates = np.array([q])

    conc = groundwater_plume_concentration(
        src, rates, probes,
        flow_velocity=0.0, longitudinal_dispersivity=10.0,
        decay_rate=lam, domain_size=domain, depth=16, p=10,
        molecular_diffusion=D,
    )
    exact = np.array([
        q * np.exp(-kappa * np.linalg.norm(p - src[0])) /
        (4.0 * np.pi * D * np.linalg.norm(p - src[0]))
        for p in probes
    ])
    rel = np.linalg.norm(conc - exact) / max(1e-30, np.linalg.norm(exact))
    print(f"  T-G3 pure-diffusion analytic: kappa={kappa:.4f} 1/m, rel-L2 = {rel:.4e}")
    assert rel < 1e-6, f"pure-diffusion analytic rel-L2 {rel} >= 1e-6"
    print("  T-G3 pure-diffusion analytic: PASS")
    return True


def test_groundwater_advected_analytic():
    """Analytic anchor (2): advected single source vs closed form
    c = (q/(4 pi D)) exp((v.(x-x_s))/(2D)) exp(-kappa r)/r on random directions."""
    domain = 100.0
    D = 2.0
    lam = 0.002
    v_vec = np.array([0.4, 0.3, 0.0])  # |v| = 0.5
    v_mag = float(np.linalg.norm(v_vec))
    kappa = float(np.sqrt(lam / D + v_mag * v_mag / (4.0 * D * D)))
    q = 1.7
    src = np.array([[40.0, 50.0, 50.0]])
    rng = np.random.RandomState(7)
    probes = src[0] + rng.uniform(-25.0, 25.0, size=(8, 3))
    # keep probes inside domain
    probes = np.clip(probes, 5.0, 95.0)
    rates = np.array([q])

    conc = groundwater_plume_concentration(
        src, rates, probes,
        flow_velocity=v_mag, longitudinal_dispersivity=0.0,
        decay_rate=lam, domain_size=domain, depth=16, p=10,
        flow_direction=v_vec,
        molecular_diffusion=D,
    )
    exact = np.array([
        (q / (4.0 * np.pi * D)) *
        np.exp(np.dot(v_vec, (p - src[0])) / (2.0 * D)) *
        np.exp(-kappa * np.linalg.norm(p - src[0])) /
        np.linalg.norm(p - src[0])
        for p in probes
    ])
    rel = np.linalg.norm(conc - exact) / max(1e-30, np.linalg.norm(exact))
    print(f"  T-G3 advected analytic: |v|={v_mag:.3f} m/day, kappa={kappa:.4f} 1/m, rel-L2 = {rel:.4e}")
    assert rel < 1e-6, f"advected analytic rel-L2 {rel} >= 1e-6"
    print("  T-G3 advected analytic: PASS")
    return True


if __name__ == "__main__":
    test_groundwater_plume()
    test_groundwater_pure_diffusion_analytic()
    test_groundwater_advected_analytic()
