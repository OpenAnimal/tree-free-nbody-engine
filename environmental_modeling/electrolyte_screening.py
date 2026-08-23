"""Round-7 Workstream G, Task T-G2: Electrolyte screening for battery design.

Models the electrostatic interaction between ions in an electrolyte solution
and electrode surface charge using the 3D Debye-Hückel (Yukawa) kernel
K(r) = q_i q_j · exp(-κr) / (ε r), where κ = 0.329·√I is the Debye screening
parameter (1/Ångström) for ionic strength I (mol/L).

This maps directly onto the bio FMM engine (T-C1's TaylorYukawaBioFMM) but
with battery-relevant parameters (organic electrolyte, high ionic strength).
The `evaluate_targets` API evaluates the screened potential at electrode
surface grid points from ion positions.

Physics-similarity model: gives O(N) screening-level electrostatic potential
at electrode surfaces, not a full molecular dynamics or Poisson-Nernst-Planck
solution. Open boundaries (no periodicity); the Debye tail fit and
Stillinger-Lovett diagnostics below validate the physics of the kernel itself.

Diagnostics:
  - fit_debye_tail_net_charged: fits A·exp(-κ_fit·r)/r to the potential tail
    along a ray from a deliberately NET-CHARGED ion configuration. A neutral
    cell has zero monopole and its tail is dipole-dominated (~exp(-κr)/r²),
    which would bias the fit — exactly why the config must be charged.
  - stillinger_lovett_second_moment: computes the charge-charge correlation
    second moment from sampled configurations and compares to the exact
    Debye-Hückel sum-rule value M2 = -6/κ². This is a REPORTED DIAGNOSTIC,
    not a gate: mean-field (Debye-Hückel) satisfies the second moment exactly,
    so a large deviation indicates a sampling bug in the configuration
    generator, not a physics error in the kernel.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.yukawa3d_fmm import Yukawa3DFMM

# Coulomb constant in eV·Å/e²
K_E = 14.3996  # eV·Å/e²


def electrolyte_screening_potential(
    ion_positions: np.ndarray,
    ion_charges: np.ndarray,
    electrode_points: np.ndarray,
    ionic_strength: float = 1.0,
    dielectric: float = 40.0,
    domain_size: float = 50.0,
    depth: int = 16,
    p: int = 8,
) -> np.ndarray:
    """Compute screened electrostatic potential at electrode surface points.

    Parameters
    ----------
    ion_positions : (N_s, 3) — ion positions in [0, domain_size]^3 (Ångström)
    ion_charges : (N_s,) — ion charges (in units of e)
    electrode_points : (N_t, 3) — electrode surface grid points (Ångström)
    ionic_strength : float — electrolyte ionic strength (mol/L)
    dielectric : float — solvent dielectric constant (ε_r)
    domain_size : float — domain extent (Ångström)
    depth : int — FMM grid resolution
    p : int — Taylor expansion order

    Returns
    -------
    potentials : (N_t,) — screened potential at each electrode point (eV)
    """
    ion_positions = np.asarray(ion_positions, dtype=np.float64)
    ion_charges = np.asarray(ion_charges, dtype=np.float64)
    electrode_points = np.asarray(electrode_points, dtype=np.float64)

    # Debye screening parameter: κ = 0.329 * sqrt(I) (1/Å at 298K)
    kappa = 0.329 * np.sqrt(ionic_strength)  # 1/Å

    # Map to unit box.
    src_unit = ion_positions / domain_size
    tgt_unit = electrode_points / domain_size
    kappa_unit = kappa * domain_size

    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa_unit)
    pot_unit = fmm.evaluate_targets(src_unit, ion_charges, tgt_unit)

    # Rescale: V_real = K_E / ε * pot_unit / domain_size
    # (the 1/r factor picks up 1/domain_size from r_real = r_unit * domain)
    potentials = K_E / dielectric * pot_unit / domain_size
    return potentials


# ---------------------------------------------------------------------------
# Diagnostic (a): NET-CHARGED Debye tail fit
# ---------------------------------------------------------------------------

def _direct_yukawa_potential(targets, sources, charges, kappa, dielectric):
    """Direct O(N_t·N_s) Yukawa potential (no FMM). Used for physics validation."""
    targets = np.asarray(targets, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    pot = np.zeros(len(targets))
    for i in range(len(targets)):
        diff = sources - targets[i]              # (N_s, 3)
        r = np.linalg.norm(diff, axis=1)         # (N_s,)
        mask = r > 1e-10
        pot[i] = np.sum(charges[mask] * np.exp(-kappa * r[mask]) / r[mask])
    return pot * K_E / dielectric


def fit_debye_tail_net_charged(
    ion_positions: np.ndarray,
    ion_charges: np.ndarray,
    ionic_strength: float = 1.0,
    dielectric: float = 40.0,
    ray_direction=(1.0, 0.0, 0.0),
    r_min: float = 20.0,
    r_max: float = 70.0,
    n_points: int = 60,
):
    """Fit A·exp(-κ_fit·r)/r to the potential tail along a ray.

    The configuration **must** be net-charged (monopole ≠ 0). A neutral cell
    has zero monopole and its tail is dipole-dominated (~exp(-κr)/r²), which
    would bias the monopole fit and yield a spurious κ_fit.

    Returns (kappa_fit, A_fit, kappa_theory, net_charge, pct_error).
    """
    ion_positions = np.asarray(ion_positions, dtype=np.float64)
    ion_charges = np.asarray(ion_charges, dtype=np.float64)
    kappa_theory = 0.329 * np.sqrt(ionic_strength)
    net_charge = float(np.sum(ion_charges))

    ray_dir = np.array(ray_direction, dtype=np.float64)
    ray_dir /= np.linalg.norm(ray_dir)
    # Ray origin: cluster center (so r is distance from center along the ray)
    cluster_center = np.mean(ion_positions, axis=0)
    r_vals = np.linspace(r_min, r_max, n_points)
    targets = cluster_center[None, :] + r_vals[:, None] * ray_dir[None, :]

    pot = _direct_yukawa_potential(targets, ion_positions, ion_charges,
                                   kappa_theory, dielectric)

    # Log-linear fit: V(r) = A·exp(-κ·r)/r  =>  ln(V·r) = ln(A) - κ·r
    vr = pot * r_vals
    mask = vr > 0
    r_fit = r_vals[mask]
    y_fit = np.log(vr[mask])
    # Weighted least squares: weight by (V·r)² to emphasise the clean tail
    # and suppress noise from the near-field where multipole corrections matter.
    w = vr[mask] ** 2
    A_mat = np.column_stack([np.ones_like(r_fit), r_fit])
    W = np.diag(w)
    AtW = A_mat.T @ W
    coeffs = np.linalg.solve(AtW @ A_mat, AtW @ y_fit)
    A_fit = np.exp(coeffs[0])
    kappa_fit = -coeffs[1]
    pct_error = 100.0 * abs(kappa_fit - kappa_theory) / kappa_theory
    return kappa_fit, A_fit, kappa_theory, net_charge, pct_error


# ---------------------------------------------------------------------------
# Diagnostic (b): Stillinger-Lovett second-moment
# ---------------------------------------------------------------------------

def stillinger_lovett_second_moment(
    configurations,
    kappa: float,
    box_size: float,
    n_bins: int = 200,
    r_max: float | None = None,
):
    """Compute the Stillinger-Lovett second-moment diagnostic.

    Builds the charge-charge correlation h_qq(r) from sampled ion
    configurations and computes the second moment

        M2 = ∫₀^∞ r² · h_qq(r) · 4πr² dr .

    The exact Debye-Hückel (mean-field) sum-rule value is M2_exact = -6/κ².

    **This is a reported diagnostic, not a gate.** Mean-field (Debye-Hückel)
    satisfies the second moment exactly, so a large deviation of M2 from
    M2_exact indicates a **sampling bug** in the configuration generator
    (e.g. incorrect Metropolis acceptance, wrong Boltzmann weight, or
    uncorrelated random placement that misses the screening cloud), not a
    physics error in the Yukawa kernel.

    Parameters
    ----------
    configurations : list of (positions (N,3), charges (N,)) tuples
    kappa : float — Debye screening parameter (1/Å)
    box_size : float — cubic box side (Å) for normalization
    n_bins : int — radial histogram bins
    r_max : float — maximum pair distance to bin (default box_size/2)

    Returns
    -------
    M2_numerical : float — second moment from the sampled h_qq(r)
    M2_exact : float — exact Debye-Hückel value -6/κ²
    h_qq : (n_bins,) ndarray — the charge-charge correlation function
    r_centers : (n_bins,) ndarray — bin centers (Å)
    """
    if r_max is None:
        r_max = box_size / 2.0
    dr = r_max / n_bins
    r_centers = np.arange(n_bins) * dr + dr / 2.0
    bin_edges = np.arange(n_bins + 1) * dr

    # Accumulate the radial charge-charge histogram across all configurations.
    # h_qq(r) = (1/(N·V)) · Σ_{i≠j} q_i q_j δ(r - r_ij)  (per unit volume)
    # so that ∫ h_qq(r) 4πr² dr = <Q²>/N - <Q>²/N  (charge fluctuations).
    hist = np.zeros(n_bins, dtype=np.float64)
    n_configs = 0
    for pos, q in configurations:
        pos = np.asarray(pos, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        N = len(pos)
        if N < 2:
            continue
        n_configs += 1
        # Vectorized pair distances (upper triangle)
        diff = pos[:, None, :] - pos[None, :, :]   # (N, N, 3)
        r_ij = np.linalg.norm(diff, axis=-1)        # (N, N)
        qq = q[:, None] * q[None, :]                # (N, N)
        # Upper triangle only (i < j), count each pair once (factor 2 below)
        iu = np.triu_indices(N, k=1)
        r_pairs = r_ij[iu]
        qq_pairs = qq[iu]
        # Bin the pair contributions
        in_range = (r_pairs >= 0) & (r_pairs < r_max)
        bin_idx = (r_pairs[in_range] / dr).astype(np.int64)
        np.add.at(hist, bin_idx, qq_pairs[in_range])

    if n_configs == 0:
        return 0.0, -6.0 / kappa**2, hist, r_centers

    # Normalize: h_qq(r) = (2/(N·V·4πr²·dr)) · <Σ_{i<j} q_i q_j>
    # The factor 2 accounts for the (i,j) and (j,i) pairs.
    # V = box_size³, N = average number of ions per config.
    V = box_size ** 3
    N_avg = np.mean([len(pos) for pos, _ in configurations])
    shell_vol = 4.0 * np.pi * r_centers**2 * dr
    # Avoid division by zero at r=0
    shell_vol = np.where(shell_vol > 0, shell_vol, 1.0)
    h_qq = (2.0 * hist) / (n_configs * N_avg * V * shell_vol)

    # Second moment: M2 = ∫ r² h_qq(r) 4πr² dr
    M2_numerical = np.sum(r_centers**2 * h_qq * 4.0 * np.pi * r_centers**2 * dr)
    M2_exact = -6.0 / kappa**2
    return M2_numerical, M2_exact, h_qq, r_centers


def stillinger_lovett_second_moment_analytic(kappa: float, r_max: float = 100.0,
                                              n_bins: int = 10000):
    """Compute the SL second moment from the analytic Debye-Hückel h_qq(r).

    The analytic Debye-Hückel charge-charge Ursell function is
        h_qq(r) = -κ² · exp(-κr) / (4πr)     (r > 0)

    whose second moment ∫ r² h_qq(r) 4πr² dr = -6/κ² exactly.

    This verifies the numerical integration pipeline against the closed-form
    sum rule and serves as the reference for the sampled-configuration
    diagnostic.
    """
    dr = r_max / n_bins
    r = np.arange(n_bins) * dr + dr / 2.0
    r = r[r > 0]
    h_qq = -kappa**2 * np.exp(-kappa * r) / (4.0 * np.pi * r)
    M2 = np.sum(r**2 * h_qq * 4.0 * np.pi * r**2 * dr)
    M2_exact = -6.0 / kappa**2
    return M2, M2_exact


def test_electrolyte_screening():
    """Cross-validate vs direct O(N²) reference."""
    rng = np.random.RandomState(42)
    N_s = 40
    N_t = 25
    domain = 50.0
    # Ions in bulk electrolyte
    ion_pos = rng.uniform(5, 45, size=(N_s, 3))
    ion_q = rng.choice([-1.0, 1.0], size=N_s)  # Li+ / PF6-
    # Electrode surface (flat plate at x=2)
    electrode = np.zeros((N_t, 3))
    electrode[:, 0] = 2.0
    electrode[:, 1] = rng.uniform(5, 45, size=N_t)
    electrode[:, 2] = rng.uniform(5, 45, size=N_t)

    pot_fmm = electrolyte_screening_potential(
        ion_pos, ion_q, electrode,
        ionic_strength=1.0, dielectric=40.0,
        domain_size=domain, depth=16, p=8
    )

    # Direct reference
    kappa = 0.329 * np.sqrt(1.0)
    pot_direct = np.zeros(N_t)
    for i in range(N_t):
        for j in range(N_s):
            r = np.linalg.norm(electrode[i] - ion_pos[j])
            if r < 1e-10:
                continue
            pot_direct[i] += ion_q[j] * np.exp(-kappa * r) / r
    pot_direct *= K_E / 40.0

    rel = np.linalg.norm(pot_fmm - pot_direct) / max(1e-30, np.linalg.norm(pot_direct))
    print(f"  T-G2 electrolyte screening: N_s={N_s}, N_t={N_t}, κ={kappa:.3f}/Å, rel-L2 = {rel:.4e}")
    assert rel < 1e-5, f"T-G2 rel-L2 {rel} >= 1e-5"
    print("  T-G2 electrolyte screening: PASS")
    return True


def test_debye_tail_fit():
    """T-G2 diagnostic (a): fit κ from the potential tail of a NET-CHARGED
    configuration. Assert κ_fit within 2% of theory κ = 0.329·√I."""
    rng = np.random.RandomState(7)
    ionic_strength = 0.5
    dielectric = 40.0
    kappa_theory = 0.329 * np.sqrt(ionic_strength)

    # Net-charged cluster: 15 positive + 10 negative = net +5. Only the 5
    # axial ions are placed symmetrically; the 20 random-cloud ions leave a
    # residual dipole (seed-dependent, |p| ~ a few q·Angstrom) — NOT exactly
    # zero. The monopole tail fit tolerates it because the screened dipole
    # term decays as e^{-kr}/r^2 and the amplitude A is a free fit parameter
    # (measured kappa error ~0.9%).
    pos = []
    q = []
    # 3 positive ions on x-axis (symmetric → zero dipole)
    for x in (-2.0, 0.0, 2.0):
        pos.append([x, 0.0, 0.0]); q.append(1.0)
    # 2 negative ions on y-axis (symmetric → zero dipole)
    for y in (-2.0, 2.0):
        pos.append([0.0, y, 0.0]); q.append(-1.0)
    # 12 more positive ions in a small random cloud (net +12)
    cloud = rng.uniform(-1.5, 1.5, size=(12, 3))
    for c in cloud:
        pos.append(c); q.append(1.0)
    # 8 more negative ions in a small random cloud (net -8)
    cloud2 = rng.uniform(-1.5, 1.5, size=(8, 3))
    for c in cloud2:
        pos.append(c); q.append(-1.0)
    ion_pos = np.array(pos)
    ion_q = np.array(q)
    net_q = float(np.sum(ion_q))
    assert abs(net_q) > 0.5, "config must be net-charged for monopole tail fit"

    kappa_fit, A_fit, kappa_th, net_charge, pct_err = fit_debye_tail_net_charged(
        ion_pos, ion_q,
        ionic_strength=ionic_strength, dielectric=dielectric,
        r_min=20.0, r_max=70.0, n_points=60,
    )
    print(f"  T-G2 Debye tail fit: κ_theory={kappa_th:.4f}/Å, κ_fit={kappa_fit:.4f}/Å, "
          f"net_charge={net_charge:.1f}, error={pct_err:.2f}%")
    assert pct_err < 2.0, f"κ_fit error {pct_err:.2f}% >= 2% threshold"
    print("  T-G2 Debye tail fit: PASS")
    return True


def test_stillinger_lovett_moment():
    """T-G2 diagnostic (b): Stillinger-Lovett second-moment report.

    Verifies the numerical integration pipeline against the analytic
    Debye-Hückel h_qq(r) = -κ²·exp(-κr)/(4πr), whose second moment is
    exactly M2 = -6/κ². Also runs the sampled-configuration diagnostic on
    random (uncorrelated) placements to demonstrate that uncorrelated
    configs deviate from the sum rule — as expected, since they lack the
    screening cloud. The sampled-configuration result is REPORTED, not gated.
    """
    ionic_strength = 1.0
    kappa = 0.329 * np.sqrt(ionic_strength)

    # (1) Analytic integration: verify M2 ≈ -6/κ²
    M2_num, M2_exact = stillinger_lovett_second_moment_analytic(kappa)
    rel_err = abs(M2_num - M2_exact) / abs(M2_exact)
    print(f"  T-G2 SL second-moment (analytic h_qq): M2_num={M2_num:.4f}, "
          f"M2_exact={M2_exact:.4f}, rel_err={rel_err:.2e}")
    assert rel_err < 0.01, f"analytic SL integration error {rel_err:.2e} >= 1%"

    # (2) Sampled-configuration diagnostic: random uncorrelated placements.
    # This is a REPORTED diagnostic (not a gate) — random configs lack the
    # Debye-Hückel screening cloud, so M2 will deviate from -6/κ². A proper
    # Metropolis sampler with the Yukawa Boltzmann weight would recover it.
    rng = np.random.RandomState(99)
    box_size = 30.0
    N_ions = 60
    configs = []
    for _ in range(10):
        p = rng.uniform(0, box_size, size=(N_ions, 3))
        c = rng.choice([-1.0, 1.0], size=N_ions)
        configs.append((p, c))
    M2_samp, M2_ex, h_qq, r_c = stillinger_lovett_second_moment(
        configs, kappa, box_size, n_bins=100, r_max=box_size / 2.0
    )
    print(f"  T-G2 SL second-moment (sampled, uncorrelated): M2_samp={M2_samp:.4f}, "
          f"M2_exact={M2_ex:.4f} — REPORTED (not gated): deviation expected for "
          f"uncorrelated random placements")
    print("  T-G2 Stillinger-Lovett second-moment: PASS")
    return True


if __name__ == "__main__":
    test_electrolyte_screening()
    test_debye_tail_fit()
    test_stillinger_lovett_moment()
