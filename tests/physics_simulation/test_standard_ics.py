"""Standard self-gravity scenario math for the WebGPU demo (round 19).

The demo's particle-particle gravity is 2D logarithmic and ATTRACTIVE, from
the confining softened pair potential

    U(r) = +(Gp/2) * ln(r^2 + eps^2),      F_ij = -Gp (r_i - r_j)/(r^2 + eps^2)

(the exact kernel the WGSL P2P / direct / FMM multipoles implement; eps^2 =
4e-5). Log gravity in 2D obeys a Gauss law, so any axisymmetric disk has
F(R) = G*M_enc(R)/R. The two standard scenarios (GPU_NOTES §15) are defined
by closed forms of that law:

  virialized log-disk:  Sigma(R) = M/(pi a^2) (1+R^2/a^2)^-2
                          M_enc(R) = M R^2/(R^2 + a^2)
                          v_c(R)   = sqrt(G M) R / sqrt(R^2 + a^2)
                        supported by rotation + dispersion (v_phi =
                        v_c sqrt(1-s^2), sigma = s v_c, s = 0.3) — virial by
                        construction (2K = sum R*F), NOT an exact DF solution
                        (the exact isotropic Jeans dispersion of this profile
                        diverges at the center; see GPU_NOTES §15);
  cold collapse:        uniform disk R0, v = 0 (Q = 0); the interior force is
                        exactly harmonic, F = G*pi*Sigma0*R, so the unsoftened
                        free-fall time is t_ff = pi/(2*sqrt(G M)/R0).

These tests mirror the JavaScript generators in index.html (generateParticles,
standard-IC branch) with the same formulas and validate the physics claims
independently in numpy:

  - the inverse-CDF radial sampler reproduces the analytic M_enc(R);
  - the IC construction satisfies the virial identity 2K = sum R*F within
    sampling noise (direct O(n^2) softened forces, no FMM involved);
  - the Hamiltonian convention (U = +(Gp/2) ln(r^2+eps^2)) is CONSISTENT
    with the mirrored force law: a leapfrog integrator with exactly those
    forces conserves exactly that E to floating-point roundoff — this is the
    same convention the demo's energy_phi WGSL kernel and the validate.html
    rig use for dE/E;
  - a cold uniform disk collapsed under the softened law reaches minimum
    half-mass radius near the analytic unsoftened t_ff (softening delays the
    collapse slightly).
"""
import numpy as np
import pytest

# --- constants mirrored from index.html (standard-IC block) ---------------
MU = 0.002        # total G*M (G = 1), N-independent
A = 0.1           # log-disk scale radius
R0 = 0.3          # cold-collapse disk radius
EPS2 = 4.0e-5     # shared softening (P2P_EPS2)
S_FRAC = 0.3      # dispersion fraction of the warm-disk construction


def sample_log_disk(n, rng):
    """Mirror of the demo's virIALIZED log-disk generator (plummer IC)."""
    u = np.maximum(1e-9, rng.random(n))
    r = A * np.sqrt(u / (1.0 - u))
    th = rng.random(n) * 2.0 * np.pi
    ct, st = np.cos(th), np.sin(th)
    v_c = np.sqrt(MU) * r / np.sqrt(r * r + A * A)
    sigma = S_FRAC * v_c
    gm = np.sqrt(-2.0 * np.log(np.maximum(1e-12, rng.random(n))))
    ang = rng.random(n) * 2.0 * np.pi
    v_r = sigma * gm * np.cos(ang)
    v_t = v_c * np.sqrt(1.0 - S_FRAC * S_FRAC) + sigma * gm * np.sin(ang)
    pos = np.stack([0.5 + r * ct, 0.5 + r * st], axis=1)
    vel = np.stack([v_r * ct - v_t * st, v_r * st + v_t * ct], axis=1)
    # exact COM / net-momentum framing (mirror of generateParticles' pass)
    pos -= pos.mean(axis=0) - 0.5
    vel -= vel.mean(axis=0)
    return pos, vel


def sample_cold_disk(n, rng):
    """Mirror of the demo's cold-collapse generator."""
    r = R0 * np.sqrt(rng.random(n))
    th = rng.random(n) * 2.0 * np.pi
    pos = np.stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)], axis=1)
    pos -= pos.mean(axis=0) - 0.5
    return pos, np.zeros((n, 2))


def softened_log_forces(pos, gp):
    """Direct O(n^2) forces of the demo's kernel: F_i = -gp sum_j (d)/(d^2+eps^2).

    Returns (forces, pair_potential_per_particle) where
    phi_i = sum_{j != i} ln(r_ij^2 + eps^2)  (the energy_phi kernel's .x).
    """
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d) + EPS2
    np.fill_diagonal(r2, 1.0)  # self-term excluded below via mask
    inv = 1.0 / r2
    np.fill_diagonal(inv, 0.0)
    f = -gp * np.einsum('ijk,ij->ik', d, inv)
    phi = np.log(r2).sum(axis=1)  # diagonal was overwritten -> no self-term
    return f, phi


def total_energy(pos, vel, gp):
    """E = sum |v|^2/2 + (gp/4) sum_i sum_{j!=i} ln(r_ij^2+eps^2) (GPU_NOTES §15)."""
    _, phi = softened_log_forces(pos, gp)
    return 0.5 * np.einsum('ij,ij->', vel, vel) + 0.25 * gp * phi.sum()


def test_log_disk_radial_sampling_matches_enclosed_mass():
    """Inverse-CDF sample of Sigma(R) ~ (1+R^2/a^2)^-2 must reproduce
    M_enc(R)/M = R^2/(R^2+a^2) (the 2D Gauss law of log gravity)."""
    rng = np.random.default_rng(0x600D5EED)
    n = 60000
    pos, _ = sample_log_disk(n, rng)
    r = np.hypot(pos[:, 0] - 0.5, pos[:, 1] - 0.5)
    for frac, tol in [(0.25, 0.02), (0.5, 0.02), (0.75, 0.02)]:
        # radius containing `frac` of the mass, sample vs analytic
        r_emp = np.quantile(r, frac)
        # analytic: M_enc/M = u -> R = a sqrt(u/(1-u))
        r_ana = A * np.sqrt(frac / (1.0 - frac))
        assert abs(r_emp / r_ana - 1.0) < tol, (frac, r_emp, r_ana)


def test_log_disk_virial_identity():
    """2K = sum_i R_i |F_i| within sampling noise for the warm rotating disk
    (centrifugal + dispersion support; the exact statement behind 'virialized
    by construction' in GPU_NOTES §15)."""
    rng = np.random.default_rng(0x600D5EED)
    n = 1500
    gp = MU / n
    pos, vel = sample_log_disk(n, rng)
    f, _ = softened_log_forces(pos, gp)
    r_vec = pos - 0.5
    # inward radial force magnitude vs outward centrifugal+pressure demand
    two_k = np.einsum('ij,ij->', vel, vel)          # 2K = sum |v|^2
    sum_r_f = -np.einsum('ij,ij->', r_vec, f)        # sum R * F_inward > 0
    # The virial identity for the exact DF would be 2K = sum R F; the warm
    # construction satisfies it to the dispersion-split approximation.
    assert abs(two_k / sum_r_f - 1.0) < 0.15, (two_k, sum_r_f)


def test_energy_convention_consistent_with_force_law():
    """Leapfrog under exactly the mirrored softened-log forces conserves
    exactly the demo's Hamiltonian convention (U = +(gp/2) ln(r^2+eps^2)) —
    the same convention energy_phi and the validate.html rig use. A sign
    error in either E or F shows up as O(1) drift, not roundoff."""
    rng = np.random.default_rng(7)
    n = 200
    gp = MU / n
    pos, vel = sample_log_disk(n, rng)
    dt = 0.01
    e0 = total_energy(pos, vel, gp)
    for _ in range(200):
        f, _ = softened_log_forces(pos, gp)
        vel = vel + 0.5 * dt * f
        pos = pos + dt * vel
        f, _ = softened_log_forces(pos, gp)
        vel = vel + 0.5 * dt * f
    e1 = total_energy(pos, vel, gp)
    assert abs((e1 - e0) / abs(e0)) < 1e-6, (e0, e1)


def test_cold_collapse_time_near_analytic_free_fall():
    """A cold uniform disk collapsed under the softened law reaches minimum
    half-mass radius near the analytic unsoftened t_ff = pi*R0/(2 sqrt(G M))
    (softening + particle noise delay it slightly)."""
    rng = np.random.default_rng(0xC011)
    n = 600
    gp = MU / n
    pos, vel = sample_cold_disk(n, rng)
    t_ff = np.pi * R0 / (2.0 * np.sqrt(MU))
    dt = t_ff / 400
    r_prev = np.hypot(pos[:, 0] - 0.5, pos[:, 1] - 0.5)
    r50_prev = np.quantile(r_prev, 0.5)
    best_t, min_r50 = 0.0, np.inf
    t = 0.0
    for step in range(800):
        f, _ = softened_log_forces(pos, gp)
        vel = vel + 0.5 * dt * f
        pos = pos + dt * vel
        f, _ = softened_log_forces(pos, gp)
        vel = vel + 0.5 * dt * f
        t += dt
        r = np.hypot(pos[:, 0] - 0.5, pos[:, 1] - 0.5)
        r50 = np.quantile(r, 0.5)
        if r50 < min_r50:
            min_r50, best_t = r50, t
        if r50 > 1.5 * r50_prev and step > 20:
            break  # bounced: past minimum
        r50_prev = r50
    assert min_r50 < 0.5 * R0, "no collapse happened"
    assert 0.7 * t_ff <= best_t <= 1.5 * t_ff, (best_t, t_ff)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
