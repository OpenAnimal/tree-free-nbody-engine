"""Round-7 Workstream G, Task T-G4: Airborne pollutant exposure.

Screening prototype; point-source steady-state assumptions; not a certification
tool.

Models atmospheric dispersion of pollutants from emission sources (industrial
stacks, traffic, etc.) to population receptors using a 3D Yukawa kernel
K(r) = Q · exp(-r/λ) / r, where λ is the atmospheric mixing length.

The screening length λ captures:
- Turbulent diffusion (σ_z atmospheric stability class)
- Wet/dry deposition (first-order removal)
- Wind-driven advection (directional bias folded into effective λ)

This is a physics-similarity model (Gaussian-plume-like), not a full
CALPUFF/AERMOD regulatory dispersion model. It gives O(N) screening-level
exposure estimates at population receptors.

Room-scale diagnostics:
  - airborne_exposure_room_images: FIRST-ORDER image-source method. Mirrors all
    sources across the 6 room walls (7× total: original + 6 reflections) to
    approximate the no-flux wall condition to **first order**. No corner or
    edge images are included (these would be needed for second and higher
    order). Uses the free-space Yukawa Green's function; does NOT capture the
    well-mixed (n=0) mode — use the eigenfunction expansion for that regime.
  - airborne_exposure_room_eigen: eigenfunction (cosine) expansion for a
    rectangular room with Neumann (no-flux) wall BCs. This is the exact
    spectral solution and naturally includes the well-mixed mode
    C_wm = Q_total / (V·λ).
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.yukawa3d_fmm import Yukawa3DFMM


def airborne_exposure(
    sources: np.ndarray,
    emission_rates: np.ndarray,
    targets: np.ndarray,
    wind_speed: float = 3.0,
    mixing_height: float = 500.0,
    deposition_rate: float = 0.0001,
    domain_size: float = 1000.0,
    depth: int = 16,
    p: int = 8,
) -> np.ndarray:
    """Compute steady-state pollutant concentration at population receptors.

    Parameters
    ----------
    sources : (N_s, 3) — emission source locations in [0, domain_size]^3 (meters)
    emission_rates : (N_s,) — emission rates (g/s)
    targets : (N_t, 3) — receptor locations (meters)
    wind_speed : float — mean wind speed (m/s)
    mixing_height : float — atmospheric mixing height (meters)
    deposition_rate : float — dry+wet deposition rate (1/s)
    domain_size : float — domain extent (meters)
    depth : int — FMM grid resolution
    p : int — Taylor expansion order

    Returns
    -------
    concentrations : (N_t,) — pollutant concentration (μg/m³)
    """
    sources = np.asarray(sources, dtype=np.float64)
    emission_rates = np.asarray(emission_rates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Effective atmospheric mixing length:
    # λ = sqrt(D_atm / deposition_rate) where D_atm ~ wind_speed * mixing_height
    D_atm = wind_speed * mixing_height  # m²/s (turbulent diffusivity scale)
    lam = np.sqrt(D_atm / max(deposition_rate, 1e-10))  # meters
    kappa = 1.0 / lam  # 1/meters

    # Map to unit box.
    src_unit = sources / domain_size
    tgt_unit = targets / domain_size
    kappa_unit = kappa * domain_size

    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa_unit)
    pot_unit = fmm.evaluate_targets(src_unit, emission_rates, tgt_unit)

    # Rescale: concentration = pot_unit / domain_size
    concentrations = pot_unit / domain_size
    return concentrations


# ---------------------------------------------------------------------------
# Room-scale diagnostics: image sources + eigenfunction expansion
# ---------------------------------------------------------------------------

def _build_first_order_images(sources, room_dims):
    """Build the 7× source list: original + 6 wall reflections (first order).

    No corner or edge images (reflections of reflections) are included —
    these would be needed for second and higher order. This is the standard
    first-order image-source approximation for the no-flux (Neumann) wall
    condition.
    """
    Lx, Ly, Lz = room_dims
    src = np.asarray(sources, dtype=np.float64)
    images = [src]
    # Reflect across x=0 and x=Lx
    r = src.copy(); r[:, 0] = -src[:, 0];           images.append(r)
    r = src.copy(); r[:, 0] = 2 * Lx - src[:, 0];   images.append(r)
    # Reflect across y=0 and y=Ly
    r = src.copy(); r[:, 1] = -src[:, 1];           images.append(r)
    r = src.copy(); r[:, 1] = 2 * Ly - src[:, 1];   images.append(r)
    # Reflect across z=0 and z=Lz
    r = src.copy(); r[:, 2] = -src[:, 2];           images.append(r)
    r = src.copy(); r[:, 2] = 2 * Lz - src[:, 2];   images.append(r)
    return images  # list of 7 arrays


def airborne_exposure_room_images(
    sources: np.ndarray,
    emission_rates: np.ndarray,
    targets: np.ndarray,
    D_t: float,
    removal_rate: float,
    room_dims: tuple,
) -> np.ndarray:
    """First-order image-source concentration in a rectangular room.

    Mirrors all sources across the 6 room walls (7× total: original + 6
    reflections) to approximate the no-flux (Neumann) wall condition to
    **first order**. No corner or edge images are included.

    Uses the free-space Yukawa Green's function
    G(r) = exp(-κr) / (4π D_t r)  with  κ = √(λ/D_t).

    NOTE: this method does NOT capture the well-mixed (n=0) mode. In the
    uniformity regime (ℓ = 1/κ >> room scale) use ``airborne_exposure_room_eigen``
    for the correct mean concentration.
    """
    sources = np.asarray(sources, dtype=np.float64)
    emission_rates = np.asarray(emission_rates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    kappa = np.sqrt(removal_rate / D_t)

    img_list = _build_first_order_images(sources, room_dims)
    all_src = np.vstack(img_list)
    all_q = np.concatenate([emission_rates] * len(img_list))

    conc = np.zeros(len(targets))
    for i in range(len(targets)):
        diff = all_src - targets[i]
        r = np.linalg.norm(diff, axis=1)
        mask = r > 1e-10
        conc[i] = np.sum(all_q[mask] * np.exp(-kappa * r[mask]) / r[mask])
    conc *= 1.0 / (4.0 * np.pi * D_t)
    return conc


def airborne_exposure_room_eigen(
    sources: np.ndarray,
    emission_rates: np.ndarray,
    targets: np.ndarray,
    D_t: float,
    removal_rate: float,
    room_dims: tuple,
    n_max: int = 6,
) -> np.ndarray:
    """Eigenfunction (cosine) expansion for a rectangular room with Neumann BCs.

    C(r) = Σ_n [Σ_s Q_s φ_n(r_s)] / (λ + D_t k_n²) · φ_n(r)

    where φ_n are the normalized cosine eigenfunctions of -∇² with Neumann
    wall BCs. This is the exact spectral solution and includes the well-mixed
    n=(0,0,0) mode C_wm = Q_total / (V·λ).
    """
    sources = np.asarray(sources, dtype=np.float64)
    emission_rates = np.asarray(emission_rates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    Lx, Ly, Lz = room_dims
    lam = removal_rate
    V = Lx * Ly * Lz

    conc = np.zeros(len(targets))
    for n_x in range(n_max + 1):
        cx = np.cos(n_x * np.pi * targets[:, 0] / Lx)
        cxs = np.cos(n_x * np.pi * sources[:, 0] / Lx)
        for n_y in range(n_max + 1):
            cy = np.cos(n_y * np.pi * targets[:, 1] / Ly)
            cys = np.cos(n_y * np.pi * sources[:, 1] / Ly)
            for n_z in range(n_max + 1):
                cz = np.cos(n_z * np.pi * targets[:, 2] / Lz)
                czs = np.cos(n_z * np.pi * sources[:, 2] / Lz)
                k2 = (n_x * np.pi / Lx) ** 2 + (n_y * np.pi / Ly) ** 2 + (n_z * np.pi / Lz) ** 2
                denom = lam + D_t * k2
                if denom < 1e-30:
                    continue
                norm = 1.0
                if n_x > 0: norm *= 2
                if n_y > 0: norm *= 2
                if n_z > 0: norm *= 2
                C_norm = np.sqrt(norm / V)
                phi_tgt = C_norm * cx * cy * cz
                phi_src = C_norm * cxs * cys * czs
                A_n = np.sum(emission_rates * phi_src) / denom
                conc += A_n * phi_tgt
    return conc


def test_airborne_exposure():
    """Cross-validate vs direct O(N²) reference."""
    rng = np.random.RandomState(42)
    N_s = 25
    N_t = 30
    domain = 1000.0
    sources = rng.uniform(50, 950, size=(N_s, 3))
    # Stack heights: elevate z
    sources[:, 2] = rng.uniform(50, 200, size=N_s)
    emission_rates = rng.uniform(10, 100, size=N_s)
    targets = rng.uniform(50, 950, size=(N_t, 3))
    targets[:, 2] = rng.uniform(0, 20, size=N_t)  # ground-level receptors

    conc_fmm = airborne_exposure(
        sources, emission_rates, targets,
        wind_speed=3.0, mixing_height=500.0,
        deposition_rate=0.0001, domain_size=domain, depth=16, p=8
    )

    # Direct reference
    D_atm = 3.0 * 500.0
    lam = np.sqrt(D_atm / 0.0001)
    kappa = 1.0 / lam
    conc_direct = np.zeros(N_t)
    for i in range(N_t):
        for j in range(N_s):
            r = np.linalg.norm(targets[i] - sources[j])
            if r < 1e-10:
                continue
            conc_direct[i] += emission_rates[j] * np.exp(-kappa * r) / r

    rel = np.linalg.norm(conc_fmm - conc_direct) / max(1e-30, np.linalg.norm(conc_direct))
    print(f"  T-G4 airborne exposure: N_s={N_s}, N_t={N_t}, λ={lam:.1f}m, rel-L2 = {rel:.4e}")
    assert rel < 1e-5, f"T-G4 rel-L2 {rel} >= 1e-5"
    print("  T-G4 airborne exposure: PASS")
    return True


def test_well_mixed_anchor():
    """T-G4 well-mixed anchor: in the uniformity regime (ℓ >> room scale) the
    mean concentration must match Q_total/(V·λ) and the field must be nearly
    uniform (max/min ≤ 1.1).

    Uses the eigenfunction expansion (exact Neumann solution) which includes
    the well-mixed n=0 mode. Parameters: D_t=10 m²/s, λ=1e-3 1/s, room
    10×10×3 m → ℓ = √(D_t/λ) = 100 m >> 10 m room scale.
    """
    D_t = 10.0
    lam = 1e-3
    room = (10.0, 10.0, 3.0)
    Lx, Ly, Lz = room
    V = Lx * Ly * Lz
    ell = np.sqrt(D_t / lam)

    sources = np.array([[2.0, 3.0, 1.0], [7.0, 7.0, 2.0], [5.0, 5.0, 1.5]])
    Q = np.array([1.0, 1.5, 0.8])
    Q_total = float(np.sum(Q))
    C_wm = Q_total / (V * lam)

    # Receptor grid (avoid exact source locations)
    gx = np.linspace(0.5, 9.5, 10)
    gy = np.linspace(0.5, 9.5, 10)
    gz = np.linspace(0.3, 2.7, 5)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    targets = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    conc = airborne_exposure_room_eigen(sources, Q, targets, D_t, lam, room, n_max=6)
    mean_c = float(np.mean(conc))
    max_c = float(np.max(conc))
    min_c = float(np.min(conc))
    rel_err = abs(mean_c - C_wm) / C_wm
    ratio = max_c / min_c

    print(f"  T-G4 well-mixed anchor: ℓ={ell:.0f}m >> room, Q_total={Q_total}, "
          f"V={V}, C_wm={C_wm:.4f}")
    print(f"    mean(C)={mean_c:.6f}, rel_err={rel_err:.2e}, "
          f"max/min={ratio:.6f} (max={max_c:.4f}, min={min_c:.4f})")
    assert rel_err < 0.05, f"well-mixed mean error {rel_err:.2e} >= 5%"
    assert ratio <= 1.1, f"max/min ratio {ratio:.4f} > 1.1"
    print("  T-G4 well-mixed anchor: PASS")
    return True


def test_wall_symmetry():
    """T-G4 wall-symmetry: with a mirror-symmetric source layout, the
    image-source field must be mirror-symmetric to ≤ 1e-10."""
    D_t = 10.0
    lam = 0.01
    room = (10.0, 10.0, 3.0)

    # Source at room center — perfectly symmetric about all 3 mid-planes.
    sources = np.array([[5.0, 5.0, 1.5]])
    Q = np.array([1.0])

    # Symmetric receptor pairs about x=5, y=5, z=1.5
    targets_left = np.array([
        [3.0, 5.0, 1.5],   # mirror about x=5
        [5.0, 2.0, 1.5],   # mirror about y=5
        [5.0, 5.0, 0.5],   # mirror about z=1.5
        [2.0, 3.0, 1.0],   # mirror about x=5 and y=5
    ])
    targets_right = np.array([
        [7.0, 5.0, 1.5],
        [5.0, 8.0, 1.5],
        [5.0, 5.0, 2.5],
        [8.0, 7.0, 1.0],
    ])

    c_left = airborne_exposure_room_images(sources, Q, targets_left, D_t, lam, room)
    c_right = airborne_exposure_room_images(sources, Q, targets_right, D_t, lam, room)
    max_err = float(np.max(np.abs(c_left - c_right)))
    print(f"  T-G4 wall-symmetry: max |C_left - C_right| = {max_err:.2e}")
    assert max_err <= 1e-10, f"symmetry error {max_err:.2e} > 1e-10"
    print("  T-G4 wall-symmetry: PASS")
    return True


if __name__ == "__main__":
    test_airborne_exposure()
    test_well_mixed_anchor()
    test_wall_symmetry()
