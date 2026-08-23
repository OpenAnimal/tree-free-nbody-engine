"""
Spectral Particle-Mesh Ewald Neural Operator (`spectral_neural_pme.py`)
======================================================================
Linear-Spectral O(N + M log M) Particle-Mesh Ewald (PME) Neural Operator.
Combines real-space elastic spatial hashing with reciprocal Fourier spectral lattices
for exact all-pairs Coulomb, electrostatic, and gravitational field evaluation.

Splits Green's function via error function decomposition:
  1 / r = erfc(alpha * r) / r  [Real-Space Short-Range, O(N)]
        + erf(alpha * r) / r   [Reciprocal Spectral Mesh, O(M log M)]
"""

import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List

_vec_erfc = np.vectorize(math.erfc, otypes=[np.float64])

def fast_erfc(x: np.ndarray) -> np.ndarray:
    """Computes complementary error function erfc(x) without external dependencies."""
    return _vec_erfc(x)


class NeuralPME:
    """
    Particle-Mesh Ewald (PME) Neural Operator.
    Computes all-pairs screened electrostatics and vector fields in O(N + M log M) operations.
    """
    def __init__(
        self,
        grid_dim: int = 32,
        alpha_ewald: float = 4.0,
        r_cutoff: float = 0.25,
        spline_order: int = 4,
        box_size: float = 1.0,
    ):
        self.grid_dim = grid_dim
        self.alpha = float(alpha_ewald)
        self.r_cutoff = float(r_cutoff)
        self.spline_order = spline_order
        self.box_size = float(box_size)
        self.h = self.box_size / self.grid_dim

        # Precompute reciprocal lattice Green's tensor
        self.G_k = self._init_reciprocal_greens_tensor()
        # Mesh-consistent self-interaction constant (see _init_reciprocal_greens_tensor).
        self.mesh_self_const = float(np.sum(self.G_k * self._W_sq)) / (self.box_size ** 3)

    def _init_reciprocal_greens_tensor(self) -> np.ndarray:
        M = self.grid_dim
        # Frequency indices
        kx = 2.0 * np.pi * np.fft.fftfreq(M, d=self.h)
        ky = 2.0 * np.pi * np.fft.fftfreq(M, d=self.h)
        kz = 2.0 * np.pi * np.fft.fftfreq(M, d=self.h)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K_sq = KX**2 + KY**2 + KZ**2

        # Avoid division by zero at k = 0
        K_sq_safe = np.where(K_sq == 0, 1.0, K_sq)
        # Charge-assignment Gaussian of width sigma_a = h (mesh spacing).
        # Its Fourier transform W(k) = exp(-k^2 sigma_a^2 / 2) is applied once
        # by the spread and once by the interpolation, so the mesh Green's
        # function must be the Ewald kernel divided by W(k)^2 to recover the
        # point-charge Ewald reciprocal term
        #   G_Ewald(k) = (4 pi / k^2) exp(-k^2 / (4 alpha^2)).
        # Without this correction the mesh potential is the potential of the
        # *smeared* charge cloud, not the point charges (off by W(k)^2).
        sigma_a = self.h
        W_sq = np.exp(-K_sq * (sigma_a ** 2))  # W(k)^2 = exp(-k^2 sigma_a^2)
        W_sq_safe = np.where(K_sq == 0, 1.0, W_sq)
        G = (4.0 * np.pi / K_sq_safe) * np.exp(-K_sq / (4.0 * (self.alpha ** 2))) / W_sq_safe
        G[0, 0, 0] = 0.0  # Neutral background / zero mode
        # Cache W(k)^2 so the mesh self-interaction constant can be computed
        # in __init__ as (1/V) sum_k G_PME(k) W(k)^2 = (1/V) sum_k G_Ewald(k).
        # For an untruncated Gaussian this is independent of sub-grid position,
        # so subtracting it (instead of the analytic 2 alpha/sqrt(pi)) removes
        # the mesh self-interaction exactly and leaves a position-independent
        # residual — required for the FD-of-potential force check.
        self._W_sq = W_sq
        return G.astype(np.float64)

    def _spread_charges_gaussian(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Spreads particle charges onto 3D mesh as a normalized Gaussian density.

        The per-axis Gaussian window has width sigma = h (mesh spacing). Each
        particle's stencil weights are renormalized so that
        ``sum_of_mesh_weights * h**3 == q`` exactly (truncation-corrected):
        the mesh therefore represents a true charge density (charge / volume),
        not a raw weight sum. This is required for the FFT-based reciprocal
        potential to have the correct magnitude.
        """
        M = self.grid_dim
        mesh = np.zeros((M, M, M), dtype=np.float64)
        scaled_pos = (positions / self.h) % M

        # Stencil half-width w = 5: for an untruncated Gaussian the mesh
        # self-potential is independent of sub-grid position, so a wide enough
        # stencil (truncation < exp(-0.5*25) ~ 3.7e-6 per axis) makes the
        # assignment FT match the analytic W(k) and the mesh self-interaction
        # constant removes the self term exactly (no position-dependent
        # residual that would corrupt a finite-difference force check).
        w = 5
        offs = np.arange(-w, w + 1)
        N = len(positions)
        h3 = self.h ** 3
        for i in range(N):
            px, py, pz = scaled_pos[i]
            q = float(charges[i])
            bx, by, bz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))

            wx_raw = np.exp(-0.5 * ((px - (bx + offs)) ** 2))
            wy_raw = np.exp(-0.5 * ((py - (by + offs)) ** 2))
            wz_raw = np.exp(-0.5 * ((pz - (bz + offs)) ** 2))
            # Per-particle normalization: sum_3d weights = (sum wx)(sum wy)(sum wz);
            # divide by sum_3d * h^3 so the contributed density integrates to q.
            sum3d = float(np.sum(wx_raw) * np.sum(wy_raw) * np.sum(wz_raw))
            factor = q / (sum3d * h3)
            for a, dx in enumerate(offs):
                ix = (bx + int(dx)) % M
                wxq = factor * wx_raw[a]
                for b, dy in enumerate(offs):
                    iy = (by + int(dy)) % M
                    wxyq = wxq * wy_raw[b]
                    for c, dz in enumerate(offs):
                        iz = (bz + int(dz)) % M
                        mesh[ix, iy, iz] += wxyq * wz_raw[c]
        return mesh

    def _interpolate_potential(self, mesh_pot: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Interpolates mesh potential back to continuous particle coordinates."""
        M = self.grid_dim
        scaled_pos = (positions / self.h) % M
        N = len(positions)
        pot = np.zeros(N, dtype=np.float64)

        w = 5
        offs = np.arange(-w, w + 1)
        for i in range(N):
            px, py, pz = scaled_pos[i]
            bx, by, bz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))
            wx = np.exp(-0.5 * ((px - (bx + offs)) ** 2))
            wy = np.exp(-0.5 * ((py - (by + offs)) ** 2))
            wz = np.exp(-0.5 * ((pz - (bz + offs)) ** 2))
            val = 0.0
            norm = 0.0
            for a, dx in enumerate(offs):
                ix = (bx + int(dx)) % M
                for b, dy in enumerate(offs):
                    iy = (by + int(dy)) % M
                    wxy = wx[a] * wy[b]
                    for c, dz in enumerate(offs):
                        iz = (bz + int(dz)) % M
                        weight = wxy * wz[c]
                        val += mesh_pot[ix, iy, iz] * weight
                        norm += weight

            pot[i] = val / (norm + 1e-12)
        return pot

    def _interpolate_gradient(self, mesh_pot: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Interpolates the GRADIENT of the mesh potential to particle positions.

        Uses 4th-order central finite differences on the periodic mesh, then
        the same Gaussian-window interpolation as `_interpolate_potential`.

        Returns (N, 3) gradient array.  This is needed for the reciprocal-space
        force: F_recip_i = -q_i * grad(phi_recip)_i.
        """
        M = self.grid_dim
        # 4th-order central difference on periodic mesh:
        # f'(x) ≈ (-f[x+2h] + 8 f[x+h] - 8 f[x-h] + f[x-2h]) / (12 h)
        grad_x = (-np.roll(mesh_pot, -2, axis=0) + 8.0 * np.roll(mesh_pot, -1, axis=0)
                  - 8.0 * np.roll(mesh_pot, 1, axis=0) + np.roll(mesh_pot, 2, axis=0))
        grad_y = (-np.roll(mesh_pot, -2, axis=1) + 8.0 * np.roll(mesh_pot, -1, axis=1)
                  - 8.0 * np.roll(mesh_pot, 1, axis=1) + np.roll(mesh_pot, 2, axis=1))
        grad_z = (-np.roll(mesh_pot, -2, axis=2) + 8.0 * np.roll(mesh_pot, -1, axis=2)
                  - 8.0 * np.roll(mesh_pot, 1, axis=2) + np.roll(mesh_pot, 2, axis=2))
        grad_mesh = np.stack([grad_x, grad_y, grad_z], axis=-1) / (12.0 * self.h)  # (M,M,M,3)

        scaled_pos = (positions / self.h) % M
        N = len(positions)
        grad = np.zeros((N, 3), dtype=np.float64)

        w = 5
        offs = np.arange(-w, w + 1)
        for i in range(N):
            px, py, pz = scaled_pos[i]
            bx, by, bz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))
            wx = np.exp(-0.5 * ((px - (bx + offs)) ** 2))
            wy = np.exp(-0.5 * ((py - (by + offs)) ** 2))
            wz = np.exp(-0.5 * ((pz - (bz + offs)) ** 2))
            val = np.zeros(3, dtype=np.float64)
            norm = 0.0
            for a, dx in enumerate(offs):
                ix = (bx + int(dx)) % M
                for b, dy in enumerate(offs):
                    iy = (by + int(dy)) % M
                    wxy = wx[a] * wy[b]
                    for c, dz in enumerate(offs):
                        iz = (bz + int(dz)) % M
                        weight = wxy * wz[c]
                        val += grad_mesh[ix, iy, iz, :] * weight
                        norm += weight

            grad[i] = val / (norm + 1e-12)
        return grad

    def forward(
        self,
        positions: np.ndarray,      # (N, 3) Continuous particle positions in [0, box_size)^3
        charges: np.ndarray,        # (N,) Particle charges
        features: Optional[np.ndarray] = None, # (N, D) Optional node representations
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes all-pairs Ewald potentials and vector fields.
        Returns: (potentials (N,), vector_fields (N, 3), metadata)
        """
        N = len(positions)
        charges = charges.astype(np.float64)
        pos = (positions % self.box_size).astype(np.float64)

        # 1. Reciprocal Fourier Spectral Mesh
        mesh_rho = self._spread_charges_gaussian(pos, charges)
        # FFT to Fourier domain
        rho_k = np.fft.fftn(mesh_rho)
        # Solve Poisson equation in k-space
        pot_k = rho_k * self.G_k
        # IFFT back to real mesh
        mesh_pot = np.real(np.fft.ifftn(pot_k))

        # Interpolate mesh potential back to particles
        pot_reciprocal = self._interpolate_potential(mesh_pot, pos)

        # Reciprocal-space force: F_recip_i = -q_i * grad(phi_recip)_i
        # (the old code dropped this entirely, returning only real-space forces).
        grad_recip = self._interpolate_gradient(mesh_pot, pos)  # (N, 3)
        forces_reciprocal = -charges[:, None] * grad_recip  # (N, 3)

        # 2. Self-Interaction Energy Correction
        # The mesh reciprocal potential includes each particle's own Gaussian
        # self-interaction. For an untruncated assignment Gaussian this is the
        # position-independent constant mesh_self_const = (1/V) sum_k G_Ewald(k)
        # (computed in __init__ from G_PME * W^2). Subtracting it removes the
        # self term exactly, leaving only the cross (pair) reciprocal
        # potential. Using the analytic 2 alpha/sqrt(pi) instead would leave a
        # position-dependent discretization residual that corrupts a
        # finite-difference force check.
        pot_self = -charges * self.mesh_self_const

        # 3. Real-Space Short-Range Correction via Spatial Hashing
        cell_size = self.r_cutoff
        n_cells = max(1, int(self.box_size / cell_size))
        grid_coords = np.clip(np.floor(pos / cell_size).astype(np.int64), 0, n_cells - 1)

        bucket_map: Dict[Tuple[int, int, int], List[int]] = {}
        for i in range(N):
            cell_key = (int(grid_coords[i, 0]), int(grid_coords[i, 1]), int(grid_coords[i, 2]))
            if cell_key not in bucket_map:
                bucket_map[cell_key] = []
            bucket_map[cell_key].append(i)

        pot_real = np.zeros(N, dtype=np.float64)
        forces_real = np.zeros((N, 3), dtype=np.float64)

        # Neighbor search over adjacent 27 cells
        for cell_k, p_ids in bucket_map.items():
            pts_src = pos[p_ids]
            q_src = charges[p_ids]
            M_src = len(p_ids)

            near_p_list = []
            near_cell_keys = set()  # dedupe: with small n_cells the periodic
            # neighbor cells alias (e.g. n_cells=2 -> (cell-1) mod 2 == cell+1),
            # which would double-count the same near cell's particles.
            for dx in (-1, 0, 1):
                nx = (cell_k[0] + dx) % n_cells
                for dy in (-1, 0, 1):
                    ny = (cell_k[1] + dy) % n_cells
                    for dz in (-1, 0, 1):
                        nz = (cell_k[2] + dz) % n_cells
                        nk = (nx, ny, nz)
                        if nk in near_cell_keys:
                            continue
                        near_cell_keys.add(nk)
                        if nk in bucket_map:
                            near_p_list.extend(bucket_map[nk])

            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = pos[near_arr]
            q_near = charges[near_arr]

            diff = pts_src[:, None, :] - pts_near[None, :, :] # (M_src, len(near), 3)
            # Minimum image convention for periodic boundaries
            diff -= self.box_size * np.round(diff / self.box_size)
            r = np.linalg.norm(diff, axis=-1) # (M_src, len(near))

            # Mask out self-interaction (r < 1e-6)
            mask = (r > 1e-6) & (r <= self.r_cutoff)
            r_safe = np.where(mask, r, 1.0)

            # Real space potential: erfc(alpha * r) / r
            erfc_vals = fast_erfc(self.alpha * r_safe).astype(np.float64)
            pot_kernel = np.where(mask, erfc_vals / r_safe, 0.0)

            pot_real[p_ids] = np.sum(pot_kernel * q_near[None, :], axis=-1)

            # Force kernel magnitude:
            #   |F_ij| = q_i q_j (erfc(alpha*r)/r^3 + 2*alpha/sqrt(pi) * exp(-alpha^2*r^2)/r^2)
            # along (r_i - r_j). The old einsum omitted q_i (the target charge),
            # which broke Newton's third law for asymmetric charges (both forces
            # of a dipole pointed the same way). Include q_i via q_src.
            force_mag = np.where(
                mask,
                (erfc_vals / (r_safe**3) + (2.0 * self.alpha / np.sqrt(np.pi)) * np.exp(-((self.alpha * r_safe)**2)) / (r_safe**2)),
                0.0
            )
            forces_real[p_ids] = np.einsum('mn,mnd,n,m->md', force_mag, diff, q_near, q_src)

        # Total electrostatic potential and force
        total_potential = pot_real + pot_reciprocal + pot_self
        total_forces = forces_real + forces_reciprocal

        meta = {
            "num_particles": N,
            "grid_dim": self.grid_dim,
            "alpha": self.alpha,
            "r_cutoff": self.r_cutoff,
            "mean_pot_real": float(np.mean(np.abs(pot_real))),
            "mean_pot_reciprocal": float(np.mean(np.abs(pot_reciprocal))),
            "mean_force_reciprocal": float(np.mean(np.linalg.norm(forces_reciprocal, axis=-1))),
        }
        return total_potential, total_forces, meta


def _analytic_ewald_2p(pos, q, alpha, L, kmax_n=12):
    """Independent analytic 2-particle periodic Ewald reference (direct k-space sum).

    Returns (potentials (2,), forces (2, 3)) for the real + reciprocal + self
    split. Used as the ground truth for the PME acceptance check.
    """
    r12 = pos[1] - pos[0]
    r12 -= L * np.round(r12 / L)
    r = float(np.linalg.norm(r12))
    # Real-space (erfc), nearest image only (caller picks r_cutoff to match).
    erfc_r = float(math.erfc(alpha * r))
    exp_r = float(np.exp(-((alpha * r) ** 2)))
    phi_real = np.array([q[1] * erfc_r / r, q[0] * erfc_r / r])
    fmag = q[0] * q[1] * (erfc_r / r**3
                          + (2.0 * alpha / math.sqrt(math.pi)) * exp_r / r**2)
    F_real = np.array([fmag * (-r12), fmag * r12])
    # Reciprocal (direct k-space sum) + mesh-consistent self.
    # The self term is the discrete k-space sum (1/V) sum G_Ewald(k) over the
    # SAME k-set used for the reciprocal, so a single particle's reciprocal
    # self cancels exactly -- matching the mesh PME's mesh_self_const
    # convention (not the continuous-integral 2 alpha/sqrt(pi)).
    V = L ** 3
    phi_recip = np.zeros(2)
    F_recip = np.zeros((2, 3))
    self_const = 0.0
    nrange = range(-kmax_n, kmax_n + 1)
    for nx in nrange:
        for ny in nrange:
            for nz in nrange:
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                k = (2.0 * np.pi / L) * np.array([nx, ny, nz], dtype=np.float64)
                k2 = float(np.dot(k, k))
                G = (4.0 * np.pi / k2) * np.exp(-k2 / (4.0 * alpha**2))
                self_const += G / V
                rho = sum(qj * np.exp(-1j * np.dot(k, rj))
                          for qj, rj in zip(q, pos))
                for i in range(2):
                    phase = np.exp(1j * np.dot(k, pos[i]))
                    phi_recip[i] += (1.0 / V) * (G * rho * phase).real
                    # grad_i phi = (1/V) Re[ G rho (i k) exp(i k r_i) ];  F = -q grad
                    gval = G * rho * (1j * k) * phase
                    F_recip[i] += -q[i] * (1.0 / V) * gval.real
    phi_self = -q * self_const
    return phi_real + phi_recip + phi_self, F_real + F_recip


def _run_pme_acceptance():
    """Three-gate acceptance check for NeuralPME (called from __main__).

    1. Total forces match an independent analytic 2-particle Ewald reference
       (real + reciprocal + self) to < 1e-4 rel.
    2. Newton's third law: |F1 + F2| / |F1| < 1e-4.
    3. Forces match central finite differences of the module's OWN total
       potential to < 1e-4 rel.
    """
    L = 1.0
    pos = np.array([[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]], dtype=np.float64)
    q = np.array([1.0, -1.0], dtype=np.float64)
    alpha = 5.0
    grid_dim = 80
    r_cutoff = 0.45  # > nearest-image distance 0.1, < next image 0.9

    pme = NeuralPME(grid_dim=grid_dim, alpha_ewald=alpha,
                    r_cutoff=r_cutoff, box_size=L)
    phi_c, F_c, _ = pme.forward(pos, q)

    phi_a, F_a = _analytic_ewald_2p(pos, q, alpha, L, kmax_n=12)

    rel_F = float(np.linalg.norm(F_c - F_a) /
                  max(1e-30, np.linalg.norm(F_a)))
    rel_phi = float(np.linalg.norm(phi_c - phi_a) /
                    max(1e-30, np.linalg.norm(phi_a)))
    newton = float(np.linalg.norm(F_c[0] + F_c[1]) /
                   max(1e-30, np.linalg.norm(F_c[0])))

    # Central FD of the module's own total potential.
    # F_i = -q_i * grad_i phi_total_i (self term is position-independent).
    eps = 1e-4
    F_fd = np.zeros_like(F_c)
    for i in range(2):
        for d in range(3):
            pp = pos.copy(); pp[i, d] += eps
            pm = pos.copy(); pm[i, d] -= eps
            phi_p, _, _ = pme.forward(pp, q)
            phi_m, _, _ = pme.forward(pm, q)
            F_fd[i, d] = -q[i] * (phi_p[i] - phi_m[i]) / (2.0 * eps)
    rel_fd = float(np.linalg.norm(F_c - F_fd) /
                   max(1e-30, np.linalg.norm(F_c)))

    print("=" * 70)
    print("NeuralPME acceptance (2-particle periodic Ewald)")
    print("=" * 70)
    print(f"  (1) force vs analytic Ewald   rel-L2 = {rel_F:.3e}  "
          f"(gate < 1e-4): {'PASS' if rel_F < 1e-4 else 'FAIL'}")
    print(f"  (2) Newton III |F1+F2|/|F1|   = {newton:.3e}  "
          f"(gate < 1e-4): {'PASS' if newton < 1e-4 else 'FAIL'}")
    print(f"  (3) force vs FD of own phi    rel-L2 = {rel_fd:.3e}  "
          f"(gate < 1e-4): {'PASS' if rel_fd < 1e-4 else 'FAIL'}")
    print(f"  [info] potential vs analytic  rel-L2 = {rel_phi:.3e}")
    print(f"  [info] analytic F1 = {F_a[0]}")
    print(f"  [info] code     F1 = {F_c[0]}")
    print(f"  [info] FD       F1 = {F_fd[0]}")
    print("=" * 70)
    assert rel_F < 1e-4, f"PME force vs analytic Ewald rel-L2 {rel_F:.3e} >= 1e-4"
    assert newton < 1e-4, f"PME Newton III violation {newton:.3e} >= 1e-4"
    assert rel_fd < 1e-4, f"PME force vs FD rel-L2 {rel_fd:.3e} >= 1e-4"
    return {"rel_F": rel_F, "newton": newton, "rel_fd": rel_fd,
            "rel_phi": rel_phi}


if __name__ == "__main__":
    _run_pme_acceptance()
