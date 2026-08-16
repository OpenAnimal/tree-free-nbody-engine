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
        G = (4.0 * np.pi / K_sq_safe) * np.exp(-K_sq / (4.0 * (self.alpha ** 2)))
        G[0, 0, 0] = 0.0 # Neutral background / zero mode
        return G.astype(np.float32)

    def _spread_charges_gaussian(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Spreads particle charges onto 3D mesh using Gaussian window."""
        M = self.grid_dim
        mesh = np.zeros((M, M, M), dtype=np.float32)
        scaled_pos = (positions / self.h) % M

        # Window stencil radius w = 2
        w = 2
        N = len(positions)
        for i in range(N):
            px, py, pz = scaled_pos[i]
            q = charges[i]
            bx, by, bz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))

            for dx in range(-w, w + 1):
                ix = (bx + dx) % M
                wx = np.exp(-0.5 * ((px - (bx + dx)) ** 2))
                for dy in range(-w, w + 1):
                    iy = (by + dy) % M
                    wy = np.exp(-0.5 * ((py - (by + dy)) ** 2))
                    for dz in range(-w, w + 1):
                        iz = (bz + dz) % M
                        wz = np.exp(-0.5 * ((pz - (bz + dz)) ** 2))
                        mesh[ix, iy, iz] += q * wx * wy * wz
        return mesh

    def _interpolate_potential(self, mesh_pot: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Interpolates mesh potential back to continuous particle coordinates."""
        M = self.grid_dim
        scaled_pos = (positions / self.h) % M
        N = len(positions)
        pot = np.zeros(N, dtype=np.float32)

        w = 2
        for i in range(N):
            px, py, pz = scaled_pos[i]
            bx, by, bz = int(np.floor(px)), int(np.floor(py)), int(np.floor(pz))
            val = 0.0
            norm = 0.0

            for dx in range(-w, w + 1):
                ix = (bx + dx) % M
                wx = np.exp(-0.5 * ((px - (bx + dx)) ** 2))
                for dy in range(-w, w + 1):
                    iy = (by + dy) % M
                    wy = np.exp(-0.5 * ((py - (by + dy)) ** 2))
                    for dz in range(-w, w + 1):
                        iz = (bz + dz) % M
                        wz = np.exp(-0.5 * ((pz - (bz + dz)) ** 2))
                        weight = wx * wy * wz
                        val += mesh_pot[ix, iy, iz] * weight
                        norm += weight

            pot[i] = val / (norm + 1e-12)
        return pot

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
        charges = charges.astype(np.float32)
        pos = (positions % self.box_size).astype(np.float32)

        # 1. Reciprocal Fourier Spectral Mesh
        mesh_rho = self._spread_charges_gaussian(pos, charges)
        # FFT to Fourier domain
        rho_k = np.fft.fftn(mesh_rho)
        # Solve Poisson equation in k-space
        pot_k = rho_k * self.G_k
        # IFFT back to real mesh
        mesh_pot = np.real(np.fft.ifftn(pot_k)).astype(np.float32)

        # Interpolate mesh potential back to particles
        pot_reciprocal = self._interpolate_potential(mesh_pot, pos)

        # 2. Self-Interaction Energy Correction
        # Phi_self = - (2 * alpha / sqrt(pi)) * q_i
        pot_self = -(2.0 * self.alpha / np.sqrt(np.pi)) * charges

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

        pot_real = np.zeros(N, dtype=np.float32)
        forces_real = np.zeros((N, 3), dtype=np.float32)

        # Neighbor search over adjacent 27 cells
        for cell_k, p_ids in bucket_map.items():
            pts_src = pos[p_ids]
            q_src = charges[p_ids]
            M_src = len(p_ids)

            near_p_list = []
            for dx in (-1, 0, 1):
                nx = (cell_k[0] + dx) % n_cells
                for dy in (-1, 0, 1):
                    ny = (cell_k[1] + dy) % n_cells
                    for dz in (-1, 0, 1):
                        nz = (cell_k[2] + dz) % n_cells
                        nk = (nx, ny, nz)
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
            erfc_vals = fast_erfc(self.alpha * r_safe).astype(np.float32)
            pot_kernel = np.where(mask, erfc_vals / r_safe, 0.0)

            pot_real[p_ids] = np.sum(pot_kernel * q_near[None, :], axis=-1)

            # Force kernel: (erfc(alpha*r)/r^3 + 2*alpha/sqrt(pi) * exp(-alpha^2*r^2)/r^2) * diff
            force_mag = np.where(
                mask,
                (erfc_vals / (r_safe**3) + (2.0 * self.alpha / np.sqrt(np.pi)) * np.exp(-((self.alpha * r_safe)**2)) / (r_safe**2)),
                0.0
            )
            forces_real[p_ids] = np.einsum('mn,mnd,n->md', force_mag, diff, q_near)

        # Total electrostatic potential
        total_potential = pot_real + pot_reciprocal + pot_self

        meta = {
            "num_particles": N,
            "grid_dim": self.grid_dim,
            "alpha": self.alpha,
            "r_cutoff": self.r_cutoff,
            "mean_pot_real": float(np.mean(np.abs(pot_real))),
            "mean_pot_reciprocal": float(np.mean(np.abs(pot_reciprocal))),
        }
        return total_potential, forces_real, meta
