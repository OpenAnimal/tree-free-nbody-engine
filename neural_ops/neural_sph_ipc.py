"""
Neural SPH & Incremental Potential Contact (IPC) Continuum Mechanics Layer (`neural_sph_ipc.py`)
================================================================================================
Linear-Time O(N) Physical Neural Operator combining:
1. Smoothed Particle Hydrodynamics (SPH) Navier-Stokes fluid pressure & viscosity.
2. Incremental Potential Contact (IPC) exact non-penetration barrier gradients.
3. Neural latent feature message passing without explicit dynamic edge-lists.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class NeuralSPHIPCLayer:
    """
    All-Pairs Continuum Mechanics & Neural SPH / IPC Layer.
    Computes fluid density, SPH pressure/viscosity forces, IPC barrier contact forces,
    and neural state updates in O(N) linear time.
    """
    def __init__(
        self,
        hidden_dim: int = 32,
        smoothing_h: float = 0.1,
        contact_dhat: float = 0.05,
        barrier_stiffness: float = 1e4,
        rest_density: float = 1000.0,
        bulk_modulus: float = 50.0,
        viscosity_mu: float = 0.1,
        grid_depth: int = 4,
    ):
        self.hidden_dim = hidden_dim
        self.h = float(smoothing_h)
        self.d_hat = float(contact_dhat)
        self.kappa_barrier = float(barrier_stiffness)
        self.rho_0 = float(rest_density)
        self.k_bulk = float(bulk_modulus)
        self.mu = float(viscosity_mu)
        self.grid_depth = grid_depth
        self.grid_res = 1 << grid_depth

        # SPH neighbor ring: the implemented cubic spline kernel
        # (`_eval_sph_kernel`) uses q = r/h with support q <= 1, i.e. the
        # kernel's support is h (NOT 2*h). The neighbor search therefore
        # covers ceil(h / cell) cells per axis, which is correct for this
        # kernel. (A textbook cubic spline with q = r/(2h) would have support
        # 2*h and need a wider ring; this code does not use that form.)
        cell = 1.0 / self.grid_res
        self.sph_ring = max(1, int(np.ceil(self.h / cell)))

        # Cubic spline kernel normalization in 3D
        self.sigma_3d = 8.0 / (np.pi * (self.h ** 3))

        # Neural feature update weights
        rng = np.random.RandomState(42)
        # Input features: hidden_dim + 1 (density) + 3 (pressure force) + 3 (visc force) + 3 (contact force)
        in_dim = hidden_dim + 10
        scale = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.normal(0, scale, size=(in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

    def _eval_sph_kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluates 3D cubic spline SPH smoothing kernel W(r, h)."""
        q = r / self.h
        mask1 = q <= 0.5
        mask2 = (q > 0.5) & (q <= 1.0)
        
        w = np.zeros_like(r, dtype=np.float32)
        w[mask1] = self.sigma_3d * (6.0 * (q[mask1]**3 - q[mask1]**2) + 1.0)
        w[mask2] = self.sigma_3d * (2.0 * ((1.0 - q[mask2]) ** 3))
        return w

    def _eval_sph_grad_kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluates dW/dr (magnitude of gradient)."""
        q = r / self.h
        mask1 = q <= 0.5
        mask2 = (q > 0.5) & (q <= 1.0)

        dw_dr = np.zeros_like(r, dtype=np.float32)
        dw_dr[mask1] = (self.sigma_3d / self.h) * (18.0 * (q[mask1]**2) - 12.0 * q[mask1])
        dw_dr[mask2] = (self.sigma_3d / self.h) * (-6.0 * ((1.0 - q[mask2]) ** 2))
        return dw_dr

    def _eval_ipc_barrier_grad(self, d: np.ndarray) -> np.ndarray:
        """
        Evaluates IPC log-barrier force magnitude:
        B'(d) = -kappa * [ 2*(d - d_hat)/d_hat^2 * ln(d/d_hat) + (d - d_hat)^2 / (d_hat^2 * d) ]
        for d < d_hat.
        """
        mask = (d > 1e-6) & (d < self.d_hat)
        d_safe = np.where(mask, d, self.d_hat)

        ratio = d_safe / self.d_hat
        diff = d_safe - self.d_hat
        
        grad = np.zeros_like(d, dtype=np.float32)
        term1 = (2.0 * diff / (self.d_hat ** 2)) * np.log(ratio)
        term2 = (diff ** 2) / ((self.d_hat ** 2) * d_safe)
        grad[mask] = -self.kappa_barrier * (term1[mask] + term2[mask])
        return grad

    def _morton_encode(self, coords: np.ndarray) -> np.ndarray:
        res = self.grid_res
        grid_indices = np.clip(np.floor(coords * res).astype(np.int64), 0, res - 1)
        return grid_indices[:, 0] + grid_indices[:, 1] * res + grid_indices[:, 2] * (res ** 2)

    def forward(
        self,
        positions: np.ndarray,      # (N, 3) Particle positions in [0, 1)^3
        velocities: np.ndarray,     # (N, 3) Particle velocities
        masses: np.ndarray,         # (N,) Particle masses
        hidden_states: np.ndarray,  # (N, hidden_dim) Latent neural features
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes physical fluid/contact forces and updates neural hidden states.
        Returns: (new_hidden (N, hidden_dim), total_forces (N, 3), densities (N,), metadata)
        """
        N = len(positions)
        masses = masses.astype(np.float32)
        vel = velocities.astype(np.float32)
        pos = np.clip(positions, 1e-4, 1.0 - 1e-4).astype(np.float32)

        # 1. Bucket particles into spatial hash grid
        keys = self._morton_encode(pos)
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            k = int(keys[i])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        cluster_keys = list(bucket_map.keys())
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        # 2. Phase 1: Compute SPH Densities rho_i = sum_j m_j W(r_ij, h)
        densities = np.zeros(N, dtype=np.float32)
        res = self.grid_res

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = pos[p_src_arr]
            center_src = np.mean(pts_src, axis=0)

            src_grid = np.floor(center_src * res).astype(np.int64)
            near_p_list = []
            ring = self.sph_ring
            for dx in range(-ring, ring + 1):
                nx = src_grid[0] + dx
                if 0 <= nx < res:
                    for dy in range(-ring, ring + 1):
                        ny = src_grid[1] + dy
                        if 0 <= ny < res:
                            for dz in range(-ring, ring + 1):
                                nz = src_grid[2] + dz
                                if 0 <= nz < res:
                                    nk = int(nx + ny * res + nz * (res ** 2))
                                    if nk in key_to_idx:
                                        near_p_list.extend(bucket_map[nk])

            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = pos[near_arr]
            m_near = masses[near_arr]

            diff = pts_src[:, None, :] - pts_near[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)

            W_vals = self._eval_sph_kernel(dist)
            densities[p_src_arr] = np.sum(W_vals * m_near[None, :], axis=-1)

        # Tait's Equation of State for pressure: P_i = k_bulk * ((rho_i / rho_0)^7 - 1)
        densities = np.maximum(densities, 1e-4)
        pressures = self.k_bulk * np.maximum(0.0, ((densities / self.rho_0) ** 7) - 1.0)

        # 3. Phase 2: Compute SPH Pressure, Viscosity, and IPC Contact Barrier Forces
        f_pressure = np.zeros((N, 3), dtype=np.float32)
        f_viscosity = np.zeros((N, 3), dtype=np.float32)
        f_contact = np.zeros((N, 3), dtype=np.float32)

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = pos[p_src_arr]
            v_src = vel[p_src_arr]
            rho_src = densities[p_src_arr]
            p_src_val = pressures[p_src_arr]
            center_src = np.mean(pts_src, axis=0)

            src_grid = np.floor(center_src * res).astype(np.int64)
            near_p_list = []
            ring = self.sph_ring
            for dx in range(-ring, ring + 1):
                nx = src_grid[0] + dx
                if 0 <= nx < res:
                    for dy in range(-ring, ring + 1):
                        ny = src_grid[1] + dy
                        if 0 <= ny < res:
                            for dz in range(-ring, ring + 1):
                                nz = src_grid[2] + dz
                                if 0 <= nz < res:
                                    nk = int(nx + ny * res + nz * (res ** 2))
                                    if nk in key_to_idx:
                                        near_p_list.extend(bucket_map[nk])

            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = pos[near_arr]
            v_near = vel[near_arr]
            m_near = masses[near_arr]
            rho_near = densities[near_arr]
            p_near_val = pressures[near_arr]

            diff = pts_src[:, None, :] - pts_near[None, :, :] # (M_src, len(near), 3)
            dist = np.linalg.norm(diff, axis=-1)              # (M_src, len(near))
            mask = dist > 1e-6

            dist_safe = np.where(mask, dist, 1.0)
            grad_W_mag = np.where(mask, self._eval_sph_grad_kernel(dist_safe), 0.0)
            dir_unit = np.where(mask[:, :, None], diff / dist_safe[:, :, None], 0.0)
            grad_W = grad_W_mag[:, :, None] * dir_unit # (M_src, len(near), 3)

            # SPH Pressure force: - m_j * (p_i / rho_i^2 + p_j / rho_j^2) * grad_W
            press_factor = (p_src_val[:, None] / (rho_src[:, None]**2)) + (p_near_val[None, :] / (rho_near[None, :]**2))
            f_press_sub = -np.einsum('mn,mnd,n->md', press_factor, grad_W, m_near)
            f_pressure[p_src_arr] = f_press_sub

            # SPH Viscosity force: mu * sum_j m_j (v_j - v_i)/rho_j * (2 * ||grad_W|| / ||r||)
            v_diff = v_near[None, :, :] - v_src[:, None, :] # (M_src, len(near), 3)
            visc_factor = (2.0 * np.abs(grad_W_mag) / dist_safe) * (m_near[None, :] / rho_near[None, :])
            f_visc_sub = self.mu * np.einsum('mn,mnd->md', visc_factor, v_diff)
            f_viscosity[p_src_arr] = f_visc_sub

            # IPC Barrier Contact Force
            ipc_grad_mag = self._eval_ipc_barrier_grad(dist)
            f_ipc_sub = -np.einsum('mn,mnd->md', ipc_grad_mag, dir_unit)
            f_contact[p_src_arr] = f_ipc_sub

        total_forces = f_pressure + f_viscosity + f_contact

        # 4. Neural Latent Feature Update (MLP Layer)
        # Concatenate: [h_i, rho_i, f_press, f_visc, f_contact]
        rho_norm = (densities[:, None] - self.rho_0) / self.rho_0
        mlp_input = np.concatenate([hidden_states, rho_norm, f_pressure, f_viscosity, f_contact], axis=-1)
        hidden_l1 = np.maximum(0.0, mlp_input @ self.W1 + self.b1)
        new_hidden = hidden_states + (hidden_l1 @ self.W2 + self.b2)

        meta = {
            "num_particles": N,
            "mean_density": float(np.mean(densities)),
            "mean_pressure": float(np.mean(pressures)),
            "max_contact_force": float(np.max(np.linalg.norm(f_contact, axis=-1))),
            "active_clusters": len(bucket_map),
        }
        return new_hidden, total_forces, densities, meta


def _test_density_vs_exact():
    """Density-vs-exact test: verify the spatial-hash SPH density matches
    the direct all-pairs sum.  The old ring=1 truncation dropped pairs with
    r in (cell_extent, 2*h]; the corrected ring = ceil(h/cell) must recover
    them.
    """
    rng = np.random.RandomState(42)
    N = 200
    h = 0.15
    pos = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    vel = np.zeros_like(pos)
    masses = np.ones(N, dtype=np.float32) * 0.01
    hidden = np.zeros((N, 32), dtype=np.float32)

    layer = NeuralSPHIPCLayer(hidden_dim=32, smoothing_h=h, grid_depth=4)
    _, _, densities, meta = layer.forward(pos, vel, masses, hidden)

    # Exact all-pairs density: rho_i = sum_j m_j W(|x_i - x_j|, h)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    # Reuse the layer's kernel evaluator (vectorized over the full N×N).
    W_exact = layer._eval_sph_kernel(dist)
    densities_exact = np.sum(W_exact * masses[None, :], axis=-1)

    rel_err = np.linalg.norm(densities - densities_exact) / max(1e-30, np.linalg.norm(densities_exact))
    print(f"  SPH density vs exact (N={N}, h={h}, ring={layer.sph_ring}):")
    print(f"    rel-L2 = {rel_err:.4e}")
    assert rel_err < 1e-6, f"SPH density mismatch: rel_err={rel_err:.4e}"
    print("  -> PASS (no kernel-support truncation)")


if __name__ == "__main__":
    _test_density_vs_exact()
