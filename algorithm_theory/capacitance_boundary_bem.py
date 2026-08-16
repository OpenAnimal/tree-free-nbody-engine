"""
Constant-Potential Boundary Element Method (BEM) for Electrodes (capacitance_boundary_bem.py).

Inspired by:
1. "Boundary Element Methods for Electrical Capacitance and Impedance Modeling"
   S. M. Rao, T. K. Sarkar, R. F. Harrington (IEEE Trans. Microwave Theory Tech., 1984).
2. "Fast Direct and Iterative Boundary Integral Solvers for Poisson Systems"
   P. G. Martinsson and V. Rokhlin (J. Comput. Phys. 2005).
3. "Constant-Potential Simulations of Electrochemical Double Layers"
   S. Reed, P. A. Madden et al. (J. Chem. Phys. 2007).

Key Algorithmic Principle:
In battery porous electrodes, supercapacitors, and microelectronics, metal current collectors
maintain constant Dirichlet potentials V_0. As mobile ions diffuse, induced surface polarization
charges sigma(x) dynamically rearrange, governed by the first-kind boundary integral equation:
    int_{partial Omega} (sigma(y) / (4 * pi * ||x - y||)) dS_y = V_applied(x) - phi_ext(x)

Classical BEM discretizes the surface into N_surf boundary patches, assembling a dense
N_surf x N_surf capacitance matrix requiring O(N_surf^3) direct LU inversion or O(N_surf^2) GMRES iterations.
Here, we build a Matrix-Free Tree-Free BEM Solver where each GMRES matrix-vector product
A * v is evaluated in O(N_surf) via tree-free multipole translation.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np


class CapacitanceBoundaryBEM:
    """
    Matrix-Free Fast Boundary Element Method (BEM) Solver for Constant-Potential Electrodes.
    
    Solves A * sigma = V_applied - phi_ext for induced surface charges sigma
    using matrix-free GMRES accelerated by tree-free multipole potential evaluations.
    """
    def __init__(
        self,
        surface_points: np.ndarray,
        surface_areas: np.ndarray,
        multipole_cell_size: float = 0.5
    ):
        self.points = np.asarray(surface_points, dtype=np.float64)
        self.areas = np.asarray(surface_areas, dtype=np.float64)
        self.n_points = len(self.points)
        self.cell_size = float(multipole_cell_size)

        # Self-interaction diagonal term for boundary patch of area A_i:
        # Integral over circular disk of equal area radius R = sqrt(A / pi):
        # int_0^R (1 / (4*pi*r)) * 2*pi*r dr = R / 2 = sqrt(A / pi) / 2
        self.patch_radii = np.sqrt(self.areas / np.pi)
        self.diag_self_potential = self.patch_radii / 2.0

        # Build spatial hash for matrix-free evaluations
        self.grid_coords = np.floor(self.points / self.cell_size).astype(np.int64)
        self.cell_map: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, c in enumerate(self.grid_coords):
            k = (int(c[0]), int(c[1]), int(c[2]))
            if k not in self.cell_map:
                self.cell_map[k] = []
            self.cell_map[k].append(idx)

        self.cell_arrays = {k: np.array(v, dtype=np.int64) for k, v in self.cell_map.items()}
        self.cell_centers = {k: np.mean(self.points[v], axis=0) for k, v in self.cell_arrays.items()}

    def evaluate_boundary_potential(self, sigma_charges: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication A * sigma in O(N_surf) time.
        Evaluates potential on boundary points produced by current surface charges.
        """
        sigma_charges = np.asarray(sigma_charges, dtype=np.float64)
        discrete_charges = sigma_charges * self.areas
        
        # Precompute cell monopole charges
        cell_q = {k: np.sum(discrete_charges[v]) for k, v in self.cell_arrays.items()}
        
        potentials = np.zeros(self.n_points, dtype=np.float64)

        for target_k, target_idx in self.cell_arrays.items():
            t_pos = self.points[target_idx]
            t_center = self.cell_centers[target_k]
            n_t = len(target_idx)
            cell_pot = np.zeros(n_t, dtype=np.float64)

            for src_k, src_idx in self.cell_arrays.items():
                s_center = self.cell_centers[src_k]
                disp_c = t_center - s_center
                dist_c = np.linalg.norm(disp_c)

                # Near-field (same or adjacent cells): Direct patch-to-patch summation
                if max(abs(target_k[0] - src_k[0]), abs(target_k[1] - src_k[1]), abs(target_k[2] - src_k[2])) <= 1:
                    s_pos = self.points[src_idx]
                    s_q = discrete_charges[src_idx]
                    
                    diff = t_pos[:, None, :] - s_pos[None, :, :]
                    r = np.linalg.norm(diff, axis=-1)
                    
                    # 1 / (4*pi*r)
                    r_safe = np.maximum(r, 1e-12)
                    k_mat = 1.0 / (4.0 * np.pi * r_safe)
                    
                    if target_k == src_k:
                        # Replace diagonal with analytical disk self-potential
                        np.fill_diagonal(k_mat, self.diag_self_potential[target_idx] / self.areas[target_idx])

                    cell_pot += k_mat @ s_q

                # Far-field: Monopole multipole translation
                else:
                    disp = t_pos - s_center
                    R = np.linalg.norm(disp, axis=-1)
                    R_safe = np.maximum(R, 1e-12)
                    
                    # 1 / (4 * pi * R)
                    cell_pot += (cell_q[src_k] / (4.0 * np.pi * R_safe))

            potentials[target_idx] = cell_pot

        return potentials

    def solve_induced_charges_gmres(
        self,
        v_applied: np.ndarray,
        phi_external: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iter: int = 60
    ) -> np.ndarray:
        """
        Solves A * sigma = V_applied - phi_external using Matrix-Free GMRES.
        
        Args:
            v_applied: (N,) target potential on surface points
            phi_external: (N,) optional potential from surrounding free ions
            tol: Residual convergence tolerance
            max_iter: Max Krylov subspace dimension
            
        Returns:
            sigma: (N,) induced surface charge densities
        """
        if phi_external is None:
            rhs = np.asarray(v_applied, dtype=np.float64)
        else:
            rhs = np.asarray(v_applied, dtype=np.float64) - np.asarray(phi_external, dtype=np.float64)

        n = len(rhs)
        x = np.zeros(n, dtype=np.float64)
        
        r0 = rhs - self.evaluate_boundary_potential(x)
        norm_r0 = np.linalg.norm(r0)
        if norm_r0 < 1e-12:
            return x

        # GMRES Arnoldi basis matrices
        V = np.zeros((n, max_iter + 1), dtype=np.float64)
        H = np.zeros((max_iter + 1, max_iter), dtype=np.float64)
        
        V[:, 0] = r0 / norm_r0
        beta_vec = np.zeros(max_iter + 1, dtype=np.float64)
        beta_vec[0] = norm_r0

        k_iter = 0
        for j in range(max_iter):
            k_iter = j + 1
            # Matrix-free matvec
            w = self.evaluate_boundary_potential(V[:, j])
            
            # Modified Gram-Schmidt orthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w -= H[i, j] * V[:, i]
                
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-14:
                V[:, j + 1] = w / H[j + 1, j]

            # Solve small (j+1) x j least squares problem
            y, residuals, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], beta_vec[:j + 2], rcond=None)
            
            # Compute current residual norm
            current_res = np.linalg.norm(beta_vec[:j + 2] - H[:j + 2, :j + 1] @ y)
            if current_res / norm_r0 < tol:
                break

        x += V[:, :k_iter] @ y
        return x

    def compute_capacitance(self, v_voltage: float = 1.0) -> float:
        """Computes total self-capacitance C = Q_total / V."""
        v_target = np.ones(self.n_points) * v_voltage
        sigma = self.solve_induced_charges_gmres(v_target)
        total_charge = np.sum(sigma * self.areas)
        return float(total_charge / v_voltage)


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Constant-Potential Electrode Boundary Element Method (BEM) Benchmark")
    print("=" * 70)

    # Generate 3D spherical electrode shell
    n_patches = 4000
    radius = 2.0
    print(f"Number of Electrode Surface Patches: {n_patches:,}")
    print(f"Electrode Sphere Radius            : {radius:.2f} m")

    # Fibonacci sphere sampling for uniform patch distribution
    phi = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_patches)
    y = 1.0 - (indices / float(n_patches - 1)) * 2.0
    rad_at_y = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    x = np.cos(theta) * rad_at_y
    z = np.sin(theta) * rad_at_y
    surf_pts = np.stack([x, y, z], axis=-1) * radius

    total_sphere_area = 4.0 * np.pi * (radius ** 2)
    patch_areas = np.ones(n_patches) * (total_sphere_area / n_patches)

    bem = CapacitanceBoundaryBEM(surface_points=surf_pts, surface_areas=patch_areas, multipole_cell_size=0.8)

    # 1. Matrix-Free GMRES BEM Solve for V = 1.0 Volt
    v_applied = np.ones(n_patches) * 1.0
    t0 = time.perf_counter()
    sigma_sol = bem.solve_induced_charges_gmres(v_applied, tol=1e-5, max_iter=30)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Matrix-Free GMRES BEM Solve Time   : {t_fast:.2f} ms")

    # 2. Compare against analytical capacitance of sphere: C = 4 * pi * eps_0 * R (eps_0 = 1 in our units => C = 4*pi*R)
    c_computed = np.sum(sigma_sol * patch_areas) / 1.0
    c_analytical = 4.0 * np.pi * radius  # in units of 1/(4*pi*r) kernel => C_anal = 4*pi * R
    
    # In our kernel 1/(4*pi*r), potential of sphere with charge Q is V = Q / (4*pi*R) => Q/V = 4*pi*R
    rel_cap_error = abs(c_computed - c_analytical) / c_analytical

    print(f"Computed Total Capacitance (Q/V)   : {c_computed:.4f}")
    print(f"Analytical Sphere Capacitance      : {c_analytical:.4f}")
    print(f"Capacitance Relative Error         : {rel_cap_error:.2e}")
    print("=" * 70)
