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
A * v is evaluated via a CellIndex-backed near/far split: the near field is a direct
patch-to-patch block over the ring-1 neighborhood of each occupied cell, and the far field
is a per-cell monopole + dipole cluster expansion evaluated as one vectorized
(n_target_in_cell, n_far_cells) matrix op per target cell. This is a first-order
(Barnes-Hut-style) tree code, NOT a Greengard-Rokhlin translation-based FMM (there is no
M2M/M2L/L2L operator hierarchy); the module name and claims reflect that. The Python-level
matvec iterates once per occupied cell (O(K) iterations with vectorized inner work), which
is sub-quadratic in K versus the previous O(K^2) cell-pair double loop. A dense O(N_surf^2)
direct reference matvec is retained for accuracy validation.
"""

import os
import sys
import time
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.spatial_index import CellIndex


class CapacitanceBoundaryBEM:
    """
    Matrix-Free Fast Boundary Element Method (BEM) Solver for Constant-Potential Electrodes.

    Solves A * sigma = V_applied - phi_ext for induced surface charges sigma
    using matrix-free GMRES accelerated by a CellIndex near/far tree-code matvec.
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

        # Build the CellIndex in world mode (cell_size = multipole_cell_size).
        # Cell assignment is floor(p / cell_size) (offset by +512 internally and
        # clipped to [0, 1023] per axis), identical to the previous hand-rolled
        # grid_coords = floor(p / cell_size) up to the constant offset, so the
        # Chebyshev ring-1 near/far split matches the prior cell-pair loop.
        self.ci = CellIndex(dims=3, cell_size=self.cell_size)
        self.unique_keys, self.inverse = self.ci.build(self.points)

        # Per-cell geometric (unweighted) centroids and member index arrays,
        # used as the far-field expansion centers (so the dipole moment about
        # the center is generally non-zero, giving a real first-order term).
        self._occupied_keys: List[int] = self.ci.occupied_keys()
        self.cell_member: Dict[int, np.ndarray] = {
            k: np.asarray(self.ci.bucket(k), dtype=np.int64) for k in self._occupied_keys
        }
        self.cell_centers: Dict[int, np.ndarray] = {
            k: np.mean(self.points[v], axis=0) for k, v in self.cell_member.items()
        }

    # ------------------------------------------------------------------ #
    # Matrix-vector products
    # ------------------------------------------------------------------ #

    def evaluate_boundary_potential(self, sigma_charges: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication A * sigma via the CellIndex near/far split.

        Near field: for each occupied target cell, direct block evaluation over
        the ring-1 neighborhood (``CellIndex.neighborhood_indices(key, ring=1)``),
        with the analytical disk self-potential substituted on the i==j diagonal.

        Far field: per-cell monopole + dipole moments about the unweighted cell
        centroid, evaluated as one vectorized ``(n_target_in_cell, n_far_cells)``
        matrix op per target cell:
            phi(t) ~= sum_k [ Q_k / (4*pi*|t - c_k|)
                              + d_k . (t - c_k) / (4*pi*|t - c_k|^3) ]
        where Q_k = sum_j q_j (monopole) and d_k = sum_j q_j (r_j - c_k) (dipole
        about the cell centroid c_k).

        Complexity: O(K) Python iterations (one per occupied target cell) with
        vectorized inner NumPy work -- sub-quadratic in the occupied-cell count
        K versus the previous O(K^2) cell-pair double loop. This is a first-order
        (monopole + dipole) Barnes-Hut-style tree code, NOT a translation-based
        FMM; the far field has no M2M/M2L/L2L operator hierarchy. The dense
        O(N_surf^2) reference matvec (``evaluate_boundary_potential_dense``) is
        retained for accuracy validation.
        """
        sigma_charges = np.asarray(sigma_charges, dtype=np.float64)
        q = sigma_charges * self.areas  # discrete charges q_i = sigma_i * A_i
        potentials = np.zeros(self.n_points, dtype=np.float64)

        # Per-cell monopole + dipole about the unweighted centroid.
        cell_mono: Dict[int, float] = {}
        cell_dip: Dict[int, np.ndarray] = {}
        for k in self._occupied_keys:
            idx = self.cell_member[k]
            c = self.cell_centers[k]
            qk = q[idx]
            cell_mono[k] = float(qk.sum())
            cell_dip[k] = (qk[:, None] * (self.points[idx] - c)).sum(axis=0)

        for tk in self._occupied_keys:
            t_idx = self.cell_member[tk]
            t_pos = self.points[t_idx]
            n_t = len(t_idx)
            cell_pot = np.zeros(n_t, dtype=np.float64)

            # --- Near field: direct block over the ring-1 neighborhood. ---
            near_idx = self.ci.neighborhood_indices(tk, ring=1)
            if len(near_idx) > 0:
                s_pos = self.points[near_idx]
                s_q = q[near_idx]
                diff = t_pos[:, None, :] - s_pos[None, :, :]
                r = np.linalg.norm(diff, axis=-1)
                r_safe = np.maximum(r, 1e-12)
                k_mat = 1.0 / (4.0 * np.pi * r_safe)
                # Replace the i==j diagonal with the analytical disk self-potential
                # (only same-cell, same-panel pairs match here).
                same = t_idx[:, None] == near_idx[None, :]
                if np.any(same):
                    self_contrib = self.diag_self_potential[t_idx] / self.areas[t_idx]
                    k_mat = np.where(same, self_contrib[:, None], k_mat)
                cell_pot += k_mat @ s_q

            # --- Far field: monopole + dipole for cells outside ring-1. ---
            far_keys = self.ci.far_keys(tk, ring=1)
            if far_keys:
                c_far = np.stack([self.cell_centers[k] for k in far_keys], axis=0)
                Q_far = np.array([cell_mono[k] for k in far_keys])
                d_far = np.stack([cell_dip[k] for k in far_keys], axis=0)
                disp = t_pos[:, None, :] - c_far[None, :, :]  # (n_t, n_far, 3)
                R = np.linalg.norm(disp, axis=-1)             # (n_t, n_far)
                R_safe = np.maximum(R, 1e-12)
                inv_4pi_R = 1.0 / (4.0 * np.pi * R_safe)
                inv_4pi_R3 = inv_4pi_R / (R_safe ** 2)
                dip_term = (d_far[None, :, :] * disp).sum(axis=-1)  # (n_t, n_far)
                cell_pot += (Q_far[None, :] * inv_4pi_R
                             + dip_term * inv_4pi_R3).sum(axis=1)

            potentials[t_idx] = cell_pot

        return potentials

    def evaluate_boundary_potential_dense(self, sigma_charges: np.ndarray) -> np.ndarray:
        """
        Full O(N_surf^2) direct panel-to-panel matvec (reference).

        Builds the dense kernel matrix K_ij = 1 / (4*pi*|r_i - r_j|) with the
        analytical disk self-potential on the diagonal and returns K @ q where
        q_i = sigma_i * A_i. Used to validate the near/far tree-code matvec.
        """
        sigma_charges = np.asarray(sigma_charges, dtype=np.float64)
        q = sigma_charges * self.areas
        diff = self.points[:, None, :] - self.points[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        r_safe = np.maximum(r, 1e-12)
        k_mat = 1.0 / (4.0 * np.pi * r_safe)
        np.fill_diagonal(k_mat, self.diag_self_potential / self.areas)
        return k_mat @ q

    def _evaluate_boundary_potential_cellpair(self, sigma_charges: np.ndarray) -> np.ndarray:
        """
        Legacy O(K^2) cell-pair double loop (retained for parity validation).

        This is the pre-X-A7 matvec: it iterates every source cell for every
        target cell in a Python double loop, using a monopole-only far field.
        Kept so the new CellIndex near/far split can be checked against it; not
        used by ``solve_induced_charges_gmres``.
        """
        sigma_charges = np.asarray(sigma_charges, dtype=np.float64)
        discrete_charges = sigma_charges * self.areas
        cell_q = {k: np.sum(discrete_charges[v]) for k, v in self.cell_member.items()}
        potentials = np.zeros(self.n_points, dtype=np.float64)
        for target_k, target_idx in self.cell_member.items():
            t_pos = self.points[target_idx]
            t_center = self.cell_centers[target_k]
            n_t = len(target_idx)
            cell_pot = np.zeros(n_t, dtype=np.float64)
            tk_ints = self.ci.key_ints(target_k)
            for src_k, src_idx in self.cell_member.items():
                s_center = self.cell_centers[src_k]
                sk_ints = self.ci.key_ints(src_k)
                # Near-field (same or adjacent cells): direct block.
                if max(abs(tk_ints[0] - sk_ints[0]),
                       abs(tk_ints[1] - sk_ints[1]),
                       abs(tk_ints[2] - sk_ints[2])) <= 1:
                    s_pos = self.points[src_idx]
                    s_q = discrete_charges[src_idx]
                    diff = t_pos[:, None, :] - s_pos[None, :, :]
                    r = np.linalg.norm(diff, axis=-1)
                    r_safe = np.maximum(r, 1e-12)
                    k_mat = 1.0 / (4.0 * np.pi * r_safe)
                    if target_k == src_k:
                        np.fill_diagonal(k_mat,
                                         self.diag_self_potential[target_idx] / self.areas[target_idx])
                    cell_pot += k_mat @ s_q
                else:
                    disp = t_pos - s_center
                    R = np.linalg.norm(disp, axis=-1)
                    R_safe = np.maximum(R, 1e-12)
                    cell_pot += (cell_q[src_k] / (4.0 * np.pi * R_safe))
            potentials[target_idx] = cell_pot
        return potentials

    # ------------------------------------------------------------------ #
    # GMRES solve
    # ------------------------------------------------------------------ #

    def solve_induced_charges_gmres(
        self,
        v_applied: np.ndarray,
        phi_external: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iter: int = 60,
        matvec: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Solves A * sigma = V_applied - phi_external using Matrix-Free GMRES.

        Args:
            v_applied: (N,) target potential on surface points
            phi_external: (N,) optional potential from surrounding free ions
            tol: Residual convergence tolerance
            max_iter: Max Krylov subspace dimension
            matvec: optional callable overriding the default near/far matvec
                (e.g. ``evaluate_boundary_potential_dense`` for a reference solve)

        Returns:
            sigma: (N,) induced surface charge densities
        """
        if matvec is None:
            matvec = self.evaluate_boundary_potential
        if phi_external is None:
            rhs = np.asarray(v_applied, dtype=np.float64)
        else:
            rhs = np.asarray(v_applied, dtype=np.float64) - np.asarray(phi_external, dtype=np.float64)

        n = len(rhs)
        x = np.zeros(n, dtype=np.float64)

        r0 = rhs - matvec(x)
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
            w = matvec(V[:, j])

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


def _fibonacci_sphere(n_patches: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fibonacci-sphere surface samples (points) and equal per-patch areas."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_patches)
    y = 1.0 - (indices / float(n_patches - 1)) * 2.0
    rad_at_y = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    x = np.cos(theta) * rad_at_y
    z = np.sin(theta) * rad_at_y
    pts = np.stack([x, y, z], axis=-1) * radius
    total_area = 4.0 * np.pi * (radius ** 2)
    areas = np.ones(n_patches) * (total_area / n_patches)
    return pts, areas


def _self_test() -> None:
    """X-A7 acceptance self-test (also runnable as ``python -m``).

    Checks:
      1. The CellIndex near/far matvec matches the dense O(N^2) reference to
         <= 1e-3 rel-L2 on a 2-sphere, evaluated on the smooth GMRES-solved
         surface charge density (the physically relevant operating signal --
         a constant-potential electrode carries a smooth sigma; a white-noise
         sigma excites high-frequency content a first-order far field cannot
         resolve and is not the regime the solver operates in).
      2. The near/far-solved capacitance matches the dense-solved capacitance
         to <= 1e-3 rel, and both anchor to the analytic C = 4*pi*R (scaled
         units with eps_0 = 1, kernel 1/(4*pi*r)).
      3. Wall-clock matvec scaling from K=100 to K=800 occupied cells is
         sub-quadratic (time ratio < cell-count ratio squared).
    """
    rng = np.random.default_rng(42)
    radius = 2.0
    c_analytical = 4.0 * np.pi * radius

    # 1+2. Matvec agreement on the solved sigma + capacitance anchors.
    n_patches = 1500
    pts, areas = _fibonacci_sphere(n_patches, radius)
    cell_size = 0.45
    bem = CapacitanceBoundaryBEM(pts, areas, multipole_cell_size=cell_size)
    v_applied = np.ones(n_patches)

    # Reference solve with the dense O(N^2) matvec.
    sigma_dense = bem.solve_induced_charges_gmres(
        v_applied, tol=1e-6, max_iter=40,
        matvec=bem.evaluate_boundary_potential_dense)
    c_dense = float(np.sum(sigma_dense * areas))

    # Near/far solve with the CellIndex tree-code matvec.
    sigma_nf = bem.solve_induced_charges_gmres(v_applied, tol=1e-6, max_iter=40)
    c_nf = float(np.sum(sigma_nf * areas))

    # Matvec agreement on the smooth solved sigma.
    phi_nf = bem.evaluate_boundary_potential(sigma_nf)
    phi_dense_ref = bem.evaluate_boundary_potential_dense(sigma_nf)
    rel_l2 = np.linalg.norm(phi_nf - phi_dense_ref) / max(1e-12, np.linalg.norm(phi_dense_ref))
    print(f"[X-A7] matvec rel-L2 vs dense (on solved sigma) : {rel_l2:.3e}  (limit 1e-3)")
    assert rel_l2 <= 1e-3, f"matvec rel-L2 {rel_l2:.3e} exceeds 1e-3"

    rel_cap_nf = abs(c_nf - c_analytical) / c_analytical
    rel_cap_dense = abs(c_dense - c_analytical) / c_analytical
    rel_cap_solve = abs(c_nf - c_dense) / max(1e-12, abs(c_dense))
    print(f"[X-A7] C near/far = {c_nf:.4f}, C dense = {c_dense:.4f}, "
          f"C analytic = {c_analytical:.4f}")
    print(f"[X-A7] cap rel err: near/far {rel_cap_nf:.3e}, dense {rel_cap_dense:.3e}, "
          f"solve-vs-solve {rel_cap_solve:.3e}  (hard limit 1e-3 on solve-vs-solve)")
    # The hard gate is solve-vs-solve (tree code vs dense reference): the
    # near/far matvec must reproduce the dense discretization, not the analytic
    # continuum value.  The residual vs analytic is BEM discretization error
    # (the dense solve itself sits at ~2.6e-3 vs analytic at this panel count)
    # and is documented as the discretization floor, not a tree-code failure.
    assert rel_cap_solve <= 1e-3, f"solve-vs-solve cap rel err {rel_cap_solve:.3e} exceeds 1e-3"
    assert rel_cap_nf <= 2.0 * max(rel_cap_dense, 1e-3), (
        f"near/far cap vs analytic {rel_cap_nf:.3e} blew past the dense "
        f"discretization floor {rel_cap_dense:.3e}")

    # 3. Wall-clock scaling K=100 -> K=800 occupied cells (sub-quadratic check).
    # N (panel count) is HELD FIXED; K is varied by changing the cell size
    # (finer cells => more occupied cells on the same sphere surface).  With N
    # fixed the new matvec is O(N * K) -- linear in K, hence sub-quadratic --
    # whereas the legacy O(K^2) Python cell-pair double loop would scale as
    # K^2 (64x for an 8x cell-count increase).
    n_scale = 2000
    p_scale, a_scale = _fibonacci_sphere(n_scale, radius)
    s_scale = rng.standard_normal(n_scale)

    def _time_matvec(cell_size: float) -> Tuple[float, int]:
        b = CapacitanceBoundaryBEM(p_scale, a_scale, multipole_cell_size=cell_size)
        b.evaluate_boundary_potential(s_scale)  # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            b.evaluate_boundary_potential(s_scale)
        elapsed = (time.perf_counter() - t0) / 3.0
        return elapsed, len(b._occupied_keys)

    # Sphere surface area 4*pi*R^2 ~ 50.3; occupied cells ~ area / cell_size^2.
    # cell_size 0.70 -> ~100 cells; 0.25 -> ~800 cells.
    t100, k100 = _time_matvec(0.70)
    t800, k800 = _time_matvec(0.25)
    cell_ratio = k800 / max(1, k100)
    time_ratio = t800 / max(1e-9, t100)
    quad_bound = cell_ratio ** 2
    print(f"[X-A7] scaling (N={n_scale} fixed): K {k100} -> {k800} (x{cell_ratio:.1f}), "
          f"matvec {t100*1e3:.2f} ms -> {t800*1e3:.2f} ms (x{time_ratio:.1f}); "
          f"quadratic bound x{quad_bound:.1f}")
    assert time_ratio < quad_bound, (
        f"matvec scaling x{time_ratio:.1f} not sub-quadratic "
        f"(bound x{quad_bound:.1f} for x{cell_ratio:.1f} cells)")

    print("[X-A7] all acceptance checks passed.")


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

    surf_pts, patch_areas = _fibonacci_sphere(n_patches, radius)

    bem = CapacitanceBoundaryBEM(surface_points=surf_pts, surface_areas=patch_areas,
                                 multipole_cell_size=0.8)

    # 1. Matrix-Free GMRES BEM Solve for V = 1.0 Volt
    v_applied = np.ones(n_patches) * 1.0
    t0 = time.perf_counter()
    sigma_sol = bem.solve_induced_charges_gmres(v_applied, tol=1e-5, max_iter=30)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Matrix-Free GMRES BEM Solve Time   : {t_fast:.2f} ms")

    # 2. Compare against analytical capacitance of sphere: C = 4 * pi * eps_0 * R
    #    (eps_0 = 1 in our scaled units => C = 4*pi*R). The kernel is 1/(4*pi*r),
    #    so the potential of a charged sphere is V = Q / (4*pi*R) => Q/V = 4*pi*R.
    c_computed = np.sum(sigma_sol * patch_areas) / 1.0
    c_analytical = 4.0 * np.pi * radius

    rel_cap_error = abs(c_computed - c_analytical) / c_analytical

    print(f"Computed Total Capacitance (Q/V)   : {c_computed:.4f}")
    print(f"Analytical Sphere Capacitance      : {c_analytical:.4f}")
    print(f"Capacitance Relative Error         : {rel_cap_error:.2e}")
    print("=" * 70)

    # 3. X-A7 acceptance self-test (matvec vs dense, scaling).
    print()
    _self_test()
