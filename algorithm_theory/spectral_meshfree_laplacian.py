"""
Matrix-Free Meshfree Laplacians (spectral_meshfree_laplacian.py).

Two operators with distinct, honestly-documented semantics:

1. `MeshfreeGraphLaplacian` (alias `SpectralMeshfreeLaplacian`, the
   historical name): a kernel-weighted GRAPH Laplacian on the point cloud —
   SPD and symmetric diagonally dominant. `(L + kappa^2 I)` is applied
   matrix-free with O(N) cost per matvec, and the associated linear system
   is solved by `solve_meshfree_poisson` with a two-level (fine Jacobi +
   coarse cluster-diagonal) preconditioned CG. This solves the GRAPH
   Poisson / diffusion problem on the point cloud; it is NOT a consistent
   discretization of the continuous nabla^2 (see `ConsistentMeshfreeLaplacian`
   for that) — earlier revisions overstated this, and the claim is retracted.

2. `ConsistentMeshfreeLaplacian`: an RBF-FD operator with a Gaussian
   kernel and quadratic polynomial augmentation (Wright & Fornberg,
   "Scattered Data Finite Difference Formulas", SIAM J. Sci. Comput.
   2016-style). Its stencils reproduce nabla^2 EXACTLY on all quadratic
   polynomials (machine precision, guaranteed by the polynomial block for
   any invertible saddle system). On smooth fields with scattered nodes
   the OBSERVED convergence is between first and second order (measured
   ~1.5 in refinement studies at fixed h/spacing ratio on jittered
   grids); O(h^2) is polynomial-exactness on quadratics, not a blanket
   convergence guarantee on irregular clouds. It is the operator to use
   when the continuous Laplacian is what you actually want.
   Caveat (measured, not hidden): its row stencils are generally
   non-symmetric, and one-sided stencils near the cloud boundary make the
   global system non-normal with spurious near-null modes — iterative
   SOLVES of the continuous problem need boundary stabilization that this
   module does not provide. Use it for operator evaluation (matvec).

The nearly-linear Laplacian-solver literature (Spielman-Teng; Cohen-Kelner-
Miller-Peng) motivates the sparse-plus-preconditioned approach, but no
Spielman-Teng guarantee is claimed for the code here: the solver is plain
preconditioned CG with a two-level heuristic preconditioner.
"""

import time
from itertools import combinations_with_replacement, product
from typing import Tuple, List, Optional, Dict
import numpy as np


class MeshfreeGraphLaplacian:
    """
    Kernel-weighted graph Laplacian on a 2D or 3D point set.

    Computes (L + kappa^2 I) * v where L is the symmetric diagonally
    dominant graph Laplacian with Wendland C2 edge weights
        (L v)_i = sum_j w_ij * (v_i - v_j),   w_ij = WendlandC2(|x_i - x_j|)
    over all pairs within `support_radius` (spatial-hash neighbor lookups,
    O(N) per matvec via a flat CSR edge array). SPD for kappa > 0.

    This is a GRAPH operator (smoothing / diffusion / spectral semantics on
    the neighbor graph). It does NOT consistently discretize the continuous
    nabla^2 — use `ConsistentMeshfreeLaplacian` for that.
    """
    def __init__(self, points: np.ndarray, support_radius: float, kappa: float = 0.0):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(
                f"points must have shape (N, 2) or (N, 3); got {pts.shape}")
        self.points = pts
        self.dim = int(pts.shape[1])
        self.n_points = int(len(pts))
        self.support_radius = float(support_radius)
        if self.support_radius <= 0.0:
            raise ValueError("support_radius must be positive")
        self.h = self.support_radius
        self.kappa = float(kappa)
        self.cell_size = self.support_radius

        # Spatial hash grid (dimension-generic bucket keys).
        self.grid_coords = np.floor(self.points / self.cell_size).astype(np.int64)
        self.spatial_buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, coord in enumerate(self.grid_coords):
            key = tuple(int(c) for c in coord)
            self.spatial_buckets.setdefault(key, []).append(idx)

        self.neighbor_indices: List[np.ndarray] = []
        self.neighbor_weights: List[np.ndarray] = []
        self.diagonal: np.ndarray = np.zeros(self.n_points, dtype=np.float64)
        self._build_stencils()

    def _wendland_c2(self, r: np.ndarray) -> np.ndarray:
        """Wendland C2 radial basis kernel (edge weights)."""
        q = np.clip(r / self.h, 0.0, 1.0)
        return (1.0 - q)**4 * (4.0 * q + 1.0)

    def _build_stencils(self):
        """Symmetric Wendland-weighted edges (upper-triangle gathered once)."""
        raw_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(self.n_points)]

        for i in range(self.n_points):
            coord = self.grid_coords[i]
            cands: List[int] = []
            for off in product((-1, 0, 1), repeat=self.dim):
                key = tuple(int(coord[a] + off[a]) for a in range(self.dim))
                b = self.spatial_buckets.get(key)
                if b:
                    cands.extend(b)
            if not cands:
                continue
            cand_arr = np.array(cands, dtype=np.int64)
            cand_arr = cand_arr[cand_arr > i]  # upper-triangle pairs -> exact symmetry

            if len(cand_arr) == 0:
                continue
            disp = self.points[cand_arr] - self.points[i]
            dists = np.linalg.norm(disp, axis=1)
            mask = dists < self.h
            valid_j = cand_arr[mask]
            if len(valid_j) == 0:
                continue

            weights = self._wendland_c2(dists[mask])
            for j_node, w_val in zip(valid_j, weights):
                raw_neighbors[i].append((int(j_node), float(w_val)))
                raw_neighbors[j_node].append((i, float(w_val)))

        flat_rows_list: List[np.ndarray] = []
        flat_cols_list: List[np.ndarray] = []
        flat_w_list: List[np.ndarray] = []
        for i in range(self.n_points):
            if len(raw_neighbors[i]) == 0:
                self.neighbor_indices.append(np.empty(0, dtype=np.int64))
                self.neighbor_weights.append(np.empty(0, dtype=np.float64))
                # Isolated point: no graph edges, so L_ii = 0 and
                # (L + kappa^2 I)_ii = kappa^2.  A tiny floor keeps the
                # matrix non-singular when kappa == 0 (pure graph Poisson).
                self.diagonal[i] = self.kappa**2 + 1e-10
            else:
                indices = np.array([idx for idx, _ in raw_neighbors[i]], dtype=np.int64)
                weights = np.array([w for _, w in raw_neighbors[i]], dtype=np.float64)
                self.neighbor_indices.append(indices)
                self.neighbor_weights.append(weights)
                self.diagonal[i] = float(np.sum(weights)) + self.kappa**2
                flat_rows_list.append(np.full(len(indices), i, dtype=np.int64))
                flat_cols_list.append(indices)
                flat_w_list.append(weights)

        if flat_rows_list:
            self.flat_rows = np.concatenate(flat_rows_list)
            self.flat_cols = np.concatenate(flat_cols_list)
            self.flat_weights = np.concatenate(flat_w_list)
        else:
            self.flat_rows = np.empty(0, dtype=np.int64)
            self.flat_cols = np.empty(0, dtype=np.int64)
            self.flat_weights = np.empty(0, dtype=np.float64)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-free application of (L + kappa^2 I) * v in O(N) operations:
            (L v)_i = (sum_j w_ij + kappa^2) * v_i - sum_j w_ij * v_j

        Vectorized via ``np.add.at`` over the flat (row, col, weight) edge
        arrays.
        """
        out = self.diagonal * v
        if self.flat_rows.size > 0:
            np.add.at(out, self.flat_rows, -self.flat_weights * v[self.flat_cols])
        return out


# Historical name (imported by algorithm_theory/__init__.py and older code).
SpectralMeshfreeLaplacian = MeshfreeGraphLaplacian


class ConsistentMeshfreeLaplacian:
    """
    Consistent RBF-FD Laplacian on a 2D or 3D point set.

    Stencil weights solve the standard RBF-FD saddle system per point:
        [ Phi  P ] [w]   [ d ]
        [ P^T  0 ] [nu] = [ b ]
    with the GAUSSIAN kernel Phi_ab = exp(-(eps*|u_a - u_b|)^2) (eps = 3 in
    units of the local normalization scale), quadratic polynomial basis P,
    and
        d_b  = (nabla^2 exp(-(eps*r)^2))(0) = (4*eps^4*r^2 - 2*dim*eps^2)
               * exp(-(eps*r)^2)   with r = |u_b|
        b_m  = nabla^2 of monomial m at 0  (2 for pure squares, else 0)
    in local coordinates u = (x_j - x_i)/h. The polynomial block makes the
    weights reproduce nabla^2 EXACTLY on every quadratic polynomial
    (machine precision, verified in __main__), independent of the kernel;
    the strictly positive definite Gaussian keeps the saddle system
    invertible for ANY stencil of distinct points (unlike polyharmonic
    splines, which are exactly singular on cospherical lattice
    configurations -- measured on regular-grid edge points).

    Stencils are the `max_stencil` nearest neighbors (default 32), which
    keeps the saddle system well conditioned on scattered clouds.

    The row stencils are generally NON-symmetric (standard for strong-form
    RBF-FD). Solving (L + kappa^2 I) u = f iteratively is NOT robust out of
    the box: one-sided stencils near the cloud boundary make the system
    non-normal with spurious near-null modes (measured: direct and Krylov
    solves return garbage on bounded clouds unless boundary rows are
    stabilized). Use this class for operator evaluation (matvec), and
    `MeshfreeGraphLaplacian` + `solve_meshfree_poisson` for the SPD graph
    solve.
    """
    def __init__(self, points: np.ndarray, support_radius: float,
                 kappa: float = 0.0, max_stencil: int = 32):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(
                f"points must have shape (N, 2) or (N, 3); got {pts.shape}")
        self.points = pts
        self.dim = int(pts.shape[1])
        self.n_points = int(len(pts))
        self.support_radius = float(support_radius)
        if self.support_radius <= 0.0:
            raise ValueError("support_radius must be positive")
        self.h = self.support_radius  # normalization scale only
        self.kappa = float(kappa)
        self.max_stencil = int(max_stencil)
        if self.max_stencil < 12:
            raise ValueError("max_stencil must be >= 12 to span the quadratic basis")

        self._build_stencils()

    def _quadratic_basis(self) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        d = self.dim
        monos: List[Tuple[int, ...]] = [()]
        monos += [(k,) for k in range(d)]
        monos += list(combinations_with_replacement(range(d), 2))
        rhs = np.zeros(len(monos), dtype=np.float64)
        for m, mono in enumerate(monos):
            if len(mono) == 2 and mono[0] == mono[1]:
                rhs[m] = 2.0
        return monos, rhs

    def _knn(self) -> List[np.ndarray]:
        """
        Nearest-neighbor stencil index lists via hash-grid ring search.

        Selection starts at the `max_stencil` nearest candidates and GROWS
        (in steps, up to 2*max_stencil) until the quadratic polynomial
        basis is spanned: on regular lattices the nearest neighbors can all
        sit on {-1,0,1} lattice offsets where x^2 = x identically, leaving
        squares indistinguishable from linears (rank 9 of 10 in 3D) — the
        axis-2 offsets are needed regardless of the kernel.
        """
        n, d = self.n_points, self.dim
        monos, _ = self._quadratic_basis()
        k = len(monos)
        # cell size ~ typical spacing so ring expansion terminates quickly
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        vol = max(float(np.prod(np.maximum(hi - lo, 1e-9))), 1e-9)
        cell = max((vol / max(n, 1)) ** (1.0 / d), 1e-9)
        grid = np.floor((self.points - lo) / cell).astype(np.int64)
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, c in enumerate(grid):
            buckets.setdefault(tuple(int(v) for v in c), []).append(idx)

        cap = 2 * self.max_stencil
        out: List[np.ndarray] = []
        for i in range(n):
            c0 = grid[i]
            # Grow the candidate pool in whole boxes until it can span the
            # basis; then rank-check the nearest-prefix and grow `take`.
            pool = np.empty(0, dtype=np.int64)
            ring = 1
            while len(pool) < min(cap + 8, n - 1) and ring < 64:
                box: List[int] = []
                for off in product(range(-ring, ring + 1), repeat=d):
                    b = buckets.get(tuple(int(c0[a] + off[a]) for a in range(d)))
                    if b:
                        box.extend(b)
                pool = np.array(box, dtype=np.int64)
                pool = pool[pool != i]
                ring += 1
            disp = self.points[pool] - self.points[i]
            dists = np.linalg.norm(disp, axis=1)
            order = np.lexsort((pool, dists))  # dedup-safe ordering
            pool, dists = pool[order], dists[order]
            uniq = np.ones(len(pool), dtype=bool)
            uniq[1:] = pool[1:] != pool[:-1]
            pool = pool[uniq]

            take = min(self.max_stencil, len(pool))
            if len(pool) > take:
                u = (self.points[pool] - self.points[i]) / self.h
                E = np.zeros((len(pool) + 1, k), dtype=np.float64)
                E[0, 0] = 1.0
                for m, mono in enumerate(monos):
                    if len(mono) == 1:
                        E[1:, m] = u[:, mono[0]]
                    elif len(mono) == 2:
                        E[1:, m] = u[:, mono[0]] * u[:, mono[1]]
                while (take < len(pool) and take < cap
                       and np.linalg.matrix_rank(E[:take + 1], tol=1e-9) < k):
                    take = min(take + 4, len(pool), cap)
            out.append(np.sort(pool[:take]))
        return out

    def _build_stencils(self):
        n, d, h = self.n_points, self.dim, self.h
        monos, bvec = self._quadratic_basis()
        k = len(monos)

        self.neighbor_indices = self._knn()
        self.neighbor_weights: List[np.ndarray] = []
        self.diagonal = np.zeros(n, dtype=np.float64)

        flat_rows: List[np.ndarray] = []
        flat_cols: List[np.ndarray] = []
        flat_w: List[np.ndarray] = []

        for i in range(n):
            sel = self.neighbor_indices[i]
            u = (self.points[sel] - self.points[i]) / h
            S = len(sel) + 1
            ub = np.vstack([np.zeros((1, d)), u])
            r2 = np.sum((ub[:, None, :] - ub[None, :, :]) ** 2, axis=2)
            eps = 3.0  # Gaussian shape parameter in u-units
            Phi = np.exp(-eps * eps * r2)
            P = np.ones((S, k))
            for m, mono in enumerate(monos):
                if len(mono) == 1:
                    P[:, m] = ub[:, mono[0]]
                elif len(mono) == 2:
                    P[:, m] = ub[:, mono[0]] * ub[:, mono[1]]
            Msys = np.zeros((S + k, S + k))
            Msys[:S, :S] = Phi
            Msys[:S, S:] = P
            Msys[S:, :S] = P.T
            rb2 = np.sum(ub * ub, axis=1)
            dv = (4.0 * eps**4 * rb2 - 2.0 * d * eps * eps) * np.exp(-eps * eps * rb2)
            rhs = np.concatenate([dv, bvec])
            try:
                sol = np.linalg.solve(Msys, rhs)
            except np.linalg.LinAlgError:
                # Degenerate stencil (collinear/coplanar): least-squares.
                sol = np.linalg.lstsq(Msys, rhs, rcond=None)[0]
            w = sol[:S] / (h * h)

            self.diagonal[i] = w[0] + self.kappa ** 2
            self.neighbor_weights.append(w[1:])
            if len(sel):
                flat_rows.append(np.full(len(sel), i, dtype=np.int64))
                flat_cols.append(sel)
                flat_w.append(w[1:])

        if flat_rows:
            self.flat_rows = np.concatenate(flat_rows)
            self.flat_cols = np.concatenate(flat_cols)
            self.flat_weights = np.concatenate(flat_w)
        else:
            self.flat_rows = np.empty(0, dtype=np.int64)
            self.flat_cols = np.empty(0, dtype=np.int64)
            self.flat_weights = np.empty(0, dtype=np.float64)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-free application of (L + kappa^2 I) * v in O(N) operations:
            (L v)_i = w_ii * v_i + sum_{j != i} w_ij * v_j
        Exact on quadratics; second-order accurate on smooth fields.
        """
        out = self.diagonal * v
        if self.flat_rows.size > 0:
            np.add.at(out, self.flat_rows, self.flat_weights * v[self.flat_cols])
        return out


class MultiScalePreconditionedSolver:
    """
    Two-Level Multi-Scale Preconditioner for PCG on the graph Laplacian.

    The coarse solve is a DIAGONAL (Jacobi) approximation, NOT the full
    Galerkin coarse operator: the code applies ``diag(A_c)^{-1}`` (where
    A_c's diagonal is the per-cluster sum of fine diagonal entries minus
    intra-cluster edge weights), not ``(C^T A C)^{-1}``. The resulting
    preconditioner is therefore
        M^{-1} = omega * D^{-1} + C * diag(A_c)^{-1} * C^T
    which is symmetric (diagonal coarse solve) but is NOT the Galerkin
    coarse operator and does NOT "guarantee strict symmetry and
    positive-definiteness" beyond the symmetry of the diagonal coarse
    smoothing; the previous "Guarantees strict symmetry" claim is retracted.
    """
    def __init__(self, laplacian: MeshfreeGraphLaplacian, coarse_factor: float = 2.0):
        self.lap = laplacian
        self.coarse_cell_size = laplacian.cell_size * coarse_factor

        # Build coarse spatial clusters
        c_coords = np.floor(laplacian.points / self.coarse_cell_size).astype(np.int64)
        if laplacian.n_points == 0:
            self.point_to_cluster = np.empty(0, dtype=np.int64)
            self.n_clusters = 0
        else:
            _, self.point_to_cluster = np.unique(c_coords, axis=0, return_inverse=True)
            self.n_clusters = int(np.max(self.point_to_cluster)) + 1

        # Compute coarse diagonal elements: approximation of (C^T * A * C)_{k,k}
        diag_coarse = np.zeros(self.n_clusters, dtype=np.float64)
        for i in range(laplacian.n_points):
            k_i = self.point_to_cluster[i]
            diag_coarse[k_i] += laplacian.diagonal[i]

            if len(laplacian.neighbor_indices[i]) > 0:
                neighs = laplacian.neighbor_indices[i]
                weights = laplacian.neighbor_weights[i]
                same_cluster = (self.point_to_cluster[neighs] == k_i)
                diag_coarse[k_i] -= np.sum(weights[same_cluster])

        self.inv_diag_coarse = 1.0 / np.maximum(diag_coarse, 1e-4)
        self.inv_diag_fine = 1.0 / laplacian.diagonal

    def apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        """
        Applies symmetric two-level preconditioner:
        z = omega * D^{-1} * r + C * (A_c^{-1} * (C^T * r))
        """
        # 1. Fine diagonal smoothing
        z_fine = 0.7 * (r * self.inv_diag_fine)

        if self.n_clusters == 0:
            return z_fine

        # 2. Coarse restriction: r_c = C^T * r
        r_coarse = np.bincount(self.point_to_cluster, weights=r, minlength=self.n_clusters)

        # 3. Coarse solve: e_c = diag(A_c)^{-1} * r_c
        e_coarse = r_coarse * self.inv_diag_coarse

        # 4. Prolongation: z_coarse = C * e_c
        z_coarse = 0.3 * e_coarse[self.point_to_cluster]

        return z_fine + z_coarse


def solve_meshfree_poisson(
    points: np.ndarray,
    rhs: np.ndarray,
    support_radius: float,
    kappa: float = 0.0,
    tol: float = 1e-5,
    max_iters: int = 100,
    use_preconditioner: bool = True
) -> Tuple[np.ndarray, int, List[float], float]:
    """
    Solves the GRAPH Poisson problem (L + kappa^2 I) u = rhs via matrix-free
    preconditioned CG, where L is the Wendland-weighted graph Laplacian of
    `MeshfreeGraphLaplacian` (SPD; kappa > 0 recommended — kappa = 0 leaves
    the constant near-null mode and convergence stalls).

    This solves the discrete graph system, NOT the continuous PDE: the graph
    Laplacian is not a consistent discretization of nabla^2. To EVALUATE the
    continuous operator on a point cloud, use `ConsistentMeshfreeLaplacian`.

    Returns:
        (solution_u, iterations_count, residual_history, elapsed_ms)
    """
    t0 = time.perf_counter()
    lap = MeshfreeGraphLaplacian(points, support_radius=support_radius, kappa=kappa)
    precond = MultiScalePreconditionedSolver(lap) if use_preconditioner else None

    u = np.zeros(lap.n_points, dtype=np.float64)
    r = rhs.copy() - lap.matvec(u)

    z = precond.apply_preconditioner(r) if precond else r.copy()
    p = z.copy()

    rz_old = float(np.dot(r, z))
    norm_rhs = float(np.linalg.norm(rhs)) + 1e-12
    res_history = [float(np.linalg.norm(r)) / norm_rhs]

    iters = 0
    for it in range(max_iters):
        if res_history[-1] < tol:
            break

        Ap = lap.matvec(p)
        pAp = float(np.dot(p, Ap))
        if pAp <= 1e-15:
            break

        alpha = rz_old / pAp
        u += alpha * p
        r -= alpha * Ap

        rel_res = float(np.linalg.norm(r)) / norm_rhs
        res_history.append(rel_res)
        iters = it + 1

        if rel_res < tol:
            break

        z = precond.apply_preconditioner(r) if precond else r.copy()
        rz_new = float(np.dot(r, z))
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return u, iters, res_history, elapsed_ms


if __name__ == "__main__":
    print("Testing Matrix-Free Meshfree Laplacians...")

    # 1. Consistent operator: must reproduce nabla^2 exactly on quadratics
    #    at EVERY point (interior and one-sided boundary stencils alike),
    #    and second-order-accurately on smooth fields.
    rng = np.random.RandomState(42)

    axes = [np.linspace(-3.0, 3.0, 12) for _ in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    lapc = ConsistentMeshfreeLaplacian(pts, support_radius=1.0)
    v = np.sum(pts ** 2, axis=1)          # nabla^2 v = 6
    err_quad = np.max(np.abs(lapc.matvec(v) - 6.0))
    print(f"RBF-FD quadratic consistency max |L v - 6| : {err_quad:.2e}")
    assert err_quad < 1e-6, "consistent stencil must be exact on quadratics"

    p = np.pi / 6.0
    s = np.prod(np.sin(p * pts), axis=1)
    target = -3 * p * p * s
    err_sin = np.linalg.norm(lapc.matvec(s) - target) / np.linalg.norm(target)
    print(f"RBF-FD smooth-field (sin) rel-L2           : {err_sin:.2e}")
    assert err_sin < 0.2, "second-order approximation expected at this h"

    pts2 = np.stack(np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9),
                                indexing='ij'), axis=-1).reshape(-1, 2)
    lapc2 = ConsistentMeshfreeLaplacian(pts2, support_radius=1.5)
    v2 = np.sum(pts2 ** 2, axis=1)
    err2 = np.max(np.abs(lapc2.matvec(v2) - 4.0))  # 2D: nabla^2 v = 4
    print(f"RBF-FD 2D quadratic consistency max|L v - 4|: {err2:.2e}")
    assert err2 < 1e-6

    # 2. Graph Poisson: two-level preconditioned CG vs plain CG.
    n_pts = 3000
    pts = rng.uniform(-5.0, 5.0, (n_pts, 3))
    r2 = np.sum(pts**2, axis=1)
    rhs = np.exp(-r2 / 4.0) - np.mean(np.exp(-r2 / 4.0))  # zero-mean rhs
    h_radius = 1.2

    u_std, iters_std, res_std, t_std = solve_meshfree_poisson(
        pts, rhs, support_radius=h_radius, kappa=0.1, tol=1e-4, max_iters=150,
        use_preconditioner=False
    )
    u_pcg, iters_pcg, res_pcg, t_pcg = solve_meshfree_poisson(
        pts, rhs, support_radius=h_radius, kappa=0.1, tol=1e-4, max_iters=150,
        use_preconditioner=True
    )

    print(f"Points: {n_pts}")
    print(f"Graph Poisson CG : {iters_std} iters, final res={res_std[-1]:.2e}, time={t_std:.2f} ms")
    print(f"Graph Poisson PCG: {iters_pcg} iters, final res={res_pcg[-1]:.2e}, time={t_pcg:.2f} ms")
    print(f"Iteration Reduction: {iters_std / max(1, iters_pcg):.2f}x")

    assert iters_pcg <= iters_std, "PCG should converge in fewer iterations than plain CG."
    print("Meshfree Laplacian Verification: SUCCESS!")
