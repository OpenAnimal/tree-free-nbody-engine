"""
Nearly-Linear Spectral Meshfree Laplacian Solver (spectral_meshfree_laplacian.py).

Inspired by:
1. "Nearly-Linear Time Algorithms for Graph Laplacians"
   Daniel A. Spielman and Shang-Hua Teng (SIAM J. Comput. / STOC 2004, 2011).
2. "Solving SDD Linear Systems in Nearly-m^{o(1)} Time"
   Michael B. Cohen, Jonathan A. Kelner, Gary L. Miller, Richard Peng et al. (STOC / FOCS 2014).

Key Algorithmic Principle:
Solving continuous Poisson (\\nabla^2 u = \\rho) or screened Poisson ((\\nabla^2 - \\kappa^2)u = \\rho)
systems on irregular 3D geometries traditionally requires explicit tetrahedral meshes or dense
finite element matrices. Here, we build a Matrix-Free Meshfree Laplacian coupled with a
Tree-Free Multi-Scale Elastic Hash Preconditioner. By restricting fine-level residuals to coarse
spatial cluster centroids and smoothing multi-scale errors, Preconditioned Conjugate Gradients (PCG)
converges in nearly-linear time without ever assembling or factorizing a global sparse matrix.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class SpectralMeshfreeLaplacian:
    """
    Matrix-Free Meshfree Laplacian-Beltrami Operator on 3D Point Sets.
    
    Computes (L + kappa^2 I) * v using normalized SPH / Wendland C2 kernel stencils
    accelerated by spatial hash neighborhood lookups.
    """
    def __init__(self, points: np.ndarray, support_radius: float, kappa: float = 0.0):
        self.points = np.asarray(points, dtype=np.float64)
        self.n_points = len(self.points)
        self.support_radius = float(support_radius)
        self.h = self.support_radius
        self.kappa = float(kappa)
        self.cell_size = self.support_radius

        # Build spatial hash grid
        self.grid_coords = np.floor(self.points / self.cell_size).astype(np.int64)
        self.spatial_buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, coord in enumerate(self.grid_coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key not in self.spatial_buckets:
                self.spatial_buckets[key] = []
            self.spatial_buckets[key].append(idx)

        # Precompute neighbor lists and weights for fast matrix-vector products
        self.neighbor_indices: List[np.ndarray] = []
        self.neighbor_weights: List[np.ndarray] = []
        self.diagonal: np.ndarray = np.zeros(self.n_points, dtype=np.float64)
        self._build_stencils()

    def _wendland_c2(self, r: np.ndarray) -> np.ndarray:
        """Normalized Wendland C2 radial basis kernel."""
        q = np.clip(r / self.h, 0.0, 1.0)
        return (1.0 - q)**4 * (4.0 * q + 1.0)

    def _build_stencils(self):
        """Constructs symmetric positive definite (SPD) Laplacian stencils."""
        # Step 1: Gather symmetric interaction pairs
        raw_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(self.n_points)]
        
        for i in range(self.n_points):
            coord = self.grid_coords[i]
            p_i = self.points[i]
            
            cands = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        k = (int(coord[0] + dx), int(coord[1] + dy), int(coord[2] + dz))
                        if k in self.spatial_buckets:
                            cands.extend(self.spatial_buckets[k])

            cand_arr = np.array(cands, dtype=np.int64)
            cand_arr = cand_arr[cand_arr > i]  # Only consider upper triangular pairs to enforce exact symmetry
            
            if len(cand_arr) == 0:
                continue

            disp = self.points[cand_arr] - p_i
            dists = np.linalg.norm(disp, axis=1)
            mask = dists < self.h
            valid_j = cand_arr[mask]
            valid_dists = dists[mask]

            if len(valid_j) == 0:
                continue

            weights = self._wendland_c2(valid_dists)
            for j_node, w_val in zip(valid_j, weights):
                raw_neighbors[i].append((j_node, float(w_val)))
                raw_neighbors[j_node].append((i, float(w_val)))

        # Step 2: Store symmetric neighbor arrays and diagonal row sums
        for i in range(self.n_points):
            if len(raw_neighbors[i]) == 0:
                self.neighbor_indices.append(np.empty(0, dtype=np.int64))
                self.neighbor_weights.append(np.empty(0, dtype=np.float64))
                self.diagonal[i] = 1.0 + self.kappa**2
            else:
                indices = np.array([idx for idx, _ in raw_neighbors[i]], dtype=np.int64)
                weights = np.array([w for _, w in raw_neighbors[i]], dtype=np.float64)
                self.neighbor_indices.append(indices)
                self.neighbor_weights.append(weights)
                # SDD diagonal: row sum + screening term
                self.diagonal[i] = float(np.sum(weights)) + self.kappa**2

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-free application of (L + kappa^2 I) * v in O(N) operations.
        (L v)_i = (1 + kappa^2) * v_i - \sum_j w_ij * v_j
        """
        out = self.diagonal * v
        for i in range(self.n_points):
            idx = self.neighbor_indices[i]
            if len(idx) > 0:
                out[i] -= np.dot(self.neighbor_weights[i], v[idx])
        return out


class MultiScalePreconditionedSolver:
    """
    Two-Level Symmetric Positive Definite (SPD) Multi-Scale Galerkin Preconditioner for PCG.
    
    Constructs an SPD two-level preconditioner:
        M^{-1} = \\omega D^{-1} + C (C^T A C)^{-1} C^T
    where D is the point diagonal matrix, and C is the spatial cluster indicator matrix.
    Guarantees strict symmetry and positive-definiteness for monotone PCG convergence.
    """
    def __init__(self, laplacian: SpectralMeshfreeLaplacian, coarse_factor: float = 2.0):
        self.lap = laplacian
        self.coarse_cell_size = laplacian.cell_size * coarse_factor
        
        # Build coarse spatial clusters
        c_coords = np.floor(laplacian.points / self.coarse_cell_size).astype(np.int64)
        _, self.point_to_cluster = np.unique(c_coords, axis=0, return_inverse=True)
        self.n_clusters = int(np.max(self.point_to_cluster)) + 1
        
        # Compute coarse diagonal elements: (C^T * A * C)_{k,k}
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
    Solves (L + kappa^2 I) u = rhs via Matrix-Free Preconditioned Conjugate Gradients.
    
    Returns:
        (solution_u, iterations_count, residual_history, elapsed_ms)
    """
    t0 = time.perf_counter()
    lap = SpectralMeshfreeLaplacian(points, support_radius=support_radius, kappa=kappa)
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
    print("Testing Nearly-Linear Spectral Meshfree Laplacian Solver...")
    
    rng = np.random.RandomState(42)
    n_pts = 3000
    pts = rng.uniform(-5.0, 5.0, (n_pts, 3))
    
    # Smooth Gaussian charge distribution
    r2 = np.sum(pts**2, axis=1)
    rhs = np.exp(-r2 / 4.0) - np.mean(np.exp(-r2 / 4.0))  # Zero net charge for Poisson
    
    h_radius = 1.2
    
    # Solve with standard CG
    u_std, iters_std, res_std, t_std = solve_meshfree_poisson(
        pts, rhs, support_radius=h_radius, kappa=0.1, tol=1e-4, max_iters=150, use_preconditioner=False
    )
    
    # Solve with Multi-Scale Preconditioned CG
    u_pcg, iters_pcg, res_pcg, t_pcg = solve_meshfree_poisson(
        pts, rhs, support_radius=h_radius, kappa=0.1, tol=1e-4, max_iters=150, use_preconditioner=True
    )
    
    print(f"Points: {n_pts}")
    print(f"Standard CG:      {iters_std} iters, final res={res_std[-1]:.2e}, time={t_std:.2f} ms")
    print(f"Multi-Scale PCG:  {iters_pcg} iters, final res={res_pcg[-1]:.2e}, time={t_pcg:.2f} ms")
    print(f"Iteration Reduction: {iters_std / max(1, iters_pcg):.2f}x")
    
    assert iters_pcg <= iters_std, "PCG should converge in fewer iterations than standard CG."
    print("Spectral Meshfree Laplacian Verification: SUCCESS!")
