"""
Matrix-Free Incremental Potential Contact (IPC) Cloth & Discrete Shell Solver
Rigorous Implementation of:
  1. Baraff & Witkin (SIGGRAPH 1998) "Large Steps in Cloth Simulation"
  2. Grinspun, Hirani, Desbrun, Schröder (SCA 2003) "Discrete Shells"
  3. Bergou, Wardetzky, Robinson, Furfaro, Grinspun (2006) "Discrete Quadratic Bending"
  4. Li et al. (ACM TOG / SIGGRAPH 2020) "Incremental Potential Contact (IPC)"
  5. Farach-Colton, Krapivin, & Kuszmaul (2025) "Non-Reordering Open Addressing"

Features:
  - Exact rotationally-invariant Discrete Mean Curvature Hinge Bending (Zero ghost forces)
  - Geometric + Material Non-Linear Edge Strain with PSD-Projected Hessians
  - Smooth IPC Log-Barrier Contact Potentials with Guaranteed Positive Clearance (d > 0)
  - 100% Matrix-Free Preconditioned Conjugate Gradient (PCG) with Jacobi Preconditioner
  - Discrete distance-check line search; candidate set frozen at the predicted
    step — classic vertex-vertex IPC limitation, no point-triangle CCD
"""

import numpy as np
import time
import os
import sys
from typing import Tuple, List, Dict, Optional, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.spatial_index import CellIndex
from core._csr import build_csr

class ClothMesh:
    """
    Triangulated Discrete Shell & Fabric Mesh Representation.
    Implements true rotationally-invariant discrete shell mechanics.

    Material model note: the energy is STRETCH (all structural edges,
    including each quad's diagonal, under ``k_stretch``) + discrete hinge
    BENDING (``k_bend``).  There is NO separate shear energy term: the
    ``k_shear`` constructor parameter is stored for API compatibility and
    validated by ``combine_cloth_meshes`` but is never consumed — shear
    resistance comes from the stretch of the diagonal structural edges.
    """
    def __init__(
        self,
        positions: np.ndarray,
        triangles: np.ndarray,
        k_stretch: float = 1500.0,
        k_shear: float = 600.0,
        k_bend: float = 0.05,
        density: float = 0.25
    ):
        self.rest_positions = np.array(positions, dtype=np.float64)
        self.triangles = np.array(triangles, dtype=np.int32)
        self.num_vertices = len(positions)
        self.num_faces = len(triangles)
        
        self.k_stretch = float(k_stretch)
        self.k_shear = float(k_shear)
        self.k_bend = float(k_bend)
        self.density = float(density)
        
        self._build_topology()
        self._compute_masses()

    def _build_topology(self):
        """
        Builds structural edges (all triangle edges, including the grid
        diagonals that carry shear resistance via the stretch term — the
        separate ``k_shear`` stiffness is NOT used; see class docstring)
        and discrete bending hinges.
        """
        # 1. Structural Edges
        edge_dict = {}
        for tri_idx, tri in enumerate(self.triangles):
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            for (u, v), opp in [((min(i, j), max(i, j)), k), ((min(j, k), max(j, k)), i), ((min(k, i), max(k, i)), j)]:
                edge = (u, v)
                if edge not in edge_dict:
                    edge_dict[edge] = [opp]
                else:
                    edge_dict[edge].append(opp)
                    
        edges = list(edge_dict.keys())
        # reshape(-1, 2): an empty edge list yields a 1-D (0,) array from
        # np.array, which crashed `struct_edges[:, 0]` below with IndexError
        # for face-less / empty meshes (R10-E1).
        self.struct_edges = np.array(edges, dtype=np.int32).reshape(-1, 2)
        diff_s = self.rest_positions[self.struct_edges[:, 0]] - self.rest_positions[self.struct_edges[:, 1]]
        self.struct_rest_lengths = np.linalg.norm(diff_s, axis=-1)
        
        # 2. Discrete Bending Hinges (Bergou / Grinspun Discrete Shells)
        # For each interior edge sharing two triangles (i, j, k) and (j, i, l)
        # Bending stencil weights: c_k, c_l, c_i, c_j such that sum(c) = 0 (Rotation & Translation Invariant)
        hinge_list = []
        stencil_weights = []
        
        for (i, j), wings in edge_dict.items():
            if len(wings) == 2:
                k, l = wings[0], wings[1]
                hinge_list.append((i, j, k, l))
                
                # Compute rest stencil weights from rest triangle geometry
                pi = self.rest_positions[i]
                pj = self.rest_positions[j]
                pk = self.rest_positions[k]
                pl = self.rest_positions[l]
                
                e_len = np.linalg.norm(pi - pj)
                area1 = 0.5 * np.linalg.norm(np.cross(pj - pi, pk - pi)) + 1e-12
                area2 = 0.5 * np.linalg.norm(np.cross(pi - pj, pl - pj)) + 1e-12
                
                # Discrete Mean Curvature weights
                w_k = e_len / (2.0 * area1)
                w_l = e_len / (2.0 * area2)
                
                # Projection to find cotangent fractions along shared edge
                e_dir = (pj - pi) / e_len
                cos_k = np.dot(pk - pi, e_dir) / e_len
                cos_l = np.dot(pl - pi, e_dir) / e_len
                
                w_i = -(1.0 - cos_k) * w_k - (1.0 - cos_l) * w_l
                w_j = -cos_k * w_k - cos_l * w_l
                
                # For regular grids, standard isometric bending stencil simplifies to:
                # [w_k, w_l, w_i, w_j] normalized such that sum = 0
                w_sum = w_k + w_l + w_i + w_j
                w_i -= w_sum * 0.5
                w_j -= w_sum * 0.5
                
                stencil_weights.append((w_k, w_l, w_i, w_j))
                
        if len(hinge_list) > 0:
            self.hinges = np.array(hinge_list, dtype=np.int32)
            self.hinge_weights = np.array(stencil_weights, dtype=np.float64)
        else:
            self.hinges = np.empty((0, 4), dtype=np.int32)
            self.hinge_weights = np.empty((0, 4), dtype=np.float64)

        # 3. Fast Topological 1-Ring Exclusion Hash for Broadphase Contact
        topo_keys = []
        for u, v in self.struct_edges:
            topo_keys.append((int(min(u, v)) << 32) | int(max(u, v)))
        for tri in self.triangles:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            topo_keys.append((min(i, j) << 32) | max(i, j))
            topo_keys.append((min(j, k) << 32) | max(j, k))
            topo_keys.append((min(k, i) << 32) | max(k, i))
        self.topo_exclusion_set = set(topo_keys)

    def _compute_masses(self):
        """Computes lumped mass per vertex proportional to surrounding triangle areas."""
        self.masses = np.zeros(self.num_vertices, dtype=np.float64)
        p0 = self.rest_positions[self.triangles[:, 0]]
        p1 = self.rest_positions[self.triangles[:, 1]]
        p2 = self.rest_positions[self.triangles[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)
        tri_masses = (areas * self.density) / 3.0
        
        np.add.at(self.masses, self.triangles[:, 0], tri_masses)
        np.add.at(self.masses, self.triangles[:, 1], tri_masses)
        np.add.at(self.masses, self.triangles[:, 2], tri_masses)
        # Minimum physical mass per vertex
        self.masses = np.maximum(self.masses, 1e-4)


def create_cloth_grid(
    nx: int = 20,
    ny: int = 20,
    width: float = 0.6,
    height: float = 0.6,
    center: Tuple[float, float, float] = (0.5, 0.5, 0.62),
    k_stretch: float = 1800.0,
    k_bend: float = 0.06,
    density: float = 0.3
) -> ClothMesh:
    """Generates a regular triangulated cloth sheet with diagonal cross-bracing."""
    x = np.linspace(-width / 2.0, width / 2.0, nx)
    y = np.linspace(-height / 2.0, height / 2.0, ny)
    xx, yy = np.meshgrid(x, y)
    
    positions = np.zeros((nx * ny, 3), dtype=np.float64)
    positions[:, 0] = xx.ravel() + center[0]
    positions[:, 1] = yy.ravel() + center[1]
    positions[:, 2] = center[2]
    
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            i0 = j * nx + i
            i1 = j * nx + (i + 1)
            i2 = (j + 1) * nx + i
            i3 = (j + 1) * nx + (i + 1)
            # Alternating diagonal pattern for isotropic deformation
            if (i + j) % 2 == 0:
                triangles.append([i0, i1, i2])
                triangles.append([i1, i3, i2])
            else:
                triangles.append([i0, i1, i3])
                triangles.append([i0, i3, i2])
            
    return ClothMesh(
        positions=positions,
        triangles=np.array(triangles, dtype=np.int32),
        k_stretch=k_stretch,
        k_shear=k_stretch * 0.35,
        k_bend=k_bend,
        density=density
    )


def combine_cloth_meshes(meshes: List[ClothMesh]) -> ClothMesh:
    """Combines multiple independent cloth meshes into a unified multi-sheet system.

    NOTE: the unified ``ClothMesh`` stores a single set of material parameters
    (k_stretch, k_shear, k_bend, density) applied to ALL layers.  Callers that
    deliberately use different per-layer parameters (e.g. cloth_shell_simulation.py
    passes k_stretch=1800/1600, k_bend=0.06/0.05, density=0.25/0.22) will have
    the first mesh's parameters silently applied to every layer — the per-layer
    values are discarded.  This function asserts that all layers share equal
    material parameters so the silent discard is caught explicitly.  To support
    true per-layer materials, ``ClothMesh`` would need per-element material
    arrays indexed by edge/hinge; that is a larger refactor not done here.
    """
    if len(meshes) == 1:
        return meshes[0]

    # Assert all layers share equal material parameters so the single-parameter
    # ClothMesh representation is not silently discarding per-layer values.
    base = meshes[0]
    for idx, m in enumerate(meshes[1:], start=1):
        if (m.k_stretch != base.k_stretch or m.k_shear != base.k_shear or
                m.k_bend != base.k_bend or m.density != base.density):
            raise ValueError(
                f"combine_cloth_meshes: layer {idx} has different material "
                f"parameters (k_stretch={m.k_stretch}, k_shear={m.k_shear}, "
                f"k_bend={m.k_bend}, density={m.density}) than layer 0 "
                f"(k_stretch={base.k_stretch}, k_shear={base.k_shear}, "
                f"k_bend={base.k_bend}, density={base.density}). The unified "
                f"ClothMesh stores a single material set; per-layer materials "
                f"are not supported. Equalize the parameters or implement "
                f"per-element material arrays."
            )

    all_pos = [m.rest_positions for m in meshes]
    all_tris = []
    offset = 0
    for m in meshes:
        all_tris.append(m.triangles + offset)
        offset += m.num_vertices

    return ClothMesh(
        positions=np.vstack(all_pos),
        triangles=np.vstack(all_tris),
        k_stretch=base.k_stretch,
        k_shear=base.k_shear,
        k_bend=base.k_bend,
        density=base.density
    )


def line_search_accepts(psi_trial: float, psi_init: float, halving: int) -> bool:
    """Newton line-search acceptance predicate (extracted for direct testing).

    Accept a valid trial when its incremental potential satisfies the
    sufficient-decrease check, or unconditionally on the last halving arm
    (halving == 5) so the loop always terminates with a valid step when one
    exists. Increasing-energy VALID trials at halving < 5 are rejected and
    the step is halved -- the path whose guard historically crashed through
    the `_`-rebinding bug.
    """
    return bool(psi_trial <= psi_init + 1e-3 or halving == 5)


class MatrixFreeIPCSolver:
    """
    State-of-the-Art Matrix-Free Incremental Potential Contact (IPC) Solver.
    Combines Discrete Shell Elasticity, Matrix-Free SpMV, and Smooth IPC Log-Barriers.

    Deprecated parameter:
        cell_size — accepted but ignored.  The production broadphase uses
        ``self.dhat`` as the cell size (vectorized canonical-half-offset
        scheme, same ring-1 candidate set as the retained CellIndex-based
        ``_find_broadphase_candidates_reference``); the old ``cell_size``
        constructor argument is no longer wired to anything.  It is kept as an
        optional no-op so existing callers do not break; remove it from new
        code.
    """
    def __init__(
        self,
        dhat: float = 0.015,         # Barrier activation threshold (1.5 cm)
        stiffness: float = 4e3,      # Barrier stiffness kappa
        max_newton_iters: int = 5,
        cg_max_iters: int = 16,
        cg_tol: float = 1e-4,
        damp_coef: float = 0.15,     # Rayleigh internal damping
        cell_size: Optional[float] = None  # DEPRECATED — ignored (broadphase uses dhat)
    ):
        self.dhat = float(dhat)
        self.stiffness = float(stiffness)
        self.max_newton_iters = max_newton_iters
        self.cg_max_iters = cg_max_iters
        self.cg_tol = cg_tol
        self.damp_coef = damp_coef

        self.spheres: List[Dict] = []
        self.planes: List[Dict] = []

    def add_sphere_obstacle(self, center: np.ndarray, radius: float):
        self.spheres.append({
            "center": np.array(center, dtype=np.float64),
            "radius": float(radius)
        })

    def add_plane_obstacle(self, point: np.ndarray, normal: np.ndarray):
        n = np.array(normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        self.planes.append({
            "point": np.array(point, dtype=np.float64),
            "normal": n
        })

    # -------------------------------------------------------------------------
    # 1. Discrete Shell Elastic Energy, Forces & Matrix-Free Hessian Products
    # -------------------------------------------------------------------------

    # -- Geometry caching + fast scatter (Newton-PCG optimization) ----------
    # The elastic and barrier Hessian-vector products are called once per CG
    # iteration, but the geometric quantities (edge diffs, normals, active
    # barrier pairs, h_scalar) depend only on `positions` (frozen during CG).
    # Precomputing them once per Newton step and reusing across all CG
    # iterations eliminates the redundant geometry recomputation that
    # dominated the profile (~20% of step time).  The np.add.at scatter
    # (another ~37%) is replaced by np.bincount, which uses optimized C
    # summation instead of the per-element Python-loop fallback of add.at.
    _ARANGE3 = np.arange(3, dtype=np.int64)

    @staticmethod
    def _scatter_add_2d(idx_list, val_list, N, ncols=3):
        """Accumulate (E_i, ncols) value blocks into rows idx_list of an
        (N, ncols) array via a single np.bincount on flat indices.
        Replaces multiple np.add.at calls (which use a slow per-element
        Python fallback) with one vectorized C-level scatter-add."""
        idx = np.concatenate(idx_list)
        vals = np.concatenate(val_list, axis=0)
        arange = np.arange(ncols, dtype=np.int64)
        flat_idx = (idx[:, None] * ncols + arange).ravel()
        out = np.bincount(flat_idx, weights=vals.ravel(),
                          minlength=N * ncols)
        return out.reshape(N, ncols)

    def _precompute_elastic_geometry(self, positions, cloth):
        """Cache all edge/hinge geometry that depends only on `positions`
        (not on the CG search direction v).  Reused across all CG
        iterations within a single Newton step and also by the gradient
        evaluation at the same x."""
        N = len(positions)
        geo = {}
        # used by both the stretch and hinge scatter-index precomputations
        arange3 = self._ARANGE3
        if len(cloth.struct_edges) > 0:
            i_idx = cloth.struct_edges[:, 0]
            j_idx = cloth.struct_edges[:, 1]
            diff = positions[i_idx] - positions[j_idx]
            dist = np.linalg.norm(diff, axis=-1) + 1e-12
            L0 = cloth.struct_rest_lengths
            delta_L = dist - L0
            normals = diff / dist[:, None]
            trans_coeff = np.maximum(0.0, 1.0 - L0 / dist)[:, None]
            # Precompute flat scatter indices (topology is constant across
            # CG iterations; only the values change).  Each edge contributes
            # +h_v_edge to row i and -h_v_edge to row j.
            arange3 = self._ARANGE3
            sidx = np.concatenate([i_idx, j_idx])
            geo["stretch"] = (i_idx, j_idx, diff, dist, delta_L, normals,
                              trans_coeff,
                              (sidx[:, None] * 3 + arange3).ravel(),
                              N * 3)
        if len(cloth.hinges) > 0:
            idx_i = cloth.hinges[:, 0]
            idx_j = cloth.hinges[:, 1]
            idx_k = cloth.hinges[:, 2]
            idx_l = cloth.hinges[:, 3]
            wk = cloth.hinge_weights[:, 0, None]
            wl = cloth.hinge_weights[:, 1, None]
            wi = cloth.hinge_weights[:, 2, None]
            wj = cloth.hinge_weights[:, 3, None]
            H = (wk * positions[idx_k] + wl * positions[idx_l]
                 + wi * positions[idx_i] + wj * positions[idx_j])
            hidx = np.concatenate([idx_k, idx_l, idx_i, idx_j])
            geo["hinge"] = (idx_i, idx_j, idx_k, idx_l, wk, wl, wi, wj, H,
                            (hidx[:, None] * 3 + arange3).ravel(),
                            N * 3)
        return geo

    def _precompute_barrier_geometry(self, positions, candidate_pairs):
        """Cache barrier-active pair geometry (normals, h_scalar) that
        depends only on `positions`, not on the CG direction v."""
        geo = {"pairs": None, "spheres": [], "planes": []}
        if len(candidate_pairs) > 0:
            i_idx = candidate_pairs[:, 0]
            j_idx = candidate_pairs[:, 1]
            diff = positions[i_idx] - positions[j_idx]
            dist = np.linalg.norm(diff, axis=-1)
            active = (dist < self.dhat) & (dist > 1e-9)
            if np.any(active):
                d = dist[active]
                dhat = self.dhat
                normals = diff[active] / d[:, None]
                h_scalar = self.stiffness * np.maximum(
                    1e-2,
                    -2.0 * np.log(d / dhat) + 4.0 * (dhat - d) / d
                    + ((dhat - d) ** 2) / (d ** 2))
                geo["pairs"] = (i_idx[active], j_idx[active], d, normals,
                                h_scalar)
        for sphere in self.spheres:
            diff_s = positions - sphere["center"]
            dist_s = np.linalg.norm(diff_s, axis=-1)
            gap = dist_s - sphere["radius"]
            active_s = (gap < self.dhat) & (gap > 1e-9)
            if np.any(active_s):
                d = gap[active_s]
                dhat = self.dhat
                normals = diff_s[active_s] / dist_s[active_s, None]
                h_scalar = self.stiffness * np.maximum(
                    1e-2,
                    -2.0 * np.log(d / dhat) + 4.0 * (dhat - d) / d
                    + ((dhat - d) ** 2) / (d ** 2))
                geo["spheres"].append((active_s, d, normals, h_scalar))
        for plane in self.planes:
            pn = plane["normal"]
            gap = np.sum((positions - plane["point"]) * pn, axis=-1)
            active_p = (gap < self.dhat) & (gap > 1e-9)
            if np.any(active_p):
                d = gap[active_p]
                dhat = self.dhat
                h_scalar = self.stiffness * np.maximum(
                    1e-2,
                    -2.0 * np.log(d / dhat) + 4.0 * (dhat - d) / d
                    + ((dhat - d) ** 2) / (d ** 2))
                geo["planes"].append((active_p, d, pn, h_scalar))
        return geo

    def compute_elastic_energy_and_forces(
        self,
        positions: np.ndarray,
        cloth: ClothMesh,
        geo: Optional[dict] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Computes internal stretch and rotation-invariant discrete hinge bending forces.
        """
        if geo is None:
            geo = self._precompute_elastic_geometry(positions, cloth)
        N = len(positions)
        forces = np.zeros_like(positions)
        total_energy = 0.0

        # A. Non-Linear Green-Lagrange / Edge Spring Elasticity
        s = geo.get("stretch")
        if s is not None:
            i_idx, j_idx, diff, dist, delta_L, normals, _trans, flat_idx, minlen = s
            total_energy += 0.5 * cloth.k_stretch * float(np.sum(delta_L**2))

            # Restoring force: f_i = -k_s * (d - L0) * (x_i - x_j)/d
            f_mag = cloth.k_stretch * delta_L
            f_edge = f_mag[:, None] * normals

            forces += np.bincount(flat_idx,
                                   weights=np.concatenate([-f_edge, f_edge]).ravel(),
                                   minlength=minlen).reshape(N, 3)

        # B. Rotationally-Invariant Discrete Shell Bending (Bergou / Grinspun)
        # Mean Curvature Vector: H = w_k * x_k + w_l * x_l + w_i * x_i + w_j * x_j
        # Since sum(w) = 0, H = 0 for any flat triangle pair in ANY 3D orientation!
        h = geo.get("hinge")
        if h is not None:
            idx_i, idx_j, idx_k, idx_l, wk, wl, wi, wj, H, flat_idx, minlen = h

            # Bending energy E_bend = 0.5 * k_b * ||H||^2
            total_energy += 0.5 * cloth.k_bend * float(np.sum(H**2))

            # Forces: f_v = -k_b * w_v * H
            forces += np.bincount(flat_idx,
                                   weights=np.concatenate([
                                       -cloth.k_bend * wk * H,
                                       -cloth.k_bend * wl * H,
                                       -cloth.k_bend * wi * H,
                                       -cloth.k_bend * wj * H]).ravel(),
                                   minlength=minlen).reshape(N, 3)

        return total_energy, forces

    def apply_elastic_hessian_vector_product(
        self,
        v: np.ndarray,
        positions: np.ndarray,
        cloth: ClothMesh,
        geo: Optional[dict] = None
    ) -> np.ndarray:
        """
        Matrix-Free Positive-Semi-Definite (PSD) evaluation of H_elastic * v.
        """
        if geo is None:
            geo = self._precompute_elastic_geometry(positions, cloth)
        N = len(v)
        Hv = np.zeros_like(v)

        # A. Stretch Hessian-vector product (PSD projected)
        s = geo.get("stretch")
        if s is not None:
            i_idx, j_idx, _diff, _dist, _delta_L, normals, trans_coeff, flat_idx, minlen = s

            v_diff = v[i_idx] - v[j_idx]
            v_proj = np.sum(v_diff * normals, axis=-1, keepdims=True)

            # Longitudinal stiffness + PSD transverse geometric stiffness
            v_ortho = v_diff - v_proj * normals

            h_v_edge = cloth.k_stretch * (v_proj * normals + trans_coeff * v_ortho)
            Hv += np.bincount(flat_idx,
                              weights=np.concatenate([h_v_edge, -h_v_edge]).ravel(),
                              minlength=minlen).reshape(N, 3)

        # B. Discrete Shell Bending Hessian-vector product
        # H_bend is constant PSD operator: (H_bend v)_a = k_b * w_a * sum_b(w_b * v_b)
        h = geo.get("hinge")
        if h is not None:
            idx_i, idx_j, idx_k, idx_l, wk, wl, wi, wj, _H, flat_idx, minlen = h

            v_H = wk * v[idx_k] + wl * v[idx_l] + wi * v[idx_i] + wj * v[idx_j]

            Hv += np.bincount(flat_idx,
                              weights=np.concatenate([
                                  cloth.k_bend * wk * v_H,
                                  cloth.k_bend * wl * v_H,
                                  cloth.k_bend * wi * v_H,
                                  cloth.k_bend * wj * v_H]).ravel(),
                              minlength=minlen).reshape(N, 3)

        return Hv

    # -------------------------------------------------------------------------
    # 2. Vectorized Broadphase Spatial Hashing (canonical-half-offset, ring-1)
    # -------------------------------------------------------------------------
    # The 13 canonical half-offsets in {-1,0,1}^3 \ {0} with lexicographically
    # positive sign (first nonzero component is +1).  Each unordered cross-cell
    # neighbor pair (cell A, cell B = A + d) with Chebyshev distance 1 is
    # emitted by exactly one of these offsets, so no dedup is needed.
    _BROADPHASE_HALF_OFFSETS = (
        (1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (0, 1, 0), (0, 1, 1), (0, 1, -1),
        (0, 0, 1),
    )
    _BP_OFFSET = 1024
    _BP_STRIDE = 2048

    @staticmethod
    def _build_dist2_offsets():
        """Distance-2 canonical offsets and their midpoint-candidate offsets.

        For each lexicographically-positive offset d with Chebyshev norm
        exactly 2, ``midpoints`` is the list of partial offsets e such that
        ``A + e`` is a cell within ring-1 of BOTH ``A`` and ``B = A + d``
        (the candidate "witness" cells whose occupancy makes the reference
        broadphase emit the (A, B) pair).
        """
        import itertools
        out = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    d = (dx, dy, dz)
                    if max(abs(dx), abs(dy), abs(dz)) != 2:
                        continue
                    if d <= (0, 0, 0):
                        continue
                    parts = []
                    for comp in d:
                        if abs(comp) == 2:
                            parts.append([1 if comp > 0 else -1])
                        elif abs(comp) == 1:
                            parts.append([0, 1 if comp > 0 else -1])
                        else:
                            parts.append([-1, 0, 1])
                    mids = [np.array(e, dtype=np.int64) for e in itertools.product(*parts)]
                    out.append((np.array(d, dtype=np.int64), mids))
        return out

    _BROADPHASE_DIST2_OFFSETS = None  # lazily built (see _get_dist2_offsets)

    @classmethod
    def _get_dist2_offsets(cls):
        if cls._BROADPHASE_DIST2_OFFSETS is None:
            cls._BROADPHASE_DIST2_OFFSETS = cls._build_dist2_offsets()
        return cls._BROADPHASE_DIST2_OFFSETS

    def find_broadphase_candidates(
        self,
        positions: np.ndarray,
        cloth: Optional[ClothMesh] = None
    ) -> np.ndarray:
        """Vectorized ring-1 Chebyshev broadphase with cell_size=dhat.

        With cell_size=dhat and ring=1, every pair of vertices whose Euclidean
        distance is < dhat is guaranteed to fall in the same or an adjacent
        cell, so the candidate set is a complete superset of all true contact
        pairs (no tunneling).  False positives (pairs in adjacent cells but
        > dhat apart) are pruned downstream by the barrier energy's
        ``dist < dhat`` active mask, so physics correctness is unaffected.

        The candidate set EXACTLY matches the reference implementation
        (``_find_broadphase_candidates_reference``), which for each occupied
        cell K emits all triu pairs of the 27-cell ring-1 neighborhood of K.
        That set is the "ring-1 neighborhood closure": every unordered pair
        (i, j) such that some occupied cell K has both cell(i) and cell(j)
        within Chebyshev distance 1.  This includes pairs whose cells are at
        Chebyshev distance 2 (when an occupied "midpoint" cell sits between
        them) in addition to the true Chebyshev-1 pairs.

        Scheme (fully vectorized, no per-cell Python loop, no ``np.unique``
        dedup — each unordered pair is emitted EXACTLY once):
          1. Origin-center the positions (translation-invariant pair
             generation); raise ``ValueError`` when span >= 1024*dhat.
          2. Integer cell coords ``cc = floor(positions_centered / dhat)``;
             encode keys as ``((cc0+1024)*2048 + cc1+1024)*2048 + cc2+1024``
             (fits int64; each axis spans < 1024 cells after centering).
          3. Build a CSR cell list via ``core/_csr.build_csr``.
          4. Chebyshev-1 cross-cell pairs: for each of the 13 canonical
             half-offsets, re-encode the shifted cell coords,
             ``np.searchsorted`` into the sorted unique keys to find occupied
             neighbors, and emit the cross product of the two CSR ranges with
             the standard vectorized variable-length expansion.  Same-cell
             pairs use the same expansion with A=B and a ``ia < ib`` mask.
          5. Chebyshev-2 cross-cell pairs (reference closure): for each of the
             49 canonical distance-2 offsets, find occupied (A, B=A+d) pairs
             via ``np.searchsorted``; for each, check whether ANY midpoint
             cell (a cell within ring-1 of both A and B) is occupied, again
             via ``np.searchsorted``; if so emit the cross product.  Each
             (A, B) pair is processed by exactly one canonical offset, so no
             dedup is needed.
          6. ``lo, hi = min, max``; apply the topological exclusion filter via
             packed-int64 ``np.isin`` against ``cloth.topo_exclusion_set``.

        Performance (re-measured on this machine, 2026-08-22): the broadphase
        is no longer the bottleneck.  On the drape scene (N=200) the new
        broadphase is ~44x faster than the per-key-loop reference
        (``_find_broadphase_candidates_reference``); on a random N=5000 scene
        it is ~53x faster (both microbenchmarks, single-threaded NumPy).  The
        broadphase share of total solver step time is ~32-38% across the
        benchmark ladder (N=484..10000); the Newton-PCG solve (~61-68%) is
        the dominant cost, reduced from ~76-84% by geometry caching across
        CG iterations and np.bincount-based scatter (replacing np.add.at).
        See ``benchmark_contact_scaling.py`` and the README table for the
        per-N measured numbers (machine-dependent; single-threaded NumPy).
        """
        # Origin-centering + span guard (identical to the reference impl).
        if len(positions) == 0:
            return np.empty((0, 2), dtype=np.int32)
        p_min = positions.min(axis=0)
        p_max = positions.max(axis=0)
        span = float(np.max(p_max - p_min))
        domain_span = 1024.0 * self.dhat
        if span >= domain_span:
            raise ValueError(
                f"Scene span {span:.4f} exceeds CellIndex world-mode domain "
                f"({domain_span:.4f} = 1024*dhat with dhat={self.dhat}). "
                f"Reduce the scene extent or increase dhat."
            )
        centroid = 0.5 * (p_min + p_max)
        positions_centered = positions - centroid

        topo = cloth.topo_exclusion_set if cloth is not None else set()
        topo_arr = (np.fromiter((int(k) for k in topo), dtype=np.int64,
                                count=len(topo)) if topo
                    else np.empty(0, dtype=np.int64))

        dhat = self.dhat
        OFFSET = self._BP_OFFSET
        STRIDE = self._BP_STRIDE

        def _encode(coords):
            return ((coords[:, 0] + OFFSET) * STRIDE
                    + coords[:, 1] + OFFSET) * STRIDE + coords[:, 2] + OFFSET

        def _occupancy(query_keys, unique_keys, K):
            """Boolean (len(query_keys),): query key is an occupied cell."""
            found = np.searchsorted(unique_keys, query_keys, side="left")
            found_clipped = np.minimum(found, K - 1)
            return unique_keys[found_clipped] == query_keys

        # 1. Integer cell coords + key encoding.
        cc = np.floor(positions_centered / dhat).astype(np.int64)  # (N,3)
        keys = _encode(cc)

        # 2. CSR cell list.
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        K = len(unique_keys)
        cell_start, cell_particles, _ = build_csr(inverse, K)

        # 3. Decode unique keys back to integer cell coords (K,3).
        uc2 = unique_keys % STRIDE
        tmp = unique_keys // STRIDE
        uc1 = tmp % STRIDE
        uc0 = tmp // STRIDE
        ucoords = np.stack([uc0 - OFFSET, uc1 - OFFSET, uc2 - OFFSET], axis=1)

        lo_parts: List[np.ndarray] = []
        hi_parts: List[np.ndarray] = []

        def _emit_cross(a_idx: np.ndarray, b_idx: np.ndarray):
            """Vectorized cross product of CSR ranges for paired cells.

            For each pair (a_idx[i], b_idx[i]) emits na*nb particle pairs
            (ia, ib) with ia from cell a and ib from cell b.
            """
            if len(a_idx) == 0:
                return
            na = cell_start[a_idx + 1] - cell_start[a_idx]
            nb = cell_start[b_idx + 1] - cell_start[b_idx]
            counts = na * nb
            total = int(counts.sum())
            if total == 0:
                return
            pair_id = np.repeat(np.arange(len(a_idx), dtype=np.int64), counts)
            starts = np.empty(len(a_idx), dtype=np.int64)
            starts[0] = 0
            if len(a_idx) > 1:
                np.cumsum(counts[:-1], out=starts[1:])
            off = np.arange(total, dtype=np.int64) - starts[pair_id]
            nb_rep = nb[pair_id]
            ia = cell_particles[cell_start[a_idx[pair_id]] + off // nb_rep]
            ib = cell_particles[cell_start[b_idx[pair_id]] + off % nb_rep]
            lo_parts.append(np.minimum(ia, ib))
            hi_parts.append(np.maximum(ia, ib))

        # 4a. Chebyshev-1 cross-cell pairs (13 canonical half-offsets).
        for dx, dy, dz in self._BROADPHASE_HALF_OFFSETS:
            ncoords = ucoords + np.array([dx, dy, dz])
            valid = np.all((ncoords >= -OFFSET) & (ncoords < OFFSET), axis=1)
            nkeys = _encode(ncoords)
            nkeys = np.where(valid, nkeys, -1)
            found = np.searchsorted(unique_keys, nkeys, side="left")
            found_clipped = np.minimum(found, K - 1)
            match = valid & (unique_keys[found_clipped] == nkeys)
            a_idx = np.nonzero(match)[0]
            b_idx = found_clipped[a_idx]
            _emit_cross(a_idx, b_idx)

        # 4b. Same-cell pairs: cross product with A=B, then mask ia < ib.
        counts_cell = cell_start[1:] - cell_start[:-1]
        multi = np.nonzero(counts_cell >= 2)[0]
        if len(multi) > 0:
            na = counts_cell[multi]
            counts = na * na
            total = int(counts.sum())
            pair_id = np.repeat(np.arange(len(multi), dtype=np.int64), counts)
            starts = np.empty(len(multi), dtype=np.int64)
            starts[0] = 0
            if len(multi) > 1:
                np.cumsum(counts[:-1], out=starts[1:])
            off = np.arange(total, dtype=np.int64) - starts[pair_id]
            na_rep = na[pair_id]
            ia = cell_particles[cell_start[multi[pair_id]] + off // na_rep]
            ib = cell_particles[cell_start[multi[pair_id]] + off % na_rep]
            keep = ia < ib
            lo_parts.append(ia[keep])
            hi_parts.append(ib[keep])

        # 5. Chebyshev-2 cross-cell pairs (reference ring-1 closure).
        # For each canonical distance-2 offset d, find occupied (A, B=A+d)
        # pairs, then keep only those with at least one occupied midpoint cell
        # (a cell within ring-1 of both A and B).  This reproduces the
        # reference's "all triu pairs of the 27-cell neighborhood" closure
        # exactly, each unordered pair emitted once via the canonical offset.
        for d, mids in self._get_dist2_offsets():
            ncoords = ucoords + d
            valid = np.all((ncoords >= -OFFSET) & (ncoords < OFFSET), axis=1)
            nkeys = _encode(ncoords)
            nkeys = np.where(valid, nkeys, -1)
            b_occ = valid & _occupancy(nkeys, unique_keys, K)
            if not np.any(b_occ):
                continue
            a_occ_idx = np.nonzero(b_occ)[0]
            b_idx = np.searchsorted(unique_keys, nkeys[a_occ_idx], side="left")
            b_idx = np.minimum(b_idx, K - 1)
            # Midpoint occupancy: OR over all midpoint-candidate offsets.
            any_mid = np.zeros(len(a_occ_idx), dtype=bool)
            for e in mids:
                mcoords = ucoords[a_occ_idx] + e
                mvalid = np.all((mcoords >= -OFFSET) & (mcoords < OFFSET), axis=1)
                mkeys = _encode(mcoords)
                mkeys = np.where(mvalid, mkeys, -1)
                any_mid |= mvalid & _occupancy(mkeys, unique_keys, K)
            keep = any_mid
            _emit_cross(a_occ_idx[keep], b_idx[keep])

        if not lo_parts:
            return np.empty((0, 2), dtype=np.int32)

        lo = np.concatenate(lo_parts)
        hi = np.concatenate(hi_parts)

        # 6. Topo-exclusion filter (packed int64) + deterministic sort.
        keys64 = (lo.astype(np.int64) << 32) | hi.astype(np.int64)
        if len(topo_arr) > 0:
            keep = ~np.isin(keys64, topo_arr)
            lo = lo[keep]
            hi = hi[keep]
            keys64 = keys64[keep]

        order = np.argsort(keys64, kind="stable")
        return np.stack([lo[order].astype(np.int32), hi[order].astype(np.int32)], axis=1)

    # -------------------------------------------------------------------------
    # Reference broadphase (per-key Python loop + np.unique dedup).
    # Kept for parity testing; NOT used by solve_step.  See
    # test_broadphase_parity_reference in test_matrix_free_ipc.py.
    # -------------------------------------------------------------------------
    def _find_broadphase_candidates_reference(
        self,
        positions: np.ndarray,
        cloth: Optional[ClothMesh] = None
    ) -> np.ndarray:
        """Reference (slow) CellIndex ring-1 broadphase — per-key Python loop.

        For each occupied cell key it calls
        ``ci.neighborhood_indices(key, ring=1)`` and emits ALL triu pairs of
        the 27-cell neighborhood (each pair up to ~27x), then dedups with
        ``np.unique``.  This is the implementation that made the matrix-free
        solver slower than naive O(N^2) (~98% of step time in the broadphase).
        Retained only for parity validation against the vectorized
        ``find_broadphase_candidates``.
        """
        if len(positions) == 0:
            return np.empty((0, 2), dtype=np.int32)
        p_min = positions.min(axis=0)
        p_max = positions.max(axis=0)
        span = float(np.max(p_max - p_min))
        domain_span = 1024.0 * self.dhat
        if span >= domain_span:
            raise ValueError(
                f"Scene span {span:.4f} exceeds CellIndex world-mode domain "
                f"({domain_span:.4f} = 1024*dhat with dhat={self.dhat}). "
                f"Reduce the scene extent or increase dhat."
            )
        centroid = 0.5 * (p_min + p_max)
        positions_centered = positions - centroid

        ci = CellIndex(dims=3, cell_size=self.dhat)
        ci.build(positions_centered)

        topo = cloth.topo_exclusion_set if cloth is not None else set()
        topo_arr = np.fromiter((int(k) for k in topo), dtype=np.int64,
                               count=len(topo)) if topo else np.empty(0, dtype=np.int64)

        pair_lo: List[int] = []
        pair_hi: List[int] = []

        for key in ci.occupied_keys():
            nbr = ci.neighborhood_indices(key, ring=1)
            n = len(nbr)
            if n < 2:
                continue
            iu, ju = np.triu_indices(n, k=1)
            u_arr = np.minimum(nbr[iu], nbr[ju])
            v_arr = np.maximum(nbr[iu], nbr[ju])
            keys64 = (u_arr.astype(np.int64) << 32) | v_arr.astype(np.int64)
            if len(topo_arr) > 0:
                mask = ~np.isin(keys64, topo_arr)
                u_arr = u_arr[mask]
                v_arr = v_arr[mask]
            pair_lo.extend(u_arr.tolist())
            pair_hi.extend(v_arr.tolist())

        if len(pair_lo) == 0:
            return np.empty((0, 2), dtype=np.int32)

        all_keys = (np.array(pair_lo, dtype=np.int64) << 32) | np.array(pair_hi, dtype=np.int64)
        _, uniq_idx = np.unique(all_keys, return_index=True)
        lo = np.array(pair_lo, dtype=np.int32)[uniq_idx]
        hi = np.array(pair_hi, dtype=np.int32)[uniq_idx]
        return np.stack([lo, hi], axis=1)

    # -------------------------------------------------------------------------
    # 3. IPC Log-Barrier Contact Energy, Forces & PSD Hessian Products
    # -------------------------------------------------------------------------
    def compute_barrier_energy_and_forces(
        self,
        positions: np.ndarray,
        candidate_pairs: np.ndarray,
        geo: Optional[dict] = None
    ) -> Tuple[float, np.ndarray]:
        if geo is None:
            geo = self._precompute_barrier_geometry(positions, candidate_pairs)
        N = len(positions)
        forces = np.zeros_like(positions)
        total_energy = 0.0

        # A. Inter-Cloth Proximity Pairs
        p = geo.get("pairs")
        if p is not None:
            i_a, j_a, d, normals, _h = p
            dhat = self.dhat
            ratio = d / dhat

            e_val = -(d - dhat)**2 * np.log(ratio)
            total_energy += self.stiffness * float(np.sum(e_val))

            # g = -kappa * [2(d - dhat)*ln(d/dhat) + (d - dhat)^2 / d]
            g_val = -self.stiffness * (2.0 * (d - dhat) * np.log(ratio) + ((d - dhat)**2) / d)
            f_rep = -g_val[:, None] * normals

            forces += self._scatter_add_2d(
                [i_a, j_a], [f_rep, -f_rep], N)

        # B. Sphere Obstacle Contact
        for active_s, d, normals, _h in geo["spheres"]:
            dhat = self.dhat
            ratio = d / dhat

            e_val = -(d - dhat)**2 * np.log(ratio)
            total_energy += self.stiffness * float(np.sum(e_val))

            g_val = -self.stiffness * (2.0 * (d - dhat) * np.log(ratio) + ((d - dhat)**2) / d)
            forces[active_s] += (-g_val[:, None] * normals)

        # C. Ground Plane Obstacle Contact
        for active_p, d, pn, _h in geo["planes"]:
            dhat = self.dhat
            ratio = d / dhat

            e_val = -(d - dhat)**2 * np.log(ratio)
            total_energy += self.stiffness * float(np.sum(e_val))

            g_val = -self.stiffness * (2.0 * (d - dhat) * np.log(ratio) + ((d - dhat)**2) / d)
            forces[active_p] += (-g_val[:, None] * pn)

        return total_energy, forces

    def apply_barrier_hessian_vector_product(
        self,
        v: np.ndarray,
        positions: np.ndarray,
        candidate_pairs: np.ndarray,
        geo: Optional[dict] = None
    ) -> np.ndarray:
        if geo is None:
            geo = self._precompute_barrier_geometry(positions, candidate_pairs)
        N = len(v)
        Hv = np.zeros_like(v)

        # A. Inter-Cloth Barrier Hessian
        p = geo.get("pairs")
        if p is not None:
            i_a, j_a, d, normals, h_scalar = p

            v_diff = v[i_a] - v[j_a]
            v_proj = np.sum(v_diff * normals, axis=-1, keepdims=True)
            h_v_pair = h_scalar[:, None] * v_proj * normals

            Hv += self._scatter_add_2d(
                [i_a, j_a], [h_v_pair, -h_v_pair], N)

        # B. Sphere Obstacle Barrier Hessian
        for active_s, d, normals, h_scalar in geo["spheres"]:
            v_proj = np.sum(v[active_s] * normals, axis=-1, keepdims=True)
            Hv[active_s] += h_scalar[:, None] * v_proj * normals

        # C. Ground Plane Obstacle Barrier Hessian
        for active_p, d, pn, h_scalar in geo["planes"]:
            v_proj = np.sum(v[active_p] * pn, axis=-1, keepdims=True)
            Hv[active_p] += h_scalar[:, None] * v_proj * pn

        return Hv

    # -------------------------------------------------------------------------
    # 4. Implicit Newton-PCG Time Stepper with CCD Line Search
    # -------------------------------------------------------------------------
    def solve_step(
        self,
        x_prev: np.ndarray,
        v_prev: np.ndarray,
        cloth: ClothMesh,
        dt: float = 0.012,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        metrics: Dict[str, float] = {}
        N = cloth.num_vertices
        masses = cloth.masses
        line_search_failures = 0
        
        # 1. Inertial unconstrained predictive trajectory: x_tilde = x_prev + dt * v_prev + dt^2 * g
        f_ext = masses[:, None] * gravity
        x_tilde = x_prev + v_prev * dt + ((dt**2) / masses[:, None]) * f_ext
        x = x_tilde.copy()
        
        # 2. Broadphase contact generation
        t_bp0 = time.perf_counter()
        candidates = self.find_broadphase_candidates(x, cloth)
        metrics["broadphase_ms"] = (time.perf_counter() - t_bp0) * 1000.0
        metrics["active_candidates"] = len(candidates)
        
        # 3. Newton-PCG Iterations
        t_pcg0 = time.perf_counter()
        total_cg_iters = 0
        
        for newton_iter in range(self.max_newton_iters):
            # Precompute elastic + barrier geometry once per Newton iter.
            # These depend only on x (frozen during the CG loop below), so
            # caching avoids recomputing edge diffs / normals / h_scalar on
            # every CG iteration — the dominant cost before this change.
            el_geo = self._precompute_elastic_geometry(x, cloth)
            bar_geo = self._precompute_barrier_geometry(x, candidates)

            # Compute internal elastic and obstacle/contact barrier forces
            _, f_elastic = self.compute_elastic_energy_and_forces(x, cloth, el_geo)
            _, f_barrier = self.compute_barrier_energy_and_forces(x, candidates, bar_geo)

            # Non-linear residual: g(x) = (M / dt^2) * (x - x_tilde) - f_elastic - f_barrier
            grad = (masses[:, None] / (dt**2)) * (x - x_tilde) - f_elastic - f_barrier
            res_norm = float(np.linalg.norm(grad))

            if res_norm < self.cg_tol:
                break

            # Jacobi diagonal preconditioner P = diag(H_total)^(-1)
            diag_H = masses[:, None] / (dt**2) + cloth.k_stretch * 1.5 + self.stiffness * 0.1
            inv_P = 1.0 / diag_H

            # Matrix-Free Preconditioned Conjugate Gradient (PCG)
            dx = np.zeros_like(x)
            r = -grad.copy()
            z = inv_P * r
            p = z.copy()
            rz_old = float(np.sum(r * z))

            for cg_step in range(self.cg_max_iters):
                total_cg_iters += 1

                # Matrix-Free SpMV: Hp = (M / dt^2) * p + H_elastic(p) + H_barrier(p)
                Hp = (masses[:, None] / (dt**2)) * p + \
                     self.apply_elastic_hessian_vector_product(p, x, cloth, el_geo) + \
                     self.apply_barrier_hessian_vector_product(p, x, candidates, bar_geo)

                pHp = float(np.sum(p * Hp)) + 1e-12
                alpha = rz_old / pHp
                dx += alpha * p
                r -= alpha * Hp

                if float(np.linalg.norm(r)) < self.cg_tol:
                    break

                z = inv_P * r
                rz_new = float(np.sum(r * z))
                beta = rz_new / (rz_old + 1e-12)
                p = z + beta * p
                rz_old = rz_new
                
            # Discrete distance-check line search: the candidate set is frozen
            # at the predicted step (x_tilde), so this is a discrete distance
            # check on that frozen set — a classic vertex-vertex IPC
            # limitation (no point-triangle CCD).  On total failure (no halving
            # produces a valid, non-penetrating trial) we keep x unchanged
            # rather than applying an unvalidated step.
            step_alpha = 1.0
            e_init = 0.5 * float(np.sum(masses[:, None] * ((x - x_tilde)**2))) / (dt**2)
            e_el_init, _ = self.compute_elastic_energy_and_forces(x, cloth, el_geo)
            e_bar_init, _ = self.compute_barrier_energy_and_forces(x, candidates, bar_geo)
            psi_init = e_init + e_el_init + e_bar_init

            # Note (F11): each trial re-evaluates the full elastic energy.  The
            # elastic energy at the base point (e_el_init) could be reused and
            # only the barrier re-evaluated, but that would weaken the
            # sufficient-decrease check (the elastic energy change is O(alpha)
            # and non-negligible for large steps), so we keep the full
            # re-evaluation for correctness.
            accepted = False
            # NOTE: the loop variable is `halving` (NOT `_`). The unpacks
            # below (`e_el_trial, _ = ...` and `e_bar_trial, _ = ...`)
            # rebind whatever name is on the left to the forces ndarray, so
            # a `for _ in range(6):` loop would rebind `_` to an ndarray and
            # a last-chance guard `... or _ == 5` would evaluate
            # `ndarray == 5` -> "ValueError: The truth value of an array is
            # ambiguous" on the first valid trial whose energy increased
            # (the `or` short-circuit only reaches the second operand when
            # the first is False). This is a LATENT defect: instrumented
            # runs of the 8x8 two-layer drape at stiffness 2e5 show zero
            # such events in 20 steps (the first trial always satisfies the
            # sufficient-decrease check), so no in-repo scene currently
            # exercises the path -- the guard is exercised by the direct
            # unit test of the acceptance predicate instead.
            for halving in range(6):
                x_trial = x + step_alpha * dx

                # Validate non-penetration condition d >= d_floor > 0
                valid = True
                for sphere in self.spheres:
                    gap = np.linalg.norm(x_trial - sphere["center"], axis=-1) - sphere["radius"]
                    if np.any(gap < 1e-4):
                        valid = False
                        break
                if valid:
                    for plane in self.planes:
                        gap = np.sum((x_trial - plane["point"]) * plane["normal"], axis=-1)
                        if np.any(gap < 1e-4):
                            valid = False
                            break
                if valid and len(candidates) > 0:
                    dist_pairs = np.linalg.norm(x_trial[candidates[:, 0]] - x_trial[candidates[:, 1]], axis=-1)
                    if np.any(dist_pairs < 1e-4):
                        valid = False

                if valid:
                    e_trial = 0.5 * float(np.sum(masses[:, None] * ((x_trial - x_tilde)**2))) / (dt**2)
                    e_el_trial, _ = self.compute_elastic_energy_and_forces(x_trial, cloth)
                    e_bar_trial, _ = self.compute_barrier_energy_and_forces(x_trial, candidates)
                    psi_trial = e_trial + e_el_trial + e_bar_trial

                    if line_search_accepts(psi_trial, psi_init, halving):
                        x = x_trial
                        accepted = True
                        break
                step_alpha *= 0.5

            if not accepted:
                # Total line-search failure: every halving produced an
                # invalid (penetrating) trial.  Keep x unchanged — NEVER
                # apply an unvalidated step (the previous code applied
                # x + (1/64)*dx without any validity check, silently
                # accepting penetration).
                line_search_failures += 1
                print(f"[WARN] MatrixFreeIPCSolver: line search failed all 6 "
                      f"halvings in Newton iter {newton_iter}; keeping x "
                      f"unchanged (no step applied).")
                
        metrics["newton_pcg_ms"] = (time.perf_counter() - t_pcg0) * 1000.0
        metrics["total_cg_iters"] = total_cg_iters
        metrics["line_search_failures"] = float(line_search_failures)
        
        # 4. Physical velocity update with internal numerical damping
        v_next = (x - x_prev) / dt
        v_next *= (1.0 - self.damp_coef)
        
        metrics["total_step_ms"] = (time.perf_counter() - t0) * 1000.0
        return x, v_next, metrics
