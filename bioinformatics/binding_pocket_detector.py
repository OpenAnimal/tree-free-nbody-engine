"""
Non-FMM Module: Fast Protein Binding Pocket & Cavity Detector.
Uses 3D Morton Elastic Spatial Hashing to identify concave active sites & druggable pockets in O(N) time
without allocating dense 3D voxel matrices.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Any
from .pdb_loader import MolecularSystem
from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d, morton_decode_3d


class BindingPocketDetector:
    """
    Lock-Free Grid-Free Protein Binding Pocket & Catalytic Cavity Identifier.
    Scans solvent-excluded voids and groups pocket points into ranked drug-binding sites.
    """
    def __init__(
        self,
        grid_spacing: float = 1.5,     # Angstroms per probe sample
        probe_radius: float = 1.4,     # Water probe radius (Angstroms)
        min_pocket_points: int = 8,
        ray_directions: int = 14,      # Number of spatial direction rays for concavity check
    ):
        self.grid_spacing = float(grid_spacing)
        self.probe_radius = float(probe_radius)
        self.min_pocket_points = int(min_pocket_points)
        self.num_rays = ray_directions

        # Stencil ray directions (Cartesian cube faces, edges, corners)
        dirs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    v = np.array([dx, dy, dz], dtype=np.float64)
                    dirs.append(v / np.linalg.norm(v))
        self.ray_vectors = np.array(dirs[:ray_directions], dtype=np.float64)

    def detect_pockets(self, system: MolecularSystem) -> Dict[str, Any]:
        """
        Identifies druggable pockets and active site cavities across the protein surface.
        """
        t0 = time.perf_counter()
        coords = system.coords
        vdw = system.radii
        N = len(coords)

        # 1. 3D Morton Spatial Hash of Protein Atoms
        cell_size = 6.0
        atom_hash = ElasticSpatialHash3D(cell_size=cell_size, capacity_hint=N * 2)
        origin = np.min(coords, axis=0) - cell_size
        _, unique_keys, inverse = atom_hash.build_from_coords(coords, origin=origin)
        K = len(unique_keys)

        # 2. Fast Surface Shell Point Generation
        # Generate candidate probe points around a representative subsample of atoms
        stride = max(1, N // 400)
        sample_indices = np.arange(0, N, stride)
        sample_coords = coords[sample_indices]
        sample_radii = vdw[sample_indices]

        # 12-direction surface offset points
        offsets = self.ray_vectors * 3.2  # Probe distance ~3.2 Angstroms from atom center
        probe_points = (sample_coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

        # 3. Filter probe points: must not clash with any atom (dist > vdw) and within outer shell
        # Hash probe points into Morton grid to find nearby atoms in O(1)
        p_shifted = probe_points - origin
        p_ix = np.maximum(0, (p_shifted[:, 0] / cell_size).astype(np.int64))
        p_iy = np.maximum(0, (p_shifted[:, 1] / cell_size).astype(np.int64))
        p_iz = np.maximum(0, (p_shifted[:, 2] / cell_size).astype(np.int64))
        p_morton = morton_encode_3d(p_ix, p_iy, p_iz)

        pocket_candidate_pts = []
        pocket_burial_scores = []

        # Vectorized batch check of probe points against atoms in 3x3x3 local neighborhood
        for p_idx, pt in enumerate(probe_points):
            px_i, py_i, pz_i = int(p_ix[p_idx]), int(p_iy[p_idx]), int(p_iz[p_idx])
            local_atom_list = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nx, ny, nz = px_i + dx, py_i + dy, pz_i + dz
                        if nx < 0 or ny < 0 or nz < 0:
                            continue
                        k_n = int(morton_encode_3d(np.array([nx]), np.array([ny]), np.array([nz]))[0])
                        c_idx = atom_hash.lookup(k_n)
                        if c_idx is not None:
                            local_atom_list.extend(np.where(inverse == c_idx)[0])

            if not local_atom_list:
                continue

            local_atoms = np.unique(local_atom_list)

            # Check distance to local atoms
            d_local = np.linalg.norm(coords[local_atoms] - pt, axis=1)
            # Must not clash with local atoms (d > vdw)
            if np.any(d_local < vdw[local_atoms]):
                continue

            # Concavity Test: Count how many rays hit surrounding protein atoms
            ray_endpoints = pt + self.ray_vectors * 7.0  # (num_rays, 3)
            # Sample distance along rays to entire cluster neighborhood
            diff = ray_endpoints[:, None, :] - coords[local_atoms][None, :, :]  # (num_rays, L, 3)
            min_ray_dist = np.min(np.linalg.norm(diff, axis=-1), axis=1)
            
            ray_hits = np.sum(min_ray_dist < 4.0)

            # High enclosure fraction -> Concave pocket point
            if ray_hits >= 4:
                pocket_candidate_pts.append(pt)
                pocket_burial_scores.append(ray_hits / self.num_rays)

        elapsed = time.perf_counter() - t0

        if not pocket_candidate_pts:
            return {
                "num_pockets": 0,
                "pockets": [],
                "total_pocket_points": 0,
                "elapsed_seconds": elapsed
            }

        pocket_pts_arr = np.array(pocket_candidate_pts, dtype=np.float64)
        scores_arr = np.array(pocket_burial_scores, dtype=np.float64)

        # 4. Cluster Pocket Points into Discrete Binding Sites
        clusters = []
        visited = np.zeros(len(pocket_pts_arr), dtype=bool)

        for i in range(len(pocket_pts_arr)):
            if visited[i]:
                continue
            
            cluster_members = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr = queue.pop(0)
                dists = np.linalg.norm(pocket_pts_arr - pocket_pts_arr[curr], axis=1)
                neighbors = np.where((dists < 6.0) & (~visited))[0]
                for n_idx in neighbors:
                    visited[n_idx] = True
                    cluster_members.append(n_idx)
                    queue.append(n_idx)

            if len(cluster_members) >= self.min_pocket_points:
                c_pts = pocket_pts_arr[cluster_members]
                c_center = np.mean(c_pts, axis=0)
                c_volume = len(cluster_members) * (self.grid_spacing ** 3) * 6.0
                druggability_score = float(np.mean(scores_arr[cluster_members]) * np.log10(len(cluster_members) + 1))

                clusters.append({
                    "pocket_id": len(clusters) + 1,
                    "center": c_center.tolist(),
                    "volume_angstrom3": float(c_volume),
                    "num_points": len(cluster_members),
                    "druggability_score": druggability_score,
                    "points": c_pts
                })

        clusters.sort(key=lambda x: x["druggability_score"], reverse=True)

        return {
            "num_pockets": len(clusters),
            "pockets": clusters,
            "total_pocket_points": len(pocket_pts_arr),
            "elapsed_seconds": elapsed
        }
