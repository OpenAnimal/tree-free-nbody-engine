"""
Non-FMM Module: O(N) Protein Residue Contact Map & Allosteric Graph Builder.
Constructs residue interaction networks (RINs) and contact graphs using 3D Morton Elastic Spatial Hashing.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Set
try:
    from .pdb_loader import MolecularSystem
    from .core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d, morton_decode_3d
    from core._csr import build_csr
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.pdb_loader import MolecularSystem
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D, morton_encode_3d, morton_decode_3d
    from core._csr import build_csr


class ContactMapGraphBuilder:
    """
    Fast O(N) Macromolecular Contact Network & Graph Generator.
    """
    def __init__(self, contact_cutoff: float = 8.0, cell_size: float = 8.0):
        self.cutoff = float(contact_cutoff)
        self.cell_size = max(float(cell_size), self.cutoff)

    def build_ca_contact_graph(self, system: MolecularSystem) -> Dict[str, Any]:
        """
        Builds C-alpha / residue-level contact graph in linear O(N) time.
        """
        t0 = time.perf_counter()
        
        # Filter for CA atoms
        ca_mask = np.array([name == "CA" for name in system.atom_names])
        if not np.any(ca_mask):
            ca_mask = np.ones(system.num_atoms, dtype=bool)

        ca_indices = np.where(ca_mask)[0]
        ca_coords = system.coords[ca_indices]
        ca_resids = system.residue_ids[ca_indices]
        ca_resnames = [system.residue_names[i] for i in ca_indices]
        N_ca = len(ca_indices)

        # Build 3D Morton Spatial Hash
        hash_table = ElasticSpatialHash3D(cell_size=self.cell_size, capacity_hint=N_ca * 2)
        origin = np.min(ca_coords, axis=0) - self.cell_size
        _, unique_keys, inverse = hash_table.build_from_coords(ca_coords, origin=origin)
        K = len(unique_keys)

        # Decode Morton keys to identify adjacent 3x3x3 cells
        cluster_grid = np.array([morton_decode_3d(int(k)) for k in unique_keys], dtype=np.int64)
        g_diff = np.abs(cluster_grid[:, None, :] - cluster_grid[None, :, :])
        is_near_cluster = np.all(g_diff <= 1, axis=-1)

        # Finding N: CSR cell lists replace the per-cluster np.where(inverse==c)
        # O(N*K) scans at the c1 / c2 gathers.
        cell_start, cell_particles, _ = build_csr(inverse, K)

        edges = []
        edge_distances = []
        degrees = np.zeros(N_ca, dtype=np.int32)

        # Collect contacts
        for c1 in range(K):
            idx1 = cell_particles[cell_start[c1]:cell_start[c1 + 1]]
            if len(idx1) == 0:
                continue

            p1 = ca_coords[idx1]

            # Neighboring clusters
            near_clusters = np.where(is_near_cluster[c1])[0]
            for c2 in near_clusters:
                if c2 < c1:
                    continue

                idx2 = cell_particles[cell_start[c2]:cell_start[c2 + 1]]
                if len(idx2) == 0:
                    continue

                p2 = ca_coords[idx2]
                delta = p1[:, None, :] - p2[None, :, :]
                dist = np.linalg.norm(delta, axis=-1)

                if c1 == c2:
                    # Upper triangle only
                    i_grid, j_grid = np.triu_indices(len(idx1), k=1)
                    valid = (dist[i_grid, j_grid] < self.cutoff) & (np.abs(ca_resids[idx1[i_grid]] - ca_resids[idx1[j_grid]]) > 1)
                    
                    for vi, vj, vd in zip(idx1[i_grid][valid], idx1[j_grid][valid], dist[i_grid, j_grid][valid]):
                        edges.append((int(vi), int(vj)))
                        edge_distances.append(float(vd))
                        degrees[vi] += 1
                        degrees[vj] += 1
                else:
                    valid_mask = (dist < self.cutoff)
                    r_i, r_j = np.where(valid_mask)
                    # Exclude immediate sequence neighbors (|i - j| <= 1)
                    seq_sep = np.abs(ca_resids[idx1[r_i]] - ca_resids[idx2[r_j]]) > 1
                    r_i = r_i[seq_sep]
                    r_j = r_j[seq_sep]

                    for ri, rj, vi, vj in zip(r_i, r_j, idx1[r_i], idx2[r_j]):
                        d_val = float(dist[ri, rj])
                        edges.append((int(vi), int(vj)))
                        edge_distances.append(d_val)
                        degrees[vi] += 1
                        degrees[vj] += 1

        elapsed = time.perf_counter() - t0

        # Hub residues (top central nodes in allosteric network)
        hub_ranking = np.argsort(degrees)[::-1]
        top_hubs = []
        for h in hub_ranking[:10]:
            top_hubs.append({
                "residue_id": int(ca_resids[h]),
                "residue_name": ca_resnames[h],
                "degree_centrality": int(degrees[h])
            })

        return {
            "num_residues": N_ca,
            "num_contacts": len(edges),
            "contact_cutoff_angstrom": self.cutoff,
            "edges": edges,
            "edge_distances": edge_distances,
            "degrees": degrees.tolist(),
            "top_hub_residues": top_hubs,
            "elapsed_seconds": elapsed
        }
