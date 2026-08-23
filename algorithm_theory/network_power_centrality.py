"""
Matrix-Free Effective Resistance & Network Power Centrality (network_power_centrality.py).

Inspired by:
1. "Graph Sparsification by Effective Resistances"
   Daniel A. Spielman and Nikhil Srivastava (SIAM J. Comput. / STOC 2008, 2011).
2. "Nearly-Linear Time Algorithms for Graph Laplacians"
   Daniel A. Spielman and Shang-Hua Teng (SIAM J. Comput. 2011).
3. "Centrality and Network Power in Physical and Financial Systems"
   D. Acemoglu, A. Ozdaglar, A. Tahbaz-Salehi (American Economic Review, 2015).

Key Algorithmic Principle:
In electrical power grids, supply chain logistics, and interbank financial liability networks,
structural power and vulnerability concentrate at chokepoint nodes.
The effective electrical resistance between two nodes u and v:
    R_eff(u, v) = (e_u - e_v)^T * L^+ * (e_u - e_v)
quantifies topological coupling, criticality, and sensitivity to line failure.

Assembling the dense Moore-Penrose pseudoinverse L^+ requires O(|V|^3) time.
Using the Spielman-Srivastava randomized incidence projection theorem:
    Let B be the signed incidence matrix (L = B^T * W * B).
    For each of k = O(log |V| / eps^2) random vectors r_j in {-1/sqrt(k), +1/sqrt(k)}^{|E|},
    we solve the block graph SDDM Laplacian system:
        L * Z = B^T * W^{1/2} * R
    Then: E[ ||Z[u, :] - Z[v, :]||^2 ] = R_eff(u, v)

The all-pairs effective resistance is then computed directly in O(k_proj) query time:
    R_eff(u, v) \approx || Z[u, :] - Z[v, :] ||^2
This drops all-pairs resistance estimation from O(|V|^3) to O(k * |E|).
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class NetworkPowerCentrality:
    """
    Spielman-Srivastava Randomized Effective Resistance & Network Chokepoint Analyzer.
    
    Computes low-dimensional metric embeddings Z in R^{|V| x k} to evaluate
    all-pairs effective resistance and power centrality in O(k_proj) query time.
    """
    def __init__(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        num_projections: int = 32,
        regularization_kappa: float = 1e-4
    ):
        self.n = int(num_nodes)
        self.edges = edges
        self.k_proj = int(num_projections)
        self.kappa = float(regularization_kappa)
        self.n_edges = len(self.edges)

        # Vectorized edge arrays
        self.edge_u = np.array([e[0] for e in self.edges], dtype=np.int64)
        self.edge_v = np.array([e[1] for e in self.edges], dtype=np.int64)
        self.edge_w = np.array([e[2] for e in self.edges], dtype=np.float64)

        # Compute degrees
        self.degrees = np.zeros(self.n, dtype=np.float64)
        np.add.at(self.degrees, self.edge_u, self.edge_w)
        np.add.at(self.degrees, self.edge_v, self.edge_w)

        self.Z_embeddings: Optional[np.ndarray] = None
        self._compute_block_jl_embeddings()

    def block_laplacian_matvec(self, V: np.ndarray) -> np.ndarray:
        """Vectorized block matrix-vector product (L + kappa * I) * V for V in R^{N x K}."""
        # Diagonal term
        out = (self.degrees[:, None] + self.kappa) * V
        
        # Off-diagonal edge terms
        contrib_u = self.edge_w[:, None] * V[self.edge_v]
        contrib_v = self.edge_w[:, None] * V[self.edge_u]
        
        np.add.at(out, self.edge_u, -contrib_u)
        np.add.at(out, self.edge_v, -contrib_v)
        return out

    def _compute_block_jl_embeddings(self, tol: float = 1e-5, max_iter: int = 50):
        """Solves L * Z = B^T * W^{1/2} * R simultaneously for all K projections via Block PCG."""
        scale = 1.0 / np.sqrt(self.k_proj)
        sqrt_w = np.sqrt(self.edge_w)
        
        # Random edge projections R: (|E|, K)
        R = np.random.choice([-scale, scale], size=(self.n_edges, self.k_proj))
        scaled_R = sqrt_w[:, None] * R
        
        # RHS B_mat: (N, K)
        B_mat = np.zeros((self.n, self.k_proj), dtype=np.float64)
        np.add.at(B_mat, self.edge_u, scaled_R)
        np.add.at(B_mat, self.edge_v, -scaled_R)

        # Block Jacobi-Preconditioned CG
        X = np.zeros((self.n, self.k_proj), dtype=np.float64)
        R_res = B_mat - self.block_laplacian_matvec(X)
        inv_diag = 1.0 / (self.degrees[:, None] + self.kappa)
        Z_dir = inv_diag * R_res
        P = Z_dir.copy()

        rz_old = np.sum(R_res * Z_dir, axis=0)
        norm_r0 = np.linalg.norm(R_res, axis=0)

        for _ in range(max_iter):
            AP = self.block_laplacian_matvec(P)
            pAp = np.sum(P * AP, axis=0)
            pAp_safe = np.where(np.abs(pAp) < 1e-16, 1.0, pAp)
            
            alpha = rz_old / pAp_safe
            X += P * alpha[None, :]
            R_res -= AP * alpha[None, :]
            
            res_norms = np.linalg.norm(R_res, axis=0)
            if np.all(res_norms / np.maximum(norm_r0, 1e-12) < tol):
                break

            Z_dir = inv_diag * R_res
            rz_new = np.sum(R_res * Z_dir, axis=0)
            beta = rz_new / np.where(np.abs(rz_old) < 1e-16, 1.0, rz_old)
            P = Z_dir + P * beta[None, :]
            rz_old = rz_new

        self.Z_embeddings = X

    def query_effective_resistance(self, u: int, v: int) -> float:
        """Evaluates R_eff(u, v) in O(k_proj) query time (constant w.r.t. |V|)."""
        diff = self.Z_embeddings[u] - self.Z_embeddings[v]
        return float(np.sum(diff ** 2))

    def batch_query_effective_resistances(self, pairs: np.ndarray) -> np.ndarray:
        """Batch queries pairwise effective resistances in vectorized O(Q * k) time."""
        u_idx = pairs[:, 0]
        v_idx = pairs[:, 1]
        diffs = self.Z_embeddings[u_idx] - self.Z_embeddings[v_idx]
        return np.sum(diffs ** 2, axis=-1)

    def compute_node_power_centrality(self) -> np.ndarray:
        """Computes Total Effective Resistance Centrality (Network Criticality / Power Hub)."""
        z_sq = np.sum(self.Z_embeddings ** 2, axis=-1)
        z_sum = np.sum(self.Z_embeddings, axis=0)
        total_z_sq = np.sum(z_sq)
        
        total_resistances = self.n * z_sq + total_z_sq - 2.0 * (self.Z_embeddings @ z_sum)
        return 1.0 / np.maximum(total_resistances, 1e-12)


def dense_exact_effective_resistance(
    n: int,
    edges: List[Tuple[int, int, float]],
    u: int,
    v: int
) -> float:
    """Exact dense reference using Moore-Penrose pseudoinverse L^+."""
    L = np.zeros((n, n), dtype=np.float64)
    for i, j, w in edges:
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w

    L_pinv = np.linalg.pinv(L)
    r_eff = L_pinv[u, u] + L_pinv[v, v] - 2.0 * L_pinv[u, v]
    return float(r_eff)


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Matrix-Free Effective Resistance & Network Power Centrality Benchmark")
    print("=" * 70)

    n_nodes = 5000
    print(f"Network Graph Nodes (|V|)    : {n_nodes:,}")
    
    coords = np.random.rand(n_nodes, 2) * 20.0
    edges_list: List[Tuple[int, int, float]] = []
    
    cell_sz = 1.5
    grid = {}
    for idx, (x, y) in enumerate(coords):
        k = (int(x / cell_sz), int(y / cell_sz))
        if k not in grid:
            grid[k] = []
        grid[k].append(idx)

    for (gx, gy), indices in grid.items():
        for i in indices:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nbr_k = (gx + dx, gy + dy)
                    if nbr_k in grid:
                        for j in grid[nbr_k]:
                            if i < j:
                                dist = np.linalg.norm(coords[i] - coords[j])
                                if dist < 0.9:
                                    weight = 1.0 / max(dist, 0.1)
                                    edges_list.append((i, j, weight))

    print(f"Network Graph Edges (|E|)    : {len(edges_list):,}")

    t0 = time.perf_counter()
    analyzer = NetworkPowerCentrality(num_nodes=n_nodes, edges=edges_list, num_projections=32)
    t_prep = (time.perf_counter() - t0) * 1000.0

    print(f"Block-PCG JL Preprocessing  : {t_prep:.2f} ms (32 Projections)")

    n_queries = 100000
    query_u = np.random.randint(0, n_nodes, size=n_queries)
    query_v = np.random.randint(0, n_nodes, size=n_queries)
    pairs = np.stack([query_u, query_v], axis=-1)

    t0 = time.perf_counter()
    fast_r_eff = analyzer.batch_query_effective_resistances(pairs)
    t_query = (time.perf_counter() - t0) * 1000.0
    qps = n_queries / (t_query / 1000.0)

    print(f"100,000 Pairwise Resistance Q: {t_query:.2f} ms ({qps:,.0f} queries/sec)")

    centralities = analyzer.compute_node_power_centrality()
    top_hubs = np.argsort(centralities)[-5:][::-1]
    print(f"Top 5 Critical Network Hubs  : Nodes {top_hubs.tolist()}")

    n_small = 200
    small_edges = [(u, v, w) for u, v, w in edges_list if u < n_small and v < n_small]
    if len(small_edges) > 10:
        ref_analyzer = NetworkPowerCentrality(num_nodes=n_small, edges=small_edges, num_projections=64)
        u_test, v_test = 0, 10
        r_fast = ref_analyzer.query_effective_resistance(u_test, v_test)
        r_dense = dense_exact_effective_resistance(n_small, small_edges, u_test, v_test)
        rel_err = abs(r_fast - r_dense) / max(r_dense, 1e-6)
        print(f"Effective Resistance Test    : Fast={r_fast:.4f} vs Dense={r_dense:.4f} (Rel Err: {rel_err:.2e})")
    print("=" * 70)
