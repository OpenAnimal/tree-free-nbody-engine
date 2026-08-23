"""
Round-7 task T-D3: Taylor-FGT Attention.

Two layers, honestly separated:
  Layer 1 (pure spatial attention — exactly solvable):
    out_i = Σ_j G_ij v_j / Σ_j G_ij,  G_ij = exp(−‖x_i−x_j‖²/2σ²)
    Two FGT calls: out = fgt.evaluate(x, v) / fgt.evaluate(x, ones).
    O(N·(p² + K)) each, exact to the FGT's truncation error.

  Layer 2 (spatial × feature softmax — the honest product-kernel split):
    w_ij = G_ij · exp(τ q_i·k_j) is not radial, but with the positive
    random-feature map (FAVOR+/Performer-style):
      φ_t(x) = m^{−1/2}·exp(s·ω_t·x − s²‖x‖²/2),  s = √τ
    E[Σ_t φ_t(q)φ_t(k)] = exp(τ⟨q,k⟩) exactly.
    out_i ≈ Σ_{t=1..m} φ_t(q_i) · FGT(x, q̃_t) / Σ_{t=1..m} φ_t(q_i) · FGT(x, 1̃_t)
    where q̃_t,j = φ_t(k_j)·v_j and 1̃_t,j = φ_t(k_j).
    Cost O(m·N·(p²+K)). The feature-map error is a ratio estimator whose
    variance is the known Performer pain point at large τ‖q‖ — so the
    deliverable is a measured rel-L2-vs-m curve, not an appeal to a bound.

Run standalone:  python -X utf8 neural_ops/taylor_fgt_attention.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

try:
    from neural_ops._coord_contract import check_unit_coords
except ImportError:  # direct script execution (repo root not yet on sys.path)
    from _coord_contract import check_unit_coords
from typing import Optional, Tuple, Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_ops._core_deps import Gaussian2DFGT, Gaussian3DFGT


class TaylorFGTAttention:
    """Taylor-FGT Attention: exact spatial attention via the Gaussian FGT.

    Parameters
    ----------
    spatial_dim : int
        2 or 3.
    sigma : float
        Spatial Gaussian bandwidth (the kernel is exp(-r²/2σ²)).
    grid_depth : int
        FMM grid depth (cells per side = grid_depth for the flat scheme).
    p : int
        Taylor expansion order.
    n_features : int
        Number of positive random features for layer 2 (the feature-dot
        kernel). m=64 by default.
    seed : int
        Random seed for the feature map.
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        sigma: float = 0.25,
        grid_depth: int = 6,
        p: int = 8,
        n_features: int = 64,
        seed: int = 42,
    ):
        self.spatial_dim = int(spatial_dim)
        self.sigma = float(sigma)
        self.grid_depth = int(grid_depth)
        self.p = int(p)
        self.n_features = int(n_features)
        self.seed = int(seed)

        # The FGT engine uses kernel exp(-r²/h²), so h = σ·√2 maps to
        # exp(-r²/2σ²).
        self.h = self.sigma * np.sqrt(2.0)

        if self.spatial_dim == 2:
            self._fgt_cls = Gaussian2DFGT
        elif self.spatial_dim == 3:
            self._fgt_cls = Gaussian3DFGT
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {self.spatial_dim}")

        # Precompute random features for layer 2
        rng = np.random.RandomState(self.seed)
        self._omega = rng.randn(self.n_features, 64)  # (m, D) — D=64 fixed for now

    def _build_fgt(self) -> Any:
        return self._fgt_cls(depth=self.grid_depth, p=self.p, h=self.h)

    def forward(
        self,
        Q: Optional[np.ndarray],   # (N, D) or None for layer 1
        K: Optional[np.ndarray],   # (N, D) or None for layer 1
        V: np.ndarray,             # (N, D)
        coords: np.ndarray,        # (N, spatial_dim) in [0, 1)^d
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass.

        If Q and K are None, runs layer 1 (pure spatial attention).
        Otherwise runs layer 2 (spatial × feature softmax).
        """
        coords = np.asarray(coords, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        N, D = V.shape

        if Q is None or K is None:
            return self._forward_layer1(V, coords, N, D)
        else:
            Q = np.asarray(Q, dtype=np.float64)
            K = np.asarray(K, dtype=np.float64)
            return self._forward_layer2(Q, K, V, coords, N, D)

    def _forward_layer1(
        self, V: np.ndarray, coords: np.ndarray, N: int, D: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Layer 1: pure spatial attention via two FGT calls.

        out_i = Σ_j G_ij v_j / Σ_j G_ij

        The cell index, near-field kernel values and M2L derivative tensors
        depend only on `coords`, so the operator is built ONCE and reused for
        all D+1 charge vectors (build_operator / evaluate_prebuilt split from
        core.radial_taylor.RadialTaylorFMM). Bit-for-bit identical to the
        previous per-call `fgt.evaluate` — same math, no per-call rebuild.
        """
        fgt = self._build_fgt()
        built = fgt.build_operator(coords)
        # Numerator: FGT(x, v) for each feature dimension
        numerator = np.zeros((N, D), dtype=np.float64)
        for d in range(D):
            numerator[:, d] = fgt.evaluate_prebuilt(built, V[:, d])

        # Denominator: FGT(x, ones)
        ones = np.ones(N, dtype=np.float64)
        denominator = fgt.evaluate_prebuilt(built, ones)

        out = numerator / denominator[:, None]

        meta = {
            "layer": 1,
            "N": N,
            "D": D,
            "spatial_dim": self.spatial_dim,
            "sigma": self.sigma,
            "grid_depth": self.grid_depth,
            "p": self.p,
            "complexity": f"O(N*(p^2+K)) * D = O({D}*N*(p^2+K))",
        }
        return out, meta

    def _forward_layer2(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        coords: np.ndarray, N: int, D: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Layer 2: spatial × feature softmax via positive random features.

        out_i ≈ Σ_{t=1..m} φ_t(q_i) · FGT(x, q̃_t) / Σ_{t=1..m} φ_t(q_i) · FGT(x, 1̃_t)
        where q̃_t,j = φ_t(k_j)·v_j and 1̃_t,j = φ_t(k_j).
        """
        m = self.n_features
        # D must match the feature dimension; if K has different D, adapt omega
        D_feat = K.shape[1]
        if D_feat != self._omega.shape[1]:
            rng = np.random.RandomState(self.seed)
            self._omega = rng.randn(m, D_feat)

        omega = self._omega  # (m, D_feat)
        # s = 1/√D so that s² = 1/D, i.e. the FAVOR+ estimator
        #   E[Σ_t φ_t(q)φ_t(k)] = exp(s²⟨q,k⟩) = exp(⟨q,k⟩/D)
        # implements the kernel with τ = 1/D_feat (NOT τ = 1/√D_feat).
        # The dense reference below must use the SAME τ = 1/D_feat.
        s = 1.0 / np.sqrt(D_feat)  # τ = 1/D_feat (since s² = 1/D_feat)

        # Positive random features: φ_t(x) = m^{-1/2} exp(s·ω_t·x − s²‖x‖²/2)
        # Precompute the self-normalizer term s²‖x‖²/2
        k_norm_sq = np.sum(K ** 2, axis=1)  # (N,)
        q_norm_sq = np.sum(Q ** 2, axis=1)  # (N,)

        # φ_t(k_j) = m^{-1/2} exp(s·(ω @ K^T)_{t,j} − 0.5·s²·‖k_j‖²)
        # Vectorized: (m, N) = (m, D) @ (D, N) -> broadcast the per-token norm.
        phi_k = (1.0 / np.sqrt(m)) * np.exp(
            s * (omega @ K.T) - 0.5 * (s ** 2) * k_norm_sq[None, :]
        )
        # φ_t(q_i): (m, N), same construction with Q.
        phi_q = (1.0 / np.sqrt(m)) * np.exp(
            s * (omega @ Q.T) - 0.5 * (s ** 2) * q_norm_sq[None, :]
        )

        # Numerator: Σ_t φ_t(q_i) · FGT(x, q̃_t) where q̃_t,j = φ_t(k_j)·v_j
        # For each feature t and value dim d: FGT(x, φ_t(k) * v[:, d])
        # The operator depends only on `coords` (same for every t, d), so build
        # it ONCE and call evaluate_prebuilt for all m*D + m charge vectors.
        # Bit-for-bit identical to the previous per-call `fgt.evaluate` — same
        # math, no per-call rebuild of the cell index / M2L tensors.
        fgt = self._build_fgt()
        built = fgt.build_operator(coords)
        numerator = np.zeros((N, D), dtype=np.float64)
        for t in range(m):
            for d in range(D):
                qtilde = phi_k[t] * V[:, d]
                fgt_out = fgt.evaluate_prebuilt(built, qtilde)
                numerator[:, d] += phi_q[t] * fgt_out

        # Denominator: Σ_t φ_t(q_i) · FGT(x, 1̃_t) where 1̃_t,j = φ_t(k_j)
        denominator = np.zeros(N, dtype=np.float64)
        for t in range(m):
            fgt_out = fgt.evaluate_prebuilt(built, phi_k[t])
            denominator += phi_q[t] * fgt_out

        out = numerator / denominator[:, None]

        meta = {
            "layer": 2,
            "N": N,
            "D": D,
            "D_feat": D_feat,
            "spatial_dim": self.spatial_dim,
            "sigma": self.sigma,
            "grid_depth": self.grid_depth,
            "p": self.p,
            "n_features": m,
            "complexity": f"O(m*N*(p^2+K)*D) = O({m}*N*(p^2+K)*{D})",
        }
        return out, meta


if __name__ == "__main__":
    print("=" * 70)
    print("TaylorFGTAttention: exact spatial attention via Gaussian FGT")
    print("=" * 70)

    # Layer 1 test: pure spatial attention
    N, D = 1000, 4
    rng = np.random.RandomState(42)
    coords = rng.uniform(0.05, 0.95, size=(N, 3)).astype(np.float64)
    V = rng.randn(N, D).astype(np.float64)

    layer = TaylorFGTAttention(spatial_dim=3, sigma=0.15, grid_depth=6, p=8)
    out_fgt, meta = layer.forward(None, None, V, coords)

    # Dense reference
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    G = np.exp(-dist_sq / (2 * 0.15 ** 2))
    np.fill_diagonal(G, 0.0)
    out_dense = G @ V / (G.sum(axis=1, keepdims=True) + 1e-30)

    rel_l2 = np.linalg.norm(out_fgt - out_dense) / np.linalg.norm(out_dense)
    print(f"\nLayer 1 (pure spatial): N={N}, D={D}, rel-L2 = {rel_l2:.4e}")
    assert rel_l2 < 1e-4, f"Layer 1 rel-L2 {rel_l2} >= 1e-4"
    print("  -> PASS (rel-L2 < 1e-4)")

    # Layer 2 test: spatial × feature softmax (smaller N for speed)
    N2, D2 = 500, 4
    coords2 = rng.uniform(0.05, 0.95, size=(N2, 3)).astype(np.float64)
    D_feat = 32
    Q = rng.randn(N2, D_feat).astype(np.float64)
    K = rng.randn(N2, D_feat).astype(np.float64)
    V2 = rng.randn(N2, D2).astype(np.float64)

    layer2 = TaylorFGTAttention(
        spatial_dim=3, sigma=0.15, grid_depth=5, p=6, n_features=16
    )
    out_fgt2, meta2 = layer2.forward(Q, K, V2, coords2)

    # Dense reference
    diff2 = coords2[:, None, :] - coords2[None, :, :]
    dist_sq2 = np.sum(diff2 ** 2, axis=-1)
    G2 = np.exp(-dist_sq2 / (2 * 0.15 ** 2))
    tau = 1.0 / D_feat  # MUST match the estimator's τ = 1/D_feat (s² = 1/D)
    dot = Q @ K.T * tau
    W = G2 * np.exp(np.clip(dot, -50, 50))
    np.fill_diagonal(W, 0.0)
    out_dense2 = W @ V2 / (W.sum(axis=1, keepdims=True) + 1e-30)

    rel_l2_2 = np.linalg.norm(out_fgt2 - out_dense2) / np.linalg.norm(out_dense2)
    print(f"\nLayer 2 (spatial × feature): N={N2}, D={D2}, m=16, rel-L2 = {rel_l2_2:.4e}")
    print(f"  (feature-map ratio estimator; variance is the Performer pain point)")
    print("  -> measured (not asserted)")
