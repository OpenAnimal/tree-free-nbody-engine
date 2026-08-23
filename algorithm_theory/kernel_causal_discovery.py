"""
Fast Hilbert-Schmidt Kernel Independence & Causal Direction Discovery (kernel_causal_discovery.py).

Inspired by:
1. "Measuring Statistical Dependence with Hilbert-Schmidt Norms"
   A. Gretton, O. Bousquet, A. Smola, B. Scholkopf (ALT 2005).
2. "Kernel-based Conditional Independence Test and Application in Causal Discovery"
   K. Zhang, J. Peters, D. Janzing, B. Scholkopf (UAI 2011).
3. "Random Features for Large-Scale Kernel Machines"
   Ali Rahimi and Benjamin Recht (NeurIPS 2007).
4. "Optimal Bounds for Open Addressing Without Reordering"
   Farach-Colton, Krapivin, & Kuszmaul (2025). IEEE FOCS 2024 / arXiv:2501.02305.

Key Algorithmic Principle:
In non-linear causal discovery and structural equation modeling, testing conditional independence
(X _||_ Y | Z) without linear Gaussian assumptions requires evaluating the Hilbert-Schmidt Independence
Criterion (HSIC) in Reproducing Kernel Hilbert Spaces (RKHS):
    HSIC(X, Y) = (1 / N^2) * Tr(K_X * H * K_Y * H)

Evaluating exact dense Gram matrices K_X and K_Y requires quadratic O(N^2) time and memory.
Using Random Fourier Feature (RFF) approximations:
    phi(x) = sqrt(2 / D_rff) * cos(W * x + b)
the infinite-dimensional kernel inner product is mapped to a low-dimensional Euclidean embedding:
    K_X \approx Phi_X * Phi_X^T
This reduces HSIC dependence computation to O(N * D_rff) linear time, enabling rapid
causal direction inference (e.g. distinguishing X -> Y from Y -> X via Additive Noise Models).
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class FastKernelCausalDiscovery:
    """
    Random Fourier Feature (RFF) Accelerated HSIC & Causal Direction Finder.
    
    Computes non-linear statistical independence and infers causal direction in O(N * D_rff) time.
    """
    def __init__(
        self,
        num_random_features: int = 48,
        rbf_gamma: float = 1.0,
        random_seed: int = 42
    ):
        self.D_rff = int(num_random_features)
        self.gamma = float(rbf_gamma)
        self.seed = int(random_seed)
        if self.D_rff < 2 or self.D_rff % 2 != 0:
            raise ValueError("num_random_features must be an even integer >= 2")
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("rbf_gamma must be finite and positive")

    def _generate_rff_projection(self, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates random Gaussian frequency weights W and uniform phase shifts b."""
        rng = np.random.RandomState(self.seed)
        # For Gaussian RBF kernel exp(-gamma * ||x - y||^2): W ~ N(0, 2 * gamma * I)
        W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(input_dim, self.D_rff))
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.D_rff)
        return W, b

    def compute_rff_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Maps continuous variables X in R^{N x D} to explicit feature map Phi(X) in R^{N x D_rff}:
            Phi(x) = sqrt(2 / D_rff) * cos(X * W + b)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        N, D = X.shape

        W, b = self._generate_rff_projection(D)
        projection = X @ W + b[None, :]
        phi = np.sqrt(2.0 / self.D_rff) * np.cos(projection)
        return phi

    def compute_fast_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes non-linear statistical dependence HSIC(X, Y) in O(N * D_rff) time.
        
        HSIC(X, Y) \approx (1 / N^2) * || Phi_X_centered^T * Phi_Y_centered ||_F^2
        """
        Phi_X = self.compute_rff_embedding(X)
        Phi_Y = self.compute_rff_embedding(Y)
        if len(Phi_X) != len(Phi_Y) or len(Phi_X) < 2:
            raise ValueError("X and Y must contain the same number of samples (>= 2)")
        N = len(Phi_X)

        # Center feature maps: H * Phi = Phi - mean(Phi)
        Phi_Xc = Phi_X - np.mean(Phi_X, axis=0, keepdims=True)
        Phi_Yc = Phi_Y - np.mean(Phi_Y, axis=0, keepdims=True)

        # Cross-covariance operator C_XY in R^{D_rff x D_rff}
        C_XY = (Phi_Xc.T @ Phi_Yc) / float(N)

        # HSIC is Frobenius norm squared of cross-covariance operator
        hsic_val = float(np.sum(C_XY ** 2))
        return hsic_val

    def infer_causal_direction(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        poly_degree: int = 3
    ) -> Dict[str, float]:
        """
        Infers causal arrow direction between X and Y using Additive Noise Models (ANM):
        Hypothesis 1 (X -> Y): Y = f(X) + N_Y  ==>  N_Y _||_ X  (low HSIC(X, residual_Y))
        Hypothesis 2 (Y -> X): X = g(Y) + N_X  ==>  N_X _||_ Y  (low HSIC(Y, residual_X))
        
        Returns:
            result: HSIC scores and inferred causal direction
        """
        x = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(Y, dtype=np.float64).ravel()
        if len(x) != len(y) or len(x) < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("X and Y must be finite vectors of equal length >= 3")
        poly_degree = int(poly_degree)
        if poly_degree < 1 or poly_degree >= len(x):
            raise ValueError("poly_degree must be in [1, n_samples - 1]")

        # Fit non-linear regression Y ~ f(X)
        poly_x = np.polyfit(x, y, deg=poly_degree)
        f_x = np.polyval(poly_x, x)
        residual_y = y - f_x

        # Fit non-linear regression X ~ g(Y)
        poly_y = np.polyfit(y, x, deg=poly_degree)
        g_y = np.polyval(poly_y, y)
        residual_x = x - g_y

        # Compute independence scores
        hsic_forward = self.compute_fast_hsic(x, residual_y)   # Score for X -> Y
        hsic_backward = self.compute_fast_hsic(y, residual_x)  # Score for Y -> X

        inferred_direction = "X -> Y" if hsic_forward < hsic_backward else "Y -> X"
        confidence_ratio = float(max(hsic_forward, hsic_backward) / max(min(hsic_forward, hsic_backward), 1e-12))

        return {
            "inferred_direction": inferred_direction,
            "hsic_X_to_Y": float(hsic_forward),
            "hsic_Y_to_X": float(hsic_backward),
            "confidence_ratio": confidence_ratio
        }


def direct_dense_hsic(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Exact dense O(N^2) reference HSIC computation."""
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    N = len(x)

    diff_x = x[:, None, :] - x[None, :, :]
    K_x = np.exp(-gamma * np.sum(diff_x ** 2, axis=-1))

    diff_y = y[:, None, :] - y[None, :, :]
    K_y = np.exp(-gamma * np.sum(diff_y ** 2, axis=-1))

    # Centering matrix H = I - 1/N
    H = np.eye(N) - (1.0 / N)
    K_xc = H @ K_x @ H
    K_yc = H @ K_y @ H

    return float(np.trace(K_xc @ K_yc) / (N ** 2))


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Fast Hilbert-Schmidt Independence & Causal Direction Benchmark")
    print("=" * 70)

    n_samples = 15000
    print(f"Number of Observational Samples: {n_samples:,}")

    # Ground truth causal process: X -> Y
    # X ~ Uniform(-2, 2)
    # Y = X^3 - 2*X + Noise
    X_true = np.random.uniform(-2.0, 2.0, size=n_samples)
    noise_Y = np.random.randn(n_samples) * 0.4
    Y_true = X_true**3 - 1.5 * X_true + noise_Y

    causal_engine = FastKernelCausalDiscovery(num_random_features=64, rbf_gamma=0.5)

    # 1. Fast RFF HSIC Computation
    t0 = time.perf_counter()
    hsic_fast = causal_engine.compute_fast_hsic(X_true, Y_true)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Fast RFF HSIC Runtime        : {t_fast:.2f} ms")
    print(f"Estimated HSIC Dependency    : {hsic_fast:.6f}")

    # 2. Dense Reference HSIC on subset
    n_sub = 1500
    t0 = time.perf_counter()
    hsic_dense_sub = direct_dense_hsic(X_true[:n_sub], Y_true[:n_sub], gamma=0.5)
    t_dense_sub = (time.perf_counter() - t0) * 1000.0
    t_dense_proj = t_dense_sub * ((n_samples * n_samples) / (n_sub * n_sub))

    print(f"Projected Dense O(N^2) Time  : {t_dense_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_dense_proj / max(t_fast, 1e-6):.1f}x")

    # 3. Infer Causal Arrow Direction
    t0 = time.perf_counter()
    causal_report = causal_engine.infer_causal_direction(X_true, Y_true)
    t_anm = (time.perf_counter() - t0) * 1000.0

    print(f"Causal Arrow Inference Time  : {t_anm:.2f} ms")
    print(f"Inferred Causal Direction    : {causal_report['inferred_direction']} (Ground Truth: X -> Y)")
    print(f"Forward Model HSIC (X -> Y)  : {causal_report['hsic_X_to_Y']:.6f} (Lower is better)")
    print(f"Backward Model HSIC (Y -> X) : {causal_report['hsic_Y_to_X']:.6f}")
    print(f"Causal Confidence Margin     : {causal_report['confidence_ratio']:.1f}x")
    print("=" * 70)
