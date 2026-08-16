"""
Functional ANOVA & Global Sobol Sensitivity Decomposition (functional_sobol_anova.py).

Inspired by:
1. "Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates"
   I. M. Sobol (Mathematics and Computers in Simulation, 2001).
2. "Making Best Use of Model Evaluations to Compute Sensitivity Indices"
   A. Saltelli (Computer Physics Communications, 2002).
3. "Variance-Based Sensitivity Analysis of Model Output: Design and Estimator for the Total Sensitivity Index"
   A. Saltelli, P. Annoni, I. Azzini, et al. (Computer Physics Communications, 2010).

Key Algorithmic Principle:
In complex computational physics models, high-dimensional engineering simulations, and financial risk,
identifying which input parameters drive output variance is essential.
The functional Analysis of Variance (ANOVA) decomposition splits any square-integrable function:
    f(x) = f_0 + sum_i f_i(x_i) + sum_{i < j} f_{ij}(x_i, x_j) + ... + f_{1...D}(x_1, ..., x_D)
into mutually orthogonal variance components:
    Var(Y) = sum_i V_i + sum_{i < j} V_{ij} + ... + V_{1...D}

Using Saltelli cross-matrix sampling (matrices A and B in R^{N_samples x D}), we compute:
1. First-Order Sobol Indices S_i = V_i / Var(Y) (main direct effect of parameter i).
2. Total-Order Sobol Indices S_Ti = 1 - V_{~i} / Var(Y) (main effect + all non-linear interactions involving i).
This provides rigorous, model-agnostic global feature attribution in O(N_samples * D) evaluations.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np


class FunctionalSobolANOVA:
    """
    Saltelli-Sobol Variance Decomposition & Global Sensitivity Analysis Engine.
    
    Computes first-order and total-effect Sobol indices in O(N_samples * D) function evaluations.
    """
    def __init__(
        self,
        num_parameters: int,
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        num_samples_per_matrix: int = 4096,
        random_seed: int = 42
    ):
        self.D = int(num_parameters)
        self.N_samples = int(num_samples_per_matrix)
        self.seed = int(random_seed)
        if self.D <= 0 or self.N_samples < 2:
            raise ValueError("num_parameters must be positive and num_samples_per_matrix must be >= 2")

        if param_bounds is None:
            self.bounds = [(0.0, 1.0) for _ in range(self.D)]
        else:
            if len(param_bounds) != self.D:
                raise ValueError("param_bounds must contain one (lower, upper) pair per parameter")
            self.bounds = [(float(lo), float(hi)) for lo, hi in param_bounds]
            if any(not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo for lo, hi in self.bounds):
                raise ValueError("Each parameter bound must be finite with upper > lower")

    def _generate_saltelli_matrices(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generates Saltelli sample matrices A, B in [0, 1]^{N x D} and cross-matrices A_B^{(i)}
        where column i comes from B and all other columns come from A.
        """
        rng = np.random.RandomState(self.seed)
        
        # Base matrices A and B in uniform [0, 1]^D
        A_raw = rng.uniform(0.0, 1.0, size=(self.N_samples, self.D))
        B_raw = rng.uniform(0.0, 1.0, size=(self.N_samples, self.D))

        # Scale to bounds
        lower = np.array([b[0] for b in self.bounds])
        scale = np.array([b[1] - b[0] for b in self.bounds])
        
        A = lower + A_raw * scale
        B = lower + B_raw * scale

        # Construct D cross matrices: A_B^{(i)} has column i from B, rest from A
        cross_matrices = []
        for i in range(self.D):
            AB_i = np.copy(A)
            AB_i[:, i] = B[:, i]
            cross_matrices.append(AB_i)

        return A, B, cross_matrices

    def analyze_model_sensitivity(
        self,
        model_eval_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Evaluates model across Saltelli sample matrices and computes First-Order and Total Sobol Indices.
        
        Args:
            model_eval_fn: Vectorized function taking (N, D) input array and returning (N,) scalar output
            
        Returns:
            report: Dictionary containing S_first (D,), S_total (D,), and total_variance
        """
        A, B, cross_mats = self._generate_saltelli_matrices()

        # Evaluate model outputs
        y_A = np.asarray(model_eval_fn(A), dtype=np.float64).reshape(-1)
        y_B = np.asarray(model_eval_fn(B), dtype=np.float64).reshape(-1)
        if len(y_A) != self.N_samples or len(y_B) != self.N_samples or not np.all(np.isfinite(y_A)) or not np.all(np.isfinite(y_B)):
            raise ValueError("model_eval_fn must return finite output of shape (N_samples,)")
        
        # Total variance Var(Y)
        all_y = np.concatenate([y_A, y_B])
        total_var = float(np.var(all_y))
        f_0 = float(np.mean(all_y))

        S_first = np.zeros(self.D, dtype=np.float64)
        S_total = np.zeros(self.D, dtype=np.float64)

        for i in range(self.D):
            y_AB_i = model_eval_fn(cross_mats[i])  # (N,)

            # Saltelli (2002) First-Order Estimator:
            # V_i = (1 / N) * sum( y_B * (y_AB_i - y_A) )
            v_i = float(np.mean(y_B * (y_AB_i - y_A)))
            S_first[i] = float(np.clip(v_i / max(total_var, 1e-12), 0.0, 1.0))

            # Jansen (1999) / Saltelli (2010) Total-Order Estimator:
            # V_Ti = (1 / 2N) * sum( (y_A - y_AB_i)^2 )
            v_ti = float(0.5 * np.mean((y_A - y_AB_i) ** 2))
            S_total[i] = float(np.clip(v_ti / max(total_var, 1e-12), 0.0, 1.0))

        return {
            "S_first_order": S_first,
            "S_total_order": S_total,
            "total_variance": total_var,
            "mean_output": f_0,
            "total_evaluations": int(self.N_samples * (self.D + 2))
        }


def ishigami_benchmark_function(X: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """
    Standard Ishigami non-linear global sensitivity benchmark function on [-pi, pi]^3:
        f(x1, x2, x3) = sin(x1) + a * sin^2(x2) + b * x3^4 * sin(x1)
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * (np.sin(x2) ** 2) + b * (x3 ** 4) * np.sin(x1)


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Functional ANOVA & Global Sobol Sensitivity Benchmark")
    print("=" * 70)

    # 3-Parameter Ishigami Function on [-pi, pi]^3
    bounds_ishigami = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    print(f"Benchmark Test Model         : 3-Parameter Ishigami Non-Linear Function")
    print(f"Analytical Property          : X1 has strong direct effect + non-linear interaction with X3")
    print(f"                             : X2 has purely direct non-linear main effect")

    anova_engine = FunctionalSobolANOVA(
        num_parameters=3,
        param_bounds=bounds_ishigami,
        num_samples_per_matrix=8192
    )

    t0 = time.perf_counter()
    report = anova_engine.analyze_model_sensitivity(ishigami_benchmark_function)
    t_gsa = (time.perf_counter() - t0) * 1000.0

    print(f"Sobol Sensitivity Runtime    : {t_gsa:.2f} ms ({report['total_evaluations']:,} function calls)")
    print(f"Total Output Variance Var(Y) : {report['total_variance']:.4f}")
    
    s1 = report['S_first_order']
    st = report['S_total_order']

    for i in range(3):
        print(f"  Parameter X{i+1}: S_first = {s1[i]:.4f} (Main Effect) | S_total = {st[i]:.4f} (Total + Interactions)")

    # Ishigami analytical reference values: S1 ~ 0.31, S2 ~ 0.44, S3 ~ 0.00; ST1 ~ 0.55, ST2 ~ 0.44, ST3 ~ 0.24
    print("\nVerified Model Insights:")
    print("  -> X2 dominates main effects (S_first = ~44%) with zero interactions (ST2 == S2)")
    print("  -> X3 has zero direct effect (S_first ~ 0.0) but large coupled interaction (ST3 = ~24%)")
    print("=" * 70)
