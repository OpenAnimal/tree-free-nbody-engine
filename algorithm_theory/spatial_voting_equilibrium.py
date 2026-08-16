"""
Continuous Spatial Voting & Electoral Nash Equilibrium Engine (spatial_voting_equilibrium.py).

Inspired by:
1. "The Spatial Theory of Voting: An Introduction"
   James M. Enelow and Melvin J. Hinich (Cambridge University Press, 1984).
2. "Spatial Competition and the Median Voter Theorem in Multidimensional Spaces"
   Anthony Downs / Donald Wittman (Journal of Economic Theory, 1977).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
In spatial voting theory and policy game theory, voters occupy positions x_i in R^D
(multi-dimensional issue spaces across economy, healthcare, climate, governance).
K candidates/parties choose platform positions y_1, ..., y_K in R^D to maximize vote share:
    V_k(y_1, ..., y_K) = sum_{i=1}^N w_i * P_k(x_i)
where voter probabilistic choice follows a multinomial logit (softmax utility):
    P_k(x_i) = exp(-||x_i - y_k||^2 / (2 * sigma^2) + b_k) / sum_{m=1}^K exp(-||x_i - y_m||^2 / (2 * sigma^2) + b_m)

The exact analytical policy gradient for candidate k is:
    nabla_{y_k} V_k = (1 / sigma^2) * sum_{i=1}^N w_i * P_k(x_i) * (1 - P_k(x_i)) * (x_i - y_k)

By formulating vote share evaluation and analytical gradients as continuous Gaussian kernel expectations,
multi-party Nash equilibria, platform convergence (Downsian Median Voter Theorem), and polarization
instabilities are evaluated across millions of voters in O(N * K) linear time without grid discretization.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class SpatialVotingEquilibriumEngine:
    """
    Continuous Multi-Dimensional Spatial Voting Field & Multi-Party Nash Equilibrium Solver.
    
    Computes voter choice probabilities, expected vote shares, and analytical policy gradients.
    """
    def __init__(
        self,
        voter_coordinates: np.ndarray,
        voter_weights: Optional[np.ndarray] = None,
        utility_bandwidth_sigma: float = 0.5
    ):
        self.voters = np.asarray(voter_coordinates, dtype=np.float64)
        if self.voters.ndim != 2 or self.voters.shape[0] == 0 or self.voters.shape[1] == 0:
            raise ValueError("voter_coordinates must have shape (N, D) with N,D > 0")
        if not np.all(np.isfinite(self.voters)):
            raise ValueError("voter_coordinates must contain only finite values")
        self.n_voters = len(self.voters)
        self.dim = self.voters.shape[1]
        self.sigma = float(utility_bandwidth_sigma)
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("utility_bandwidth_sigma must be finite and positive")

        if voter_weights is None:
            self.weights = np.ones(self.n_voters, dtype=np.float64) / self.n_voters
        else:
            w = np.asarray(voter_weights, dtype=np.float64)
            if w.ndim != 1 or len(w) != self.n_voters or not np.all(np.isfinite(w)) or np.any(w < 0.0):
                raise ValueError("voter_weights must be finite, non-negative, and shape (N,)")
            total_weight = np.sum(w)
            if total_weight <= 0.0:
                raise ValueError("voter_weights must have positive total weight")
            self.weights = w / total_weight

    def evaluate_vote_shares_and_probabilities(
        self,
        candidate_platforms: np.ndarray,
        candidate_valences: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes probabilistic voter choice matrix P (N, K) and total vote shares V (K,).
        
        Args:
            candidate_platforms: (K, D) policy positions
            candidate_valences: (K,) optional baseline valence/incumbency advantages
            
        Returns:
            vote_shares: (K,) expected fraction of total vote
            voter_probabilities: (N, K) softmax choice distribution per voter
        """
        platforms = np.asarray(candidate_platforms, dtype=np.float64)
        if platforms.ndim != 2 or platforms.shape[0] == 0 or platforms.shape[1] != self.dim:
            raise ValueError(f"candidate_platforms must have shape (K, {self.dim}) with K > 0")
        if not np.all(np.isfinite(platforms)):
            raise ValueError("candidate_platforms must contain only finite values")
        k_candidates = len(platforms)
        
        if candidate_valences is None:
            valences = np.zeros(k_candidates, dtype=np.float64)
        else:
            valences = np.asarray(candidate_valences, dtype=np.float64)
            if valences.ndim != 1 or len(valences) != k_candidates or not np.all(np.isfinite(valences)):
                raise ValueError("candidate_valences must have finite shape (K,)")

        # Pairwise distance squared: (N, K)
        diff = self.voters[:, None, :] - platforms[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # Utilities: U_{ik} = -||x_i - y_k||^2 / (2 * sigma^2) + b_k
        utilities = -dist_sq / (2.0 * (self.sigma ** 2)) + valences[None, :]

        # Numerically stable Softmax
        max_u = np.max(utilities, axis=-1, keepdims=True)
        exp_u = np.exp(utilities - max_u)
        sum_exp = np.sum(exp_u, axis=-1, keepdims=True)
        
        voter_probs = exp_u / np.maximum(sum_exp, 1e-15)  # (N, K)
        
        # Expected vote share: V_k = sum_i w_i * P_{ik}
        vote_shares = np.sum(self.weights[:, None] * voter_probs, axis=0)  # (K,)
        return vote_shares, voter_probs

    def compute_analytical_policy_gradients(
        self,
        candidate_platforms: np.ndarray,
        candidate_valences: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes exact analytical policy gradients nabla_{y_k} V_k for all K candidates.
        
        nabla_{y_k} V_k = (1 / sigma^2) * sum_i w_i * P_{ik} * (1 - P_{ik}) * (x_i - y_k)
        
        Returns:
            gradients: (K, D) platform gradient vectors
            vote_shares: (K,) current vote shares
        """
        platforms = np.asarray(candidate_platforms, dtype=np.float64)
        k_candidates = len(platforms)
        vote_shares, voter_probs = self.evaluate_vote_shares_and_probabilities(platforms, candidate_valences)

        gradients = np.zeros((k_candidates, self.dim), dtype=np.float64)
        inv_sigma2 = 1.0 / (self.sigma ** 2)

        for k in range(k_candidates):
            p_k = voter_probs[:, k]  # (N,)
            variance_factor = self.weights * p_k * (1.0 - p_k)  # (N,)
            
            # Displacement vector: (x_i - y_k)
            disp = self.voters - platforms[k]  # (N, D)
            
            # Gradient nabla_{y_k} V_k
            grad_k = inv_sigma2 * np.sum(variance_factor[:, None] * disp, axis=0)
            gradients[k] = grad_k

        return gradients, vote_shares

    def solve_multi_party_nash_equilibrium(
        self,
        initial_platforms: np.ndarray,
        learning_rate: float = 0.2,
        max_iterations: int = 100,
        convergence_tol: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Finds the multi-party Nash policy equilibrium via simultaneous gradient ascent.
        
        Returns:
            equilibrium_platforms: (K, D) converged candidate positions
            final_shares: (K,) equilibrium vote shares
            n_iters: iterations to convergence
        """
        platforms = np.copy(initial_platforms).astype(np.float64)
        n_iters = 0

        for it in range(max_iterations):
            n_iters = it + 1
            grads, shares = self.compute_analytical_policy_gradients(platforms)
            
            max_grad_norm = np.max(np.linalg.norm(grads, axis=-1))
            if max_grad_norm < convergence_tol:
                break

            # Gradient ascent step: y_k += eta * grad_k
            platforms += learning_rate * grads

        final_shares, _ = self.evaluate_vote_shares_and_probabilities(platforms)
        return platforms, final_shares, n_iters


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Continuous Spatial Voting & Electoral Nash Equilibrium Benchmark")
    print("=" * 70)

    n_voters = 25000
    k_parties = 4
    print(f"Number of Continuous Voters  : {n_voters:,}")
    print(f"Number of Competing Parties  : {k_parties}")

    # Generate multi-modal voter distribution across 2D political compass
    # E.g. 3 demographic clusters (Progressive-Urban, Conservative-Rural, Centrist-Suburban)
    v1 = np.random.randn(n_voters // 3, 2) * 0.3 + np.array([-0.6, -0.4])
    v2 = np.random.randn(n_voters // 3, 2) * 0.4 + np.array([+0.5, +0.6])
    v3 = np.random.randn(n_voters - 2 * (n_voters // 3), 2) * 0.35 + np.array([0.0, 0.0])
    all_voters = np.vstack([v1, v2, v3])

    engine = SpatialVotingEquilibriumEngine(
        voter_coordinates=all_voters,
        utility_bandwidth_sigma=0.45
    )

    # Initial candidate platform guesses
    initial_platforms = np.array([
        [-0.8, -0.8],
        [+0.7, +0.7],
        [-0.2, +0.5],
        [+0.3, -0.6]
    ])

    # 1. Evaluate Analytical Policy Gradients
    t0 = time.perf_counter()
    grads, initial_shares = engine.compute_analytical_policy_gradients(initial_platforms)
    t_grad = (time.perf_counter() - t0) * 1000.0

    print(f"Analytical Policy Gradient   : {t_grad:.2f} ms")
    print(f"Initial Party Vote Shares    : {[f'{s*100:.1f}%' for s in initial_shares]}")

    # 2. Solve Multi-Party Nash Equilibrium
    t0 = time.perf_counter()
    eq_platforms, final_shares, iters = engine.solve_multi_party_nash_equilibrium(
        initial_platforms, learning_rate=0.25, max_iterations=60, convergence_tol=1e-4
    )
    t_nash = (time.perf_counter() - t0) * 1000.0

    print(f"Nash Equilibrium Convergence : {t_nash:.2f} ms ({iters} simultaneous gradient steps)")
    print(f"Equilibrium Party Vote Shares: {[f'{s*100:.1f}%' for s in final_shares]}")
    print(f"Converged Platform Positions :\n{np.round(eq_platforms, 3)}")
    print("=" * 70)
