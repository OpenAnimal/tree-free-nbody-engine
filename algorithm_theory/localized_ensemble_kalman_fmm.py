"""
Localized Ensemble Kalman Filter (LEnKF) with Spatial Tapering (localized_ensemble_kalman_fmm.py).

Inspired by:
1. "Construction of Correlation Functions in Two and Three Dimensions"
   G. Gaspari and S. E. Cohn (Q. J. R. Meteorol. Soc. 1999).
2. "A Local Ensemble Transform Kalman Filter for Data Assimilation"
   B. R. Hunt, E. J. Kostelich, I. Szunyogh (Physica D: Nonlinear Phenomena, 2007).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Farach-Colton, Krapivin, & Kuszmaul (2025). IEEE FOCS 2024 / arXiv:2501.02305.

Key Algorithmic Principle:
In high-dimensional physical state estimation (atmospheric weather forecasting, oceanography,
geophysical reservoir tracking, massive smart grids), the state dimension N is huge (N = 10^5 to 10^8),
while the ensemble size M is small (M = 20 to 80).
Classical Kalman filtering requires storing and inverting a dense N x N covariance matrix P (O(N^3)).
Standard EnKF suffers from catastrophic spurious long-range sampling correlations due to low ensemble rank.

The Localized Ensemble Kalman Filter (LEnKF) applies a TRUNCATED Gaspari-Cohn
5th-order correlation taper:
    rho(z) = 1 - (5/3)*z^2 + (5/8)*z^3 + (1/2)*z^4 - (1/4)*z^5,  for z = dist/r_loc <= 1
and rho(z) = 0 for z > 1. Note this is the first Gaspari-Cohn piece only (the
full kernel has a second piece on z in [1, 2] that tapers smoothly to 0); the
truncation here produces a DISCONTINUITY at r_loc (rho(1) ~ 0.208, then jumps
to 0), which is a deliberate compact-support approximation, not the smooth
full kernel.
Using a uniform grid hash (dict-based) with cell size r_loc, each state
variable i gathers its local observations in O(1) average time and performs an
independent local subspace Kalman update. The per-state update cost is
O(k_act^2 * M + k_act^3) where k_act is the number of local observations
within r_loc of state i (the local innovation covariance P_yy is k_act x
k_act and is solved directly); this is O(M^2)-class only when k_act = O(M).
The global assimilation cost is therefore O(N * (k_act^2 * M + k_act^3)),
i.e. it depends on the local observation count k_act, not purely on N and M.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class LocalizedEnsembleKalmanFilter:
    """
    Tree-Free Localized Ensemble Kalman Filter (LEnKF) Engine.

    Performs spatial covariance localization. Per-state update cost is
    O(k_act^2 * M + k_act^3) where k_act is the local observation count within
    r_loc; global cost is O(N * (k_act^2 * M + k_act^3)) -- it depends on the
    local observation count k_act, not purely on N and M.
    """
    def __init__(
        self,
        state_coords: np.ndarray,
        localization_radius: float = 2.0,
        obs_noise_variance: float = 0.25
    ):
        self.coords = np.asarray(state_coords, dtype=np.float64)
        self.n_states = len(self.coords)
        self.dim = self.coords.shape[1]
        self.r_loc = float(localization_radius)
        self.sigma_obs2 = float(obs_noise_variance)
        self.cell_size = self.r_loc

    def _gaspari_cohn_weight(self, r: np.ndarray) -> np.ndarray:
        """Evaluates 5th-order Gaspari-Cohn compactly supported correlation polynomial."""
        z = np.clip(r / self.r_loc, 0.0, 1.0)
        # Bounded 5th-order correlation
        w = 1.0 - (5.0 / 3.0) * (z ** 2) + (5.0 / 8.0) * (z ** 3) + 0.5 * (z ** 4) - 0.25 * (z ** 5)
        return np.maximum(w, 0.0)

    def assimilate_observations(
        self,
        prior_ensemble: np.ndarray,
        obs_indices: np.ndarray,
        obs_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs localized ensemble assimilation update.
        
        Args:
            prior_ensemble: (N_state, M_ensemble) prior forecast ensemble
            obs_indices: (K_obs,) indices of observed state variables
            obs_values: (K_obs,) noisy sensor measurements
            
        Returns:
            posterior_ensemble: (N_state, M_ensemble) analysis updated ensemble
            posterior_mean: (N_state,) mean state estimate
        """
        E_prior = np.asarray(prior_ensemble, dtype=np.float64)
        N_state, M_ens = E_prior.shape
        obs_idx = np.asarray(obs_indices, dtype=np.int64)
        y_obs = np.asarray(obs_values, dtype=np.float64)
        n_obs = len(obs_idx)

        # Observation coordinates
        obs_coords = self.coords[obs_idx]

        # Prior mean and ensemble anomalies
        prior_mean = np.mean(E_prior, axis=1)  # (N,)
        A_prior = E_prior - prior_mean[:, None]  # (N, M)

        # Model observation ensemble at sensor points
        HX_ens = E_prior[obs_idx]  # (K, M)
        HX_mean = np.mean(HX_ens, axis=1)  # (K,)
        HA_prior = HX_ens - HX_mean[:, None]  # (K, M)

        # Spatial Hash Partitioning of Observations
        obs_grid = np.floor(obs_coords / self.cell_size).astype(np.int64)
        obs_buckets: Dict[Tuple[int, ...], List[int]] = {}
        for local_k, coord in enumerate(obs_grid):
            key = tuple(coord)
            if key not in obs_buckets:
                obs_buckets[key] = []
            obs_buckets[key].append(local_k)

        from itertools import product
        neighbor_offsets = tuple(product((-1, 0, 1), repeat=self.dim))

        # Hash State Coordinates
        state_grid = np.floor(self.coords / self.cell_size).astype(np.int64)
        state_buckets: Dict[Tuple[int, ...], List[int]] = {}
        for s_idx, coord in enumerate(state_grid):
            key = tuple(coord)
            if key not in state_buckets:
                state_buckets[key] = []
            state_buckets[key].append(s_idx)

        E_posterior = np.copy(E_prior)

        # Localized Block Assimilation
        for cell_k, state_indices in state_buckets.items():
            # Gather nearby observations within adjacent cells
            cand_obs_list = []
            for offset in neighbor_offsets:
                nbr_k = tuple(c + delta for c, delta in zip(cell_k, offset))
                if nbr_k in obs_buckets:
                    cand_obs_list.append(obs_buckets[nbr_k])

            if len(cand_obs_list) == 0:
                continue

            local_obs_idx = np.concatenate(cand_obs_list)
            local_obs_coords = obs_coords[local_obs_idx]
            local_y = y_obs[local_obs_idx]
            local_HA = HA_prior[local_obs_idx]  # (k_loc, M)
            local_hx_mean = HX_mean[local_obs_idx]  # (k_loc,)
            n_loc_obs = len(local_obs_idx)

            for s_idx in state_indices:
                p_s = self.coords[s_idx]
                dist_obs = np.linalg.norm(local_obs_coords - p_s, axis=-1)
                
                mask = dist_obs <= self.r_loc
                if not np.any(mask):
                    continue

                valid_obs = local_obs_idx[mask]
                dists = dist_obs[mask]
                rho_weights = self._gaspari_cohn_weight(dists)

                y_active = y_obs[valid_obs]
                hx_active = HX_mean[valid_obs]
                HA_active = HA_prior[valid_obs]  # (k_act, M)
                k_act = len(valid_obs)

                # Local Innovation Covariance: R_local + (HA * HA^T) / (M - 1)
                P_yy = (HA_active @ HA_active.T) / float(M_ens - 1)
                # Apply Gaspari-Cohn tapering
                P_yy = P_yy * np.outer(rho_weights, rho_weights) + self.sigma_obs2 * np.eye(k_act)

                # Cross-covariance P_xy between single state variable s_idx and observations
                A_s = A_prior[s_idx]  # (M,)
                P_xy = (A_s @ HA_active.T) / float(M_ens - 1)  # (k_act,)
                P_xy = P_xy * rho_weights

                # Kalman gain for state s_idx: K_s = P_xy * P_yy^{-1}
                K_s = np.linalg.solve(P_yy, P_xy)  # (k_act,)

                # Analysis Update: posterior mean = prior_mean + K * (y - H * prior_mean)
                innovation = y_active - hx_active
                update_mean = float(np.dot(K_s, innovation))
                
                # Update posterior ensemble around updated mean.
                # NOTE: the prior anomaly is rescaled by a HARD-CODED 0.85
                # posterior-spread shrink factor (an ad-hoc inflation/damping
                # constant, not derived from the Kalman gain or the ensemble
                # covariance); this deliberately under-spreads the posterior
                # ensemble relative to the prior anomalies.
                E_posterior[s_idx] = prior_mean[s_idx] + update_mean + A_prior[s_idx] * 0.85

        posterior_mean = np.mean(E_posterior, axis=1)
        return E_posterior, posterior_mean


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Localized Ensemble Kalman Filter (LEnKF) Benchmark")
    print("=" * 70)

    n_grid_states = 10000
    m_ensemble = 30
    n_sensors = 500
    print(f"Physical State Grid Size (N) : {n_grid_states:,}")
    print(f"Ensemble Members (M)         : {m_ensemble}")
    print(f"Active Sensor Observations   : {n_sensors}")

    # 2D Spatial Grid Coordinates
    coords_2d = np.random.rand(n_grid_states, 2) * 20.0
    
    # Synthetic true state field: smooth temperature/vorticity waves
    true_state = np.sin(coords_2d[:, 0] * 0.5) * np.cos(coords_2d[:, 1] * 0.5)

    # Prior forecast has smooth structural background displacement error
    forecast_bias = 0.6 * np.cos(coords_2d[:, 0] * 0.3)
    prior_ens = np.zeros((n_grid_states, m_ensemble))
    for m in range(m_ensemble):
        prior_ens[:, m] = true_state + forecast_bias + np.random.randn(n_grid_states) * 0.2

    # Sample random sensor locations
    obs_indices = np.random.choice(n_grid_states, size=n_sensors, replace=False)
    noise_sigma = 0.15
    obs_measurements = true_state[obs_indices] + np.random.randn(n_sensors) * noise_sigma

    filter_engine = LocalizedEnsembleKalmanFilter(
        state_coords=coords_2d,
        localization_radius=2.5,
        obs_noise_variance=noise_sigma**2
    )

    t0 = time.perf_counter()
    post_ens, post_mean = filter_engine.assimilate_observations(
        prior_ensemble=prior_ens,
        obs_indices=obs_indices,
        obs_values=obs_measurements
    )
    t_enkf = (time.perf_counter() - t0) * 1000.0

    prior_rmse = np.sqrt(np.mean((np.mean(prior_ens, axis=1) - true_state) ** 2))
    post_rmse = np.sqrt(np.mean((post_mean - true_state) ** 2))

    print(f"LEnKF Assimilation Runtime   : {t_enkf:.2f} ms")
    print(f"Prior Forecast RMSE          : {prior_rmse:.4f}")
    print(f"Posterior Analysis RMSE      : {post_rmse:.4f} (Error Reduction: {(1.0 - post_rmse/prior_rmse)*100:.1f}%)")
    print("=" * 70)
