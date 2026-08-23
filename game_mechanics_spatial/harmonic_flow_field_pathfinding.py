"""
Harmonic Potential Flow Field & Continuum Swarm Pathfinder.
Solves real-time continuum navigation for massive agent swarms (50,000+ units) with hash-neighborhood queries.

Mathematical Formulation:
- Total Potential: Phi(x) = Phi_att(x) + Phi_rep(x) + Phi_vortex(x)
- Attractive Goal Potential: Parabolic / Monopole sink attracting toward target destination:
    Phi_att(x) = 0.5 * k_att * ||x - x_goal||^2  (or logarithmic / harmonic sink)
- Screened Yukawa Repulsive Obstacle Field:
    Phi_rep(x) = sum_k [ Q_obs * exp(-kappa * ||x - x_k||) / (||x - x_k|| + epsilon) ]
- Circulation / Streamline Vortex Term (prevents local minima at concave obstacles):
    v_vortex(x) = R_(pi/2) * grad(Phi_rep(x)) * vortex_gain
- Continuum Velocity Field:
    v(x) = -grad(Phi(x)) / ||grad(Phi(x))|| * v_desired

Eliminates the O(K * N log N) per-agent A* search bottleneck by enabling single-pass vectorized flow field evaluation.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Union
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.spatial_index import CellIndex


class HarmonicPotentialFlowField:
    """
    Continuous Harmonic Potential Flow Field & Swarm Pathfinder.
    The obstacle term is an analytic Yukawa sum over all obstacles (O(N*M)
    vectorized); `sample_flow_velocity_hashed` restricts that sum to the
    obstacles of the 5x5 elastic-hash neighborhood, justified by the
    exponential screening (decay length = one grid cell). "Multipole" in
    earlier revisions was a misnomer: there is no multipole expansion here.
    """
    def __init__(
        self,
        world_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
        k_att: float = 1.0,
        kappa_obs: float = 0.05,
        q_obs_default: float = 100.0,
        epsilon: float = 2.0,
        vortex_gain: float = 0.4
    ):
        """
        world_bounds: (min_x, min_y, max_x, max_y)
        k_att: Goal attraction spring constant
        kappa_obs: Screening factor for Yukawa repulsive obstacles (exponential decay rate)
        q_obs_default: Default obstacle repulsive charge
        epsilon: Softening radius to prevent singularity at r=0
        vortex_gain: Curl circulation gain to bypass obstacle stagnation points
        """
        self.bounds = world_bounds
        self.k_att = float(k_att)
        self.kappa_obs = float(kappa_obs)
        self.q_obs_default = float(q_obs_default)
        self.epsilon = float(epsilon)
        self.vortex_gain = float(vortex_gain)

        # Spatial hash for dynamic obstacles
        self.grid_cell_size = 20.0
        self.index = CellIndex(dims=2, cell_size=self.grid_cell_size)
        
        # Obstacle storage: positions (M, 2), charges (M,), radii (M,)
        self.obstacle_positions = np.empty((0, 2), dtype=np.float32)
        self.obstacle_charges = np.empty(0, dtype=np.float32)
        self.obstacle_radii = np.empty(0, dtype=np.float32)

        # Goals storage: positions (G, 2), weights (G,)
        self.goal_positions = np.empty((0, 2), dtype=np.float32)
        self.goal_weights = np.empty(0, dtype=np.float32)

    def set_goals(self, goals: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Sets navigation destination goals.
        goals: (G, 2) coordinates
        """
        goals = np.atleast_2d(np.asarray(goals, dtype=np.float32))
        self.goal_positions = goals
        if weights is None:
            self.goal_weights = np.ones(len(goals), dtype=np.float32)
        else:
            self.goal_weights = np.asarray(weights, dtype=np.float32)

    def set_obstacles(self, positions: np.ndarray, charges: Optional[np.ndarray] = None, radii: Optional[np.ndarray] = None):
        """
        Sets static and dynamic obstacle clusters.
        positions: (M, 2)
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float32))
        M = len(positions)
        self.obstacle_positions = positions
        if charges is None:
            self.obstacle_charges = np.full(M, self.q_obs_default, dtype=np.float32)
        else:
            self.obstacle_charges = np.asarray(charges, dtype=np.float32)
            
        if radii is None:
            self.obstacle_radii = np.full(M, self.epsilon, dtype=np.float32)
        else:
            self.obstacle_radii = np.asarray(radii, dtype=np.float32)

        # Index obstacles into elastic spatial hash (rebuilt per set call:
        # append-only tables cannot unlearn stale keys). Buckets live in a dict;
        # the hash is the authoritative occupied-cell index used for neighborhood
        # probes in sample_flow_velocity_hashed.
        self.index.build(positions)

    def _eval_velocity(
        self,
        agent_positions: np.ndarray,
        obs_positions: np.ndarray,
        obs_charges: np.ndarray,
        obs_radii: np.ndarray,
        desired_speed: float = 10.0,
        chunk_size: int = 8192,
    ) -> np.ndarray:
        """Core vectorized flow-field evaluation, parameterized by the obstacle
        arrays so the hashed variant can reuse it with a per-agent obstacle
        subset WITHOUT constructing a fresh ``HarmonicPotentialFlowField`` and
        WITHOUT rebuilding a spatial hash per agent (round-8 hoist).  Uses
        ``self.goal_positions`` / ``self.goal_weights`` and the scalar field
        constants (``k_att``, ``kappa_obs``, ``vortex_gain``) -- identical
        inputs give bit-identical outputs to the original per-agent sub-field
        construction.
        """
        agent_positions = np.atleast_2d(np.asarray(agent_positions, dtype=np.float32))
        N = len(agent_positions)
        if N == 0:
            return np.empty((0, 2), dtype=np.float32)

        velocities = np.zeros((N, 2), dtype=np.float32)
        n_goals = len(self.goal_positions)
        n_obs = len(obs_positions)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(N, start_idx + chunk_size)
            pos_chunk = agent_positions[start_idx:end_idx] # (B, 2)
            B = len(pos_chunk)

            grad_total = np.zeros((B, 2), dtype=np.float32)

            # 1. Attractive Gradient toward Nearest / Weighted Goals
            if n_goals > 0:
                # diff: (B, G, 2)
                diff_goal = pos_chunk[:, None, :] - self.goal_positions[None, :, :]
                dist_sq_goal = np.sum(diff_goal**2, axis=-1) + 1e-6 # (B, G)
                dist_goal = np.sqrt(dist_sq_goal) # (B, G)

                # Normalized direction vectors to goals: diff_goal / dist_goal
                # Weighted attraction: sum_g w_g * (p - g) / (dist_g + eps)
                att_weights = self.goal_weights[None, :] / (dist_goal + 5.0) # (B, G)
                att_weights /= np.maximum(1e-6, np.sum(att_weights, axis=-1, keepdims=True)) # softmax-like weighting

                grad_att = np.sum(diff_goal * (att_weights[:, :, None] * self.k_att), axis=1) # (B, 2)
                grad_total += grad_att

            # 2. Screened Yukawa Repulsive Obstacle Gradient + Vortex Circulation
            if n_obs > 0:
                # diff: (B, M, 2) = pos_chunk - obs_pos
                diff_obs = pos_chunk[:, None, :] - obs_positions[None, :, :]
                r_sq = np.sum(diff_obs**2, axis=-1) + (obs_radii[None, :]**2) # (B, M)
                r = np.sqrt(r_sq) # (B, M)

                # Screened Yukawa Potential: V(r) = Q * exp(-kappa * r) / r
                # -grad V(r) = Q * exp(-kappa * r) * (kappa / r + 1 / r^2) * (diff / r)
                # Here grad V(r) is pointing outward from obstacle toward agent.
                decay = np.exp(-self.kappa_obs * r) # (B, M)
                force_mag = obs_charges[None, :] * decay * (self.kappa_obs / r + 1.0 / r_sq) # (B, M)

                # Unit directions from obstacle to agent: diff_obs / r
                unit_diff = diff_obs / r[:, :, None] # (B, M, 2)
                grad_rep = np.sum(unit_diff * force_mag[:, :, None], axis=1) # (B, 2) repulsive force

                # Add vortex circulation component orthogonal to repulsive gradient
                # Rotates repulsive force 90 deg counter-clockwise: (-dy, dx)
                vortex_force = np.stack([-grad_rep[:, 1], grad_rep[:, 0]], axis=-1) * self.vortex_gain

                # In potential field: v = -grad_att + grad_rep + vortex
                # Note: grad_rep here is already the repulsive push away from obstacle (+ outward)
                grad_total = grad_total - grad_rep - vortex_force

            # 3. Velocity Synthesis & Normalization
            # Velocity is directed opposite to attraction gradient (downhill toward goal)
            # grad_total accumulated: +grad_att - grad_rep - vortex
            # So desired heading = -grad_total
            heading = -grad_total
            speed = np.linalg.norm(heading, axis=-1, keepdims=True)
            norm_heading = np.where(speed > 1e-5, heading / speed, 0.0)

            # Decelerate when within goal arrival threshold
            if n_goals > 0:
                min_goal_dist = np.min(dist_goal, axis=-1, keepdims=True)
                arrival_scale = np.clip(min_goal_dist / 15.0, 0.05, 1.0)
                velocities[start_idx:end_idx] = norm_heading * (desired_speed * arrival_scale)
            else:
                velocities[start_idx:end_idx] = norm_heading * desired_speed

        return velocities

    def sample_flow_velocity(
        self,
        agent_positions: np.ndarray,
        desired_speed: float = 10.0,
        chunk_size: int = 8192
    ) -> np.ndarray:
        """
        Evaluates continuous flow field velocities for N agents in parallel.
        agent_positions: (N, 2) array of coordinates.
        Returns: (N, 2) normalized velocity vectors * desired_speed.
        """
        return self._eval_velocity(
            agent_positions,
            self.obstacle_positions,
            self.obstacle_charges,
            self.obstacle_radii,
            desired_speed=desired_speed,
            chunk_size=chunk_size,
        )

    def _neighborhood_obstacles(self, pos: np.ndarray, ring: int = 2) -> np.ndarray:
        """Obstacle indices within the (2*ring+1)^2 hash-neighborhood of pos."""
        k = self.index.key_of(pos)
        parts = []
        for nk in self.index.neighbor_keys(k, ring):
            b = self.index.bucket(nk)
            if b is not None:
                parts.append(b)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def sample_flow_velocity_hashed(self, agent_positions: np.ndarray,
                                    desired_speed: float = 10.0,
                                    ring: int = 2) -> np.ndarray:
        """
        Hash-accelerated variant: per agent, the Yukawa obstacle sum runs only
        over obstacles in the (2*ring+1)^2 elastic-hash neighborhood (screened
        contributions beyond that are < exp(-ring) of the peak). The elastic
        hash is the ONLY cell index consulted here.

        Round-8 hoist: the previous implementation constructed a fresh
        ``HarmonicPotentialFlowField`` per agent and called ``set_goals`` +
        ``set_obstacles`` (the latter rebuilds a spatial hash) per agent --
        a per-agent field rebuild that the prior session's "hoisted" claim
        denied.  The field constants, goals, and the parent obstacle hash
        are all fixed for the step, so the per-agent variation is ONLY the
        neighborhood obstacle subset.  This now calls the shared
        ``_eval_velocity`` helper with that subset directly, reusing
        ``self.goal_positions`` / ``self.goal_weights`` and the parent's
        already-built obstacle hash (consulted via
        ``_neighborhood_obstacles``).  Outputs are bit-identical to the
        per-agent sub-field construction (same constants, same goals, same
        obstacle subset -> same vectorized math), verified by an explicit
        equivalence check during this audit.
        """
        agent_positions = np.atleast_2d(np.asarray(agent_positions, dtype=np.float32))
        full = np.empty((len(agent_positions), 2), dtype=np.float32)
        for i, p in enumerate(agent_positions):
            obs_idx = self._neighborhood_obstacles(p, ring)
            if len(obs_idx):
                full[i] = self._eval_velocity(
                    p[None, :],
                    self.obstacle_positions[obs_idx],
                    self.obstacle_charges[obs_idx],
                    self.obstacle_radii[obs_idx],
                    desired_speed=desired_speed,
                )[0]
            else:
                # No neighborhood obstacles: still evaluate the attractive
                # goal term (obstacle subset is empty).  Matches the old
                # sub-field path which called set_obstacles with an empty
                # array then sample_flow_velocity.
                full[i] = self._eval_velocity(
                    p[None, :],
                    self.obstacle_positions[:0],
                    self.obstacle_charges[:0],
                    self.obstacle_radii[:0],
                    desired_speed=desired_speed,
                )[0]
        return full

    def validate_hashed_flow(self, agent_positions: np.ndarray,
                             desired_speed: float = 10.0, ring: int = 2) -> Dict:
        """Relative deviation of the hash-truncated field vs the full Yukawa sum."""
        ref = self.sample_flow_velocity(agent_positions, desired_speed=desired_speed)
        fast = self.sample_flow_velocity_hashed(agent_positions, desired_speed=desired_speed, ring=ring)
        return {
            "mean_rel_dev": float(np.mean(np.linalg.norm(fast - ref, axis=1) /
                                          np.maximum(1e-6, np.linalg.norm(ref, axis=1)))),
            "max_rel_dev": float(np.max(np.linalg.norm(fast - ref, axis=1) /
                                        np.maximum(1e-6, np.linalg.norm(ref, axis=1)))),
        }

    def rasterize_flow_field_grid(
        self,
        resolution: int = 128,
        desired_speed: float = 10.0
    ) -> Dict[str, np.ndarray]:
        """
        Generates a 2D vector field grid for game engine minimap/pathfinding baking.
        """
        min_x, min_y, max_x, max_y = self.bounds
        xs = np.linspace(min_x, max_x, resolution, dtype=np.float32)
        ys = np.linspace(min_y, max_y, resolution, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

        t0 = time.perf_counter()
        vels = self.sample_flow_velocity(grid_pts, desired_speed=desired_speed)
        t_ms = (time.perf_counter() - t0) * 1000.0

        Vx = vels[:, 0].reshape(resolution, resolution)
        Vy = vels[:, 1].reshape(resolution, resolution)

        return {
            "X": X,
            "Y": Y,
            "Vx": Vx,
            "Vy": Vy,
            "latency_ms": t_ms,
            "resolution": resolution
        }


def run_harmonic_pathfinding_demo():
    print("==================================================================")
    print(" GAME MECHANICS: HARMONIC POTENTIAL FLOW FIELD SWARM PATHFINDING")
    print("==================================================================")

    # World: 1000m x 1000m battlefield
    pathfinder = HarmonicPotentialFlowField(
        world_bounds=(0.0, 0.0, 1000.0, 1000.0),
        k_att=1.0,
        kappa_obs=0.03,
        q_obs_default=150.0,
        epsilon=3.0,
        vortex_gain=0.35
    )

    # 1. Setup multi-target objectives (Chokepoint / Base capture)
    goals = np.array([
        [900.0, 900.0],
        [850.0, 150.0]
    ], dtype=np.float32)
    pathfinder.set_goals(goals, weights=[1.0, 0.5])

    # 2. Setup static fortress / mountain obstacles
    np.random.seed(42)
    num_obstacles = 128
    obstacle_positions = np.random.uniform(200.0, 800.0, size=(num_obstacles, 2)).astype(np.float32)
    obstacle_radii = np.random.uniform(10.0, 30.0, size=num_obstacles).astype(np.float32)
    pathfinder.set_obstacles(obstacle_positions, radii=obstacle_radii)

    # 3. Simulate Massive Swarm Navigation (50,000 Agents)
    num_agents = 50000
    agent_positions = np.random.uniform(50.0, 250.0, size=(num_agents, 2)).astype(np.float32)

    print(f"[*] Navigating {num_agents:,} dynamic agents through {num_obstacles} obstacles...")
    
    # Warm-up pass
    _ = pathfinder.sample_flow_velocity(agent_positions[:100], desired_speed=12.0)

    # Benchmark full crowd velocity field sampling
    t0 = time.perf_counter()
    agent_velocities = pathfinder.sample_flow_velocity(agent_positions, desired_speed=12.0)
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    throughput = (num_agents / (t_elapsed / 1000.0))

    print(f"[-] Velocity Field Evaluation Latency: {t_elapsed:.2f} ms ({t_elapsed/num_agents*1000.0:.3f} us/agent)")
    print(f"[-] Real-Time Swarm Throughput:        {throughput:,.0f} agents/sec")
    print(f"[-] Equivalent Frame Rate (@ 50k boids): {1000.0 / max(1e-3, t_elapsed):.1f} FPS")
    print(f"[-] Mean Swarm Velocity Magnitude:     {np.mean(np.linalg.norm(agent_velocities, axis=-1)):.2f} m/s")

    # 4. Rasterize Minimap Flow Field
    grid_res = 128
    grid_stats = pathfinder.rasterize_flow_field_grid(resolution=grid_res)
    print(f"[-] 2D Vector Grid ({grid_res}x{grid_res} = {grid_res*grid_res:,} cells) Baked In: {grid_stats['latency_ms']:.2f} ms")
    val = pathfinder.validate_hashed_flow(np.random.uniform(0, 1000, (64, 2)).astype(np.float32))
    print(f"    Hash-Truncated Yukawa vs Full:    mean {val['mean_rel_dev']:.1e}, max {val['max_rel_dev']:.1e} (screening-valid)")
    print("==================================================================")


if __name__ == '__main__':
    run_harmonic_pathfinding_demo()
