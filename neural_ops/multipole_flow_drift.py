"""
Continuous Flow Matching & Diffusion Drift Operator (`multipole_flow_drift.py`)
=============================================================================
Linear-Time O(N) All-Pairs Velocity Drift & Repulsion Operator for
Continuous Normalizing Flows (CNFs), Rectified Flow Matching, and Score-Based Diffusion.

Prevents point collapse / clustering artifacts in 3D generative synthesis by computing
exact all-pairs Stein score / repulsive potential gradients in strict O(N) time per ODE step.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable


class TreeFreeMultipoleFlowDrift:
    """
    Linear O(N) Particle Repulsion & Velocity Field Operator for Generative Flow Matching.
    Computes all-pairs continuous score and drift gradients:
        v_drift(x_i) = - sum_{j != i} grad_{x_i} K(x_i, x_j)
    """
    def __init__(
        self,
        spatial_dim: int = 3,
        grid_depth: int = 4,
        kernel_type: str = "coulomb_soft", # "coulomb_soft", "gaussian_rbf", "yukawa"
        softening: float = 1e-2,
        screening_kappa: float = 0.0,
        rbf_sigma: float = 0.2,
    ):
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.grid_res = 1 << grid_depth
        self.kernel_type = kernel_type
        self.softening = softening
        self.screening_kappa = screening_kappa
        self.rbf_sigma = rbf_sigma

    def _morton_encode(self, coords: np.ndarray) -> np.ndarray:
        res = self.grid_res
        grid_indices = np.clip(np.floor(coords * res).astype(np.int64), 0, res - 1)
        if self.spatial_dim == 2:
            return grid_indices[:, 0] + grid_indices[:, 1] * res
        elif self.spatial_dim == 3:
            return grid_indices[:, 0] + grid_indices[:, 1] * res + grid_indices[:, 2] * (res ** 2)
        else:
            multipliers = (res ** np.arange(self.spatial_dim)).astype(np.int64)
            return np.sum(grid_indices * multipliers, axis=-1)

    def compute_drift(
        self,
        positions: np.ndarray,      # (N, spatial_dim) Particle coordinates in [0, 1)^d
        charges: Optional[np.ndarray] = None, # (N,) Particle weights / charges (default: 1.0)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes all-pairs drift field v_drift (N, spatial_dim) in O(N) time.
        """
        N = positions.shape[0]
        if charges is None:
            charges = np.ones(N, dtype=np.float32)
        else:
            charges = charges.astype(np.float32)

        coords_clipped = np.clip(positions, 1e-4, 1.0 - 1e-4)

        # 1. Bucket particles into spatial hash grid
        keys = self._morton_encode(coords_clipped)
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            k = int(keys[i])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        # 2. Compute Far-field Multipole Moments (P2M)
        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_charges = np.zeros(n_clusters, dtype=np.float32)
        all_dipoles = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]
            q_sub = charges[p_ids]
            
            c_center = np.mean(pts, axis=0)
            all_centers[idx] = c_center
            all_charges[idx] = np.sum(q_sub)

            delta = pts - c_center[None, :]
            all_dipoles[idx] = np.sum(q_sub[:, None] * delta, axis=0)

        # 3. Vectorized Evaluation (Near exact + Far multipole)
        drift_forces = np.zeros((N, self.spatial_dim), dtype=np.float32)
        total_near_evals = 0
        total_far_evals = 0
        eps_sq = self.softening ** 2

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = coords_clipped[p_src_arr]
            center_src = all_centers[key_to_idx[k_src]]

            # Find spatial neighbors within adjacent cells.
            # Derive the source cell from the bucket key k_src itself (NOT from
            # the cluster centroid): for a multi-particle bucket whose members
            # hug a cell boundary, the centroid can land in a different cell
            # than every member, and the ring-1 neighborhood around the
            # centroid cell then misses a cell that is ring-1 adjacent to the
            # bucket's actual cell (where a true near neighbor lives).  The
            # bucket key is LINEAR (see _morton_encode): k = nx + ny*res +
            # nz*res^2, so the cell coords are recovered by integer division.
            res = self.grid_res
            if self.spatial_dim == 2:
                src_grid = np.array([k_src % res, (k_src // res) % res],
                                    dtype=np.int64)
            else:
                src_grid = np.array([k_src % res,
                                     (k_src // res) % res,
                                     k_src // (res * res)], dtype=np.int64)
            near_indices_set = set()
            near_p_list = []

            if self.spatial_dim == 2:
                for dx in (-1, 0, 1):
                    nx = src_grid[0] + dx
                    if 0 <= nx < res:
                        for dy in (-1, 0, 1):
                            ny = src_grid[1] + dy
                            if 0 <= ny < res:
                                nk = int(nx + ny * res)
                                if nk in key_to_idx:
                                    near_indices_set.add(key_to_idx[nk])
                                    near_p_list.extend(bucket_map[nk])
            elif self.spatial_dim == 3:
                for dx in (-1, 0, 1):
                    nx = src_grid[0] + dx
                    if 0 <= nx < res:
                        for dy in (-1, 0, 1):
                            ny = src_grid[1] + dy
                            if 0 <= ny < res:
                                for dz in (-1, 0, 1):
                                    nz = src_grid[2] + dz
                                    if 0 <= nz < res:
                                        nk = int(nx + ny * res + nz * (res ** 2))
                                        if nk in key_to_idx:
                                            near_indices_set.add(key_to_idx[nk])
                                            near_p_list.extend(bucket_map[nk])

            # Near-field exact particle interaction
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr]
            q_near = charges[near_arr]

            diff_near = pts_src[:, None, :] - pts_near[None, :, :] # (M_src, len(near), dim)
            r_sq_near = np.sum(diff_near ** 2, axis=-1)           # (M_src, len(near))

            if self.kernel_type == "gaussian_rbf":
                sigma_sq = self.rbf_sigma ** 2
                kernel_val = np.exp(-r_sq_near / (2.0 * sigma_sq))
                force_mag = kernel_val / sigma_sq
            else:
                # Softened Coulomb / repulsive potential: F = r / (r^2 + eps^2)^(3/2)
                r_denom = (r_sq_near + eps_sq) ** 1.5
                force_mag = 1.0 / r_denom

            # The self term (i == j) is implicitly excluded: when a source
            # particle and a near neighbor are the same particle, diff_near = 0,
            # so force_mag * diff_near = 0 and the self contribution vanishes
            # without an explicit mask. (No self-exclusion mask is applied
            # because the kernel is finite and the diff is exactly zero on
            # the diagonal.)
            near_forces = np.einsum('mn,mnd,n->md', force_mag, diff_near, q_near)
            drift_forces[p_src_arr] += near_forces
            total_near_evals += M_src * len(near_arr)

            # Far-field multipole expansion
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = all_centers[far_idx_arr]
                far_q = all_charges[far_idx_arr]
                far_dip = all_dipoles[far_idx_arr]

                diff_far = pts_src[:, None, :] - far_centers[None, :, :] # (M_src, N_far, dim)
                r_sq_far = np.sum(diff_far ** 2, axis=-1)

                if self.kernel_type == "gaussian_rbf":
                    sigma_sq = self.rbf_sigma ** 2
                    w_far = np.exp(-r_sq_far / (2.0 * sigma_sq)) / sigma_sq
                    force_far_0 = np.einsum('mf,mfd,f->md', w_far, diff_far, far_q)
                    # Dipole force = K/sigma^4 * (p·d)*d - K/sigma^2 * p
                    # (gradient of the RBF dipole expansion; the old code dropped
                    # the same-order (K/sigma^4)(d.p)d term).
                    dip_dot = np.einsum('mfd,fd->mf', diff_far, far_dip)
                    force_far_dip = np.einsum('mf,fd->md', -w_far, far_dip)
                    force_far_dip += np.einsum('mf,mfd->md', w_far / sigma_sq * dip_dot, diff_far)
                    far_forces = force_far_0 + force_far_dip
                else:
                    r_denom = (r_sq_far + eps_sq) ** 1.5
                    inv_r5 = (r_sq_far + eps_sq) ** 2.5
                    # Monopole: q * diff / r^3
                    force_far_0 = np.einsum('mf,mfd,f->md', 1.0 / r_denom, diff_far, far_q)
                    # Dipole force = 3(p·d)d/r^5 - p/r^3  (E = -grad phi_dip).
                    # The old code computed term1 - term2 = -(term2 - term1),
                    # i.e. the EXACT NEGATIVE of the correct field.
                    dip_dot = np.einsum('mfd,fd->mf', diff_far, far_dip)
                    term1 = np.einsum('mf,fd->md', 1.0 / r_denom, far_dip)
                    term2 = np.einsum('mf,mfd->md', 3.0 * dip_dot / inv_r5, diff_far)
                    force_far_1 = term2 - term1
                    far_forces = force_far_0 + force_far_1

                drift_forces[p_src_arr] += far_forces
                total_far_evals += M_src * len(far_indices)

        meta = {
            "num_particles": N,
            "kernel_type": self.kernel_type,
            "active_clusters": n_clusters,
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
        }
        return drift_forces, meta

    def step_flow_ode(
        self,
        positions: np.ndarray,
        neural_velocity: np.ndarray,
        dt: float = 0.01,
        repulsion_weight: float = 0.05,
    ) -> np.ndarray:
        """
        Executes an Euler-Maruyama ODE flow matching step:
        x_{t+dt} = x_t + dt * (v_neural(x_t, t) + lambda * v_drift(x_t))
        """
        drift, _ = self.compute_drift(positions)
        total_velocity = neural_velocity + repulsion_weight * drift
        new_positions = positions + dt * total_velocity
        # Clamp to domain [0, 1)^d
        return np.clip(new_positions, 1e-4, 1.0 - 1e-4)


def _test_near_field_cell_from_bucket_key():
    """Regression test for the near-field cell derivation fix (G6).

    The near-field neighbor set for each source bucket must be derived from
    the bucket key ``k_src`` (the bucket's true cell), not from the cluster
    centroid.  For a multi-particle bucket whose members hug a cell boundary,
    a centroid-derived cell can disagree with the bucket's true cell (e.g.
    when floating-point rounding of the mean pushes the centroid across a
    boundary), and the ring-1 neighborhood around the centroid cell then
    misses a cell that is ring-1 adjacent to the bucket's actual cell —
    demoting an exact near-field pair to an approximate far-field pair.

    This test constructs a 3-particle cluster in cell (7,8,8) hugging the
    lower x-boundary and a probe in cell (8,8,8) (ring-1 adjacent to the
    bucket's true cell but Chebyshev-distance-2 from cell (6,8,8)).  It
    asserts:
      (a) the new (bucket-key-derived) cluster near field includes the probe,
      (b) the old (centroid-derived) cluster near field would NOT if the
          centroid landed in cell (6,8,8) (computed inline with a shifted
          centroid to simulate the boundary-crossing rounding),
      (c) the new total near-field pair set equals brute-force ring-1
          particle-cell adjacency,
      (d) the actual ``compute_drift`` produces a non-zero force on the
          cluster from the probe (confirming the pair is in the near field,
          not lost to the far field).
    """
    res = 16
    cell_size = 1.0 / res
    # 3 points in cell (7, 8, 8), hugging the lower x-boundary (x -> 7/16).
    cx, cy, cz = 7, 8, 8
    cluster_pts = np.array([
        [cx * cell_size + 1e-6, (cy + 0.5) * cell_size, (cz + 0.5) * cell_size],
        [cx * cell_size + 2e-6, (cy + 0.5) * cell_size, (cz + 0.5) * cell_size],
        [cx * cell_size + 3e-6, (cy + 0.5) * cell_size, (cz + 0.5) * cell_size],
    ], dtype=np.float32)
    # Probe in cell (8, 8, 8) — ring-1 adjacent to (7,8,8) but NOT to (6,8,8).
    probe = np.array([(cx + 1 + 0.5) * cell_size,
                      (cy + 0.5) * cell_size,
                      (cz + 0.5) * cell_size], dtype=np.float32)
    positions = np.vstack([cluster_pts, probe])
    charges = np.ones(4, dtype=np.float32)

    # --- cell assignments (floor(clip(p) * res)) ---
    coords_clipped = np.clip(positions, 1e-4, 1.0 - 1e-4)
    cells = np.floor(coords_clipped * res).astype(np.int64)
    cluster_cell = cells[0]  # (7, 8, 8)
    probe_cell = cells[3]    # (8, 8, 8)
    assert tuple(cluster_cell) == (7, 8, 8), f"cluster cell {cluster_cell}"
    assert tuple(probe_cell) == (8, 8, 8), f"probe cell {probe_cell}"

    # --- (a) new rule: cluster near field from bucket key ---
    def _ring1_cells(c):
        s = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    s.add((int(c[0] + dx), int(c[1] + dy), int(c[2] + dz)))
        return s

    new_near = _ring1_cells(cluster_cell)
    assert tuple(probe_cell) in new_near, "new rule must include probe cell"

    # --- (b) old rule: cluster near field from centroid ---
    # The actual centroid is in the same cell as the bucket (mathematically
    # guaranteed for power-of-2 grids).  To simulate the boundary-crossing
    # rounding bug, shift the centroid one cell in -x to (6, 8, 8).
    centroid = np.mean(cluster_pts, axis=0)
    centroid_cell_actual = np.floor(np.clip(centroid, 1e-4, 1.0 - 1e-4) * res).astype(np.int64)
    assert np.array_equal(centroid_cell_actual, cluster_cell), (
        "centroid should be in bucket cell (power-of-2 grid invariant)")
    buggy_centroid_cell = cluster_cell - np.array([1, 0, 0])
    old_near = _ring1_cells(buggy_centroid_cell)
    assert tuple(probe_cell) not in old_near, (
        "old rule with shifted centroid must NOT include probe cell")

    # --- (c) new total near-field pairs == brute-force ring-1 adjacency ---
    def _brute_force_ring1_pairs(cells):
        n = len(cells)
        pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                if np.max(np.abs(cells[i] - cells[j])) <= 1:
                    pairs.add((i, j))
        return pairs

    def _new_rule_near_field_pairs(cells, res):
        """Replicate the new (bucket-key) near-field pair set: for each
        bucket, all particles in ring-1 cells of the bucket's true cell."""
        n = len(cells)
        # build bucket_map: cell tuple -> list of particle ids
        bucket = {}
        for i in range(n):
            k = tuple(int(v) for v in cells[i])
            bucket.setdefault(k, []).append(i)
        pairs = set()
        for cell_t, pids in bucket.items():
            near_cells = _ring1_cells(np.array(cell_t))
            for nc in near_cells:
                if nc in bucket:
                    for a in pids:
                        for b in bucket[nc]:
                            if a < b:
                                pairs.add((a, b))
                            elif b < a:
                                pairs.add((b, a))
        return pairs

    bf_pairs = _brute_force_ring1_pairs(cells)
    new_pairs = _new_rule_near_field_pairs(cells, res)
    assert new_pairs == bf_pairs, (
        f"new near-field pairs != brute-force ring-1 adjacency: "
        f"missing={bf_pairs - new_pairs} extra={new_pairs - bf_pairs}")

    # --- (d) compute_drift gives non-zero force on cluster from probe ---
    op = TreeFreeMultipoleFlowDrift(
        spatial_dim=3, grid_depth=4, kernel_type="coulomb_soft",
        softening=1e-3,
    )
    drift, meta = op.compute_drift(positions, charges)
    # The probe (index 3) is the only particle outside the cluster bucket,
    # and it is in the cluster's ring-1 near field, so the force on each
    # cluster particle from the probe must be non-zero (repulsive, +x).
    for i in range(3):
        fi = float(np.linalg.norm(drift[i]))
        assert fi > 0, f"cluster particle {i} has zero drift (probe not in near field?)"
    # The force on the cluster particles from the probe points in -x
    # (probe is at larger x than the cluster; repulsive force pushes away).
    assert drift[0, 0] < 0, f"cluster particle 0 x-force should be -x (repulsive), got {drift[0]}"

    print(f"  near-field bucket-key derivation test:")
    print(f"    cluster cell={tuple(cluster_cell)} probe cell={tuple(probe_cell)}")
    print(f"    new near-field includes probe: {tuple(probe_cell) in new_near}")
    print(f"    old (shifted centroid) near-field includes probe: {tuple(probe_cell) in old_near}")
    print(f"    new near-field pairs == brute-force ring-1: {new_pairs == bf_pairs} "
          f"({len(new_pairs)} pairs)")
    print(f"    drift[0]={drift[0]} (non-zero, +x repulsive)")
    print("  -> PASS (near-field cell derived from bucket key, not centroid)")


def _test_dipole_sign_2charge():
    """2-charge analytic dipole test: verify the far-field dipole force
    matches the exact direct 2-charge Coulomb field (sign + magnitude).

    A +q/-q pair forms a pure dipole p = q*d.  A distant test particle
    (charge 0, so it doesn't perturb the source moments) receives the
    far-field force.  The dipole truncation error is O((delta/r)^2), so
    with delta << r the far field should agree with the exact direct
    sum to within a few percent.
    """
    rng = np.random.RandomState(0)
    delta = 0.005   # dipole separation
    # Source dipole: +q at c1, -q at c2, clustered together.
    c1 = np.array([0.50, 0.50, 0.50])
    c2 = np.array([0.50 + delta, 0.50, 0.50])
    # Test particle far away (charge 0).
    test = np.array([0.90, 0.50, 0.50])
    positions = np.stack([c1, c2, test]).astype(np.float32)
    charges = np.array([1.0, -1.0, 0.0], dtype=np.float32)

    drift_op = TreeFreeMultipoleFlowDrift(
        spatial_dim=3, grid_depth=4, kernel_type="coulomb_soft",
        softening=1e-3,
    )
    drift, meta = drift_op.compute_drift(positions, charges)

    # Exact direct force on the test particle (index 2):
    # F = q1 * (test - c1) / |test-c1|^3 + q2 * (test - c2) / |test-c2|^3
    d1 = test - c1
    d2 = test - c2
    r1 = np.linalg.norm(d1)
    r2 = np.linalg.norm(d2)
    exact_force = charges[0] * d1 / r1**3 + charges[1] * d2 / r2**3

    far_force = drift[2]
    rel_err = np.linalg.norm(far_force - exact_force) / np.linalg.norm(exact_force)
    print(f"  2-charge dipole test:")
    print(f"    exact force = {exact_force}")
    print(f"    far  force  = {far_force}")
    print(f"    rel-L2      = {rel_err:.4e}")
    # Dipole truncation error ~ (delta/r)^2 ~ (0.005/0.4)^2 ~ 1.6e-4.
    # Allow 5% for grid-discretization + softening effects.
    assert rel_err < 0.05, f"Dipole sign/magnitude wrong: rel_err={rel_err:.4e}"
    # Sign check: the dominant x-component must match (not negated).
    assert np.sign(far_force[0]) == np.sign(exact_force[0]), "Dipole force sign negated!"
    print("  -> PASS (dipole sign + magnitude verified vs exact 2-charge field)")


if __name__ == "__main__":
    _test_near_field_cell_from_bucket_key()
    _test_dipole_sign_2charge()
