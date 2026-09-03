"""
Fully Vectorized 2D Gaussian Fast Gaussian Transform (FGT) with Forces.

All cell loops eliminated — the only Python loops are over the
(2*ring+1)^2 neighbor offsets (25 for ring=2), and each offset is a
fully vectorized NumPy operation.

Near-field:  exact particle-particle Gaussian force, pairs built via
              np.repeat + np.cumsum (no per-cell Python loop).
Far-field:   monopole approximation at cell centers, one einsum over
              all (N, K) cell pairs.

The force convention matches the repulsive drift:
    F_i = sum_j q_j * 2*(x_i - x_j) / h^2 * exp(-|x_i - x_j|^2 / h^2)

Kernel:  G(r) = exp(-r^2 / h^2)   (standard FGT form, NOT exp(-r^2/2h^2))
         To match a brute-force that uses exp(-r^2/(2*sigma^2)),
         construct with h = sigma * sqrt(2).

References
----------
- Greengard & Strain (1991). The Fast Gauss Transform. SIAM J. Sci. Comput.
- The grid-based flat scheme follows the same P2M / direct-near / far-field
  structure as core/gaussian2d_fgt.py and core/radial_taylor.py, but
  eliminates all Python loops over cells.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class VectorizedGaussian2DFGT:
    """Fully vectorized 2D Gaussian FGT computing repulsive forces.

    Parameters
    ----------
    depth : int
        Grid resolution (cells per side).  Total cells = depth^2.
        Linear semantics (depth=16 -> 16 cells/side, NOT 2^16).
    h : float
        Gaussian bandwidth.  Kernel = exp(-r^2 / h^2).
        To match a brute-force using exp(-r^2 / (2*sigma^2)),
        set h = sigma * sqrt(2).
    ring : int
        Near-field neighborhood ring (default 2 = 5x5 box).
        Cells within `ring` of a particle's cell get exact
        particle-particle interactions; cells outside get the
        monopole far-field approximation.
    dtype : np.dtype
        Working precision (default float32).  float64 was the original
        default but doubles memory for all (N, K) intermediates — the
        main leak vector at N~18k, K~1k.  float32 gives <0.5% error.
    far_chunk_size : int
        Far-field (N, K, 2) broadcast is chunked over N in blocks of
        this size to bound peak memory.  0 = no chunking (legacy).
    near_chunk_size : int
        Near-field exact pair computation is chunked over target
        particles in blocks of this size.  Without chunking, concentrated
        particle distributions can produce O(N * max_cell_size) pairs
        in a single allocation (multi-GB at N~29k).  Default 512.
    """

    def __init__(self, depth: int = 16, h: float = 0.2, ring: int = 2,
                 dtype=np.float32, far_chunk_size: int = 4096,
                 near_chunk_size: int = 512):
        self.depth = int(depth)
        self.h = float(h)
        self.ring = int(ring)
        self.h2 = self.h * self.h
        self.grid_res = self.depth
        self._n_cells = self.depth * self.depth
        self.dtype = np.dtype(dtype)
        self.far_chunk_size = int(far_chunk_size)
        self.near_chunk_size = int(near_chunk_size)

    def evaluate_forces(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
    ) -> np.ndarray:
        """Compute repulsive forces F_i = sum_j q_j * K(x_i, x_j).

        K(x_i, x_j) = 2*(x_i - x_j)/h^2 * exp(-|x_i - x_j|^2 / h^2)

        Returns (N, 2) force array.
        """
        dt = self.dtype
        positions = np.asarray(positions, dtype=dt)
        charges = np.asarray(charges, dtype=dt)
        N = len(positions)
        if N == 0:
            return np.empty((0, 2), dtype=dt)

        depth = self.depth
        h2 = dt.type(self.h2)
        ring = self.ring
        n_cells = self._n_cells
        inv_h2 = dt.type(2.0 / self.h2)  # prefactor: 2/h^2

        # ── 1. Cell assignment ──────────────────────────────────────────
        ix = np.clip((positions[:, 0] * depth).astype(np.int32), 0, depth - 1)
        iy = np.clip((positions[:, 1] * depth).astype(np.int32), 0, depth - 1)
        cell_id = (ix * depth + iy).astype(np.int32)  # (N,)

        # Sort particles by cell for O(1) cell->particles lookup
        sort_idx = np.argsort(cell_id, kind="stable")
        sorted_cell = cell_id[sort_idx]

        # Dense cell arrays: counts and starts in the sorted array
        cell_counts = np.bincount(sorted_cell, minlength=n_cells)  # (n_cells,)
        cell_starts = np.concatenate([[0], np.cumsum(cell_counts)])  # (n_cells+1,)

        # Occupied cells
        occupied = np.flatnonzero(cell_counts > 0)  # (K,)
        K = len(occupied)

        # Cell centers, total charges (monopoles), and dipole moments
        occ_ix = occupied // depth
        occ_iy = occupied % depth
        centers = np.stack(
            [(occ_ix + 0.5) / depth, (occ_iy + 0.5) / depth], axis=1
        ).astype(dt)  # (K, 2)
        cell_charges = np.bincount(
            cell_id, weights=charges, minlength=n_cells
        )[occupied].astype(dt)  # (K,)

        # Map cell_id -> index into `occupied` array (for far-field lookup)
        # cell_to_occ[cell_id] = index in occupied, or -1 if unoccupied
        cell_to_occ = np.full(n_cells, -1, dtype=np.int32)
        cell_to_occ[occupied] = np.arange(K)

        # Dipole moments: p_s = sum_{j in cell s} q_j * (x_j - c_s)
        disp_from_center = positions - centers[cell_to_occ[cell_id]]  # (N, 2)
        dipoles = np.stack([
            np.bincount(cell_id, weights=charges * disp_from_center[:, d],
                        minlength=n_cells)[occupied]
            for d in range(2)
        ], axis=1).astype(dt)  # (K, 2)

        # ── 2. Far-field: monopole + dipole at cell centers (ALL cells) ─
        # F_i = sum_s [Q_s * K(x_i, c_s) + (p_s · ∇_{c_s}) K(x_i, c_s)]
        # where K(x, c) = 2*(x-c)/h^2 * exp(-|x-c|^2/h^2) is the force kernel.
        #
        # Dipole correction: ∂K_d/∂c_e = 2*G/h^2 * [-δ_{de} + 2*d_d*d_e/h^2]
        # so ΔF = sum_s 2*G/h^2 * [-p_s + 2*(p_s·d)*d/h^2]
        #
        # This includes near-field cells; we subtract the near mono+dipole
        # below and add the exact near-field.
        #
        # Chunked over N to bound peak memory: the (N, K, 2) broadcast
        # is the single largest allocation.  At N=18k, K=1k, float64 that
        # was ~293 MB per array; float32 + chunking cuts it to ~33 MB.
        far_forces = np.zeros((N, 2), dtype=dt)
        if K > 0:
            chunk = self.far_chunk_size if self.far_chunk_size > 0 else N
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                pos_c = positions[i0:i1]  # (nc, 2)
                diff_far = pos_c[:, None, :] - centers[None, :, :]  # (nc, K, 2)
                r2_far = np.sum(diff_far * diff_far, axis=-1)  # (nc, K)
                kernel_far = np.exp(-r2_far / h2)  # (nc, K)
                prefactor = kernel_far * inv_h2  # (nc, K)  — 2*G/h^2

                # Monopole: Q_s * 2*d/h^2 * G
                ff_chunk = np.einsum(
                    "nk,nkd,k->nd", prefactor, diff_far, cell_charges
                )  # (nc, 2)

                # Dipole: 2*G/h^2 * [-p_s + 2*(p_s·d)*d/h^2]
                p_dot_d = np.einsum("nkd,kd->nk", diff_far, dipoles)  # (nc, K)
                dipole_term1 = np.einsum(
                    "nk,kd->nd", prefactor, dipoles
                )  # (nc, 2)
                dipole_term2 = np.einsum(
                    "nk,nkd->nd", prefactor * p_dot_d, diff_far
                ) * (dt.type(2.0) / h2)  # (nc, 2)
                far_forces[i0:i1] = ff_chunk - dipole_term1 + dipole_term2

        # ── 3. Near-field: exact + mono+dipole subtraction ──────────────
        # For each offset (dx, dy) in the ring:
        #   near_exact  += sum_{j in neighbor cell} q_j * K(x_i, x_j)
        #   near_approx += Q_s*K(x_i,c_s) + dipole correction  (to subtract)
        # Total = far_forces - near_approx + near_exact
        near_exact = np.zeros((N, 2), dtype=dt)
        near_approx = np.zeros((N, 2), dtype=dt)

        offsets = range(-ring, ring + 1)
        for dx in offsets:
            for dy in offsets:
                # Offset validity: skip out-of-bounds offsets instead of
                # clipping (clipping causes duplicate cell visits at
                # boundaries, amplifying the near-field correction).
                offset_valid = (
                    (ix + dx >= 0) & (ix + dx < depth)
                    & (iy + dy >= 0) & (iy + dy < depth)
                )  # (N,) bool

                # Neighbor cell for each particle (clipped for safe indexing,
                # but offset_valid gates the contribution)
                nbr_ix = np.clip(ix + dx, 0, depth - 1)
                nbr_iy = np.clip(iy + dy, 0, depth - 1)
                nbr_cell = (nbr_ix * depth + nbr_iy).astype(np.int32)  # (N,)

                # Which particles have a valid, occupied neighbor cell?
                nbr_occ_idx = cell_to_occ[nbr_cell]  # (N,) — -1 if unoccupied
                valid = offset_valid & (nbr_occ_idx >= 0)
                if not np.any(valid):
                    continue

                # ── 3a. Mono+dipole subtraction for this offset ─────────
                nbr_centers = centers[nbr_occ_idx[valid]]  # (n_valid, 2)
                nbr_q = cell_charges[nbr_occ_idx[valid]]  # (n_valid,)
                nbr_p = dipoles[nbr_occ_idx[valid]]  # (n_valid, 2)
                diff_mono = positions[valid] - nbr_centers  # (n_valid, 2)
                r2_mono = np.sum(diff_mono * diff_mono, axis=-1)
                kernel_mono = np.exp(-r2_mono / h2)
                pref_mono = kernel_mono * inv_h2  # 2*G/h^2

                # Monopole: Q_s * 2*d/h^2 * G
                mono_force = nbr_q[:, None] * pref_mono[:, None] * diff_mono

                # Dipole: 2*G/h^2 * [-p_s + 2*(p_s·d)*d/h^2]
                p_dot_d_mono = np.sum(nbr_p * diff_mono, axis=-1)  # (n_valid,)
                dip_force = (
                    -pref_mono[:, None] * nbr_p
                    + pref_mono[:, None] * (dt.type(2.0) / h2) * p_dot_d_mono[:, None] * diff_mono
                )
                np.add.at(near_approx, np.flatnonzero(valid),
                          mono_force + dip_force)

                # ── 3b. Exact particle-particle for this offset (chunked) ─
                # Build pairs (i, j) where j is in the neighbor cell of i.
                # Use repeat + cumsum to vectorize the variable-length join.
                #
                # Chunked over target particles to bound peak memory: without
                # chunking, total_pairs can reach O(N * max_cell_size) which
                # at N=29k with concentrated gradients exceeds 800M pairs
                # (6+ GiB for a single int64 index array).
                nbr_counts = cell_counts[nbr_cell]  # (N,) — particles in neighbor cell
                # Only keep particles whose neighbor cell is non-empty
                has_pairs = nbr_counts > 0
                if not np.any(has_pairs):
                    continue

                hp_idx = np.flatnonzero(has_pairs)  # (n_hp,)
                hp_counts = nbr_counts[has_pairs]  # (n_hp,)
                hp_nbr_cell = nbr_cell[has_pairs]  # (n_hp,)
                hp_starts = cell_starts[hp_nbr_cell]  # (n_hp,) — start in sort_idx

                near_chunk = self.near_chunk_size
                n_hp = len(hp_idx)
                for ci in range(0, n_hp, near_chunk):
                    ce = min(ci + near_chunk, n_hp)
                    ci_idx = hp_idx[ci:ce]       # (nc,)
                    ci_counts = hp_counts[ci:ce]  # (nc,)
                    ci_starts = hp_starts[ci:ce]  # (nc,)
                    chunk_pairs = int(np.sum(ci_counts))
                    if chunk_pairs == 0:
                        continue

                    # Target particle indices (repeated)
                    target_idx = np.repeat(ci_idx, ci_counts)  # (chunk_pairs,)

                    # Within-cell offset for each expanded position
                    ci_cumsum = np.cumsum(ci_counts)  # (nc,)
                    expanded_start = np.repeat(ci_starts, ci_counts)
                    within = np.arange(chunk_pairs) - np.repeat(
                        np.concatenate([[0], ci_cumsum[:-1]]), ci_counts
                    )
                    source_idx = sort_idx[expanded_start + within]  # (chunk_pairs,)

                    # Compute exact force for all pairs in this chunk
                    diff_pair = positions[target_idx] - positions[source_idx]  # (P, 2)
                    r2_pair = np.sum(diff_pair * diff_pair, axis=-1)  # (P,)
                    kernel_pair = np.exp(-r2_pair / h2)  # (P,)

                    # Self-mask: exclude i == j (only matters for offset (0,0))
                    self_mask = target_idx == source_idx
                    kernel_pair = np.where(self_mask, 0.0, kernel_pair)

                    pair_force = (
                        charges[source_idx, None]
                        * kernel_pair[:, None]
                        * diff_pair
                        * inv_h2
                    )  # (P, 2)
                    np.add.at(near_exact, target_idx, pair_force)

        # ── 4. Total: far - near_approx + near_exact ───────────────────
        # far_forces includes mono+dipole for ALL cells.
        # near_approx is the mono+dipole for near cells (to subtract).
        # near_exact is the exact particle-particle for near cells (to add).
        forces = far_forces - near_approx + near_exact
        return forces
