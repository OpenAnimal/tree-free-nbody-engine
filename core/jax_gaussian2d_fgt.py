"""
JAX GPU 2D Gaussian Fast Gaussian Transform (FGT) with Forces.

A fully on-device implementation of the same monopole + dipole far-field /
exact near-field scheme as ``core/vectorized_gaussian2d_fgt.py``, but
expressed in JAX so it runs on GPU/TPU inside a larger JIT'd pipeline
(no host round-trips).

The design mirrors ``core/jax_tree_free_fmm.py``:
  - ``segment_sum`` for cell aggregation (P2M moments)
  - ``jnp.argsort`` for counting-sort by cell key
  - static ``max_K = depth * depth`` cell arrays with zero-masking for
    unoccupied cells (no dynamic-shape occupied-cell compaction)
  - padded cell slots for the near-field exact pair sweep (static
    ``max_cell_size``), avoiding ragged while-loops

Unlike ``jax_tree_free_fmm.py`` this module does **not** enable x64:
the Gaussian FGT uses only real arithmetic and float32 gives <0.5%
relative error vs the f64 brute-force reference.

Kernel
------
    G(r) = exp(-r^2 / h^2)
    F_i  = sum_j q_j * 2*(x_i - x_j)/h^2 * exp(-|x_i - x_j|^2 / h^2)

To match a brute-force using exp(-r^2/(2*sigma^2)), construct with
h = sigma * sqrt(2).

Complexity
----------
Flat-grid scheme (single level, uniform depth x depth grid):

  Far-field :  O(N * K)        K = depth^2  (all cells, masked)
  Near-field:  O(N * 25 * C)   C = max_cell_size (padded)

With adaptive depth = ceil(sqrt(N / 8)):
  K ~ N/8, C ~ 8  =>  far = O(N^2/8), near = O(200*N)

The far-field dominates asymptotically, but on GPU it is a single
batched einsum that is highly parallelisable.  For N up to ~100k
this is well under 1 second on a modern GPU.  A hierarchical
multi-level scheme (Greengard & Strain, 1991) would give true O(N)
but adds significant trace-time complexity for modest gain at
these sizes.

References
----------
- Greengard & Strain (1991). The Fast Gauss Transform. SIAM J. Sci. Comput.
- The flat P2M / direct-near / far-field structure follows
  ``core/vectorized_gaussian2d_fgt.py`` and ``core/gaussian2d_fgt.py``.
"""

from __future__ import annotations

from typing import Tuple
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax
    try:
        from jax.lax import segment_sum
    except ImportError:  # pragma: no cover - JAX >= 0.10 layout
        from jax._src.ops.scatter import segment_sum
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    lax = None
    segment_sum = None
    partial = None
    HAS_JAX = False


if HAS_JAX:
    # ─────────────────────────────────────────────────────────────────────
    # Cell binning helpers (shared structure with jax_tree_free_fmm.py)
    # ─────────────────────────────────────────────────────────────────────

    def _bin_particles(positions: jnp.ndarray, depth: int):
        """Quantise positions to row-major cell keys and counting-sort.

        Returns
        -------
        keys        : (N,)   raw cell key in [0, depth*depth)
        sort_idx    : (N,)   stable argsort by key (sorted -> original)
        cell_counts : (K,)   particles per cell  (K = depth*depth)
        cell_starts : (K+1,) exclusive prefix sum of counts
        within_slot : (N,)   within-cell ordinal for each sorted particle
        """
        N = positions.shape[0]
        K = depth * depth
        ix = jnp.clip((positions[:, 0] * depth).astype(jnp.int32),
                      0, depth - 1)
        iy = jnp.clip((positions[:, 1] * depth).astype(jnp.int32),
                      0, depth - 1)
        keys = (ix * depth + iy).astype(jnp.int32)     # (N,) row-major
        sort_idx = jnp.argsort(keys, stable=True).astype(jnp.int32)
        sorted_keys = keys[sort_idx]
        cell_counts = segment_sum(
            jnp.ones(N, dtype=jnp.int32), sorted_keys,
            num_segments=K)                            # (K,)
        # exclusive prefix sum
        cell_starts = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(cell_counts),
        ]).astype(jnp.int32)                           # (K+1,)
        # within-cell slot:  arange(N) - cell_starts[sorted_keys]
        within_slot = (jnp.arange(N, dtype=jnp.int32)
                       - cell_starts[sorted_keys])
        return keys, sort_idx, cell_counts, cell_starts, within_slot

    def _build_padded_cells(positions, charges, keys, sort_idx,
                            cell_counts, cell_starts, within_slot,
                            depth: int, max_cell_size: int):
        """Scatter sorted particles into a (K, max_cell_size, D) pad.

        Returns
        -------
        cell_pos    : (K, max_cell_size, 2)  positions, 0-padded
        cell_q      : (K, max_cell_size)     charges, 0-padded
        cell_orig   : (K, max_cell_size)     original particle id, -1 padded
        cell_valid  : (K, max_cell_size)     bool mask (slot < cell_count)
        """
        K = depth * depth
        N = positions.shape[0]
        sorted_pos = positions[sort_idx]               # (N, 2)
        sorted_q = charges[sort_idx]                   # (N,)
        sorted_orig = sort_idx                         # (N,)
        sorted_keys = keys[sort_idx]                   # (N,)

        # Scatter into padded grid — excess particles (slot >= max_cell_size)
        # are silently dropped.  With adaptive depth targeting ~8/cell this
        # is vanishingly rare; callers should verify max(cell_counts) <<
        # max_cell_size.
        slot_ok = within_slot < max_cell_size
        safe_slot = jnp.where(slot_ok, within_slot, jnp.int32(0)).astype(jnp.int32)
        safe_keys = jnp.where(slot_ok, sorted_keys, jnp.int32(0)).astype(jnp.int32)

        cell_pos = jnp.zeros((K, max_cell_size, 2),
                             dtype=positions.dtype)
        cell_pos = cell_pos.at[safe_keys, safe_slot].set(sorted_pos)

        cell_q = jnp.zeros((K, max_cell_size), dtype=charges.dtype)
        cell_q = cell_q.at[safe_keys, safe_slot].set(sorted_q)

        cell_orig = jnp.full((K, max_cell_size), -1, dtype=jnp.int32)
        cell_orig = cell_orig.at[safe_keys, safe_slot].set(sorted_orig)

        # Validity mask: slot < actual cell count
        cell_valid = (jnp.arange(max_cell_size)[None, :]
                      < cell_counts[:, None])          # (K, max_cell_size)
        return cell_pos, cell_q, cell_orig, cell_valid

    # ─────────────────────────────────────────────────────────────────────
    # Main FGT force evaluator
    # ─────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(2, 3, 4, 5))
    def jax_gaussian2d_fgt_forces(
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        h: float,
        depth: int = 32,
        ring: int = 2,
        max_cell_size: int = 64,
    ) -> jnp.ndarray:
        """Compute repulsive Gaussian forces on GPU/TPU.

        Parameters
        ----------
        positions : (N, 2) in [0, 1)^2
        charges   : (N,)
        h         : Gaussian bandwidth (kernel = exp(-r^2/h^2))
        depth     : grid resolution (cells per side, static)
        ring      : near-field neighbourhood ring (2 = 5x5 box)
        max_cell_size : static pad for near-field cell occupancy

        Returns
        -------
        forces : (N, 2)  same dtype as positions
        """
        dt = positions.dtype
        N = positions.shape[0]
        K = depth * depth
        h2 = dt.type(h * h)
        inv_h2 = dt.type(2.0 / (h * h))

        # ── 1. Bin & counting-sort ──────────────────────────────────────
        keys, sort_idx, cell_counts, cell_starts, within_slot = \
            _bin_particles(positions, depth)

        # ── 2. Cell moments (P2M) via segment_sum ───────────────────────
        # All K cells; unoccupied cells get zero moments.
        cell_id_arr = jnp.arange(K)
        # Key encoding is ix * depth + iy (see _bin_particles), so:
        cell_ix = cell_id_arr // depth
        cell_iy = cell_id_arr % depth
        centers = jnp.stack([
            (cell_ix.astype(dt) + 0.5) / depth,
            (cell_iy.astype(dt) + 0.5) / depth,
        ], axis=-1)                                     # (K, 2)

        cell_charges = segment_sum(charges, keys,
                                   num_segments=K)      # (K,)

        # Dipole moments: p_s = sum_{j in cell s} q_j * (x_j - c_s)
        disp_from_center = positions - centers[keys]   # (N, 2)
        dipoles = jnp.stack([
            segment_sum(charges * disp_from_center[:, d],
                        keys, num_segments=K)
            for d in range(2)
        ], axis=1).astype(dt)                           # (K, 2)

        # ── 3. Far-field: monopole + dipole at ALL cell centers ─────────
        # F_i = sum_s [Q_s * K(x_i, c_s) + dipole correction]
        # Computed via vmap over particles to bound peak memory.
        # (N, K, 2) would be ~320 MB at N=18k, K=2k; vmap keeps it at
        # (K, 2) per particle and lets XLA schedule the parallelism.)
        cc = centers        # (K, 2)  — captured by closure
        cq = cell_charges   # (K,)
        cd = dipoles        # (K, 2)

        def _far_one(pos_i):
            diff = pos_i - cc                        # (K, 2)
            r2 = jnp.sum(diff * diff, axis=-1)       # (K,)
            kernel = jnp.exp(-r2 / h2)               # (K,)
            pref = kernel * inv_h2                   # (K,)  2*G/h^2

            # Monopole: Q_s * 2*d/h^2 * G
            mono = jnp.einsum("k,kd->d", pref * cq, diff)  # (2,)

            # Dipole: 2*G/h^2 * [-p_s + 2*(p_s . d)*d/h^2]
            p_dot_d = jnp.sum(cd * diff, axis=-1)    # (K,)
            dip1 = jnp.sum(pref[:, None] * cd, axis=0)  # (2,)
            dip2 = jnp.sum((pref * p_dot_d)[:, None] * diff,
                           axis=0) * (dt.type(2.0) / h2)
            return mono - dip1 + dip2

        far_forces = jax.vmap(_far_one)(positions)        # (N, 2)

        # ── 4. Near-field: exact + mono+dipole subtraction ──────────────
        # Build padded cell arrays for the 25-offset ring sweep.
        cell_pos, cell_q, cell_orig, cell_valid = _build_padded_cells(
            positions, charges, keys, sort_idx,
            cell_counts, cell_starts, within_slot,
            depth, max_cell_size)

        # Per-particle cell indices (unsorted — original order)
        ix_p = jnp.clip((positions[:, 0] * depth).astype(jnp.int32),
                        0, depth - 1)
        iy_p = jnp.clip((positions[:, 1] * depth).astype(jnp.int32),
                        0, depth - 1)
        tgt_id = jnp.arange(N, dtype=jnp.int32)

        near_exact = jnp.zeros((N, 2), dtype=dt)
        near_approx = jnp.zeros((N, 2), dtype=dt)

        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                # Offset validity: skip out-of-bounds offsets instead of
                # clipping (clipping causes duplicate cell visits at
                # boundaries, amplifying the near-field correction).
                offset_valid = (
                    (ix_p + dx >= 0) & (ix_p + dx < depth)
                    & (iy_p + dy >= 0) & (iy_p + dy < depth)
                )                                          # (N,) bool

                # Neighbour cell (only meaningful where offset_valid)
                nbr_ix = jnp.clip(ix_p + dx, 0, depth - 1)
                nbr_iy = jnp.clip(iy_p + dy, 0, depth - 1)
                nbr_cell = nbr_ix * depth + nbr_iy          # (N,)

                # Gather padded cell data for each particle's neighbour
                nbr_pos = cell_pos[nbr_cell]                # (N, mcs, 2)
                nbr_q = cell_q[nbr_cell]                    # (N, mcs)
                nbr_orig = cell_orig[nbr_cell]              # (N, mcs)
                nbr_valid = cell_valid[nbr_cell]            # (N, mcs)

                # Self-mask (only relevant for offset (0,0))
                self_mask = nbr_orig != tgt_id[:, None]     # (N, mcs)
                pair_mask = nbr_valid & self_mask           # (N, mcs)
                # Zero out invalid offsets
                pair_mask = pair_mask & offset_valid[:, None]  # (N, mcs)

                # ── 4a. Mono+dipole subtraction for this offset ────────
                nbr_center = centers[nbr_cell]              # (N, 2)
                nbr_charge = cell_charges[nbr_cell]         # (N,)
                nbr_dipole = dipoles[nbr_cell]              # (N, 2)

                diff_mono = positions - nbr_center          # (N, 2)
                r2_mono = jnp.sum(diff_mono * diff_mono, axis=-1)  # (N,)
                kernel_mono = jnp.exp(-r2_mono / h2)        # (N,)
                pref_mono = kernel_mono * inv_h2            # (N,)

                # Monopole force: Q_s * 2*d/h^2 * G
                mono_force = (nbr_charge[:, None]
                              * pref_mono[:, None]
                              * diff_mono)                 # (N, 2)

                # Dipole force: 2*G/h^2 * [-p_s + 2*(p_s.d)*d/h^2]
                p_dot_d_mono = jnp.sum(nbr_dipole * diff_mono,
                                       axis=-1)             # (N,)
                dip_force = (
                    -pref_mono[:, None] * nbr_dipole
                    + pref_mono[:, None] * (dt.type(2.0) / h2)
                    * p_dot_d_mono[:, None] * diff_mono
                )                                          # (N, 2)
                # Zero out invalid offsets
                ov2 = offset_valid[:, None].astype(dt)     # (N, 1)
                near_approx = near_approx + (
                    mono_force + dip_force) * ov2

                # ── 4b. Exact particle-particle for this offset ─────────
                diff_pair = positions[:, None, :] - nbr_pos  # (N, mcs, 2)
                r2_pair = jnp.sum(diff_pair * diff_pair,
                                  axis=-1)                  # (N, mcs)
                kernel_pair = jnp.exp(-r2_pair / h2)        # (N, mcs)
                kernel_pair = jnp.where(pair_mask,
                                        kernel_pair,
                                        dt.type(0.0))

                pair_force = (
                    nbr_q[:, :, None]
                    * kernel_pair[:, :, None]
                    * diff_pair
                    * inv_h2
                )                                          # (N, mcs, 2)
                near_exact = near_exact + jnp.sum(
                    jnp.where(pair_mask[:, :, None],
                              pair_force,
                              dt.type(0.0)),
                    axis=1)                                 # (N, 2)

        # ── 5. Total: far - near_approx + near_exact ───────────────────
        forces = far_forces - near_approx + near_exact
        return forces

    # ─────────────────────────────────────────────────────────────────────
    # Convenience class (mirrors VectorizedGaussian2DFGT interface)
    # ─────────────────────────────────────────────────────────────────────

    class JaxGaussian2DFGT:
        """JAX GPU 2D Gaussian FGT with forces.

        Drop-in replacement for ``VectorizedGaussian2DFGT`` that runs
        entirely on-device and can be fused inside a larger JIT'd
        pipeline.  ``depth``, ``ring`` and ``max_cell_size`` are static
        (they size every padded buffer).

        Parameters
        ----------
        depth : int
            Grid resolution (cells per side).  Total cells = depth^2.
        h : float
            Gaussian bandwidth.  Kernel = exp(-r^2 / h^2).
        ring : int
            Near-field neighbourhood ring (default 2 = 5x5 box).
        max_cell_size : int
            Static pad for near-field cell occupancy.  Particles in
            cells exceeding this are silently dropped from the near
            field.  With adaptive depth (~8/cell average), 64 is
            generous; 128 is very safe.
        """

        def __init__(self, depth: int = 32, h: float = 0.2, ring: int = 2,
                     max_cell_size: int = 64):
            self.depth = int(depth)
            self.h = float(h)
            self.ring = int(ring)
            self.max_cell_size = int(max_cell_size)

        def evaluate_forces(self, positions: jnp.ndarray,
                            charges: jnp.ndarray) -> jnp.ndarray:
            """Compute repulsive forces.  Returns (N, 2) on-device."""
            return jax_gaussian2d_fgt_forces(
                positions, charges, self.h,
                self.depth, self.ring, self.max_cell_size)

else:
    # JAX not available
    jax_gaussian2d_fgt_forces = None
    JaxGaussian2DFGT = None
