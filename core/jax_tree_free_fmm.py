"""
JAX Adaptive FMM Operator Library + Differentiable Dense Reference (`jax_tree_free_fmm.py`)
====================================================================================
Verified JAX implementations of the Carrier, Greengard, & Rokhlin (1988) multipole
operator primitives (P2M, M2M, M2L, L2L, L2P, P2P) plus an exact O(N^2) dense
reference and its reverse-mode autodiff gradient.

Honest scope (Round-7 audit, finding F-06):
- This module ships verified adaptive FMM operator primitives and a differentiable
  dense reference. It does NOT contain an assembled JAX FMM pipeline (no
  upward/downward pass driver wiring the operators together over a tree).
  The earlier "end-to-end-differentiable JAX Tree-Free FMM Engine" wording
  overstated this; the assembled pipeline is task T-D4 of the Round-7 plan.
- The legacy `jax_multi_level_probe_lookup` (2-slot-per-level probe, the
  pre-funnel scheme that `core/elastic_hash.py` disavows) has been removed.
  JAX with x64-disabled cannot express the 64-bit funnel mixer; the funnel
  hash stays CPU/Zig/WGSL only (see the `funnel_probe` docstring in `core/elastic_hash.py`).
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import time
import math

try:
    import jax
    # The adaptive FMM M2L/M2M operators and the assembled flat FMM pipeline rely on
    # complex128 expansions (jnp.log of complex deltas, high-order binomial
    # sums). The acceptance tolerance is rel-L2 <= 1e-6 vs an f64 direct
    # reference, which is unreachable in float32/complex64 roundoff. Enable
    # x64 before any jax op is built.
    jax.config.update("jax_enable_x64", True)  # process-wide x64 (required: FMM coefficients underflow in f32)
    import jax.numpy as jnp
    from jax import jit, vmap, grad, lax
    # segment_sum's public path has moved across JAX releases
    # (jax.ops -> jax.lax -> jax._src.ops.scatter). Try them in order;
    # swallowing an ImportError here would silently set HAS_JAX=False and
    # make every JAX test skip-while-passing even with JAX installed
    # (audit finding: jax 0.10.2 ships no jax.lax.segment_sum).
    try:
        from jax.lax import segment_sum
    except ImportError:  # pragma: no cover - JAX >= 0.10 layout
        from jax._src.ops.scatter import segment_sum
    from functools import partial
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False
    partial = None
    segment_sum = None


if HAS_JAX:
    # ---------------------------------------------------------------------------
    # 1. JAX Vectorized Non-Reordering Elastic Spatial Hash Table (Farach-Colton, Krapivin, & Kuszmaul, 2025)
    # ---------------------------------------------------------------------------
    @jit
    def jax_morton_encode_2d(x: jnp.ndarray, y: jnp.ndarray, depth: int = 5) -> jnp.ndarray:
        """Vectorized Morton encoding on GPU/TPU."""
        grid_res = 1 << depth
        ix = jnp.clip((x * grid_res).astype(jnp.int32), 0, grid_res - 1)
        iy = jnp.clip((y * grid_res).astype(jnp.int32), 0, grid_res - 1)

        def spread_bits(v):
            v = jnp.bitwise_and(jnp.bitwise_or(v, jnp.left_shift(v, 8)), 0x00FF00FF)
            v = jnp.bitwise_and(jnp.bitwise_or(v, jnp.left_shift(v, 4)), 0x0F0F0F0F)
            v = jnp.bitwise_and(jnp.bitwise_or(v, jnp.left_shift(v, 2)), 0x33333333)
            v = jnp.bitwise_and(jnp.bitwise_or(v, jnp.left_shift(v, 1)), 0x55555555)
            return v

        return jnp.bitwise_or(spread_bits(ix), jnp.left_shift(spread_bits(iy), 1))

    # ---------------------------------------------------------------------------
    # 2. Differentiable adaptive FMM Multipole Operators in Complex Representation
    # ---------------------------------------------------------------------------
    # `order` is declared static (static_argnums=(3,)) on every operator below
    # because the M2M/M2L/L2L bodies use Python `for l in range(1, order+1)`
    # loops and math.comb -- these require a concrete Python int, so passing
    # `order` as a traced value would crash at trace time. P2M/L2P/L2P-force
    # also slice `coeffs[1:order+1]` / build `jnp.arange(order+1)`, which need
    # a static bound too, so they are treated identically for consistency.
    @partial(jit, static_argnums=(3,))
    def jax_p2m_expansion(points: jnp.ndarray, charges: jnp.ndarray, center: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        P2M: Particle-to-Multipole expansion (CGR88 Eq. 2.1 - 2.2):
        a_0 = sum(q_i)
        a_k = - sum(q_i * (z_i - z_0)^k) / k  for k=1..order
        """
        z_pts = points[:, 0] + 1j * points[:, 1]
        z_c = center[0] + 1j * center[1]
        dz = z_pts - z_c

        a0 = jnp.sum(charges)
        powers = jnp.arange(1, order + 1)
        dz_pow = dz[:, None] ** powers[None, :]
        ak = -jnp.sum(charges[:, None] * dz_pow, axis=0) / powers
        return jnp.concatenate([jnp.array([a0]), ak])

    @partial(jit, static_argnums=(3,))
    def jax_m2m_translation(m_coeffs: jnp.ndarray, center_child: jnp.ndarray, center_parent: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        M2M: Multipole-to-Multipole translation (CGR88 Theorem 2.2).
        b_0 = a_0
        b_l = - a_0 * delta^l / l + sum_{k=1}^l a_k * binom(l-1, k-1) * delta^(l-k)
        """
        delta = (center_child[0] - center_parent[0]) + 1j * (center_child[1] - center_parent[1])
        b0 = m_coeffs[0]
        
        # Build binomial coefficient matrix on host or static
        # For JIT, we compute terms
        b_list = [b0]
        for l in range(1, order + 1):
            term = -b0 * (delta ** l) / l
            for k in range(1, l + 1):
                binom_val = math.comb(l - 1, k - 1)
                term = term + m_coeffs[k] * binom_val * (delta ** (l - k))
            b_list.append(term)
        return jnp.stack(b_list)

    @partial(jit, static_argnums=(3,))
    def jax_m2l_translation(multipole_coeffs: jnp.ndarray, center_src: jnp.ndarray, center_tgt: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        M2L: Multipole-to-Local translation (CGR88 Theorem 2.3).
        delta = center_tgt - center_src
        c_0 = a_0 * ln(delta) + sum_{k=1}^p a_k / delta^k
        c_l = (a_0 * (-1)^(l-1)) / (l * delta^l) + sum_{k=1}^p [ (-1)^l * binom(k+l-1, l) * a_k ] / delta^(k+l)
        """
        delta = (center_tgt[0] - center_src[0]) + 1j * (center_tgt[1] - center_src[1])
        a0 = multipole_coeffs[0]
        ak = multipole_coeffs[1:order + 1]

        k_idx = jnp.arange(1, order + 1)
        c0 = a0 * jnp.log(delta) + jnp.sum(ak / (delta ** k_idx))

        c_list = [c0]
        for l in range(1, order + 1):
            term = a0 * ((-1.0) ** (l - 1)) / (l * (delta ** l))
            for k in range(1, order + 1):
                binom_factor = ((-1.0) ** l) * float(math.comb(k + l - 1, l))
                term = term + binom_factor * ak[k - 1] / (delta ** (k + l))
            c_list.append(term)
        return jnp.stack(c_list)

    @partial(jit, static_argnums=(3,))
    def jax_l2l_translation(local_coeffs: jnp.ndarray, center_src: jnp.ndarray, center_dst: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        L2L: Local-to-Local translation (CGR88 Theorem 2.4).
        d_l = sum_{k=l}^p c_k * binom(k, l) * delta^(k-l)
        """
        delta = (center_dst[0] - center_src[0]) + 1j * (center_dst[1] - center_src[1])
        d_list = []
        for l in range(order + 1):
            term = 0.0 + 0.0j
            for k in range(l, order + 1):
                binom_val = float(math.comb(k, l))
                term = term + local_coeffs[k] * binom_val * (delta ** (k - l))
            d_list.append(term)
        return jnp.stack(d_list)

    @partial(jit, static_argnums=(3,))
    def jax_l2p_evaluation(local_coeffs: jnp.ndarray, points: jnp.ndarray, center_tgt: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        L2P: Local-to-Particle potential evaluation: Phi(z) = Re( sum_{l=0}^p c_l * (z - z_0)^l ).
        """
        z_pts = points[:, 0] + 1j * points[:, 1]
        z_c = center_tgt[0] + 1j * center_tgt[1]
        dz = z_pts - z_c

        powers = jnp.arange(order + 1)
        dz_pow = dz[:, None] ** powers[None, :]
        phi_complex = jnp.sum(local_coeffs[None, :] * dz_pow, axis=-1)
        return jnp.real(phi_complex)

    @partial(jit, static_argnums=(3,))
    def jax_l2p_force_evaluation(local_coeffs: jnp.ndarray, points: jnp.ndarray, center_tgt: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        L2P Vector Force evaluation: F = ( -Re(Psi'), Im(Psi') )
        where Psi'(z) = sum_{l=1}^p l * c_l * (z - z0)^(l-1).
        """
        z_pts = points[:, 0] + 1j * points[:, 1]
        z_c = center_tgt[0] + 1j * center_tgt[1]
        dz = z_pts - z_c

        l_idx = jnp.arange(1, order + 1)
        dz_pow = dz[:, None] ** (l_idx[None, :] - 1)
        deriv = jnp.sum(l_idx[None, :] * local_coeffs[None, 1:order + 1] * dz_pow, axis=-1)
        fx = -jnp.real(deriv)
        fy = jnp.imag(deriv)
        return jnp.stack([fx, fy], axis=-1)

    @jit
    def jax_p2p_near_field(points_tgt: jnp.ndarray, points_src: jnp.ndarray, charges_src: jnp.ndarray, softening: float = 1e-4) -> jnp.ndarray:
        """
        P2P: Vectorized near-field direct potential: Phi_i = sum_j q_j * log(r_ij + eps).
        """
        diff = points_tgt[:, None, :] - points_src[None, :, :]
        r_sq = jnp.sum(diff ** 2, axis=-1) + (softening ** 2)
        r = jnp.sqrt(r_sq)
        pot = jnp.sum(charges_src[None, :] * jnp.log(r), axis=-1)
        return pot

    @jit
    def jax_direct_nbody_reference(positions: jnp.ndarray, charges: jnp.ndarray, softening: float = 1e-4) -> jnp.ndarray:
        """Exact O(N^2) reference potential for verification."""
        N = positions.shape[0]
        diff = positions[:, None, :] - positions[None, :, :]
        r_sq = jnp.sum(diff ** 2, axis=-1) + (softening ** 2)
        r = jnp.sqrt(r_sq)
        eye = jnp.eye(N)
        r_diag_safe = r * (1.0 - eye) + eye
        pot = jnp.sum(charges[None, :] * jnp.log(r_diag_safe) * (1.0 - eye), axis=-1)
        return pot

    # Differentiable force evaluator via JAX automatic differentiation
    def compute_nbody_forces_jax(positions: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        """Computes all-pairs forces F = -grad(Phi_total) via reverse-mode autodiff."""
        def total_potential_energy(pos):
            return jnp.sum(jax_direct_nbody_reference(pos, charges))
        return -0.5 * grad(total_potential_energy)(positions)

    # ---------------------------------------------------------------------------
    # 3b. Differentiable CSR cell-list pair sweep (near-field primitive)
    # ---------------------------------------------------------------------------
    # Exact ragged block sweep over CSR row-runs: `lax.while_loop` with a
    # device-side (data-dependent) trip count gives exact, capacity-free
    # enumeration of the ring-2 neighborhood. JAX cannot reverse-differentiate
    # such a loop directly, so the transpose is supplied analytically via
    # jax.custom_vjp, exploiting that the log kernel is SYMMETRIC (K_ij=K_ji)
    # and the near pair set is symmetric (Chebyshev ring, self pairs excluded):
    #   s0_i = sum_j w_j log(r_ij)          (scalar-kernel sweep)
    #   A_i  = sum_j w_j (x_i - x_j)/r_ij^2 (vector-kernel sweep, same loop)
    # For near_pot = s0(pos, q) and loss cotangent t:
    #   dL/dq = s0(pos, t)                    (K^T t = K t)
    #   dL/dx = t * A(q) + q * A(t)           (target + source side; verified
    #                                           by finite differences in
    #                                           tests/core/test_jax_pipeline.py)
    # Forward-mode (jacfwd) is NOT defined for this primitive (defvjp only);
    # reverse-mode through the near field is exact.

    def _csr_pair_sweep_impl(pos, w, rs_all, rl_all, perm, block_ar, soft_sq):
        """Ragged block sweep over the CSR row-runs of the near field.

        Parameters (pos/w differentiable; the rest are bookkeeping arrays):
          pos : (N, 2) targets;  w : (N,) source weights (charges)
          rs_all, rl_all : (n_offsets, N) start/length of each target's
            row-runs in the cell-key-sorted order (n_offsets = 2*ring+1)
          perm : (N,) sorted->original particle id (stable argsort by key)
          block_ar : (block,) 0..block-1 (static-sized lane index vector)
          soft_sq : softening^2 added to r^2 (matches direct reference)

        Returns (s0, A): the scalar- and vector-kernel sweeps described above.
        The while_loop trip count is the max block count over targets/offsets
        -- resolved on device, so no per-cell capacity is assumed and no
        source is ever dropped or double counted.
        """
        N = pos.shape[0]
        # Sentinel padding: when `off` advances past the last real offset the
        # cond/body gathers must read ZERO-length runs so the loop exits --
        # JAX gathers CLAMP out-of-bounds indices instead of erroring, so an
        # unpadded (n_off, N) stack would re-read the last row forever.
        zero_row = jnp.zeros((1, pos.shape[0]), dtype=rl_all.dtype)
        rs_p = jnp.concatenate([rs_all, zero_row])
        rl_p = jnp.concatenate([rl_all, zero_row])
        n_off = rs_all.shape[0] + 1     # one past the last real offset
        s0 = jnp.zeros(N, dtype=pos.dtype)
        A = jnp.zeros_like(pos)
        off = jnp.zeros((), dtype=jnp.int32)
        c = jnp.zeros(N, dtype=jnp.int32)
        tgt_id = jnp.arange(N)
        block = block_ar.shape[0]

        def _cond(state):
            _, _, o, c = state
            return (o < (n_off - 1)) | jnp.any(c < rl_p[o])

        def _body(state):
            s0, A, o, c = state
            rs = rs_p[o]
            rl = rl_p[o]
            idx = rs[:, None] + c[:, None] + block_ar[None, :]   # (N, block)
            valid = block_ar[None, :] < (rl - c)[:, None]
            idx_c = jnp.minimum(idx, N - 1)          # safe gather for padding
            src = perm[idx_c]                         # original particle id
            m = valid & (src != tgt_id[:, None])      # padding + self pairs
            d = pos[:, None, :] - pos[src]            # (N, block, 2)
            r2 = jnp.sum(d * d, axis=-1) + soft_sq
            r2_safe = jnp.where(m, r2, 1.0)           # log(1)=0 padding lanes
            s0 = s0 + jnp.sum(w[src] * jnp.log(jnp.sqrt(r2_safe)) * m, axis=1)
            A = A + jnp.sum((w[src] * m / r2_safe)[:, :, None] * d, axis=1)
            c_new = c + block
            exhausted = ~jnp.any(c_new < rl)
            return s0, A, o + exhausted, jnp.where(exhausted, 0, c_new)

        s0, A, _, _ = lax.while_loop(_cond, _body, (s0, A, off, c))
        return s0, A

    def _csr_pair_sweep_s0(pos, w, rs_all, rl_all, perm, block_ar, soft_sq):
        """Scalar-kernel sweep only (the near-field potential).

        The bookkeeping args (rs_all, rl_all, perm, block_ar) are integer
        arrays (non-differentiable dtype); `soft_sq` is treated as a constant
        (no gradient w.r.t. the softening is provided)."""
        s0, _ = _csr_pair_sweep_impl(pos, w, rs_all, rl_all, perm, block_ar,
                                     soft_sq)
        return s0

    _csr_pair_sweep = jax.custom_vjp(_csr_pair_sweep_s0)

    def _csr_pair_sweep_fwd(pos, w, rs_all, rl_all, perm, block_ar, soft_sq):
        s0, A_w = _csr_pair_sweep_impl(pos, w, rs_all, rl_all, perm, block_ar,
                                       soft_sq)
        # Residuals carry the bookkeeping arrays through to the transpose.
        return s0, (pos, w, A_w, rs_all, rl_all, perm, block_ar, soft_sq)

    def _csr_pair_sweep_bwd(res, t):
        pos, w, A_w, rs_all, rl_all, perm, block_ar, soft_sq = res
        s0_t, A_t = _csr_pair_sweep_impl(pos, t, rs_all, rl_all, perm,
                                         block_ar, soft_sq)
        vjp_pos = t[:, None] * A_w + w[:, None] * A_t
        return vjp_pos, s0_t, None, None, None, None, None

    _csr_pair_sweep.defvjp(_csr_pair_sweep_fwd, _csr_pair_sweep_bwd)

    # ---------------------------------------------------------------------------
    # 4. Assembled flat-scheme 2D log-kernel FMM pipeline (Round-7 task T-D4)
    # ---------------------------------------------------------------------------
    # A single-level flat scheme: bin particles on a uniform depth x depth grid,
    # P2M per occupied cell, M2L over all well-separated (K,K) cell pairs,
    # L2P per particle, near field = CSR cell-list direct sum over ring-2
    # row-runs of the cell-sorted order (O(N * 25 * occupancy) work instead of
    # the previous dense O(N^2) masked pairwise tensor; see step 5 below).
    # The funnel hash stays CPU/WGSL (documented above); on-device binning uses
    # jnp.argsort (the standard accelerator route). K <= depth^2 static-bounded.

    @partial(jit, static_argnums=(2, 3))
    def jax_flat_fmm_evaluate(pos: jnp.ndarray, q: jnp.ndarray,
                              depth: int = 5, order: int = 8,
                              softening: float = 0.0) -> jnp.ndarray:
        """Assembled flat 2D log-kernel FMM (adaptive FMM complex form).

        Single-level flat scheme on a `depth` x `depth` uniform grid. The
        near field is an exact direct sum over the ring-2 (5x5) cell
        neighborhood, evaluated with a CSR cell-list (spatial-hash) sweep
        instead of a dense (N, N) masked pairwise tensor (the Round-9/10
        O(N^2) residual risk); the far field is the adaptive FMM M2L
        re-expansion evaluated per particle via L2P. `depth` and `order` are
        JIT static (they size every padded buffer); `softening` is a dynamic
        scalar (added to r^2 in the near field, matching
        `jax_direct_nbody_reference`).

        Near-field mechanism (mirrors `core/_csr.py` + `core/csr_p2p.py`):
        particles are counting-sorted by raw cell key (`jnp.argsort` +
        `segment_sum`/`cumsum`), so each cell's particle ids form one
        contiguous run of the sorted order. A target's ring-2 (5x5)
        neighborhood is exactly 5 such row-runs (rows iy-2..iy+2, columns
        clamped to the grid). Each ragged run is consumed in fixed-size
        blocks by a `lax.while_loop` whose trip count is data-dependent but
        resolved ON DEVICE -- there is no host-side control flow on tracer
        values, no per-cell capacity assumption (a cell with any occupancy
        is swept exactly), and static block size + per-block validity masks
        follow the repo's padded-buffer/mask pattern. The pair SET and the
        per-pair kernel q_j * log(sqrt(r_ij^2 + s^2)) are identical to the
        previous dense implementation; only the floating-point summation
        order differs (machine-noise level).

        Index space (Round-7 audit fix): ALL cell bookkeeping uses ONE
        consistent space -- the raw row-major cell key in [0, depth^2). Cell
        centers, multipole moments, local expansions and per-particle cell
        ids are all addressed by this same key, so unoccupied cells simply
        carry zero moments and a well-defined (unused) center. The previous
        implementation mixed compact occupied-key ranks (cell centers, far
        mask source coords) with raw keys (cell_start, M_all, inverse),
        which computed P2M moments about the wrong centers whenever any cell
        was unoccupied, and sliced `z_sorted[start:end]` with vmapped
        (traced) bounds -- an impossible index in JAX.

        Parameters
        ----------
        pos : (N, 2) float64 in [0, 1)^2
        q : (N,) charges
        depth : int -- cells per side (linear, T-C8 semantics)
        order : int -- Taylor expansion order
        softening : float -- near-field r^2 softening (matches direct ref)

        Returns
        -------
        pot : (N,) float64 -- per-particle potential (near + far)
        """
        N = pos.shape[0]
        grid_res = depth
        h = 1.0 / depth
        max_K = depth * depth
        ring = 2

        # 1. Bin: quantize to row-major cell keys (the single index space).
        ix = jnp.clip((pos[:, 0] * grid_res).astype(jnp.int32), 0, grid_res - 1)
        iy = jnp.clip((pos[:, 1] * grid_res).astype(jnp.int32), 0, grid_res - 1)
        keys = iy * grid_res + ix            # (N,) raw key per particle
        key_mask = jnp.zeros(max_K, dtype=jnp.bool_).at[keys].set(True)

        # Cell centers for EVERY raw key (well-defined for unoccupied cells
        # too; their moments are zero so the center is never used in a sum).
        cell_id = jnp.arange(max_K)
        cell_ix = cell_id % grid_res
        cell_iy = cell_id // grid_res
        centers = jnp.stack([
            (cell_ix.astype(jnp.float64) + 0.5) * h,
            (cell_iy.astype(jnp.float64) + 0.5) * h,
        ], axis=-1)                          # (max_K, 2)
        centers_c = centers[:, 0] + 1j * centers[:, 1]   # (max_K,) complex

        # 2. P2M via segment_sum over the raw-key index space (no slicing,
        # no vmap over traced bounds). Moments are computed about the cell
        # center indexed by the SAME raw key as the segment id.
        z_pts = pos[:, 0] + 1j * pos[:, 1]               # (N,)
        dz_p = z_pts - centers_c[keys]                   # (N,) per-particle
        a0 = segment_sum(q, keys, num_segments=max_K)    # (max_K,)
        powers = jnp.arange(1, order + 1)
        dz_pow = dz_p[:, None] ** powers[None, :]         # (N, order)
        weighted = q[:, None] * dz_pow                    # (N, order)
        ak = -segment_sum(weighted, keys, num_segments=max_K) / powers  # (max_K, order)
        M_all = jnp.concatenate([a0[:, None], ak], axis=1)  # (max_K, order+1)
        M_all = M_all * key_mask[:, None]                 # zero unoccupied (safe)

        # 3. M2L: for each target raw key, sum local expansions from all
        # well-separated (Chebyshev > ring) occupied source cells. Vectorized
        # over sources; vmap over targets. delta_safe avoids log(0)/0**k for
        # self / non-far pairs before the mask zeroes their contribution.
        t_ix = cell_ix[:, None]
        t_iy = cell_iy[:, None]
        s_ix = cell_ix[None, :]
        s_iy = cell_iy[None, :]
        cheb = jnp.maximum(jnp.abs(s_ix - t_ix), jnp.abs(s_iy - t_iy))  # (max_K, max_K)
        far_mask = (cheb > ring) & key_mask[None, :]      # (max_K, max_K) [tgt, src]

        def m2l_for_target(t_idx):
            s_idx = cell_id                              # (max_K,)
            mask = far_mask[t_idx, :]                    # (max_K,)
            delta = centers_c[t_idx] - centers_c[s_idx]  # (max_K,)
            delta_safe = jnp.where(mask, delta, 1.0 + 0.0j)
            a0_s = M_all[s_idx, 0]                       # (max_K,)
            ak_s = M_all[s_idx, 1:order + 1]             # (max_K, order)
            k_idx = jnp.arange(1, order + 1)
            c0 = a0_s * jnp.log(delta_safe) + \
                jnp.sum(ak_s / (delta_safe[:, None] ** k_idx[None, :]), axis=1)
            c_list = [c0]
            for l in range(1, order + 1):
                term = a0_s * ((-1.0) ** (l - 1)) / (l * (delta_safe ** l))
                for k in range(1, order + 1):
                    binom_factor = ((-1.0) ** l) * float(math.comb(k + l - 1, l))
                    term = term + binom_factor * ak_s[:, k - 1] / (delta_safe ** (k + l))
                c_list.append(term)
            L = jnp.stack(c_list, axis=1)                # (max_K, order+1)
            return jnp.sum(L * mask[:, None], axis=0)    # (order+1,)

        L_all = jax.vmap(m2l_for_target)(cell_id)        # (max_K, order+1)
        L_all = L_all * key_mask[:, None]

        # 4. L2P: per particle, evaluate its cell's local expansion.
        def l2p_one(p_idx):
            c = keys[p_idx]
            dz = z_pts[p_idx] - centers_c[c]
            pw = jnp.arange(order + 1)
            phi = jnp.sum(L_all[c, :] * dz ** pw)
            return jnp.real(phi)

        far_pot = jax.vmap(l2p_one)(jnp.arange(N))

        # 5. Near field: exact direct sum over the ring-2 (5x5) cell
        # neighborhood, evaluated as a CSR cell-list (spatial-hash) sweep
        # instead of the previous dense (N, N, 2) masked pairwise tensor
        # (the Round-9/10 O(N^2) residual risk). Same pair set, same kernel
        # q_j * log(sqrt(r_ij^2 + s^2)); only summation order differs.
        #
        # 5a. Counting sort by raw cell key (JAX mirror of core/_csr.py
        # build_csr): stable argsort groups each cell's particle ids into
        # one contiguous run; `cum`/`start` give run boundaries per key.
        perm = jnp.argsort(keys, stable=True)            # (N,) sorted->orig
        cnt = segment_sum(jnp.ones(N, dtype=jnp.int32), keys,
                          num_segments=max_K)              # (max_K,)
        cum = jnp.cumsum(cnt)                              # inclusive prefix
        start = cum - cnt                                  # exclusive prefix
        # run_end[k] = #particles with key <= k  (cum padded for k = max_K-1)
        run_end_padded = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), cum])

        # 5b. Per target, the ring-2 neighborhood is 5 contiguous row-runs of
        # the sorted order (rows iy-2..iy+2, columns ix-2..ix+2 clamped to
        # the grid) -- the same cells the dense near_mask selected via
        # Chebyshev distance <= ring on the (already clipped) cell indices.
        ix_t = keys % grid_res
        iy_t = keys // grid_res
        ix_lo = jnp.maximum(ix_t - ring, 0)
        ix_hi = jnp.minimum(ix_t + ring, grid_res - 1)

        rs_list = []
        rl_list = []
        for dy in range(-ring, ring + 1):
            row = iy_t + dy
            row_ok = (row >= 0) & (row <= grid_res - 1)
            row_c = jnp.where(row_ok, row, 0)
            k_lo = row_c * grid_res + ix_lo               # first key in run
            k_hi = row_c * grid_res + ix_hi               # last key in run
            rs_list.append(start[k_lo])                   # (N,) run start
            rl_list.append(jnp.where(
                row_ok,
                run_end_padded[k_hi + 1] - run_end_padded[k_lo],
                0))                                       # (N,) run length
        rs_all = jnp.stack(rs_list)                       # (2*ring+1, N)
        rl_all = jnp.stack(rl_list)

        # Static source-block size (shapes must be static under jit): sized
        # from N and max_K (both static) for the expected row-run length
        # 5*N/max_K. Exactness never depends on it -- the while_loop inside
        # _csr_pair_sweep sweeps every run to its true dynamic length; the
        # block size only sets work granularity, and per-block masks drop
        # padding lanes.
        expected_run = 5.0 * N / max_K
        block = max(16, min(512, 1 << int(math.ceil(
            math.log2(max(1.0, expected_run))))))
        block_ar = jnp.arange(block)

        # 5c. Differentiable ragged block sweep (see section 3b above).
        near_pot = _csr_pair_sweep(pos, q, rs_all, rl_all, perm, block_ar,
                                   softening ** 2)

        return far_pot + near_pot

    # API Aliases
    jax_direct_nbody = jax_direct_nbody_reference
else:
    jax_morton_encode_2d = None
    jax_p2m_expansion = None
    jax_m2m_translation = None
    jax_m2l_translation = None
    jax_l2l_translation = None
    jax_l2p_evaluation = None
    jax_l2p_force_evaluation = None
    jax_p2p_near_field = None
    jax_direct_nbody_reference = None
    jax_direct_nbody = None
    jax_flat_fmm_evaluate = None
    compute_nbody_forces_jax = None


def benchmark_jax_engine():
    if not HAS_JAX:
        print("[INFO] JAX is not installed in the active environment. Run `pip install jax jaxlib` to run JIT GPU kernels.")
        return

    print("=" * 70)
    print(">>> BENCHMARKING JAX VECTORIZED TREE-FREE FMM & AUTODIFF ENGINE")
    print("=" * 70)

    N = 4000
    order = 6
    print(f"[*] Compiling JAX JIT kernels for N = {N:,} particles (Order p = {order})...")

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    pos = jax.random.uniform(k1, shape=(N, 2), minval=0.05, maxval=0.95)
    charges = jax.random.uniform(k2, shape=(N,), minval=-1.0, maxval=1.0)

    # 1. Warmup & JIT Compile
    _ = jax_direct_nbody_reference(pos[:100], charges[:100]).block_until_ready()
    center = jnp.array([0.5, 0.5])
    m_coeffs = jax_p2m_expansion(pos[:100], charges[:100], center, order=order).block_until_ready()

    # 2. Benchmark Multipole Expansion (P2M)
    t0 = time.perf_counter()
    m_coeffs = jax_p2m_expansion(pos, charges, center, order=order).block_until_ready()
    t_p2m = (time.perf_counter() - t0) * 1000.0
    print(f"[-] JAX Vectorized P2M Expansion ({N:,} pts): {t_p2m:.3f} ms")

    # 3. Benchmark Local Evaluation (L2P)
    l_coeffs = jnp.ones(order + 1, dtype=jnp.complex64)
    t0 = time.perf_counter()
    phi_eval = jax_l2p_evaluation(l_coeffs, pos, center, order=order).block_until_ready()
    t_l2p = (time.perf_counter() - t0) * 1000.0
    print(f"[-] JAX Vectorized L2P Evaluation ({N:,} pts): {t_l2p:.3f} ms")

    # 4. Benchmark Direct N-Body
    t0 = time.perf_counter()
    pot_direct = jax_direct_nbody_reference(pos, charges).block_until_ready()
    t_direct = (time.perf_counter() - t0) * 1000.0
    print(f"[-] JAX Exact N-Body Potential ({N:,} pts):   {t_direct:.2f} ms")

    # 5. Differentiable Autodiff Forces
    t0 = time.perf_counter()
    forces = compute_nbody_forces_jax(pos[:500], charges[:500]).block_until_ready()
    t_grad = (time.perf_counter() - t0) * 1000.0
    print(f"[-] JAX Reverse-Mode Autodiff Forces (500 pts): {t_grad:.2f} ms | Shape: {forces.shape}")
    print("=" * 70)
