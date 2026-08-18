"""
High-Performance, End-to-End Differentiable JAX Tree-Free FMM Engine
=====================================================================
Combines Vectorized Non-Reordering Open Addressing Hash (Farach-Colton et al. 2025)
with Carrier, Greengard, & Rokhlin (1988) Multipole Kernels (P2M, M2M, M2L, L2L, L2P, P2P)
and JAX Autodiff Gradients.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import time
import math

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


if HAS_JAX:
    # ---------------------------------------------------------------------------
    # 1. JAX Vectorized Non-Reordering Elastic Spatial Hash Table (Farach-Colton 2025)
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

    @jit
    def jax_multi_level_probe_lookup(
        keys_table: jnp.ndarray,
        query_keys: jnp.ndarray,
        level_offsets: jnp.ndarray,
        level_sizes: jnp.ndarray,
        seeds_a: jnp.ndarray,
        seeds_b: jnp.ndarray,
        num_levels: int = 4
    ) -> jnp.ndarray:
        def probe_single(key):
            result = -1
            for lvl in range(4):
                offset = level_offsets[lvl]
                size = level_sizes[lvl]
                for att in range(2):
                    h = jnp.bitwise_and(key * seeds_a[lvl, att] + seeds_b[lvl, att] + att * 2654435761, 0x7FFFFFFF) % size
                    pos = offset + h
                    hit = (keys_table[pos] == key)
                    result = jnp.where((result == -1) & hit, pos, result)
            return result

        return vmap(probe_single)(query_keys)

    # ---------------------------------------------------------------------------
    # 2. Differentiable CGR88 Multipole Operators in Complex Representation
    # ---------------------------------------------------------------------------
    @jit
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

    @jit
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

    @jit
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

    @jit
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

    @jit
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

    @jit
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

    # API Aliases
    jax_direct_nbody = jax_direct_nbody_reference
    jax_elastic_probe_lookup = jax_multi_level_probe_lookup
else:
    jax_morton_encode_2d = None
    jax_multi_level_probe_lookup = None
    jax_elastic_probe_lookup = None
    jax_p2m_expansion = None
    jax_m2m_translation = None
    jax_m2l_translation = None
    jax_l2l_translation = None
    jax_l2p_evaluation = None
    jax_l2p_force_evaluation = None
    jax_p2p_near_field = None
    jax_direct_nbody_reference = None
    jax_direct_nbody = None
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
