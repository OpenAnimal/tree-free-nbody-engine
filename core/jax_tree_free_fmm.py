"""
High-Performance, End-to-End Differentiable JAX Tree-Free FMM Engine
=====================================================================
Combines Vectorized Non-Reordering Open Addressing Hash (Farach-Colton et al. 2025)
with Complex Harmonic Multipole Kernels (P2M, M2L, L2P, P2P) and Autodiff Gradients.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import time

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    HAS_JAX = True
except ImportError:
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
        """
        Batched parallel probe lookup across multi-level geometric sub-arrays
        with bounded O(log 1/delta) worst-case search complexity.
        """
        def probe_single(key):
            result = -1
            # Probe sequentially through geometric funnel levels
            for lvl in range(4): # Unrolled 4 levels
                offset = level_offsets[lvl]
                size = level_sizes[lvl]
                # Try primary and secondary probes per level
                for att in range(2):
                    h = jnp.bitwise_and(key * seeds_a[lvl, att] + seeds_b[lvl, att] + att * 2654435761, 0x7FFFFFFF) % size
                    pos = offset + h
                    hit = (keys_table[pos] == key)
                    result = jnp.where((result == -1) & hit, pos, result)
            return result

        return vmap(probe_single)(query_keys)

    # ---------------------------------------------------------------------------
    # 2. Differentiable Multipole Operators in Complex Representation
    # ---------------------------------------------------------------------------
    @jit
    def jax_p2m_expansion(points: jnp.ndarray, charges: jnp.ndarray, center: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        P2M: Particle-to-Multipole expansion:
        a_0 = sum(q_i)
        a_k = - sum(q_i * (z_i - z_0)^k) / k  for k=1..order
        """
        z_pts = points[:, 0] + 1j * points[:, 1]
        z_c = center[0] + 1j * center[1]
        dz = z_pts - z_c # (N,)

        a0 = jnp.sum(charges)
        powers = jnp.arange(1, order + 1)
        dz_pow = dz[:, None] ** powers[None, :] # (N, order)
        ak = -jnp.sum(charges[:, None] * dz_pow, axis=0) / powers
        return jnp.concatenate([jnp.array([a0]), ak])

    @jit
    def jax_m2l_translation(multipole_coeffs: jnp.ndarray, center_src: jnp.ndarray, center_tgt: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        M2L: Multipole-to-Local translation:
        Translates distant multipole moments into a local Taylor series expansion at target center.
        """
        z0 = (center_src[0] - center_tgt[0]) + 1j * (center_src[1] - center_tgt[1])
        a0 = multipole_coeffs[0]
        ak = multipole_coeffs[1:order + 1]

        # Match the NumPy M2L convention used by core.tree_free_fmm.m2l.
        neg_z0 = -z0
        b0 = a0 * jnp.log(neg_z0) + jnp.sum(ak / (neg_z0 ** jnp.arange(1, order + 1)))
        l = jnp.arange(1, order + 1)
        k = jnp.arange(1, order + 1)
        monopole_terms = ((-1) ** l) * a0 / (l * (z0 ** l))
        higher_terms = jnp.sum(ak[None, :] / (neg_z0 ** (k[:, None] + l[None, :])), axis=1)
        return jnp.concatenate([jnp.asarray([b0]), monopole_terms + higher_terms])

    @jit
    def jax_l2p_evaluation(local_coeffs: jnp.ndarray, points: jnp.ndarray, center_tgt: jnp.ndarray, order: int = 6) -> jnp.ndarray:
        """
        L2P: Local-to-Particle potential evaluation from local Taylor series.
        """
        z_pts = points[:, 0] + 1j * points[:, 1]
        z_c = center_tgt[0] + 1j * center_tgt[1]
        dz = z_pts - z_c

        powers = jnp.arange(order + 1)
        dz_pow = dz[:, None] ** powers[None, :]
        phi_complex = jnp.sum(local_coeffs[None, :] * dz_pow, axis=-1)
        return jnp.real(phi_complex)

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
        # The per-particle reference potentials count every pair twice.
        return -0.5 * grad(total_potential_energy)(positions)

    # API Aliases for benchmark suite compatibility
    jax_direct_nbody = jax_direct_nbody_reference
    jax_elastic_probe_lookup = jax_multi_level_probe_lookup
else:
    jax_morton_encode_2d = None
    jax_multi_level_probe_lookup = None
    jax_elastic_probe_lookup = None
    jax_p2m_expansion = None
    jax_m2l_translation = None
    jax_l2p_evaluation = None
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


if __name__ == "__main__":
    benchmark_jax_engine()
