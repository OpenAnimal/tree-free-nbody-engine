/**
 * Ultra-Fast Tree-Free Fast Multipole Method (FMM) CUDA/HIP C++ Kernel
 * ====================================================================
 * Combines Lock-Free Non-Reordering Open Addressing Hash Table (Farach-Colton et al. 2025)
 * with Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole Expansions (P2M, M2L, L2P)
 * and Fused Shared-Memory P2P Interaction Tiles.
 *
 * Target Architectures:
 *  - NVIDIA Ampere (sm_80), Ada Lovelace (sm_89), Hopper (sm_90), Blackwell (sm_100)
 *  - AMD RDNA 2/3/4 (gfx1030/gfx1100/gfx1200) and CDNA 1/2/3 via HIP compilation
 */

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define WARP_SIZE 64
#else
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#define WARP_SIZE 32
#endif

#include <stdint.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define MAX_LEVELS 4
#define MAX_ORDER 8

// Structure for 2D/3D particle state
struct __align__(16) Particle2D {
    float x, y;
    float q;
    float pad;
};

struct ComplexFloat {
    float r, i;
};

__device__ __forceinline__ ComplexFloat c_make(float r, float i) {
    ComplexFloat z; z.r = r; z.i = i; return z;
}

__device__ __forceinline__ ComplexFloat c_add(ComplexFloat a, ComplexFloat b) {
    return c_make(a.r + b.r, a.i + b.i);
}

__device__ __forceinline__ ComplexFloat c_sub(ComplexFloat a, ComplexFloat b) {
    return c_make(a.r - b.r, a.i - b.i);
}

__device__ __forceinline__ ComplexFloat c_mul(ComplexFloat a, ComplexFloat b) {
    return c_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

__device__ __forceinline__ ComplexFloat c_div(ComplexFloat a, ComplexFloat b) {
    float den = b.r * b.r + b.i * b.i + 1e-18f;
    return c_make((a.r * b.r + a.i * b.i) / den, (a.i * b.r - a.r * b.i) / den);
}

__device__ __forceinline__ ComplexFloat c_log(ComplexFloat a) {
    float r = sqrtf(a.r * a.r + a.i * a.i + 1e-18f);
    float theta = atan2f(a.i, a.r);
    return c_make(logf(r), theta);
}

// ---------------------------------------------------------------------------
// 1. Device Morton 2D Coordinate Encoding
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t expand_bits_2d(uint32_t v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__device__ __forceinline__ uint32_t morton_encode_2d_dev(float x, float y, int depth) {
    uint32_t grid_res = 1U << depth;
    uint32_t ix = fminf(fmaxf(x * grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iy = fminf(fmaxf(y * grid_res, 0.0f), (float)(grid_res - 1));
    return expand_bits_2d(ix) | (expand_bits_2d(iy) << 1);
}

// ---------------------------------------------------------------------------
// 2. Lock-Free Non-Reordering Elastic Hash Table Insert (atomicCAS)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hash_probe_dev(uint32_t key, uint32_t seed_a, uint32_t seed_b, uint32_t size, int attempt) {
    return ((key * seed_a + seed_b + attempt * 2654435761U) & 0x7FFFFFFFU) % size;
}

__global__ void insert_particles_elastic_hash_2d_kernel(
    const Particle2D* __restrict__ particles,
    int num_particles,
    uint32_t* __restrict__ table_keys,
    int* __restrict__ table_head_indices,
    int* __restrict__ particle_next_ptrs,
    const uint32_t* __restrict__ level_offsets,
    const uint32_t* __restrict__ level_sizes,
    const uint32_t* __restrict__ seeds_a,
    const uint32_t* __restrict__ seeds_b,
    int depth
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles) return;

    Particle2D p = particles[tid];
    uint32_t key = morton_encode_2d_dev(p.x, p.y, depth);

    bool inserted = false;
    for (int lvl = 0; lvl < MAX_LEVELS && !inserted; ++lvl) {
        uint32_t offset = level_offsets[lvl];
        uint32_t size = level_sizes[lvl];
        int max_attempts = 4 + lvl * 2;

        for (int att = 0; att < max_attempts && !inserted; ++att) {
            uint32_t slot = offset + hash_probe_dev(key, seeds_a[lvl * 4 + (att % 4)], seeds_b[lvl * 4 + (att % 4)], size, att);
            uint32_t old_key = atomicCAS(&table_keys[slot], 0xFFFFFFFFU, key);
            if (old_key == 0xFFFFFFFFU || old_key == key) {
                int old_head = atomicExch(&table_head_indices[slot], tid);
                particle_next_ptrs[tid] = old_head;
                inserted = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Fused CGR88 Local to Particle (L2P) + Near-Field P2P Interaction Kernel
// ---------------------------------------------------------------------------
__global__ void evaluate_cgr88_fmm_2d_kernel(
    const Particle2D* __restrict__ particles,
    int num_particles,
    const ComplexFloat* __restrict__ cluster_local_coeffs, // [num_clusters * (order + 1)]
    const float2* __restrict__ cluster_centers,
    const int* __restrict__ particle_cluster_ids,
    float* __restrict__ out_potentials,
    float2* __restrict__ out_forces,
    int order,
    float softening,
    float cell_size // width of one uniform-grid leaf cell (world units)
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles) return;

    Particle2D p_i = particles[tid];
    int c_id = particle_cluster_ids[tid];
    float2 c_center = cluster_centers[c_id];

    // 1. Far-field L2P evaluation
    ComplexFloat dz = c_make(p_i.x - c_center.x, p_i.y - c_center.y);
    ComplexFloat dz_k = c_make(1.0f, 0.0f);

    ComplexFloat l0 = cluster_local_coeffs[c_id * (order + 1) + 0];
    ComplexFloat pot_complex = l0;
    ComplexFloat deriv_complex = c_make(0.0f, 0.0f);

    for (int l = 1; l <= order; ++l) {
        ComplexFloat coeff = cluster_local_coeffs[c_id * (order + 1) + l];
        ComplexFloat term_deriv = c_mul(c_make((float)l, 0.0f), c_mul(coeff, dz_k));
        deriv_complex = c_add(deriv_complex, term_deriv);
        dz_k = c_mul(dz_k, dz);
        pot_complex = c_add(pot_complex, c_mul(coeff, dz_k));
    }

    float phi = pot_complex.r;
    float2 force = make_float2(-deriv_complex.r, deriv_complex.i);
    float eps_sq = softening * softening;

    // 2. Near-field P2P: direct summation restricted to particles whose
    // cluster is the SAME as or GEOMETRICALLY ADJACENT to ours (Chebyshev
    // center distance <= one cell). This is exactly the complement of the
    // well-separated pairs handled by the far-field local expansions, so
    // every pair is counted once. NOTE: still an O(N * particles_per_cell-
    // neighborhood) scan over all N; a per-cluster particle-range CSR would
    // remove the full scan but requires additional device buffers.
    for (int j = 0; j < num_particles; ++j) {
        if (j == tid) continue;
        int c_j = particle_cluster_ids[j];
        bool near_cell =
            c_j == c_id ||
            (fabsf(cluster_centers[c_j].x - c_center.x) <= 1.5f * cell_size &&
             fabsf(cluster_centers[c_j].y - c_center.y) <= 1.5f * cell_size);
        if (!near_cell) continue;
        Particle2D p_j = particles[j];
        float dx = p_i.x - p_j.x;
        float dy = p_i.y - p_j.y;
        float r_sq = dx * dx + dy * dy + eps_sq;
        float inv_r2 = 1.0f / r_sq;
        phi += p_j.q * 0.5f * logf(r_sq);
        force.x -= p_j.q * dx * inv_r2;
        force.y -= p_j.q * dy * inv_r2;
    }

    out_potentials[tid] = phi;
    out_forces[tid] = force;
}
