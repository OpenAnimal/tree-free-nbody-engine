/**
 * Ultra-Fast Tree-Free Fast Multipole Method (FMM) AMD ROCm / HIP C++ Kernel
 * =========================================================================
 * Combines Lock-Free Non-Reordering Open Addressing Hash Table (Farach-Colton et al. 2025)
 * with Carrier, Greengard, & Rokhlin (1988) Multipole Expansions (P2M, M2L, L2P)
 * and Fused Shared-Memory / Local Data Share (LDS) P2P Interaction Tiles.
 *
 * Target Architectures:
 *  - AMD RDNA 1/2/3/4 (gfx1010/gfx1030/gfx1100/gfx1200)
 *  - AMD CDNA 1/2/3 (gfx906/gfx908/gfx90a/gfx940/gfx942)
 */

#include <hip/hip_runtime.h>
#define WARP_SIZE 64
#define AMD_WAVE_SIZE 64

#include <stdint.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define MAX_LEVELS 4
#define MAX_ORDER 8

struct __align__(16) Particle3D {
    float x, y, z, q;
};

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

__device__ __forceinline__ uint32_t expand_bits_2d(uint32_t v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__device__ __forceinline__ uint32_t expand_bits_3d(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}

__device__ __forceinline__ uint32_t morton_encode_2d_hip(float x, float y, int depth) {
    uint32_t grid_res = 1U << depth;
    uint32_t ix = fminf(fmaxf(x * grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iy = fminf(fmaxf(y * grid_res, 0.0f), (float)(grid_res - 1));
    return expand_bits_2d(ix) | (expand_bits_2d(iy) << 1);
}

__device__ __forceinline__ uint32_t hip_morton_encode_3d(float x, float y, float z, int depth) {
    uint32_t grid_res = 1U << depth;
    uint32_t ix = fminf(fmaxf(x * grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iy = fminf(fmaxf(y * grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iz = fminf(fmaxf(z * grid_res, 0.0f), (float)(grid_res - 1));
    return expand_bits_3d(ix) | (expand_bits_3d(iy) << 1) | (expand_bits_3d(iz) << 2);
}

__device__ __forceinline__ uint32_t hash_probe_hip(uint32_t key, uint32_t seed_a, uint32_t seed_b, uint32_t size, int attempt) {
    return ((key * seed_a + seed_b + attempt * 2654435761U) & 0x7FFFFFFFU) % size;
}

__global__ void hip_insert_particles_elastic_hash_kernel(
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
    int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= num_particles) return;

    Particle2D p = particles[tid];
    uint32_t key = morton_encode_2d_hip(p.x, p.y, depth);

    bool inserted = false;
    for (int lvl = 0; lvl < MAX_LEVELS && !inserted; ++lvl) {
        uint32_t offset = level_offsets[lvl];
        uint32_t size = level_sizes[lvl];
        int max_attempts = 4 + lvl * 2;

        for (int att = 0; att < max_attempts && !inserted; ++att) {
            uint32_t slot = offset + hash_probe_hip(key, seeds_a[lvl * 4 + (att % 4)], seeds_b[lvl * 4 + (att % 4)], size, att);
            uint32_t old_key = atomicCAS(&table_keys[slot], 0xFFFFFFFFU, key);
            if (old_key == 0xFFFFFFFFU || old_key == key) {
                int old_head = atomicExch(&table_head_indices[slot], tid);
                particle_next_ptrs[tid] = old_head;
                inserted = true;
            }
        }
    }
}

__global__ void hip_evaluate_near_field_p2p_kernel(
    const Particle3D* __restrict__ particles,
    int num_particles,
    float* __restrict__ out_potentials,
    float3* __restrict__ out_forces,
    float softening
) {
    __shared__ Particle3D lds_particles[BLOCK_SIZE];
    int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int lid = hipThreadIdx_x;

    Particle3D my_p = (tid < num_particles) ? particles[tid] : Particle3D{0,0,0,0};
    float phi = 0.0f;
    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    float eps_sq = softening * softening;

    int num_tiles = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int load_idx = t * BLOCK_SIZE + lid;
        lds_particles[lid] = (load_idx < num_particles) ? particles[load_idx] : Particle3D{0,0,0,0};
        __syncthreads();

        int tile_limit = min(BLOCK_SIZE, num_particles - t * BLOCK_SIZE);
        if (tid < num_particles) {
            #pragma unroll 4
            for (int j = 0; j < tile_limit; ++j) {
                int global_j = t * BLOCK_SIZE + j;
                if (global_j != tid) {
                    Particle3D pj = lds_particles[j];
                    float dx = my_p.x - pj.x;
                    float dy = my_p.y - pj.y;
                    float dz = my_p.z - pj.z;
                    float r_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                    float inv_r = rsqrtf(r_sq);
                    float inv_r3 = inv_r * inv_r * inv_r;
                    phi += pj.q * inv_r;
                    float f_scalar = my_p.q * pj.q * inv_r3;
                    f.x += f_scalar * dx;
                    f.y += f_scalar * dy;
                    f.z += f_scalar * dz;
                }
            }
        }
        __syncthreads();
    }

    if (tid < num_particles) {
        out_potentials[tid] = phi;
        out_forces[tid] = f;
    }
}

__device__ __forceinline__ float hip_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = AMD_WAVE_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, AMD_WAVE_SIZE);
    }
    return val;
}

__global__ void hip_compute_cluster_multipoles_kernel(
    const Particle3D* __restrict__ particles,
    const int* __restrict__ cluster_particle_indices,
    int num_particles_in_cluster,
    float3 cluster_center,
    float* __restrict__ out_monopole,
    float3* __restrict__ out_dipole
) {
    int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    float q_sum = 0.0f;
    float3 dipole = make_float3(0.0f, 0.0f, 0.0f);

    if (tid < num_particles_in_cluster) {
        int p_idx = cluster_particle_indices[tid];
        Particle3D p = particles[p_idx];
        q_sum = p.q;
        dipole.x = p.q * (p.x - cluster_center.x);
        dipole.y = p.q * (p.y - cluster_center.y);
        dipole.z = p.q * (p.z - cluster_center.z);
    }

    q_sum = hip_warp_reduce_sum(q_sum);
    dipole.x = hip_warp_reduce_sum(dipole.x);
    dipole.y = hip_warp_reduce_sum(dipole.y);
    dipole.z = hip_warp_reduce_sum(dipole.z);

    if ((hipThreadIdx_x & (AMD_WAVE_SIZE - 1)) == 0) {
        atomicAdd(out_monopole, q_sum);
        atomicAdd(&out_dipole->x, dipole.x);
        atomicAdd(&out_dipole->y, dipole.y);
        atomicAdd(&out_dipole->z, dipole.z);
    }
}

__global__ void hip_evaluate_cgr88_fmm_2d_kernel(
    const Particle2D* __restrict__ particles,
    int num_particles,
    const ComplexFloat* __restrict__ cluster_local_coeffs,
    const float2* __restrict__ cluster_centers,
    const int* __restrict__ particle_cluster_ids,
    float* __restrict__ out_potentials,
    float2* __restrict__ out_forces,
    int order,
    float softening
) {
    int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= num_particles) return;

    Particle2D p_i = particles[tid];
    int c_id = particle_cluster_ids[tid];
    float2 c_center = cluster_centers[c_id];

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

    for (int j = 0; j < num_particles; ++j) {
        if (j != tid) {
            Particle2D p_j = particles[j];
            float dx = p_i.x - p_j.x;
            float dy = p_i.y - p_j.y;
            float r_sq = dx * dx + dy * dy + eps_sq;
            if (r_sq < 0.0016f) {
                float inv_r2 = 1.0f / (r_sq + 1e-12f);
                phi += p_j.q * 0.5f * logf(r_sq + 1e-12f);
                force.x -= p_j.q * dx * inv_r2;
                force.y -= p_j.q * dy * inv_r2;
            }
        }
    }

    out_potentials[tid] = phi;
    out_forces[tid] = force;
}
