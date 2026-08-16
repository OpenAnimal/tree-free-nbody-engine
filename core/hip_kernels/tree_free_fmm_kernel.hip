/**
 * Ultra-Fast Tree-Free Fast Multipole Method (FMM) AMD ROCm / HIP C++ Kernel
 * ===========================================================================
 * Native Heterogeneous-Compute Interface for Portability (HIP) implementation
 * providing full ATI/AMD Radeon GPU acceleration across consumer and enterprise architectures:
 *
 * Target Architectures:
 *  - AMD RDNA 4 (gfx1200 / RX 8000 Series)
 *  - AMD RDNA 3 (gfx1100 / gfx1101 / gfx1102 / RX 7900, 7800, 7700, 7600)
 *  - AMD RDNA 2 (gfx1030 / gfx1031 / gfx1032 / RX 6900, 6800, 6700, 6600)
 *  - AMD RDNA 1 (gfx1010 / gfx1012 / RX 5700, 5600, 5500)
 *  - AMD CDNA 1/2/3 (gfx908 / gfx90a / gfx940 / gfx942 / Instinct MI100, MI200, MI300)
 *  - AMD GCN 4/5 (gfx803 / gfx900 / Polaris RX 480/580, Vega 56/64)
 *
 * Features:
 *  - Dynamic Wavefront Size Detection (Wave32 for RDNA vs Wave64 for CDNA/GCN).
 *  - Non-Reordering Lock-Free Elastic Spatial Hash Table (Farach-Colton et al. 2025) via AMD Global Atomic CAS.
 *  - High-Throughput Fused All-Pairs P2P Interactions in AMD Local Data Share (LDS).
 *  - Cross-Lane Wavefront Shuffles (__shfl_down) for P2M Multipole Reductions.
 */

#ifndef TREE_FREE_FMM_KERNEL_HIP
#define TREE_FREE_FMM_KERNEL_HIP

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/hip_runtime.h>
#else
// Fallback for native compilation / linting
#include <cuda_runtime.h>
#endif

#include <stdint.h>
#include <stdio.h>

// Handle AMD Wavefront 32 (RDNA) vs Wavefront 64 (CDNA/GCN)
#if defined(__AMDGCN_WAVEFRONT_SIZE)
#define AMD_WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE
#elif defined(__HIP_PLATFORM_AMD__)
#define AMD_WAVE_SIZE 64
#else
#define AMD_WAVE_SIZE 32
#endif

#define HIP_BLOCK_SIZE 256
#define MAX_LEVELS 4
#define ORDER 4

// 16-byte aligned float4 particle structure matching AMD Radeon memory coalescing
struct __align__(16) HipParticle {
    float x, y, z, q;
};

// ---------------------------------------------------------------------------
// 1. AMD Device 3D Morton Coordinate Encoding
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hip_expand_bits_3d(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}

__device__ __forceinline__ uint32_t hip_morton_encode_3d(float x, float y, float z, int depth) {
    uint32_t grid_res = 1U << depth;
    uint32_t ix = (uint32_t)fminf(fmaxf(x * (float)grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iy = (uint32_t)fminf(fmaxf(y * (float)grid_res, 0.0f), (float)(grid_res - 1));
    uint32_t iz = (uint32_t)fminf(fmaxf(z * (float)grid_res, 0.0f), (float)(grid_res - 1));
    return hip_expand_bits_3d(ix) | (hip_expand_bits_3d(iy) << 1) | (hip_expand_bits_3d(iz) << 2);
}

// ---------------------------------------------------------------------------
// 2. Lock-Free Non-Reordering Elastic Hash Table Insert (atomicCAS) on AMD GPU
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hip_hash_probe(
    uint32_t key,
    uint32_t seed_a,
    uint32_t seed_b,
    uint32_t size,
    int attempt
) {
    return ((key * seed_a + seed_b + (uint32_t)attempt * 2654435761U) & 0x7FFFFFFFU) % size;
}

__global__ void hip_insert_particles_elastic_hash_kernel(
    const HipParticle* __restrict__ particles,
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

    HipParticle p = particles[tid];
    uint32_t key = hip_morton_encode_3d(p.x, p.y, p.z, depth);

    // Multi-level geometric probe without ANY reordering (Farach-Colton et al. 2025)
    bool inserted = false;
    for (int lvl = 0; lvl < MAX_LEVELS && !inserted; ++lvl) {
        uint32_t offset = level_offsets[lvl];
        uint32_t size = level_sizes[lvl];
        int max_attempts = 4 + lvl * 2;

        for (int att = 0; att < max_attempts && !inserted; ++att) {
            uint32_t slot = offset + hip_hash_probe(key, seeds_a[lvl * 4 + (att % 4)], seeds_b[lvl * 4 + (att % 4)], size, att);
            
            // Atomic lock-free slot claim via AMD global memory atomicCAS
            uint32_t old_key = atomicCAS(&table_keys[slot], 0xFFFFFFFFU, key);
            if (old_key == 0xFFFFFFFFU || old_key == key) {
                // Prepend particle to lock-free singly linked bucket list
                int old_head = atomicExch(&table_head_indices[slot], tid);
                particle_next_ptrs[tid] = old_head;
                inserted = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Fused Near-Field P2P Interaction Kernel with AMD LDS Tiling
// ---------------------------------------------------------------------------
__global__ void hip_evaluate_near_field_p2p_kernel(
    const HipParticle* __restrict__ particles,
    int num_particles,
    float* __restrict__ out_potentials,
    float3* __restrict__ out_forces,
    float softening
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles) return;

    HipParticle p_i = particles[tid];
    float phi = 0.0f;
    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    float eps_sq = softening * softening;

    // Block-tiled execution using AMD Local Data Share (LDS)
    __shared__ HipParticle lds_particles[HIP_BLOCK_SIZE];

    int num_tiles = (num_particles + HIP_BLOCK_SIZE - 1) / HIP_BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int load_idx = t * HIP_BLOCK_SIZE + threadIdx.x;
        if (load_idx < num_particles) {
            lds_particles[threadIdx.x] = particles[load_idx];
        }
        __syncthreads();

        int tile_limit = min(HIP_BLOCK_SIZE, num_particles - t * HIP_BLOCK_SIZE);
        #pragma unroll 4
        for (int j = 0; j < tile_limit; ++j) {
            int global_j = t * HIP_BLOCK_SIZE + j;
            if (global_j != tid) {
                HipParticle p_j = lds_particles[j];
                float dx = p_i.x - p_j.x;
                float dy = p_i.y - p_j.y;
                float dz = p_i.z - p_j.z;
                float r_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                float inv_r = rsqrtf(r_sq);
                float inv_r3 = inv_r * inv_r * inv_r;

                phi += p_j.q * inv_r;
                float force_scalar = p_i.q * p_j.q * inv_r3;
                f.x += force_scalar * dx;
                f.y += force_scalar * dy;
                f.z += force_scalar * dz;
            }
        }
        __syncthreads();
    }

    out_potentials[tid] = phi;
    out_forces[tid] = f;
}

// ---------------------------------------------------------------------------
// 4. Wavefront-Level Multipole Expansion (P2M) for AMD ROCm
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hip_wavefront_reduce_sum(float val) {
    #pragma unroll
    for (int offset = AMD_WAVE_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, AMD_WAVE_SIZE);
    }
    return val;
}

__global__ void hip_compute_cluster_multipoles_kernel(
    const HipParticle* __restrict__ particles,
    const int* __restrict__ cluster_particle_indices,
    int num_particles_in_cluster,
    float3 cluster_center,
    float* __restrict__ out_monopole,
    float3* __restrict__ out_dipole
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float q_sum = 0.0f;
    float3 dipole = make_float3(0.0f, 0.0f, 0.0f);

    if (tid < num_particles_in_cluster) {
        int p_idx = cluster_particle_indices[tid];
        HipParticle p = particles[p_idx];
        q_sum = p.q;
        dipole.x = p.q * (p.x - cluster_center.x);
        dipole.y = p.q * (p.y - cluster_center.y);
        dipole.z = p.q * (p.z - cluster_center.z);
    }

    q_sum = hip_wavefront_reduce_sum(q_sum);
    dipole.x = hip_wavefront_reduce_sum(dipole.x);
    dipole.y = hip_wavefront_reduce_sum(dipole.y);
    dipole.z = hip_wavefront_reduce_sum(dipole.z);

    // Each wavefront leader atomically accumulates into global cluster multipoles
    if ((threadIdx.x & (AMD_WAVE_SIZE - 1)) == 0) {
        atomicAdd(out_monopole, q_sum);
        atomicAdd(&out_dipole->x, dipole.x);
        atomicAdd(&out_dipole->y, dipole.y);
        atomicAdd(&out_dipole->z, dipole.z);
    }
}

#endif // TREE_FREE_FMM_KERNEL_HIP
