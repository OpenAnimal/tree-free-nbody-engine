/**
 * Ultra-Fast Tree-Free Fast Multipole Method (FMM) CUDA/HIP C++ Kernel
 * ====================================================================
 * Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole Expansions
 * (P2M, M2L, L2P) with a Fused Shared-Memory P2P Interaction Tile.
 *
 * Neighbor / index construction (task T-E1 CUDA side): the per-cell
 * particle ranges are built with a counting-sort CSR pipeline that mirrors
 * the WGSL reference kernels clear_cells / count_cells / scan_cells /
 * scatter_cells in index.html:
 *   A) clear_cell_counts: cellCount[c] = 0 for every leaf cell c
 *   B) count_cells:       atomicAdd(cellCount[leafIndex(pos)], 1) per particle
 *   C) scan_cells:        single-block (256) exclusive prefix sum over
 *                         cellCount -> cellStart[c]; cellCursor[c] = cellStart[c]
 *   D) scatter_cells:     slot = atomicAdd(cellCursor[cell], 1);
 *                         sortedIndex[slot] = particleId
 * The P2P pass then iterates CONTIGUOUS ranges
 * sortedIndex[cellStart[c] .. +cellCount[c]] over the 3x3 leaf neighborhood
 * — no linked lists, no hash table, no next pointers, no O(N) masked scan.
 * The previous generic open-addressing hash insert (atomicCAS + linear
 * probing with particle_next_ptrs) has been removed accordingly; the FKK
 * funnel hash schedule lives only in core/elastic_hash.py and the WGSL demo.
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
// 1. Leaf cell index (uniform 2D grid, row-major).
//    Mirrors the WGSL `leafIndex(p)` in index.html: side = 1 << leaf_bits,
//    cell = cy * side + cx, with positions clamped to [0, 1).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t leaf_index_2d(float x, float y, uint32_t side) {
    uint32_t cx = min((uint32_t)(fminf(fmaxf(x, 0.0f), 0.999999f) * (float)side), side - 1u);
    uint32_t cy = min((uint32_t)(fminf(fmaxf(y, 0.0f), 0.999999f) * (float)side), side - 1u);
    return cy * side + cx;
}

// ---------------------------------------------------------------------------
// 2. Counting-sort CSR cell-list construction (T-E1 CUDA side).
//    Four launch_bounds-decorated passes mirroring the WGSL reference
//    clear_cells / count_cells / scan_cells / scatter_cells. Produces
//    cellStart / cellCount / sortedIndex so that particles of cell c occupy
//    sortedIndex[cellStart[c] .. cellStart[c] + cellCount[c] - 1].
//    No thrust/CUB; only CUDA runtime atomics + a single-block scan.
// ---------------------------------------------------------------------------

// Pass A: zero per-cell counts. Dispatch over num_cells.
__global__ __launch_bounds__(BLOCK_SIZE) void clear_cell_counts(
    uint32_t* __restrict__ cellCount,
    uint32_t num_cells)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_cells) cellCount[tid] = 0u;
}

// Pass B: per-particle histogram into cell counts. Dispatch over num_particles.
__global__ __launch_bounds__(BLOCK_SIZE) void count_cells(
    const Particle2D* __restrict__ particles,
    int num_particles,
    uint32_t* __restrict__ cellCount,
    uint32_t side)
{
    int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= num_particles) return;
    Particle2D p = particles[tid];
    uint32_t cell = leaf_index_2d(p.x, p.y, side);
    atomicAdd(&cellCount[cell], 1u);
}

// Pass C: single-block (256) exclusive prefix sum over cellCount.
// Each thread sequentially scans one contiguous chunk of cells (chunk-local
// exclusive prefix + chunk sum), thread 0 scans the 256 chunk sums, then
// every thread offsets its chunk by the totals of all earlier chunks.
// Writes cellStart[c] (exclusive prefix) and initializes cellCursor[c] =
// cellStart[c] for the scatter pass. Mirrors the WGSL scan_cells exactly.
// Launch with <<<1, 256>>>; num_cells must fit a single-block scan
// (chunk = ceil(num_cells/256) cells per thread).
__global__ __launch_bounds__(256) void scan_cells(
    const uint32_t* __restrict__ cellCount,
    uint32_t* __restrict__ cellStart,
    uint32_t* __restrict__ cellCursor,
    uint32_t num_cells)
{
    __shared__ uint32_t scanPartial[256];
    __shared__ uint32_t scanBase[256];
    uint32_t tid = threadIdx.x;
    uint32_t chunk = (num_cells + 255u) / 256u;
    uint32_t lo = tid * chunk;
    uint32_t hi = min(lo + chunk, num_cells);

    // Chunk-local exclusive prefix sum.
    uint32_t run = 0u;
    for (uint32_t i = lo; i < hi; ++i) {
        cellStart[i] = run;
        run += cellCount[i];
    }
    scanPartial[tid] = run;
    __syncthreads();

    // Thread 0 scans the 256 chunk sums into per-chunk bases.
    if (tid == 0u) {
        uint32_t acc = 0u;
        for (uint32_t t = 0u; t < 256u; ++t) {
            scanBase[t] = acc;
            acc += scanPartial[t];
        }
    }
    __syncthreads();

    // Offset each chunk by the totals of all earlier chunks; seed cellCursor.
    uint32_t base = scanBase[tid];
    for (uint32_t j = lo; j < hi; ++j) {
        uint32_t s = cellStart[j] + base;
        cellStart[j] = s;
        cellCursor[j] = s; // no atomic: only this thread writes cell j
    }
}

// Pass D: scatter particle ids into sortedIndex via per-cell atomicAdd on
// cellCursor. Dispatch over num_particles.
__global__ __launch_bounds__(BLOCK_SIZE) void scatter_cells(
    const Particle2D* __restrict__ particles,
    int num_particles,
    uint32_t* __restrict__ cellCursor,
    int* __restrict__ sortedIndex,
    uint32_t side)
{
    int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= num_particles) return;
    Particle2D p = particles[tid];
    uint32_t cell = leaf_index_2d(p.x, p.y, side);
    uint32_t slot = atomicAdd(&cellCursor[cell], 1u);
    sortedIndex[slot] = tid;
}

// ---------------------------------------------------------------------------
// 3. Fused adaptive FMM Local to Particle (L2P) + Near-Field P2P Interaction Kernel
//    The far-field L2P math is unchanged; only the near-field neighbor
//    traversal has been switched from an O(N) masked scan to CSR cell-list
//    ranges (clear/count/scan/scatter output). The 3x3 leaf neighborhood of
//    particle i's cell is iterated via cellStart/cellCount/sortedIndex,
//    matching the WGSL reference P2P neighbor traversal.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_SIZE) void evaluate_adaptive_fmm_2d_kernel(
    const Particle2D* __restrict__ particles,
    int num_particles,
    const ComplexFloat* __restrict__ cluster_local_coeffs, // [num_clusters * (order + 1)]
    const float2* __restrict__ cluster_centers,
    const int* __restrict__ particle_cluster_ids,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellCount,
    const int* __restrict__ sortedIndex,
    uint32_t leaf_bits,
    int order,
    float softening,
    float* __restrict__ out_potentials,
    float2* __restrict__ out_forces
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

    // 2. Near-field P2P via CSR cell lists (counting-sort ranges from
    //    clear_cell_counts / count_cells / scan_cells / scatter_cells).
    //    Iterate the 3x3 leaf neighborhood of particle i's cell; each cell c
    //    contributes sortedIndex[cellStart[c] .. +cellCount[c]]. This is the
    //    complement of the well-separated pairs handled by the far-field local
    //    expansions, so every near pair is counted once. Unsigned wraparound
    //    on ny/nx handles border cells (out-of-range >= side is skipped).
    uint32_t side = 1u << leaf_bits;
    uint32_t cx = min((uint32_t)(fminf(fmaxf(p_i.x, 0.0f), 0.999999f) * (float)side), side - 1u);
    uint32_t cy = min((uint32_t)(fminf(fmaxf(p_i.y, 0.0f), 0.999999f) * (float)side), side - 1u);
    for (int oy = -1; oy <= 1; ++oy) {
        uint32_t ny = cy + (uint32_t)oy; // underflow -> >= side, skipped below
        if (ny >= side) continue;
        for (int ox = -1; ox <= 1; ++ox) {
            uint32_t nx = cx + (uint32_t)ox;
            if (nx >= side) continue;
            uint32_t c = ny * side + nx;
            int start = (int)cellStart[c];
            int cnt = (int)cellCount[c];
            for (int k = 0; k < cnt; ++k) {
                int j = sortedIndex[start + k];
                if (j == tid) continue;
                Particle2D p_j = particles[j];
                float dx = p_i.x - p_j.x;
                float dy = p_i.y - p_j.y;
                float r_sq = dx * dx + dy * dy + eps_sq;
                float inv_r2 = 1.0f / r_sq;
                phi += p_j.q * 0.5f * logf(r_sq);
                force.x -= p_j.q * dx * inv_r2;
                force.y -= p_j.q * dy * inv_r2;
            }
        }
    }

    out_potentials[tid] = phi;
    out_forces[tid] = force;
}

// ---------------------------------------------------------------------------
// 4. Host launcher (CUDA runtime API only; HIP builds compile this file
//    with hipcc and should provide an equivalent launcher with the hip*
//    memory APIs).
//    Chains the four CSR build passes (clear -> count -> scan -> scatter)
//    and the fused L2P + P2P evaluate kernel. Transient CSR buffers
//    (cellCount / cellStart / cellCursor / sortedIndex) are allocated and
//    freed per call for simplicity; production pipelines should reuse them
//    across frames (ping-pong, see docs/GPU_NOTES.md). No thrust/CUB.
// ---------------------------------------------------------------------------
#ifndef __HIPCC__
extern "C" void launch_tree_free_fmm_2d(
    const Particle2D* d_particles,
    int num_particles,
    const ComplexFloat* d_local_coeffs,
    const float2* d_cluster_centers,
    const int* d_particle_cluster_ids,
    int order,
    float softening,
    uint32_t leaf_bits,
    float* d_out_potentials,
    float2* d_out_forces)
{
    uint32_t side = 1u << leaf_bits;
    uint32_t num_cells = side * side;
    int nblocks_p = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int nblocks_c = (int)((num_cells + (uint32_t)BLOCK_SIZE - 1u) / (uint32_t)BLOCK_SIZE);

    uint32_t* d_cellCount  = nullptr;
    uint32_t* d_cellStart  = nullptr;
    uint32_t* d_cellCursor = nullptr;
    int*      d_sortedIdx  = nullptr;
    cudaMalloc(&d_cellCount,  sizeof(uint32_t) * num_cells);
    cudaMalloc(&d_cellStart,  sizeof(uint32_t) * num_cells);
    cudaMalloc(&d_cellCursor, sizeof(uint32_t) * num_cells);
    cudaMalloc(&d_sortedIdx,  sizeof(int) * (size_t)num_particles);

    clear_cell_counts<<<nblocks_c, BLOCK_SIZE>>>(d_cellCount, num_cells);
    count_cells<<<nblocks_p, BLOCK_SIZE>>>(d_particles, num_particles, d_cellCount, side);
    scan_cells<<<1, 256>>>(d_cellCount, d_cellStart, d_cellCursor, num_cells);
    scatter_cells<<<nblocks_p, BLOCK_SIZE>>>(d_particles, num_particles, d_cellCursor, d_sortedIdx, side);

    evaluate_adaptive_fmm_2d_kernel<<<nblocks_p, BLOCK_SIZE>>>(
        d_particles, num_particles, d_local_coeffs, d_cluster_centers,
        d_particle_cluster_ids, d_cellStart, d_cellCount, d_sortedIdx,
        leaf_bits, order, softening, d_out_potentials, d_out_forces);

    cudaFree(d_cellCount);
    cudaFree(d_cellStart);
    cudaFree(d_cellCursor);
    cudaFree(d_sortedIdx);
}
#endif // __HIPCC__
