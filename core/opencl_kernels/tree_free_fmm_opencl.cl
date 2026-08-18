/**
 * Ultra-Fast Tree-Free Fast Multipole Method (FMM) OpenCL C Compute Kernels
 * =========================================================================
 * Cross-platform hardware acceleration compliant with ATI/AMD Radeon (RDNA, CDNA, GCN, APU),
 * Intel Arc/Iris, and NVIDIA GPU architectures via OpenCL 1.2 / 2.0 / 3.0.
 *
 * Kernels:
 *  1. `opencl_morton_encode_3d`: High-throughput bit-interleaving coordinate hash.
 *  2. `opencl_p2p_coulomb_nbody`: Block-tiled Local Memory (LDS) all-pairs potential and vector forces.
 *  3. `opencl_cgr88_fmm_2d`: Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole L2P & P2P evaluator.
 *  4. `opencl_volumetric_ao_sample`: High-speed ambient occlusion / density field evaluation.
 */

#ifndef TREE_FREE_FMM_OPENCL_CL
#define TREE_FREE_FMM_OPENCL_CL

typedef struct {
    float x;
    float y;
    float z;
    float q;
} OpenCLParticle;

typedef struct {
    float r;
    float i;
} OpenCLComplex;

inline OpenCLComplex ocl_c_make(float r, float i) {
    OpenCLComplex z; z.r = r; z.i = i; return z;
}

inline OpenCLComplex ocl_c_add(OpenCLComplex a, OpenCLComplex b) {
    return ocl_c_make(a.r + b.r, a.i + b.i);
}

inline OpenCLComplex ocl_c_mul(OpenCLComplex a, OpenCLComplex b) {
    return ocl_c_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

// ---------------------------------------------------------------------------
// 1. Bit-Interleaved 3D Morton Coordinate Encoding
// ---------------------------------------------------------------------------
inline uint opencl_expand_bits_3d(uint v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}

__kernel void opencl_morton_encode_3d(
    __global const float* coords, // (N, 3)
    __global uint* out_keys,      // (N,)
    const int num_particles,
    const int depth
) {
    int gid = get_global_id(0);
    if (gid >= num_particles) return;

    float x = coords[gid * 3 + 0];
    float y = coords[gid * 3 + 1];
    float z = coords[gid * 3 + 2];

    uint grid_res = 1U << depth;
    uint ix = (uint)clamp(x * (float)grid_res, 0.0f, (float)(grid_res - 1));
    uint iy = (uint)clamp(y * (float)grid_res, 0.0f, (float)(grid_res - 1));
    uint iz = (uint)clamp(z * (float)grid_res, 0.0f, (float)(grid_res - 1));

    uint key = opencl_expand_bits_3d(ix) | 
               (opencl_expand_bits_3d(iy) << 1) | 
               (opencl_expand_bits_3d(iz) << 2);

    out_keys[gid] = key;
}

// ---------------------------------------------------------------------------
// 2. Block-Tiled All-Pairs N-Body / P2P Kernel in AMD Local Data Share (__local)
// ---------------------------------------------------------------------------
#define WORKGROUP_SIZE 256

__kernel void opencl_p2p_coulomb_nbody(
    __global const float* coords,       // (N, 3)
    __global const float* charges,      // (N,)
    __global float* out_potentials,     // (N,)
    __global float* out_forces,         // (N, 3)
    const int num_particles,
    const float softening_sq,
    __local float4* local_particles     // __local scratchpad in AMD LDS
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    float4 my_p = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    if (gid < num_particles) {
        my_p.x = coords[gid * 3 + 0];
        my_p.y = coords[gid * 3 + 1];
        my_p.z = coords[gid * 3 + 2];
        my_p.w = charges[gid];
    }

    float acc_phi = 0.0f;
    float3 acc_f = (float3)(0.0f, 0.0f, 0.0f);

    int num_tiles = (num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int load_idx = t * WORKGROUP_SIZE + lid;
        if (load_idx < num_particles) {
            local_particles[lid] = (float4)(
                coords[load_idx * 3 + 0],
                coords[load_idx * 3 + 1],
                coords[load_idx * 3 + 2],
                charges[load_idx]
            );
        } else {
            local_particles[lid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int tile_count = min(WORKGROUP_SIZE, num_particles - t * WORKGROUP_SIZE);
        if (gid < num_particles) {
            #pragma unroll 4
            for (int j = 0; j < tile_count; ++j) {
                int global_j = t * WORKGROUP_SIZE + j;
                if (global_j != gid) {
                    float4 p_j = local_particles[j];
                    float dx = my_p.x - p_j.x;
                    float dy = my_p.y - p_j.y;
                    float dz = my_p.z - p_j.z;
                    float r_sq = dx * dx + dy * dy + dz * dz + softening_sq;
                    float inv_r = rsqrt(r_sq);
                    float inv_r3 = inv_r * inv_r * inv_r;

                    acc_phi += p_j.w * inv_r;
                    float f_scalar = my_p.w * p_j.w * inv_r3;
                    acc_f.x += f_scalar * dx;
                    acc_f.y += f_scalar * dy;
                    acc_f.z += f_scalar * dz;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < num_particles) {
        out_potentials[gid] = acc_phi;
        out_forces[gid * 3 + 0] = acc_f.x;
        out_forces[gid * 3 + 1] = acc_f.y;
        out_forces[gid * 3 + 2] = acc_f.z;
    }
}

// ---------------------------------------------------------------------------
// 3. Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole L2P & P2P Kernel
// ---------------------------------------------------------------------------
__kernel void opencl_cgr88_fmm_2d(
    __global const float2* positions,           // (N,)
    __global const float* charges,              // (N,)
    __global const int* particle_cluster_ids,   // (N,)
    __global const float2* cluster_centers,     // (num_clusters,)
    __global const float2* cluster_local_coeffs,// (num_clusters * (order + 1),) stored as complex float2
    __global float* out_potentials,             // (N,)
    __global float2* out_forces,                // (N,)
    const int num_particles,
    const int order,
    const float softening_sq
) {
    int gid = get_global_id(0);
    if (gid >= num_particles) return;

    float2 pos_i = positions[gid];
    float q_i = charges[gid];
    int c_id = particle_cluster_ids[gid];
    float2 c_center = cluster_centers[c_id];

    // L2P evaluation
    OpenCLComplex dz = ocl_c_make(pos_i.x - c_center.x, pos_i.y - c_center.y);
    OpenCLComplex dz_k = ocl_c_make(1.0f, 0.0f);

    float2 l0_raw = cluster_local_coeffs[c_id * (order + 1) + 0];
    OpenCLComplex pot_comp = ocl_c_make(l0_raw.x, l0_raw.y);
    OpenCLComplex deriv_comp = ocl_c_make(0.0f, 0.0f);

    for (int l = 1; l <= order; ++l) {
        float2 cl_raw = cluster_local_coeffs[c_id * (order + 1) + l];
        OpenCLComplex coeff = ocl_c_make(cl_raw.x, cl_raw.y);
        OpenCLComplex term_deriv = ocl_c_mul(ocl_c_make((float)l, 0.0f), ocl_c_mul(coeff, dz_k));
        deriv_comp = ocl_c_add(deriv_comp, term_deriv);
        dz_k = ocl_c_mul(dz_k, dz);
        pot_comp = ocl_c_add(pot_comp, ocl_c_mul(coeff, dz_k));
    }

    float phi = pot_comp.r;
    float2 force = (float2)(-deriv_comp.r, deriv_comp.i);

    // Direct near-field summation
    for (int j = 0; j < num_particles; ++j) {
        if (j != gid) {
            float2 pos_j = positions[j];
            float q_j = charges[j];
            float2 diff = pos_i - pos_j;
            float r_sq = dot(diff, diff) + softening_sq;
            if (r_sq < 0.0025f) {
                float r_sq_safe = max(r_sq, 1e-12f);
                float inv_r2 = 1.0f / r_sq_safe;
                phi += q_j * 0.5f * log(r_sq_safe);
                force -= q_j * diff * inv_r2;
            }
        }
    }

    out_potentials[gid] = phi;
    out_forces[gid] = force;
}

// ---------------------------------------------------------------------------
// 4. Ambient Occlusion / Volumetric Multipole Evaluation Kernel
// ---------------------------------------------------------------------------
__kernel void opencl_volumetric_ao_sample(
    __global const float* query_points,   // (N_queries, 3)
    __global const float* cluster_centers, // (N_clusters, 3)
    __global const float* cluster_masses,  // (N_clusters,)
    __global const float* cluster_radii,   // (N_clusters,)
    __global float* out_ao_values,        // (N_queries,)
    const int num_queries,
    const int num_clusters
) {
    int gid = get_global_id(0);
    if (gid >= num_queries) return;

    float qx = query_points[gid * 3 + 0];
    float qy = query_points[gid * 3 + 1];
    float qz = query_points[gid * 3 + 2];

    float total_occ = 0.0f;

    for (int c = 0; c < num_clusters; ++c) {
        float cx = cluster_centers[c * 3 + 0];
        float cy = cluster_centers[c * 3 + 1];
        float cz = cluster_centers[c * 3 + 2];
        float mass = cluster_masses[c];
        float r_eff = cluster_radii[c];

        float dx = qx - cx;
        float dy = qy - cy;
        float dz = qz - cz;
        float dist_sq = dx * dx + dy * dy + dz * dz + 1e-6f;
        float dist = sqrt(dist_sq);

        float occ = (mass * r_eff) / (dist_sq + r_eff * r_eff);
        total_occ += occ;
    }

    out_ao_values[gid] = clamp(1.0f - total_occ * 0.05f, 0.0f, 1.0f);
}

#endif // TREE_FREE_FMM_OPENCL_CL
