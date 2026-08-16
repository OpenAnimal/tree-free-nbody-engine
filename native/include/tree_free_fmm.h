/**
 * Tree-Free Fast Multipole Method (FMM) Native C-ABI Header
 * ========================================================
 * High-performance SIMD CPU backend, Morton bitpacking, 2D/3D Multipole Expansions,
 * and zero-allocation contact solver.
 * Designed for Unreal Engine 5, C++ robotics controllers (MuJoCo/AT-ST/Morphex), and Python ctypes.
 */

#ifndef TREE_FREE_FMM_H
#define TREE_FREE_FMM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef TREE_FREE_FMM_EXPORTS
        #define FMM_API __declspec(dllexport)
    #else
        #define FMM_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define FMM_API __attribute__((visibility("default")))
    #else
        #define FMM_API
    #endif
#endif

// ----------------------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------------------

/**
 * 3D Cartesian Multipole Moments up to Quadrupole (Order 2): 10 floats.
 * Layout: [M0, Dx, Dy, Dz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
 */
typedef struct {
    float m0;
    float dx, dy, dz;
    float qxx, qyy, qzz;
    float qxy, qxz, qyz;
} fmm_multipole_3d_t;

/**
 * 3D Local Taylor Series Coefficients up to 2nd Order: 10 floats.
 * Layout: [L0, Gx, Gy, Gz, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz]
 */
typedef struct {
    float l0;
    float gx, gy, gz;
    float hxx, hyy, hzz;
    float hxy, hxz, hyz;
} fmm_local_3d_t;

// ----------------------------------------------------------------------------
// Core P2P & Spatial Functions
// ----------------------------------------------------------------------------

/**
 * Direct N-Body P2P Potential calculation via SIMD @Vector(8, f32).
 * Computes: out_potentials[i] = \sum_{j \neq i} mass[j] / sqrt(|r_i - r_j|^2 + softening_sq)
 */
void zig_fmm_p2p_potentials(
    const float* px,
    const float* py,
    const float* pz,
    const float* masses,
    size_t num_particles,
    float softening_sq,
    float* out_potentials
);

/**
 * Direct N-Body P2P Force calculation (Fx, Fy, Fz) via SIMD vectorization.
 */
void zig_fmm_p2p_forces(
    const float* px,
    const float* py,
    const float* pz,
    const float* masses,
    size_t num_particles,
    float softening_sq,
    float* out_fx,
    float* out_fy,
    float* out_fz
);

/**
 * 3D Morton-64 Quantization & Bit-Packing (21 bits per axis into 64-bit word).
 */
void zig_fmm_encode_morton3d(
    const float* px,
    const float* py,
    const float* pz,
    size_t num_particles,
    float min_bound,
    float max_bound,
    uint32_t depth,
    uint64_t* out_morton
);

/**
 * 64-Bit Bitboard Occupancy Extraction.
 */
uint64_t zig_fmm_build_bitboard64(
    const uint64_t* morton_codes,
    size_t num_particles,
    uint32_t shift
);

/**
 * Zero-Allocation Matrix-Free IPC Contact Force Solver.
 */
void zig_fmm_contact_forces(
    const float* px,
    const float* py,
    const float* pz,
    size_t num_nodes,
    float dhat,
    float kappa,
    float* out_fx,
    float* out_fy,
    float* out_fz
);

// ----------------------------------------------------------------------------
// 2D Complex Multipole Expansions (Greengard-Rokhlin Laurent series)
// ----------------------------------------------------------------------------

void zig_fmm_2d_p2m(
    const float* px,
    const float* py,
    const float* charges,
    size_t num_particles,
    float cx,
    float cy,
    uint32_t order,
    float* out_moments_re,
    float* out_moments_im
);

void zig_fmm_2d_m2l(
    const float* src_moments_re,
    const float* src_moments_im,
    float src_cx,
    float src_cy,
    float dst_cx,
    float dst_cy,
    uint32_t order,
    float* out_local_re,
    float* out_local_im
);

void zig_fmm_2d_l2p(
    const float* local_re,
    const float* local_im,
    uint32_t order,
    float cx,
    float cy,
    const float* tx,
    const float* ty,
    size_t num_targets,
    float* out_potentials
);

// ----------------------------------------------------------------------------
// 3D Cartesian Multipole Expansions (1/r Gravitational / Coulomb / UE5)
// ----------------------------------------------------------------------------

void zig_fmm_3d_p2m(
    const float* px,
    const float* py,
    const float* pz,
    const float* masses,
    size_t num_particles,
    float cx,
    float cy,
    float cz,
    fmm_multipole_3d_t* out_moments
);

void zig_fmm_3d_m2p(
    const fmm_multipole_3d_t* moments,
    float cx,
    float cy,
    float cz,
    const float* tx,
    const float* ty,
    const float* tz,
    size_t num_targets,
    float softening_sq,
    float* out_potentials
);

void zig_fmm_3d_m2l(
    const fmm_multipole_3d_t* moments,
    float src_cx,
    float src_cy,
    float src_cz,
    float dst_cx,
    float dst_cy,
    float dst_cz,
    float softening_sq,
    fmm_local_3d_t* out_local
);

void zig_fmm_3d_l2p(
    const fmm_local_3d_t* local,
    float cx,
    float cy,
    float cz,
    const float* tx,
    const float* ty,
    const float* tz,
    size_t num_targets,
    float* out_potentials
);

/**
 * Version check (e.g. 110 for 1.1.0).
 */
uint32_t zig_fmm_version(void);

#ifdef __cplusplus
}
#endif

#endif // TREE_FREE_FMM_H
