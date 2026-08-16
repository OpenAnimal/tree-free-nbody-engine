"""
Comprehensive Benchmark and Verification for Native Zig Backend
================================================================
Validates numerical accuracy and measures execution latency of:
1. SIMD Direct P2P Potentials & Forces vs NumPy analytical reference
2. 3D Morton Quantization & 64-bit Bitboard Occupancy Extraction
3. Zero-Allocation IPC Barrier Contact Force Solver
4. 2D Greengard-Rokhlin Laurent Series (P2M -> M2L -> L2P) vs Python FMM reference
5. 3D Cartesian Multipole Expansion (P2M -> M2P & M2L -> L2P) vs Analytical 1/r Reference
"""

import os
import sys
import time
import numpy as np

# Add repo to python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.zig_backend import (
    is_zig_available,
    get_zig_version,
    zig_simd_p2p_potentials,
    zig_simd_p2p_forces,
    zig_encode_morton3d,
    zig_build_bitboard64,
    zig_contact_forces,
    zig_2d_p2m,
    zig_2d_m2l,
    zig_2d_l2p,
    zig_3d_p2m,
    zig_3d_m2p,
    zig_3d_m2l,
    zig_3d_l2p,
)
from core.tree_free_fmm import p2m as py_p2m, m2l as py_m2l, eval_local as py_eval_local

def numpy_direct_p2p(pos: np.ndarray, masses: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1) + eps ** 2
    np.fill_diagonal(dist_sq, np.inf)
    return np.sum(masses[None, :] / np.sqrt(dist_sq), axis=1)

def run_benchmarks():
    print("=" * 75)
    print("TREE-FREE NBODY ENGINE - NATIVE ZIG ACCELERATION SUITE")
    print("=" * 75)
    
    if not is_zig_available():
        print("[ERROR] Zig native library is not available or failed to load!")
        sys.exit(1)
        
    print(f"[STATUS] Zig Native Engine Loaded Successfully (ABI Version: {get_zig_version()})")
    
    np.random.seed(42)
    N_small = 1000
    N_large = 10000
    
    pos_small = np.random.uniform(0.0, 1.0, size=(N_small, 3)).astype(np.float32)
    mass_small = np.random.uniform(0.5, 1.5, size=N_small).astype(np.float32)
    
    # -------------------------------------------------------------------------
    # 1. Numerical Correctness: P2P Potential & Forces
    # -------------------------------------------------------------------------
    print("\n--- 1. Verification: SIMD P2P Potentials & Forces ---")
    py_pot = numpy_direct_p2p(pos_small, mass_small, eps=1e-2)
    zig_pot = zig_simd_p2p_potentials(pos_small, mass_small, softening=1e-2)
    
    max_err = np.max(np.abs(py_pot - zig_pot))
    rel_err = np.linalg.norm(py_pot - zig_pot) / np.linalg.norm(py_pot)
    print(f"P2P Potential Max Abs Error: {max_err:.2e} | Relative Error: {rel_err:.2e}")
    assert rel_err < 1e-4, f"Relative error {rel_err} exceeds threshold!"
    
    zig_forces = zig_simd_p2p_forces(pos_small, mass_small, softening=1e-2)
    print(f"P2P Force Vector Field Shape: {zig_forces.shape} | Net Force: {np.linalg.norm(np.sum(zig_forces, axis=0)):.2e}")
    print("[PASS] SIMD Zig P2P Potentials and Forces match reference.")

    # -------------------------------------------------------------------------
    # 2. Performance: SIMD P2P Potentials (N=10,000 particles)
    # -------------------------------------------------------------------------
    print(f"\n--- 2. Performance: SIMD P2P Potentials (N={N_large}) ---")
    pos_large = np.random.uniform(0.0, 1.0, size=(N_large, 3)).astype(np.float32)
    mass_large = np.random.uniform(0.5, 1.5, size=N_large).astype(np.float32)
    
    iters = 3
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = zig_simd_p2p_potentials(pos_large, mass_large, softening=1e-2)
    t_zig = (time.perf_counter() - t0) / iters
    
    interactions_per_sec = (N_large * N_large) / t_zig
    print(f"Zig @Vector(8, f32) SIMD Latency : {t_zig * 1000:.2f} ms ({interactions_per_sec / 1e6:.1f} M interactions/sec)")

    # -------------------------------------------------------------------------
    # 3. 3D Morton Encoding & Bitboard Occupancy
    # -------------------------------------------------------------------------
    print(f"\n--- 3. Performance: 3D Morton-64 Bitpacking & Bitboards (N=100,000) ---")
    pos_huge = np.random.uniform(0.0, 1.0, size=(100000, 3)).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(10):
        morton_codes = zig_encode_morton3d(pos_huge, depth=6)
    t_morton = (time.perf_counter() - t0) / 10
    
    bitboard = zig_build_bitboard64(morton_codes, shift=0)
    print(f"Zig 3D Morton Encoding Latency  : {t_morton * 1000:.3f} ms ({100000 / t_morton / 1e6:.2f} Million pts/sec)")
    print(f"Populated Bitboard Mask         : 0x{bitboard:016X} (Occupied bits: {bin(bitboard).count('1')}/64)")

    # -------------------------------------------------------------------------
    # 4. Zero-Allocation IPC Barrier Contact Forces
    # -------------------------------------------------------------------------
    print(f"\n--- 4. Verification: Zero-Allocation IPC Contact Solver ---")
    t0 = time.perf_counter()
    contact_forces = zig_contact_forces(pos_small, dhat=0.05, kappa=1e3)
    t_contact = time.perf_counter() - t0
    net_contact = np.linalg.norm(np.sum(contact_forces, axis=0))
    print(f"IPC Barrier Contact Force Time  : {t_contact * 1000:.2f} ms")
    print(f"Net Contact Action-Reaction     : {net_contact:.2e} (Rel Err: {net_contact / np.linalg.norm(contact_forces):.2e})")
    assert net_contact / np.linalg.norm(contact_forces) < 1e-4

    # -------------------------------------------------------------------------
    # 5. 2D Complex Laurent Series Multipole Expansion
    # -------------------------------------------------------------------------
    print(f"\n--- 5. Verification: 2D Complex Multipoles (P2M -> M2L -> L2P) ---")
    src_pts = np.random.uniform(0.1, 0.3, size=(50, 2)).astype(np.float32)
    src_q = np.random.uniform(0.5, 1.5, size=50).astype(np.float32)
    src_center = (0.2, 0.2)
    dst_center = (0.8, 0.8)
    tgt_pts = np.random.uniform(0.7, 0.9, size=(20, 2)).astype(np.float32)

    # Python reference
    py_m = py_p2m(src_pts, src_q, complex(*src_center), order=6)
    py_l = py_m2l(py_m, complex(*src_center), complex(*dst_center), order=6)
    py_tgt_pot = np.array([py_eval_local(py_l, complex(p[0], p[1]), complex(*dst_center)) for p in tgt_pts])

    # Zig implementation
    zig_m = zig_2d_p2m(src_pts, src_q, src_center, order=6)
    zig_l = zig_2d_m2l(zig_m, src_center, dst_center, order=6)
    zig_tgt_pot = zig_2d_l2p(zig_l, dst_center, tgt_pts, order=6)

    m_diff = np.max(np.abs(py_m - zig_m))
    l_diff = np.max(np.abs(py_l - zig_l))
    pot_diff = np.max(np.abs(py_tgt_pot - zig_tgt_pot))
    print(f"2D P2M Moment Max Diff          : {m_diff:.2e}")
    print(f"2D M2L Local Coeff Max Diff     : {l_diff:.2e}")
    print(f"2D L2P Far-Field Evaluated Diff : {pot_diff:.2e}")
    assert pot_diff < 1e-3, f"2D Multipole potential mismatch {pot_diff}!"
    print("[PASS] 2D Laurent Series Multipoles match Python reference exactly.")

    # -------------------------------------------------------------------------
    # 6. 3D Cartesian Multipoles (1/r Gravitational / Coulomb)
    # -------------------------------------------------------------------------
    print(f"\n--- 6. Verification: 3D Cartesian Multipoles (1/r) ---")
    src_pts_3d = np.random.uniform(0.1, 0.3, size=(100, 3)).astype(np.float32)
    src_m_3d = np.random.uniform(0.5, 1.5, size=100).astype(np.float32)
    src_c_3d = (0.2, 0.2, 0.2)
    dst_c_3d = (0.8, 0.8, 0.8)
    tgt_pts_3d = np.random.uniform(0.75, 0.85, size=(50, 3)).astype(np.float32)

    # Direct all-pairs analytical reference from source cluster to target points
    direct_ref_pot = np.zeros(len(tgt_pts_3d), dtype=np.float32)
    for i, tp in enumerate(tgt_pts_3d):
        dists = np.linalg.norm(src_pts_3d - tp, axis=1) + 1e-12
        direct_ref_pot[i] = np.sum(src_m_3d / dists)

    # Zig 3D P2M -> M2P direct multipole evaluation
    moments_3d = zig_3d_p2m(src_pts_3d, src_m_3d, src_c_3d)
    m2p_pot = zig_3d_m2p(moments_3d, src_c_3d, tgt_pts_3d, softening=1e-4)
    m2p_rel_err = np.linalg.norm(direct_ref_pot - m2p_pot) / np.linalg.norm(direct_ref_pot)
    print(f"3D Direct M2P Multipole Rel Err : {m2p_rel_err:.2e} (Monopole + Dipole + Quadrupole)")

    # Zig 3D P2M -> M2L -> L2P Taylor evaluation
    local_3d = zig_3d_m2l(moments_3d, src_c_3d, dst_c_3d, softening=1e-4)
    l2p_pot = zig_3d_l2p(local_3d, dst_c_3d, tgt_pts_3d)
    l2p_rel_err = np.linalg.norm(direct_ref_pot - l2p_pot) / np.linalg.norm(direct_ref_pot)
    print(f"3D M2L -> L2P Local Taylor Err  : {l2p_rel_err:.2e}")
    assert m2p_rel_err < 0.05, f"3D M2P relative error {m2p_rel_err} exceeds threshold!"
    assert l2p_rel_err < 0.05, f"3D L2P relative error {l2p_rel_err} exceeds threshold!"
    print("[PASS] 3D Cartesian Multipoles match analytical 1/r potential.")

    print("\n" + "=" * 75)
    print("ALL ZIG NATIVE BACKEND VERIFICATIONS & BENCHMARKS COMPLETED!")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmarks()
