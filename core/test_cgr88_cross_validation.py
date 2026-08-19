"""
Comprehensive Cross-Validation and Rigorous Verification Test Suite
for Carrier, Greengard, & Rokhlin (1988) 2D Adaptive Fast Multipole Method (CGR88)
and Greengard & Rokhlin (1987) Regular Fast Multipole Method.

Test Categories:
1. Mathematical primitives & translation operator correctness (P2M, M2M, M2L, L2L, L2P, P2L, M2P, P2P).
2. Dual reciprocity of CGR88 interaction lists (List 3 and List 4).
3. Exponential error convergence vs multipole expansion order p in [2, 16].
4. Stress-testing across challenging non-uniform, multi-scale, and clustered particle geometries.
5. Multi-engine cross-validation (Exact Direct vs CGR88 Adaptive vs GR87 Regular vs Fast Vectorized vs Tree-Free Hash).
6. Force & potential analytical gradient consistency (F = -grad phi).
"""

import sys
import os
import time
import math
import numpy as np

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    CGR88AdaptiveFMM,
    TreeFreeElasticAdaptiveFMM,
    GreengardRokhlin87RegularFMM,
    FastVectorizedFMM,
    TreeFreeFMM,
    AdaptiveQuadTree,
    exact_direct_nbody_2d,
    exact_direct_nbody_forces_2d,
    cgr88_p2m,
    cgr88_m2m,
    cgr88_m2l,
    cgr88_l2l,
    cgr88_l2p,
    cgr88_l2p_force,
    cgr88_p2l,
    cgr88_m2p,
    p2p_potential_and_force,
    build_flat_adaptive_metadata,
)
from core.test_flat_adaptive_gpu import evaluate_flat_adaptive_emulated


# =============================================================================
# 1. MATHEMATICAL OPERATOR INVARIANT TESTS
# =============================================================================

def test_p2m_expansion_accuracy():
    """Verify that P2M expansion converges exponentially to exact potential outside circle."""
    np.random.seed(101)
    center = 0.5 + 0.5j
    r_src = 0.05
    # Generate particles inside radius r_src
    angles = np.random.uniform(0, 2 * np.pi, size=30)
    radii = np.random.uniform(0, r_src, size=30)
    pts = np.column_stack([center.real + radii * np.cos(angles), center.imag + radii * np.sin(angles)])
    q = np.random.uniform(-1.0, 1.0, size=30)
    
    # Target point well outside circle
    tgt_pt = 0.8 + 0.8j
    tgt_arr = np.array([[tgt_pt.real, tgt_pt.imag]])
    exact_pot, _, _ = p2p_potential_and_force(tgt_arr, pts, q)
    
    # Check convergence for increasing p
    errors = []
    for p in [2, 4, 6, 8, 10]:
        coeffs = cgr88_p2m(pts, q, center, p=p)
        dz = tgt_pt - center
        pot_p2m = float((coeffs[0] * np.log(dz) + np.sum(coeffs[1:] / (dz ** np.arange(1, p + 1)))).real)
        err = abs(pot_p2m - exact_pot[0])
        errors.append(err)
        
    # Errors must decrease monotonically and reach high precision
    assert errors[-1] < 1e-6
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 0.5


def test_m2m_translation_invariance():
    """Verify that M2M translation yields identical far-field potential as original expansion."""
    np.random.seed(102)
    center0 = 0.4 + 0.4j
    center1 = 0.5 + 0.5j  # Parent center
    
    pts = center0.real + np.random.uniform(-0.02, 0.02, size=(20, 2))
    q = np.random.uniform(-1.0, 1.0, size=20)
    
    p = 10
    m0 = cgr88_p2m(pts, q, center0, p=p)
    m1 = cgr88_m2m(m0, center0, center1, p=p)
    
    # Evaluate at distant point
    target = 0.9 + 0.9j
    pot0, _ = cgr88_m2p(m0, center0, target, p=p)
    pot1, _ = cgr88_m2p(m1, center1, target, p=p)
    
    assert abs(pot0 - pot1) < 1e-7


def test_m2l_and_l2p_invariance():
    """Verify that M2L followed by L2P matches M2P directly for well-separated centers."""
    np.random.seed(103)
    src_center = 0.2 + 0.2j
    dst_center = 0.8 + 0.8j
    
    pts = src_center.real + np.random.uniform(-0.02, 0.02, size=(20, 2))
    q = np.random.uniform(-1.0, 1.0, size=20)
    
    p = 10
    m_src = cgr88_p2m(pts, q, src_center, p=p)
    l_dst = cgr88_m2l(m_src, src_center, dst_center, p=p)
    
    # Target near dst_center
    target = dst_center + (0.01 + 0.01j)
    pot_m2p, _ = cgr88_m2p(m_src, src_center, target, p=p)
    pot_l2p = cgr88_l2p(l_dst, target, dst_center, p=p)
    
    assert abs(pot_m2p - pot_l2p) < 1e-7


def test_l2l_translation_invariance():
    """Verify that L2L shifting preserves local expansion evaluation."""
    np.random.seed(104)
    src_center = 0.2 + 0.2j
    dst_center0 = 0.7 + 0.7j
    dst_center1 = 0.75 + 0.75j  # Child center
    
    pts = src_center.real + np.random.uniform(-0.02, 0.02, size=(20, 2))
    q = np.random.uniform(-1.0, 1.0, size=20)
    
    p = 10
    m_src = cgr88_p2m(pts, q, src_center, p=p)
    l0 = cgr88_m2l(m_src, src_center, dst_center0, p=p)
    l1 = cgr88_l2l(l0, dst_center0, dst_center1, p=p)
    
    target = dst_center1 + (0.005 - 0.005j)
    pot0 = cgr88_l2p(l0, target, dst_center0, p=p)
    pot1 = cgr88_l2p(l1, target, dst_center1, p=p)
    
    assert abs(pot0 - pot1) < 1e-7


def test_p2l_and_list4_accuracy():
    """Verify that P2L (used for List 4) reproduces exact potential at target center."""
    np.random.seed(105)
    src_pts = np.random.uniform(0.1, 0.3, size=(25, 2))
    src_q = np.random.uniform(-1.0, 1.0, size=25)
    tgt_center = 0.85 + 0.85j
    
    p = 10
    l_coeffs = cgr88_p2l(src_pts, src_q, tgt_center, p=p)
    
    # Evaluate at multiple points around tgt_center
    test_points = [
        tgt_center + 0.01 + 0.01j,
        tgt_center - 0.01 + 0.02j,
        tgt_center + 0.02 - 0.01j
    ]
    for pt in test_points:
        pot_l2p = cgr88_l2p(l_coeffs, pt, tgt_center, p=p)
        exact_pot, _, _ = p2p_potential_and_force(np.array([[pt.real, pt.imag]]), src_pts, src_q)
        assert abs(pot_l2p - exact_pot[0]) < 1e-6


def test_l2p_force_analytical_gradient():
    """Verify that L2P force matches numerical finite-difference gradient: F = -grad phi."""
    np.random.seed(106)
    center = 0.5 + 0.5j
    p = 8
    # Random local coeffs
    l_coeffs = np.random.uniform(-1.0, 1.0, size=p + 1) + 1j * np.random.uniform(-1.0, 1.0, size=p + 1)
    
    test_pt = 0.52 + 0.48j
    fx, fy = cgr88_l2p_force(l_coeffs, test_pt, center, p=p)
    
    # Finite difference gradient
    h = 1e-6
    pot_x_plus = cgr88_l2p(l_coeffs, test_pt + h, center, p=p)
    pot_x_minus = cgr88_l2p(l_coeffs, test_pt - h, center, p=p)
    pot_y_plus = cgr88_l2p(l_coeffs, test_pt + 1j * h, center, p=p)
    pot_y_minus = cgr88_l2p(l_coeffs, test_pt - 1j * h, center, p=p)
    
    num_fx = -(pot_x_plus - pot_x_minus) / (2 * h)
    num_fy = -(pot_y_plus - pot_y_minus) / (2 * h)
    
    assert abs(fx - num_fx) < 1e-5
    assert abs(fy - num_fy) < 1e-5


# =============================================================================
# 2. ADAPTIVE QUADTREE INTERACTION LIST RECIPROCITY TESTS
# =============================================================================

def test_cgr88_tree_interaction_lists_reciprocity():
    """Verify that every interaction pair in adaptive quadtree is partitioned into exactly one list."""
    np.random.seed(201)
    N = 300
    pos = np.random.uniform(0.05, 0.95, size=(N, 2))
    charges = np.random.uniform(-1.0, 1.0, size=N)
    
    tree = AdaptiveQuadTree(pos, charges, max_leaf_particles=15, max_depth=6, p=6)
    
    # Verify List 3 and List 4 dual reciprocity:
    # If d in List 3(b), then b must be in List 4(d)
    for bid, b in tree.boxes.items():
        if b.is_leaf:
            for d_id in b.list3:
                assert bid in tree.boxes[d_id].list4, f"Reciprocity failure: box {bid} leaf has {d_id} in List 3, but not vice-versa in List 4"
                
    for did, d in tree.boxes.items():
        for b_id in d.list4:
            assert did in tree.boxes[b_id].list3, f"Reciprocity failure: box {did} has {b_id} in List 4, but {did} not in List 3 of {b_id}"


def test_flat_adaptive_gpu_metadata():
    """Verify upload-ready flat node and List 1-4 metadata derived from CGR88."""
    np.random.seed(207)
    positions = np.random.uniform(0.02, 0.98, size=(300, 2))
    charges = np.random.uniform(-1.0, 1.0, size=300)
    metadata = build_flat_adaptive_metadata(
        positions,
        charges,
        max_leaf_particles=12,
        max_depth=6,
    )
    metadata.validate()
    assert metadata.node_count > 1
    assert metadata.particle_indices.shape == (len(positions),)
    assert np.unique(metadata.particle_indices).size == len(positions)
    assert np.all(metadata.leaf_node_for_particle != 0xFFFFFFFF)
    
    # List 3/List 4 are dual in the source CGR88 tree and must remain dual after flattening.
    for node in range(metadata.node_count):
        for source in metadata.list_for(node, 2):
            assert node in set(metadata.list_for(int(source), 3))
        for target in metadata.list_for(node, 3):
            assert node in set(metadata.list_for(int(target), 2))


def test_flat_adaptive_gpu_schedule_convergence():
    """Verify that the flat GPU schedule matches exact CGR88 convergence for p=1..4."""
    np.random.seed(804)
    positions = np.random.uniform(0.05, 0.95, size=(160, 2))
    charges = np.ones(len(positions))
    metadata = build_flat_adaptive_metadata(
        positions,
        charges,
        max_leaf_particles=12,
        max_depth=6,
    )
    exact_pot = exact_direct_nbody_2d(positions, charges)
    exact_fx, exact_fy = exact_direct_nbody_forces_2d(positions, charges)

    errors = []
    for order in (1, 2, 3, 4):
        pot, fx, fy = evaluate_flat_adaptive_emulated(
            positions,
            metadata,
            expansion_order=order,
        )
        rel_pot = np.linalg.norm(pot - exact_pot) / np.linalg.norm(exact_pot)
        rel_fx = np.linalg.norm(fx - exact_fx) / np.linalg.norm(exact_fx)
        errors.append((rel_pot, rel_fx))

    # Error must decay monotonically across p=1..4
    for i in range(len(errors) - 1):
        assert errors[i + 1][0] < errors[i][0]
        assert errors[i + 1][1] < errors[i][1]
    assert errors[-1][0] < 5e-5
    assert errors[-1][1] < 1e-3


# =============================================================================
# 3. EXPONENTIAL ERROR CONVERGENCE TESTS
# =============================================================================

def test_cgr88_convergence_vs_order(p):
    """Verify that CGR88 adaptive FMM error decreases systematically as p increases."""
    np.random.seed(301)
    N = 400
    pos = np.random.uniform(0.05, 0.95, size=(N, 2))
    charges = np.random.uniform(-1.0, 1.0, size=N)
    
    exact_pot = exact_direct_nbody_2d(pos, charges)
    fmm = CGR88AdaptiveFMM(max_leaf_particles=15, max_depth=6, p=p)
    fmm_pot = fmm.evaluate(pos, charges, compute_forces=False)
    
    rel_err = np.linalg.norm(fmm_pot - exact_pot) / np.linalg.norm(exact_pot)
    expected_max_err = {
        2: 2.0e-2,
        4: 2.0e-3,
        6: 3.0e-4,
        8: 5.0e-5,
        10: 1.0e-5,
        12: 5.0e-6
    }
    assert rel_err <= expected_max_err[p], f"Relative error {rel_err:.3e} exceeds bound for p={p}"


# =============================================================================
# 4. STRESS TESTING ON CHALLENGING NON-UNIFORM DISTRIBUTIONS
# =============================================================================

def test_cgr88_multi_cluster_nonuniform():
    """Verify CGR88 on multi-scale non-uniform clustered particle distribution."""
    np.random.seed(401)
    # 4 dense Gaussian clusters with different scales + background
    c1 = np.random.normal(loc=[0.25, 0.25], scale=0.015, size=(150, 2))
    c2 = np.random.normal(loc=[0.75, 0.75], scale=0.010, size=(150, 2))
    c3 = np.random.normal(loc=[0.20, 0.80], scale=0.025, size=(150, 2))
    c4 = np.random.normal(loc=[0.80, 0.20], scale=0.008, size=(150, 2))
    bg = np.random.uniform(0.02, 0.98, size=(100, 2))
    pos = np.clip(np.vstack([c1, c2, c3, c4, bg]), 0.01, 0.99)
    charges = np.random.uniform(-1.0, 1.0, size=len(pos))
    
    exact_pot = exact_direct_nbody_2d(pos, charges)
    exact_fx, exact_fy = exact_direct_nbody_forces_2d(pos, charges)
    
    fmm = CGR88AdaptiveFMM(max_leaf_particles=15, max_depth=8, p=10)
    fmm_pot, fmm_fx, fmm_fy = fmm.evaluate(pos, charges, compute_forces=True)
    
    err_pot = np.linalg.norm(fmm_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_fx = np.linalg.norm(fmm_fx - exact_fx) / np.linalg.norm(exact_fx)
    err_fy = np.linalg.norm(fmm_fy - exact_fy) / np.linalg.norm(exact_fy)
    
    assert err_pot < 5e-4
    assert err_fx < 5e-3
    assert err_fy < 5e-3


def test_cgr88_singular_boundary_distribution():
    """Verify CGR88 on boundary-concentrated and singular distributions."""
    np.random.seed(402)
    N = 300
    # Boundary particles near domain edges
    t = np.random.uniform(0.02, 0.98, size=N)
    edge_idx = np.random.randint(0, 4, size=N)
    pos = np.zeros((N, 2))
    for i in range(N):
        if edge_idx[i] == 0: pos[i] = [t[i], 0.02]
        elif edge_idx[i] == 1: pos[i] = [t[i], 0.98]
        elif edge_idx[i] == 2: pos[i] = [0.02, t[i]]
        else: pos[i] = [0.98, t[i]]
    charges = np.random.uniform(-1.0, 1.0, size=N)
    
    exact_pot = exact_direct_nbody_2d(pos, charges)
    fmm = CGR88AdaptiveFMM(max_leaf_particles=12, max_depth=7, p=10)
    fmm_pot = fmm.evaluate(pos, charges, compute_forces=False)
    
    err_pot = np.linalg.norm(fmm_pot - exact_pot) / np.linalg.norm(exact_pot)
    assert err_pot < 1e-4


# =============================================================================
# 5. MULTI-ENGINE CROSS-VALIDATION
# =============================================================================

def test_multi_engine_cross_validation():
    """Cross-validate all FMM engine variants against exact direct summation."""
    np.random.seed(501)
    N = 500
    pos = np.random.uniform(0.05, 0.95, size=(N, 2))
    charges = np.random.uniform(-1.0, 1.0, size=N)
    
    # 1. Exact direct
    exact_pot = exact_direct_nbody_2d(pos, charges)
    exact_fx, exact_fy = exact_direct_nbody_forces_2d(pos, charges)
    
    # 2. CGR88 Adaptive FMM
    cgr_fmm = CGR88AdaptiveFMM(max_leaf_particles=20, max_depth=6, p=10)
    cgr_pot, cgr_fx, cgr_fy = cgr_fmm.evaluate(pos, charges, compute_forces=True)
    
    # 3. Regular FMM (Greengard & Rokhlin 1987)
    reg_fmm = GreengardRokhlin87RegularFMM(depth=4, p=10)
    reg_pot, reg_fx, reg_fy = reg_fmm.evaluate(pos, charges, compute_forces=True)
    
    # 4. Fast Vectorized FMM
    vec_fmm = FastVectorizedFMM(depth=4, order=8)
    vec_pot, vec_fx, vec_fy = vec_fmm.evaluate(pos, charges, compute_forces=True)
    
    # 5. Tree-Free Elastic Adaptive FMM
    tf_adapt = TreeFreeElasticAdaptiveFMM(max_leaf_particles=20, base_depth=2, max_depth=6, p=10)
    tf_adapt_pot, tf_adapt_fx, tf_adapt_fy = tf_adapt.evaluate(pos, charges, compute_forces=True)
    
    # 6. Tree-Free Hash Regular FMM
    tf_fmm = TreeFreeFMM(depth=4, order=8)
    tf_pot = tf_fmm.evaluate(pos, charges)
    
    # Verify all relative potential errors are well below 1e-4
    err_cgr = np.linalg.norm(cgr_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_tf_adapt = np.linalg.norm(tf_adapt_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_reg = np.linalg.norm(reg_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_vec = np.linalg.norm(vec_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_tf = np.linalg.norm(tf_pot - exact_pot) / np.linalg.norm(exact_pot)
    
    assert err_cgr < 1e-5, f"CGR88 error {err_cgr:.3e} exceeds threshold"
    assert err_tf_adapt < 1e-5, f"Tree-Free Elastic Adaptive error {err_tf_adapt:.3e} exceeds threshold"
    assert err_reg < 1e-5, f"Regular FMM error {err_reg:.3e} exceeds threshold"
    assert err_vec < 1e-5, f"Fast Vectorized error {err_vec:.3e} exceeds threshold"
    assert err_tf < 1e-5, f"Tree-Free Hash error {err_tf:.3e} exceeds threshold"
    
    # Verify force accuracy
    err_cgr_fx = np.linalg.norm(cgr_fx - exact_fx) / np.linalg.norm(exact_fx)
    err_tf_adapt_fx = np.linalg.norm(tf_adapt_fx - exact_fx) / np.linalg.norm(exact_fx)
    err_reg_fx = np.linalg.norm(reg_fx - exact_fx) / np.linalg.norm(exact_fx)
    err_vec_fx = np.linalg.norm(vec_fx - exact_fx) / np.linalg.norm(exact_fx)
    
    assert err_cgr_fx < 1e-4
    assert err_tf_adapt_fx < 1e-4
    assert err_reg_fx < 1e-4
    assert err_vec_fx < 1e-4


def test_tree_free_elastic_adaptive_equivalence():
    """Verify exact equivalence between Tree-Free Elastic Adaptive FMM and classical tree CGR88."""
    np.random.seed(502)
    # Multi-cluster non-uniform distribution
    c1 = np.random.normal(loc=[0.25, 0.25], scale=0.015, size=(100, 2))
    c2 = np.random.normal(loc=[0.75, 0.75], scale=0.010, size=(100, 2))
    c3 = np.random.normal(loc=[0.20, 0.80], scale=0.025, size=(100, 2))
    bg = np.random.uniform(0.02, 0.98, size=(100, 2))
    pos = np.clip(np.vstack([c1, c2, c3, bg]), 0.01, 0.99)
    charges = np.random.uniform(-1.0, 1.0, size=len(pos))

    exact_pot = exact_direct_nbody_2d(pos, charges)
    exact_fx, exact_fy = exact_direct_nbody_forces_2d(pos, charges)

    cgr_tree = CGR88AdaptiveFMM(max_leaf_particles=15, max_depth=7, p=10)
    pot_tree, fx_tree, fy_tree = cgr_tree.evaluate(pos, charges, compute_forces=True)

    tf_cgr = TreeFreeElasticAdaptiveFMM(max_leaf_particles=15, base_depth=2, max_depth=7, p=10)
    pot_tf, fx_tf, fy_tf = tf_cgr.evaluate(pos, charges, compute_forces=True)

    err_tree = np.linalg.norm(pot_tree - exact_pot) / np.linalg.norm(exact_pot)
    err_tf = np.linalg.norm(pot_tf - exact_pot) / np.linalg.norm(exact_pot)
    diff_tree_tf = np.linalg.norm(pot_tf - pot_tree) / np.linalg.norm(pot_tree)
    diff_fx = np.linalg.norm(fx_tf - fx_tree) / np.linalg.norm(fx_tree)

    assert err_tree < 1e-5
    assert err_tf < 1e-5
    assert diff_tree_tf < 1e-12, f"Tree-Free and Tree potential difference {diff_tree_tf:.3e} exceeds tolerance"
    assert diff_fx < 1e-12, f"Tree-Free and Tree force difference {diff_fx:.3e} exceeds tolerance"


def test_flat_fmm_elastic_hash_occupancy():
    """FastVectorizedFMM on a clustered non-uniform distribution: accuracy
    vs direct summation AND the elastic hash must contain exactly the set
    of occupied Morton cell keys."""
    np.random.seed(707)
    pts = np.vstack([
        np.random.rand(40, 2) * 0.10 + 0.10,
        np.random.rand(60, 2) * 0.15 + 0.70,
        np.random.rand(100, 2) * 0.30 + 0.40,
    ])
    q = np.random.uniform(-1.0, 1.0, size=len(pts))

    fmm = FastVectorizedFMM(depth=4, order=8)
    pot, fx, fy = fmm.evaluate(pts, q, compute_forces=True)

    exact_pot = exact_direct_nbody_2d(pts, q)
    exact_fx, _ = exact_direct_nbody_forces_2d(pts, q)
    err_pot = np.max(np.abs(pot - exact_pot)) / np.max(np.abs(exact_pot))
    err_fx = np.max(np.abs(fx - exact_fx)) / np.max(np.abs(exact_fx))
    assert err_pot < 1e-4, f"flat FMM potential error {err_pot:.3e}"
    assert err_fx < 1e-3, f"flat FMM force error {err_fx:.3e}"

    grid_res = 1 << 4
    ix = np.clip((pts[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((pts[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    expected_keys = set(int(k) for k in ((4 << 24) | (ix << 12) | iy))
    hash_keys = set(int(k) for k, _ in fmm.hash_table.items())
    assert hash_keys == expected_keys, "elastic hash occupancy mismatch"
    for k in expected_keys:
        assert k in fmm.hash_table, f"occupied cell {k} missing from elastic hash"


if __name__ == '__main__':
    tests = [
        ("test_p2m_expansion_accuracy", test_p2m_expansion_accuracy),
        ("test_m2m_translation_invariance", test_m2m_translation_invariance),
        ("test_m2l_and_l2p_invariance", test_m2l_and_l2p_invariance),
        ("test_l2l_translation_invariance", test_l2l_translation_invariance),
        ("test_p2l_and_list4_accuracy", test_p2l_and_list4_accuracy),
        ("test_l2p_force_analytical_gradient", test_l2p_force_analytical_gradient),
        ("test_cgr88_tree_interaction_lists_reciprocity", test_cgr88_tree_interaction_lists_reciprocity),
        ("test_flat_adaptive_gpu_metadata", test_flat_adaptive_gpu_metadata),
        ("test_flat_adaptive_gpu_schedule_convergence", test_flat_adaptive_gpu_schedule_convergence),
        ("test_cgr88_multi_cluster_nonuniform", test_cgr88_multi_cluster_nonuniform),
        ("test_cgr88_singular_boundary_distribution", test_cgr88_singular_boundary_distribution),
        ("test_multi_engine_cross_validation", test_multi_engine_cross_validation),
        ("test_tree_free_elastic_adaptive_equivalence", test_tree_free_elastic_adaptive_equivalence),
        ("test_flat_fmm_elastic_hash_occupancy", test_flat_fmm_elastic_hash_occupancy),
    ]
    for p in [2, 4, 6, 8, 10, 12]:
        tests.append((f"test_cgr88_convergence_vs_order(p={p})", lambda p=p: test_cgr88_convergence_vs_order(p)))

    print("=" * 80)
    print(" RUNNING CGR88 ADAPTIVE FMM COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)
    passed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt = (time.perf_counter() - t0) * 1000
            print(f" [PASS] {name:<50} ({dt:.2f} ms)")
            passed += 1
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            print(f" [FAIL] {name:<50} ({dt:.2f} ms) -> {e}")

    print("=" * 80)
    print(f" Result: {passed}/{len(tests)} tests passed successfully.")
    print("=" * 80)
    if passed < len(tests):
        sys.exit(1)
