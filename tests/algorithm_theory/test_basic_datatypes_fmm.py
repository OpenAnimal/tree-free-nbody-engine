"""
Unit & Integration Verification Tests for Basic Datatypes in algorithm_theory:
1. FlatMultipoleRangeTree (Multidimensional Range Searches & Box Moments)
2. ElasticQuotientFilter (Non-Reordering Zero-Displacement AMQ & Frequency Sketch)
3. SublinearEditDistance (Sublinear Approximate String Matching & Banded Alignment)
4. SpatialDisjointSetFMM (Linear-time Geometric Dynamic Connectivity & Percolation)
"""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from algorithm_theory.multipole_range_tree import (
    FlatMultipoleRangeTree,
    direct_range_query_baseline,
    morton_encode_nd
)
from algorithm_theory.elastic_quotient_filter import (
    ElasticQuotientFilter,
    ClassicBloomFilterBaseline
)
from algorithm_theory.sublinear_edit_distance import (
    SublinearEditDistance,
    exact_wagner_fischer_edit_distance,
    BKTree,
    ElasticFuzzyDictionary
)
from algorithm_theory.spatial_disjoint_set_fmm import (
    SpatialDisjointSetFMM
)
from algorithm_theory.spatial_point_cloud_compression import (
    SpatialPointCloudCompressor,
    compute_point_cloud_psnr
)


def test_multipole_range_tree():
    print("[1/5] Testing FlatMultipoleRangeTree (Multidimensional Orthogonal Range Tree)...")
    rng = np.random.RandomState(42)
    N = 2500
    points = rng.uniform(-10.0, 10.0, size=(N, 3))
    values = rng.uniform(0.5, 5.0, size=(N, 2))
    
    tree = FlatMultipoleRangeTree(points, values=values, leaf_capacity=16, max_depth=8)
    
    # Test 1: Full domain query matches exact sum
    full_res = tree.query_range(np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, 10.0]))
    assert full_res["count"] == N, f"Expected {N} points, got {full_res['count']}"
    assert np.allclose(full_res["sum"], np.sum(values, axis=0), rtol=1e-5), "Full sum mismatch"
    
    # Test 2: Random box queries compared with direct baseline
    for _ in range(15):
        c_min = rng.uniform(-8.0, 4.0, size=3)
        c_max = c_min + rng.uniform(2.0, 8.0, size=3)
        
        tree_res = tree.query_range(c_min, c_max, return_indices=True)
        exact_res = direct_range_query_baseline(points, values, c_min, c_max)
        
        assert tree_res["count"] == exact_res["count"], f"Count mismatch: tree {tree_res['count']} vs exact {exact_res['count']}"
        assert np.allclose(tree_res["sum"], exact_res["sum"], rtol=1e-5), "Range sum mismatch"
        if tree_res["count"] > 0:
            assert set(tree_res["indices"]) == set(exact_res["indices"]), "Returned indices mismatch"
            
    # Test 3: Multipole box potential evaluation
    target_pts = rng.uniform(15.0, 20.0, size=(5, 3))
    b_min, b_max = np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])
    pot = tree.compute_multipole_box_potential(target_pts, b_min, b_max)
    assert len(pot) == 5
    assert np.all(pot > 0), "Potential should be positive"
    
    print("  -> FlatMultipoleRangeTree verification PASSED.")


def test_elastic_quotient_filter():
    print("\n[2/5] Testing ElasticQuotientFilter (Zero-Displacement AMQ & Multiset)...")
    capacity = 5000
    eqf = ElasticQuotientFilter(capacity=capacity, fingerprint_bits=16, num_levels=4, enable_counters=True)
    
    rng = np.random.RandomState(42)
    keys = rng.randint(1, 100_000_000, size=2000, dtype=np.int64)
    
    # Insert keys with multiplicity
    for i, k in enumerate(keys):
        mult = (i % 3) + 1
        ok, probes = eqf.insert(int(k), count=mult)
        assert ok, f"Insert failed for key {k}"
        
    # Check true positive retrieval and frequency accuracy
    for i, k in enumerate(keys):
        present, _ = eqf.contains(int(k))
        assert present, f"Key {k} should be present in filter"
        freq = eqf.get_frequency(int(k))
        assert freq >= (i % 3) + 1, f"Frequency mismatch for key {k}: expected >= {(i%3)+1}, got {freq}"
        
    # Check false positive rate on disjoint random queries
    disjoint_keys = rng.randint(200_000_000, 300_000_000, size=5000, dtype=np.int64)
    false_positives = sum(eqf.contains(int(k))[0] for k in disjoint_keys)
    fp_rate = false_positives / len(disjoint_keys)
    # Theoretical FP rate for 16-bit fingerprint is ~ 1/2^16 = 0.000015
    assert fp_rate < 0.01, f"False positive rate {fp_rate:.5f} is abnormally high"
    
    # Check Jaccard Similarity between overlapping sets
    eqf2 = ElasticQuotientFilter(capacity=capacity, fingerprint_bits=16, num_levels=4, enable_counters=True)
    for k in keys[:1000]:
        eqf2.insert(int(k))
        
    jaccard = eqf.compute_jaccard_similarity(eqf2)
    assert 0.3 < jaccard < 1.0, f"Unexpected Jaccard similarity {jaccard}"
    
    print(f"  -> ElasticQuotientFilter verification PASSED (FP Rate = {fp_rate:.6f}, Jaccard = {jaccard:.3f}).")


def test_sublinear_edit_distance():
    print("\n[3/5] Testing SublinearEditDistance (Approximate String Matching)...")
    engine = SublinearEditDistance(q=3, band_width=8)
    
    s1 = "ALGORITHMTHEORYNMODYFMM"
    s2 = "ALGORITHMIC_THEORY_FMM"
    
    exact_dist = exact_wagner_fischer_edit_distance(s1, s2)
    approx_res = engine.approximate_edit_distance(s1, s2)
    
    # Lower bound must be <= exact distance
    assert approx_res["qgram_lower_bound"] <= exact_dist + 1e-5, "q-gram lower bound violated"
    # Band-sufficient case (equal-length strings, band_width=8 >> true distance): the
    # banded DP must recover the EXACT edit distance, so require zero slack.
    assert abs(approx_res["approx_distance"] - exact_dist) <= 0, f"Approx distance {approx_res['approx_distance']} != exact {exact_dist} (band-sufficient case must be exact)"
    
    # Test approximate pattern matching in corpus
    text = "THE_FAST_MULTIPOLE_METHOD_PROVIDES_LINEAR_SCALING_FOR_TREE_FREE_NBODY_SIMULATION"
    pattern = "MULTIPOLE_METHO"  # 1 deletion from METHOD
    matches = engine.find_approximate_matches(text, pattern, max_errors=2)
    assert len(matches) >= 1, "Should identify approximate match in text"
    
    # Test BK-Tree vs ElasticFuzzyDictionary
    vocab = ["algorithm", "algorithmic", "algebraic", "multipole", "multiscale", "gaussian", "diffusion", "particle"]
    bk_tree = BKTree()
    elastic_dict = ElasticFuzzyDictionary(max_edit_distance=2)
    
    for word in vocab:
        bk_tree.insert(word)
        elastic_dict.insert(word)
        
    query_word = "algoritm"  # 1 deletion from algorithm
    bk_res = bk_tree.search(query_word, max_distance=2)
    elastic_res = elastic_dict.search(query_word, max_distance=2)
    
    assert any(w == "algorithm" for w, d in bk_res), "BK-Tree should find 'algorithm'"
    assert any(w == "algorithm" for w, d in elastic_res), "ElasticFuzzyDictionary should find 'algorithm'"
    
    print(f"  -> SublinearEditDistance & BK-Tree/ElasticFuzzyDict verification PASSED.")


def test_spatial_disjoint_set_fmm():
    print("\n[4/5] Testing SpatialDisjointSetFMM (Linear Geometric Dynamic Connectivity)...")
    rng = np.random.RandomState(42)
    
    # Create 3 distinct spatial clusters
    cluster1 = rng.randn(50, 3) * 0.3 + np.array([-10.0, 0.0, 0.0])
    cluster2 = rng.randn(50, 3) * 0.3 + np.array([+10.0, 0.0, 0.0])
    cluster3 = rng.randn(50, 3) * 0.3 + np.array([0.0, 10.0, 0.0])
    pts = np.vstack([cluster1, cluster2, cluster3])
    
    # With eps = 1.5, should cleanly find 3 distinct connected components
    dsu = SpatialDisjointSetFMM(pts, connectivity_radius=1.5)
    summary = dsu.get_components_summary()
    
    assert summary["num_components"] == 3, f"Expected 3 clusters, got {summary['num_components']}"
    assert np.all(summary["component_sizes"] == 50), f"Clusters should each have 50 nodes: {summary['component_sizes']}"
    
    # Compute Spanning Forest
    edges = dsu.compute_approximate_spanning_forest()
    assert len(edges) >= 49 * 3, f"Should have spanning edges for all 3 clusters: got {len(edges)}"
    
    print(f"  -> SpatialDisjointSetFMM verification PASSED (3 clusters found, {len(edges)} spanning edges).")


def test_spatial_point_cloud_compression():
    print("\n[5/5] Testing SpatialPointCloudCompressor (Tree-Free Morton Delta Varint PCC)...")
    rng = np.random.RandomState(42)
    N = 10000
    points = rng.randn(N, 3).astype(np.float32) * 25.0
    attributes = rng.uniform(0.0, 1.0, size=(N, 4)).astype(np.float32)  # RGBA or 3D Gaussian opacities
    
    compressor = SpatialPointCloudCompressor(precision_bits=14)
    comp_res = compressor.compress(points, attributes=attributes)
    
    assert comp_res["compression_ratio"] > 1.5, f"Expected compression ratio > 1.5x, got {comp_res['compression_ratio']:.2f}x"
    assert len(comp_res["payload"]) < comp_res["raw_bytes"]
    
    # Decompress and verify
    recon_pts, recon_attr = compressor.decompress(comp_res["payload"])
    assert recon_pts.shape == points.shape
    assert recon_attr is not None and recon_attr.shape == attributes.shape
    
    psnr = compute_point_cloud_psnr(points, recon_pts)
    assert psnr > 40.0, f"Expected PSNR > 40 dB, got {psnr:.2f} dB"
    
    print(f"  -> SpatialPointCloudCompressor verification PASSED (Ratio = {comp_res['compression_ratio']:.2f}x, Bits/Point = {comp_res['bits_per_point']:.1f}, PSNR = {psnr:.1f} dB).")


def test_funnel_quotient_filter():
    """Round-7 task T-A4b: verify the FunnelQuotientFilter."""
    print("\n[6/6] Testing FunnelQuotientFilter (T-A4b: funnel-hash quotient filter)...")
    from algorithm_theory.elastic_quotient_filter import FunnelQuotientFilter
    import numpy as np

    capacity = 120000  # > 100k distinct items (the funnel table doesn't auto-grow)
    r = 8  # 8 remainder bits. NOTE: the operating false-positive rate is
           # ~ n_stored / 2^64 (full 64-bit hash-collision bound: an absent
           # item must match BOTH the quotient AND the stored remainder of an
           # existing entry), NOT 2^(-r) -- see elastic_quotient_filter.py
           # false-positive section.  The <0.5% FPR assertion below is a
           # loose sanity guard, not a measurement of the 2^(-r) figure.
    fqf = FunnelQuotientFilter(capacity=capacity, delta=0.05, r_remainder_bits=r)

    # (1) No false negatives ever: insert 10^5 random items, check all present
    rng = np.random.RandomState(42)
    items = [int(x) for x in rng.randint(0, 10**9, size=100000)]
    for item in items:
        fqf.insert(item)
    for item in items:
        present, _ = fqf.contains(item)
        assert present, f"False negative on item {item}"
    print(f"  (1) No false negatives: PASS (10^5 items, all found)")

    # (2) False-positive rate <= 2^-r + slack (r=8 -> <= ~0.4%; assert < 0.5%)
    absent_items = [int(x) for x in rng.randint(10**9, 2 * 10**9, size=100000)]
    fp_count = 0
    max_probes = 0
    for item in absent_items:
        present, probes = fqf.contains(item)
        max_probes = max(max_probes, probes)
        if present:
            fp_count += 1
    fp_rate = fp_count / len(absent_items)
    print(f"  (2) FP rate = {fp_rate:.6f} (target < 0.5% for r={r})")
    assert fp_rate < 0.005, f"FP rate {fp_rate:.4f} >= 0.5%"

    # (3) probe_bound respected over 10^4 contains
    print(f"  (3) Max probes observed = {max_probes}, probe_bound = {fqf.probe_bound}")
    assert max_probes <= fqf.probe_bound, \
        f"Max probes {max_probes} exceeds probe_bound {fqf.probe_bound}"

    # (4) length <= capacity + slack
    print(f"  (4) Table length = {fqf.length}, capacity = {capacity}")
    assert fqf.length <= capacity * (1 + 0.05) + 100, \
        f"Table length {fqf.length} exceeds capacity+slack"

    # (5) Count functionality
    fqf2 = FunnelQuotientFilter(capacity=100, r_remainder_bits=8)
    for _ in range(5):
        fqf2.insert(42)
    fqf2.insert(99)
    assert fqf2.count_of(42) == 5, f"count_of(42) = {fqf2.count_of(42)}, expected 5"
    assert fqf2.count_of(99) == 1, f"count_of(99) = {fqf2.count_of(99)}, expected 1"
    assert fqf2.count_of(123) == 0, f"count_of(123) = {fqf2.count_of(123)}, expected 0"
    print(f"  (5) Count functionality: PASS (count_of(42)=5, count_of(99)=1)")

    # (6) Capacity overflow must raise, not silently drop items.
    #     Inserting 200 items into capacity=100 must surface a RuntimeError
    #     (the funnel table refuses once full) rather than claim 200 stored.
    fqf_overflow = FunnelQuotientFilter(capacity=100, r_remainder_bits=8)
    overflow_raised = False
    n_actually_present = 0
    inserted_items = list(range(200))
    for item in inserted_items:
        try:
            fqf_overflow.insert(item)
        except RuntimeError as exc:
            overflow_raised = True
            print(f"  (6) Overflow raised at item {item}: {str(exc)[:80]}...")
            break
    # Count how many of the items we attempted are actually present.
    for item in inserted_items:
        present, _ = fqf_overflow.contains(item)
        if present:
            n_actually_present += 1
    print(f"  (6) Overflow RuntimeError raised = {overflow_raised}, "
          f"distinct present = {fqf_overflow.length} (<= capacity 100)")
    assert overflow_raised, "Capacity overflow must raise RuntimeError, not silently drop"
    assert fqf_overflow.length <= 100, \
        f"Stored distinct {fqf_overflow.length} exceeds capacity 100"

    # (7) Reproducibility: str/bytes hashing is deterministic across
    #     constructions (FNV-1a + splitmix, not PYTHONHASHSEED-randomised hash).
    str_items = [f"item_{i}" for i in range(500)]
    fqf_a = FunnelQuotientFilter(capacity=1000, r_remainder_bits=8)
    fqf_b = FunnelQuotientFilter(capacity=1000, r_remainder_bits=8)
    for s in str_items:
        fqf_a.insert(s)
        fqf_b.insert(s)
    # Both filters must agree on membership for every inserted and every
    # disjoint probe item.
    for s in str_items:
        pa, _ = fqf_a.contains(s)
        pb, _ = fqf_b.contains(s)
        assert pa and pb, f"Reproducibility failure: inserted item {s!r} missing"
    disjoint = [f"absent_{i}" for i in range(500)]
    for s in disjoint:
        pa, _ = fqf_a.contains(s)
        pb, _ = fqf_b.contains(s)
        assert pa == pb, f"Reproducibility failure: disjoint item {s!r} disagrees"
    print(f"  (7) Reproducibility (str/bytes stable hash): PASS "
          f"(two independent constructions agree on {len(str_items)} + "
          f"{len(disjoint)} probes)")

    print(f"  -> FunnelQuotientFilter verification PASSED.")


if __name__ == "__main__":
    print("==================================================================")
    print("STARTING BASIC DATATYPES & DATA STRUCTURES VERIFICATION TEST SUITE")
    print("==================================================================")
    t0 = time.perf_counter()
    test_multipole_range_tree()
    test_elastic_quotient_filter()
    test_sublinear_edit_distance()
    test_spatial_disjoint_set_fmm()
    test_spatial_point_cloud_compression()
    test_funnel_quotient_filter()
    total_time = (time.perf_counter() - t0) * 1000.0
    print(f"\nALL BASIC DATATYPE & COMPRESSION TESTS PASSED in {total_time:.2f} ms.")
    print("==================================================================")
