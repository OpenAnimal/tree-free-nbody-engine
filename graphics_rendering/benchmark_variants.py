"""Standardized variant benchmark: standard vs +elastichash (+quantized where applicable)."""
import numpy as np, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark
from core.validation import cross_validate

def run_ao_variants(n_occluders=4000, n_queries=1000):
    from graphics_rendering.volumetric_fmm_ao import VolumetricFMMAmbientOcclusion
    rng = np.random.default_rng(3)
    p = rng.uniform(-5, 5, (n_occluders, 3)).astype(np.float32)
    r = rng.uniform(0.05, 0.3, n_occluders).astype(np.float32)
    o = rng.uniform(0.5, 1.0, n_occluders).astype(np.float32)
    q = rng.uniform(-6, 6, (n_queries, 3)).astype(np.float32)
    vao = VolumetricFMMAmbientOcclusion(cell_size=1.0)
    vao.insert_occluders(p, r, o)
    bench = VariantBenchmark("Volumetric AO (3D inverse-square kernel; no FMM applies — core FMM is 2D log kernel)")
    bench.add("standard (exact per-particle)", lambda: vao.evaluate_ao_exact(q), note="O(Q*N) reference")
    bench.add("+elastichash near/far", lambda: vao.evaluate_ao_field_near_far(q), accuracy_vs="standard (exact per-particle)", note="order-0 far field")
    bench.add("+quantized (all-cluster)", lambda: vao._evaluate_ao_cpu(
        q, np.stack([vao.macro_clusters[k]["center"] for k in sorted(vao.macro_clusters)]),
        np.array([vao.macro_clusters[k]["mass"] for k in sorted(vao.macro_clusters)], np.float32),
        np.array([vao.macro_clusters[k]["eff_radius"] for k in sorted(vao.macro_clusters)], np.float32),
        4096), accuracy_vs="standard (exact per-particle)", note="cluster-quantized far field")
    return bench.run()

if __name__ == "__main__":
    run_ao_variants()
