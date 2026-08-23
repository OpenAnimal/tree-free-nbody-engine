"""Regression tests for the Round-10 Wave-B fixes (R10-B1/B2/B3).

- B1: MeshfreeGraphLaplacian must accept 2D points (previously crashed
  with IndexError on the hardcoded 3D bucket key).
- B3: ConsistentMeshfreeLaplacian must reproduce nabla^2 exactly on
  quadratics (the previous graph-Laplacian operator presented as a
  continuous-Laplacian discretization was off by orders of magnitude);
  the graph operator is now documented as a graph operator and the
  consistent operator carries the continuous semantics.
- B2: ContinuousFockExchangeFMM far field must use the erf-screened
  monopole kernel (the previous bare 1/R far field diverged as cell_size
  shrank).
"""

import numpy as np


def _cml():
    from algorithm_theory.spectral_meshfree_laplacian import (
        ConsistentMeshfreeLaplacian)
    return ConsistentMeshfreeLaplacian


def test_graph_laplacian_accepts_2d_points():
    from algorithm_theory.spectral_meshfree_laplacian import (
        MeshfreeGraphLaplacian)
    pts = np.stack(np.meshgrid(np.linspace(-2, 2, 9),
                               np.linspace(-2, 2, 9),
                               indexing='ij'), axis=-1).reshape(-1, 2)
    lap = MeshfreeGraphLaplacian(pts, support_radius=1.5)
    v = np.random.RandomState(0).rand(len(pts))
    out = lap.matvec(v)
    assert out.shape == v.shape
    assert np.all(np.isfinite(out))


def test_graph_laplacian_rejects_bad_dims():
    from algorithm_theory.spectral_meshfree_laplacian import (
        MeshfreeGraphLaplacian)
    with np.testing.assert_raises(ValueError):
        MeshfreeGraphLaplacian(np.zeros((5, 4)), support_radius=1.0)


def test_consistent_laplacian_exact_on_quadratics_3d():
    CML = _cml()
    axes = [np.linspace(-3.0, 3.0, 12) for _ in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    lap = CML(pts, support_radius=1.0)
    v = np.sum(pts ** 2, axis=1)          # nabla^2 v = 6 everywhere
    err = np.max(np.abs(lap.matvec(v) - lap.kappa ** 2 * v - 6.0))
    assert err < 1e-6, f"quadratic consistency violated: {err:.3e}"


def test_consistent_laplacian_exact_on_quadratics_2d():
    CML = _cml()
    pts = np.stack(np.meshgrid(np.linspace(-2, 2, 9),
                               np.linspace(-2, 2, 9),
                               indexing='ij'), axis=-1).reshape(-1, 2)
    lap = CML(pts, support_radius=1.5)
    v = np.sum(pts ** 2, axis=1)          # nabla^2 v = 4 in 2D
    err = np.max(np.abs(lap.matvec(v) - 4.0))
    assert err < 1e-6, f"2D quadratic consistency violated: {err:.3e}"


def test_consistent_laplacian_smooth_field_accuracy():
    CML = _cml()
    axes = [np.linspace(-3.0, 3.0, 12) for _ in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    lap = CML(pts, support_radius=1.0)
    p = np.pi / 6.0
    s = np.prod(np.sin(p * pts), axis=1)
    target = -3 * p * p * s
    rel = np.linalg.norm(lap.matvec(s) - target) / np.linalg.norm(target)
    # Second-order behavior at this (coarse) h; the pre-fix graph operator
    # scored ~1.0 (indistinguishable from noise) on this probe.
    assert rel < 0.2, f"smooth-field accuracy degraded: {rel:.3e}"


def test_consistent_laplacian_scattered_points():
    CML = _cml()
    rng = np.random.RandomState(3)
    pts = rng.uniform(-3, 3, (400, 3))
    lap = CML(pts, support_radius=1.2)
    v = np.sum(pts ** 2, axis=1)
    err = np.max(np.abs(lap.matvec(v) - 6.0))
    assert err < 1e-3, f"scattered quadratic consistency: {err:.3e}"


def test_fock_exchange_far_field_screened_kernel():
    """R10-B2: with small cells (many far-field interactions) the CFMM must
    stay close to the direct erf-screened evaluation."""
    from algorithm_theory.quantum_fock_exchange_fmm import (
        ContinuousFockExchangeFMM)
    rng = np.random.RandomState(0)
    n = 60
    coords = rng.randn(n, 3) * 4.0
    exps = rng.uniform(0.5, 3.5, size=n)
    dens = rng.randn(n, n) * 0.05
    dens = (dens + dens.T) / 2.0
    ref = ContinuousFockExchangeFMM(coords, exps, cell_size=50.0)
    j_ref = ref.direct_coulomb_matrix_reference(dens)
    far = ContinuousFockExchangeFMM(coords, exps, cell_size=0.3)
    j_far = far.compute_coulomb_matrix_cfmm(dens)
    rel = np.linalg.norm(j_far - j_ref) / np.linalg.norm(j_ref)
    # Pre-fix: rel ~ 0.7 at cell_size=0.3 (bare 1/R far field).
    assert rel < 0.15, f"far-field screening degraded: {rel:.3e}"


if __name__ == "__main__":
    test_graph_laplacian_accepts_2d_points()
    test_graph_laplacian_rejects_bad_dims()
    test_consistent_laplacian_exact_on_quadratics_3d()
    test_consistent_laplacian_exact_on_quadratics_2d()
    test_consistent_laplacian_smooth_field_accuracy()
    test_consistent_laplacian_scattered_points()
    test_fock_exchange_far_field_screened_kernel()
    print("ALL ROUND-10 FIX REGRESSION TESTS PASSED")
