"""Round-10 Wave B probe: algorithm_theory modules vs independent references.

Each check builds an independent reference and compares values, not just shapes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: algorithm_theory deep verification")
print("=" * 70)

rng = np.random.RandomState(42)

# ============================================================
# 1. Koopman: eigenvalue recovery from linear dynamics
# ============================================================
print("\n[1] Koopman spectral operator")
from algorithm_theory.koopman_spectral_operator import ContinuousKoopmanOperator

A = np.array([[0.9, 0.1], [0.0, 0.8]])
N_steps = 300
traj = np.zeros((N_steps, 2))
traj[0] = rng.randn(2)
for s in range(1, N_steps):
    traj[s] = A @ traj[s - 1] + 0.01 * rng.randn(2)

koop = ContinuousKoopmanOperator(poly_degree=2, n_rbf_centers=16)
koop.fit(traj)

eig_A = np.sort(np.linalg.eigvals(A).real)
eig_K_all = koop.eigenvalues.real
eig_K_nontrivial = np.sort(eig_K_all[np.abs(eig_K_all - 1.0) > 0.01])
check("Koopman: nontrivial eigenvalues near A's eigenvalues",
      len(eig_K_nontrivial) >= 2 and
      np.any(np.abs(eig_K_nontrivial - 0.9) < 0.15) and
      np.any(np.abs(eig_K_nontrivial - 0.8) < 0.15),
      f"eig_A={eig_A}, eig_K_nontrivial={eig_K_nontrivial[:5]}")

x_test = traj[50].copy()
try:
    pred = koop.predict_future_trajectory(x_test, num_future_steps=1)
    expected = A @ x_test
    check("Koopman: 1-step prediction matches linear dynamics",
          np.allclose(pred[0], expected, atol=0.2),
          f"pred={pred[0]}, expected={expected}")
except Exception as e:
    check("Koopman: predict runs", False, str(e))

# ============================================================
# 2. LEnKF: analysis step vs direct Kalman
# ============================================================
print("\n[2] Localized Ensemble Kalman Filter")
from algorithm_theory.localized_ensemble_kalman_fmm import LocalizedEnsembleKalmanFilter

n_ens = 100
n_dim = 5
sigma_prior = 1.0
sigma_obs = 0.3
truth = np.array([2.0] * n_dim)
obs = truth + sigma_obs * rng.randn(n_dim)

ens = truth[:, None] + sigma_prior * rng.randn(n_dim, n_ens)
state_coords = np.linspace(0, 1, n_dim).reshape(-1, 1)

P = np.var(ens, axis=1)
analysis_direct = np.mean(ens, axis=1) + P / (P + sigma_obs**2) * (obs - np.mean(ens, axis=1))

try:
    lenkf = LocalizedEnsembleKalmanFilter(
        state_coords=state_coords,
        localization_radius=10.0,
        obs_noise_variance=sigma_obs**2,
    )
    post_ens, post_mean = lenkf.assimilate_observations(
        prior_ensemble=ens.copy(),
        obs_indices=np.arange(n_dim),
        obs_values=obs,
    )
    check("LEnKF: no NaN in posterior", not np.any(np.isnan(post_ens)),
          f"nan count={np.sum(np.isnan(post_ens))}")
    check("LEnKF: analysis mean close to direct Kalman",
          np.allclose(post_mean, analysis_direct, atol=0.3),
          f"lenkf={post_mean}, direct={analysis_direct}")
    post_var = np.var(post_ens, axis=1)
    check("LEnKF: analysis reduces spread",
          np.all(post_var < P),
          f"posterior var={post_var}, prior var={P}")
except Exception as e:
    check("LEnKF: analysis runs without error", False, str(e))

# ============================================================
# 3. Matrix-free Gaussian Process: mean vs dense (variance is sparse approx)
# ============================================================
print("\n[3] Matrix-free Gaussian Process")
from algorithm_theory.matrix_free_gaussian_process import MatrixFreeGaussianProcess

N_train = 50
N_test = 20
pts_train = rng.uniform(0, 1, size=(N_train, 1))
pts_test = rng.uniform(0, 1, size=(N_test, 1))
y_train = np.sin(2 * np.pi * pts_train[:, 0]) + 0.1 * rng.randn(N_train)

ls, sf, sn = 0.3, 1.0, 0.1
gp = MatrixFreeGaussianProcess(lengthscale=ls, signal_variance=sf, noise_variance=sn)
gp.fit(pts_train, y_train)
pred_mean, pred_var = gp.predict(pts_test)

from scipy.spatial.distance import cdist
def rbf_kernel(X1, X2, lengthscale, signal_var):
    D2 = cdist(X1, X2, 'sqeuclidean')
    return signal_var**2 * np.exp(-0.5 * D2 / lengthscale**2)

K = rbf_kernel(pts_train, pts_train, ls, sf) + sn**2 * np.eye(N_train)
K_star = rbf_kernel(pts_test, pts_train, ls, sf)
alpha = np.linalg.solve(K, y_train)
pred_direct = K_star @ alpha

rel_mean = np.linalg.norm(pred_mean - pred_direct) / max(1e-30, np.linalg.norm(pred_direct))
check("GP: posterior mean vs dense solve", rel_mean < 0.1,
      f"rel-L2={rel_mean:.2e}")
check("GP: variance non-negative", np.all(pred_var >= -1e-10),
      f"min var={np.min(pred_var):.2e}")
# Variance is a sparse approximation (upper bound) — just check it's reasonable
check("GP: variance <= signal_variance", np.all(pred_var <= sf**2 + 1e-6),
      f"max var={np.max(pred_var):.4f}, sf^2={sf**2}")

# ============================================================
# 4. NUFFT: type-1 and type-2 vs direct DFT (numpy fft order)
# ============================================================
print("\n[4] Non-uniform FFT")
from algorithm_theory.non_uniform_fourier_hash import NonUniformFourierHash

N = 64
n_pts = 50
x = rng.uniform(0, 2 * np.pi, n_pts)
f = rng.randn(n_pts)

nufft = NonUniformFourierHash(grid_shape=N, dim=1, oversampling_factor=2.0, window_width=8)

try:
    result = nufft.type1_nonuniform_to_uniform(x, f)
    k_fft = np.concatenate([np.arange(N // 2), np.arange(-N // 2, 0)])
    direct = np.array([np.sum(f * np.exp(-1j * k_fft[m] * x)) for m in range(N)])
    rel = np.linalg.norm(result - direct) / max(1e-30, np.linalg.norm(direct))
    check("NUFFT Type-1 vs direct DFT", rel < 2e-4,
          f"rel-L2={rel:.2e}")
except Exception as e:
    check("NUFFT Type-1 runs", False, str(e))

try:
    F = rng.randn(N) + 1j * rng.randn(N)
    result2 = nufft.type2_uniform_to_nonuniform(F, x)
    direct2 = np.array([np.sum(F * np.exp(1j * k_fft * x[j])) for j in range(n_pts)])
    rel2 = np.linalg.norm(result2 - direct2) / max(1e-30, np.linalg.norm(direct2))
    check("NUFFT Type-2 vs direct DFT", rel2 < 1e-4,
          f"rel-L2={rel2:.2e}")
except Exception as e:
    check("NUFFT Type-2 runs", False, str(e))

# ============================================================
# 5. Optimal Transport: marginal constraints (with large cutoff)
# ============================================================
print("\n[5] Fast Entropic Optimal Transport")
from algorithm_theory.optimal_transport_fmm import FastEntropicOptimalTransport
from scipy.spatial.distance import cdist as cdist_scipy

n_src = 10
n_tgt = 10
src = rng.uniform(0, 1, size=(n_src, 2))
tgt = rng.uniform(0, 1, size=(n_tgt, 2))
a = np.ones(n_src) / n_src
b = np.ones(n_tgt) / n_tgt

try:
    ot = FastEntropicOptimalTransport(regularization_gamma=1.0, max_iterations=2000,
                                       tolerance=1e-8, cutoff_sigma_multiplier=10.0)
    u, v, cost, iters = ot.solve_transport_plan(src, a, tgt, b)
    # Reconstruct plan: T = diag(u) K diag(v), K = exp(-r^2/(2*gamma^2))
    C = cdist_scipy(src, tgt) ** 2
    K_mat = np.exp(-C / (2 * 1.0**2))
    T = np.diag(u) @ K_mat @ np.diag(v)
    check("OT: plan shape", T.shape == (n_src, n_tgt),
          f"got {T.shape}")
    check("OT: source marginals", np.allclose(T.sum(axis=1), a, atol=1e-6),
          f"row sums={T.sum(axis=1)}")
    check("OT: target marginals", np.allclose(T.sum(axis=0), b, atol=1e-6),
          f"col sums={T.sum(axis=0)}")
    check("OT: plan non-negative", np.all(T >= -1e-10),
          f"min={np.min(T):.2e}")
except Exception as e:
    check("OT: solve runs", False, str(e))

# ============================================================
# 6. Spectral meshfree Laplacian: 3D only (2D crashes — R10-B1)
# ============================================================
print("\n[6] Spectral meshfree Laplacian (3D)")
from algorithm_theory.spectral_meshfree_laplacian import SpectralMeshfreeLaplacian

# 3D grid: f(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)
# Laplacian = -3*pi^2 * f
N_side = 12
h = 1.0 / (N_side - 1)
xs = np.linspace(0, 1, N_side)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
pts3d = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
f3d = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel()) * np.sin(np.pi * Z.ravel())
f_lap_analytic = -3 * np.pi**2 * f3d

try:
    sml = SpectralMeshfreeLaplacian(points=pts3d, support_radius=3 * h)
    f_sml = sml.matvec(f3d)
    interior = (X.ravel() > 0.15) & (X.ravel() < 0.85) & \
               (Y.ravel() > 0.15) & (Y.ravel() < 0.85) & \
               (Z.ravel() > 0.15) & (Z.ravel() < 0.85)
    rel_sml = np.linalg.norm(f_sml[interior] - f_lap_analytic[interior]) / np.linalg.norm(f_lap_analytic[interior])
    check("Spectral Laplacian vs analytic (3D)", rel_sml < 0.2,
          f"rel-L2={rel_sml:.2e}")
except Exception as e:
    check("Spectral Laplacian runs (3D)", False, str(e))

# 2D should crash (R10-B1 bug)
try:
    pts2d_bug = np.column_stack([X.ravel()[:10], Y.ravel()[:10]])
    sml_2d = SpectralMeshfreeLaplacian(points=pts2d_bug, support_radius=0.1)
    check("R10-B1: 2D points crash (bug confirmed)", False,
          "2D points did NOT crash — bug may be fixed")
except IndexError:
    check("R10-B1: 2D points crash (bug confirmed)", True,
          "IndexError as expected — hardcoded 3D indexing")
except Exception as e:
    check("R10-B1: 2D points crash (bug confirmed)", False,
          f"Unexpected error: {e}")

# ============================================================
# 7. Quantum Fock: CFMM vs direct (cell_size=0.5 = all direct)
# ============================================================
print("\n[7] Quantum Fock Exchange FMM")
from algorithm_theory.quantum_fock_exchange_fmm import ContinuousFockExchangeFMM

N_q = 30
basis_coords = rng.uniform(0, 1, size=(N_q, 3))
basis_exponents = rng.uniform(0.5, 2.0, size=N_q)
density = rng.uniform(0.1, 1.0, size=(N_q, N_q))
density = 0.5 * (density + density.T)

# With cell_size=0.5, all interactions are direct -> exact match
try:
    fock_exact = ContinuousFockExchangeFMM(
        basis_coords=basis_coords, basis_exponents=basis_exponents, cell_size=0.5,
    )
    mat_exact = fock_exact.compute_coulomb_matrix_cfmm(density)
    mat_direct = fock_exact.direct_coulomb_matrix_reference(density)
    rel_exact = np.linalg.norm(mat_exact - mat_direct) / max(1e-30, np.linalg.norm(mat_direct))
    check("Fock: cell_size=0.5 (all direct) matches reference", rel_exact < 1e-10,
          f"rel-L2={rel_exact:.2e}")
except Exception as e:
    check("Fock: cell_size=0.5 runs", False, str(e))

# With smaller cell_size, far-field expansion is used -> R10-B2 bug
try:
    fock_far = ContinuousFockExchangeFMM(
        basis_coords=basis_coords, basis_exponents=basis_exponents, cell_size=0.2,
    )
    mat_far = fock_far.compute_coulomb_matrix_cfmm(density)
    rel_far = np.linalg.norm(mat_far - mat_direct) / max(1e-30, np.linalg.norm(mat_direct))
    check("Fock: cell_size=0.2 (far-field) matches reference", rel_far < 0.05,
          f"rel-L2={rel_far:.2e}  [R10-B2: far-field expansion bug if FAIL]")
except Exception as e:
    check("Fock: cell_size=0.2 runs", False, str(e))

# ============================================================
# 8. Sublinear edit distance: vs Wagner-Fischer
# ============================================================
print("\n[8] Sublinear edit distance")
from algorithm_theory.sublinear_edit_distance import SublinearEditDistance, exact_wagner_fischer_edit_distance

pairs = [
    ("kitten", "sitting"),
    ("sunday", "saturday"),
    ("abc", "abc"),
    ("", "abc"),
    ("abc", ""),
    ("a", "b"),
]
all_ok = True
for s1, s2 in pairs:
    ed_exact = exact_wagner_fischer_edit_distance(s1, s2)
    try:
        sed = SublinearEditDistance(q=3, band_width=16)
        result = sed.approximate_edit_distance(s1, s2)
        if isinstance(result, dict):
            ed_sub = result.get('approx_distance', result.get('distance', 0))
        else:
            ed_sub = result
        if not (ed_sub >= ed_exact and ed_sub <= ed_exact + 3):
            all_ok = False
            print(f"    FAIL: '{s1}' vs '{s2}': exact={ed_exact}, sublinear={ed_sub}")
    except Exception as e:
        all_ok = False
        print(f"    ERROR: '{s1}' vs '{s2}': {e}")
        break
check("edit distance: sublinear within bounds of exact", all_ok)

# ============================================================
# 9. Geodesic SSSP: vs Dijkstra baseline
# ============================================================
print("\n[9] Geodesic SSSP")
from algorithm_theory.tree_free_geodesic_fmm import FrontierClusteredSSSP, DijkstraBaselineSSSP

N_g = 100
adj = [[] for _ in range(N_g)]
for i in range(N_g):
    for j in range(i + 1, min(i + 5, N_g)):
        w = rng.uniform(0.1, 2.0)
        adj[i].append((j, w))
        adj[j].append((i, w))

try:
    dijk = DijkstraBaselineSSSP(num_nodes=N_g, adj_list=adj)
    dist_dijk = dijk.compute(source=0)

    frontier = FrontierClusteredSSSP(num_nodes=N_g, adj_list=adj)
    dist_frontier = frontier.compute(source=0)

    check("SSSP: frontier matches Dijkstra", np.allclose(dist_dijk, dist_frontier),
          f"max diff={np.max(np.abs(dist_dijk - dist_frontier)):.2e}")
except Exception as e:
    check("SSSP runs", False, str(e))

# ============================================================
# 10. Capacitance BEM: vs analytic sphere
# ============================================================
print("\n[10] Capacitance BEM vs analytic sphere")
from algorithm_theory.capacitance_boundary_bem import CapacitanceBoundaryBEM

R = 2.0
n_el = 300
phi = np.pi * (3.0 - np.sqrt(5.0))
indices = np.arange(n_el)
y_sphere = 1 - 2 * (indices + 0.5) / n_el
r_sphere = np.sqrt(1 - y_sphere**2)
theta_sphere = phi * indices
surface_pts = np.column_stack([
    R * r_sphere * np.cos(theta_sphere),
    R * r_sphere * np.sin(theta_sphere),
    R * y_sphere,
])
areas = np.full(n_el, 4 * np.pi * R**2 / n_el)

try:
    bem = CapacitanceBoundaryBEM(
        surface_points=surface_pts,
        surface_areas=areas,
        multipole_cell_size=0.5,
    )
    C = bem.compute_capacitance()
    C_analytic = 4 * np.pi * R
    rel_C = abs(C - C_analytic) / C_analytic
    check("BEM: capacitance vs analytic 4*pi*R", rel_C < 0.1,
          f"num={C:.4f}, analytic={C_analytic:.4f}, rel={rel_C:.2e}")
except Exception as e:
    check("BEM runs", False, str(e))

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
