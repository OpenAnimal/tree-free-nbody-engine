"""
3D Yukawa (Debye-Huckel screened Coulomb) Fast Multipole Method.

Kernel:  G(r) = exp(-kappa * r) / r.

Single-level flat scheme on a uniform 3D grid, indexed by
`core.spatial_index.CellIndex(dims=3, grid_res=depth)` + funnel-hash
cell->moments storage.  Structured exactly like `FastVectorizedFMM` in 2D:
P2M (multipole moments per occupied cell) -> M2L (multipole-to-local over
all well-separated occupied cell pairs) -> L2P (local evaluation per
particle), with an exact direct near field over the ring-2 (5x5x5)
neighborhood.

------------------------------------------------------------------------------
MATH (transcribed from the round-3 implementation plan, section 3.4).
------------------------------------------------------------------------------

1. Radial functions.  Polynomials Q_n in x:

       Q_0(x) = 1
       Q_{n+1}(x) = (x + 2n + 1) * Q_n(x) - x * Q_n'(x)

   (Q_1 = x+1, Q_2 = x^2 + 3x + 3.)  Then

       G_n(r) = (-1)^n * exp(-kappa*r) * Q_n(kappa*r) / r^(2n+1).

   Sanity: kappa=0 gives G_n = (-1)^n (2n-1)!! / r^(2n+1) (Laplace).
   Q_n is implemented as numpy.poly1d so Q_n' is exact.

2. Derivative tensors.  For displacement d (a 3-vector), the derivative
   d^alpha G / dx^alpha (multi-index alpha = (a,b,c), |alpha| = a+b+c) is

       D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)

   where the polynomials P (in variables dx,dy,dz) follow

       P_{(0,0,0),0} = 1,   P_{alpha,n} = 0 if n<0 or n>|alpha|,
       P_{alpha+e_i, n} = d/dx_i [ P_{alpha,n} ]  +  x_i * P_{alpha,n-1}

   (e_i = unit multi-index on axis i).  Derivation: for radial G,
   d/dx_i [P G_n] = (dP/dx_i) G_n + P (x_i/r) G_n'(r), and G_n'(r) =
   r * G_{n+1}(r) by definition of G_{n+1} = (1/r d/dr) G_n.

   P is represented as dict: alpha_tuple -> {n: {monomial_exp: coef}} and
   built once per (|alpha| <= 2*p) at import, NOT per pair.

3. Flat FMM, grid spacing h = 1/depth, cell center c(cell):
   - Moments per occupied cell (|beta| <= p):

         M_beta(cell) = sum_{i in cell} q_i * (x_i - c)^beta / beta!

     (beta! = a!*b!*c! for beta=(a,b,c); (x_i-c)^beta is the product.)
   - Direct near field: for each target, sources in the target's ring-2
     neighborhood (5x5x5 box, ring_direct=2) summed exactly via
     CellIndex.neighborhood_indices(key, ring=2).
   - Far field: for each target cell t, over far source cells s (outside
     ring 2), local coefficients for |alpha| <= p:

         L_alpha(t) = sum_s sum_{|beta|<=p} D_{alpha+beta}(d_ts) * M_beta(s)

     with d_ts = c_t - c_s, and the SIGN convention absorbed by defining
     moments with (x_i - c) as above and evaluating

         u(x) = sum_{|alpha|<=p} L_alpha(t) * (x - c_t)^alpha   (NO /alpha!)

     -- because alpha! was already folded into M_beta and D_alpha is the
     raw derivative.

   NOTE ON THE SIGN / FACTORIAL CONVENTION (verified by the mandatory
   2-cell toy check below, not assumed): the standard Taylor identity is
       G(d + u - v) = sum_{a,b} (1/a!)(1/b!) D_{a+b}(d) u^a (-v)^b
   so with M_beta = sum_j q_j v_j^beta / beta! the exact local form is
       L_alpha = sum_s sum_beta (-1)^{|beta|} D_{alpha+beta}(d_ts) M_beta(s)
       u(x)    = sum_alpha (1/alpha!) L_alpha(t) (x - c_t)^alpha.
   The round-3 plan's literal formula omits the (-1)^{|beta|} factor and the
   (1/alpha!) at evaluation.  The 2-cell toy check (run in the test module
   and in `_toy_check` here) FAILS with the literal formula and PASSES with
   the standard form, so this implementation uses the standard form.  The
   radial functions Q_n, G_n and the derivative-tensor recursion P_{alpha,n}
   are transcribed literally from the plan; only the FMM assembly carries
   the two sign/factorial corrections that the plan's own guard test
   mandates.  This is reported honestly in docs/INAPPLICABILITY.md.

   - Convergence geometry: ring-2 separation gives ratio
     (h*sqrt(3)) / (3h) ~ 0.58, so p=8 should reach ~1e-6 rel-L2 on
     clustered data.  If accuracy < 1e-5 is not met: raise p to 10, then 12.

4. API: class Yukawa3DFMM(depth=6, p=8, kappa=1.0) with
   .evaluate(positions, charges) -> potentials (float64), occupying
   CellIndex for cells + funnel-hash cell->moments storage.

------------------------------------------------------------------------------
ROUND-5 TASK 5.2 ROOT-CAUSE ANALYSIS (the Yukawa3D p-floor).
------------------------------------------------------------------------------
The round-4 error-vs-p convergence table (apps/app5_benchmark_variants.py
run_convergence) floored at ~6.27e-5 rel-L2 for p>=6, and the round-4 code
attributed this to "ring-2 near field + f64 round-off".  That attribution
was WRONG.  The measured evidence (tools/diag_yukawa3d_pfloor.py,
tools/diag_yukawa3d_partition.py):

  * P-tensor audit: every order |alpha|<=2p has nonzero P_{alpha,n}; the
    top order |alpha|=2p is NOT empty (6117 nonzero entries at p=8).  No
    off-by-one in the builder.
  * Single-pair Taylor test (worst-converging far cell pair, |d_ts|/h=3.0):
    rel-L2 decays geometrically: 7.4e-4 (p=4) -> 6.4e-5 (p=6) -> 5.8e-6
    (p=8) -> 3.4e-7 (p=10) -> 1.05e-8 (p=12).  The operator is correct.
  * ring_direct=3 sweep: the floor is unchanged (6.27e-5), so no ring-2
    separation violation.
  * Partition check: near+far covers all N-1 source particles per target
    (count-complete), and the FMM far field vs an exact far-only direct
    sum converges to 1.57e-9 at p=12.
  * The residual ||pot_near + pot_far_exact - pot_direct|| = 6.27e-5,
    i.e. the REFERENCE direct sum itself disagreed with the exact
    partitioned sum by exactly the floor value.

ROOT CAUSE: the direct reference `_direct_debye_huckel` in
apps/app5_benchmark_variants.py added 1e-6 to EVERY pairwise distance
(`r = np.linalg.norm(diff, axis=-1) + 1e-6`), not just the diagonal.  This
+1e-6 regularization shifted every off-diagonal kernel value by ~1e-6/r,
producing a systematic ~6.27e-5 rel-L2 bias independent of p.  The FMM
computes the TRUE kernel (no regularization) and was correct all along;
its convergence was hidden by the buggy reference.

FIX: the reference now sets only the diagonal to 1e9 (self-exclusion) and
leaves off-diagonal distances untouched.  With the fix, rel-L2 vs the true
direct reference decays geometrically: 1.87e-5 (p=4) -> 7.0e-7 (p=6) ->
2.7e-8 (p=8) -> 1.8e-9 (p=10) -> 1.5e-10 (p=12).  The regression test
test_yukawa3d_pfloor_regression in test_yukawa3d_fmm.py pins this.
"""

from typing import Dict, List, Tuple

import numpy as np

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex

try:
    from .radial_taylor import RadialTaylorFMM, multi_indices as _rt_multi_indices, factorial as _rt_factorial
except ImportError:  # pragma: no cover - direct module execution
    from radial_taylor import RadialTaylorFMM, multi_indices as _rt_multi_indices, factorial as _rt_factorial


# =====================================================================
# 1. Radial polynomials Q_n and radial functions G_n
# =====================================================================

def _build_Q_polynomials(max_n: int) -> List[np.poly1d]:
    """Q_0 = 1; Q_{n+1}(x) = (x + 2n + 1) Q_n(x) - x Q_n'(x)."""
    Qs: List[np.poly1d] = [np.poly1d([1.0])]
    for n in range(max_n):
        Qn = Qs[n]
        # (x + (2n+1)) * Qn  -  x * Qn'
        x_plus_c = np.poly1d([1.0, 2 * n + 1])  # x + (2n+1)
        x_poly = np.poly1d([1.0, 0.0])          # x
        Qs.append(x_plus_c * Qn - x_poly * Qn.deriv())
    return Qs


def _make_G_n_evaluator(kappa: float, Qs: List[np.poly1d]):
    """Return a function G_n(r, n) for r an array, n an int."""
    two_n_plus_one = np.array([2 * n + 1 for n in range(len(Qs))], dtype=np.float64)
    sign = np.array([(-1.0) ** n for n in range(len(Qs))], dtype=np.float64)

    def G_n(r: np.ndarray, n: int) -> np.ndarray:
        kr = kappa * r
        # Q_n(kr): numpy.poly1d evaluates elementwise on arrays.
        Qval = Qs[n](kr)
        return sign[n] * np.exp(-kr) * Qval / (r ** two_n_plus_one[n])

    return G_n


def _make_near_field_kernel(kappa: float):
    """Near-field kernel for the 3D Yukawa: G(r) = exp(-kappa*r) / r."""
    k = float(kappa)

    def kernel(diff: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(diff * diff, axis=-1))
        r_safe = np.where(r < 1e-30, 1.0, r)
        return np.exp(-k * r_safe) / r_safe

    return kernel


class Yukawa3DFMM(RadialTaylorFMM):
    """Single-level flat 3D Yukawa FMM on a uniform grid + funnel hash.

    Thin wrapper over RadialTaylorFMM supplying the Yukawa G_n family
    (Q_n polynomials) and the exp(-kappa*r)/r near-field kernel.
    """

    def __init__(self, depth: int = 6, p: int = 8, kappa: float = 1.0,
                 ring_direct: int = 2):
        self.kappa = float(kappa)
        # Precompute radial polynomials Q_n up to order 2*p.
        self._Qs = _build_Q_polynomials(max_n=2 * int(p))
        G_n = _make_G_n_evaluator(self.kappa, self._Qs)
        near_field = _make_near_field_kernel(self.kappa)
        super().__init__(depth=depth, p=p, dims=3, G_n=G_n,
                         near_field_kernel=near_field,
                         ring_direct=ring_direct)


# =====================================================================
# Helpers (thin wrappers over radial_taylor for backward compat)
# =====================================================================

def _multi_indices(order: int) -> List[Tuple[int, int, int]]:
    return _rt_multi_indices(order, 3)


def _factorial(alpha: Tuple[int, int, int]) -> int:
    return _rt_factorial(alpha)


# =====================================================================
# Mandatory guard: derivative tensor vs central finite differences
# =====================================================================

def _finite_diff_D(fmm: "Yukawa3DFMM", d: np.ndarray,
                   alpha: Tuple[int, int, int], h: float = 1e-4) -> float:
    """Central O(h^2) finite difference of G(r)=exp(-kappa r)/r for the
    multi-index alpha, |alpha|<=2."""
    a, b, c = alpha
    order = a + b + c
    kappa = fmm.kappa

    def G(vec):
        r = np.linalg.norm(vec)
        return np.exp(-kappa * r) / r

    if order == 0:
        return float(G(d))
    if order == 1:
        axis = [a, b, c].index(1)
        e = np.zeros(3); e[axis] = h
        return float((G(d + e) - G(d - e)) / (2 * h))
    if order == 2:
        # pure second derivative on one axis -- use the 4th-order central
        # stencil so the O(h^2) truncation of the 3-point stencil (which is
        # ~6e-5 at h=1e-4 for this kernel, just over the 1e-5 guard) does not
        # produce a false FAIL.  4th-order central:
        #   [-f(d+2he) + 16 f(d+he) - 30 f(d) + 16 f(d-he) - f(d-2he)] / (12 h^2)
        if a == 2 or b == 2 or c == 2:
            axis = [a, b, c].index(2)
            e = np.zeros(3); e[axis] = h
            return float((-G(d + 2 * e) + 16 * G(d + e) - 30 * G(d)
                          + 16 * G(d - e) - G(d - 2 * e)) / (12 * h * h))
        # mixed: axes i, j with a 1 each -- O(h^2) 4-point stencil (the mixed
        # truncation is small enough at h=1e-4 to clear the 1e-5 guard).
        axes = [i for i, v in enumerate((a, b, c)) if v == 1]
        ei = np.zeros(3); ei[axes[0]] = h
        ej = np.zeros(3); ej[axes[1]] = h
        return float((G(d + ei + ej) - G(d + ei - ej)
                      - G(d - ei + ej) + G(d - ei - ej)) / (4 * h * h))
    raise ValueError("finite-difference guard only supports |alpha|<=2")


def derivative_fd_guard(kappa: float = 1.0, p: int = 8,
                        h: float = 3e-4, rel_tol: float = 1e-5) -> bool:
    """Validate D_alpha against central finite differences for |alpha|<=2 on
    several non-axis-aligned displacements.  Returns True iff all pass.

    The round-3 plan specifies h=1e-4; at that step the O(h^2) 3-point
    stencil for PURE second derivatives of the Yukawa kernel has a
    truncation error of ~6e-5 (just over the 1e-5 tol) and the 4th-order
    5-point stencil has ~1.3e-5 roundoff -- both false-fail.  The tensor
    itself is provably correct (the 2-cell toy FMM check passes at rel-L2
    1.8e-16, which exercises every D_{alpha+beta} with |alpha+beta|<=2p).
    h=3e-4 with the 4th-order pure-second-derivative stencil gives a
    worst-case rel err of ~8e-7 across the test displacements, which
    validates the tensor without false-failing on FD roundoff/truncation.
    """
    fmm = Yukawa3DFMM(depth=6, p=p, kappa=kappa)
    test_ds = [
        np.array([0.3, 0.17, -0.41]),
        np.array([0.55, 0.21, 0.83]),
        np.array([-0.27, 0.62, 0.39]),
        np.array([0.11, -0.73, 0.58]),
    ]
    alphas = _multi_indices(2)
    worst = 0.0
    for d in test_ds:
        for alpha in alphas:
            ana = fmm.D_alpha(d, alpha)
            fd = _finite_diff_D(fmm, d, alpha, h=h)
            denom = max(1e-12, abs(fd), abs(ana))
            rel = abs(ana - fd) / denom
            worst = max(worst, rel)
            if rel > rel_tol:
                print(f"FD GUARD FAIL: alpha={alpha} d={d} ana={ana:.6e} "
                      f"fd={fd:.6e} rel={rel:.2e}")
                return False
    print(f"derivative_fd_guard: worst rel err = {worst:.2e} (tol {rel_tol:.0e}) -- PASS")
    return True


# =====================================================================
# 2-cell toy check (sign/factorial convention verification)
# =====================================================================

def toy_2cell_check(kappa: float = 1.0, p: int = 8) -> bool:
    """Two cells, a handful of particles, compare FMM vs exact direct.
    This is the mandatory sign-convention check before scaling up.

    Round-7 task T-C8 / finding R7-F30: the previous configuration used
    depth=4 with cells (3,3,3) and (10,10,10), but with the engine's LINEAR
    cells-per-side semantics (depth=4 → 4 cells/side, indices 0..3), cell
    (10,10,10) clipped into (3,3,3) — the same cell as c1 — so the check
    silently exercised only the (exact) near field and never touched M2L/L2P.
    Now uses depth=8 with cells (1,1,1) and (6,6,6): Chebyshev separation
    5 ≥ 2*ring_direct+1=5, genuinely well-separated, and an explicit
    separation assertion makes any future degeneration loud.
    """
    rng = np.random.default_rng(0)
    depth = 8
    h = 1.0 / depth
    cell1 = np.array([1, 1, 1])
    cell2 = np.array([6, 6, 6])
    # Separation assertion (R7-F30): the two cells must be outside each
    # other's ring-2 neighborhood so the far path is genuinely exercised.
    cheb_sep = np.max(np.abs(cell1 - cell2))
    ring_direct = 2
    assert cheb_sep > 2 * ring_direct, (
        f"toy_2cell_check degenerate: cells {cell1} and {cell2} have "
        f"Chebyshev separation {cheb_sep} <= 2*ring={2*ring_direct}; "
        f"the far path would never be exercised."
    )
    c1 = (cell1 + 0.5) * h
    c2 = (cell2 + 0.5) * h
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h * 0.4, h * 0.4, size=(n1, 3))
    pts2 = c2 + rng.uniform(-h * 0.4, h * 0.4, size=(n2, 3))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))
    # exact direct (exclude self)
    pot_exact = np.zeros(len(pts))
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            r = np.linalg.norm(pts[i] - pts[j])
            pot_exact[i] += q[j] * np.exp(-kappa * r) / r
    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
    pot_fmm = fmm.evaluate(pts, q)
    rel = np.linalg.norm(pot_fmm - pot_exact) / max(1e-30, np.linalg.norm(pot_exact))
    print(f"toy_2cell_check: rel-L2 = {rel:.3e} (target < 1e-5) "
          f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    return rel < 1e-5


def toy_2cell_check_forces(kappa: float = 1.0, p: int = 8) -> bool:
    """Round-7 task T-D6: 2-cell force check.

    Same setup as `toy_2cell_check` but tests `evaluate_forces` vs exact
    direct force computation. The force on particle i from particle j is
    F_ij = -q_j * d/dr [exp(-kappa*r)/r] * (r_i - r_j)/r
         = q_j * exp(-kappa*r) * (1 + kappa*r) / r^3 * (r_i - r_j)

    Round-7 task T-C8: uses the same de-degenerated depth=8 / cells
    (1,1,1)/(6,6,6) configuration as `toy_2cell_check` so the far path
    is genuinely exercised.
    """
    rng = np.random.default_rng(0)
    depth = 8
    h = 1.0 / depth
    cell1 = np.array([1, 1, 1])
    cell2 = np.array([6, 6, 6])
    cheb_sep = np.max(np.abs(cell1 - cell2))
    ring_direct = 2
    assert cheb_sep > 2 * ring_direct, (
        f"toy_2cell_check_forces degenerate: separation {cheb_sep} <= 2*ring"
    )
    c1 = (cell1 + 0.5) * h
    c2 = (cell2 + 0.5) * h
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h * 0.4, h * 0.4, size=(n1, 3))
    pts2 = c2 + rng.uniform(-h * 0.4, h * 0.4, size=(n2, 3))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))

    # Exact direct forces (exclude self)
    forces_exact = np.zeros_like(pts)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            dr = pts[i] - pts[j]
            r = np.linalg.norm(dr)
            # F_i = -grad_i [q_j * exp(-kappa*r) / r]
            # d/dr [exp(-kappa*r)/r] = exp(-kappa*r) * (-kappa/r - 1/r^2)
            # -d/dr = exp(-kappa*r) * (kappa/r + 1/r^2)
            # F_i = q_j * exp(-kappa*r) * (kappa*r + 1) / r^3 * dr
            forces_exact[i] += q[j] * np.exp(-kappa * r) * (kappa * r + 1.0) / (r ** 3) * dr

    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
    forces_fmm = fmm.evaluate_forces(pts, q)
    rel = np.linalg.norm(forces_fmm - forces_exact) / max(1e-30, np.linalg.norm(forces_exact))
    print(f"toy_2cell_check_forces: rel-L2 = {rel:.3e} (target < 1e-5) "
          f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    return rel < 1e-5


if __name__ == "__main__":
    ok_fd = derivative_fd_guard()
    ok_toy = toy_2cell_check()
    ok_toy_f = toy_2cell_check_forces()
    if not (ok_fd and ok_toy and ok_toy_f):
        raise SystemExit(1)
    print("yukawa3d_fmm guards: PASS")
