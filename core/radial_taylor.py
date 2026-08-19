"""Unified radial Taylor FMM engine (dimension-parameterized).

Extracts the shared structure from the three copy-pasted engines:
  - core/gaussian2d_fgt.py      (2D, G_n = (-2/h^2)^n exp(-r^2/h^2))
  - core/yukawa3d_fmm.py        (3D, G_n = (-1)^n exp(-kr) Q_n(kr) / r^(2n+1))
  - core/screened_yukawa2d_fmm.py (2D, G_n = kappa^(2n) [a_n K0 + b_n K1])

Shared code (this module):
  - multi-index helpers (multi_indices, factorial) parameterized by dims
  - P_{alpha,n} polynomial-tensor builder (dims=2|3)
  - polynomial helpers (deriv_xi, mul_xi, eval_poly)
  - ring-2 flat scheme driver (P2M / M2L / L2P / near-field via CellIndex)

Kernel-specific code (each engine module):
  - the G_n evaluator (the radial function family)
  - the near-field kernel (exp(-r2/h2), exp(-kr)/r, K0(kr))
  - the guard functions (derivative_fd_guard, toy_2cell_check, etc.)

The three engine classes subclass RadialTaylorFMM and supply their G_n
and near-field kernel in __init__; everything else is inherited.

------------------------------------------------------------------------------
MATH (identical to the three engine modules; see those for full derivations).

1. Radial functions G_n(r): kernel-specific, supplied as a callable
   G_n(r: ndarray, n: int) -> ndarray.

2. Derivative tensors: for displacement d (a dims-vector),
       D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)
   with the dimension-independent recursion
       P_{(0,...,0),0} = 1;  P_{alpha,n} = 0 if n<0 or n>|alpha|;
       P_{alpha+e_i, n} = d/dx_i[P_{alpha,n}] + x_i * P_{alpha,n-1}.

3. Flat FMM (grid spacing h = 1/depth, cell center c):
   - P2M:  M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
   - Near: exact direct over ring-2 neighborhood
   - M2L:  L_alpha(t) = sum_s sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s)
   - L2P:  u(x) = sum_alpha (1/alpha!) L_alpha(t) (x - c_t)^alpha

SIGN / FACTORIAL CONVENTION: the standard Taylor identity (corrected
round-3 form, verified by the mandatory 2-cell toy check in each engine):
   G(d + u - v) = sum_{a,b} (1/a!)(1/b!) D_{a+b}(d) u^a (-v)^b
so L_alpha = sum_s sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s)
and u(x) = sum_alpha (1/alpha!) L_alpha(t) (x - c_t)^alpha.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex


# =====================================================================
# Multi-index helpers (dimension-parameterized)
# =====================================================================

def multi_indices(order: int, dims: int) -> List[Tuple[int, ...]]:
    """All multi-indices alpha with |alpha| <= order in dims dimensions,
    ordered by total degree then lexicographic (matching the original
    per-dimension _multi_indices functions)."""
    out: List[Tuple[int, ...]] = []
    for total in range(order + 1):
        _gen_indices(total, dims, (), out)
    return out


def _gen_indices(remaining: int, dims: int,
                 prefix: Tuple[int, ...],
                 out: List[Tuple[int, ...]]) -> None:
    if dims == 1:
        out.append(prefix + (remaining,))
        return
    for a in range(remaining + 1):
        _gen_indices(remaining - a, dims - 1, prefix + (a,), out)


def factorial(alpha: Tuple[int, ...]) -> int:
    """Product of factorials of the components of alpha."""
    from math import factorial as _f
    r = 1
    for a in alpha:
        r *= _f(a)
    return r


# =====================================================================
# Polynomial helpers (dimension-independent, operate on tuple keys)
# =====================================================================
# Representation: poly = {exp_tuple: coef} where exp_tuple has length dims.

def deriv_xi(poly: Dict[Tuple[int, ...], float],
             axis: int) -> Dict[Tuple[int, ...], float]:
    """d/dx_axis of a polynomial represented as {exp: coef}."""
    out: Dict[Tuple[int, ...], float] = {}
    for exp_tuple, c in poly.items():
        e = exp_tuple[axis]
        if e > 0:
            exp2 = list(exp_tuple)
            exp2[axis] = e - 1
            out[tuple(exp2)] = out.get(tuple(exp2), 0.0) + c * e
    return out


def mul_xi(poly: Dict[Tuple[int, ...], float],
           axis: int) -> Dict[Tuple[int, ...], float]:
    """Multiply by x_axis (shift exponent on `axis` by +1)."""
    out: Dict[Tuple[int, ...], float] = {}
    for exp_tuple, c in poly.items():
        exp2 = list(exp_tuple)
        exp2[axis] += 1
        out[tuple(exp2)] = c
    return out


def eval_poly(poly: Dict[Tuple[int, ...], float],
              disp_components: List[np.ndarray]) -> np.ndarray:
    """Evaluate a polynomial at a batch of displacements.
    disp_components = [dx, dy, ...] (one array per axis, same shape)."""
    out = np.zeros_like(disp_components[0])
    for exp_tuple, c in poly.items():
        term = c
        for axis, e in enumerate(exp_tuple):
            term = term * (disp_components[axis] ** e)
        out += term
    return out


# =====================================================================
# P_{alpha, n} tensor builder (dimension-parameterized)
# =====================================================================

def build_P_tensors(max_order: int,
                    dims: int) -> Dict[Tuple[int, ...], Dict[int, Dict[Tuple[int, ...], float]]]:
    """Build P_{alpha, n} for all |alpha| <= max_order, 0 <= n <= |alpha|.
    Recurrence (dimension-independent):
        P_{(0,...,0), 0} = 1
        P_{alpha+e_i, n} = d/dx_i[P_{alpha,n}] + x_i * P_{alpha, n-1}
        P_{alpha, n} = 0 if n<0 or n>|alpha|.
    """
    P: Dict[Tuple[int, ...], Dict[int, Dict[Tuple[int, ...], float]]] = {}
    zero = tuple([0] * dims)
    P[zero] = {0: {zero: 1.0}}

    all_alphas = multi_indices(max_order, dims)

    for alpha in all_alphas:
        if alpha == zero:
            continue
        P.setdefault(alpha, {})
        order = sum(alpha)
        # Use the FIRST valid predecessor axis (the recurrence is
        # consistent across all valid predecessor axes; summing would
        # double-count).
        chosen_axis = -1
        for axis, comp in enumerate(alpha):
            if comp > 0:
                chosen_axis = axis
                break
        alpha_prev = list(alpha)
        alpha_prev[chosen_axis] -= 1
        alpha_prev = tuple(alpha_prev)
        P_prev = P.get(alpha_prev, {})
        for n in range(order + 1):
            # term 1: d/dx_i [P_{alpha_prev, n}]
            poly_n = P_prev.get(n)
            if poly_n:
                d = deriv_xi(poly_n, axis=chosen_axis)
                if d:
                    P[alpha].setdefault(n, {})
                    for exp, cc in d.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
            # term 2: x_i * P_{alpha_prev, n-1}
            poly_nm1 = P_prev.get(n - 1)
            if poly_nm1:
                m = mul_xi(poly_nm1, axis=chosen_axis)
                if m:
                    P[alpha].setdefault(n, {})
                    for exp, cc in m.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
    return P


# =====================================================================
# Unified ring-2 flat scheme driver
# =====================================================================

class RadialTaylorFMM:
    """Single-level flat radial Taylor FMM on a uniform grid.

    Parameters
    ----------
    depth : int
        Grid resolution is 2^depth cells per side.
    p : int
        Truncation order (moments and locals up to |alpha| <= p).
    dims : int
        Spatial dimensionality (2 or 3).
    G_n : callable
        Radial function evaluator G_n(r: ndarray, n: int) -> ndarray.
    near_field_kernel : callable
        Maps diff (nt, ns, dims) -> kernel values (nt, ns).  Self-pair
        entries (zero displacement) may produce any value; the driver
        masks them to 0.
    ring_direct : int
        Near-field neighborhood ring (default 2 = 5x5[x5] box).
    """

    def __init__(self, depth: int, p: int, dims: int,
                 G_n: Callable[[np.ndarray, int], np.ndarray],
                 near_field_kernel: Callable[[np.ndarray], np.ndarray],
                 ring_direct: int = 2):
        self.depth = int(depth)
        self.p = int(p)
        self.dims = int(dims)
        self.ring_direct = int(ring_direct)
        self._G_n = G_n
        self._near_field_kernel = near_field_kernel
        self.grid_res = 1 << self.depth
        # Derivative-tensor polynomials P_{alpha, n} for |alpha| <= 2*p.
        self._P = build_P_tensors(max_order=2 * self.p, dims=self.dims)
        # Multi-index lists.
        self._alphas_p = multi_indices(self.p, self.dims)
        self._alphas_2p = multi_indices(2 * self.p, self.dims)
        self._alpha_p_index = {a: i for i, a in enumerate(self._alphas_p)}
        self._alpha_fact = np.array([factorial(a) for a in self._alphas_p])

    # ------------------------------------------------------------------

    def evaluate(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            return np.empty(0, dtype=np.float64)
        p = self.p
        depth = self.depth
        dims = self.dims
        h_grid = 1.0 / depth

        # 1. Build the CellIndex over occupied cells.
        cell_index = CellIndex(dims=dims, grid_res=depth)
        unique_keys, inverse = cell_index.build(positions)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        # Decode cell integer coords and centers.
        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)  # (K, dims)
        centers = (cell_ints.astype(np.float64) + 0.5) * h_grid  # (K, dims)

        # 2. P2M: M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
        n_mom = len(self._alphas_p)
        pc = centers[inverse]                       # (N, dims)
        disp_p = positions - pc                     # (N, dims)
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            beta_arr = np.array(beta, dtype=np.float64)
            w = charges * np.prod(disp_p ** beta_arr, axis=1) / factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # 3. Near field: exact direct over ring-2 neighborhood of each
        #    target cell.
        pot = np.zeros(N, dtype=np.float64)
        for c_id, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            near_idx = cell_index.neighborhood_indices(int(key), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xt = positions[idx_t]            # (nt, dims)
            xs = positions[near_idx]         # (ns, dims)
            qs = charges[near_idx]           # (ns,)
            diff = xt[:, None, :] - xs[None, :, :]   # (nt, ns, dims)
            g = self._near_field_kernel(diff)         # (nt, ns)
            # zero out self pairs (same particle index).
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            self_mask = (id_t == id_s)
            g = np.where(self_mask, 0.0, g)
            pot[idx_t] += np.sum(qs[None, :] * g, axis=1)

        # 4. Far field: M2L over all well-separated (outside ring-2) cell
        #    pairs, then L2P per particle.
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]   # (K, K, dims)
            cheb = np.max(np.abs(dci), axis=-1)      # (K, K)
            far_mask = cheb > self.ring_direct       # (K, K) bool
            d_ts = centers[:, None, :] - centers[None, :, :]   # (K, K, dims)
            disp_components = [d_ts[:, :, i] for i in range(dims)]
            r_ts = np.sqrt(np.sum(d_ts * d_ts, axis=-1))
            # G_n(|d|) for n=0..2p; near pairs zeroed.
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)  # (K, K, 2p+1)
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)

            n_loc = n_mom
            L = np.zeros((n_loc, K), dtype=np.float64)
            decomps = self._decompositions()
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                D_gamma = self._eval_D_tensor(gamma, disp_components, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                Mstack = M[betas_idx] * signs[:, None]    # (m, K)
                contrib = D_gamma @ Mstack.T              # (K, m)
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            # 5. L2P: u_far(x_i) = sum_alpha (1/alpha!) L_alpha(cell_i) (x_i-c)^alpha
            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]   # (n_loc, N)
            far_pot = np.zeros(N, dtype=np.float64)
            for ai, alpha in enumerate(self._alphas_p):
                alpha_arr = np.array(alpha, dtype=np.float64)
                far_pot += one_over_fact[ai] * Lp[ai] * np.prod(disp_p ** alpha_arr, axis=1)
            pot += far_pot

        self.cell_index = cell_index
        self._last_M = M
        return pot

    # ------------------------------------------------------------------

    def _eval_D_tensor(self, gamma: Tuple[int, ...],
                       disp_components: List[np.ndarray],
                       Gn: np.ndarray) -> np.ndarray:
        """D_gamma(d) = sum_n P_{gamma,n}(d) * G_n(|d|)."""
        out = np.zeros_like(disp_components[0])
        Pgamma = self._P.get(gamma, {})
        for n, poly in Pgamma.items():
            Pval = eval_poly(poly, disp_components)
            out += Pval * Gn[:, :, n]
        return out

    def _decompositions(self) -> Dict[Tuple[int, ...], List[Tuple[int, int, float]]]:
        """For each gamma with |gamma|<=2p, list (alpha_idx, beta_idx, sign)
        with |alpha|<=p, |beta|<=p, alpha+beta=gamma, sign=(-1)^|beta|."""
        p = self.p
        out: Dict[Tuple[int, ...], List[Tuple[int, int, float]]] = {}
        for ai, alpha in enumerate(self._alphas_p):
            for bi, beta in enumerate(self._alphas_p):
                gamma = tuple(a + b for a, b in zip(alpha, beta))
                if sum(gamma) > 2 * p:
                    continue
                out.setdefault(gamma, []).append((ai, bi, (-1.0) ** sum(beta)))
        return out

    # ------------------------------------------------------------------

    def D_alpha(self, d: np.ndarray, alpha: Tuple[int, ...]) -> float:
        """Single-point derivative tensor D_alpha(d) for a dims-vector d.
        Used by the FD guard tests in the engine modules."""
        d = np.asarray(d, dtype=np.float64).reshape(self.dims)
        r = float(np.linalg.norm(d))
        if r < 1e-30:
            return 0.0
        val = 0.0
        Palpha = self._P.get(alpha, {})
        for n, poly in Palpha.items():
            Pval = 0.0
            for exp_tuple, c in poly.items():
                term = c
                for axis, e in enumerate(exp_tuple):
                    term *= d[axis] ** e
                Pval += term
            val += Pval * float(self._G_n(np.array([r]), n)[0])
        return val
