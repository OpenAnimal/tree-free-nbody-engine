"""Unified radial Taylor FMM engine (dimension-parameterized).

Extracts the shared structure from the three copy-pasted engines:
  - core/gaussian2d_fgt.py      (2D, G_n = (-2/h^2)^n exp(-r^2/h^2))
  - core/yukawa3d_fmm.py        (3D, G_n = (-1)^n exp(-kr) Q_n(kr) / r^(2n+1))
  - core/screened_yukawa2d_fmm.py (2D, G_n = kappa^(2n) [a_n K0 + b_n K1])

Shared code (this module):
  - multi-index helpers (multi_indices, factorial) parameterized by dims
  - P_{alpha,n} polynomial-tensor builder (arbitrary spatial dims)
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

from typing import Callable, Dict, List, Tuple, Any

import numpy as np

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex

try:
    from ._csr import build_csr
    from .csr_p2p import _vectorized_neighbor_ids, _build_flat_sources
except ImportError:  # pragma: no cover - direct module execution
    from _csr import build_csr
    from csr_p2p import _vectorized_neighbor_ids, _build_flat_sources


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
        Cells per side = depth (LINEAR semantics — Round-7 task T-C8 / finding
        R7-F30: the docstring previously said "2^depth cells per side", which
        was wrong; `evaluate` uses `h_grid = 1/depth` and
        `CellIndex(dims, grid_res=depth)`, so cells-per-side is exactly
        `depth`. The vestigial `self.grid_res = 1 << depth` is replaced by
        `self.grid_res = self.depth` for readability; use `cells_per_side`
        in new code.)
    p : int
        Truncation order (moments and locals up to |alpha| <= p).
    dims : int
        Spatial dimensionality (any positive integer supported by CellIndex;
        1D-3D retain their historical key formats and higher dimensions use
        generic Morton keys).
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
        # Round-7 T-C8 / R7-F30: cells-per-side is `depth` (linear), NOT 2^depth.
        # The old `1 << self.depth` was vestigial and contradicted `evaluate`'s
        # actual `h_grid = 1/depth` + `CellIndex(grid_res=depth)` usage.
        self.grid_res = self.depth
        # Derivative-tensor polynomials P_{alpha, n} for |alpha| <= 2*p.
        self._P = build_P_tensors(max_order=2 * self.p, dims=self.dims)
        # Multi-index lists.
        self._alphas_p = multi_indices(self.p, self.dims)
        self._alphas_2p = multi_indices(2 * self.p, self.dims)
        self._alpha_p_index = {a: i for i, a in enumerate(self._alphas_p)}
        self._alpha_fact = np.array([factorial(a) for a in self._alphas_p])
        # Charge-independent M2L decomposition structure (alpha_idx, beta_idx,
        # sign) for each gamma with |gamma|<=2p. Depends only on (p, dims), so
        # compute once in __init__ and reuse across every evaluate call
        # (previously recomputed on every evaluate at :317/:418/:448/:594).
        self._decomp = self._decompositions()

    @property
    def cells_per_side(self) -> int:
        """Cells per side of the flat grid (= depth, linear semantics)."""
        return self.depth

    # ------------------------------------------------------------------

    def evaluate(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Single-shot potential evaluation.

        Delegates to `build_operator(positions)` (charge-independent build)
        followed by `evaluate_prebuilt(built, charges)` (charge-dependent
        contraction), so callers with fixed positions and many charge
        vectors can reuse the built operator across calls. The split
        performs the same operations in the same accumulation order as a
        monolithic body written against the same (CSR-ordered) near blocks,
        so results agree to last-ulp roundoff.
        """
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.float64)
        built = self.build_operator(positions)
        return self.evaluate_prebuilt(built, charges)

    # ------------------------------------------------------------------

    def build_operator(self, positions: np.ndarray) -> Dict[str, Any]:
        """Build the charge-independent operator for `positions`.

        Computes everything that does NOT depend on the charge vector:
          - the CellIndex over occupied cells (keys, inverse, centers),
          - the per-particle displacement from its cell center (disp_p),
          - the near-field kernel values per target cell with self-pairs
            already masked (so evaluate_prebuilt only has to weight by q),
          - the M2L derivative tensors D_gamma(d_ts) for every gamma with a
            non-empty decomposition (the expensive _eval_D_tensor / G_n
            work), plus the per-gamma (betas_idx, signs, d_list).

        Returns an opaque dict to pass to `evaluate_prebuilt`. The far-field
        D_gamma tensors dominate memory: O(n_gamma * K^2) where K <= depth^dims
        (e.g. depth=6 in 3D -> K<=216), which is modest for the flat scheme.

        Round-7 task T-D7: this enables the build-once / evaluate-many pattern
        for callers with fixed positions and many charge vectors (e.g. the
        Yukawa3D force/field sweeps and the IPC barrier Jacobian).
        """
        positions = np.asarray(positions, dtype=np.float64)
        N = len(positions)
        if N == 0:
            return {"N": 0, "empty": True}
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

        n_mom = len(self._alphas_p)
        pc = centers[inverse]                       # (N, dims)
        disp_p = positions - pc                     # (N, dims)

        # 2. Near field: precompute kernel values per target cell with self
        #    pairs masked (charge-independent).
        #
        #    Rewired (Round-7 T-E1 gate passed at >= 1.5x): the per-cell
        #    `cell_index.bucket` / `cell_index.neighborhood_indices` loop
        #    (125 elastic-hash probes per cell at ring=2 in 3D) is replaced
        #    by CSR cell-list ranges + vectorized `searchsorted` neighbor
        #    occupancy.  The near_blocks contain the same particle sets and
        #    the same kernel values; the source-particle ORDER within a
        #    block may differ from the old hash-probe order, so the
        #    charge-weighted sums in evaluate_prebuilt agree to last-ulp
        #    roundoff, NOT bit-for-bit (floating-point sums are
        #    order-dependent at the ~1e-16 relative level).
        cell_start, cell_particles, _ = build_csr(inverse, K)
        neighbor_ids = _vectorized_neighbor_ids(
            unique_keys, cell_index, K, self.ring_direct)

        # Precompute flat source-particle array with per-cell offsets.
        flat_sources, source_offsets = _build_flat_sources(
            neighbor_ids, cell_start, cell_particles, K)

        near_blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for c_id in range(K):
            idx_t = cell_particles[cell_start[c_id]:cell_start[c_id + 1]]
            if len(idx_t) == 0:
                continue
            s_lo = source_offsets[c_id]
            s_hi = source_offsets[c_id + 1]
            if s_hi == s_lo:
                continue
            near_idx = flat_sources[s_lo:s_hi]
            xt = positions[idx_t]            # (nt, dims)
            xs = positions[near_idx]         # (ns, dims)
            diff = xt[:, None, :] - xs[None, :, :]   # (nt, ns, dims)
            g = self._near_field_kernel(diff)         # (nt, ns)
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            g[id_t == id_s] = 0.0
            near_blocks.append((idx_t, near_idx, g))

        # 3. Far field: precompute the M2L derivative tensors D_gamma (the
        #    expensive _eval_D_tensor + G_n work) and per-gamma contraction
        #    metadata. All charge-independent.
        m2l_terms: List[Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray, np.ndarray]] = []
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]   # (K, K, dims)
            cheb = np.max(np.abs(dci), axis=-1)      # (K, K)
            far_mask = cheb > self.ring_direct       # (K, K) bool
            d_ts = centers[:, None, :] - centers[None, :, :]   # (K, K, dims)
            disp_components = [d_ts[:, :, i] for i in range(dims)]
            r_ts = np.sqrt(np.sum(d_ts * d_ts, axis=-1))
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)  # (K, K, 2p+1)
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)

            decomps = self._decomp
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                D_gamma = self._eval_D_tensor(gamma, disp_components, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                m2l_terms.append((d_list, D_gamma, betas_idx, signs))

        self.cell_index = cell_index
        return {
            "N": N, "K": K, "inverse": inverse, "disp_p": disp_p,
            "n_mom": n_mom, "near_blocks": near_blocks,
            "m2l_terms": m2l_terms, "cell_index": cell_index,
            "empty": False,
        }

    # ------------------------------------------------------------------

    def evaluate_prebuilt(self, built: Dict[str, Any],
                          charges: np.ndarray) -> np.ndarray:
        """Evaluate the potential for `charges` using a prebuilt operator.

        Performs only the charge-dependent work: P2M (charge moments), the
        near-field q-weighted sum (reusing the precomputed kernel values),
        the M2L contraction L = sum D_gamma * sign * M (reusing the
        precomputed D_gamma tensors), and L2P. Matches `evaluate` for the
        same (positions, charges) to last-ulp roundoff (source order within
        a near block is fixed by the CSR build, so repeated calls on the
        same built operator ARE bit-identical).
        """
        if built.get("empty", False):
            return np.empty(0, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = built["N"]
        K = built["K"]
        inverse = built["inverse"]
        disp_p = built["disp_p"]
        n_mom = built["n_mom"]

        # P2M: M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            beta_arr = np.array(beta, dtype=np.float64)
            w = charges * np.prod(disp_p ** beta_arr, axis=1) / factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # Near field: q-weighted sum over precomputed kernel values.
        pot = np.zeros(N, dtype=np.float64)
        for idx_t, near_idx, g in built["near_blocks"]:
            qs = charges[near_idx]           # (ns,)
            pot[idx_t] += np.sum(qs[None, :] * g, axis=1)

        # Far field: M2L contraction (precomputed D_gamma) then L2P.
        m2l_terms = built["m2l_terms"]
        if m2l_terms:
            n_loc = n_mom
            L = np.zeros((n_loc, K), dtype=np.float64)
            for d_list, D_gamma, betas_idx, signs in m2l_terms:
                Mstack = M[betas_idx] * signs[:, None]    # (m, K)
                contrib = D_gamma @ Mstack.T              # (K, m)
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]   # (n_loc, N)
            far_pot = np.zeros(N, dtype=np.float64)
            for ai, alpha in enumerate(self._alphas_p):
                alpha_arr = np.array(alpha, dtype=np.float64)
                far_pot += one_over_fact[ai] * Lp[ai] * np.prod(disp_p ** alpha_arr, axis=1)
            pot += far_pot

        self.cell_index = built["cell_index"]
        self._last_M = M
        return pot

    # ------------------------------------------------------------------

    def evaluate_targets(self, sources: np.ndarray, charges: np.ndarray,
                         targets: np.ndarray) -> np.ndarray:
        """Round-7 Workstream-G enabler: evaluate the kernel at `targets` from
        `sources` with `charges`, where sources and targets may differ.

        The sources are binned into the CellIndex (P2M + M2L as in `evaluate`);
        each target then gets:
          - Far field: L2P from the local expansion of the source cell it
            falls into; for targets in EMPTY source cells a per-target local
            expansion is built via M2L from all far source cells (full p-th
            order accuracy).
          - Near field: exact direct over the ring-2 source neighborhood of
            the target's cell.

        This is a ~20-line refactor of `evaluate` that separates the source
        binning from the target evaluation. Both T-G1 (radiotherapy dose
        grid) and T-G3 (groundwater targets) depend on it.

        Parameters
        ----------
        sources : (N_s, dims) in [0,1)^dims
        charges : (N_s,)
        targets : (N_t, dims) in [0,1)^dims

        Returns
        -------
        pot : (N_t,) — per-target potential (no self-pair masking needed;
            targets are distinct from sources).
        """
        sources = np.asarray(sources, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        N_s = len(sources)
        N_t = len(targets)
        if N_s == 0:
            return np.zeros(N_t, dtype=np.float64)
        if N_t == 0:
            return np.empty(0, dtype=np.float64)
        p = self.p
        depth = self.depth
        dims = self.dims
        h_grid = 1.0 / depth

        # 1. Bin SOURCES into the CellIndex.
        cell_index = CellIndex(dims=dims, grid_res=depth)
        unique_keys, inverse = cell_index.build(sources)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)
        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)
        centers = (cell_ints.astype(np.float64) + 0.5) * h_grid

        # 2. P2M on sources (same as evaluate).
        n_mom = len(self._alphas_p)
        pc = centers[inverse]
        disp_p = sources - pc
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            beta_arr = np.array(beta, dtype=np.float64)
            w = charges * np.prod(disp_p ** beta_arr, axis=1) / factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # 3. M2L over source cells (same as evaluate).
        L = np.zeros((n_mom, K), dtype=np.float64)
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]
            cheb = np.max(np.abs(dci), axis=-1)
            far_mask = cheb > self.ring_direct
            d_ts = centers[:, None, :] - centers[None, :, :]
            disp_components = [d_ts[:, :, i] for i in range(dims)]
            r_ts = np.sqrt(np.sum(d_ts * d_ts, axis=-1))
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)
            decomps = self._decomp
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                D_gamma = self._eval_D_tensor(gamma, disp_components, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                Mstack = M[betas_idx] * signs[:, None]
                contrib = D_gamma @ Mstack.T
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

        # 4. For each target, find its source cell (quantize to the grid).
        #    Build a key->cell_id map for the source cells.
        key_to_cid = {int(k): c for c, k in enumerate(unique_keys)}
        t_cell_ids = np.full(N_t, -1, dtype=np.int64)
        t_keys = np.empty(N_t, dtype=np.int64)
        for i in range(N_t):
            t_keys[i] = int(cell_index.key_of(targets[i]))
            t_cell_ids[i] = key_to_cid.get(int(t_keys[i]), -1)

        # 5. Far field L2P at each target. For targets in occupied source
        #    cells, use that cell's precomputed local expansion L. For targets
        #    in empty cells, compute a per-target local expansion via M2L from
        #    all far source cells (same derivative-tensor math, single target
        #    row), then L2P — full p-th order accuracy.
        pot = np.zeros(N_t, dtype=np.float64)
        one_over_fact = 1.0 / self._alpha_fact
        decomps = self._decomp

        for i in range(N_t):
            c = t_cell_ids[i]
            if c >= 0:
                # Target in an occupied source cell: use precomputed L.
                ct = centers[c]
                disp_t = targets[i] - ct
                for ai, alpha in enumerate(self._alphas_p):
                    alpha_arr = np.array(alpha, dtype=np.float64)
                    pot[i] += one_over_fact[ai] * L[ai, c] * np.prod(disp_t ** alpha_arr)
            else:
                # Target in an empty cell: compute L_target via M2L from all
                # far source cells, then L2P.
                t_ints = np.array(cell_index.key_ints(int(t_keys[i])), dtype=np.int64)
                ct = (t_ints.astype(np.float64) + 0.5) * h_grid
                # Chebyshev distance from target cell to each source cell.
                cheb_src = np.max(np.abs(cell_ints - t_ints[None, :]), axis=-1)
                far_src_mask = cheb_src > self.ring_direct
                if not np.any(far_src_mask):
                    continue
                # d = center_target - center_source for far source cells.
                far_idx = np.where(far_src_mask)[0]
                d_ts = ct[None, :] - centers[far_idx]  # (K_far, dims)
                disp_components_t = [d_ts[:, j] for j in range(dims)]
                r_ts = np.sqrt(np.sum(d_ts * d_ts, axis=-1))
                Gn_t = np.stack([self._G_n(r_ts, n) for n in range(2 * p + 1)], axis=-1)
                # Compute L_target via M2L.
                L_target = np.zeros(n_mom, dtype=np.float64)
                for gamma in self._alphas_2p:
                    d_list = decomps.get(gamma)
                    if not d_list:
                        continue
                    D_gamma = self._eval_D_tensor(gamma, disp_components_t, Gn_t)  # (K_far,)
                    for j, (ai, bi, sign) in enumerate(d_list):
                        L_target[ai] += sign * np.sum(D_gamma * M[bi, far_idx])
                # L2P at target.
                disp_t = targets[i] - ct
                for ai, alpha in enumerate(self._alphas_p):
                    alpha_arr = np.array(alpha, dtype=np.float64)
                    pot[i] += one_over_fact[ai] * L_target[ai] * np.prod(disp_t ** alpha_arr)

        # 6. Near field: for each target, gather sources from the ring-2
        #    neighborhood of the target's cell.
        for i in range(N_t):
            near_idx = cell_index.neighborhood_indices(int(t_keys[i]), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xs = sources[near_idx]
            qs = charges[near_idx]
            diff = targets[i][None, :] - xs  # (ns, dims)
            g = self._near_field_kernel(diff[None, :, :])  # (1, ns)
            pot[i] += np.sum(qs * g[0])

        return pot

    def evaluate_forces(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Round-7 task T-D6: compute the FIELD E_i = -∇u(x_i) from the local
        Taylor polynomial of the far field, plus exact near-field gradient.

        Convention: this returns the FIELD (force per unit charge), NOT the
        mechanical force. The potential u_i = sum_j q_j G(x_i - x_j) already
        includes the source charges q_j; the target charge q_i is NOT folded
        in. Callers that need the mechanical force must multiply by q_i:
            F_i = q_i * E_i.
        This convention is consistent across all callers (e.g.
        `bioinformatics/core/fast_multipole_kernel.py` multiplies by q_i
        explicitly, `neural_ops/equivariant_field_layer.py` uses the field
        directly as a vector feature).

        The far-field force uses the same L coefficients as `evaluate` (the
        M2L step is identical); only the L2P contraction changes:
        F_i^{(d)} = -Σ_{α: α_d>0} (1/α!) L_α · α_d · (x_i-c)^{α-e_d}.

        The near-field force is a finite-difference approximation
        (O(eps_fd^2), central differences, eps_fd=1e-6) of the near-field
        kernel gradient -- NOT an exact analytic gradient. The kernel is
        smooth away from r=0 and self-pairs are masked, but distinct-but-
        nearly-coincident source/target pairs closer than eps_fd are
        approximated poorly (the central-difference stencil straddles the
        singularity).

        Returns (N, dims) force array.
        """
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        dims = self.dims
        if N == 0:
            return np.empty((0, dims), dtype=np.float64)
        p = self.p
        depth = self.depth
        h_grid = 1.0 / depth

        # 1. Build the CellIndex (same as evaluate)
        cell_index = CellIndex(dims=dims, grid_res=depth)
        unique_keys, inverse = cell_index.build(positions)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)
        centers = (cell_ints.astype(np.float64) + 0.5) * h_grid

        # 2. P2M (same as evaluate)
        n_mom = len(self._alphas_p)
        pc = centers[inverse]
        disp_p = positions - pc
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            beta_arr = np.array(beta, dtype=np.float64)
            w = charges * np.prod(disp_p ** beta_arr, axis=1) / factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        forces = np.zeros((N, dims), dtype=np.float64)

        # 3. Near field: exact gradient via finite differences of the kernel
        #    (the kernel is smooth away from r=0; self-pairs masked).
        eps_fd = 1e-6
        for c_id, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            near_idx = cell_index.neighborhood_indices(int(key), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xt = positions[idx_t]
            xs = positions[near_idx]
            qs = charges[near_idx]
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            self_mask = (id_t == id_s)

            for d in range(dims):
                # Central finite difference of kernel w.r.t. x_d (target coord)
                diff_plus = xt[:, None, :] - xs[None, :, :]
                diff_plus[:, :, d] += eps_fd
                diff_minus = xt[:, None, :] - xs[None, :, :]
                diff_minus[:, :, d] -= eps_fd
                g_plus = self._near_field_kernel(diff_plus)
                g_minus = self._near_field_kernel(diff_minus)
                grad_d = (g_plus - g_minus) / (2.0 * eps_fd)
                grad_d = np.where(self_mask, 0.0, grad_d)
                # F = -grad u, and u = sum_j q_j G(x_i - x_j)
                forces[idx_t, d] -= np.sum(qs[None, :] * grad_d, axis=1)

        # 4. Far field: M2L (same as evaluate) then L2P gradient
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]
            cheb = np.max(np.abs(dci), axis=-1)
            far_mask = cheb > self.ring_direct
            d_ts = centers[:, None, :] - centers[None, :, :]
            disp_components = [d_ts[:, :, i] for i in range(dims)]
            r_ts = np.sqrt(np.sum(d_ts * d_ts, axis=-1))
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)

            n_loc = n_mom
            L = np.zeros((n_loc, K), dtype=np.float64)
            decomps = self._decomp
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                D_gamma = self._eval_D_tensor(gamma, disp_components, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                Mstack = M[betas_idx] * signs[:, None]
                contrib = D_gamma @ Mstack.T
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            # 5. L2P gradient: F_i^{(d)} = -Σ_{α: α_d>0} (1/α!) L_α · α_d · (x-c)^{α-e_d}
            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]   # (n_loc, N)
            for d in range(dims):
                far_force_d = np.zeros(N, dtype=np.float64)
                for ai, alpha in enumerate(self._alphas_p):
                    if alpha[d] == 0:
                        continue
                    # (x-c)^{alpha - e_d}
                    alpha_shifted = list(alpha)
                    alpha_shifted[d] -= 1
                    alpha_shifted_arr = np.array(alpha_shifted, dtype=np.float64)
                    disp_term = np.prod(disp_p ** alpha_shifted_arr, axis=1)
                    far_force_d += one_over_fact[ai] * alpha[d] * Lp[ai] * disp_term
                forces[:, d] -= far_force_d

        self.cell_index = cell_index
        self._last_M = M
        return forces

    # ------------------------------------------------------------------

    def _eval_D_tensor(self, gamma: Tuple[int, ...],
                       disp_components: List[np.ndarray],
                       Gn: np.ndarray) -> np.ndarray:
        """D_gamma(d) = sum_n P_{gamma,n}(d) * G_n(|d|).

        Gn may be 3D (K, K, 2p+1) for the standard M2L or 2D (K_far, 2p+1)
        for the single-target M2L used by `evaluate_targets`.
        """
        out = np.zeros_like(disp_components[0])
        Pgamma = self._P.get(gamma, {})
        for n, poly in Pgamma.items():
            Pval = eval_poly(poly, disp_components)
            out += Pval * Gn[..., n]
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
