"""Level-batched vectorized 2D adaptive FMM (CGR88 four-list scheme).

High-throughput sibling of ``core.adaptive_fmm.TreeFreeElasticAdaptiveFMM``
(the classical engine stays as the pedagogical / cross-validation reference).
Same mathematics -- Carrier, Greengard, & Rokhlin (1988) adaptive multipole
expansions with the exact four interaction lists -- but every pass is batched
per tree level into dense ``(n_boxes, p+1)`` complex arrays, following the
standard optimization lineage of production FMM codes:

- **2:1 level-balanced quadtree** so adjacent leaves never differ by more
  than one level; this bounds all four interaction lists to constant size
  (Sundar, Sampath, & Biros, 2008; Ying, Biros, & Zorin, 2004).
- **Interaction lists built from the bounded colleague ring**, never by root
  recursion (Carrier, Greengard, & Rokhlin, 1988, Section 3; Yokota, 2012).
  List-2 sources live on the FMMLIB2D-style ``(-3..3)^2 \\ 3x3`` stencil
  with a parity-dependent colleague-ring test; List-3/4 separated children
  sit at per-axis offsets ``{-2, +3}`` relative to the target leaf box.
- **M2L operators precomputed per (level, relative offset)** as dense
  ``(p+1, p+1)`` matrices, so each offset class collapses to one BLAS matmul
  (Gimbutas & Greengard, 2012, FMMLIB2D ``itable(-3:3,-3:3)``; exafmm-t
  ``M2L_setup``).
- **Vectorized List-4 P2L**: the per-particle P2L of CGR88 is kept exactly
  (M2L from the coarse leaf's multipole would converge at ratio
  (sqrt(2)+sqrt(2)/2)/2.5 ~ 0.85 per term and lose an order of accuracy);
  it is batched into ragged (particle, cell) rows with sorted
  ``np.add.reduceat`` segment sums instead of Python loops.
- **CSR near-field P2P** with per-leaf concatenated source blocks (Lashuk et
  al., 2012: sorted/concatenated particle arrays).

Cell index: as everywhere in this repo, the authoritative cell index is the
funnel hash (Farach-Colton, Krapivin, & Kuszmaul, 2025,
``core.elastic_hash.ElasticHashTable``) mapping each level-tagged Morton cell
key to its dense cell id. The hot passes work on per-level dense occupancy
grids -- the "implicit lattice" -- which are the vectorizable O(1) equivalent
of hash membership probes (the same hybrid ``FastVectorizedFMM`` documents).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .adaptive_fmm import morton_encode_box
    from .elastic_hash import ElasticHashTable
except ImportError:  # standalone execution
    from adaptive_fmm import morton_encode_box
    from elastic_hash import ElasticHashTable

__all__ = ["FastAdaptiveFMM"]


# ---------------------------------------------------------------------------
# Batched translation operators (flattened over n box pairs)
# ---------------------------------------------------------------------------

def _m2m_batch(a: np.ndarray, delta: np.ndarray, p: int) -> np.ndarray:
    """M2M (CGR88 Theorem 2.2) for n pairs at once.

    a: (n, p+1) source multipoles; delta: (n,) complex = src - dst centers.
        b_0 = a_0
        b_l = -a_0 delta^l / l + sum_{k=1..l} binom(l-1, k-1) a_k delta^(l-k)
    """
    n = len(delta)
    dpow = np.empty((n, p + 1), dtype=np.complex128)
    dpow[:, 0] = 1.0
    for l in range(1, p + 1):
        dpow[:, l] = dpow[:, l - 1] * delta
    b = np.empty((n, p + 1), dtype=np.complex128)
    b[:, 0] = a[:, 0]
    for l in range(1, p + 1):
        term = -a[:, 0] * dpow[:, l] / l
        for k in range(1, l + 1):
            term = term + math.comb(l - 1, k - 1) * a[:, k] * dpow[:, l - k]
        b[:, l] = term
    return b


def _l2l_batch(c: np.ndarray, delta: np.ndarray, p: int) -> np.ndarray:
    """L2L (CGR88 Theorem 2.4) for n pairs at once.

    c: (n, p+1) source locals; delta: (n,) = dst - src centers.
        d_l = sum_{k=l..p} binom(k, l) c_k delta^(k-l)
    """
    n = len(delta)
    dpow = np.empty((n, p + 1), dtype=np.complex128)
    dpow[:, 0] = 1.0
    for l in range(1, p + 1):
        dpow[:, l] = dpow[:, l - 1] * delta
    d = np.empty((n, p + 1), dtype=np.complex128)
    for l in range(p + 1):
        term = c[:, l].copy()
        for k in range(l + 1, p + 1):
            term = term + math.comb(k, l) * c[:, k] * dpow[:, k - l]
        d[:, l] = term
    return d


_M2L_TABLES: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _m2l_tables(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-p cached tables for the M2L matrix: (signs, binomials, k+l power
    indices), all (p+1, p+1) with row index l, column index k."""
    if p not in _M2L_TABLES:
        l_idx, k_idx = np.mgrid[0:p + 1, 0:p + 1]
        signs = np.ones((p + 1, p + 1), dtype=np.float64)
        signs[1:, :] = (-1.0) ** l_idx[1:, :]         # (-1)^l for l >= 1
        signs[1:, 0] = (-1.0) ** (l_idx[1:, 0] - 1)   # (-1)^(l-1) in column 0
        binoms = np.ones((p + 1, p + 1), dtype=np.float64)
        for l in range(1, p + 1):
            for k in range(1, p + 1):
                binoms[l, k] = math.comb(k + l - 1, l)
        pow_idx = (l_idx + k_idx).astype(np.int64)
        _M2L_TABLES[p] = (signs, binoms, pow_idx)
    return _M2L_TABLES[p]


def _m2l_matrix(delta: complex, p: int) -> np.ndarray:
    """Dense (p+1, p+1) M2L matrix for fixed separation delta = dst - src,
    valid for |delta| > r_src + r_dst. local = multipole @ M.T:
        c_0 = a_0 ln(delta) + sum_{k>=1} a_k delta^{-k}
        c_l = a_0 (-1)^{l-1}/(l delta^l)
              + sum_{k>=1} (-1)^l binom(k+l-1, l) a_k delta^{-(k+l)}
    """
    signs, binoms, pow_idx = _m2l_tables(p)
    dp = (1.0 / complex(delta)) ** np.arange(2 * p + 1)
    M = signs * binoms * dp[pow_idx]
    M[1:, 0] /= np.arange(1, p + 1)   # the 1/l factor in c_l, l >= 1
    M[0, 0] = np.log(complex(delta))
    return M


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FastAdaptiveFMM:
    """Vectorized CGR88 adaptive FMM on a 2:1-balanced, hash-indexed quadtree.

    Parameters mirror ``TreeFreeElasticAdaptiveFMM`` so the two engines are
    drop-in comparable in the benchmark table.
    """

    def __init__(
        self,
        max_leaf_particles: int = 24,
        base_depth: int = 2,
        max_depth: int = 9,
        p: int = 10,
        softening: float = 0.0,
    ):
        if max_depth > 12:
            raise ValueError(
                "max_depth > 12 allocates dense per-level occupancy grids "
                f"((2^max_depth)^2 int64 each); got {max_depth}")
        self.max_leaf_particles = max_leaf_particles
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.p = p
        self.softening = softening
        # Funnel hash: authoritative cell index (level-tagged Morton key -> id).
        self.hash_table: Optional[ElasticHashTable] = None
        self.cell_keys: List[int] = []

    # ------------------------------------------------------------------ build

    def _grow(self, k: int) -> int:
        """Extend cell storage by k slots; returns first new id."""
        old = self.n_cells
        need = old + k
        if need > self._cap:
            new_cap = max(self._cap * 2, need)
            for name, dt in (("lvl", np.int64), ("cx", np.int64),
                             ("cy", np.int64), ("par", np.int64),
                             ("cnt", np.int64), ("cen", np.complex128),
                             ("leaf", np.bool_), ("chb", np.int64)):
                arr = getattr(self, "_" + name)
                grown = np.empty(new_cap, dtype=dt)
                grown[:old] = arr[:old]
                setattr(self, "_" + name, grown)
            self._cap = new_cap
        self.n_cells = need
        return old

    def _add_level_grid(self, lvl: int) -> np.ndarray:
        if lvl not in self._occ:
            self._occ[lvl] = np.full((1 << lvl, 1 << lvl), -1, dtype=np.int64)
        return self._occ[lvl]

    def _split_cells(self, parents: np.ndarray, positions: np.ndarray) -> None:
        """Split leaf cells `parents` (same level) into 4 children each and
        rebin their particles one level deeper. Fully vectorized."""
        lvl = int(self._lvl[parents[0]])
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0
        npar = len(parents)
        base = self._grow(4 * npar)
        pix = self._cx[parents]
        piy = self._cy[parents]

        q = np.arange(4)
        qx = (q & 1)[None, :] + 2 * pix[:, None]      # (npar, 4)
        qy = (q >> 1)[None, :] + 2 * piy[:, None]
        ids = base + 4 * np.arange(npar)[:, None] + q[None, :]

        self._lvl[ids.ravel()] = lvl + 1
        self._cx[ids.ravel()] = qx.ravel()
        self._cy[ids.ravel()] = qy.ravel()
        self._par[ids.ravel()] = np.repeat(parents, 4)
        self._leaf[ids.ravel()] = True
        self._cnt[ids.ravel()] = 0
        self._chb[ids.ravel()] = -1
        hx = Wx / (1 << (lvl + 1))
        hy = Wy / (1 << (lvl + 1))
        self._cen[ids.ravel()] = (bx0 + (qx.ravel() + 0.5) * hx) + \
            1j * (by0 + (qy.ravel() + 0.5) * hy)
        self._leaf[parents] = False
        self._chb[parents] = base + 4 * np.arange(npar)

        grid = self._add_level_grid(lvl + 1)
        grid[qx.ravel(), qy.ravel()] = ids.ravel()

        # rebin particles of the split parents
        member = np.isin(self.pcell, parents)
        if member.any():
            sub = np.nonzero(member)[0]
            sc = self.pcell[sub]
            six = self._cx[sc]
            siy = self._cy[sc]
            cix = np.clip(((positions[sub, 0] - bx0) / Wx * (1 << (lvl + 1)))
                          .astype(np.int64), 2 * six, 2 * six + 1)
            ciy = np.clip(((positions[sub, 1] - by0) / Wy * (1 << (lvl + 1)))
                          .astype(np.int64), 2 * siy, 2 * siy + 1)
            cid = self._chb[sc] + (cix - 2 * six) + 2 * (ciy - 2 * siy)
            self.pcell[sub] = cid
            np.add.at(self._cnt, cid, 1)
            self._cnt[sc] = 0

    def _build(self, positions: np.ndarray, N: int) -> None:
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0

        cap = 1024
        self._cap = cap
        self.n_cells = 0
        self._lvl = np.zeros(cap, dtype=np.int64)
        self._cx = np.zeros(cap, dtype=np.int64)
        self._cy = np.zeros(cap, dtype=np.int64)
        self._par = np.full(cap, -1, dtype=np.int64)
        self._cnt = np.zeros(cap, dtype=np.int64)
        self._cen = np.zeros(cap, dtype=np.complex128)
        self._leaf = np.zeros(cap, dtype=bool)
        self._chb = np.full(cap, -1, dtype=np.int64)
        self._occ: Dict[int, np.ndarray] = {}

        # base-level binning
        b = self.base_depth
        grid = self._add_level_grid(b)
        fx = (positions[:, 0] - bx0) / Wx * (1 << b)
        fy = (positions[:, 1] - by0) / Wy * (1 << b)
        ix_b = np.clip(fx.astype(np.int64), 0, (1 << b) - 1)
        iy_b = np.clip(fy.astype(np.int64), 0, (1 << b) - 1)
        keys = ix_b * (1 << b) + iy_b
        uniq, inv = np.unique(keys, return_inverse=True)
        n0 = len(uniq)
        base = self._grow(n0)
        uix = uniq // (1 << b)
        uiy = uniq % (1 << b)
        ids = base + np.arange(n0)
        self._lvl[ids] = b
        self._cx[ids] = uix
        self._cy[ids] = uiy
        self._par[ids] = -1
        self._leaf[ids] = True
        self._chb[ids] = -1
        hbx = Wx / (1 << b)
        hby = Wy / (1 << b)
        self._cen[ids] = (bx0 + (uix + 0.5) * hbx) + 1j * (by0 + (uiy + 0.5) * hby)
        grid[uix, uiy] = ids
        self.pcell = ids[inv]
        self._cnt[ids] = np.bincount(inv, minlength=n0)

        # occupancy-driven splitting (level by level until quiescent)
        while True:
            did = False
            for lvl in range(self.base_depth, self.max_depth):
                g = self._occ.get(lvl)
                if g is None:
                    continue
                ids = g[g >= 0]
                over = ids[self._leaf[ids] & (self._cnt[ids] > self.max_leaf_particles)]
                if len(over):
                    self._split_cells(over, positions)
                    did = True
            if not did:
                break

        # 2:1 balance is REQUIRED, not optional: the bounded interaction
        # list construction (colleague rings, one-level descent) is only
        # complete for level-balanced trees.
        self._balance(positions)

        self._build_hash_index()

    def _balance(self, positions: np.ndarray) -> None:
        """Enforce 2:1 balance: split any leaf >= 2 levels coarser than an
        adjacent leaf, via 3x3 grid dilations projected onto coarser levels."""
        for _ in range(2 * self.max_depth + 4):
            to_split: List[np.ndarray] = []
            for lvl in range(self.max_depth, self.base_depth, -1):
                grid = self._occ.get(lvl)
                if grid is None:
                    continue
                leaf_cells = grid[grid >= 0]
                leaf_cells = leaf_cells[self._leaf[leaf_cells]]
                if len(leaf_cells) == 0:
                    continue
                mask = np.zeros_like(grid, dtype=bool)
                mask[self._cx[leaf_cells], self._cy[leaf_cells]] = True
                # Chebyshev 3x3 dilation = cross dilation applied twice
                dil = mask.copy()
                for _ in range(2):
                    nxt = dil.copy()
                    for ax in (0, 1):
                        for off in (1, -1):
                            a = [slice(None)] * 2
                            b_ = [slice(None)] * 2
                            if off == 1:
                                a[ax] = slice(off, None)
                                b_[ax] = slice(0, -off)
                            else:
                                a[ax] = slice(0, off)
                                b_[ax] = slice(-off, None)
                            shifted = np.zeros_like(dil)
                            shifted[tuple(a)] = dil[tuple(b_)]
                            nxt |= shifted
                    dil = nxt

                cur = dil
                cur_lvl = lvl
                for lc in range(lvl - 2, self.base_depth - 1, -1):
                    h = 1 << (cur_lvl - lc)
                    coarse = cur.reshape(1 << lc, h, 1 << lc, h).any(axis=(1, 3))
                    g = self._occ.get(lc)
                    if g is not None:
                        cc = g[g >= 0]
                        if len(cc):
                            marked = coarse[self._cx[cc], self._cy[cc]]
                            bad = cc[marked & self._leaf[cc]]
                            if len(bad):
                                to_split.append(bad)
                    cur = coarse
                    cur_lvl = lc
            if not to_split:
                return
            # A cell can be marked from several level projections in one
            # round; deduplicate before splitting (splitting a cell twice
            # would orphan its first children).
            allbad = np.unique(np.concatenate(to_split))
            for lvl in np.unique(self._lvl[allbad]):
                self._split_cells(allbad[self._lvl[allbad] == lvl], positions)

    def _build_hash_index(self) -> None:
        n = self.n_cells
        self.hash_table = ElasticHashTable(capacity=max(16384, 4 * n), delta=0.05)
        self.cell_keys = []
        for c in range(n):
            key = morton_encode_box(int(self._lvl[c]), int(self._cx[c]),
                                    int(self._cy[c]))
            ok, _ = self.hash_table.insert(key, int(c))
            if ok:
                self.cell_keys.append(key)
        self._list3_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ utils

    def _lookup(self, lvl: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Occupancy lookup at `lvl`; -1 for out-of-bounds/unoccupied."""
        g = self._occ.get(lvl)
        n = 1 << lvl
        ix = np.asarray(ix, dtype=np.int64)
        iy = np.asarray(iy, dtype=np.int64)
        valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        out = np.full(len(ix), -1, dtype=np.int64)
        if g is not None and valid.any():
            out[valid] = g[ix[valid], iy[valid]]
        return out

    def _children_matrix(self) -> np.ndarray:
        """(n_cells, 4) child ids (-1 for unsplit)."""
        n = self.n_cells
        ch = np.full((n, 4), -1, dtype=np.int64)
        has = np.nonzero(self._chb[:n] >= 0)[0]
        if len(has):
            ch[has] = self._chb[has][:, None] + np.arange(4)[None, :]
        return ch

    # ------------------------------------------------------------------ main

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        p = self.p
        xmin, xmax = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
        ymin, ymax = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
        margin = max(1e-4, 0.02 * max(xmax - xmin, ymax - ymin, 1e-3))
        self.bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0

        self._build(positions, N)
        n_cells = self.n_cells
        lvl = self._lvl[:n_cells]
        cix_all = self._cx[:n_cells]
        ciy_all = self._cy[:n_cells]
        leaf = self._leaf[:n_cells]
        centers = self._cen[:n_cells]
        children = self._children_matrix()
        max_lvl = int(lvl.max())

        # leaf CSR over particles
        leaves = np.nonzero(leaf)[0]
        leaf_slot = np.full(n_cells, -1, dtype=np.int64)
        leaf_slot[leaves] = np.arange(len(leaves))
        leaf_of = self.pcell
        counts = np.bincount(leaf_slot[leaf_of], minlength=len(leaves))
        cell_start = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(counts, out=cell_start[1:])
        cell_particles = np.argsort(leaf_slot[leaf_of], kind="stable")

        m = np.zeros((n_cells, p + 1), dtype=np.complex128)
        lcl = np.zeros((n_cells, p + 1), dtype=np.complex128)

        # ---- P2M ------------------------------------------------------------
        z = positions[:, 0] + 1j * positions[:, 1]
        m[:, 0] = np.bincount(leaf_of, weights=charges, minlength=n_cells)
        dz_leaf = z - centers[leaf_of]
        dz_pow = np.ones(N, dtype=np.complex128)
        for k in range(1, p + 1):
            dz_pow *= dz_leaf
            w = charges * dz_pow / (-k)
            m[:, k] = (np.bincount(leaf_of, weights=w.real, minlength=n_cells)
                       + 1j * np.bincount(leaf_of, weights=w.imag,
                                          minlength=n_cells))

        # ---- M2M upward -----------------------------------------------------
        for l in range(max_lvl - 1, -1, -1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            parents = cells[~leaf[cells]]
            if len(parents) == 0:
                continue
            ch = children[parents]                    # (n_par, 4)
            a = m[ch]                                 # (n_par, 4, p+1)
            d = centers[ch] - centers[parents][:, None]
            b = _m2m_batch(a.reshape(-1, p + 1), d.reshape(-1), p)
            m[parents] = b.reshape(len(parents), 4, p + 1).sum(axis=1)

        # ---- downward: L2L + List-2 M2L + List-3/4 ---------------------------
        m2l_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        list3_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        for l in range(1, max_lvl + 1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            if len(cells) == 0:
                continue
            cix = cix_all[cells]
            ciy = ciy_all[cells]
            cpar = self._par[cells]

            # L2L from parent (parent local finalized at level l-1). Base
            # cells have no parent cell (the ancestors above base_depth do
            # not exist as cells; their far field enters via List-2 M2L at
            # the base level, the flat-scheme geometry).
            cpar = self._par[cells]
            has_par = cpar >= 0
            if has_par.any():
                hc = cells[has_par]
                hp = cpar[has_par]
                lcl[hc] += _l2l_batch(lcl[hp], centers[hc] - centers[hp], p)

            # List 2: children of the parent's colleagues that are not
            # adjacent. Ring test is parity-dependent, so it is applied per
            # cell rather than baked into the offset enumeration.
            ppx = cix >> 1
            ppy = ciy >> 1
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        continue
                    sx = cix + dx
                    sy = ciy + dy
                    src = self._lookup(l, sx, sy)
                    in_ring = (src >= 0) & \
                        (np.abs((sx >> 1) - ppx) <= 1) & \
                        (np.abs((sy >> 1) - ppy) <= 1)
                    if in_ring.any():
                        s = src[in_ring]
                        key = (l, dx, dy)
                        M = m2l_cache.get(key)
                        if M is None:
                            # delta = dst_center - src_center; the source sits
                            # at (+dx, +dy) relative to the target.
                            M = _m2l_matrix(
                                complex(-dx * Wx / (1 << l),
                                        -dy * Wy / (1 << l)), p)
                            m2l_cache[key] = M
                        lcl[cells[in_ring]] += m[s] @ M.T

            # List 3 (M2P at evaluation) + List 4 (P2L into the separated
            # child's local from the coarse leaf's particles). Child offsets
            # are expressed in level-(l+1) units relative to 2*cix (the
            # target box origin at l+1): adjacent children occupy
            # [-1, 2]^2; separated ones have at least one axis in {-2, +3}.
            # The colleague of the target is at cix + (vx2 >> 1)
            # (exact: 2*cix is even). List 4 uses true per-particle P2L --
            # M2L from the coarse leaf's multipole would converge at ratio
            # (sqrt(2)+sqrt(2)/2)/2.5 ~ 0.85 and lose ~an order of accuracy.
            leaf_mask = leaf[cells]
            for vx2 in range(-2, 4):
                for vy2 in range(-2, 4):
                    if -1 <= vx2 <= 2 and -1 <= vy2 <= 2:
                        continue  # adjacent child: belongs to List 1, not 3
                    coll = self._lookup(l, cix + (vx2 >> 1),
                                        ciy + (vy2 >> 1))
                    child = self._lookup(l + 1, 2 * cix + vx2, 2 * ciy + vy2)
                    ok = (coll >= 0) & (~leaf[coll.clip(0)]) & (child >= 0) \
                        & leaf_mask
                    if not ok.any():
                        continue
                    tgt = cells[ok]
                    src3 = child[ok]
                    list3_pairs.append((tgt, src3))
                    # List 4 P2L, vectorized over (leaf particle, dst cell)
                    slot_t = leaf_slot[tgt]
                    n_b = counts[slot_t]
                    T = int(n_b.sum())
                    if T == 0:
                        continue
                    reps = np.repeat(np.arange(len(tgt)), n_b)
                    prev = np.concatenate(([0], np.cumsum(n_b)[:-1]))
                    within = np.arange(T) - np.repeat(prev, n_b)
                    pids = cell_particles[cell_start[slot_t][reps] + within]
                    dcells = src3[reps]
                    dzz = centers[dcells] - z[pids]  # z0_D - z_i
                    w = 1.0 / dzz
                    wpow = np.empty((T, p + 1), dtype=np.complex128)
                    wpow[:, 0] = 1.0
                    for k in range(1, p + 1):
                        wpow[:, k] = wpow[:, k - 1] * w
                    qw = charges[pids][:, None] * wpow  # (T, p+1)
                    order = np.argsort(dcells, kind="stable")
                    d_sorted = dcells[order]
                    starts = np.concatenate(
                        ([0], np.nonzero(d_sorted[1:] != d_sorted[:-1])[0] + 1))
                    dsts = d_sorted[starts]
                    qw_s = qw[order]
                    # c_0 = sum q ln(z0 - z_i); no contribution from the
                    # w^0 column (classical P2L has no constant term).
                    log_term = np.add.reduceat(
                        (charges[pids] * np.log(dzz)).real[order], starts)
                    lcl[dsts, 0] += log_term
                    for kl in range(1, p + 1):
                        coef = ((-1.0) ** (kl - 1)) / kl
                        lcl[dsts, kl] += coef * (
                            np.add.reduceat(qw_s[:, kl].real, starts)
                            + 1j * np.add.reduceat(qw_s[:, kl].imag, starts))

        # ---- List 1 (adjacent leaves) for near-field P2P ----------------------
        l1_tgt_parts: List[np.ndarray] = []
        l1_src_parts: List[np.ndarray] = []
        l1_counts = np.zeros(len(leaves), dtype=np.int64)

        def add_l1(tgt: np.ndarray, src: np.ndarray) -> None:
            ok = counts[leaf_slot[src]] > 0  # drop empty neighbor leaves
            if not ok.any():
                return
            t = tgt[ok]
            s = src[ok]
            l1_tgt_parts.append(leaf_slot[t])
            l1_src_parts.append(s)
            l1_counts[:] += np.bincount(leaf_slot[t], minlength=len(leaves))

        # (a) each leaf's own cell (self pairs masked in the P2P kernel)
        add_l1(leaves, leaves)
        for l in range(0, max_lvl + 1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            if len(cells) == 0:
                continue
            cix = cix_all[cells]
            ciy = ciy_all[cells]
            leaf_mask = leaf[cells]

            # (b) same-level leaf colleagues (includes siblings)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    src = self._lookup(l, cix + dx, ciy + dy)
                    ok = (src >= 0) & leaf_mask & leaf[src.clip(0)]
                    if ok.any():
                        add_l1(cells[ok], src[ok])

            # (c) coarser adjacent leaves: 3x3 ring around the parent, with
            # the mixed-level touch test (a parent-colleague cell only
            # touches the child if the child's quadrant faces it).
            if l >= 1:
                px = cix >> 1
                py = ciy >> 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        sx = px + dx
                        sy = py + dy
                        src = self._lookup(l - 1, sx, sy)
                        touch = (2 * sx <= cix + 1) & (2 * sx + 2 >= cix) & \
                                (2 * sy <= ciy + 1) & (2 * sy + 2 >= ciy)
                        ok = (src >= 0) & touch & leaf_mask & leaf[src.clip(0)]
                        if ok.any():
                            add_l1(cells[ok], src[ok])

            # (d) adjacent children (qx,qy in [-1,2]^2) of split same-level
            # colleagues. The (0,0) colleague is the target itself (a leaf,
            # never split), so own-box children cannot leak in.
            for qx in range(-1, 3):
                for qy in range(-1, 3):
                    if 0 <= qx <= 1 and 0 <= qy <= 1:
                        continue  # own parent's quadrant block: tgt is a leaf
                    coll = self._lookup(l, cix + (qx >> 1), ciy + (qy >> 1))
                    child = self._lookup(l + 1, 2 * cix + qx, 2 * ciy + qy)
                    ok = ((coll >= 0) & (~leaf[coll.clip(0)]) & (child >= 0)
                          & leaf_mask & leaf[child.clip(0)])
                    if ok.any():
                        add_l1(cells[ok], child[ok])

        l1_start = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(l1_counts, out=l1_start[1:])
        l1_tgt = (np.concatenate(l1_tgt_parts) if l1_tgt_parts
                  else np.empty(0, dtype=np.int64))
        l1_src = (np.concatenate(l1_src_parts) if l1_src_parts
                  else np.empty(0, dtype=np.int64))
        order = np.argsort(l1_tgt, kind="stable")
        l1_src = l1_src[order]

        # per-leaf concatenated near-field source particles. `l1_src` is
        # sorted by target leaf slot, so the flat layout (leaf-major, then
        # entry order within the leaf) means destination offsets are simply
        # sequential.
        n_src_cells = counts[leaf_slot[l1_src]]
        flat_total = int(n_src_cells.sum())
        flat_sources = np.empty(flat_total, dtype=np.int64)
        if flat_total:
            entry_of_row = np.repeat(np.arange(len(l1_src)), n_src_cells)
            within_entry = np.arange(flat_total) - np.repeat(
                np.concatenate(([0], np.cumsum(n_src_cells)[:-1])), n_src_cells)
            src_pid = cell_particles[cell_start[leaf_slot[l1_src[entry_of_row]]]
                                     + within_entry]
            flat_sources[:] = src_pid  # dest == arange(flat_total)
        src_off = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(np.bincount(
            np.repeat(np.arange(len(leaves)), l1_counts.astype(np.int64)),
            weights=n_src_cells, minlength=len(leaves)), out=src_off[1:])

        # ---- evaluation -------------------------------------------------------
        potentials = np.zeros(N, dtype=np.float64)
        forces_x = np.zeros(N, dtype=np.float64)
        forces_y = np.zeros(N, dtype=np.float64)

        # L2P (+ forces), vectorized over all particles
        c_loc = lcl[leaf_of]
        dzp = z - centers[leaf_of]
        val = c_loc[:, 0].copy()
        deriv = np.zeros(N, dtype=np.complex128)
        zp = np.ones(N, dtype=np.complex128)
        for l in range(1, p + 1):
            deriv += l * c_loc[:, l] * zp
            zp *= dzp
            val += c_loc[:, l] * zp
        potentials += val.real
        if compute_forces:
            forces_x += -deriv.real
            forces_y += deriv.imag

        # M2P over List-3 pairs (leaf particles x separated source boxes)
        for tgt, src3 in list3_pairs:
            slot_t = leaf_slot[tgt]
            n_b = counts[slot_t]
            reps = np.repeat(np.arange(len(tgt)), n_b)
            prev = np.concatenate(([0], np.cumsum(n_b)[:-1]))
            within = np.arange(int(n_b.sum())) - np.repeat(prev, n_b)
            pids = cell_particles[cell_start[slot_t][reps] + within]
            m_src = m[src3][reps]
            dzs = z[pids] - centers[src3][reps]
            a0 = m_src[:, 0]
            dinv = 1.0 / dzs
            pot = a0 * np.log(dzs)
            dsum = a0 * dinv.copy()
            dinv_pow = dinv.copy()
            for k in range(1, p + 1):
                pot += m_src[:, k] * dinv_pow
                if compute_forces:
                    dsum -= k * m_src[:, k] * (dinv_pow * dinv)
                if k < p:
                    dinv_pow = dinv_pow * dinv
            np.add.at(potentials, pids, pot.real)
            if compute_forces:
                np.add.at(forces_x, pids, -dsum.real)
                np.add.at(forces_y, pids, dsum.imag)

        # near-field P2P per leaf block
        eps2 = self.softening * self.softening
        for s in range(len(leaves)):
            t_ids = cell_particles[cell_start[s]:cell_start[s + 1]]
            if len(t_ids) == 0:
                continue
            lo, hi = src_off[s], src_off[s + 1]
            if hi <= lo:
                continue
            s_ids = flat_sources[lo:hi]
            xt = positions[t_ids]
            xs = positions[s_ids]
            qs = charges[s_ids]
            ddx = xt[:, 0][:, None] - xs[None, :, 0]
            ddy = xt[:, 1][:, None] - xs[None, :, 1]
            r2 = ddx * ddx + ddy * ddy + eps2
            r2_safe = np.where(r2 < 1e-28, 1.0, r2)
            g = 0.5 * np.log(r2_safe)
            self_mask = t_ids[:, None] == s_ids[None, :]
            g = np.where(self_mask, 0.0, g)
            potentials[t_ids] += g @ qs
            if compute_forces:
                inv = np.where(self_mask, 0.0, 1.0 / r2_safe)
                forces_x[t_ids] -= (ddx * inv) @ qs
                forces_y[t_ids] -= (ddy * inv) @ qs

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials
