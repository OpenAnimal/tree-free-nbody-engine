#!/usr/bin/env python3
"""Cross-validation of the multi-level uniform-grid CGR88 FMM used by index.html.

The browser engine runs a fixed-depth uniform-lattice FMM whose zone
decomposition generalizes the original 2-level scheme to K levels:

    leaf Chebyshev offset <= 1            -> direct P2P (main shader, exact)
    leaf offset 2..3, parents adjacent    -> leaf multipole M2P ring (l2p)
    level-l Chebyshev >= 2, parents at
      level l+1 adjacent (l = 1..K-1)     -> M2L at level l
    top level (l = K, 4x4 grid) Chebyshev >= 2 -> M2L at the top level

which is the classic "interact at the finest well-separated level" rule.
This script is the mathematical ground truth for the WGSL port:

  Test A (partition): every ordered leaf-cell pair is covered by EXACTLY ONE
      zone, both under the min-level rule and under a literal simulation of
      each kernel's loop conditions (including the parity-dependent
      parent-adjacency checks). K=1 must reproduce the old 2-level scheme.
  Test B (numerics): the full pipeline (exact P2P + ring + multi-level far
      field) matches the brute-force softened pairwise sum
      F_i = sum_j -(p_i - p_j)/(|p_i - p_j|^2 + eps^2). Two error sources:
      order-dependent multipole truncation, and an order-independent
      softening-granularity floor eps^2/(2w)^2 (ring zones soften the cell
      center distance; the reference softens pointwise). Tolerances encode
      both.
  Test C (level scaling): same at B=5 and B=6 (more levels).

All formulas mirror the WGSL sources term-for-term (complex arithmetic;
softened denominator on monopole leading terms only).
"""

import sys
from math import comb

import numpy as np

EPS2 = 4e-5  # matches P2P_EPS2 in index.html
ORDER = 4


def level_res(bits: int, level: int) -> int:
    return 1 << (bits - level)


def level_count(bits: int) -> int:
    return max(1, bits - 2)


def cell_of(pts: np.ndarray, side: int) -> np.ndarray:
    cx = np.clip((pts[:, 0] * side).astype(int), 0, side - 1)
    cy = np.clip((pts[:, 1] * side).astype(int), 0, side - 1)
    return cy * side + cx


def cell_center(cell: int, side: int) -> complex:
    cx, cy = cell % side, cell // side
    return complex(cx + 0.5, cy + 0.5) / side


def chebyshev(a: int, b: int, side: int) -> int:
    ax, ay = a % side, a // side
    bx, by = b % side, b // side
    return max(abs(ax - bx), abs(ay - by))


# ----------------------------------------------------------------------------
# Zone classification. Parent adjacency is computed from the ACTUAL cell
# coordinates, because an offset of 3 maps to a parent offset of 1 or 2
# depending on parity.
# ----------------------------------------------------------------------------

def parents_adjacent_xy(ax: int, ay: int, bx: int, by: int) -> bool:
    """Parent-adjacency from actual leaf/level coordinates (parity-aware)."""
    return abs((ax >> 1) - (bx >> 1)) <= 1 and abs((ay >> 1) - (by >> 1)) <= 1


def zone_min_rule(ax: int, ay: int, bx: int, by: int, bits: int, k: int | None = None) -> str:
    """Finest-well-separated-level classification of a leaf-cell pair.

    A pair interacts at the finest level l whose cells are Chebyshev >= 2
    apart AND whose parents (at l+1) are adjacent — the top level has no
    adjacency requirement. Offset 4+ at a level implies parents >= 2 apart,
    so such pairs correctly defer to a coarser level (this is why the V-list
    loop only needs offsets -3..3).
    """
    dx, dy = abs(ax - bx), abs(ay - by)
    if max(dx, dy) <= 1:
        return "p2p"
    if max(dx, dy) <= 3 and parents_adjacent_xy(ax, ay, bx, by):
        return "ring"
    top = level_count(bits) if k is None else k
    for level in range(1, top + 1):
        lx, ly, rx, ry = ax >> level, ay >> level, bx >> level, by >> level
        if max(abs(lx - rx), abs(ly - ry)) >= 2:
            if level == top or parents_adjacent_xy(lx, ly, rx, ry):
                return f"m2l@{level}"
    raise AssertionError(f"pair ({ax},{ay})/({bx},{by}) never well separated")


def zone_kernel_simulation(ax: int, ay: int, bx: int, by: int, bits: int) -> set:
    """Which kernel loops cover the pair, mirroring the WGSL loop conditions."""
    zones = set()
    dx, dy = abs(ax - bx), abs(ay - by)
    # P2P: 3x3 leaf neighborhood
    if max(dx, dy) <= 1:
        zones.add("p2p")
    # Ring: offsets -3..3, Chebyshev 2..3, parity parent check
    if dx <= 3 and dy <= 3 and 2 <= max(dx, dy) <= 3 and parents_adjacent_xy(ax, ay, bx, by):
        zones.add("ring")
    # m2l@l: offsets -3..3 at level l, Chebyshev >= 2, parity parent check
    for level in range(1, level_count(bits) + 1):
        lx, ly = ax >> level, ay >> level
        rx, ry = bx >> level, by >> level
        ox, oy = abs(lx - rx), abs(ly - ry)
        if max(ox, oy) < 2:
            continue
        if level == level_count(bits):
            zones.add(f"m2l@{level}")  # top level: every cell at Chebyshev >= 2
        elif ox <= 3 and oy <= 3 and parents_adjacent_xy(lx, ly, rx, ry):
            zones.add(f"m2l@{level}")
    return zones


def test_partition() -> bool:
    ok = True
    for bits in (4, 5, 6):
        side = 1 << bits
        k = level_count(bits)
        coverage_errors = 0
        mismatch = 0
        for ay in range(side):
            for ax in range(side):
                for by in range(side):
                    for bx in range(side):
                        if (ax, ay) == (bx, by):
                            continue
                        zones = zone_kernel_simulation(ax, ay, bx, by, bits)
                        if len(zones) != 1:
                            coverage_errors += 1
                            if coverage_errors <= 3:
                                print(f"  PARTITION FAIL B={bits} ({ax},{ay})->({bx},{by}): {zones}")
                        expected = zone_min_rule(ax, ay, bx, by, bits)
                        if zones != {expected}:
                            mismatch += 1
        print(f"  B={bits} K={k}: {side*side*(side*side-1)} ordered pairs, "
              f"{coverage_errors} coverage errors, {mismatch} rule mismatches")
        ok = ok and coverage_errors == 0 and mismatch == 0

    # K=1 must reproduce the original 2-level scheme: ring for leaf 2..3 with
    # adjacent parents, parent-level M2L for parent offset >= 2.
    bits, k = 5, 1
    side = 1 << bits
    for ay in range(side):
        for ax in range(side):
            for by in range(side):
                for bx in range(side):
                    if (ax, ay) == (bx, by):
                        continue
                    got = zone_min_rule(ax, ay, bx, by, bits, k=1)
                    dx, dy = abs(ax - bx), abs(ay - by)
                    parent_off = max(abs((ax >> 1) - (bx >> 1)), abs((ay >> 1) - (by >> 1)))
                    if max(dx, dy) <= 1:
                        want = "p2p"
                    elif 2 <= max(dx, dy) <= 3 and parent_off <= 1:
                        want = "ring"
                    elif parent_off >= 2:
                        want = "m2l@1"
                    else:
                        want = "UNCOVERED"
                    if got != want:
                        print(f"  K=1 FAIL ({ax},{ay})->({bx},{by}): {got} != {want}")
                        return False
    print("  K=1 equivalence with the original 2-level scheme: OK")
    return ok


# ----------------------------------------------------------------------------
# The full numeric pipeline (mirrors the WGSL term-for-term).
# ----------------------------------------------------------------------------

def build_moments(z: np.ndarray, leaf: np.ndarray, bits: int) -> list:
    """P2M at leaves + M2M up the chain. Returns per-level moment arrays."""
    k = level_count(bits)
    moments = []
    side = level_res(bits, 0)
    ncells = side * side
    mom = np.zeros((ncells, ORDER + 1), dtype=np.complex128)
    members = [[] for _ in range(ncells)]
    for i, c in enumerate(leaf):
        members[c].append(i)
    for c in range(ncells):
        if not members[c]:
            continue
        center = cell_center(c, side)
        rho = z[members[c]] - center
        mom[c, 0] = len(rho)
        for korder in range(1, ORDER + 1):
            mom[c, korder] = -(rho ** korder / korder).sum()
    moments.append(mom)
    # M2M upward (CGR88 Thm 2.2), delta = child_center - parent_center
    for level in range(1, k + 1):
        side = level_res(bits, level)
        ncells = side * side
        mom = np.zeros((ncells, ORDER + 1), dtype=np.complex128)
        child_side = side * 2
        prev = moments[level - 1]
        for cy in range(side):
            for cx in range(side):
                pc = complex(cx + 0.5, cy + 0.5) / side
                for oy in range(2):
                    for ox in range(2):
                        child = (cy * 2 + oy) * child_side + cx * 2 + ox
                        a = prev[child]
                        cc = complex(cx * 2 + ox + 0.5, cy * 2 + oy + 0.5) / child_side
                        delta = cc - pc
                        for ko in range(ORDER + 1):
                            acc = 0j
                            if ko == 0:
                                acc = a[0]
                            else:
                                acc = -a[0] * delta ** ko / ko
                                for j in range(1, ko + 1):
                                    acc += a[j] * comb(ko - 1, j - 1) * delta ** (ko - j)
                            mom[cy * side + cx, ko] += acc
        moments.append(mom)
    return moments


def build_locals(moments: list, bits: int) -> list:
    """Per-level M2L over the V-lists + L2L downward chain."""
    k = level_count(bits)
    locals_per_level = [None] + [np.zeros((level_res(bits, l) ** 2, ORDER + 1), dtype=np.complex128)
                                 for l in range(1, k + 1)]
    for level in range(1, k + 1):
        side = level_res(bits, level)
        mom = moments[level]
        loc = locals_per_level[level]
        for ty in range(side):
            for tx in range(side):
                tc = complex(tx + 0.5, ty + 0.5) / side
                if mom[ty * side + tx, 0] == 0:
                    continue  # empty target; locals never read for empty cells
                for sy in range(side):
                    for sx in range(side):
                        ox, oy = abs(sx - tx), abs(sy - ty)
                        if max(ox, oy) < 2:
                            continue
                        if level < k:
                            if not (ox <= 3 and oy <= 3 and parents_adjacent_xy(tx, ty, sx, sy)):
                                continue  # handled at a finer/coarser level
                        a = mom[sy * side + sx]
                        if a[0] == 0:
                            continue
                        sc = complex(sx + 0.5, sy + 0.5) / side
                        delta = tc - sc
                        den_soft = abs(delta) ** 2 + EPS2
                        loc[ty * side + tx, 0] += a[0] * np.log(delta)
                        for ko in range(1, ORDER + 1):
                            loc[ty * side + tx, 0] += a[ko] / delta ** ko
                        for l in range(1, ORDER + 1):
                            # softened monopole: a0 * conj(delta)^l * (-1)^(l-1)/l / (|d|^2+eps^2)^l
                            loc[ty * side + tx, l] += (
                                a[0] * np.conj(delta) ** l * ((-1) ** (l - 1) / l) / den_soft ** l
                            )
                            for ko in range(1, ORDER + 1):
                                loc[ty * side + tx, l] += (
                                    a[ko] / delta ** (ko + l) * ((-1) ** l) * comb(ko + l - 1, l)
                                )
    # L2L downward: c'_l = sum_{j>=l} c_j C(j,l) d^(j-l), d = child_center - parent_center
    for level in range(k, 1, -1):
        side = level_res(bits, level)
        child_side = side * 2
        parent_loc = locals_per_level[level]
        child_loc = locals_per_level[level - 1]
        for cy in range(child_side):
            for cx in range(child_side):
                pc = complex((cx >> 1) + 0.5, (cy >> 1) + 0.5) / side
                cc = complex(cx + 0.5, cy + 0.5) / child_side
                d = cc - pc
                c = parent_loc[(cy >> 1) * side + (cx >> 1)]
                for l in range(ORDER + 1):
                    acc = 0j
                    for j in range(l, ORDER + 1):
                        acc += c[j] * comb(j, l) * d ** (j - l)
                    child_loc[cy * child_side + cx, l] += acc
    return locals_per_level


def fmm_deriv_at(z: np.ndarray, leaf: np.ndarray, bits: int, moments: list, locals_per_level: list) -> np.ndarray:
    """Ring M2P + level-1 local expansion evaluation -> complex 'deriv'.

    The WGSL stores force = (-Re(deriv), +Im(deriv)); for a real monopole this
    is -m*d/(|d|^2+eps^2), i.e. attractive softened gravity.
    """
    side = level_res(bits, 0)
    leaf_mom = moments[0]
    loc1 = locals_per_level[1]
    pside = side // 2
    deriv = np.zeros(len(z), dtype=np.complex128)
    for i, zi in enumerate(z):
        ax, ay = int(zi.real * side), int(zi.imag * side)
        ax = min(ax, side - 1)
        ay = min(ay, side - 1)
        # Level-1 local expansion of the containing parent cell
        px, py = ax >> 1, ay >> 1
        pc = complex(px + 0.5, py + 0.5) / pside
        dz = zi - pc
        c = loc1[py * pside + px]
        d = 0j
        for l in range(1, ORDER + 1):
            d += c[l] * l * dz ** (l - 1)
        # Leaf ring M2P: offsets 2..3 with adjacent parents
        tpx, tpy = ax >> 1, ay >> 1
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                cheb = max(abs(ox), abs(oy))
                if cheb <= 1 or cheb > 3:
                    continue
                nx, ny = ax + ox, ay + oy
                if not (0 <= nx < side and 0 <= ny < side):
                    continue
                if abs((nx >> 1) - tpx) > 1 or abs((ny >> 1) - tpy) > 1:
                    continue
                a = leaf_mom[ny * side + nx]
                if a[0] == 0:
                    continue
                delta = zi - cell_center(ny * side + nx, side)
                d += a[0] * np.conj(delta) / (abs(delta) ** 2 + EPS2)
                for ko in range(1, ORDER + 1):
                    d -= ko * a[ko] / delta ** (ko + 1)
        deriv[i] = d
    return deriv


def p2p_deriv(z: np.ndarray, leaf: np.ndarray, bits: int) -> np.ndarray:
    """Exact softened P2P over the 3x3 leaf neighborhood, as complex 'deriv'."""
    side = level_res(bits, 0)
    members = [[] for _ in range(side * side)]
    for i, c in enumerate(leaf):
        members[c].append(i)
    deriv = np.zeros(len(z), dtype=np.complex128)
    for i, zi in enumerate(z):
        ax, ay = int(zi.real * side), int(zi.imag * side)
        ax = min(ax, side - 1)
        ay = min(ay, side - 1)
        d = 0j
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                nx, ny = ax + ox, ay + oy
                if not (0 <= nx < side and 0 <= ny < side):
                    continue
                for j in members[ny * side + nx]:
                    if j == i:
                        continue
                    dz = zi - z[j]
                    d += np.conj(dz) / (abs(dz) ** 2 + EPS2)  # mass 1
        deriv[i] = d
    return deriv


def test_numerics(bits: int, z: np.ndarray) -> bool:
    leaf = cell_of(np.stack([z.real, z.imag], axis=1), level_res(bits, 0))
    moments = build_moments(z, leaf, bits)
    locals_per_level = build_locals(moments, bits)
    far = fmm_deriv_at(z, leaf, bits, moments, locals_per_level)
    near = p2p_deriv(z, leaf, bits)
    fmm_force = np.stack([-(far + near).real, (far + near).imag], axis=1)

    # Brute force: softened pairwise over ALL pairs (complex form: F = -sum dz/r^2)
    diff = z[:, None] - z[None, :]
    r2 = diff.real ** 2 + diff.imag ** 2 + EPS2
    np.fill_diagonal(r2, np.inf)
    ref_c = -(diff / r2).sum(axis=1)
    ref = np.stack([ref_c.real, ref_c.imag], axis=1)

    mag = np.hypot(ref[:, 0], ref[:, 1])
    # Robust relative error: particles whose reference force nearly cancels
    # carry large relative noise from truncation that is uniform in absolute
    # size, so the denominator is floored at a fraction of the RMS force.
    floor = 0.1 * np.sqrt((mag ** 2).mean())
    mask = mag > 1e-6
    rel = np.hypot(fmm_force[:, 0] - ref[:, 0], fmm_force[:, 1] - ref[:, 1])[mask] / np.maximum(mag, floor)[mask]
    # Two error sources vs the pointwise-softened reference:
    #   1. multipole truncation ~(rho/delta)^(order+1) — converges with order;
    #   2. softening granularity ~ eps^2/(2w)^2 at the ring boundary: the
    #      scheme softens the CELL-CENTER distance in ring/far zones (for
    #      continuity with the P2P zone), the reference softens pointwise.
    #      This floor is order-INDEPENDENT and equals eps2/(2w)^2 to within a
    #      small factor (verified: order-8 residual == the bound at B=4/5/6).
    side = level_res(bits, 0)
    soft_floor = EPS2 / (2.0 / side) ** 2
    tol_mean = 3e-3
    tol_max = 2.2 * soft_floor + 5e-3
    print(f"  B={bits} K={level_count(bits)}: N={len(z)} mean_rel={rel.mean():.2e} "
          f"max_rel={rel.max():.2e} (over {mask.sum()} particles, softening floor {soft_floor:.2e})")
    return rel.mean() < tol_mean and rel.max() < tol_max


def main() -> int:
    rng = np.random.default_rng(20260819)
    uniform = rng.uniform(0.0, 1.0, size=(800, 2))
    clump1 = rng.normal(0.05, size=(200, 2)) + [0.35, 0.6]
    clump2 = rng.normal(0.08, size=(200, 2)) + [0.7, 0.3]
    pts = np.clip(np.vstack([uniform, np.clip(clump1, 0, 1), np.clip(clump2, 0, 1)]), 0.0, 0.999999)
    z = (pts[:, 0] + 1j * pts[:, 1]).astype(np.complex128)

    print("Test A: zone partition (exactly-one-coverage + K=1 equivalence)")
    ok_a = test_partition()
    print("Test B/C: numerics vs brute force (order 4, eps^2 = 4e-5)")
    ok_b = all(test_numerics(b, z) for b in (4, 5, 6))

    if ok_a and ok_b:
        print("ALL TESTS PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
