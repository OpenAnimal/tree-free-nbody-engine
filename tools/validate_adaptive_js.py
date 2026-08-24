"""Cross-validate index.html's JS adaptive-FMM metadata builder against the
direct O(N^2) sum, via the flat-schedule emulator in core.test_flat_adaptive_gpu.

The JS builder (buildAdaptiveMetadata in index.html) is executed by Node on a
synthetic scene (tools/emit_adaptive_meta.mjs slices the real shipped source
out of the page), its flat arrays are loaded here, wrapped as a
FlatAdaptiveMetadata, and evaluated with evaluate_flat_adaptive_emulated.

Round 13: the emitted binary also carries the MATERIALIZED far-field CSR
(per-leaf List-2 source lists across ancestor levels + the per-(level,
offset) M2L operator table). evaluate_flat_adaptive_materialized emulates the
far_gather/p2l GPU schedule (per-leaf flat gather through the operator
table + one-shot L2L folds of the ancestors' P2L locals) and is checked
three ways: the emitted operator table against the Python reference closed
form, the CSR contents against the List-2 sets of the ancestor chain, and
the full materialized potentials/forces against the direct O(N^2) sum at
the same tolerances as the legacy chain.

Run: python tools/validate_adaptive_js.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adaptive_gpu_metadata import FlatAdaptiveMetadata, INVALID  # noqa: E402
from core.adaptive_fmm import exact_direct_nbody_2d, exact_direct_nbody_forces_2d  # noqa: E402
from tests.core.test_flat_adaptive_gpu import evaluate_flat_adaptive_emulated  # noqa: E402


def load_binary(path: Path):
    raw = path.read_bytes()
    off = 0

    def take_u32(n):
        nonlocal off
        arr = np.frombuffer(raw, dtype=np.uint32, count=n, offset=off).copy()
        off += 4 * n
        return arr

    def take_f32(n):
        nonlocal off
        arr = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).copy()
        off += 4 * n
        return arr

    magic, N, nodeCount, numLevels, depth, leafCount, nFar = take_u32(7)
    assert magic == 0x4D504441, "bad magic"
    # listData length = total words minus every other section (the emitter
    # writes sections back to back; only listData is variably sized besides
    # farEntries, whose count rides in the header).
    total_words = len(raw) // 4
    list_data_words = (total_words - 7 - 2 * N - 4 * nodeCount - 2 * nodeCount
                       - 4 * nodeCount - 2 * nodeCount - 4 * nodeCount - 4 * nodeCount
                       - nodeCount - nodeCount - int(nFar) - 26950 - N - N)
    assert list_data_words >= 0, "binary layout mismatch (negative listData)"
    positions = take_f32(N * 2).reshape(N, 2).astype(np.float64)
    nodeCenterSize = take_f32(nodeCount * 4).reshape(nodeCount, 4)
    nodeMeta = take_u32(nodeCount * 2).reshape(nodeCount, 2)
    nodeChildren = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    nodeParticleRange = take_u32(nodeCount * 2).reshape(nodeCount, 2)
    listOffsets = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    listCounts = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    listData = take_u32(list_data_words)
    farStart = take_u32(nodeCount)
    farCount = take_u32(nodeCount)
    farEntries = take_u32(int(nFar))
    farOps = take_f32(26950)
    leafForParticle = take_u32(N)
    particleIndices = take_u32(N)
    assert off == len(raw), f"trailing bytes: {len(raw) - off}"
    node_parent = nodeMeta[:, 0].copy()
    node_flags = nodeMeta[:, 1].copy()
    return dict(
        positions=positions, nodeCenterSize=nodeCenterSize,
        # alias for the historical key used by tests/core/test_adaptive_wgsl_numeric.py
        node_center_size=nodeCenterSize,
        node_parent=node_parent, node_children=nodeChildren,
        node_flags=node_flags,
        node_particle_range=nodeParticleRange, list_offsets=listOffsets,
        list_counts=listCounts, list_data=listData,
        far_start=farStart, far_count=farCount, far_entries=farEntries,
        far_ops=farOps,
        leaf_node_for_particle=leafForParticle, particle_indices=particleIndices,
        info=dict(N=N, nodeCount=nodeCount, numLevels=numLevels, depth=depth, leafCount=leafCount),
    )


def structural_checks(d):
    """Invariants the WGSL dispatch relies on (level-contiguity, parents,
    children, list bounds, leaf coverage)."""
    info = d["info"]
    n = info["nodeCount"]
    cs = d["nodeCenterSize"]
    parent, children, flags = d["node_parent"], d["node_children"], d["node_flags"]
    ranges, l_off, l_cnt, l_data = (d["node_particle_range"], d["list_offsets"],
                                    d["list_counts"], d["list_data"])
    errs = []

    # Level-contiguous layout: nodeCenterSize.depth must be non-decreasing.
    levels = cs[:, 3].astype(int)
    if np.any(np.diff(levels) < 0):
        errs.append("node layout is not level-contiguous (depth decreases)")
    # Parent/child consistency + parent level = child level - 1.
    for node in range(n):
        p = int(parent[node])
        if p != INVALID:
            if p >= n:
                errs.append(f"node {node} parent OOB")
            elif levels[p] != levels[node] - 1:
                errs.append(f"node {node} level {levels[node]} parent level {levels[p]}")
        for slot in range(4):
            ch = int(children[node, slot])
            if ch != INVALID and levels[ch] != levels[node] + 1:
                errs.append(f"node {node} child slot {slot} level mismatch")
    # Particle ranges tile particleIndices exactly once.
    total = int(ranges[:, 1].sum())
    if total != info["N"]:
        errs.append(f"particle ranges sum {total} != N {info['N']}")
    if np.any(ranges[:, 0] + ranges[:, 1] > len(d["particle_indices"])):
        errs.append("particle range exceeds particle_indices")
    # List ranges within listData.
    if np.any(l_off + l_cnt > len(l_data)):
        errs.append("list range exceeds list_data")
    # Leaf coverage: deepest node on each particle's ancestor chain is terminal.
    leaf_for = d["leaf_node_for_particle"]
    if np.any(leaf_for == INVALID):
        errs.append("particles without leaf")
    for i in range(info["N"]):
        leaf = int(leaf_for[i])
        if not (flags[leaf] & 1):
            errs.append(f"particle {i} leaf {leaf} not terminal")
            break
    # leafForParticle must match the cell of the particle.
    for i in range(info["N"]):
        leaf = int(leaf_for[i])
        cx, cy, w = cs[leaf, 0], cs[leaf, 1], cs[leaf, 2]
        x, y = d["positions"][i]
        if not (cx - w / 2 <= x <= cx + w / 2 and cy - w / 2 <= y <= cy + w / 2):
            errs.append(f"particle {i} outside its leaf cell")
            break
    # Every leaf's List 1 includes itself.
    for node in range(n):
        if flags[node] & 1:
            s, c = int(l_off[node, 0]), int(l_cnt[node, 0])
            if c == 0 or node not in l_data[s:s + c].tolist():
                errs.append(f"leaf {node} List 1 missing self")
                break
    # List 3/4 reciprocity.
    for a in range(n):
        for b in d["list_data"][int(l_off[a, 2]):int(l_off[a, 2]) + int(l_cnt[a, 2])]:
            b = int(b)
            l4 = d["list_data"][int(l_off[b, 3]):int(l_off[b, 3]) + int(l_cnt[b, 3])].tolist()
            if a not in l4:
                errs.append(f"List3/4 reciprocity broken: {a} -> {b}")
                break
        if errs:
            break
    return errs


# =====================================================================
# Materialized far-field path (round 13)
# =====================================================================

def _reference_m2l_matrix(delta: complex, p: int) -> np.ndarray:
    """Dense (p+1, p+1) M2L operator for fixed delta = dst - src (the same
    closed form core/adaptive_fmm_fast._m2l_matrix and the WGSL m2l kernel
    implement):
        c_0 = a_0 ln(delta) + sum_{k>=1} a_k delta^{-k}
        c_l = a_0 (-1)^{l-1}/(l delta^l)
              + sum_{k>=1} (-1)^l binom(k+l-1, l) a_k delta^{-(k+l)}
    """
    M = np.zeros((p + 1, p + 1), dtype=np.complex128)
    dp = (1.0 / complex(delta)) ** np.arange(2 * p + 1)
    for l in range(p + 1):
        for k in range(p + 1):
            if l == 0 and k == 0:
                M[l, k] = np.log(complex(delta))
            elif l == 0:
                M[l, k] = dp[k]
            elif k == 0:
                M[l, k] = ((-1.0) ** (l - 1) / l) * dp[l]
            else:
                M[l, k] = ((-1.0) ** l) * math.comb(k + l - 1, l) * dp[k + l]
    return M


def _l2l_shift(c: np.ndarray, delta: complex, p: int) -> np.ndarray:
    """One-shot exact L2L recentering (same form as farShift in WGSL):
    c'_m = sum_{k>=m} binom(k, m) c_k delta^(k-m)."""
    out = np.zeros(p + 1, dtype=np.complex128)
    dp = np.empty(p + 1, dtype=np.complex128)
    dp[0] = 1.0
    for j in range(1, p + 1):
        dp[j] = dp[j - 1] * delta
    for m in range(p + 1):
        acc = 0j
        for k in range(m, p + 1):
            acc += math.comb(k, m) * c[k] * dp[k - m]
        out[m] = acc
    return out


def materialized_checks(d, p: int):
    """Validate the emitted CSR + farOps table against the Python reference.

    1. farOps rows (for every row referenced by an entry) match
       _reference_m2l_matrix(delta(l, offset)) to f32 tolerance.
    2. Each leaf's CSR multiset of (level, source) equals the List-2 sets of
       its ancestor chain, with the operator row implied by the cell offset.
    """
    cs = d["nodeCenterSize"]
    levels = cs[:, 3].astype(int)
    parent, flags = d["node_parent"], d["node_flags"]
    cell_ix = np.zeros(len(flags), dtype=np.int64)
    cell_iy = np.zeros(len(flags), dtype=np.int64)
    for node in range(len(flags)):
        w = cs[node, 2]
        cell_ix[node] = int(round(cs[node, 0] / w - 0.5))
        cell_iy[node] = int(round(cs[node, 1] / w - 0.5))
    errs = []
    far_ops = d["far_ops"]

    # 1) operator table cross-check on every referenced row.
    rows_used = set()
    for packed in d["far_entries"]:
        rows_used.add(int(packed) >> 22)
    worst = 0.0
    for row in rows_used:
        l = row // 49
        oy = (row % 49) // 7
        ox = row % 7
        w = 2.0 ** (-l)
        delta = complex((ox - 3) * w, (oy - 3) * w)
        ref = _reference_m2l_matrix(delta, 4)
        got = far_ops[row * 50:(row + 1) * 50].reshape(25, 2).astype(np.float64)
        got_c = got[:, 0] + 1j * got[:, 1]
        scale = max(1.0, float(np.abs(ref).max()))
        worst = max(worst, float(np.abs(got_c - ref.reshape(-1)).max()) / scale)
    if worst > 5e-6:
        errs.append(f"farOps table diverges from reference closed form (max rel {worst:.2e})")

    # 2) CSR content vs ancestor-chain List-2 sets.
    l_off, l_cnt, l_data = d["list_offsets"], d["list_counts"], d["list_data"]
    for t in range(len(flags)):
        if not (flags[t] & 1):
            if int(d["far_count"][t]) != 0:
                errs.append(f"non-leaf {t} has far CSR entries")
            continue
        expected = []  # (level, src, row)
        a = t
        for l in range(levels[t], 0, -1):
            for q in range(int(l_cnt[a, 1])):
                s = int(l_data[int(l_off[a, 1]) + q])
                row = (l * 49 + (cell_iy[a] - cell_iy[s] + 3) * 7
                       + (cell_ix[a] - cell_ix[s] + 3))
                expected.append((l, s, row))
            a = int(parent[a])
        start, cnt = int(d["far_start"][t]), int(d["far_count"][t])
        if start + cnt > len(d["far_entries"]):
            errs.append(f"leaf {t} CSR range exceeds far_entries")
            break
        got = []
        prev_level = 999
        for e in range(cnt):
            packed = int(d["far_entries"][start + e])
            src, row = packed & 0x3FFFFF, packed >> 22
            lvl = row // 49
            if lvl > prev_level:
                errs.append(f"leaf {t} CSR not grouped by descending level")
                break
            prev_level = lvl
            got.append((lvl, src, row))
        if got != expected:
            errs.append(f"leaf {t} CSR mismatch: {len(got)} entries vs expected {len(expected)}")
            break
    return errs, worst


def evaluate_flat_adaptive_materialized(
        positions: np.ndarray, d: dict, *, expansion_order: int = 2,
        charges: np.ndarray | None = None):
    """Execute the MATERIALIZED far-field GPU schedule (round 13) in NumPy.

    Mirrors the p2l + far_gather kernels of index.html / adaptive_fmm.wgsl:
      1. P2M at terminal leaves, M2M upward (shared with the legacy chain)
      2. p2l: per-node List-4 P2L into locals (full loops; at validator N
         the 128-sample budget never triggers, weight 1)
      3. far_gather: per leaf, flat CSR gather of List-2 sources through the
         farOps table + one-shot L2L folds of ancestors' P2L locals
      4. Particle evaluation: L2P + List 3 M2P + List 1 P2P (unchanged)
    """
    from core.adaptive_fmm import (
        p2m as adaptivefmm_p2m, m2m as adaptivefmm_m2m,
        p2l as adaptivefmm_p2l, l2p as adaptivefmm_l2p,
        l2p_force as adaptivefmm_l2p_force, m2p as adaptivefmm_m2p,
    )
    order = expansion_order
    n_nodes = d["info"]["nodeCount"]
    n_particles = len(positions)
    if charges is None:
        charges = np.ones(n_particles, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    cs = d["nodeCenterSize"]
    levels = cs[:, 3].astype(int)
    parent, flags = d["node_parent"], d["node_flags"]

    def center(node):
        return complex(float(cs[node, 0]), float(cs[node, 1]))

    multipoles = np.zeros((n_nodes, order + 1), dtype=np.complex128)
    for node in range(n_nodes):
        start, count = d["node_particle_range"][node]
        if count == 0:
            continue
        idxs = d["particle_indices"][int(start):int(start + count)]
        multipoles[node] = adaptivefmm_p2m(
            positions[idxs], charges[idxs], center(node), order)

    level_nodes: dict[int, list[int]] = {}
    for node in range(n_nodes):
        level_nodes.setdefault(int(levels[node]), []).append(node)
    max_depth = max(level_nodes)
    for lvl in range(max_depth, -1, -1):
        for node in level_nodes.get(lvl, []):
            if flags[node] & 1:
                continue
            acc = np.zeros(order + 1, dtype=np.complex128)
            for slot in range(4):
                ch = int(d["node_children"][node, slot])
                if ch == INVALID:
                    continue
                acc += adaptivefmm_m2m(multipoles[ch], center(ch), center(node), order)
            multipoles[node] = acc

    # p2l pass: each node's OWN List-4 P2L contribution only.
    p2l_locals = np.zeros((n_nodes, order + 1), dtype=np.complex128)
    for node in range(n_nodes):
        for src in d["list_data"][int(d["list_offsets"][node, 3]):
                                 int(d["list_offsets"][node, 3]) + int(d["list_counts"][node, 3])]:
            src = int(src)
            s_start, s_count = d["node_particle_range"][src]
            if s_count == 0:
                continue
            idxs = d["particle_indices"][int(s_start):int(s_start + s_count)]
            p2l_locals[node] += adaptivefmm_p2l(
                positions[idxs], charges[idxs], center(node), order)

    # far_gather pass: per leaf.
    locals_ = np.zeros((n_nodes, order + 1), dtype=np.complex128)
    far_ops = d["far_ops"]
    for t in range(n_nodes):
        if not (flags[t] & 1):
            continue
        t_center = center(t)
        t_level = int(levels[t])
        anc_idx = {}
        a = t
        for l in range(t_level, 0, -1):
            anc_idx[l] = a
            a = int(parent[a])
        acc = np.zeros(order + 1, dtype=np.complex128)
        start, cnt = int(d["far_start"][t]), int(d["far_count"][t])
        lvl_acc = np.zeros(order + 1, dtype=np.complex128)
        run_level = None
        for e in range(cnt):
            packed = int(d["far_entries"][start + e])
            src, row = packed & 0x3FFFFF, packed >> 22
            l = row // 49
            if l != run_level:
                if run_level is not None:
                    acc += _l2l_shift(lvl_acc, t_center - center(anc_idx[run_level]), order)
                lvl_acc = np.zeros(order + 1, dtype=np.complex128)
                run_level = l
            # 5x5 complex matvec through the emitted operator row.
            row_ops = far_ops[row * 50:(row + 1) * 50].reshape(25, 2).astype(np.float64)
            M = (row_ops[:, 0] + 1j * row_ops[:, 1]).reshape(5, 5)[:order + 1, :order + 1]
            lvl_acc += M @ multipoles[src]
        if run_level is not None:
            acc += _l2l_shift(lvl_acc, t_center - center(anc_idx[run_level]), order)
        # Fold ancestors' P2L locals (level t_level is t itself, zero shift).
        for l in range(1, t_level + 1):
            acc += _l2l_shift(p2l_locals[anc_idx[l]], t_center - center(anc_idx[l]), order)
        locals_[t] = acc

    potentials = np.zeros(n_particles, dtype=np.float64)
    fx = np.zeros(n_particles, dtype=np.float64)
    fy = np.zeros(n_particles, dtype=np.float64)
    for i in range(n_particles):
        pos_c = complex(positions[i, 0], positions[i, 1])
        target = int(d["leaf_node_for_particle"][i])
        tc = center(target)
        potentials[i] += adaptivefmm_l2p(locals_[target], pos_c, tc, order)
        lfx, lfy = adaptivefmm_l2p_force(locals_[target], pos_c, tc, order)
        fx[i] += lfx
        fy[i] += lfy
        for src in d["list_data"][int(d["list_offsets"][target, 2]):
                                 int(d["list_offsets"][target, 2]) + int(d["list_counts"][target, 2])]:
            src = int(src)
            sc = center(src)
            pot_d, deriv_d = adaptivefmm_m2p(multipoles[src], sc, pos_c, order)
            potentials[i] += pot_d
            fx[i] -= deriv_d.real
            fy[i] += deriv_d.imag
        for source_node in d["list_data"][int(d["list_offsets"][target, 0]):
                                          int(d["list_offsets"][target, 0]) + int(d["list_counts"][target, 0])]:
            source_node = int(source_node)
            s_start, s_count = d["node_particle_range"][source_node]
            if s_count == 0:
                continue
            for j in d["particle_indices"][int(s_start):int(s_start + s_count)]:
                j = int(j)
                if j == i:
                    continue
                diff = positions[i] - positions[j]
                r2 = max(float(diff[0] * diff[0] + diff[1] * diff[1]), 1e-12)
                qj = charges[j]
                potentials[i] += qj * 0.5 * math.log(r2)
                fx[i] -= qj * diff[0] / r2
                fy[i] -= qj * diff[1] / r2
    return potentials, fx, fy


def run_scene(scene, N, depth, order):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "meta.bin"
        proc = subprocess.run(
            ["node", str(ROOT / "tools" / "emit_adaptive_meta.mjs"), scene, str(N), str(depth), str(out)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node failed: {proc.stderr}\n{proc.stdout}")
        stats = json.loads(proc.stdout.strip().splitlines()[-1])
        d = load_binary(out)

    errs = structural_checks(d)
    mat_errs, ops_dev = materialized_checks(d, 4)
    md = FlatAdaptiveMetadata(
        node_center_size=d["nodeCenterSize"], node_parent=d["node_parent"],
        node_children=d["node_children"], node_particle_range=d["node_particle_range"],
        node_flags=d["node_flags"], particle_indices=d["particle_indices"],
        list_offsets=d["list_offsets"], list_counts=d["list_counts"],
        list_data=d["list_data"], leaf_node_for_particle=d["leaf_node_for_particle"],
        bounds=(0.0, 1.0, 0.0, 1.0),
    )
    rng = np.random.default_rng(3)
    charges = rng.uniform(0.5, 1.5, size=N)
    pot, fx, fy = evaluate_flat_adaptive_emulated(
        d["positions"], md, expansion_order=order, charges=charges)
    m_pot, m_fx, m_fy = evaluate_flat_adaptive_materialized(
        d["positions"], d, expansion_order=order, charges=charges)
    # The materialized path is the same math reordered: agreement with the
    # legacy emulator should hold near f64 rounding, far below the tolerance.
    chain_dev = float(np.linalg.norm(m_pot - pot) / max(1e-300, np.linalg.norm(pot)))
    chain_dev_f = float(np.linalg.norm(m_fx - fx) / max(1e-300, np.linalg.norm(fx)))
    e_pot = exact_direct_nbody_2d(d["positions"], charges)
    e_fx, e_fy = exact_direct_nbody_forces_2d(d["positions"], charges)
    rel_pot = np.linalg.norm(pot - e_pot) / np.linalg.norm(e_pot)
    rel_f = np.linalg.norm(fx - e_fx) / np.linalg.norm(e_fx)
    m_rel_pot = np.linalg.norm(m_pot - e_pot) / np.linalg.norm(e_pot)
    m_rel_f = np.linalg.norm(m_fx - e_fx) / np.linalg.norm(e_fx)
    return (stats, errs, mat_errs, ops_dev, chain_dev, chain_dev_f,
            rel_pot, rel_f, m_rel_pot, m_rel_f)


def main():
    ok = True
    # (scene, N, maxDepth, order, pot_tol, force_tol) — potentials converge
    # one order faster than forces in 2D logarithmic FMM; gates mirror the
    # Python reference's own cross-validation tolerances
    # (tests/core/test_flat_adaptive_gpu.py, tests/core/test_adaptive_fmm_cross_validation.py).
    scenes = [
        ("uniform", 800, 6, 2, 3e-3, 4e-2),
        ("uniform", 800, 6, 4, 5e-4, 5e-3),
        ("gaussian", 1500, 8, 4, 5e-4, 5e-3),
        ("clusters", 1600, 8, 4, 5e-4, 5e-3),
        ("hardedge", 1000, 9, 4, 5e-4, 5e-3),
        ("hardedge", 1000, 9, 2, 3e-3, 4e-2),
        ("single", 300, 10, 4, 5e-4, 5e-3),
    ]
    for scene, N, depth, order, pot_tol, f_tol in scenes:
        (stats, errs, mat_errs, ops_dev, chain_dev, chain_dev_f,
         rel_pot, rel_f, m_rel_pot, m_rel_f) = run_scene(scene, N, depth, order)
        status = "PASS" if (not errs and not mat_errs and rel_pot < pot_tol and rel_f < f_tol
                            and m_rel_pot < pot_tol and m_rel_f < f_tol) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {scene:9s} N={N:5d} depth={depth} p={order}: "
              f"chain rel_pot={rel_pot:.2e} (gate {pot_tol:.0e}) rel_f={rel_f:.2e} (gate {f_tol:.0e}) | "
              f"materialized rel_pot={m_rel_pot:.2e} rel_f={m_rel_f:.2e} | "
              f"mat-vs-chain pot={chain_dev:.1e} f={chain_dev_f:.1e} | opsDev={ops_dev:.1e} "
              f"(nodes={stats['nodes']}, leaves={stats['leaves']}, maxLeaf={stats['maxLeaf']}, "
              f"listEntries={stats['listEntries']}, farEntries={stats['farEntries']})")
        for e in errs[:10]:
            print(f"        structural: {e}")
        for e in mat_errs[:10]:
            print(f"        materialized: {e}")
    if not ok:
        sys.exit(1)
    print("All JS adaptive-builder cross-validations passed "
          "(legacy chain + materialized far path).")


if __name__ == "__main__":
    main()
