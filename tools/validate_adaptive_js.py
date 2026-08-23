"""Cross-validate index.html's JS adaptive-FMM metadata builder against the
direct O(N^2) sum, via the flat-schedule emulator in core.test_flat_adaptive_gpu.

The JS builder (buildAdaptiveMetadata in index.html) is executed by Node on a
synthetic scene (tools/emit_adaptive_meta.mjs slices the real shipped source
out of the page), its flat arrays are loaded here, wrapped as a
FlatAdaptiveMetadata, and evaluated with evaluate_flat_adaptive_emulated.

Run: python tools/validate_adaptive_js.py
"""
from __future__ import annotations

import json
import struct
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

    magic, N, nodeCount, numLevels, depth, leafCount = take_u32(6)
    assert magic == 0x4D504441, "bad magic"
    positions = take_f32(N * 2).reshape(N, 2).astype(np.float64)
    nodeCenterSize = take_f32(nodeCount * 4).reshape(nodeCount, 4)
    nodeParent = take_u32(nodeCount)
    nodeChildren = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    nodeFlags = take_u32(nodeCount)
    nodeParticleRange = take_u32(nodeCount * 2).reshape(nodeCount, 2)
    listOffsets = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    listCounts = take_u32(nodeCount * 4).reshape(nodeCount, 4)
    nListData = (len(raw) - off - 4 * (2 * N)) // 4
    listData = take_u32(nListData)
    leafForParticle = take_u32(N)
    particleIndices = take_u32(N)
    assert off == len(raw), f"trailing bytes: {len(raw) - off}"
    return dict(
        positions=positions, node_center_size=nodeCenterSize, node_parent=nodeParent,
        node_children=nodeChildren, node_flags=nodeFlags,
        node_particle_range=nodeParticleRange, list_offsets=listOffsets,
        list_counts=listCounts, list_data=listData,
        leaf_node_for_particle=leafForParticle, particle_indices=particleIndices,
        info=dict(N=N, nodeCount=nodeCount, numLevels=numLevels, depth=depth, leafCount=leafCount),
    )


def structural_checks(d):
    """Invariants the WGSL dispatch relies on (level-contiguity, parents,
    children, list bounds, leaf coverage)."""
    info = d["info"]
    n = info["nodeCount"]
    cs = d["node_center_size"]
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
    md = FlatAdaptiveMetadata(
        node_center_size=d["node_center_size"], node_parent=d["node_parent"],
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
    e_pot = exact_direct_nbody_2d(d["positions"], charges)
    e_fx, e_fy = exact_direct_nbody_forces_2d(d["positions"], charges)
    rel_pot = np.linalg.norm(pot - e_pot) / np.linalg.norm(e_pot)
    rel_f = np.linalg.norm(fx - e_fx) / np.linalg.norm(e_fx)
    return stats, errs, rel_pot, rel_f


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
        stats, errs, rel_pot, rel_f = run_scene(scene, N, depth, order)
        status = "PASS" if (not errs and rel_pot < pot_tol and rel_f < f_tol) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {scene:9s} N={N:5d} depth={depth} p={order}: "
              f"rel_pot={rel_pot:.2e} (gate {pot_tol:.0e}) rel_f={rel_f:.2e} (gate {f_tol:.0e}) "
              f"(nodes={stats['nodes']}, leaves={stats['leaves']}, maxLeaf={stats['maxLeaf']}, "
              f"listEntries={stats['listEntries']})")
        for e in errs[:10]:
            print(f"        structural: {e}")
    if not ok:
        sys.exit(1)
    print("All JS adaptive-builder cross-validations passed.")


if __name__ == "__main__":
    main()
