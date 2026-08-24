"""On-GPU numeric check of the adaptive WGSL multipole math (round-9 residual
risk: "no on-GPU numeric check of the adaptive WGSL multipole math").

What was previously covered vs. what this file adds
---------------------------------------------------
Previously (T-E1):
  * tests/core/test_adaptive_wgsl_csr.py COMPILES the whole
    core/webgpu_kernels/adaptive_fmm.wgsl module (type-checks every entry
    point incl. the 16-storage-binding layout) but only DISPATCHES the four
    counting-sort CSR passes (clear_cells/count_cells/scan_cells/
    scatter_cells).  The multipole kernels (clear/p2m/m2m/m2l/l2l/l2p) were
    never executed on a GPU.
  * tests/core/test_webgpu_parity.py runs the FIXED-GRID kernel
    (tree_free_fmm.wgsl, host-built local expansions) numerically on GPU --
    not the adaptive multipole chain.
  * tools/validate_adaptive_js.py validates index.html's JS *metadata
    builder* against O(N^2) via the pure-Python emulator
    (evaluate_flat_adaptive_emulated) -- the WGSL multipole math itself was
    never evaluated headlessly.

This file dispatches the full adaptive multipole chain of the file kernel
core/webgpu_kernels/adaptive_fmm.wgsl on a native adapter:

    clear -> p2m -> m2m (per level, bottom-up)
          -> per level top-down: m2l (List 2 M2L + List 4 P2L) then l2l
          -> [counting-sort CSR passes when the near field is enabled]
          -> l2p (L2P + List 3 M2P + near-field P2P)

Why the file kernel and not index.html's inline wgslAdaptiveFmmSource: the
inline demo kernel's readc/writec take read_write storage-pointer fn
parameters, which Tint accepts but wgpu-py's naga backend rejects
("pointer of space Storage ... can't be passed into functions"; verified
2026-08-23 with wgpu 0.32.0).  The file kernel is the naga-compatible
T-E1 reference of the same math (per tools/check_wgsl_sync.py's allowlist)
and is exactly what test_adaptive_wgsl_csr.py compiles.

Coverage contract of the file kernel (its header comment): the near-field
P2P iterates a UNIFORM 3x3 grid overlay, so the FULL pipeline is exact only
for a uniform-depth tree.  Two complementary configurations are therefore
checked:

  1. MIXED-DEPTH adaptive tree (uniform scene, N=300, maxDepth=4; leaves at
     depths 2 and 3; nonempty Lists 3/4): run with params.zeroNearP2P = 1
     (far field only) and compare against a masked direct O(N^2) sum that
     excludes exactly the List-1 pairs.  This exercises P2M, M2M, M2L,
     List-4 P2L, L2L, L2P and List-3 M2P on a genuinely occupancy-adaptive
     tree (mixed-size cells) -- including the adaptive-only translation
     paths that a uniform tree leaves empty.
  2. UNIFORM-DEPTH tree (uniform scene, N=512, maxDepth=3; all 64 leaves at
     depth 3; Lists 3/4 empty by construction): run the FULL pipeline
     (zeroNearP2P = 0, P2P overlay grid = 8x8 = leaf grid) and compare
     against (a) direct O(N^2) summation, (a') a hybrid direct sum with the
     kernel's near-field softening (eps^2 = 4e-5) applied to the same 3x3
     pair set, (b) core.adaptive_fmm.AdaptiveFMM (CGR88 f64 reference),
     and the exact-schedule f64 emulator
     (tests.core.test_flat_adaptive_gpu.evaluate_flat_adaptive_emulated).

Metadata comes from index.html's real builder via `node
tools/emit_adaptive_meta.mjs` (slices buildAdaptiveMetadata verbatim out of
the page), reusing tools/validate_adaptive_js.py's loader and structural
checks.

Tolerances: the far-field and same-basis full-pipeline gates (5e-4 potential
/ 5e-3 force at order 4) mirror tools/validate_adaptive_js.py and the
flat-GPU emulator gates in tests/core/test_flat_adaptive_gpu.py (the Python
reference's own cross-validation tolerances).  Comparisons against
UNSOFTENED references (direct O(N^2), AdaptiveFMM, the emulator) are gated
loosely because the WGSL near field applies eps^2 = 4e-5 softening those
references lack; that softening bias is measured every run (hybrid-vs-
direct, no GPU involved) and asserted inside the loose gates, so a WGSL
regression still trips the tight same-basis gate while the loose gates only
ever absorb the documented softening.

DISPATCH PROTOCOL NOTE (found while building this gate): the adaptive
kernels address nodes as `levelBase + id.x` and bound only
`node < nodeCount` -- there is no per-dispatch levelCount guard, so a
caller that launches ceil(levelCount/64) workgroups of 64 threads (exactly
what index.html's browser pipeline does per level) lets surplus threads
spill into the NEXT level's nodes; m2l/l2l accumulate read-modify-write, so
spilled-onto nodes receive 2-3x their M2L/L2L contributions (measured: a
level-2 node's local expansion came back exactly 2.00x the reference, far
field ~0.7-0.98 rel-L2 error).  m2m is immune (pure write).  This harness
pads every level to a multiple of the workgroup size with inert terminal
nodes (pad_levels_to_workgroup) so the ceil dispatch covers EXACTLY
levelCount threads, satisfying the kernels' documented contract without
modifying any shader.  See the final report: the browser-side dispatcher
needs the same treatment or a kernel-side `id.x >= levelCount` guard.

Skips (pytest.skip) when wgpu is not importable, no adapter is found, or
node is not available.

Run standalone:  python -X utf8 -m tests.core.test_adaptive_wgsl_numeric
Run under pytest: python -m pytest tests/core/test_adaptive_wgsl_numeric.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.validate_adaptive_js import load_binary, structural_checks  # noqa: E402

NEAR_SOFTENING_EPS2 = 4e-5  # file-kernel l2p near-field: r2 = dot(d,d) + 4e-5

# (scene, N, maxDepth, order) -- both N in the low hundreds, 2D unit box.
CFG_MIXED = ("uniform", 300, 4, 4)    # occupancy-adaptive: leaves at depths 2..3
CFG_UNIFORM = ("uniform", 512, 3, 4)  # uniform-depth: all 64 leaves at depth 3

# Far-field / same-basis gates (5e-4 potential / 5e-3 force at order 4)
# mirror tools/validate_adaptive_js.py and the flat-GPU emulator gates in
# tests/core/test_flat_adaptive_gpu.py (the Python reference's own
# cross-validation tolerances).
FAR_POT_TOL = 5e-4
FAR_FORCE_TOL = 5e-3
# Comparisons against UNSOFTENED references (plain direct O(N^2),
# AdaptiveFMM, the emulator) are dominated by the WGSL near-field softening
# eps^2 = 4e-5 that those references do not apply.  The softening bias is
# measured explicitly every run (hybrid-vs-direct below) and asserted inside
# these gates, so the loose gates cannot silently absorb a WGSL regression:
# any WGSL-side error is separately pinned by the tight hybrid gate.
FULL_POT_TOL = 2e-3
FULL_FORCE_TOL = 2e-1


# =====================================================================
# Metadata (reuse the JS builder through the existing emit harness)
# =====================================================================

def build_scene_metadata(scene: str, n: int, depth: int):
    """Run node tools/emit_adaptive_meta.mjs and load the flat metadata.
    Returns (metadata dict, emit stats json)."""
    if shutil.which("node") is None:
        pytest.skip("node is not available to run tools/emit_adaptive_meta.mjs")
    root = Path(_REPO_ROOT)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "meta.bin"
        proc = subprocess.run(
            ["node", str(root / "tools" / "emit_adaptive_meta.mjs"),
             scene, str(n), str(depth), str(out)],
            capture_output=True, text=True, cwd=str(root), timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node failed: {proc.stderr}\n{proc.stdout}")
        stats = json.loads(proc.stdout.strip().splitlines()[-1])
        d = load_binary(out)
    errs = structural_checks(d)
    assert not errs, f"adaptive metadata structural errors: {errs[:5]}"
    return d, stats


def level_table(d: dict):
    """Reconstruct (present_levels, levelStart, levelCount) from the flat
    nodeCenterSize.depth column (nodes are level-contiguous)."""
    levels = d["node_center_size"][:, 3].astype(int)
    present, starts, counts = [], [], []
    for lvl in np.unique(levels):
        idx = np.nonzero(levels == lvl)[0]
        present.append(int(lvl))
        starts.append(int(idx[0]))
        counts.append(int(len(idx)))
    return present, starts, counts


def pad_levels_to_workgroup(d: dict, wg: int = 64):
    """Return (padded metadata dict, level table) with every level's node
    count rounded up to a multiple of the per-level workgroup size.

    WHY: the adaptive kernels address nodes as `levelBase + id.x` with the
    ONLY bound being `node < nodeCount` -- there is no per-dispatch
    `levelCount` guard.  A dispatcher that launches ceil(levelCount/64)
    workgroups of 64 threads (exactly what index.html's browser pipeline
    does at its per-level m2m/l2l/m2l dispatches) lets the surplus threads
    spill into the NEXT level's nodes: m2l/l2l use read-modify-write
    accumulation, so those nodes receive 2-3x their M2L/L2L contribution
    (verified numerically: a level-2 node's local came back exactly 2x the
    reference).  m2m is immune (pure write, recomputes the same value).

    Padding each level to a multiple of the workgroup size makes the ceil
    dispatch cover EXACTLY levelCount threads, satisfying the kernels'
    documented dispatch contract without modifying any shader.  Pad nodes
    are inert: terminal, zero particles, empty lists, INVALID parent, and
    nothing references them (no list entries, no leafForParticle hits).
    """
    cs = d["node_center_size"]
    parent, children, flags = (d["node_parent"], d["node_children"],
                               d["node_flags"])
    rng_arr, l_off, l_cnt, l_data = (d["node_particle_range"],
                                     d["list_offsets"], d["list_counts"],
                                     d["list_data"])
    leaf, pidx = d["leaf_node_for_particle"], d["particle_indices"]
    n = int(d["info"]["N"])
    node_count = int(d["info"]["nodeCount"])
    present, starts, counts = level_table(d)

    new_counts = [((c + wg - 1) // wg) * wg for c in counts]
    new_starts, total = [], 0
    for c in new_counts:
        new_starts.append(total)
        total += c

    o2n = np.full(node_count, 0xFFFFFFFF, dtype=np.uint32)
    cs2 = np.zeros((total, 4), dtype=np.float32)
    parent2 = np.full(total, 0xFFFFFFFF, dtype=np.uint32)
    children2 = np.full((total, 4), 0xFFFFFFFF, dtype=np.uint32)
    flags2 = np.zeros(total, dtype=np.uint32)
    rng2 = np.zeros((total, 2), dtype=np.uint32)
    l_off2 = np.zeros((total, 4), dtype=np.uint32)
    l_cnt2 = np.zeros((total, 4), dtype=np.uint32)
    for i, lvl in enumerate(present):
        s, c, ns = starts[i], counts[i], new_starts[i]
        cs2[ns:ns + c] = cs[s:s + c]
        parent2[ns:ns + c] = parent[s:s + c]
        children2[ns:ns + c] = children[s:s + c]
        flags2[ns:ns + c] = flags[s:s + c]
        rng2[ns:ns + c] = rng_arr[s:s + c]
        l_off2[ns:ns + c] = l_off[s:s + c]
        l_cnt2[ns:ns + c] = l_cnt[s:s + c]
        o2n[s:s + c] = np.arange(ns, ns + c, dtype=np.uint32)
        # Pad slots for this level: inert terminal nodes.
        for j in range(ns + c, ns + new_counts[i]):
            cs2[j] = (0.5, 0.5, 1.0, lvl)
            flags2[j] = 1  # terminal
            rng2[j] = (n, 0)
    # Remap every node-index field through o2n (INVALID passes through).
    def remap(arr):
        out = arr.copy()
        valid = arr != 0xFFFFFFFF
        out[valid] = o2n[arr[valid].astype(np.int64)]
        return out
    parent2_v = remap(parent2)
    children2_v = remap(children2.reshape(-1)).reshape(total, 4)
    l_data2 = remap(l_data)
    leaf2 = remap(leaf)

    # Round 13 materialized far CSR: pad node-indexed arrays, remap entry
    # source ids. Node order is preserved (padding only inserts inert nodes
    # between levels), so per-leaf start offsets are unchanged.
    far_s2 = np.zeros(total, dtype=np.uint32)
    far_c2 = np.zeros(total, dtype=np.uint32)
    fe = d["far_entries"].astype(np.uint64)
    fe_src = (fe & np.uint64(0x3FFFFF)).astype(np.uint32)
    fe_row = (fe >> np.uint64(22)).astype(np.uint32)
    far_e2 = (remap(fe_src).astype(np.uint64)
              | (fe_row.astype(np.uint64) << np.uint64(22))).astype(np.uint32)
    for old in range(node_count):
        new = int(o2n[old])
        if new != 0xFFFFFFFF:
            far_s2[new] = d["far_start"][old]
            far_c2[new] = d["far_count"][old]

    padded = dict(d)
    padded.update(
        node_center_size=cs2, node_parent=parent2_v, node_children=children2_v,
        node_flags=flags2, node_particle_range=rng2, list_offsets=l_off2,
        list_counts=l_cnt2, list_data=l_data2, leaf_node_for_particle=leaf2,
        particle_indices=pidx, far_start=far_s2, far_count=far_c2,
        far_entries=far_e2,
        info=dict(d["info"], nodeCount=np.uint32(total)),
    )
    return padded, (present, new_starts, new_counts)


# =====================================================================
# GPU harness (buffer conventions follow tests/core/test_adaptive_wgsl_csr.py)
# =====================================================================

def _device_or_skip():
    try:
        import wgpu
    except ImportError:
        pytest.skip("wgpu not installed")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("no WebGPU adapter")
    device = adapter.request_device_sync(required_limits={})
    return wgpu, device


def run_adaptive_wgsl(d: dict, charges: np.ndarray, order: int,
                      zero_near_p2p: bool, grid_dim: int,
                      materialized: bool = False):
    """Dispatch the adaptive multipole chain of adaptive_fmm.wgsl.

    Returns (pot, fx, fy) as float64 arrays of length N read back from the
    fields buffer (fx = .x, fy = .y, potential = .z).
    """
    wgpu, device = _device_or_skip()
    from core.webgpu_kernels.webgpu_fmm_runner import get_adaptive_fmm_wgsl_source

    n = int(d["info"]["N"])
    # Pad levels to workgroup-size multiples so per-level ceil dispatches
    # cover exactly levelCount threads (see pad_levels_to_workgroup).
    pd, (present, starts, counts) = pad_levels_to_workgroup(d)
    node_count = int(pd["info"]["nodeCount"])
    module = device.create_shader_module(code=get_adaptive_fmm_wgsl_source())

    nc = grid_dim * grid_dim

    pos = d["positions"].astype(np.float32)
    particles = np.zeros((n, 4), dtype=np.float32)
    particles[:, 0] = pos[:, 0]
    particles[:, 1] = pos[:, 1]

    node_meta = np.zeros((node_count, 2), dtype=np.uint32)
    node_meta[:, 0] = pd["node_parent"]
    node_meta[:, 1] = pd["node_flags"]

    # GridParams { gridDim: u32, _pad: u32, gridOrigin: vec2, cellSize: f32 }
    # (cellSize is u32 word 4 / byte 16, after the vec2 origin at bytes 8..15;
    # same packing as tests/core/test_adaptive_wgsl_csr.py)
    gp = np.zeros(6, dtype=np.uint32)
    gp[0] = grid_dim
    gp.view(np.float32)[4] = np.float32(1.0 / grid_dim)

    SU, CD, CS = (wgpu.BufferUsage.STORAGE, wgpu.BufferUsage.COPY_DST,
                  wgpu.BufferUsage.COPY_SRC)

    def mk(data, usage=SU):
        buf = device.create_buffer(size=data.nbytes, usage=usage | CD | CS)
        device.queue.write_buffer(buf, 0, np.ascontiguousarray(data).tobytes())
        return buf

    def mkraw(nbytes, usage=SU):
        return device.create_buffer(size=nbytes, usage=usage | CS)

    params_buf = mkraw(32, wgpu.BufferUsage.UNIFORM | CD)
    device.queue.write_buffer(params_buf, 0, np.zeros(8, np.uint32).tobytes())

    far_csr = None
    n_far = int(pd["far_count"].sum())
    if materialized:
        # Packed farCSR u32 buffer: [2*nodeCount interleaved start/count |
        # entries | farOps f32 bits] — the same layout index.html uploads.
        far_csr = np.zeros(2 * node_count + n_far + 26950, dtype=np.uint32)
        far_csr[0::2][:node_count] = pd["far_start"]
        far_csr[1::2][:node_count] = pd["far_count"]
        far_csr[2 * node_count:2 * node_count + n_far] = pd["far_entries"]
        far_csr[2 * node_count + n_far:].view(np.float32)[:] = d["far_ops"]

    bufs = {
        0: mk(particles),                       # particles
        1: mk(pd["node_center_size"]),          # nodeCenterSize
        2: mk(pd["node_particle_range"]),       # nodeParticleRange
        3: mk(pd["leaf_node_for_particle"]),    # leafForParticle
        4: mk(pd["particle_indices"]),          # particleIndices
        5: mk(pd["list_offsets"]),              # listOffsets
        6: mk(pd["list_counts"]),               # listCounts
        7: mk(pd["list_data"]),                 # listData
        8: mkraw(node_count * 48),             # multipoles (3 vec4/node)
        9: mkraw(node_count * 48),             # locals (3 vec4/node)
        10: mkraw(n * 16),                      # fields
        11: params_buf,                         # params (uniform)
        12: mk(node_meta),                      # nodeMeta (parent|flags)
        13: mk(pd["node_children"]),            # nodeChildren
        15: mk(charges.astype(np.float32)),     # charges
        16: mk(gp, wgpu.BufferUsage.UNIFORM),   # gridParams
        17: mkraw(3 * nc * 4),                  # cellArrays count|cursor|start
        20: mkraw(n * 4),                       # sortedIndex
    }
    if materialized:
        bufs[14] = mk(far_csr)                  # farCSR (round 13)
    binding_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20]
    if materialized:
        binding_ids.append(14)
    ro = {"type": "read-only-storage"}
    rw = {"type": "storage"}
    types = [ro, ro, ro, ro, ro, ro, ro, ro, rw, rw, rw,
             {"type": "uniform"}, ro, ro, ro, {"type": "uniform"}, rw, rw]
    if materialized:
        types.append(ro)
    COMP = wgpu.ShaderStage.COMPUTE
    bgl = device.create_bind_group_layout(entries=[
        {"binding": bid, "visibility": COMP, "buffer": t}
        for bid, t in zip(binding_ids, types)])
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[{"binding": bid, "resource": {"buffer": b, "offset": 0}}
                 for bid, b in zip(binding_ids, [bufs[i] for i in binding_ids])])

    pipelines = {}

    def dispatch(entry, n_workgroups, level_base=0, level_count=0):
        if entry not in pipelines:
            pipelines[entry] = device.create_compute_pipeline(
                layout=layout,
                compute={"module": module, "entry_point": entry})
        # FmmParams: numParticles, nodeCount, expansionOrder, levelCount,
        #            levelBase, zeroNearP2P, _pad1, _pad2  (8 u32 = 32 B)
        params = np.zeros(8, dtype=np.uint32)
        params[0] = n
        params[1] = node_count
        params[2] = order
        params[3] = level_count
        params[4] = level_base
        params[5] = 1 if zero_near_p2p else 0
        params[6] = n_far  # farEntryCount (farOps tail base in farCSR)
        device.queue.write_buffer(params_buf, 0, params.tobytes())
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipelines[entry])
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(n_workgroups, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])

    def readback(buf, nbytes, dtype):
        st = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        enc = device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, st, 0, nbytes)
        device.queue.submit([enc.finish()])
        st.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(st.read_mapped(), dtype=dtype).copy()
        st.unmap()
        st.destroy()
        return out

    ceil256 = lambda m: (int(m) + 255) // 256
    ceil64 = lambda m: (int(m) + 63) // 64

    # 1. clear coefficient buffers, 2. leaf P2M, 3. M2M bottom-up per level.
    dispatch("clear", ceil256(node_count * 3))
    dispatch("p2m", ceil64(node_count))
    for i in range(len(present) - 2, -1, -1):
        dispatch("m2m", ceil64(counts[i]), starts[i], counts[i])
    # 4. Downward far field: the legacy per-level chain (M2L List 2 +
    #    List 4 P2L, then L2L), or the round-13 materialized path (whole-range
    #    p2l for the List-4 parts + far_gather's per-leaf CSR/operator-table
    #    gather) when materialized=True.
    if materialized:
        dispatch("p2l", ceil64(node_count))
        dispatch("far_gather", ceil64(node_count))
    else:
        for i in range(1, len(present)):
            dispatch("m2l", ceil64(counts[i]), starts[i], counts[i])
            dispatch("l2l", ceil64(counts[i]), starts[i], counts[i])
    # 5. counting-sort CSR cell lists for the near-field P2P.
    if not zero_near_p2p:
        dispatch("clear_cells", ceil256(nc))
        dispatch("count_cells", ceil256(n))
        dispatch("scan_cells", 1)
        dispatch("scatter_cells", ceil256(n))
    # 6. particle evaluation.
    dispatch("l2p", ceil256(n))

    fields = readback(bufs[10], n * 16, np.float32).reshape(n, 4)
    for b in bufs.values():
        b.destroy()
    return (fields[:, 2].astype(np.float64),
            fields[:, 0].astype(np.float64),
            fields[:, 1].astype(np.float64))


# =====================================================================
# References
# =====================================================================

def _near_leaf_matrix(d: dict) -> np.ndarray:
    """node x node bool: near[T, S] iff S is in List 1 (adjacent leaves,
    incl. self) of leaf T."""
    nn = int(d["info"]["nodeCount"])
    near = np.zeros((nn, nn), dtype=bool)
    l_off, l_cnt, l_data = d["list_offsets"], d["list_counts"], d["list_data"]
    for node in range(nn):
        s, c = int(l_off[node, 0]), int(l_cnt[node, 0])
        for src in l_data[s:s + c]:
            near[node, int(src)] = True
    return near


def direct_far_field(d: dict, charges: np.ndarray):
    """Exact f64 direct sum over every pair EXCEPT List-1 (near) pairs --
    the exact coverage of the kernel's zeroNearP2P=1 far-field output."""
    pos = d["positions"]
    n = len(pos)
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    r2 = dx * dx + dy * dy
    np.fill_diagonal(r2, 1.0)
    leaf = d["leaf_node_for_particle"].astype(np.int64)
    near = _near_leaf_matrix(d)[leaf[:, None], leaf[None, :]]
    near[np.arange(n), np.arange(n)] = False  # self is not a far pair
    q = charges[None, :]
    pot_ij = q * 0.5 * np.log(r2)
    inv_r2 = 1.0 / r2
    fx_ij = -q * dx * inv_r2
    fy_ij = -q * dy * inv_r2
    for arr in (pot_ij, fx_ij, fy_ij):
        arr[near] = 0.0
    np.fill_diagonal(pot_ij, 0.0)
    np.fill_diagonal(fx_ij, 0.0)
    np.fill_diagonal(fy_ij, 0.0)
    return pot_ij.sum(axis=1), fx_ij.sum(axis=1), fy_ij.sum(axis=1)


def direct_hybrid_softened(d: dict, charges: np.ndarray, grid_dim: int):
    """Direct O(N^2) with the kernel's near-field treatment: pairs whose
    uniform-grid overlay cells lie within the 3x3 neighborhood get
    r2 -> r2 + 4e-5; all other pairs are exact (matching the WGSL full
    pipeline's split between CSR P2P and the multipole far field)."""
    pos = d["positions"]
    n = len(pos)
    cx = np.clip((pos[:, 0] * grid_dim).astype(np.int64), 0, grid_dim - 1)
    cy = np.clip((pos[:, 1] * grid_dim).astype(np.int64), 0, grid_dim - 1)
    near = (np.abs(cx[:, None] - cx[None, :]) <= 1) & \
           (np.abs(cy[:, None] - cy[None, :]) <= 1)
    near[np.arange(n), np.arange(n)] = False
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    r2 = dx * dx + dy * dy
    r2s = r2 + NEAR_SOFTENING_EPS2
    np.fill_diagonal(r2, 1.0)
    np.fill_diagonal(r2s, 1.0)
    q = charges[None, :]
    pot = np.where(near, q * 0.5 * np.log(r2s), q * 0.5 * np.log(r2))
    inv = np.where(near, 1.0 / r2s, 1.0 / r2)
    fx = -q * dx * inv
    fy = -q * dy * inv
    for arr in (pot, fx, fy):
        np.fill_diagonal(arr, 0.0)
    return pot.sum(axis=1), fx.sum(axis=1), fy.sum(axis=1)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(1e-300, np.linalg.norm(b)))


def _max_rel(a, b):
    denom = np.maximum(np.abs(b), 1e-3 * float(np.median(np.abs(b))))
    return float(np.max(np.abs(a - b) / denom))


def _report(label, gpu, ref, pot_tol, force_tol):
    pot, fx, fy = gpu
    rp, rfx, rfy = ref
    e_pot = _rel_l2(pot, rp)
    f_gpu = np.stack([fx, fy])
    f_ref = np.stack([rfx, rfy])
    e_f = float(np.linalg.norm(f_gpu - f_ref) /
                max(1e-300, np.linalg.norm(f_ref)))
    print(f"    {label}:")
    print(f"      rel-L2  pot={e_pot:.3e} force={e_f:.3e} "
          f"(gates {pot_tol:.0e} / {force_tol:.0e})")
    print(f"      max-rel pot={_max_rel(pot, rp):.3e} fx={_max_rel(fx, rfx):.3e} "
          f"fy={_max_rel(fy, rfy):.3e}")
    assert e_pot < pot_tol, f"{label}: pot rel-L2 {e_pot:.3e} >= {pot_tol:.0e}"
    assert e_f < force_tol, f"{label}: force rel-L2 {e_f:.3e} >= {force_tol:.0e}"
    return e_pot, e_f


# =====================================================================
# Tests
# =====================================================================

def _scene_charges(n: int, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.5, 1.5, size=n)


def test_adaptive_far_field_mixed_depth():
    """Genuinely occupancy-adaptive tree: leaves at multiple depths, nonempty
    Lists 3/4.  Far-field-only GPU run (zeroNearP2P=1) vs masked direct."""
    scene, n, depth, order = CFG_MIXED
    d, stats = build_scene_metadata(scene, n, depth)
    levels = d["node_center_size"][:, 3].astype(int)
    terminal = (d["node_flags"] & 1).astype(bool)
    leaf_depths = np.unique(levels[terminal])
    assert len(leaf_depths) >= 2, \
        f"config regressed to uniform depth {leaf_depths.tolist()}"
    l3 = int(d["list_counts"][:, 2].sum())
    l4 = int(d["list_counts"][:, 3].sum())
    assert l3 > 0 and l4 > 0, "Lists 3/4 empty: adaptive P2L/M2P paths unused"
    print(f"\n[mixed-depth] {scene} N={n} depth={depth} order={order}: "
          f"nodes={stats['nodes']} leaves={stats['leaves']} "
          f"leaf depths={leaf_depths.tolist()} L3={l3} L4={l4} "
          f"maxLeaf={stats['maxLeaf']}")

    charges = _scene_charges(n)
    pot, fx, fy = run_adaptive_wgsl(d, charges, order, True, grid_dim=1 << depth)
    rp, rfx, rfy = direct_far_field(d, charges)
    _report("GPU far field (P2M+M2M+M2L+P2L+L2L+L2P+M2P) vs masked direct O(N^2)",
            (pot, fx, fy), (rp, rfx, rfy), FAR_POT_TOL, FAR_FORCE_TOL)


def test_adaptive_materialized_far_field():
    """Round-13 materialized far path: p2l + far_gather (flat per-leaf CSR
    gather through the per-(level, offset) operator table) replace the
    per-level m2l/l2l chain. Checked against the masked direct far field
    AND against the legacy-chain GPU run on the identical tree (the two
    paths compute the same sum in a different order, so their agreement is
    bounded by f32 rounding, far below the truncation gates)."""
    scene, n, depth, order = CFG_MIXED
    d, stats = build_scene_metadata(scene, n, depth)
    charges = _scene_charges(n)
    grid_dim = 1 << depth

    pot_m, fx_m, fy_m = run_adaptive_wgsl(d, charges, order, True, grid_dim,
                                          materialized=True)
    rp, rfx, rfy = direct_far_field(d, charges)
    _report("GPU materialized far field (P2M+M2M+p2l+far_gather+L2P+M2P) "
            "vs masked direct O(N^2)",
            (pot_m, fx_m, fy_m), (rp, rfx, rfy), FAR_POT_TOL, FAR_FORCE_TOL)

    pot_c, fx_c, fy_c = run_adaptive_wgsl(d, charges, order, True, grid_dim)
    _report("materialized vs legacy chain (same tree, f32 reorder only)",
            (pot_m, fx_m, fy_m), (pot_c, fx_c, fy_c), 1e-3, 1e-2)


def test_adaptive_full_pipeline_uniform_depth():
    """Uniform-depth tree (all leaves at maxDepth; overlay P2P coverage
    contract satisfied exactly): full GPU pipeline vs direct O(N^2), hybrid
    softened direct, AdaptiveFMM reference, and the exact-schedule emulator."""
    from core.adaptive_fmm import (
        AdaptiveFMM, exact_direct_nbody_2d, exact_direct_nbody_forces_2d)
    from tests.core.test_flat_adaptive_gpu import evaluate_flat_adaptive_emulated
    from core.adaptive_gpu_metadata import FlatAdaptiveMetadata

    scene, n, depth, order = CFG_UNIFORM
    d, stats = build_scene_metadata(scene, n, depth)
    levels = d["node_center_size"][:, 3].astype(int)
    terminal = (d["node_flags"] & 1).astype(bool)
    assert (levels[terminal] == depth).all(), \
        "not a uniform-depth tree: file-kernel overlay P2P coverage contract violated"
    grid_dim = 1 << depth
    print(f"\n[uniform-depth] {scene} N={n} depth={depth} order={order}: "
          f"nodes={stats['nodes']} leaves={stats['leaves']} "
          f"overlay={grid_dim}x{grid_dim} maxLeaf={stats['maxLeaf']} "
          f"L2={int(d['list_counts'][:, 1].sum())}")

    charges = _scene_charges(n)
    pos = d["positions"]

    # --- far-field isolation on the same tree (tight gate) --------------
    pot_f, fx_f, fy_f = run_adaptive_wgsl(d, charges, order, True, grid_dim)
    rp, rfx, rfy = direct_far_field(d, charges)
    _report("GPU far field vs masked direct O(N^2)",
            (pot_f, fx_f, fy_f), (rp, rfx, rfy), FAR_POT_TOL, FAR_FORCE_TOL)

    # --- full pipeline ---------------------------------------------------
    pot, fx, fy = run_adaptive_wgsl(d, charges, order, False, grid_dim)

    # Same-basis reference: the kernel's near-field softening (eps^2 = 4e-5)
    # applied to the same 3x3 pair set, exact f64 elsewhere.  This is the
    # tight full-pipeline gate on the WGSL math itself.
    hp, hfx, hfy = direct_hybrid_softened(d, charges, grid_dim)
    _report("GPU full pipeline vs hybrid direct (softened 3x3 near field)",
            (pot, fx, fy), (hp, hfx, hfy), FAR_POT_TOL, FAR_FORCE_TOL)

    ep = exact_direct_nbody_2d(pos, charges)
    efx, efy = exact_direct_nbody_forces_2d(pos, charges)
    # Softening bias baseline (no GPU involved): the unsoftened references
    # below cannot agree with the WGSL near field better than this.
    bias_pot = _rel_l2(hp, ep)
    bias_f = float(np.linalg.norm(np.stack([hfx, hfy]) - np.stack([efx, efy])) /
                   np.linalg.norm(np.stack([efx, efy])))
    print(f"    [softening bias baseline, hybrid vs unsoftened direct: "
          f"pot={bias_pot:.3e} force={bias_f:.3e} (eps^2={NEAR_SOFTENING_EPS2:g})]")
    assert bias_pot < FULL_POT_TOL and bias_f < FULL_FORCE_TOL, \
        "near-field softening changed: re-derive FULL_*_TOL"
    _report("GPU full pipeline vs direct O(N^2) (unsoftened; bias above)",
            (pot, fx, fy), (ep, efx, efy), FULL_POT_TOL, FULL_FORCE_TOL)

    ref = AdaptiveFMM(max_leaf_particles=16, max_depth=depth, p=order,
                      softening=0.0)
    ap, afx, afy = ref.evaluate(pos, charges, compute_forces=True)
    _report("GPU full pipeline vs core.adaptive_fmm.AdaptiveFMM (CGR88 f64)",
            (pot, fx, fy), (np.asarray(ap), np.asarray(afx), np.asarray(afy)),
            FULL_POT_TOL, FULL_FORCE_TOL)

    md = FlatAdaptiveMetadata(
        node_center_size=d["node_center_size"], node_parent=d["node_parent"],
        node_children=d["node_children"], node_particle_range=d["node_particle_range"],
        node_flags=d["node_flags"], particle_indices=d["particle_indices"],
        list_offsets=d["list_offsets"], list_counts=d["list_counts"],
        list_data=d["list_data"], leaf_node_for_particle=d["leaf_node_for_particle"],
        bounds=(0.0, 1.0, 0.0, 1.0),
    )
    mp, mfx, mfy = evaluate_flat_adaptive_emulated(
        pos, md, expansion_order=order, charges=charges)
    _report("GPU full pipeline vs flat-schedule f64 emulator (same tree/lists,"
            " unsoftened near field)",
            (pot, fx, fy), (mp, mfx, mfy), FULL_POT_TOL, FULL_FORCE_TOL)


def main() -> int:
    tests = [test_adaptive_far_field_mixed_depth,
             test_adaptive_full_pipeline_uniform_depth]
    failed = 0
    for t in tests:
        try:
            t()
        except BaseException as e:  # noqa: BLE001 - map pytest outcomes standalone
            if type(e).__name__ == "Skipped":
                print(f"{t.__name__}: SKIP ({e})")
                continue
            if isinstance(e, AssertionError):
                failed += 1
                print(f"{t.__name__}: FAIL: {e}")
                continue
            raise
        print(f"{t.__name__}: PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
