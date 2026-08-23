"""Adaptive FMM WGSL kernel gate: compile + counting-sort CSR validation.

Compiles the full `adaptive_fmm.wgsl` module on a native WebGPU adapter
(this alone type-checks every entry point, including the 16-storage-binding
consolidated layout: nodeMeta = parent|flags vec2, cellArrays =
count|cursor|start packed in one buffer), then dispatches the T-E1 passes
(clear_cells -> count_cells -> scan_cells -> scatter_cells) and verifies the
CSR invariants on readback:

  * cell counts sum to N
  * sortedIndex is a permutation of [0, N)
  * every slot in cell c's contiguous range holds a particle whose own cell
    index is c

Skips (exit 0) when wgpu-py is missing or no adapter is found.

Run standalone:  python -X utf8 -m core.test_adaptive_wgsl_csr
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main():
    try:
        import wgpu
    except ImportError:
        sys.exit(_skip("wgpu not installed"))

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        sys.exit(_skip("no WebGPU adapter"))
    device = adapter.request_device_sync(required_limits={})

    from core.webgpu_kernels.webgpu_fmm_runner import (
        get_adaptive_fmm_wgsl_source,
    )
    module = device.create_shader_module(code=get_adaptive_fmm_wgsl_source())
    # Module creation type-checks every entry point; pipeline creation below
    # validates the bind group layout against the 16-storage-binding cap.
    print("adaptive_fmm.wgsl compiled (16 storage + 2 uniform bindings)")

    rng = np.random.default_rng(11)
    n = 1000
    grid_dim = 16
    nc = grid_dim * grid_dim
    pts = rng.random((n, 2))

    # Particle { pos: vec2<f32>, vel: vec2<f32> } = 16 B
    particles = np.zeros((n, 4), dtype=np.float32)
    particles[:, 0] = pts[:, 0]
    particles[:, 1] = pts[:, 1]
    # FmmParams: 8 u32 words
    fmm = np.zeros(8, dtype=np.uint32)
    fmm[0] = n
    # GridParams { gridDim: u32, _pad: u32, gridOrigin: vec2, cellSize: f32 }
    # -> 24 B (struct size rounds up to the vec2 alignment)
    gp = np.zeros(6, dtype=np.uint32)
    gp[0] = grid_dim
    gp_f = gp.view(np.float32)
    gp_f[4] = 1.0 / grid_dim  # gridOrigin stays (0,0)

    SU, CD, CS = (wgpu.BufferUsage.STORAGE, wgpu.BufferUsage.COPY_DST,
                  wgpu.BufferUsage.COPY_SRC)

    def mk(data, usage=SU):
        buf = device.create_buffer(size=data.nbytes, usage=usage | CD | CS)
        device.queue.write_buffer(buf, 0, np.ascontiguousarray(data).tobytes())
        return buf

    def mkraw(nbytes, usage=SU):
        return device.create_buffer(size=nbytes, usage=usage | CS)

    bufs = [
        mk(particles),                 # 0 particles
        mkraw(16),                     # 1 nodeCenterSize (1 vec4)
        mkraw(8),                      # 2 nodeParticleRange (1 vec2)
        mk(np.zeros(n, np.uint32)),    # 3 leafForParticle
        mk(np.zeros(n, np.uint32)),    # 4 particleIndices
        mkraw(16),                     # 5 listOffsets (1 vec4)
        mkraw(16),                     # 6 listCounts (1 vec4)
        mkraw(4),                      # 7 listData (1 u32)
        mkraw(3 * 16),                 # 8 multipoles (3 vec4)
        mkraw(3 * 16),                 # 9 locals (3 vec4)
        mk(np.zeros((n, 4), np.float32)),  # 10 fields
        mk(fmm, wgpu.BufferUsage.UNIFORM),  # 11 params
        mkraw(8),                      # 12 nodeMeta (1 vec2<u32>)
        mkraw(16),                     # 13 nodeChildren (1 vec4)
        mk(np.ones(n, np.float32)),    # 15 charges
        mk(gp, wgpu.BufferUsage.UNIFORM),   # 16 gridParams
        mkraw(3 * nc * 4),             # 17 cellArrays (count|cursor|start)
        mkraw(n * 4),                  # 20 sortedIndex
    ]
    binding_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20]

    COMP = wgpu.ShaderStage.COMPUTE
    ro = {"type": "read-only-storage"}
    rw = {"type": "storage"}
    types = [ro, ro, ro, ro, ro, ro, ro, ro, rw, rw, rw,
             {"type": "uniform"}, ro, ro, ro, {"type": "uniform"}, rw, rw]
    bgl = device.create_bind_group_layout(entries=[
        {"binding": bid, "visibility": COMP, "buffer": t}
        for bid, t in zip(binding_ids, types)])
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[{"binding": bid, "resource": {"buffer": b, "offset": 0}}
                 for bid, b in zip(binding_ids, bufs)])

    def dispatch(entry, n_workgroups):
        pipeline = device.create_compute_pipeline(
            layout=layout,
            compute={"module": module, "entry_point": entry})
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(n_workgroups, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])

    def readback(buf, nbytes, dtype):
        st = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        enc = device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, st, 0, nbytes)
        device.queue.submit([enc.finish()])
        st.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(st.read_mapped(), dtype=dtype).copy()
        st.unmap()
        st.destroy()
        return out

    dispatch("clear_cells", (nc + 255) // 256)
    dispatch("count_cells", (n + 255) // 256)
    dispatch("scan_cells", 1)
    dispatch("scatter_cells", (n + 255) // 256)

    cell_arrays = readback(bufs[16], 3 * nc * 4, np.uint32)
    cell_count = cell_arrays[:nc]
    cell_start = cell_arrays[2 * nc:]
    sorted_idx = readback(bufs[17], n * 4, np.uint32)

    assert int(cell_count.sum()) == n, \
        f"CSR: counts sum {int(cell_count.sum())} != N {n}"
    assert len(np.unique(sorted_idx)) == n, "CSR: sortedIndex not a permutation"

    cx = np.clip((pts[:, 0] * grid_dim).astype(np.int64), 0, grid_dim - 1)
    cy = np.clip((pts[:, 1] * grid_dim).astype(np.int64), 0, grid_dim - 1)
    expected_cell = cy * grid_dim + cx
    for c in range(nc):
        s, cnt = int(cell_start[c]), int(cell_count[c])
        if cnt:
            got = expected_cell[sorted_idx[s:s + cnt].astype(np.int64)]
            assert (got == c).all(), f"CSR: cell {c} range holds foreign particles"

    print(f"test_adaptive_wgsl_csr: PASS (N={n}, {grid_dim}x{grid_dim} grid, "
          f"packed cellArrays CSR verified)")
    for b in bufs:
        b.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
