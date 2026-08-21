"""
WebGPU & WGSL Compute Shader Interop (`webgpu_fmm_runner.py`)
=============================================================
Provides cross-platform WebGPU compute execution for browser, WebXR, and cloud-edge deployments.
Executes natively on AMD Radeon (Vulkan/DX12), NVIDIA (Vulkan/DX12), Apple Silicon (Metal),
and Intel Arc without requiring vendor-specific toolchains.

Two roles:
  1. WGSL source access for browser/client integration (get_wgsl_source /
     get_adaptive_cgr88_wgsl_source).
  2. Host-side dispatch via wgpu-py: run_fixed_grid_fmm_forces() executes the
     file kernel `tree_free_fmm.wgsl` (T-E1 counting-sort CSR cell lists +
     CGR88 complex L2P far field) on a native adapter and returns per-particle
     forces. The far-field local-expansion coefficients are built host-side
     (exact 2D complex Taylor fit about each leaf-cell center from all
     particles outside the cell's 3x3 neighborhood), the near field + L2P
     evaluation runs on the GPU.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np

WGSL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "tree_free_fmm.wgsl")
ADAPTIVE_CGR88_WGSL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "adaptive_cgr88.wgsl")

try:
    import wgpu
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False
    wgpu = None


def get_wgsl_source() -> str:
    """Returns WGSL source code for browser / client integration."""
    with open(WGSL_SOURCE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_adaptive_cgr88_wgsl_source() -> str:
    """Return the staged flat adaptive CGR88 WGSL kernel source."""
    with open(ADAPTIVE_CGR88_WGSL_SOURCE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def is_webgpu_available() -> bool:
    """Returns True if wgpu-py is installed and an adapter is found."""
    if not HAS_WGPU:
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync()
        return adapter is not None
    except Exception:
        return False


def get_webgpu_adapter_info() -> Dict[str, Any]:
    """Queries active WebGPU backend (Vulkan / DX12 / Metal)."""
    if not is_webgpu_available():
        return {"status": "UNAVAILABLE", "has_wgpu": HAS_WGPU}
    try:
        adapter = wgpu.gpu.request_adapter_sync()
        info = adapter.summary
        return {
            "status": "READY",
            "summary": info,
            "backend": getattr(adapter, "backend_type", "Unknown")
        }
    except Exception as ex:
        return {"status": "ERROR", "error": str(ex)}


# =====================================================================
# Host-side far-field: per-leaf-cell complex local expansions (CGR88).
# The kernel's L2P evaluates  phi(z) ~= Re( sum_k l_k (z - z_c)^k )  with
# force  (-Re L'(z), Im L'(z)).  Matching coefficients for the 2D log
# kernel  phi_j(z) = q_j log|z - z_j|  expanded about the cell center z_c
# in terms of w_j = z_c - z_j (sources outside the 3x3 neighborhood, so
# |dz| / |w_j| <= ~0.5 and the order-`order` truncation converges):
#   l_0 = sum_j q_j log w_j            (complex log; Re gives log|w_j|)
#   l_k = sum_j q_j (-1)^(k+1) / (k w_j^k),  k = 1..order
# =====================================================================

def build_cluster_local_expansions(pts: np.ndarray, q: np.ndarray,
                                   grid_dim: int, order: int
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cluster_centers (C,2) f64, cluster_ids (N,) i64,
    coeffs (C, order+1, 2) f64 [real, imag]) for the uniform leaf grid."""
    n = len(pts)
    dim = grid_dim
    nc = dim * dim
    cx = np.clip((pts[:, 0] * dim).astype(np.int64), 0, dim - 1)
    cy = np.clip((pts[:, 1] * dim).astype(np.int64), 0, dim - 1)
    cluster_ids = cy * dim + cx
    centers = np.empty((nc, 2), dtype=np.float64)
    gx = (np.arange(dim) + 0.5) / dim
    # cell index c = iy * dim + ix (row-major): x varies fastest -> tile,
    # y repeats -> repeat. (Swapping these transposes every cell center and
    # makes the |dz| < |w| convergence premise of the local expansion fail.)
    centers[:, 0] = np.tile(gx, dim)
    centers[:, 1] = np.repeat(gx, dim)

    # Complex cell-center minus particle positions: (C, N)
    w_re = centers[:, 0][:, None] - pts[None, :, 0]
    w_im = centers[:, 1][:, None] - pts[None, :, 1]
    # Far mask: particle j is FAR from cell c iff its own cell is NOT in the
    # 3x3 neighborhood of c (near field is handled by the CSR P2P pass).
    cell_x = np.arange(nc) % dim    # x cell index of each grid cell
    cell_y = np.arange(nc) // dim
    ccx = np.clip((pts[:, 0] * dim).astype(np.int64), 0, dim - 1)
    ccy = np.clip((pts[:, 1] * dim).astype(np.int64), 0, dim - 1)
    dx = np.abs(cell_x[:, None] - ccx[None, :])
    dy = np.abs(cell_y[:, None] - ccy[None, :])
    far = (dx > 1) | (dy > 1)
    cluster_ids = ccy * dim + ccx
    qm = q[None, :] * far

    w_abs = np.hypot(w_re, w_im)
    w_angle = np.arctan2(w_im, w_re)
    log_w_abs = np.log(np.maximum(w_abs, 1e-300))

    coeffs = np.zeros((nc, order + 1, 2), dtype=np.float64)
    # l_0 = sum_j q_j log w_j  (real part = log|w_j|)
    l0_re = np.sum(qm * log_w_abs, axis=1)
    l0_im = np.sum(qm * w_angle, axis=1)
    coeffs[:, 0, 0] = l0_re
    coeffs[:, 0, 1] = l0_im
    # l_k via w_j^(-k) = |w|^(-k) (cos(-k angle), sin(-k angle)).
    # Mask BEFORE the negative power: a near-source particle sitting on the
    # cell center makes |w|^(-k) overflow to inf, and the far-mask multiply
    # (0 * inf) would poison every coefficient with NaN.
    for k in range(1, order + 1):
        r_k = np.where(far, np.power(np.maximum(w_abs, 1e-300), -k), 0.0)
        a_k = -k * w_angle
        wk_re = r_k * np.cos(a_k)
        wk_im = r_k * np.sin(a_k)
        sign = (-1.0) ** (k + 1) / k
        coeffs[:, k, 0] = sign * np.sum(qm * wk_re, axis=1)
        coeffs[:, k, 1] = sign * np.sum(qm * wk_im, axis=1)
    return centers, cluster_ids, coeffs


def run_fixed_grid_fmm_forces(pts: np.ndarray, q: np.ndarray,
                              softening: float = 0.02,
                              depth: int = 5, order: int = 8,
                              verify_csr: bool = True
                              ) -> np.ndarray:
    """Run the file kernel tree_free_fmm.wgsl on a native WebGPU adapter.

    Dispatch order (T-E1): clear_cells -> count_cells -> scan_cells ->
    scatter_cells -> fmm_compute_main.  Returns per-particle force vectors
    (N, 2) float32. When verify_csr, also reads back the counting-sort
    buffers and asserts the CSR invariants (each particle exactly once,
    sorted slot cell matches the particle's cell).
    """
    if not HAS_WGPU:
        raise RuntimeError("wgpu-py is not installed")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("no WebGPU adapter available")
    device = adapter.request_device_sync(required_limits={})
    code = get_wgsl_source()
    module = device.create_shader_module(code=code)

    pts = np.asarray(pts, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n = len(pts)
    grid_dim = 1 << depth
    nc = grid_dim * grid_dim
    cell_size = 1.0 / grid_dim

    centers, cluster_ids, coeffs = build_cluster_local_expansions(
        pts, q, grid_dim, order)

    # ---- host-side staging arrays (struct layouts per the WGSL) ----
    # Particle2D { pos: vec2<f32>, q: f32, cluster_id: u32 }: the u32 member
    # must hold the INTEGER cell id, so build u32 words and poke the floats
    # through a view (a float32 array would store 226.0 as 0x43620000).
    particles = np.zeros((n, 4), dtype=np.uint32)
    particles_f = particles.view(np.float32)
    particles_f[:, 0] = pts[:, 0]
    particles_f[:, 1] = pts[:, 1]
    particles_f[:, 2] = q
    particles[:, 3] = cluster_ids.astype(np.uint32)
    centers_f32 = centers.astype(np.float32)
    coeffs_f32 = coeffs.astype(np.float32)           # (C, order+1, 2)
    # SimulationParams: u32,u32,u32 | f32,f32 | u32,u32 | pad | vec2 -> 40 B.
    # Build as u32 words and poke the float members through a view — writing
    # e.g. n as a float32 bit pattern would hand the kernel 0x42800000 (1.1e9)
    # as num_particles.
    params = np.zeros(10, dtype=np.uint32)
    params_f = params.view(np.float32)
    params[0] = np.uint32(n)
    params[1] = np.uint32(nc)
    params[2] = np.uint32(order)
    params_f[3] = np.float32(softening * softening)
    params_f[4] = np.float32(cell_size)
    params[5] = np.uint32(grid_dim)
    # params[6] = _pad0; params_f[8:10] = grid_origin (0,0)

    SU = wgpu.BufferUsage.STORAGE
    CD = wgpu.BufferUsage.COPY_DST
    CS = wgpu.BufferUsage.COPY_SRC
    MR = wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST

    def mkbuf(arr_or_size, usage):
        # Every buffer gets COPY_SRC so the readback path can copy out of it.
        usage = usage | CS
        if isinstance(arr_or_size, np.ndarray):
            data = np.ascontiguousarray(arr_or_size)
            buf = device.create_buffer(size=data.nbytes, usage=usage | CD)
            device.queue.write_buffer(buf, 0, data.tobytes())
            return buf
        return device.create_buffer(size=arr_or_size, usage=usage)

    bufs = {
        "particles": mkbuf(particles, SU),
        "potentials": mkbuf(n * 4, SU),
        "forces": mkbuf(n * 16, SU),
        "cluster_centers": mkbuf(centers_f32, SU),
        "cluster_coeffs": mkbuf(coeffs_f32, SU),
        "params": mkbuf(params, wgpu.BufferUsage.UNIFORM),
        "cellCount": mkbuf(nc * 4, SU),
        "cellCursor": mkbuf(nc * 4, SU),
        "cellStart": mkbuf(nc * 4, SU),
        "sortedIndex": mkbuf(n * 4, SU),
    }

    # Explicit pipeline layout covering ALL 10 bindings: layout="auto" would
    # derive a per-entry-point layout from only the bindings each entry uses
    # (clear_cells touches 2), so one shared bind group must come from an
    # explicit layout instead.
    COMP = wgpu.ShaderStage.COMPUTE
    ro = {"type": "read-only-storage"}
    rw = {"type": "storage"}
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": COMP, "buffer": ro},   # particles
        {"binding": 1, "visibility": COMP, "buffer": rw},   # potentials
        {"binding": 2, "visibility": COMP, "buffer": rw},   # forces
        {"binding": 3, "visibility": COMP, "buffer": ro},   # cluster_centers
        {"binding": 4, "visibility": COMP, "buffer": ro},   # cluster_local_coeffs
        {"binding": 5, "visibility": COMP, "buffer": {"type": "uniform"}},
        {"binding": 6, "visibility": COMP, "buffer": rw},   # cellCount
        {"binding": 7, "visibility": COMP, "buffer": rw},   # cellCursor
        {"binding": 8, "visibility": COMP, "buffer": rw},   # cellStart
        {"binding": 9, "visibility": COMP, "buffer": rw},   # sortedIndex
    ])
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0}}
                 for i, b in enumerate(
                     [bufs["particles"], bufs["potentials"], bufs["forces"],
                      bufs["cluster_centers"], bufs["cluster_coeffs"],
                      bufs["params"], bufs["cellCount"], bufs["cellCursor"],
                      bufs["cellStart"], bufs["sortedIndex"]])])

    def dispatch(entry, n_workgroups):
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": module, "entry_point": entry})
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(n_workgroups, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])

    ceil = lambda m: (int(m) + 255) // 256
    dispatch("clear_cells", ceil(nc))
    dispatch("count_cells", ceil(n))
    dispatch("scan_cells", 1)   # single-workgroup scan by design
    dispatch("scatter_cells", ceil(n))
    dispatch("fmm_compute_main", (n + 127) // 128)

    def readback(buf, nbytes):
        staging = device.create_buffer(size=nbytes, usage=MR)
        enc = device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, staging, 0, nbytes)
        device.queue.submit([enc.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        data = staging.read_mapped()
        out = np.frombuffer(data, dtype=np.uint8).copy()
        staging.unmap()
        staging.destroy()
        return out

    forces_raw = readback(bufs["forces"], n * 16)
    forces = forces_raw.view(np.float32).reshape(n, 4)[:, :2].copy()

    if verify_csr:
        cell_start = readback(bufs["cellStart"], nc * 4).view(np.uint32)
        cell_count = readback(bufs["cellCount"], nc * 4).view(np.uint32)
        sorted_idx = readback(bufs["sortedIndex"], n * 4).view(np.uint32)
        assert int(cell_count.sum()) == n, "CSR: counts do not sum to N"
        assert len(np.unique(sorted_idx)) == n, "CSR: sortedIndex not a permutation"
        pos32 = particles_f[:, :2]  # float view — particles[] holds u32 words
        cxl = np.clip((pos32[:, 0] * grid_dim).astype(np.int64), 0, grid_dim - 1)
        cyl = np.clip((pos32[:, 1] * grid_dim).astype(np.int64), 0, grid_dim - 1)
        expected_cell = cyl * grid_dim + cxl
        for c in range(nc):
            s, cnt = int(cell_start[c]), int(cell_count[c])
            if cnt:
                got = expected_cell[sorted_idx[s:s + cnt].astype(np.int64)]
                assert (got == c).all(), f"CSR: cell {c} range holds foreign particles"

    for b in bufs.values():
        b.destroy()
    return forces


def run_webgpu_demo():
    print("=" * 70)
    print("WEBGPU / WGSL COMPUTE SHADER COMPLIANCE DEMO")
    print("=" * 70)
    info = get_webgpu_adapter_info()
    print(f"[-] WebGPU Availability: {info['status']}")
    if info['status'] != 'READY':
        print(f"[-] Python wgpu package: {HAS_WGPU}")
        print("[-] WGSL Shader Source : Loaded successfully (Length: " + str(len(get_wgsl_source())) + " bytes)")
        print("[INFO] WebGPU WGSL shaders allow the Tree-Free FMM engine to run directly")
        print("       in web browsers (WebXR / 3D web apps) and cloud edge runtimes across")
        print("       AMD Radeon, NVIDIA, Intel, and Apple GPUs with zero modifications.")
    else:
        print(f"[-] WebGPU Adapter Summary: {info.get('summary')}")
        # Headless smoke: run the fixed-grid kernel on a small random cloud.
        rng = np.random.default_rng(0)
        pts = rng.random((512, 2))
        qq = rng.uniform(-1, 1, 512)
        f = run_fixed_grid_fmm_forces(pts, qq, softening=0.02, depth=5, order=8)
        print(f"[-] Fixed-grid WGSL smoke: N=512, |F| mean "
              f"{float(np.linalg.norm(f, axis=1).mean()):.4e}, CSR verified")
    print("=" * 70)


if __name__ == "__main__":
    run_webgpu_demo()
