"""
OpenCL N-Body & FMM Accelerator Backend (`opencl_fmm_backend.py`)
================================================================
Executes Tree-Free FMM and N-Body kernels across AMD / ATI Radeon GPUs (RDNA, CDNA,
GCN, Radeon Pro, APUs), Intel Iris/Arc, and NVIDIA GPUs via OpenCL.

Features:
- Automated OpenCL device discovery prioritizing AMD/ATI Radeon hardware.
- JIT kernel compilation with caching and local workgroup memory (LDS) optimization.
- Seamless NumPy / memoryview zero-copy staging buffers.
"""

from __future__ import annotations
import os
import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

OPENCL_CL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "tree_free_fmm_opencl.cl")

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False
    cl = None


class OpenCLFMMContext:
    """
    Singleton Context manager for OpenCL hardware acceleration.
    """
    _instance: Optional[OpenCLFMMContext] = None

    def __init__(self, prefer_amd: bool = True):
        self.context = None
        self.queue = None
        self.program = None
        self.device = None
        self.platform = None
        self.is_initialized = False
        self.prefer_amd = prefer_amd

        if HAS_PYOPENCL:
            self._initialize_context()

    @classmethod
    def get_instance(cls, prefer_amd: bool = True) -> OpenCLFMMContext:
        if cls._instance is None:
            cls._instance = OpenCLFMMContext(prefer_amd=prefer_amd)
        return cls._instance

    def _initialize_context(self):
        try:
            platforms = cl.get_platforms()
            if not platforms:
                return

            chosen_platform = None
            chosen_device = None

            if self.prefer_amd:
                # Look for AMD platform first
                for p in platforms:
                    if "AMD" in p.name.upper() or "ADVANCED MICRO DEVICES" in p.vendor.upper():
                        devs = p.get_devices(device_type=cl.device_type.GPU)
                        if devs:
                            chosen_platform = p
                            chosen_device = devs[0]
                            break

            # Fallback to any GPU device
            if chosen_device is None:
                for p in platforms:
                    devs = p.get_devices(device_type=cl.device_type.GPU)
                    if devs:
                        chosen_platform = p
                        chosen_device = devs[0]
                        break

            # Fallback to any OpenCL device (including CPU)
            if chosen_device is None and platforms:
                chosen_platform = platforms[0]
                devs = chosen_platform.get_devices()
                if devs:
                    chosen_device = devs[0]

            if chosen_device is None:
                return

            self.platform = chosen_platform
            self.device = chosen_device
            self.context = cl.Context([chosen_device])
            self.queue = cl.CommandQueue(self.context)

            # Load and build OpenCL C kernel program
            with open(OPENCL_CL_SOURCE_PATH, "r", encoding="utf-8") as f:
                cl_src = f.read()

            self.program = cl.Program(self.context, cl_src).build()
            self.is_initialized = True

        except Exception as ex:
            self.is_initialized = False

    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_initialized or self.device is None:
            return {"status": "UNAVAILABLE", "has_pyopencl": HAS_PYOPENCL}
        return {
            "status": "READY",
            "platform": self.platform.name,
            "device_name": self.device.name,
            "vendor": self.device.vendor,
            "compute_units": self.device.max_compute_units,
            "global_mem_mb": self.device.global_mem_size / (1024 * 1024),
            "local_mem_kb": self.device.local_mem_size / 1024,
            "is_amd": "AMD" in self.device.vendor.upper() or "ATI" in self.device.vendor.upper()
        }


def is_opencl_available() -> bool:
    """Returns True if OpenCL is installed and a device context was successfully initialized."""
    ctx = OpenCLFMMContext.get_instance()
    return ctx.is_initialized


def opencl_tree_free_nbody(
    coords: np.ndarray,
    charges: np.ndarray,
    softening: float = 1e-3,
    workgroup_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes all-pairs Coulomb potentials & forces using AMD / OpenCL GPU kernel.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    charges = np.ascontiguousarray(charges, dtype=np.float32)
    N = len(coords)

    if not is_opencl_available():
        raise RuntimeError("OpenCL is not available or initialized. Please install PyOpenCL and GPU OpenCL drivers.")

    ctx = OpenCLFMMContext.get_instance()
    mf = cl.mem_flags

    # Allocate device buffers
    d_coords = cl.Buffer(ctx.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coords)
    d_charges = cl.Buffer(ctx.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charges)
    d_pot = cl.Buffer(ctx.context, mf.WRITE_ONLY, size=N * 4) # float32 = 4 bytes
    d_forces = cl.Buffer(ctx.context, mf.WRITE_ONLY, size=N * 3 * 4)

    # Local scratchpad memory in AMD Local Data Share (LDS)
    local_mem = cl.LocalMemory(workgroup_size * 16) # 16 bytes per float4

    global_size = ((N + workgroup_size - 1) // workgroup_size) * workgroup_size
    local_size = workgroup_size

    ctx.program.opencl_p2p_coulomb_nbody(
        ctx.queue,
        (global_size,),
        (local_size,),
        d_coords,
        d_charges,
        d_pot,
        d_forces,
        np.int32(N),
        np.float32(softening ** 2),
        local_mem
    )

    out_pot = np.empty(N, dtype=np.float32)
    out_forces = np.empty((N, 3), dtype=np.float32)

    cl.enqueue_copy(ctx.queue, out_pot, d_pot)
    cl.enqueue_copy(ctx.queue, out_forces, d_forces)
    ctx.queue.finish()

    return out_pot, out_forces


def opencl_morton_encode_3d(coords: np.ndarray, depth: int = 10) -> np.ndarray:
    """
    Vectorized 3D Morton coordinate encoding on OpenCL hardware.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    N = len(coords)

    if not is_opencl_available():
        raise RuntimeError("OpenCL is not available.")

    ctx = OpenCLFMMContext.get_instance()
    mf = cl.mem_flags

    d_coords = cl.Buffer(ctx.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coords)
    d_keys = cl.Buffer(ctx.context, mf.WRITE_ONLY, size=N * 4)

    global_size = ((N + 255) // 256) * 256
    ctx.program.opencl_morton_encode_3d(
        ctx.queue,
        (global_size,),
        (256,),
        d_coords,
        d_keys,
        np.int32(N),
        np.int32(depth)
    )

    out_keys = np.empty(N, dtype=np.uint32)
    cl.enqueue_copy(ctx.queue, out_keys, d_keys)
    ctx.queue.finish()
    return out_keys


def run_opencl_benchmark():
    print("=" * 70)
    print("ATI / AMD RADEON & OPENCL ACCELERATOR BENCHMARK")
    print("=" * 70)

    ctx = OpenCLFMMContext.get_instance(prefer_amd=True)
    info = ctx.get_device_info()

    print(f"[-] OpenCL Availability : {info['status']}")
    if info["status"] != "READY":
        print(f"[-] PyOpenCL Installed  : {HAS_PYOPENCL}")
        print("[INFO] To enable native OpenCL on AMD Radeon / Intel / NVIDIA, run: pip install pyopencl")
        return

    print(f"[-] Selected Platform   : {info['platform']}")
    print(f"[-] Device Name         : {info['device_name']}")
    print(f"[-] AMD / ATI Radeon    : {'YES' if info['is_amd'] else 'NO'}")
    print(f"[-] Compute Units (CUs) : {info['compute_units']}")
    print(f"[-] Global VRAM         : {info['global_mem_mb']:.1f} MB")
    print(f"[-] Local Memory (LDS)  : {info['local_mem_kb']:.1f} KB")

    N = 10000
    rng = np.random.RandomState(42)
    coords = rng.uniform(0.1, 0.9, (N, 3)).astype(np.float32)
    charges = rng.uniform(-1.0, 1.0, N).astype(np.float32)

    # Warmup
    _ = opencl_tree_free_nbody(coords[:100], charges[:100])

    t0 = time.perf_counter()
    pot, forces = opencl_tree_free_nbody(coords, charges)
    t_el = (time.perf_counter() - t0) * 1000.0

    print(f"\n[-] OpenCL N-Body Evaluation ({N:,} particles): {t_el:.2f} ms")
    print(f"[-] Throughput: {N / (t_el / 1000.0):,.0f} particles/sec")
    print(f"[-] Mean Potential: {float(np.mean(pot)):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_opencl_benchmark()
