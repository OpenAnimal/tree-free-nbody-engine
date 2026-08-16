"""
Native Zig CPU SIMD Acceleration Bridge for Tree-Free FMM Engine
================================================================
Provides zero-copy, highly-vectorized SIMD CPU execution, arbitrary bitwidth
Morton quantization, bitboard fast-forwarding, 2D/3D multipole expansions,
and zero-allocation contact solving.
"""

import ctypes
import os
import sys
from typing import Optional, Tuple, Dict, Any
import numpy as np

# -----------------------------------------------------------------------------
# C-ABI Structures
# -----------------------------------------------------------------------------

class FMMMultipole3D(ctypes.Structure):
    _fields_ = [
        ("m0", ctypes.c_float),
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dz", ctypes.c_float),
        ("qxx", ctypes.c_float),
        ("qyy", ctypes.c_float),
        ("qzz", ctypes.c_float),
        ("qxy", ctypes.c_float),
        ("qxz", ctypes.c_float),
        ("qyz", ctypes.c_float),
    ]

class FMMLocal3D(ctypes.Structure):
    _fields_ = [
        ("l0", ctypes.c_float),
        ("gx", ctypes.c_float),
        ("gy", ctypes.c_float),
        ("gz", ctypes.c_float),
        ("hxx", ctypes.c_float),
        ("hyy", ctypes.c_float),
        ("hzz", ctypes.c_float),
        ("hxy", ctypes.c_float),
        ("hxz", ctypes.c_float),
        ("hyz", ctypes.c_float),
    ]

def _find_zig_dll() -> Optional[str]:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # native/zig/zig-out/bin (Windows)
        os.path.join(curr_dir, "..", "native", "zig", "zig-out", "bin", "tree_free_fmm_native.dll"),
        # native/zig/zig-out/lib (Linux/macOS)
        os.path.join(curr_dir, "..", "native", "zig", "zig-out", "lib", "libtree_free_fmm_native.so"),
        os.path.join(curr_dir, "..", "native", "zig", "zig-out", "lib", "libtree_free_fmm_native.dylib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _load_library() -> Optional[ctypes.CDLL]:
    dll_path = _find_zig_dll()
    if dll_path is None:
        return None
    try:
        lib = ctypes.CDLL(dll_path)
        
        # Version
        lib.zig_fmm_version.restype = ctypes.c_uint32
        lib.zig_fmm_version.argtypes = []

        # P2P & Forces
        lib.zig_fmm_p2p_potentials.restype = None
        lib.zig_fmm_p2p_potentials.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]

        lib.zig_fmm_p2p_forces.restype = None
        lib.zig_fmm_p2p_forces.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        # Morton & Bitboard
        lib.zig_fmm_encode_morton3d.restype = None
        lib.zig_fmm_encode_morton3d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]

        lib.zig_fmm_build_bitboard64.restype = ctypes.c_uint64
        lib.zig_fmm_build_bitboard64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]

        # Contact IPC
        lib.zig_fmm_contact_forces.restype = None
        lib.zig_fmm_contact_forces.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        # 2D Complex Multipole Expansions
        lib.zig_fmm_2d_p2m.restype = None
        lib.zig_fmm_2d_p2m.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        lib.zig_fmm_2d_m2l.restype = None
        lib.zig_fmm_2d_m2l.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        lib.zig_fmm_2d_l2p.restype = None
        lib.zig_fmm_2d_l2p.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]

        # 3D Cartesian Multipole Expansions
        lib.zig_fmm_3d_p2m.restype = None
        lib.zig_fmm_3d_p2m.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(FMMMultipole3D),
        ]

        lib.zig_fmm_3d_m2p.restype = None
        lib.zig_fmm_3d_m2p.argtypes = [
            ctypes.POINTER(FMMMultipole3D),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]

        lib.zig_fmm_3d_m2l.restype = None
        lib.zig_fmm_3d_m2l.argtypes = [
            ctypes.POINTER(FMMMultipole3D),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(FMMLocal3D),
        ]

        lib.zig_fmm_3d_l2p.restype = None
        lib.zig_fmm_3d_l2p.argtypes = [
            ctypes.POINTER(FMMLocal3D),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]

        return lib
    except Exception:
        return None

_LIB = _load_library()

# -----------------------------------------------------------------------------
# Input Validation Helpers
# -----------------------------------------------------------------------------

def _prepare_positions(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (N, 2) or (N, 3)")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")
    return np.ascontiguousarray(pos)

def _prepare_particles(positions: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = _prepare_positions(positions)
    mass = np.asarray(masses, dtype=np.float32)
    if mass.ndim != 1 or len(mass) != len(pos):
        raise ValueError("masses must have shape (N,)")
    if not np.all(np.isfinite(mass)):
        raise ValueError("masses must contain only finite values")
    return pos, np.ascontiguousarray(mass)

def _validate_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def is_zig_available() -> bool:
    """Returns True if the compiled Zig native library is loaded and ready."""
    return _LIB is not None

def get_zig_version() -> Optional[int]:
    """Returns the library version (e.g. 110 for 1.1.0) or None."""
    if _LIB is None:
        return None
    return int(_LIB.zig_fmm_version())

def zig_simd_p2p_potentials(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = 1e-2
) -> np.ndarray:
    """
    Evaluates N-Body direct gravitational/Coulomb potentials using Zig @Vector(8, f32) SIMD.
    positions: (N, 3) or (N, 2) float32 array
    masses: (N,) float32 array
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    _validate_positive(softening, "softening")
    pos, m = _prepare_particles(positions, masses)
    N = len(pos)

    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    out_pot = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_p2p_potentials(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(softening ** 2),
        out_pot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_pot

def zig_simd_p2p_forces(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = 1e-2
) -> np.ndarray:
    """
    Evaluates N-Body direct gravitational/Coulomb forces using Zig @Vector(8, f32) SIMD.
    returns: (N, 3) force vector array
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    _validate_positive(softening, "softening")
    pos, m = _prepare_particles(positions, masses)
    N = len(pos)

    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    fx = np.zeros(N, dtype=np.float32)
    fy = np.zeros(N, dtype=np.float32)
    fz = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_p2p_forces(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(softening ** 2),
        fx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return np.column_stack([fx, fy, fz])

def zig_encode_morton3d(
    positions: np.ndarray,
    min_bound: float = 0.0,
    max_bound: float = 1.0,
    depth: int = 6
) -> np.ndarray:
    """
    Encodes (N, 3) continuous positions into 64-bit Morton codes via bitwise interleaving in Zig.
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    if not isinstance(depth, (int, np.integer)) or not 0 <= int(depth) <= 21:
        raise ValueError("depth must be an integer in the range [0, 21]")
    if not np.isfinite(min_bound) or not np.isfinite(max_bound) or max_bound <= min_bound:
        raise ValueError("max_bound must be finite and greater than min_bound")

    pos = _prepare_positions(positions)
    N = len(pos)
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    out_morton = np.zeros(N, dtype=np.uint64)

    _LIB.zig_fmm_encode_morton3d(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(min_bound),
        ctypes.c_float(max_bound),
        ctypes.c_uint32(depth),
        out_morton.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    return out_morton

def zig_build_bitboard64(morton_codes: np.ndarray, shift: int = 0) -> int:
    """Builds a 64-cell occupancy bitboard from Morton codes."""
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    codes = np.ascontiguousarray(morton_codes, dtype=np.uint64)
    if codes.ndim != 1:
        raise ValueError("morton_codes must have shape (N,)")
    if not isinstance(shift, (int, np.integer)) or not 0 <= int(shift) < 64:
        raise ValueError("shift must be an integer in the range [0, 63]")
    return int(_LIB.zig_fmm_build_bitboard64(
        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_size_t(len(codes)),
        ctypes.c_uint32(int(shift)),
    ))

def zig_contact_forces(
    positions: np.ndarray,
    dhat: float = 1e-3,
    kappa: float = 1e4
) -> np.ndarray:
    """
    Zero-allocation IPC barrier contact force evaluation in Zig.
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    _validate_positive(dhat, "dhat")
    _validate_positive(kappa, "kappa")

    pos = _prepare_positions(positions)
    N = len(pos)
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    fx = np.zeros(N, dtype=np.float32)
    fy = np.zeros(N, dtype=np.float32)
    fz = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_contact_forces(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(dhat),
        ctypes.c_float(kappa),
        fx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return np.column_stack([fx, fy, fz])

# -----------------------------------------------------------------------------
# 2D Complex Multipole APIs (Exact Greengard-Rokhlin Laurent Expansion)
# -----------------------------------------------------------------------------

def zig_2d_p2m(
    positions: np.ndarray,
    charges: np.ndarray,
    center: Tuple[float, float],
    order: int = 6
) -> np.ndarray:
    """
    Computes 2D Laurent multipole moments around center (cx, cy).
    returns: (order + 1,) complex128 array matching core.tree_free_fmm.p2m
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    if not 1 <= order <= 8:
        raise ValueError("order must be between 1 and 8")

    pos, q = _prepare_particles(positions, charges)
    N = len(pos)
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    cx, cy = float(center[0]), float(center[1])

    out_re = np.zeros(order + 1, dtype=np.float32)
    out_im = np.zeros(order + 1, dtype=np.float32)

    _LIB.zig_fmm_2d_p2m(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(cx),
        ctypes.c_float(cy),
        ctypes.c_uint32(order),
        out_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_re.astype(np.float64) + 1j * out_im.astype(np.float64)

def zig_2d_m2l(
    m_coeffs: np.ndarray,
    src_center: Tuple[float, float],
    dst_center: Tuple[float, float],
    order: int = 6
) -> np.ndarray:
    """
    Translates 2D multipole moments from src_center to dst_center.
    returns: (order + 1,) complex128 array matching core.tree_free_fmm.m2l
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    if not 1 <= order <= 8:
        raise ValueError("order must be between 1 and 8")

    m_c = np.asarray(m_coeffs, dtype=np.complex64)
    if m_c.ndim != 1 or len(m_c) < order + 1:
        raise ValueError(f"m_coeffs must be a 1D array with at least {order + 1} entries")
    src_re = np.ascontiguousarray(np.real(m_c[:order + 1]), dtype=np.float32)
    src_im = np.ascontiguousarray(np.imag(m_c), dtype=np.float32)

    out_re = np.zeros(order + 1, dtype=np.float32)
    out_im = np.zeros(order + 1, dtype=np.float32)

    _LIB.zig_fmm_2d_m2l(
        src_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        src_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(float(src_center[0])),
        ctypes.c_float(float(src_center[1])),
        ctypes.c_float(float(dst_center[0])),
        ctypes.c_float(float(dst_center[1])),
        ctypes.c_uint32(order),
        out_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_re.astype(np.float64) + 1j * out_im.astype(np.float64)

def zig_2d_l2p(
    local_coeffs: np.ndarray,
    center: Tuple[float, float],
    target_positions: np.ndarray,
    order: int = 6
) -> np.ndarray:
    """
    Evaluates 2D potential from local Taylor series at target points.
    returns: (N,) float32 potentials
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")
    if not 1 <= order <= 8:
        raise ValueError("order must be between 1 and 8")

    pos = _prepare_positions(target_positions)
    N = len(pos)
    tx = np.ascontiguousarray(pos[:, 0])
    ty = np.ascontiguousarray(pos[:, 1])

    l_c = np.asarray(local_coeffs, dtype=np.complex64)
    if l_c.ndim != 1 or len(l_c) < order + 1:
        raise ValueError(f"local_coeffs must be a 1D array with at least {order + 1} entries")
    l_re = np.ascontiguousarray(np.real(l_c[:order + 1]), dtype=np.float32)
    l_im = np.ascontiguousarray(np.imag(l_c), dtype=np.float32)

    out_pot = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_2d_l2p(
        l_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        l_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(order),
        ctypes.c_float(float(center[0])),
        ctypes.c_float(float(center[1])),
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        out_pot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_pot

# -----------------------------------------------------------------------------
# 3D Cartesian Multipole APIs (1/r Gravitational / Coulomb / UE5)
# -----------------------------------------------------------------------------

def zig_3d_p2m(
    positions: np.ndarray,
    masses: np.ndarray,
    center: Tuple[float, float, float]
) -> Dict[str, Any]:
    """
    Computes 3D Cartesian multipole moments (M0, D, Q) around center (cx, cy, cz).
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    pos, m = _prepare_particles(positions, masses)
    N = len(pos)
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    moments = FMMMultipole3D()

    _LIB.zig_fmm_3d_p2m(
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(cx),
        ctypes.c_float(cy),
        ctypes.c_float(cz),
        ctypes.byref(moments),
    )

    return {
        "m0": moments.m0,
        "d": np.array([moments.dx, moments.dy, moments.dz], dtype=np.float32),
        "q": np.array([
            [moments.qxx, moments.qxy, moments.qxz],
            [moments.qxy, moments.qyy, moments.qyz],
            [moments.qxz, moments.qyz, moments.qzz],
        ], dtype=np.float32),
        "_struct": moments,
    }

def zig_3d_m2p(
    moments_dict: Dict[str, Any],
    center: Tuple[float, float, float],
    target_positions: np.ndarray,
    softening: float = 1e-2
) -> np.ndarray:
    """
    Evaluates 3D Far-Field Potential (1/r) directly from multipole moments at targets.
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    _validate_positive(softening, "softening")
    pos = _prepare_positions(target_positions)
    N = len(pos)
    tx = np.ascontiguousarray(pos[:, 0])
    ty = np.ascontiguousarray(pos[:, 1])
    tz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    struct = moments_dict.get("_struct")
    if struct is None:
        struct = FMMMultipole3D(
            m0=float(moments_dict["m0"]),
            dx=float(moments_dict["d"][0]),
            dy=float(moments_dict["d"][1]),
            dz=float(moments_dict["d"][2]),
            qxx=float(moments_dict["q"][0, 0]),
            qyy=float(moments_dict["q"][1, 1]),
            qzz=float(moments_dict["q"][2, 2]),
            qxy=float(moments_dict["q"][0, 1]),
            qxz=float(moments_dict["q"][0, 2]),
            qyz=float(moments_dict["q"][1, 2]),
        )

    out_pot = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_3d_m2p(
        ctypes.byref(struct),
        ctypes.c_float(cx),
        ctypes.c_float(cy),
        ctypes.c_float(cz),
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        ctypes.c_float(softening ** 2),
        out_pot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_pot

def zig_3d_m2l(
    moments_dict: Dict[str, Any],
    src_center: Tuple[float, float, float],
    dst_center: Tuple[float, float, float],
    softening: float = 1e-2
) -> Dict[str, Any]:
    """
    Translates 3D multipole moments from src_center to dst_center into local Taylor expansion.
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    _validate_positive(softening, "softening")
    src_cx, src_cy, src_cz = float(src_center[0]), float(src_center[1]), float(src_center[2])
    dst_cx, dst_cy, dst_cz = float(dst_center[0]), float(dst_center[1]), float(dst_center[2])

    struct = moments_dict.get("_struct")
    if struct is None:
        struct = FMMMultipole3D(
            m0=float(moments_dict["m0"]),
            dx=float(moments_dict["d"][0]),
            dy=float(moments_dict["d"][1]),
            dz=float(moments_dict["d"][2]),
            qxx=float(moments_dict["q"][0, 0]),
            qyy=float(moments_dict["q"][1, 1]),
            qzz=float(moments_dict["q"][2, 2]),
            qxy=float(moments_dict["q"][0, 1]),
            qxz=float(moments_dict["q"][0, 2]),
            qyz=float(moments_dict["q"][1, 2]),
        )

    local = FMMLocal3D()

    _LIB.zig_fmm_3d_m2l(
        ctypes.byref(struct),
        ctypes.c_float(src_cx),
        ctypes.c_float(src_cy),
        ctypes.c_float(src_cz),
        ctypes.c_float(dst_cx),
        ctypes.c_float(dst_cy),
        ctypes.c_float(dst_cz),
        ctypes.c_float(softening ** 2),
        ctypes.byref(local),
    )

    return {
        "l0": local.l0,
        "g": np.array([local.gx, local.gy, local.gz], dtype=np.float32),
        "h": np.array([
            [local.hxx, local.hxy, local.hxz],
            [local.hxy, local.hyy, local.hyz],
            [local.hxz, local.hyz, local.hzz],
        ], dtype=np.float32),
        "_struct": local,
    }

def zig_3d_l2p(
    local_dict: Dict[str, Any],
    center: Tuple[float, float, float],
    target_positions: np.ndarray
) -> np.ndarray:
    """
    Evaluates 3D potential from local Taylor expansion at target points.
    """
    if _LIB is None:
        raise RuntimeError("Zig native engine not compiled or available.")

    pos = _prepare_positions(target_positions)
    N = len(pos)
    tx = np.ascontiguousarray(pos[:, 0])
    ty = np.ascontiguousarray(pos[:, 1])
    tz = np.ascontiguousarray(pos[:, 2]) if pos.shape[1] > 2 else np.zeros(N, dtype=np.float32)

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    struct = local_dict.get("_struct")
    if struct is None:
        struct = FMMLocal3D(
            l0=float(local_dict["l0"]),
            gx=float(local_dict["g"][0]),
            gy=float(local_dict["g"][1]),
            gz=float(local_dict["g"][2]),
            hxx=float(local_dict["h"][0, 0]),
            hyy=float(local_dict["h"][1, 1]),
            hzz=float(local_dict["h"][2, 2]),
            hxy=float(local_dict["h"][0, 1]),
            hxz=float(local_dict["h"][0, 2]),
            hyz=float(local_dict["h"][1, 2]),
        )

    out_pot = np.zeros(N, dtype=np.float32)

    _LIB.zig_fmm_3d_l2p(
        ctypes.byref(struct),
        ctypes.c_float(cx),
        ctypes.c_float(cy),
        ctypes.c_float(cz),
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(N),
        out_pot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out_pot
