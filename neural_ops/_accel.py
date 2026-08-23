"""
Backend acceleration shim for `neural_ops` (`neural_ops/_accel.py`).
====================================================================
Tri-backend dispatch (NumPy reference / PyTorch JIT / JAX JIT) for the
dense per-bucket linear algebra inside the neural operator layers.

Design constraints (read before editing):
- The spatial **bucketing** (`_bucketing.build_cell_index` ->
  `core.spatial_index.CellIndex`, the funnel-hash-backed cell index) is
  **CPU-only by design**. `core/jax_tree_free_fmm.py:17` documents that
  JAX with x64-disabled cannot express the 64-bit funnel mixer, and the
  funnel hash stays CPU/Zig/WGSL only. So the accel backends never touch
  bucketing: the forward loops still build the cell index on CPU and gather
  per-bucket NumPy arrays; only the per-bucket dense math (matmul / exp /
  einsum / sum) is moved to the device backend.
- The per-bucket kernels are extracted as **branch-free pure functions** so
  that `torch.compile` / `jax.jit` can compile them cleanly (no
  data-dependent control flow inside the compiled region). The
  "are there far clusters?" branch stays in the Python forward loop.
- NumPy stays the default reference path. The torch/jax paths are validated
  against it by `test_backend_parity.py` (rel-L2 + cosine + timing).

AGENTS.md JAX/GPU rule: when running interactively we must not let JAX
preallocate the default 75-90% VRAM and OOM background training. JAX only
honors `XLA_PYTHON_CLIENT_*` env vars at import time, so this module cannot
retroactively clamp them; instead `warn_jax_gpu_prealloc()` prints a notice
if a CUDA/ROCm JAX backend is detected without the preallocate guard set.
On the CPU JAX wheel (the current `aivenv` setup) this is a no-op.
"""

from __future__ import annotations
import os
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection (mirrors the HAS_TORCH / HAS_JAX pattern already used in
# neural_ops/autograd_adjoint_fmm.py and core/jax_tree_free_fmm.py).
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import jax
    # Do NOT call jax.config.update("jax_enable_x64", False) here.
    # core/jax_tree_free_fmm.py sets jax_enable_x64=True (required for adaptive FMM
    # complex128 coefficients). jax.config is process-wide, so setting it
    # False here would override that and silently break FMM tracing if
    # _accel is imported after jax_tree_free_fmm. Our layers always cast to
    # float32 explicitly, so the x64 flag has no effect on our computations.
    import jax.numpy as jnp
    from jax import jit as _jax_jit
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    _jax_jit = None
    HAS_JAX = False


VALID_BACKENDS = ("numpy", "torch", "jax")


def has_backend(name: str) -> bool:
    """True if the named backend is importable in this process."""
    if name == "numpy":
        return True
    if name == "torch":
        return HAS_TORCH
    if name == "jax":
        return HAS_JAX
    return False


def resolve_backend(backend: Optional[str]) -> str:
    """Resolve a requested backend to one of VALID_BACKENDS, with fallback.

    `None` picks torch if available (the user's primary accel target), else
    jax if available, else numpy. An explicit request for an unavailable
    backend falls back to numpy with a warning rather than raising, so the
    layer still runs (parity harness can then compare only the available
    subset).
    """
    if backend is None:
        for cand in ("torch", "jax", "numpy"):
            if has_backend(cand):
                return cand
        return "numpy"
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")
    if not has_backend(backend):
        warnings.warn(
            f"backend {backend!r} not available (not installed); falling back to numpy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "numpy"
    return backend


# ---------------------------------------------------------------------------
# Device selection.
# ---------------------------------------------------------------------------
_TORCH_DEVICE: Optional[Any] = None


def torch_device() -> Any:
    """Cached torch device (CUDA if available, else CPU)."""
    global _TORCH_DEVICE
    if _TORCH_DEVICE is None:
        if HAS_TORCH and torch.cuda.is_available():
            _TORCH_DEVICE = torch.device("cuda")
        else:
            _TORCH_DEVICE = torch.device("cpu")
    return _TORCH_DEVICE


def warn_jax_gpu_prealloc() -> None:
    """Print a notice if a GPU JAX backend is active without the
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `MEM_FRACTION` guard from
    AGENTS.md. No-op on CPU JAX."""
    if not HAS_JAX:
        return
    try:
        kinds = {d.platform for d in jax.devices()}
    except Exception:
        return
    if not (kinds & {"gpu", "cuda", "rocm"}):
        return
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false":
        return
    if os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""):
        return
    warnings.warn(
        "JAX GPU backend detected without XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "or XLA_PYTHON_CLIENT_MEM_FRACTION set (see AGENTS.md). JAX may "
        "preallocate 75-90%% VRAM and OOM background training sessions.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Tensor converters.
# ---------------------------------------------------------------------------
def as_torch(arr: Any, dtype: Optional[Any] = None) -> Any:
    """NumPy/Python -> contiguous torch tensor on the cached device."""
    if not HAS_TORCH:
        raise RuntimeError("torch not available")
    if isinstance(arr, torch.Tensor):
        t = arr
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr))
    t = t.to(torch_device())
    if dtype is not None:
        t = t.to(dtype)
    return t


def as_jax(arr: Any, dtype: Optional[Any] = None) -> Any:
    """NumPy/Python -> jax array (lives on jax.devices()[0])."""
    if not HAS_JAX:
        raise RuntimeError("jax not available")
    a = jnp.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    return a


def as_numpy(arr: Any) -> np.ndarray:
    """Any backend array -> contiguous NumPy ndarray (CPU)."""
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if HAS_JAX and isinstance(arr, jnp.ndarray):
        return np.asarray(arr)
    return np.ascontiguousarray(arr)


def to_backend(arr: Any, backend: str, dtype: Optional[Any] = None) -> Any:
    """Convert a NumPy/Python array to the named backend's array type."""
    if backend == "numpy":
        a = np.asarray(arr)
        return a.astype(dtype) if dtype is not None else a
    if backend == "torch":
        return as_torch(arr, dtype=dtype)
    if backend == "jax":
        return as_jax(arr, dtype=dtype)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Backend array namespaces.
#
# Each namespace exposes the small set of duck-typed ops the per-bucket
# kernels use, normalized so the kernel body is identical across backends.
# ---------------------------------------------------------------------------
class _NumpyNS:
    backend = "numpy"
    float32 = np.float32

    @staticmethod
    def asarray(a):
        return np.asarray(a)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def matmul(a, b):
        return np.matmul(a, b)

    @staticmethod
    def einsum(sub, *operands):
        return np.einsum(sub, *operands)

    @staticmethod
    def sum(x, axis=-1, keepdims=False):
        return x.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def to_float32(x):
        return x.astype(np.float32)

    @staticmethod
    def bool_to_float32(x):
        return x.astype(np.float32)

    @staticmethod
    def index(x, idx):
        return x[idx]

    @staticmethod
    def index_set(x, idx, val):
        x[idx] = val
        return x


class _TorchNS:
    backend = "torch"
    float32 = None  # set below after torch import

    @staticmethod
    def asarray(a):
        if isinstance(a, torch.Tensor):
            return a.to(torch_device())
        return as_torch(a)

    @staticmethod
    def zeros(shape, dtype=None):
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(shape, dtype=dtype, device=torch_device())

    @staticmethod
    def exp(x):
        return torch.exp(x)

    @staticmethod
    def clip(x, lo, hi):
        return torch.clamp(x, lo, hi)

    @staticmethod
    def matmul(a, b):
        return torch.matmul(a, b)

    @staticmethod
    def einsum(sub, *operands):
        return torch.einsum(sub, *operands)

    @staticmethod
    def sum(x, axis=-1, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims)

    @staticmethod
    def sqrt(x):
        return torch.sqrt(x)

    @staticmethod
    def maximum(a, b):
        return torch.maximum(a, b)

    @staticmethod
    def to_float32(x):
        return x.to(torch.float32)

    @staticmethod
    def bool_to_float32(x):
        return x.to(torch.float32)

    @staticmethod
    def index(x, idx):
        return x[idx]

    @staticmethod
    def index_set(x, idx, val):
        x[idx] = val
        return x


class _JaxNS:
    backend = "jax"
    float32 = None  # set below after jax import

    @staticmethod
    def asarray(a):
        return jnp.asarray(a)

    @staticmethod
    def zeros(shape, dtype=None):
        if dtype is None:
            dtype = jnp.float32
        return jnp.zeros(shape, dtype=dtype)

    @staticmethod
    def exp(x):
        return jnp.exp(x)

    @staticmethod
    def clip(x, lo, hi):
        return jnp.clip(x, lo, hi)

    @staticmethod
    def matmul(a, b):
        return jnp.matmul(a, b)

    @staticmethod
    def einsum(sub, *operands):
        return jnp.einsum(sub, *operands)

    @staticmethod
    def sum(x, axis=-1, keepdims=False):
        return x.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(x):
        return jnp.sqrt(x)

    @staticmethod
    def maximum(a, b):
        return jnp.maximum(a, b)

    @staticmethod
    def to_float32(x):
        return x.astype(jnp.float32)

    @staticmethod
    def bool_to_float32(x):
        return x.astype(jnp.float32)

    @staticmethod
    def index(x, idx):
        return x[idx]

    @staticmethod
    def index_set(x, idx, val):
        # jax arrays are immutable -> functional update.
        return x.at[idx].set(val)


if HAS_TORCH:
    _TorchNS.float32 = torch.float32
if HAS_JAX:
    _JaxNS.float32 = jnp.float32


def get_ns(backend: str):
    """Return the array namespace for the named backend."""
    if backend == "numpy":
        return _NumpyNS
    if backend == "torch":
        if not HAS_TORCH:
            raise RuntimeError("torch not available")
        return _TorchNS
    if backend == "jax":
        if not HAS_JAX:
            raise RuntimeError("jax not available")
        return _JaxNS
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Compiled-kernel cache.
#
# `torch.compile` and `jax.jit` both cache internally per input shape, so we
# compile each kernel once per backend and reuse the returned callable across
# all buckets (it re-specializes per shape on its own). Set jit=False to bypass
# compilation (useful for debugging parity divergences without recompilation
# noise).
#
# torch.compile backend selection (no Triton on this machine):
#   - `inductor` (the default, real codegen) requires Triton, which is not
#     installed in `aivenv` and is unreliable on Windows for torch 2.6. When
#     Triton is missing we fall back to `aot_eager` (ahead-of-time tracing
#     with eager execution -- no codegen, but still captures the graph and
#     avoids some per-op Python overhead). Override with the env var
#     NEURAL_OPS_TORCH_BACKEND inductor|aot_eager|cudagraphs|eager.
#   - Measured on the RTX 4070 SUPER: for *small* per-bucket kernels eager is
#     faster than aot_eager/cudagraphs (compile/capture overhead dominates).
#     torch.compile only wins for larger fused ops. The parity harness
#     reports both eager and compiled timings so the crossover is visible.
# ---------------------------------------------------------------------------
_TORCH_COMPILE_BACKEND: Optional[str] = None
_TORCH_COMPILE_WARNED: bool = False


def _resolve_torch_compile_backend() -> str:
    """Pick the torch.compile backend, honoring the env override and falling
    back from `inductor` to `aot_eager` when Triton is unavailable."""
    global _TORCH_COMPILE_WARNED
    forced = os.environ.get("NEURAL_OPS_TORCH_BACKEND", "").strip().lower()
    if forced:
        return forced
    try:
        import triton  # noqa: F401
        return "inductor"
    except ImportError:
        if not _TORCH_COMPILE_WARNED:
            warnings.warn(
                "Triton not installed: torch.compile will use the 'aot_eager' "
                "backend (no codegen). Set NEURAL_OPS_TORCH_BACKEND=inductor "
                "after installing triton, or =eager to skip compile.",
                RuntimeWarning,
                stacklevel=2,
            )
            _TORCH_COMPILE_WARNED = True
        return "aot_eager"


_COMPILE_CACHE: Dict[Tuple[str, str, bool], Callable] = {}


def get_compiled(backend: str, name: str, fn: Callable, jit: bool = True) -> Callable:
    """Return a (possibly JIT-compiled) version of `fn` for `backend`.

    Parameters
    ----------
    backend : "numpy" | "torch" | "jax"
    name    : cache key (e.g. "mpa_near") so distinct kernels don't collide.
    fn      : branch-free pure function over backend arrays (uses the ns ops).
    jit     : if False, return `fn` unchanged (no torch.compile / jax.jit).
    """
    key = (backend, name, jit)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    if not jit or backend == "numpy":
        out = fn
    elif backend == "torch":
        cb = _resolve_torch_compile_backend()
        out = torch.compile(fn, backend=cb) if cb != "eager" else fn
    elif backend == "jax":
        out = _jax_jit(fn)
    else:
        out = fn
    _COMPILE_CACHE[key] = out
    return out


def clear_compile_cache() -> None:
    """Drop all cached compiled kernels (e.g. between parity runs)."""
    _COMPILE_CACHE.clear()


# ---------------------------------------------------------------------------
# Backend status string (for the parity harness header).
# ---------------------------------------------------------------------------
def status_line() -> str:
    parts = [f"numpy={np.__version__}"]
    if HAS_TORCH:
        dev = torch_device()
        parts.append(f"torch={torch.__version__}({dev})")
    else:
        parts.append("torch=missing")
    if HAS_JAX:
        try:
            devs = ",".join(d.platform for d in jax.devices())
        except Exception:
            devs = "?"
        parts.append(f"jax={jax.__version__}({devs})")
    else:
        parts.append("jax=missing")
    return " | ".join(parts)
