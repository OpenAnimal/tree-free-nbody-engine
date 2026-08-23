"""Input contract checks for neural_ops spatial operators.

The spatial operators quantize coordinates to a unit cell grid
``floor(coords * grid_res)`` over ``[0, 1)^dims``; coordinates outside that
range are CLIPPED at the boundary. That clipping is silent in plain numpy,
and rescaled inputs (e.g. coords in [0, 10)) collapse many distinct cells
onto the boundary cells, producing plausible-looking but wrong outputs —
the exact "looks-working-but-wrong" failure class this package refuses.

``check_unit_coords`` emits a ``RuntimeWarning`` (once per call site by
default) whenever coordinates fall outside ``[0, 1)^dims`` beyond float
noise, telling the caller how to normalize. It does not raise: existing
pipelines that deliberately feed slightly-out-of-range values keep
working, just no longer silently.
"""

import warnings
from typing import Optional

import numpy as np

_WARNED: set = set()


def check_unit_coords(coords: np.ndarray, name: str = "coords",
                      once_per_site: bool = True) -> bool:
    """Warn if ``coords`` materially escapes ``[0, 1)^dims``.

    Parameters
    ----------
    coords : array (N, dims) — or (N,) for 1D operators.
    name : identifier used in the warning message (typically the operator
        and argument, e.g. ``"TreeFreeMultipoleAttention(coords)"``).
    once_per_site : warn only once per ``name`` per process (default).

    Returns True if all coordinates are inside the unit domain (within
    1e-9 tolerance), False otherwise. The function never raises and never
    modifies the input; the caller's own clipping behavior is unchanged.
    """
    arr = np.asarray(coords, dtype=np.float64)
    tol = 1e-9
    outside = (arr < -tol) | (arr >= 1.0 + tol)
    if not np.any(outside):
        return True
    if once_per_site and name in _WARNED:
        return False
    _WARNED.add(name)
    n_out = int(np.count_nonzero(outside))
    lo, hi = float(np.min(arr)), float(np.max(arr))
    warnings.warn(
        f"{name}: {n_out}/{arr.size} coordinate values fall outside the "
        f"[0,1)^dims unit domain this operator quantizes onto (observed "
        f"range [{lo:.4g}, {hi:.4g}]). Out-of-range values are CLIPPED to "
        f"the boundary cells, which collapses distinct positions and "
        f"silently degrades the result. Normalize first, e.g. "
        f"coords = (coords - coords.min(0)) / np.ptp(coords, axis=0), "
        f"or min-max onto [0, 1 - eps].",
        RuntimeWarning,
        stacklevel=3,
    )
    return False
