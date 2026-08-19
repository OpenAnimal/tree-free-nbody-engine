"""
Repo-wide numeric validation convention.

Every approximate computation (hash-truncated, cluster-far-field, quantized)
is checked against an exact reference with `cross_validate`, and every demo
prints the result via `fmt_validation`. One convention, no per-file formats.
"""

from typing import Dict

import numpy as np


def cross_validate(approx: np.ndarray, exact: np.ndarray, name: str = "") -> Dict[str, float]:
    """
    Compares an approximation against an exact reference (same shape).
    Returns {'rel_l2', 'mean_abs', 'max_abs', 'cosine'}.
    """
    approx = np.asarray(approx, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    diff = approx - exact
    denom = max(1e-12, float(np.linalg.norm(exact)))
    cos = float(np.dot(approx.ravel(), exact.ravel()) /
                max(1e-12, np.linalg.norm(approx) * np.linalg.norm(exact)))
    out = {
        "rel_l2": float(np.linalg.norm(diff) / denom),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cos,
    }
    if name:
        out["name"] = name
    return out


def fmt_validation(res: Dict[str, float]) -> str:
    """Single canonical line for demo/test output."""
    name = res.get("name", "result")
    return (f"[valid] {name}: rel L2 = {res['rel_l2']:.3e} | mean abs = {res['mean_abs']:.3e} | "
            f"max abs = {res['max_abs']:.3e} | cosine = {res['cosine']:.6f}")


def assert_accuracy(res: Dict[str, float], rel_l2_tol: float, label: str = "") -> None:
    """Assert convention used by tests and demos with hard gates."""
    label = label or res.get("name", "result")
    assert res["rel_l2"] < rel_l2_tol, (
        f"{label}: rel L2 {res['rel_l2']:.3e} exceeds tolerance {rel_l2_tol:.1e}")
