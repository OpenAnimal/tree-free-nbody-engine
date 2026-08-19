"""Standardized variant benchmark for Application 2 (hydrodynamic vortex field).

Variants:
  standard -- exact direct Biot-Savart velocity on the Eulerian probe grid
              (O(N_grid * N_vortices) -- the reference the app cross-checks)
  +fmm     -- the app's actual path: streamfunction psi evaluated on the grid
              by FastVectorizedFMM (2D log multipoles, funnel-hash cells),
              then velocity = (dpsi/dy, -dpsi/dx) by central finite differences

The +elastichash axis is OMITTED here with reason: the FMM IS the
hash-bucketed path for this kernel (the funnel-hash cell index is what the
FMM builds internally), so a separate "+elastichash" row would duplicate
+fmm rather than represent a distinct influence.

Accuracy vs `standard` on the per-grid-point velocity vector (rel L2).
The finite-difference grid spacing sets the floor on the achievable error;
that is reported in the note, not hidden.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _vortex_sheet(n_vortices: int = 400):
    """Same vortex sheet geometry as app2_hydrodynamics.simulate_vortex_sheet."""
    x = np.linspace(0.1, 0.9, n_vortices // 2)
    y1 = 0.6 + 0.03 * np.sin(4 * np.pi * x)
    gamma1 = np.ones_like(x) * 1.5
    y2 = 0.4 - 0.03 * np.sin(4 * np.pi * x)
    gamma2 = -np.ones_like(x) * 1.5
    pos = np.vstack([np.stack([x, y1], axis=1), np.stack([x, y2], axis=1)]).astype(np.float64)
    circ = np.concatenate([gamma1, gamma2]).astype(np.float64)
    return pos, circ


def _probe_grid(res: int = 80):
    gx = np.linspace(0.05, 0.95, res)
    gy = np.linspace(0.05, 0.95, res)
    X, Y = np.meshgrid(gx, gy)
    return X, Y, np.stack([X.ravel(), Y.ravel()], axis=1)


def _direct_biot_savart_grid(pos, circ, grid_pts):
    """Exact O(N_grid * N_vortices) Biot-Savart velocity at each grid point."""
    dx = grid_pts[:, 0:1] - pos[:, 0]
    dy = grid_pts[:, 1:2] - pos[:, 1]
    r2 = dx ** 2 + dy ** 2 + 1e-4
    u = -np.sum(circ * dy / r2, axis=1) / (2 * np.pi)
    v = np.sum(circ * dx / r2, axis=1) / (2 * np.pi)
    return np.stack([u, v], axis=1)


def _fmm_streamfunction_velocity(pos, circ, grid_pts, res, depth=5, order=8):
    """The app's path: FMM streamfunction on the grid, then central FD velocity."""
    from core.fast_vectorized_fmm import FastVectorizedFMM
    fmm = FastVectorizedFMM(depth=depth, order=order, softening=1e-4)
    all_pos = np.vstack([pos, grid_pts])
    all_q = np.concatenate([circ, np.zeros(len(grid_pts))])
    phi = fmm.evaluate(all_pos, all_q)[len(pos):]
    psi = -phi.reshape(res, res) / (2 * np.pi)
    gx = np.linspace(0.05, 0.95, res)
    gy = np.linspace(0.05, 0.95, res)
    u_grid, v_grid = np.gradient(psi, gy, gx)
    u_grid, v_grid = u_grid, -v_grid  # np.gradient(psi, y, x): first is d/dy
    return np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)


def run_app2_variants(n_vortices: int = 400, res: int = 80):
    pos, circ = _vortex_sheet(n_vortices=n_vortices)
    X, Y, grid_pts = _probe_grid(res=res)

    bench = VariantBenchmark(
        f"App 2 -- Hydrodynamic vortex sheet (N_vort={n_vortices}, "
        f"probe grid {res}x{res}, 2D log streamfunction kernel)"
    )
    bench.add(
        "standard (direct Biot-Savart)",
        lambda: _direct_biot_savart_grid(pos, circ, grid_pts),
        note="O(N_grid * N_vortices) exact reference",
    )
    bench.add(
        "+fmm (FMM streamfunction + FD)",
        lambda: _fmm_streamfunction_velocity(pos, circ, grid_pts, res, depth=5, order=8),
        accuracy_vs="standard (direct Biot-Savart)",
        note="FastVectorizedFMM psi on grid, velocity by central FD; "
             "error floor set by FD grid spacing (app tolerance 5e-2)",
    )
    return bench.run()


if __name__ == "__main__":
    run_app2_variants()
