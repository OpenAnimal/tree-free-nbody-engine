"""
Same-Scene Benchmark: Matrix-Free Tree-Free IPC vs ZOZO PPF Contact Solver

X-P4 task: a same-scene, same-timestep-count benchmark against
st-tech/ppf-contact-solver, NOT against our own naive baseline.

THE SCENE (replicating PPF's examples/headless.py):
  - 5 triangulated cloth sheets (res=64 => 4,096 verts each, 7,938 tris each)
    stacked along x with 0.25 spacing, pinned at the +y edge,
    strain-limit 0.05, young-mod ~1000, bend ~10.
  - 1 icosphere (r=0.5, subdiv=4 => 2,562 verts) as a static collider,
    initially at (-1, 0, 0), moving to (7, 0, 0) over t=[0, 5].
  - dt = 0.01, 60 frames, min 8 Newton steps.

WHAT PPF RUNS (measured on this machine, RTX 4070, 2026-08-21):
  - 23,042 vertices, 44,810 triangles total
  - 102 actual timesteps (TOI sub-stepping from 60 frames)
  - Total wall-clock: ~53 sec, avg ~520 msec/frame
  - GPU/CUDA/Rust solver, cubic barrier, full CCD
  - See the PPF stdout.log for per-frame breakdown.

WHAT OUR SOLVER RUNS (this script):
  - Same mesh topology (5 sheets res=64 + sphere approximation)
  - Same dt=0.01, 60 frames (no TOI sub-stepping -- our solver has no CCD)
  - CPU/NumPy, log-barrier, vertex-vertex contact only

HONEST COMPARISON CAVEATS (read before interpreting):
  1. PPF is a GPU/CUDA/Rust solver; ours is CPU/NumPy. Wall-clock
     comparison is apples-to-oranges and PPF will be faster in absolute
     time. The point is to measure the GAP, not to claim a win.
  2. PPF uses a cubic barrier (ACM TOG 2024) with full CCD; ours uses a
     log-barrier with vertex-vertex contact only (no point-triangle CCD).
     Our penetration-free guarantee is weaker.
  3. PPF sub-steps via TOI (time-of-impact) when needed; our solver takes
     fixed dt=0.01 steps. PPF's 102 actual timesteps vs our 60 means PPF
     does more work per frame for robustness. The speedup ratio is
     computed as (wall_clock / sim_time) for both solvers to account for
     this.
  4. PPF's icosphere is a real triangle mesh collider; our solver uses an
     analytic sphere obstacle (point-to-sphere-center distance). This is
     an approximation but captures the same contact physics.
  5. PPF pins mesh vertices grabbed at a spatial location; our solver
     pins by index. We approximate the pinned edge. Pinning is applied
     POST-HOC (restore position + zero velocity after solve_step), not as
     Dirichlet BCs during the Newton-PCG solve. This is weaker than PPF's
     in-solve Dirichlet handling and may inflate our CG iteration count.
  6. The sphere moves from (-1,0,0) to (7,0,0) over t=[0,5], but 60
     frames at dt=0.01 = 0.6 sec sim time, so the sphere only moves 0.96
     units (from -1 to -0.04). It NEVER reaches the sheets (at x=0..1.0)
     in 60 frames. Both solvers only see sheet-sheet gravity drape
     contacts, not sphere-sheet contacts. The sphere is a future collider
     that would only engage after ~500 frames (5 sec sim time).
  7. PPF uses FEM with Young's modulus (young-mod=1000, bend=10.0); our
     solver uses edge-spring (k_stretch=3000) + discrete hinge (k_bend=
     0.5). These are NOT the same parameterization. We tuned our values
     to be stable at dt=0.01, not to match PPF's material response.

Run:  python physics_simulation/ppf_contact_solver_fmm/benchmark_vs_ppf.py
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from physics_simulation.ppf_contact_solver_fmm.matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes,
)


# -----------------------------------------------------------------------
# PPF headless scene parameters (from examples/headless.py)
# -----------------------------------------------------------------------
PPF_SHEET_RES = 64          # res=64 => 64x64 = 4,096 verts per sheet
PPF_NUM_SHEETS = 5
PPF_SHEET_SPACING = 0.25    # sheets spaced 0.25 along x
PPF_SPHERE_RADIUS = 0.5
PPF_DT = 0.01
PPF_FRAMES = 60
PPF_MIN_NEWTON = 8

# Allow overriding via env vars for faster testing
if os.environ.get("BENCH_RES"):
    PPF_SHEET_RES = int(os.environ["BENCH_RES"])
if os.environ.get("BENCH_FRAMES"):
    PPF_FRAMES = int(os.environ["BENCH_FRAMES"])
if os.environ.get("BENCH_NEWTON"):
    PPF_MIN_NEWTON = int(os.environ["BENCH_NEWTON"])
# PPF material params (from drape.ipynb, headless uses defaults):
#   young-mod=1000, bend=10.0, strain-limit=0.05
# Our solver uses k_stretch (spring constant) and k_bend (hinge constant).
# These are NOT the same parameterization -- PPF uses FEM with Young's modulus,
# ours uses edge-spring + discrete hinge. We pick values that give similar
# visual stiffness (tuned to not explode at dt=0.01).
OUR_K_STRETCH = 3000.0
OUR_K_BEND = 0.5
OUR_DENSITY = 0.25
OUR_DHAT = 0.02             # contact barrier distance
OUR_STIFFNESS = 5e3         # barrier stiffness


def build_ppf_headless_scene():
    """Build a scene approximating PPF's headless.py: 5 sheets + sphere.

    Returns (cloth, sphere_center, sphere_radius, pinned_indices).
    The sphere starts at (-1, 0, 0) and moves to (7, 0, 0) over t=[0, 5].
    """
    sheets = []
    pinned_all = []

    for i in range(PPF_NUM_SHEETS):
        # PPF: square(res=64, ex=[0,0,1], ey=[0,1,0]) => sheet in the ZY plane
        # centered at origin, then placed at (i*0.25, 0, 0).
        # Our create_cloth_grid makes a sheet in the XY plane; we rotate
        # it to the ZY plane by swapping axes: (x, y, z) -> (x, z, y).
        # Sheet spans [-0.5, 0.5] in y and z (PPF square is unit-sized).
        sheet = create_cloth_grid(
            nx=PPF_SHEET_RES, ny=PPF_SHEET_RES,
            width=1.0, height=1.0,
            center=(i * PPF_SHEET_SPACING, 0.0, 0.0),
            k_stretch=OUR_K_STRETCH,
            k_bend=OUR_K_BEND,
            density=OUR_DENSITY,
        )
        # Rotate from XY plane to ZY plane: (x, y, z) -> (x, z, y)
        # PPF uses ex=[0,0,1], ey=[0,1,0] so the sheet spans Z and Y.
        # Our grid spans X and Y at z=0. PPF's sheet is at x=const (vertical).
        # Mapping: new_x = center, new_y = old_y, new_z = old_x - center.
        # This is a 90° rotation around Y + translation (rigid transform),
        # so hinge weights computed from the original positions are still
        # valid (distances and angles are preserved).
        # NOTE: combine_cloth_meshes rebuilds topology from rest_positions,
        # so the hinge weights will be recomputed from the rotated positions
        # anyway — the manual rest_length recomputation below is redundant
        # but harmless (it gets discarded by combine).
        pos = sheet.rest_positions.copy()
        pos[:, 2] = pos[:, 0] - (i * PPF_SHEET_SPACING)  # dx -> z
        pos[:, 0] = i * PPF_SHEET_SPACING                  # x = center
        sheet.rest_positions = pos

        sheets.append(sheet)

        # PPF pins obj.grab([0, 1, 0]) -- the +y edge.
        # In our grid, +y edge = last row of vertices (indices ny-1, 2*ny-1, ...)
        # Actually grab([0,1,0]) grabs vertices near the +y direction from center.
        # For a sheet centered at (cx, 0, 0) spanning y in [-0.5, 0.5],
        # the +y edge is at y=+0.5, which is the last row: indices [ny*(ny-1) .. ny*ny-1]
        # but with our offset, vertex (i, j) is at index j*nx + i.
        # The +y edge (y=+0.5) is j=ny-1: indices [(ny-1)*nx .. ny*nx - 1].
        # But these are LOCAL indices; we need GLOBAL indices after combine.
        # We'll collect local pinned and offset them later.
        # PPF grab([0,1,0]) grabs the edge at y=+0.5 relative to sheet center.
        # In our grid that's the last row.
        local_pinned = np.arange(PPF_SHEET_RES - 1, PPF_SHEET_RES * PPF_SHEET_RES,
                                 PPF_SHEET_RES, dtype=np.int32)
        # Actually: row j=ny-1 has indices [(ny-1)*nx, (ny-1)*nx+1, ..., ny*nx-1]
        local_pinned = np.arange((PPF_SHEET_RES - 1) * PPF_SHEET_RES,
                                 PPF_SHEET_RES * PPF_SHEET_RES, dtype=np.int32)
        pinned_all.append(local_pinned)

    cloth = combine_cloth_meshes(sheets)

    # Offset pinned indices to global
    global_pinned = []
    offset = 0
    for i, lp in enumerate(pinned_all):
        global_pinned.append(lp + offset)
        offset += sheets[i].num_vertices
    global_pinned = np.concatenate(global_pinned)

    sphere_center = np.array([-1.0, 0.0, 0.0])
    sphere_radius = PPF_SPHERE_RADIUS

    return cloth, sphere_center, sphere_radius, global_pinned


def run_our_solver_benchmark():
    """Run our solver on the PPF headless scene and report timing."""
    print("=" * 80)
    print("SAME-SCENE BENCHMARK: Our Solver vs PPF (headless.py scene)")
    print("=" * 80)

    cloth, sphere_c0, sphere_r, pinned = build_ppf_headless_scene()
    print(f"Scene: {PPF_NUM_SHEETS} sheets res={PPF_SHEET_RES} + sphere r={sphere_r}")
    print(f"  Vertices: {cloth.num_vertices:,}  (PPF: 23,042)")
    print(f"  Triangles: {cloth.num_faces:,}  (PPF: 44,810)")
    print(f"  Pinned vertices: {len(pinned)}")
    print(f"  dt={PPF_DT}, frames={PPF_FRAMES}")
    print()

    # Our solver does NOT support pinned indices in solve_step directly.
    # We handle pinning by zeroing velocity and restoring position after each step.
    solver = MatrixFreeIPCSolver(
        dhat=OUR_DHAT,
        stiffness=OUR_STIFFNESS,
        max_newton_iters=PPF_MIN_NEWTON,
        cg_max_iters=32,
        cg_tol=1e-4,
        damp_coef=0.15,
    )
    solver.add_sphere_obstacle(center=sphere_c0, radius=sphere_r)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, -2.0]),
                              normal=np.array([0.0, 0.0, 1.0]))

    x = cloth.rest_positions.copy()
    v = np.zeros_like(x)
    # Store pinned rest positions
    x_pinned = x[pinned].copy()

    gravity = np.array([0.0, 0.0, -9.81])
    frame_times = []
    min_clearances = []

    def _min_clearance(pos, solv):
        """Min distance over candidate pairs + sphere + plane obstacles.

        NOTE: this calls find_broadphase_candidates again, which adds ~broadphase_ms
        of overhead per frame NOT included in the frame_ms timing.  The clearance
        is for reporting only; it does not inflate the measured solve time.
        """
        md = float("inf")
        cands = solv.find_broadphase_candidates(pos, cloth)
        if len(cands) > 0:
            dists = np.linalg.norm(pos[cands[:, 0]] - pos[cands[:, 1]], axis=-1)
            md = min(md, float(np.min(dists)))
        for sphere in solv.spheres:
            gap = float(np.min(np.linalg.norm(pos - sphere["center"], axis=-1)
                               - sphere["radius"]))
            md = min(md, gap)
        for plane in solv.planes:
            gap = float(np.min(np.sum((pos - plane["point"]) * plane["normal"], axis=-1)))
            md = min(md, gap)
        return md

    print(f"Running {PPF_FRAMES} frames...")
    t_total0 = time.perf_counter()

    for frame in range(PPF_FRAMES):
        # Move sphere from (-1, 0, 0) to (7, 0, 0) over t=[0, 5]
        t = frame * PPF_DT
        sphere_pos = sphere_c0 + np.array([min(8.0, 8.0 * t / 5.0), 0.0, 0.0])
        solver.spheres[0]["center"] = sphere_pos

        t_frame0 = time.perf_counter()
        x, v, metrics = solver.solve_step(x, v, cloth, dt=PPF_DT, gravity=gravity)
        frame_ms = (time.perf_counter() - t_frame0) * 1000.0
        frame_times.append(frame_ms)

        # Apply pinning: restore pinned positions, zero their velocity
        x[pinned] = x_pinned
        v[pinned] = 0.0

        mc = _min_clearance(x, solver)
        min_clearances.append(mc)

        if (frame + 1) % 10 == 0 or frame == 0:
            print(f"  Frame {frame + 1:3d}/{PPF_FRAMES}: {frame_ms:8.1f} ms  "
                  f"(broadphase {metrics.get('broadphase_ms', 0):.1f} ms, "
                  f"cg_iters {metrics.get('total_cg_iters', 0)}, "
                  f"min_clear {mc:.4f})")

    total_sec = time.perf_counter() - t_total0
    avg_ms = np.mean(frame_times)
    min_ms = np.min(frame_times)
    max_ms = np.max(frame_times)
    sim_time_sec = PPF_FRAMES * PPF_DT

    # PPF baseline (measured on this machine, RTX 4070, 2026-08-21)
    ppf_total_sec = 53.0
    ppf_frames = 60
    ppf_substeps = 102
    ppf_sim_time = ppf_frames * PPF_DT  # 0.6 sec
    ppf_per_substep_ms = 520.0
    ppf_per_sim_sec = ppf_total_sec / ppf_sim_time  # 88.3x realtime
    our_per_sim_sec = total_sec / sim_time_sec       # e.g. 1010x realtime
    speedup_ratio = our_per_sim_sec / ppf_per_sim_sec

    print()
    print("=" * 80)
    print("RESULTS: Our Solver (CPU/NumPy, log-barrier, vertex-vertex)")
    print("=" * 80)
    print(f"  Frames completed:     {PPF_FRAMES}")
    print(f"  Sim time:             {sim_time_sec:.2f} sec ({PPF_FRAMES} x dt={PPF_DT})")
    print(f"  Total wall-clock:     {total_sec:.2f} sec")
    print(f"  Average per-frame:    {avg_ms:.1f} ms")
    print(f"  Min frame:            {min_ms:.1f} ms")
    print(f"  Max frame:            {max_ms:.1f} ms")
    print(f"  Min clearance:        {min(min_clearances):.6f} (must be > 0)")
    print(f"  Realtime factor:      {our_per_sim_sec:.1f}x (wall/sim)")
    print()
    print("PPF BASELINE (GPU/CUDA/Rust, cubic barrier, full CCD):")
    print(f"  Frames requested:     {ppf_frames} (sub-stepped to {ppf_substeps} via TOI)")
    print(f"  Sim time:             {ppf_sim_time:.2f} sec ({ppf_frames} x dt={PPF_DT})")
    print(f"  Total wall-clock:     ~{ppf_total_sec:.0f} sec")
    print(f"  Average per-sub-step: ~{ppf_per_substep_ms:.0f} ms (over {ppf_substeps} sub-steps)")
    print(f"  Realtime factor:      {ppf_per_sim_sec:.1f}x (wall/sim)")
    print()
    print("HONEST INTERPRETATION:")
    # Direction-correct wording: speedup_ratio = (our wall/sim) / (PPF wall/sim).
    # > 1 -> PPF is faster by that factor; < 1 -> OUR solver is faster by the
    # reciprocal (only meaningful for the full 60-frame run; short env-var
    # override runs have different per-frame warmup, see WARNING below).
    if speedup_ratio >= 1.0:
        print(f"  PPF is ~{speedup_ratio:.1f}x FASTER (wall-clock per unit sim time).")
    else:
        print(f"  OUR solver is ~{1.0 / max(speedup_ratio, 1e-9):.1f}x faster "
              f"(wall-clock per unit sim time) -- unexpected for the full run; "
              f"check the frame-count WARNING below.")
    print(f"  This compares total wall-clock / total sim time for both solvers,")
    print(f"  accounting for PPF's TOI sub-stepping (102 sub-steps for 60 frames).")
    print(f"  NOTE: our solver ran {PPF_FRAMES} frames; PPF ran {ppf_frames}.")
    if PPF_FRAMES < ppf_frames:
        print(f"  WARNING: frame counts differ ({PPF_FRAMES} vs {ppf_frames});")
        print(f"  the ratio may change with a full {ppf_frames}-frame run.")
    print(f"  This is expected: PPF runs on GPU/CUDA; ours runs on CPU/NumPy.")
    print(f"  The algorithmic advantages of our approach (no BVH rebuild,")
    print(f"   no CSR assembly) are complexity arguments that would need a")
    print(f"   GPU implementation to manifest in wall-clock time.")
    print(f"  Our solver also has a weaker penetration guarantee (no CCD).")
    print("=" * 80)

    return {
        "total_sec": total_sec,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "frames": PPF_FRAMES,
        "min_clearance": min(min_clearances),
    }


if __name__ == "__main__":
    run_our_solver_benchmark()
