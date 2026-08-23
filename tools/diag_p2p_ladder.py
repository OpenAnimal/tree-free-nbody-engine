#!/usr/bin/env python
"""Round-7 task T-E3: 5M P2P optimization ladder — sweep protocol generator.

The GPU P2P pass is the measured 86% hotspot at 5M particles (GPU_NOTES §4:
Main Compute = 59.28 ms of 68.57 ms total).  This script generates the
sweep protocol for the three-step optimization ladder documented in
GPU_NOTES.md §6.6:

  (a) leafBits sweep  — ?leafBits=6..9 at 1M/2M/5M, pin leafBitsForCount
  (b) LDS staging     — ?ldsStaging=1, measure occupancy impact
  (c) AABB cull       — ?aabbCull=1, report cull fractions

Each step is independently measured via the ?autoprint=1 60-frame rolling
mean protocol.  A documented non-improvement is a valid result (house law).

This script does NOT run the browser benchmark — it prints the URL
protocol and the expected baseline numbers so the MAIN agent (or a human)
can run the sweeps in a browser and record the results.

Usage:
    python tools/diag_p2p_ladder.py
"""

from __future__ import annotations
import sys
import os

# Baseline numbers from GPU_NOTES §4 (5M particles, round-6 measurement).
BASELINE_5M = {
    "main_compute_ms": 59.28,
    "total_ms": 68.57,
    "avg_step_ms": 39.8,
    "fmm_chain_ms": 7.5,
    "p2p_fraction": 0.86,
}

# leafBitsForCount auto-tune formula (from index.html):
#   targetSide = sqrt(N / 12)
#   bits = ceil(log2(targetSide))
#   clamped to [6, 10]
def leaf_bits_for_count(n: int) -> int:
    import math
    target_side = (n / 12.0) ** 0.5
    bits = math.ceil(math.log2(target_side)) if target_side > 1 else 6
    return max(6, min(10, bits))


def print_sweep_protocol():
    print("=" * 72)
    print("T-E3: 5M P2P Optimization Ladder — Sweep Protocol")
    print("=" * 72)
    print()
    print(f"Baseline (5M, round-6 §4):")
    for k, v in BASELINE_5M.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.2f}")
        else:
            print(f"  {k:25s} = {v}")
    print()

    # --- Step (a): leafBits sweep ---
    print("-" * 72)
    print("Step (a): leafBits sweep")
    print("-" * 72)
    print()
    print("Auto-tune formula: leafBitsForCount(n) = clamp(ceil(log2(sqrt(n/12))), 6, 10)")
    print()
    print(f"{'N':>10s} | {'auto bits':>10s} | {'side':>8s} | {'avg/cell':>10s} | URL")
    print("-" * 72)
    for n in [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
        bits = leaf_bits_for_count(n)
        side = 1 << bits
        avg_per_cell = n / (side * side)
        url = f"?N={n}&leafBits={bits}&autoprint=1"
        print(f"{n:>10,d} | {bits:>10d} | {side:>8d} | {avg_per_cell:>10.1f} | {url}")
    print()
    print("Sweep protocol: for each N in {1M, 2M, 5M}, run ?leafBits={6,7,8,9,10}&autoprint=1")
    print("and record the 60-frame Avg Step.  Pin leafBitsForCount to the measured optimum.")
    print("Acceptance: 5M Avg Step improves >= 15% vs baseline AND ?probe=1 far-field check")
    print("unchanged AND visual parity.")
    print()

    # --- Step (b): LDS staging ---
    print("-" * 72)
    print("Step (b): LDS staging")
    print("-" * 72)
    print()
    print("URL: ?N=5000000&ldsStaging=1&autoprint=1")
    print()
    print("Design: one workgroup per target leaf cell; stage each neighbor cell's")
    print("slice into var<workgroup> arrays (cap = p2pListCap, early-out beyond);")
    print("process one slice per barrier.  Removes ~9x redundant global reads per lane.")
    print("Risk: LDS pressure drops occupancy — measure both Avg Step and occupancy.")
    print()
    print("Acceptance: 5M Avg Step improves >= 15% AND occupancy does not drop > 30%.")
    print()

    # --- Step (c): AABB cull ---
    print("-" * 72)
    print("Step (c): AABB cull")
    print("-" * 72)
    print()
    print("URL: ?N=5000000&aabbCull=1&autoprint=1")
    print()
    print("Design: per-cell min/max extents (computed in a new kernel) -> skip neighbor")
    print("cells whose extent is beyond the P2P radius.  Report cull fractions from")
    print("telemetry (TELEM JSON gains a 'p2pCullFraction' field).")
    print()
    print("Acceptance: 5M Avg Step improves >= 15% AND cull fraction reported > 0.")
    print()

    # --- Summary ---
    print("=" * 72)
    print("Summary: each step = one browser session + one GPU_NOTES entry.")
    print("A documented non-improvement is a valid result (house law).")
    print("Never report a single-frame number — always use the 60-frame rolling mean.")
    print("=" * 72)


if __name__ == "__main__":
    print_sweep_protocol()
