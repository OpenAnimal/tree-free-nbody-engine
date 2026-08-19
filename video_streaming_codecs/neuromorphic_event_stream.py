"""
Neuromorphic Event-Camera Spatiotemporal Stream Reconstructor (Zero Frame-Rate Video).
Powered by Farach-Colton Non-Reordering Hashing & Spatiotemporal Morton Point Fields.

Processes asynchronous spike streams (x, y, t, polarity) from event sensors (DVS)
and accumulates per-cell event density without frame buffers; the elastic hash
is the authoritative active-cell index. NOTE: earlier claims of reconstructing
optical flow and intensity gradients in O(1) were untrue - this module
accumulates event density only, in O(N) per batch.
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.spatial_index import CellIndex

class NeuromorphicStreamReconstructor:
    """
    Continuous Event-Camera Video Reconstructor & Spatiotemporal Velocity Evaluator.
    """
    def __init__(self, width: int = 1280, height: int = 720, time_window_us: float = 20000.0):
        self.width = width
        self.height = height
        self.time_window_us = time_window_us
        self.grid_x = width // 8
        self.grid_y = height // 8

        # Non-reordering rolling space-time hash table (world mode on raw pixels)
        self.index = CellIndex(dims=2, cell_size=8.0)
        self.event_density = np.zeros((self.grid_y, self.grid_x), dtype=np.float32)

    def process_event_batch(self, events_x: np.ndarray, events_y: np.ndarray, events_t: np.ndarray, events_p: np.ndarray) -> Dict:
        """
        Ingests batch of N microsecond events:
        events_x, events_y: pixel coordinates
        events_t: microsecond timestamps
        events_p: polarity (+1 / -1)
        """
        t0 = time.perf_counter()
        N = len(events_x)

        # Build the authoritative active-cell index from raw pixel coordinates
        # (world mode: floor(coord/8) + 512). Density is accumulated per occupied
        # cell. The hash is rebuilt per batch (append-only tables cannot unlearn
        # stale keys). Pixel coords are clipped to the sensor extent first,
        # matching the old code's clip(events//8, 0, grid-1) semantics.
        cx = np.clip(events_x.astype(np.float64), 0, self.width - 1)
        cy = np.clip(events_y.astype(np.float64), 0, self.height - 1)
        positions = np.stack([cx, cy], axis=1)
        unique_keys, inverse = self.index.build(positions)
        counts = np.bincount(inverse, minlength=len(unique_keys))
        for k in unique_keys:
            ix, iy = self.index.key_ints(int(k))
            self.event_density[iy - 512, ix - 512] += counts[self.index.cell_id(int(k))]

        t_elapsed = (time.perf_counter() - t0) * 1000.0

        return {
            "num_events": N,
            "latency_ms": t_elapsed,
            "events_per_sec": N / max(1e-6, t_elapsed / 1000.0),
            "active_spatiotemporal_cells": len(unique_keys)
        }

    def is_cell_active(self, event_x: float, event_y: float) -> bool:
        """Active-cell membership test via the authoritative elastic hash."""
        cx = float(np.clip(event_x, 0, self.width - 1))
        cy = float(np.clip(event_y, 0, self.height - 1))
        return self.index.cell_id(self.index.key_of(np.array([cx, cy]))) is not None

    def validate_density(self, events_x, events_y, n_samples: int = 200) -> Dict:
        """Cross-check accumulated density against brute-force recounting."""
        bx = np.clip(events_x // 8, 0, self.grid_x - 1)
        by = np.clip(events_y // 8, 0, self.grid_y - 1)
        rng = np.random.default_rng(9)
        idx = rng.choice(len(events_x), size=min(n_samples, len(events_x)), replace=False)
        mismatches = 0
        all_keys = (by << 12) | bx
        for i in idx:
            b = int(all_keys[i])
            brute = int(np.sum(all_keys == b))
            ok_val = (brute == int(self.event_density[int(by[i]), int(bx[i])])
                      and self.is_cell_active(events_x[i], events_y[i]))
            mismatches += (not ok_val)
        return {"sampled": len(idx), "mismatches": mismatches}

def run_neuromorphic_demo():
    print("==================================================================")
    print(" VIDEO STREAMING: NEUROMORPHIC EVENT-CAMERA STREAM RECONSTRUCTOR")
    print("==================================================================")
    N_EVENTS = 500000
    print(f"Streaming {N_EVENTS:,} asynchronous microsecond event spikes...")
    
    np.random.seed(42)
    ev_x = np.random.randint(0, 1280, size=N_EVENTS)
    ev_y = np.random.randint(0, 720, size=N_EVENTS)
    ev_t = np.sort(np.random.uniform(0, 20000, size=N_EVENTS)) # 20 ms window
    ev_p = np.random.choice([-1, 1], size=N_EVENTS)
    
    reconstructor = NeuromorphicStreamReconstructor(width=1280, height=720)
    stats = reconstructor.process_event_batch(ev_x, ev_y, ev_t, ev_p)
    
    print(f"[-] Event Ingestion Time:     {stats['latency_ms']:.2f} ms")
    print(f"[-] Event Processing Speed:   {stats['events_per_sec']/1e6:.2f} Million Events/sec")
    print(f"[-] Active Spatio-Cells:      {stats['active_spatiotemporal_cells']:,}")
    val = reconstructor.validate_density(ev_x, ev_y)
    print(f"[-] Density Cross-Check:      {val['mismatches']}/{val['sampled']} mismatches (must be 0)")
    assert val['mismatches'] == 0

if __name__ == '__main__':
    run_neuromorphic_demo()
