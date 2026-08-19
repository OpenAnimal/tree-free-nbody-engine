"""
Instant Video Heatmap & Motion Highlight Accumulator.
Plain numpy spatial binning of motion-vector magnitudes.

Computes 2D activity/traffic heatmaps for security cams, gaming replays, and
sports streams. NOTE: no hash table and no FMM apply here — accumulation is
array slicing; the decorative ElasticHashTable instance and "lock-free
hashing" claims of earlier revisions were removed as untrue.
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class VideoMotionHeatmap:
    """
    Real-Time 2D Video Motion Heatmap Generator.
    """
    def __init__(self, grid_w: int = 64, grid_h: int = 36):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    def accumulate_motion_vectors(self, motion_vectors: np.ndarray):
        """
        motion_vectors: (H_blocks, W_blocks, 2)
        """
        # Calculate velocity magnitude
        speed = np.linalg.norm(motion_vectors, axis=-1)
        # Downsample/bin into heatmap grid
        h_in, w_in = speed.shape
        step_y = max(1, h_in // self.grid_h)
        step_x = max(1, w_in // self.grid_w)
        
        binned_speed = speed[::step_y, ::step_x][:self.grid_h, :self.grid_w]
        self.heatmap += binned_speed

def run_heatmap_demo():
    print("==================================================================")
    print(" VIDEO STREAMING: INSTANT VIDEO MOTION HEATMAP & HIGHLIGHT DETECTOR")
    print("==================================================================")
    N_FRAMES = 1000
    print(f"Accumulating motion heatmaps across {N_FRAMES:,} video frames...")
    
    heatmap_gen = VideoMotionHeatmap(grid_w=64, grid_h=36)
    np.random.seed(42)
    
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        # Synthetic motion vectors (e.g. traffic corridor in center)
        mvs = np.zeros((72, 128, 2), dtype=np.float32)
        mvs[30:45, :, 0] = np.random.uniform(5, 15, size=(15, 128))
        heatmap_gen.accumulate_motion_vectors(mvs)
        
    t_acc = (time.perf_counter() - t0) * 1000.0
    print(f"[-] Heatmap Processing Time:  {t_acc:.2f} ms ({N_FRAMES / (t_acc/1000.0):,.0f} Frames/sec)")
    print(f"[-] Peak Motion Zone Value:   {np.max(heatmap_gen.heatmap):.1f}")

if __name__ == '__main__':
    run_heatmap_demo()
