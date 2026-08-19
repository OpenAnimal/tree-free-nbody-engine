"""
Infinite Procedural Map & Terrain Biome Generator.
Screened-RBF potential-field terrain synthesis over macro features.

Generates boundless terrain elevation without storing tile arrays. Each query
costs O(num_macro_features) work (vectorized), not O(1). NOTE: "Multipole" in
the class name is a historical misnomer — the field is a plain screened radial
basis sum; there is no multipole expansion and no FMM here, and the elastic
hash adds nothing to this purely arithmetic generator (it was removed).
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class ProceduralMultipoleMapGenerator:
    """
    Infinite Terrain & Biome Generator using a screened RBF field (see module docstring).
    """
    def __init__(self, seed: int = 42, num_macro_features: int = 256):
        np.random.seed(seed)
        # Macro terrain poles (mountains, oceans, volcano hotspots)
        self.feature_centers = np.random.uniform(0, 1000, size=(num_macro_features, 2)).astype(np.float32)
        self.feature_amplitudes = np.random.uniform(-50.0, 100.0, size=num_macro_features).astype(np.float32)

    def query_terrain_chunk(self, chunk_x: float, chunk_y: float, chunk_size: int = 64) -> Dict:
        """
        Evaluates terrain heightfield over a (chunk_size x chunk_size) window (O(features), vectorized).
        """
        t0 = time.perf_counter()
        gx = np.linspace(chunk_x, chunk_x + 100.0, chunk_size)
        gy = np.linspace(chunk_y, chunk_y + 100.0, chunk_size)
        X, Y = np.meshgrid(gx, gy)
        grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1) # (N, 2)
        
        # Screened RBF terrain height evaluation
        diff = grid_pts[:, None, :] - self.feature_centers[None, :, :]
        r2 = np.sum(diff**2, axis=-1)
        # Screened RBF harmonic terrain height
        kernel = np.exp(-r2 / (200.0**2))
        elevation = np.sum(kernel * self.feature_amplitudes[None, :], axis=-1).reshape(chunk_size, chunk_size)
        
        t_gen = (time.perf_counter() - t0) * 1000.0
        
        return {
            "chunk_size": chunk_size,
            "latency_ms": t_gen,
            "min_elevation": float(np.min(elevation)),
            "max_elevation": float(np.max(elevation)),
            "fps_chunk_capacity": 1000.0 / max(1e-3, t_gen)
        }

def run_procedural_map_demo():
    print("==================================================================")
    print(" GAME MECHANICS: INFINITE PROCEDURAL MAP GENERATION (SCREENED RBF FIELD)")
    print("==================================================================")
    print(f"Generating 64x64 terrain chunk (4,096 vertices) on-the-fly...")
    
    map_gen = ProceduralMultipoleMapGenerator(num_macro_features=500)
    stats = map_gen.query_terrain_chunk(chunk_x=350.0, chunk_y=420.0, chunk_size=64)
    
    print(f"[-] Chunk Generation Time:    {stats['latency_ms']:.2f} ms")
    print(f"[-] Chunk Streaming FPS:      {stats['fps_chunk_capacity']:.1f} Chunks/sec")
    print(f"[-] Elevation Range:          [{stats['min_elevation']:.1f} m, {stats['max_elevation']:.1f} m]")

if __name__ == '__main__':
    run_procedural_map_demo()

ProceduralRBFMapGenerator = ProceduralMultipoleMapGenerator  # renamed alias
