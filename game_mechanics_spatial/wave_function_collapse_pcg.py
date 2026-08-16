"""
Wave Function Collapse (WFC) Procedural Content Generation Engine.
Fast Constraint Propagation (AC-4 / Min-Entropy) with Bitset-Accelerated Superposition.

Mathematical & Algorithmic Formulation:
- Wave Superposition: Each cell c in grid has a state vector W(c) in {0, 1}^K representing possible tile prototypes.
- Bitset Optimization: For K <= 64 prototypes, W(c) is packed into a uint64 bitmask, allowing bitwise AND/OR/popcount in O(1).
- Minimum Shannon Entropy Observation:
    H(c) = log(sum_{t in W(c)} w_t) - (sum_{t in W(c)} w_t * log(w_t)) / (sum_{t in W(c)} w_t) + noise
- Forward Constraint Propagation (AC-4):
    When cell c is collapsed to pattern p, adjacent neighbor n in direction d must satisfy:
    W(n) <- W(n) AND AllowedMask(p, d)
    If W(n) changes, push n to propagation queue. Contradiction occurs if W(n) == 0.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.elastic_hash import ElasticHashTable


# Direction offsets: (dx, dy) and corresponding opposite direction index
# 0: +X (Right), 1: -X (Left), 2: +Y (Down), 3: -Y (Up)
DIRECTIONS_2D = [
    (1, 0),   # 0: +X
    (-1, 0),  # 1: -X
    (0, 1),   # 2: +Y
    (0, -1)   # 3: -Y
]
OPPOSITE_DIR_2D = [1, 0, 3, 2]


class WaveFunctionCollapse2D:
    """
    High-Performance Bitset-Accelerated 2D Wave Function Collapse Engine.
    """
    def __init__(
        self,
        tile_names: List[str],
        tile_weights: Optional[List[float]] = None,
        periodic: bool = False
    ):
        self.tile_names = tile_names
        self.num_tiles = len(tile_names)
        if self.num_tiles > 64:
            raise ValueError(f"Bitset WFC currently supports up to 64 tile prototypes (received {self.num_tiles}).")

        if tile_weights is None:
            self.weights = np.ones(self.num_tiles, dtype=np.float64)
        else:
            self.weights = np.asarray(tile_weights, dtype=np.float64)

        self.periodic = periodic
        self.all_mask = (1 << self.num_tiles) - 1

        # Allowed transition masks: compatibility_mask[dir_idx][tile_idx] -> uint64 bitmask of allowed neighbor tiles
        # directions: 0:+X, 1:-X, 2:+Y, 3:-Y
        self.compatibility_mask = np.zeros((4, self.num_tiles), dtype=np.uint64)

        # Precomputed weight logs for fast Shannon entropy
        self.w_log_w = self.weights * np.log(np.maximum(1e-12, self.weights))

    def add_adjacency_rule(self, tile_a: str, dir_name: str, tile_b: str):
        """
        Adds a directional adjacency rule: tile_a placed next to tile_b in direction dir_name.
        dir_name in ['+x', '-x', '+y', '-y', 'right', 'left', 'down', 'up']
        """
        dir_map = {
            '+x': 0, 'right': 0, 'east': 0,
            '-x': 1, 'left': 1, 'west': 1,
            '+y': 2, 'down': 2, 'south': 2,
            '-y': 3, 'up': 3, 'north': 3
        }
        d_idx = dir_map[dir_name.lower()]
        opp_idx = OPPOSITE_DIR_2D[d_idx]

        idx_a = self.tile_names.index(tile_a)
        idx_b = self.tile_names.index(tile_b)

        # In direction d from A, B is allowed
        self.compatibility_mask[d_idx, idx_a] |= np.uint64(1 << idx_b)
        # In opposite direction from B, A is allowed
        self.compatibility_mask[opp_idx, idx_b] |= np.uint64(1 << idx_a)

    def collapse(self, width: int, height: int, max_restarts: int = 20, seed: Optional[int] = None) -> Dict:
        """
        Executes WFC generation over a (width x height) grid.
        Returns: Dict with collapsed tile map and generation statistics.
        """
        if seed is not None:
            np.random.seed(seed)

        t0 = time.perf_counter()

        for attempt in range(max_restarts):
            # 1. Initialize Wave Superposition Grid
            # wave_masks: (H, W) uint64
            wave = np.full((height, width), self.all_mask, dtype=np.uint64)
            num_remaining = np.full((height, width), self.num_tiles, dtype=np.int32)
            
            # Entropy cache
            sum_weights = np.sum(self.weights)
            sum_w_log_w = np.sum(self.w_log_w)
            base_entropy = np.log(sum_weights) - (sum_w_log_w / sum_weights)

            entropies = np.full((height, width), base_entropy, dtype=np.float64)
            # Add micro-noise to break entropy ties uniformly
            noise = np.random.uniform(1e-6, 1e-4, size=(height, width))
            entropies += noise

            collapsed_count = 0
            total_cells = width * height
            contradiction = False

            while collapsed_count < total_cells:
                # 2. Select Unobserved Cell with Minimal Shannon Entropy
                # Mask out already collapsed cells (num_remaining == 1) or invalid cells
                uncollapsed = (num_remaining > 1)
                if not np.any(uncollapsed):
                    break

                min_val = np.min(entropies[uncollapsed])
                min_candidates = np.argwhere((entropies == min_val) & uncollapsed)
                
                # Pick randomly among candidates with minimal entropy
                chosen_idx = np.random.randint(len(min_candidates))
                cy, cx = min_candidates[chosen_idx]

                # 3. Collapse (Observe) Selected Cell
                cell_mask = int(wave[cy, cx])
                possible_tiles = [t for t in range(self.num_tiles) if (cell_mask & (1 << t))]
                
                if not possible_tiles:
                    contradiction = True
                    break

                tile_probs = self.weights[possible_tiles]
                tile_probs /= np.sum(tile_probs)
                selected_tile = np.random.choice(possible_tiles, p=tile_probs)

                # Set collapsed state
                wave[cy, cx] = np.uint64(1 << selected_tile)
                num_remaining[cy, cx] = 1
                entropies[cy, cx] = float('inf')
                collapsed_count += 1

                # 4. Propagate Constraints (AC-4 Wavefront)
                prop_queue = [(cx, cy)]

                while prop_queue:
                    qx, qy = prop_queue.pop(0)
                    src_mask = int(wave[qy, qx])

                    for d_idx, (dx, dy) in enumerate(DIRECTIONS_2D):
                        nx, ny = qx + dx, qy + dy

                        if self.periodic:
                            nx = nx % width
                            ny = ny % height
                        else:
                            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                                continue

                        if num_remaining[ny, nx] <= 1:
                            continue

                        # Compute union of allowed patterns from src cell
                        allowed_for_neighbor = np.uint64(0)
                        for t in range(self.num_tiles):
                            if (src_mask & (1 << t)):
                                allowed_for_neighbor |= self.compatibility_mask[d_idx, t]

                        curr_mask = wave[ny, nx]
                        new_mask = curr_mask & allowed_for_neighbor

                        if new_mask != curr_mask:
                            if new_mask == 0:
                                contradiction = True
                                break

                            wave[ny, nx] = new_mask
                            # Count remaining possible states via bit popcount
                            rem = bin(int(new_mask)).count('1')
                            num_remaining[ny, nx] = rem

                            if rem == 1:
                                entropies[ny, nx] = float('inf')
                                collapsed_count += 1
                            else:
                                # Recompute Shannon entropy for cell
                                p_tiles = [t for t in range(self.num_tiles) if (int(new_mask) & (1 << t))]
                                w_sub = self.weights[p_tiles]
                                w_sum = np.sum(w_sub)
                                w_log_sum = np.sum(self.w_log_w[p_tiles])
                                entropies[ny, nx] = np.log(w_sum) - (w_log_sum / w_sum) + noise[ny, nx]

                            prop_queue.append((nx, ny))

                    if contradiction:
                        break

                if contradiction:
                    break

            if not contradiction and collapsed_count == total_cells:
                # Successful synthesis!
                t_total = (time.perf_counter() - t0) * 1000.0
                # Decode wave to tile grid
                result_grid = np.zeros((height, width), dtype=np.int32)
                for y in range(height):
                    for x in range(width):
                        mask = int(wave[y, x])
                        tile_idx = (mask & -mask).bit_length() - 1 if mask > 0 else 0
                        result_grid[y, x] = tile_idx

                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "latency_ms": t_total,
                    "grid": result_grid,
                    "tile_names": [self.tile_names[i] for i in result_grid.ravel()],
                    "cells_per_sec": (total_cells / max(1e-6, t_total)) * 1000.0
                }

        # If all attempts had contradictions
        t_total = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "attempts": max_restarts,
            "latency_ms": t_total,
            "grid": None,
            "cells_per_sec": 0.0
        }


def create_dungeon_wfc_ruleset() -> WaveFunctionCollapse2D:
    """
    Creates a standard Dungeon & Architecture WFC ruleset:
    Tiles: Void, Floor, Wall, Corridor, Door, Treasure
    """
    tiles = ["VOID", "FLOOR", "WALL", "CORRIDOR", "DOOR", "TREASURE"]
    # Weights: Void (common), Floor (common), Wall (common), Corridor (moderate), Door (rare), Treasure (rare)
    weights = [3.0, 5.0, 4.0, 3.0, 1.0, 0.5]
    wfc = WaveFunctionCollapse2D(tile_names=tiles, tile_weights=weights)

    # 1. Self-adjacencies (Homogeneous regions)
    for t in ["VOID", "FLOOR", "WALL", "CORRIDOR"]:
        for d in ['+x', '-x', '+y', '-y']:
            wfc.add_adjacency_rule(t, d, t)

    # 2. Wall boundaries between Floor/Corridor and Void
    for d in ['+x', '-x', '+y', '-y']:
        wfc.add_adjacency_rule("WALL", d, "FLOOR")
        wfc.add_adjacency_rule("WALL", d, "VOID")
        wfc.add_adjacency_rule("WALL", d, "CORRIDOR")

    # 3. Doors connect Rooms (Floor) to Corridors
    for d in ['+x', '-x', '+y', '-y']:
        wfc.add_adjacency_rule("DOOR", d, "FLOOR")
        wfc.add_adjacency_rule("DOOR", d, "CORRIDOR")
        wfc.add_adjacency_rule("DOOR", d, "WALL")

    # 4. Treasure rooms spawn inside floor interiors
    for d in ['+x', '-x', '+y', '-y']:
        wfc.add_adjacency_rule("TREASURE", d, "FLOOR")

    return wfc


def create_biome_terrain_wfc_ruleset() -> WaveFunctionCollapse2D:
    """
    Creates an Organic Biome & Terrain WFC ruleset:
    Deep Ocean -> Shallow Water -> Sand Beach -> Grass Plain -> Forest -> Mountain Peak
    """
    tiles = ["OCEAN", "WATER", "SAND", "GRASS", "FOREST", "MOUNTAIN"]
    weights = [4.0, 3.0, 2.5, 5.0, 3.5, 1.5]
    wfc = WaveFunctionCollapse2D(tile_names=tiles, tile_weights=weights)

    # Self adjacencies
    for t in tiles:
        for d in ['+x', '-x', '+y', '-y']:
            wfc.add_adjacency_rule(t, d, t)

    # Continuous elevation transitions (Ocean <-> Water <-> Sand <-> Grass <-> Forest <-> Mountain)
    transitions = [
        ("OCEAN", "WATER"),
        ("WATER", "SAND"),
        ("SAND", "GRASS"),
        ("GRASS", "FOREST"),
        ("FOREST", "MOUNTAIN"),
        ("GRASS", "MOUNTAIN")
    ]
    for t1, t2 in transitions:
        for d in ['+x', '-x', '+y', '-y']:
            wfc.add_adjacency_rule(t1, d, t2)

    return wfc


def run_wfc_demo():
    print("==================================================================")
    print(" GAME MECHANICS: WAVE FUNCTION COLLAPSE (WFC) PROCEDURAL GENERATOR")
    print("==================================================================")

    # 1. Biome & Natural Map Generation Demo
    print("[1/2] Generating Procedural Biome World Map (40x25 = 1,000 Cells)...")
    biome_wfc = create_biome_terrain_wfc_ruleset()
    res_biome = biome_wfc.collapse(width=40, height=25, seed=123)

    if res_biome["success"]:
        print(f"[-] Synthesis Succeeded in {res_biome['attempts']} attempt(s)")
        print(f"[-] Generation Latency: {res_biome['latency_ms']:.2f} ms")
        print(f"[-] Generation Throughput: {res_biome['cells_per_sec']:,.0f} cells/sec")
        
        # ASCII Visualization of Biome Map
        char_map = {0: '~', 1: '.', 2: 'o', 3: '"', 4: '#', 5: '^'}
        print("\n--- Generated Procedural Biome ASCII Map ---")
        grid = res_biome["grid"]
        for row in grid[:15]: # display first 15 rows
            print("".join(char_map.get(c, '?') for c in row))
        print("-------------------------------------------\n")

    # 2. Procedural Dungeon Room & Corridor Architecture Demo
    print("[2/2] Generating Procedural Dungeon Map (32x32 = 1,024 Cells)...")
    dungeon_wfc = create_dungeon_wfc_ruleset()
    res_dungeon = dungeon_wfc.collapse(width=32, height=32, seed=456)

    if res_dungeon["success"]:
        print(f"[-] Dungeon Synthesis Succeeded in {res_dungeon['attempts']} attempt(s)")
        print(f"[-] Generation Latency: {res_dungeon['latency_ms']:.2f} ms")
        print(f"[-] Generation Throughput: {res_dungeon['cells_per_sec']:,.0f} cells/sec")

        dungeon_chars = {0: ' ', 1: '.', 2: '#', 3: '=', 4: '+', 5: '$'}
        print("\n--- Generated Dungeon Layout ASCII Map ---")
        d_grid = res_dungeon["grid"]
        for row in d_grid[:16]:
            print("".join(dungeon_chars.get(c, '?') for c in row))
        print("------------------------------------------")
    print("==================================================================")


if __name__ == '__main__':
    run_wfc_demo()
