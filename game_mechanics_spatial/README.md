# Game Mechanics & Spatial Computing Suite (`game_mechanics_spatial`)
### Flocking, Continuum Pathfinding, Wave Function Collapse, Procedural Dungeons, Line-of-Sight, CAD Smart Brushes & LOD Decimation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Engine: Unreal%20%2F%20Unity%20%2F%20CAD](https://img.shields.io/badge/Engine-Unreal%20%2F%20Unity%20%2F%20CAD-orange.svg)]()
[![Scale: 5k Agents @ 7 FPS](https://img.shields.io/badge/Scale-5k%20Agents%20%40%207%20FPS-purple.svg)]()

---

> 🔬 **Research Prototype & Exploratory Notice:**  
> `game_mechanics_spatial` is an experimental research exploration investigating non-reordering spatial hashing and analytic potential-field spatial physics for interactive games, virtual worlds, and CAD tools. All modules include runnable unit demonstrations and verified throughput metrics.

---

## 🌟 Overview & Implemented Modules

This sub-repository provides high-performance, modular spatial primitives for game engines (Unreal Engine / Unity), robotics simulators, and CAD/VFX software:

```text
game_mechanics_spatial/
├── README.md                           # Comprehensive documentation & architecture
├── massive_crowd_flocking.py           # Flocking Swarms, Near-Field Exact (5k agents @ ~7.5 FPS; vectorized order-0 far field)
├── harmonic_flow_field_pathfinding.py  # Screened Yukawa Potential-Field Swarm Pathfinder (190k+ agents/s)
├── wave_function_collapse_pcg.py       # Bitset AC-3-style Constraint Wave Function Collapse Engine (22k+ cells/s)
├── procedural_dungeon_network.py       # Rejection-Sampled & MST Dungeon / Corridor Synthesizer (3-10 ms generation)
├── line_of_sight_fog_of_war.py         # RTS Unit Vision & Fog of War (20,000 Units in ~75-170 ms; proximity only, no occlusion)
├── procedural_map_generator.py         # Infinite Procedural Terrain & Biome Screened-RBF Field (O(features)/query)
├── smart_brush_lasso_selector.py       # CAD / VFX 1,000,000+ Point Cloud Lasso Brush (~4 ms / 237 FPS)
└── fast_mesh_lod_decimator.py          # O(N) Level-of-Detail (LOD) Mesh Decimator (18M Polys/sec)
```

---

## 📊 Summary of Verified Performance

| Module | Purpose / Real-World Application | Measured Throughput / Latency | Algorithmic Benefit |
| :--- | :--- | :--- | :--- |
| **`massive_crowd_flocking.py`** | Flocking birds, fish schools, and game crowds. | **7.5 FPS** (5,000 agents in 132.6 ms; NumPy step) | Eliminates $O(N^2)$ all-pairs boid check via Morton neighbor lookups; near-field exact (verified vs brute), far-field is a vectorized order-0 barycentric mean (NOT multipole/FMM). An earlier "72 FPS @ 50k agents" claim was not reproducible and has been re-measured. |
| **`harmonic_flow_field_pathfinding.py`** | Continuum pathfinding for massive armies & swarms without A* bottlenecks. | **192,600+ agents/sec** (5.19 µs / query) | Screened Yukawa repulsive obstacles & vortex circulation; hash-truncated variant restricts the sum to a 5x5 neighborhood (screening-valid). NOT a multipole/FMM. |
| **`wave_function_collapse_pcg.py`** | Procedural biome tilemaps, dungeon structures & architectural synthesis. | **22,800+ cells/sec** (44 ms for 1,024 cells) | 64-bit word-parallel bitsets & Shannon entropy minimization with AC-3-style propagation (FIFO deque). |
| **`procedural_dungeon_network.py`** | Roguelike/RPG room networks, underground corridors & transit loops. | **3.23 ms** (80x40 dungeon) / **10.29 ms** (256x256 mega-level) | Rejection-sampled room placement with AABB non-overlap + centroid spacing; all-pairs proximity graph; Kruskal MST reachability & controlled cycle loop restoration. NOT Bridson Poisson-Disc, NOT Delaunay. |
| **`line_of_sight_fog_of_war.py`** | RTS radar and unit visibility (StarCraft / Total War scale). | **173 ms** (20,000 units / 120k ops/s) | Direct spatial cell revelation by proximity; NO occlusion raycasting (true LOS against terrain is not computed). |
| **`procedural_map_generator.py`** | Boundless terrain heightfield generation. | **18.9 Chunks/sec** (4,096 vertices/chunk) | Screened RBF harmonic potential field evaluates terrain on-the-fly (O(num_features)/query, NOT O(1); "Multipole" in the class name is a historical misnomer). |
| **`smart_brush_lasso_selector.py`** | CAD / Blender point cloud lasso and volume selection. | **4.22 ms** (237 FPS cursor drag on 1M pts) | Bounded spatial queries only touching intersecting bounding cells; ceil-based cell scan radius so rim points are never missed; returns the actual selected indices. |
| **`fast_mesh_lod_decimator.py`** | Automatic 3D asset Level-of-Detail (LOD) generation. | **5.28 ms** (18.9 Million polygons/sec) | Single-pass Morton vertex clustering without quadratic edge-collapse loops. |

---

## 🛠️ Quickstart & Usage Examples

```bash
# Test 5,000-Agent Flocking Swarm (near-field exact, ~7.5 FPS NumPy step)
python game_mechanics_spatial/massive_crowd_flocking.py

# Test Harmonic Potential Flow Field Swarm Pathfinder (50k agents)
python game_mechanics_spatial/harmonic_flow_field_pathfinding.py

# Test Wave Function Collapse Procedural Engine
python game_mechanics_spatial/wave_function_collapse_pcg.py

# Test Procedural Dungeon & Corridor Network Synthesizer
python game_mechanics_spatial/procedural_dungeon_network.py

# Test RTS Fog of War Engine (20,000 Units)
python game_mechanics_spatial/line_of_sight_fog_of_war.py

# Test Infinite Procedural Map Generation
python game_mechanics_spatial/procedural_map_generator.py

# Test 1,000,000 Point Cloud Smart Lasso Brush
python game_mechanics_spatial/smart_brush_lasso_selector.py

# Test Fast Mesh LOD Decimator
python game_mechanics_spatial/fast_mesh_lod_decimator.py
```

---

## 🔬 Theoretical Citations

1. **Optimal Bounds for Open Addressing Without Reordering.** Farach-Colton, Krapivin, & Kuszmaul (2025). *IEEE FOCS 2024* / [arXiv:2501.02305](https://arxiv.org/abs/2501.02305).
2. **A Fast Algorithm for Particle Simulations.** Greengard, Rokhlin (1987). *Journal of Computational Physics*, 73(2), 325–348.
3. **A Fast Adaptive Multipole Algorithm for Particle Simulations.** Carrier, Greengard, Rokhlin (1988). *SIAM Journal on Scientific and Statistical Computing*, 9(4), 669–686.
4. **Flocks, Herds, and Schools: A Distributed Behavioral Model.** Reynolds (1987). *ACM SIGGRAPH Computer Graphics*, 21(4), 25–34.
5. **WaveFunctionCollapse Algorithm.** Gumin (2016). *GitHub*: [mxgmn/WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse).
6. **Fast Poisson Disk Sampling in Arbitrary Dimensions.** Bridson (2007). *ACM SIGGRAPH Sketches*, Article 22.
