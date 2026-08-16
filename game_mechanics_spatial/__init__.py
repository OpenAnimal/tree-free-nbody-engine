"""
Game Mechanics & Spatial Computing Suite (`game_mechanics_spatial`).
High-performance spatial algorithms, swarm navigation, procedural generation, and CAD mechanics.
"""

from .massive_crowd_flocking import MassiveGameCrowdEngine
from .line_of_sight_fog_of_war import GameLineOfSightRadar
from .procedural_map_generator import ProceduralMultipoleMapGenerator
from .smart_brush_lasso_selector import SmartBrushPointCloudSelector
from .fast_mesh_lod_decimator import FastMeshLODDecimator
from .harmonic_flow_field_pathfinding import HarmonicPotentialFlowField
from .wave_function_collapse_pcg import (
    WaveFunctionCollapse2D,
    create_dungeon_wfc_ruleset,
    create_biome_terrain_wfc_ruleset
)
from .procedural_dungeon_network import ProceduralDungeonSynthesizer

# Convenient standard aliases
LineOfSightFogOfWarEngine = GameLineOfSightRadar
SmartBrushLassoSelector = SmartBrushPointCloudSelector

__all__ = [
    "MassiveGameCrowdEngine",
    "GameLineOfSightRadar",
    "LineOfSightFogOfWarEngine",
    "ProceduralMultipoleMapGenerator",
    "SmartBrushPointCloudSelector",
    "SmartBrushLassoSelector",
    "FastMeshLODDecimator",
    "HarmonicPotentialFlowField",
    "WaveFunctionCollapse2D",
    "create_dungeon_wfc_ruleset",
    "create_biome_terrain_wfc_ruleset",
    "ProceduralDungeonSynthesizer",
]
