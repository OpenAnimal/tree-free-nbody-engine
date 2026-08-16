"""
Graphics & Real-Time Rendering Suite (`graphics_rendering`)
Point-Based Global Illumination, Surfel Radiosity, Volumetric AO & Gridless Irradiance Caching.
Powered by Tree-Free Fast Multipole Method (FMM) and Elastic Spatial Hashing.
"""

from .surfel_radiosity_gi import SurfelRadiosityGI, Surfel
from .volumetric_fmm_ao import VolumetricFMMAmbientOcclusion
from .dynamic_irradiance_cache import DynamicIrradianceCache

__all__ = [
    "SurfelRadiosityGI",
    "Surfel",
    "VolumetricFMMAmbientOcclusion",
    "DynamicIrradianceCache",
]
