"""
AV1/VVC Parametric Noise & Film Grain Synthesis Bridge (`film_grain_synthesizer.py`)
====================================================================================
Compatibility alias and bridge for `parametric_noise_field_codec.py`.
Provides standard codec names (FilmGrainParameters, FilmGrainAnalyzer, FilmGrainSynthesizer).
"""

from .parametric_noise_field_codec import (
    ParametricNoiseFieldDescriptor,
    FilmGrainParameters,
    NoiseFieldAnalysisResult,
    GrainAnalysisResult,
    ParametricNoiseFieldAnalyzer,
    FilmGrainAnalyzer,
    ParametricNoiseFieldSynthesizer,
    FilmGrainSynthesizer,
    run_parametric_noise_demo,
)

run_film_grain_demo = run_parametric_noise_demo

__all__ = [
    "ParametricNoiseFieldDescriptor",
    "FilmGrainParameters",
    "NoiseFieldAnalysisResult",
    "GrainAnalysisResult",
    "ParametricNoiseFieldAnalyzer",
    "FilmGrainAnalyzer",
    "ParametricNoiseFieldSynthesizer",
    "FilmGrainSynthesizer",
    "run_film_grain_demo",
    "run_parametric_noise_demo",
]

if __name__ == '__main__':
    run_film_grain_demo()
