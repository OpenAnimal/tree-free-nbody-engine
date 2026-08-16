"""
Video Streaming, Codecs & Multimodal Media Transport (`video_streaming_codecs`)
================================================================================
Hardware-optimized streaming video, 4D Gaussian Splatting, neuromorphic event streams,
macroblock compression, and biological telemetry media integration (FFmpeg / Matroska / LSL).
"""

from .volumetric_gaussian_stream import GaussianSplat4DStreamer
from .neuromorphic_event_stream import NeuromorphicStreamReconstructor
from .low_bitrate_humanitarian import LowBitrateSemanticCodec
from .frame_deduplicator import VideoFrameDeduplicator
from .video_motion_heatmap import VideoMotionHeatmap
from .lockfree_motion_estimation import LockFreeMotionEstimator
from .greedy_macroblock_merger import GreedyMacroblockMerger
from .one_euro_video_stabilizer import OneEuroVideoStabilizer, AdaptiveBitrateController
from .biosignal_media_stream import (
    BiosignalMediaStreamMuxer,
    TimedBiosignalPacket,
    MultiplexedMediaStreamReport
)
from .scenecut_gop_analyzer import (
    SceneCutGOPAnalyzer,
    SceneTransitionType,
    FrameSceneMetadata,
    SceneCutSummaryReport
)
from .perceptual_rate_controller import (
    PerceptualRateController,
    PerceptualRateControlResult
)
from .spatial_dct_entropy_codec import SpatialDCTCodec, CompressedBitstreamPacket
from .parametric_noise_field_codec import (
    ParametricNoiseFieldDescriptor,
    NoiseFieldAnalysisResult,
    ParametricNoiseFieldAnalyzer,
    ParametricNoiseFieldSynthesizer,
    FilmGrainParameters,
    GrainAnalysisResult,
    FilmGrainAnalyzer,
    FilmGrainSynthesizer,
)
from .adaptive_hls_dash_segmenter import (
    VideoRendition,
    MediaSegment,
    ABRStreamingManifests,
    AdaptiveStreamingSegmenter
)
from .ffmpeg_interop_bridge import (
    HardwareEncoderProfile,
    EncodingPipelinePlan,
    FFmpegInteropBridge
)

# Standard aliases
VolumetricGaussianStreamer = GaussianSplat4DStreamer
LowBitrateHumanitarianCodec = LowBitrateSemanticCodec
VideoMotionHeatmapGenerator = VideoMotionHeatmap

__all__ = [
    "GaussianSplat4DStreamer",
    "VolumetricGaussianStreamer",
    "NeuromorphicStreamReconstructor",
    "LowBitrateSemanticCodec",
    "LowBitrateHumanitarianCodec",
    "VideoFrameDeduplicator",
    "VideoMotionHeatmap",
    "VideoMotionHeatmapGenerator",
    "LockFreeMotionEstimator",
    "GreedyMacroblockMerger",
    "OneEuroVideoStabilizer",
    "AdaptiveBitrateController",
    "BiosignalMediaStreamMuxer",
    "TimedBiosignalPacket",
    "MultiplexedMediaStreamReport",
    "SceneCutGOPAnalyzer",
    "SceneTransitionType",
    "FrameSceneMetadata",
    "SceneCutSummaryReport",
    "PerceptualRateController",
    "PerceptualRateControlResult",
    "SpatialDCTCodec",
    "CompressedBitstreamPacket",
    "ParametricNoiseFieldDescriptor",
    "NoiseFieldAnalysisResult",
    "ParametricNoiseFieldAnalyzer",
    "ParametricNoiseFieldSynthesizer",
    "FilmGrainParameters",
    "GrainAnalysisResult",
    "FilmGrainAnalyzer",
    "FilmGrainSynthesizer",
    "VideoRendition",
    "MediaSegment",
    "ABRStreamingManifests",
    "AdaptiveStreamingSegmenter",
    "HardwareEncoderProfile",
    "EncodingPipelinePlan",
    "FFmpegInteropBridge",
]
