"""
Comprehensive Unit & Integration Test Suite for Video Streaming, Codecs & Multimodal Media (`video_streaming_codecs`).
"""

import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_streaming_codecs import (
    GaussianSplat4DStreamer,
    NeuromorphicStreamReconstructor,
    LowBitrateSemanticCodec,
    VideoFrameDeduplicator,
    VideoMotionHeatmap,
    LockFreeMotionEstimator,
    GreedyMacroblockMerger,
    OneEuroVideoStabilizer,
    AdaptiveBitrateController,
    BiosignalMediaStreamMuxer,
    SceneCutGOPAnalyzer,
    PerceptualRateController,
    ParametricNoiseFieldAnalyzer,
    ParametricNoiseFieldSynthesizer,
    FilmGrainAnalyzer,
    FilmGrainSynthesizer,
    AdaptiveStreamingSegmenter,
    SpatialDCTCodec,
    FFmpegInteropBridge
)

def test_gaussian_splat_streamer():
    print("[+] Testing GaussianSplat4DStreamer...")
    n_gaussians = 1000
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.05, 0.95, (n_gaussians, 3)).astype(np.float32)
    scales = rng.uniform(0.05, 0.2, (n_gaussians, 3)).astype(np.float32)
    rot = rng.randn(n_gaussians, 4).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=-1, keepdims=True)
    sh = rng.uniform(0, 1, (n_gaussians, 3)).astype(np.float32)

    streamer = GaussianSplat4DStreamer(depth=6)
    stats = streamer.compress_frame(pos, scales, rot, sh)
    assert stats["num_gaussians"] == n_gaussians
    assert stats["num_clusters"] > 0
    print(f"    [PASS] Compressed {stats['num_gaussians']} Gaussians into {stats['num_clusters']} clusters ({stats['compression_ratio']:.2f}x)")


def test_neuromorphic_reconstructor():
    print("[+] Testing NeuromorphicStreamReconstructor...")
    n_events = 5000
    rng = np.random.RandomState(42)
    x = rng.randint(0, 320, n_events, dtype=np.int32)
    y = rng.randint(0, 240, n_events, dtype=np.int32)
    ts = np.sort(rng.uniform(0.0, 20000.0, n_events)).astype(np.float64)
    pol = rng.choice([-1, 1], n_events).astype(np.int8)

    recon = NeuromorphicStreamReconstructor(width=320, height=240)
    stats = recon.process_event_batch(x, y, ts, pol)
    assert stats["num_events"] == n_events
    print(f"    [PASS] Processed {stats['num_events']} events in {stats['latency_ms']:.2f} ms")


def test_low_bitrate_semantic_codec():
    print("[+] Testing LowBitrateSemanticCodec...")
    codec = LowBitrateSemanticCodec(num_landmarks=68)
    rng = np.random.RandomState(42)
    landmarks = rng.uniform(0.1, 0.9, (68, 2)).astype(np.float32)
    packed, enc_time = codec.encode_frame(landmarks)
    assert len(packed) > 0
    field, dec_time = codec.decode_and_reconstruct_field(packed, grid_size=32)
    assert field.shape == (32, 32)
    print(f"    [PASS] Semantic video codec encoded {len(packed)} bytes, decoded 32x32 field in {dec_time:.2f} ms")


def test_frame_deduplicator():
    print("[+] Testing VideoFrameDeduplicator...")
    dedup = VideoFrameDeduplicator(capacity=1000)
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    is_unique1 = dedup.process_frame(frame)
    is_unique2 = dedup.process_frame(frame)
    assert is_unique1
    assert not is_unique2
    print("    [PASS] Video frame deduplication verified.")


def test_video_motion_heatmap():
    print("[+] Testing VideoMotionHeatmap...")
    heatmap = VideoMotionHeatmap(grid_w=64, grid_h=36)
    mvs = np.zeros((36, 64, 2), dtype=np.float32)
    mvs[10:20, 10:20, 0] = 5.0
    heatmap.accumulate_motion_vectors(mvs)
    assert heatmap.heatmap.shape == (36, 64)
    assert np.max(heatmap.heatmap) > 0
    print("    [PASS] Motion heatmap accumulator verified.")


def test_lockfree_motion_estimation():
    print("[+] Testing LockFreeMotionEstimator...")
    estimator = LockFreeMotionEstimator(block_size=16, width=128, height=128)
    f1 = np.full((128, 128), 100, dtype=np.uint8)
    f2 = np.full((128, 128), 100, dtype=np.uint8)
    f2[32:48, 32:48] = 200
    estimator.register_reference_frame(f1)
    mv, sad, stats = estimator.estimate_motion(f2)
    assert mv.shape == (8, 8, 2)
    print(f"    [PASS] Lock-free motion estimation evaluated {stats['total_blocks']} blocks in {stats['elapsed_ms']:.2f} ms")


def test_greedy_macroblock_merger():
    print("[+] Testing GreedyMacroblockMerger...")
    merger = GreedyMacroblockMerger(block_size=16, variance_threshold=10.0)
    frame = np.full((128, 128), 128, dtype=np.uint8)
    merged, stats = merger.merge_frame(frame)
    assert stats["compression_ratio"] >= 1.0
    print(f"    [PASS] Greedy macroblock merger achieved {stats['compression_ratio']:.2f}x block reduction.")


def test_one_euro_stabilizer():
    print("[+] Testing OneEuroVideoStabilizer & ABR Controller...")
    stab = OneEuroVideoStabilizer(min_cutoff=1.0, beta=0.007)
    pos1 = np.array([10.0, 20.0], dtype=np.float32)
    pos2 = np.array([12.0, 22.0], dtype=np.float32)
    s1 = stab.filter(pos1, rate=60.0)
    s2 = stab.filter(pos2, rate=60.0)
    assert s2.shape == (2,)

    abr = AdaptiveBitrateController()
    selected_br, filtered_bw = abr.update_and_select(measured_bandwidth_kbps=3500.0, fps=30.0)
    assert selected_br > 0
    print(f"    [PASS] 1-Euro stabilizer smoothed position and ABR selected {selected_br} kbps.")


def test_biosignal_media_stream():
    print("[+] Testing BiosignalMediaStreamMuxer...")
    channel_names = [f"EEG_{i}" for i in range(8)]
    muxer = BiosignalMediaStreamMuxer(channel_names=channel_names, sampling_rate_hz=500.0, video_fps=60.0)
    rng = np.random.RandomState(42)
    signal = rng.randn(8, 2000).astype(np.float32) * 50.0 # 8 channels x 2000 samples
    packets, report = muxer.multiplex_stream_session(signal, start_time_seconds=0.0)
    assert len(packets) > 0
    assert report.total_biosignal_samples > 0
    assert report.total_video_frames > 0
    print(f"    [PASS] Biosignal media multiplexer multiplexed {report.total_video_frames} frames ({report.muxing_throughput_samples_sec:,.0f} samples/s).")


def test_scenecut_and_perceptual_rc():
    print("[+] Testing SceneCutGOPAnalyzer & PerceptualRateController...")
    analyzer = SceneCutGOPAnalyzer(min_gop_size=1)
    rng = np.random.RandomState(42)
    f1 = rng.randint(0, 100, size=(120, 160), dtype=np.uint8)
    f2 = rng.randint(150, 255, size=(120, 160), dtype=np.uint8) # hard cut
    m1 = analyzer.analyze_frame(f1)
    m2 = analyzer.analyze_frame(f2)
    assert m1.is_keyframe # First frame is IDR
    assert m2.is_keyframe and m2.transition_type.value == "HARD_CUT"
    
    prc = PerceptualRateController()
    rc_res = prc.analyze_frame_perceptual_complexity(f2, base_qp=28)
    assert rc_res.base_qp == 28
    print(f"    [PASS] Scene cut detected (Score: {m2.scene_cut_score:.2f}) and perceptual rate controller verified.")


def test_parametric_noise_field_and_adaptive_hls():
    print("[+] Testing ParametricNoiseFieldCodec & AdaptiveStreamingSegmenter...")
    analyzer = ParametricNoiseFieldAnalyzer()
    synthesizer = ParametricNoiseFieldSynthesizer()
    frame = np.full((120, 160), 128, dtype=np.uint8)
    res = analyzer.decompose_noise_field(frame)
    synth_frame = synthesizer.reconstruct_field(res.base_frame, res.descriptor)
    assert synth_frame.shape == (120, 160)
    assert res.descriptor.ar_lag >= 0

    # Also verify AV1/VVC compatibility alias interface
    fg_analyzer = FilmGrainAnalyzer()
    fg_synthesizer = FilmGrainSynthesizer()
    fg_params = fg_analyzer.estimate_film_grain(frame)
    fg_synth = fg_synthesizer.apply_film_grain(frame, fg_params)
    assert fg_synth.shape == (120, 160)

    segmenter = AdaptiveStreamingSegmenter()
    manifests = segmenter.generate_manifests(
        keyframe_pts=[0.0, 2.0, 4.0, 6.0],
        total_duration_sec=7.0,
        stream_name="test_stream"
    )
    assert "#EXTM3U" in manifests.hls_master_m3u8
    print("    [PASS] Parametric noise field codec & ABR HLS segmenter verified.")


def test_spatial_dct_codec():
    print("[+] Testing SpatialDCTCodec...")
    codec = SpatialDCTCodec(quality_factor=85)
    frame = np.full((128, 128), 128, dtype=np.uint8)
    frame[32:96, 32:96] = 200
    pkt = codec.encode_frame(frame)
    assert pkt.byte_length > 0
    assert pkt.psnr_db > 30.0
    recon = codec.decode_frame(pkt)
    assert recon.shape == (128, 128)
    print(f"    [PASS] Spatial DCT codec round-trip verified (PSNR: {pkt.psnr_db:.2f} dB, SSIM: {pkt.ssim:.4f}).")


def test_ffmpeg_interop_bridge():
    print("[+] Testing FFmpegInteropBridge...")
    bridge = FFmpegInteropBridge()
    assert len(bridge.available_encoders) > 0
    plan = bridge.generate_encoding_plan(
        input_spec="pipe:0",
        output_path="test_out.mp4",
        codec="av1",
        width=1280,
        height=720,
        fps=30.0,
        keyframe_timestamps=[0.0, 2.0, 4.0]
    )
    assert len(plan.generated_cli_command) > 0
    assert plan.target_codec == "av1"
    print(f"    [PASS] FFmpeg interop bridge synthesized plan: {plan.encoder_flag} ({len(plan.generated_cli_command)} args).")


if __name__ == "__main__":
    print("==================================================================")
    print(" VIDEO STREAMING & CODECS: UNIT & INTEGRATION TEST HARNESS")
    print("==================================================================")
    t0 = time.perf_counter()
    test_gaussian_splat_streamer()
    test_neuromorphic_reconstructor()
    test_low_bitrate_semantic_codec()
    test_frame_deduplicator()
    test_video_motion_heatmap()
    test_lockfree_motion_estimation()
    test_greedy_macroblock_merger()
    test_one_euro_stabilizer()
    test_biosignal_media_stream()
    test_scenecut_and_perceptual_rc()
    test_parametric_noise_field_and_adaptive_hls()
    test_spatial_dct_codec()
    test_ffmpeg_interop_bridge()
    t_total = (time.perf_counter() - t0) * 1000.0
    print("==================================================================")
    print(f"ALL VIDEO STREAMING TESTS PASSED in {t_total:.2f} ms")
    print("==================================================================")
