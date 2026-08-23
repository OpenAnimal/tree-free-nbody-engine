"""
Comprehensive Unit & Integration Test Suite for Video Streaming, Codecs & Multimodal Media (`video_streaming_codecs`).
"""

import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def test_motion_estimation_uniform_region_fallback():
    """On a uniform frame every block shares a fingerprint. The hash-proposed MV
    must NOT be trusted blindly: SAD validation must force the zero-MV fallback
    (the reference and current frames are identical, so the correct MV is zero
    everywhere and the SAD must be ~0 for every block)."""
    print("[+] Testing motion estimation uniform-region SAD fallback...")
    estimator = LockFreeMotionEstimator(block_size=16, width=128, height=128)
    uniform = np.full((128, 128), 120, dtype=np.uint8)
    estimator.register_reference_frame(uniform)
    mv, sad, stats = estimator.estimate_motion(uniform)
    # Every block is identical to the reference -> zero MV and ~0 SAD everywhere.
    assert np.array_equal(mv, np.zeros_like(mv)), "uniform region must fall back to zero MV"
    assert np.max(sad) == 0.0, f"uniform region SAD must be 0 (got max {np.max(sad)})"
    print(f"    [PASS] Uniform region fell back to zero-MV (max SAD = {np.max(sad):.1f}, hash_hit_rate = {stats['hash_hit_rate']:.1f}%).")


def test_greedy_macroblock_merger():
    print("[+] Testing GreedyMacroblockMerger...")
    merger = GreedyMacroblockMerger(block_size=16, variance_threshold=10.0)
    frame = np.full((128, 128), 128, dtype=np.uint8)
    merged, stats = merger.merge_frame(frame)
    assert stats["compression_ratio"] >= 1.0
    print(f"    [PASS] Greedy macroblock merger achieved {stats['compression_ratio']:.2f}x block reduction.")


def test_greedy_macroblock_merger_edge_blocks():
    """A frame whose width/height is NOT a multiple of block_size must not silently
    drop the trailing edge rows/columns: partial edge blocks must be emitted and
    their union must cover the full frame extent."""
    print("[+] Testing GreedyMacroblockMerger edge-block coverage...")
    merger = GreedyMacroblockMerger(block_size=16, variance_threshold=10.0)
    # 1920x1080: 1080 = 67*16 + 8 -> bottom edge of 8 rows must be emitted.
    frame = np.full((1080, 1920), 128, dtype=np.uint8)
    merged, stats = merger.merge_frame(frame)
    assert stats["edge_blocks"] > 0, "non-multiple-of-block-size frame must emit edge blocks"
    # Verify the union of all block rectangles covers the full frame.
    covered = np.zeros((1080, 1920), dtype=bool)
    for b in merged:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        covered[y:y+h, x:x+w] = True
    assert covered.all(), "merged blocks must cover the entire frame (no dropped edge pixels)"
    # The bottom 8 edge rows must be covered by edge_partial blocks.
    edge_rows = [b for b in merged if b["type"] == "edge_partial"]
    assert any(b["y"] + b["h"] == 1080 for b in edge_rows), "bottom edge rows must reach frame bottom"
    print(f"    [PASS] Edge coverage ok: {stats['edge_blocks']} edge blocks, full frame covered.")


def test_greedy_macroblock_merger_coverage_sweep():
    """Parametrized coverage sweep: for a range of (w, h) frame sizes at
    block_size=16, every pixel must be covered EXACTLY once by the emitted
    block rectangles -- no gap, no overlap.

    This is the regression guard for the both-edge-residuals bug
    (greedy_macroblock_merger.py:125-156): frames with BOTH a trailing
    bottom edge AND a trailing right edge dropped the right-edge interior
    strip (the `elif edge_w > 0:` was unreachable when `edge_h > 0`).
    100x60 @ bs=16 left 192 px uncovered; 60x100 left 1152. The fix emits
    the bottom strip, right strip, and bottom-right corner independently.

    Fast step=13 per the audit acceptance; the full step=7 sweep is the
    slow variant.
    """
    print("[+] Testing GreedyMacroblockMerger exact-coverage sweep...")
    bs = 16
    merger = GreedyMacroblockMerger(block_size=bs, variance_threshold=10.0)
    sizes = list(range(30, 201, 13))
    checked = 0
    for h in sizes:
        for w in sizes:
            # A uniform frame -> every interior cell is a flat merge
            # candidate, but coverage is independent of merge decisions:
            # the edge regions are always emitted standalone. Use a uniform
            # frame so the interior collapses to one merged rectangle and
            # any coverage hole is unambiguously an edge-emission bug.
            frame = np.full((h, w), 128, dtype=np.uint8)
            merged, stats = merger.merge_frame(frame)
            counts = np.zeros((h, w), dtype=np.int32)
            for b in merged:
                x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
                counts[y:y+bh, x:x+bw] += 1
            if not (counts.min() == 1 and counts.max() == 1):
                uncovered = int(np.sum(counts == 0))
                overlapped = int(np.sum(counts > 1))
                raise AssertionError(
                    f"coverage failure at w={w} h={h} bs={bs}: "
                    f"min={counts.min()} max={counts.max()} "
                    f"uncovered={uncovered} overlapped={overlapped}")
            checked += 1
    print(f"    [PASS] Exact coverage (every pixel covered exactly once) for "
          f"{checked} (w,h) combinations in [30..200] step 13 at bs={bs}.")


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


def test_biosignal_predictor_no_desync_on_saturation():
    """Regression (audit finding #8): the encoder's temporal-delta predictor
    must advance on the CLIPPED (transmitted) deltas, NOT the unclipped
    quantized values, so it stays in sync with the decoder.

    ROOT CAUSE: the encoder advanced `self.prev_quantized = q_vals[:, -1]`
    using the unclipped quantized values, but the decoder reconstructs each
    sample as `prev + cumsum(delta_i16)` -- i.e. it advances on the
    SATURATED int16 deltas. Whenever a delta saturated (only reachable at
    q_bits >= 16, since q_bits <= 15 deltas always fit in int16), the
    encoder and decoder predictors diverged permanently, and the divergence
    accumulated every saturated frame.

    This test constructs a signal that forces delta saturation at q_bits=16
    by jumping from -v_range to +v_range between consecutive samples (a
    delta of ~2*v_range, which quantizes to ~q_levels = 65535 -- right at
    the int16 saturation boundary). It then simulates the decoder's
    cumsum-delta reconstruction and asserts the decoder's predictor state
    matches the encoder's `prev_quantized` after every frame, for many
    frames. The OLD code fails this on the first saturated frame.
    """
    print("[+] Testing BiosignalMediaStreamMuxer predictor/decoder sync...")
    n_ch = 4
    muxer = BiosignalMediaStreamMuxer(
        channel_names=[f"EEG_{i}" for i in range(n_ch)],
        sampling_rate_hz=500.0, video_fps=100.0,
        quantization_bits=16, voltage_range_uV=300.0)
    # samples_per_frame = round(500/100) = 5
    n_frames = 30
    n_s = muxer.samples_per_frame
    # Build a signal that saturates the int16 delta in every frame:
    # alternate full-scale swings between consecutive samples.
    t = np.arange(n_s * n_frames)
    sig = (300.0 * np.sign(np.sin(2 * np.pi * t / (2 * n_s)))).astype(np.float32)
    sig = np.broadcast_to(sig, (n_ch, n_s * n_frames)).copy()
    packets, _ = muxer.multiplex_stream_session(sig, start_time_seconds=0.0)
    assert len(packets) == n_frames, f"expected {n_frames} packets, got {len(packets)}"

    # Simulate the decoder: reconstruct each sample as prev + cumsum of the
    # transmitted (saturated int16) deltas. The decoder's predictor state
    # after frame f is prev + sum(delta_i16[:, f]).
    decoder_prev = np.zeros(n_ch, dtype=np.int32)
    saturation_count = 0
    for f, pkt in enumerate(packets):
        delta_i16 = np.frombuffer(pkt.packed_bytes, dtype=np.int16
                                  ).reshape(n_ch, n_s).astype(np.int32)
        # Count saturations (the audit's trigger condition).
        sat = int(np.sum((np.abs(delta_i16) >= 32767)))
        saturation_count += sat
        # Decoder reconstructs the per-sample quantized values via cumsum.
        recon = np.cumsum(delta_i16, axis=1) + decoder_prev[:, None]
        decoder_prev = recon[:, -1]
    # The test's whole point is that saturation MUST happen (otherwise the
    # bug path is not exercised). At q_bits=16, full-scale swings produce
    # deltas of ~65535 which saturate to 32767.
    assert saturation_count > 0, (
        "test setup failed: no delta saturation occurred, so the desync "
        "path was not exercised. Check q_bits=16 and the signal swing.")

    # The encoder's final prev_quantized must equal the decoder's final
    # predictor state. The OLD code (advance on unclipped q_vals) diverged
    # here by ~32768 per saturated frame.
    encoder_prev = muxer.prev_quantized
    max_diff = int(np.max(np.abs(encoder_prev - decoder_prev)))
    assert max_diff == 0, (
        f"predictor desync: encoder prev_quantized = {encoder_prev} "
        f"but decoder reconstructed prev = {decoder_prev} "
        f"(max abs diff = {max_diff}; {saturation_count} saturated deltas "
        f"across {n_frames} frames). The encoder advanced on unclipped "
        f"q_vals while the decoder advances on saturated int16 deltas.")
    print(f"    [PASS] Encoder/decoder predictor sync holds across "
          f"{n_frames} frames with {saturation_count} saturated deltas "
          f"(max abs diff = {max_diff}).")


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


def test_spatial_dct_cross_quality_decode():
    """A packet encoded at one quality must decode identically regardless of the
    quality the decoding codec was constructed with (the quant matrix is rebuilt
    from the bitstream header byte)."""
    print("[+] Testing SpatialDCTCodec cross-quality decode...")
    rng = np.random.RandomState(7)
    frame = rng.randint(0, 256, size=(64, 64)).astype(np.uint8)
    enc = SpatialDCTCodec(quality_factor=80)
    pkt = enc.encode_frame(frame)
    # Decode with a fresh codec constructed at a *different* quality.
    dec_other = SpatialDCTCodec(quality_factor=40)
    recon_other = dec_other.decode_frame(pkt)
    # Decode with the encoding codec itself (reference).
    recon_ref = enc.decode_frame(pkt)
    assert recon_other.shape == (64, 64)
    assert np.array_equal(recon_other, recon_ref), (
        "cross-quality decode mismatch: decoder must rebuild quant matrix from header q"
    )
    print(f"    [PASS] Cross-quality decode matches (enc q=80, dec q=40); max abs diff = {int(np.max(np.abs(recon_other.astype(int) - recon_ref.astype(int))))}.")


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


def test_scenecut_hard_cut_frame_accuracy():
    """A hard cut from a stable baseline must place the IDR at the CUT frame
    (frame-accurate), not one frame late, and must patch the cut frame's
    metadata to IDR in place. Regression for the deferred-flash resolution
    that used to shift keyframes by +1 and emit the first frame of the new
    scene as inter-predicted."""
    def coherent_scene(n, seed, h=36, w=64):
        r = np.random.default_rng(seed)
        base = r.random((h, w)) * 255
        return [np.clip(base + r.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8)
                for _ in range(n)]

    frames = coherent_scene(20, 101) + coherent_scene(20, 202) + coherent_scene(20, 303)
    analyzer = SceneCutGOPAnalyzer(min_gop_size=8)
    metas = [analyzer.analyze_frame(f) for f in frames]
    for cut in (20, 40):
        assert cut in analyzer.keyframe_indices, (
            f"hard cut at frame {cut} not in keyframes {analyzer.keyframe_indices}")
        m = metas[cut]
        assert m.is_keyframe and m.recommended_frame_type == "IDR", (
            f"cut frame {cut} metadata not IDR: {m.recommended_frame_type}")
    # A single-frame flash must NOT become an IDR.
    base = np.random.default_rng(11).random((36, 64)) * 255
    r = np.random.default_rng(12)
    frames2 = [np.clip(base + r.normal(0, 2, (36, 64)), 0, 255).astype(np.uint8)
               for _ in range(12)]
    flash = np.clip(base * 0.3 + 180, 0, 255).astype(np.uint8)
    frames2.insert(6, flash)
    analyzer2 = SceneCutGOPAnalyzer(min_gop_size=4)
    analyzer2.analyze_frame(frames2[0])
    metas2 = [analyzer2.analyze_frame(f) for f in frames2]
    assert 6 not in analyzer2.keyframe_indices, "single-frame flash must not be an IDR"
    print("    [PASS] Scenecut hard-cut IDR is frame-accurate; flash rejected.")


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
    test_motion_estimation_uniform_region_fallback()
    test_greedy_macroblock_merger()
    test_greedy_macroblock_merger_edge_blocks()
    test_greedy_macroblock_merger_coverage_sweep()
    test_one_euro_stabilizer()
    test_biosignal_media_stream()
    test_biosignal_predictor_no_desync_on_saturation()
    test_scenecut_and_perceptual_rc()
    test_scenecut_hard_cut_frame_accuracy()
    test_parametric_noise_field_and_adaptive_hls()
    test_spatial_dct_codec()
    test_spatial_dct_cross_quality_decode()
    test_ffmpeg_interop_bridge()
    t_total = (time.perf_counter() - t0) * 1000.0
    print("==================================================================")
    print(f"ALL VIDEO STREAMING TESTS PASSED in {t_total:.2f} ms")
    print("==================================================================")
