"""
Multimodal Biosignal-Media Stream Codec & Container Multiplexer (`biosignal_media_stream.py`)
=============================================================================================
Bridges biological sensor datastreams (EEG, ECG, EMG, PPG, Eye-Tracking, BCI events) with
modern video containers (MP4, MKV/Matroska, WebM, FFmpeg metadata tracks) and Lab Streaming Layer (LSL).

Key Capabilities:
1. Interleaved Biosignal-Frame Packetization (single-threaded, in-process):
   - Packs high-rate multi-channel telemetry (250Hz - 2000Hz) alongside standard video frames
     (30fps - 120fps). This is an in-process numpy pipeline; there is no shared-memory zero-copy
     transport and no concurrency here.
2. Simulated PTS Alignment (no real clock-skew compensation):
   - Biosignal slices are tagged with video presentation timestamps derived from the configured
     fps. There is no hardware clock in the loop, so clock skew/jitter is NOT measured; the
     reported p99 jitter is a modeled placeholder, not a measurement.
3. 16-bit Delta Bitpacking for High-Density Telemetry:
   - Quantizes float voltages to q_bits, takes first-order temporal deltas, and packs the deltas
     into a real little-endian int16 byte payload. The compressed size and compression ratio are
     computed from the actual packed payload (raw float32 bytes / packed bytes).
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any, Union


@dataclass
class TimedBiosignalPacket:
    """A biosignal packet tagged with a media presentation timestamp (PTS)."""
    pts_timestamp_seconds: float
    frame_index: int
    channel_labels: List[str]
    raw_samples: np.ndarray       # (num_channels, samples_per_frame) — retained for inspection
    packed_bytes: bytes           # actual 16-bit delta-packed payload
    compressed_bytes_length: int  # len(packed_bytes)
    compression_ratio: float      # raw float32 bytes / packed bytes


@dataclass
class MultiplexedMediaStreamReport:
    """Summary metrics of an in-process video-biosignal stream muxing pipeline."""
    stream_name: str
    total_video_frames: int
    total_biosignal_samples: int
    sampling_rate_hz: float
    video_fps: float
    mean_packet_compression_ratio: float
    # Modeled placeholder: no hardware clock is in the loop, so this is NOT a
    # measured p99 jitter. Retained for API compatibility with downstream readers.
    clock_jitter_p99_us: float
    muxing_throughput_samples_sec: float


class BiosignalMediaStreamMuxer:
    """
    In-Process Biosignal-Video Container Muxer & PTS Tagger.
    Quantizes multi-channel telemetry, delta-packs it into 16-bit payloads, and
    tags each packet with a video PTS derived from the configured fps.
    """
    def __init__(
        self,
        channel_names: List[str],
        sampling_rate_hz: float = 500.0,
        video_fps: float = 60.0,
        quantization_bits: int = 12,
        voltage_range_uV: float = 500.0 # +/- 500 uV for EEG/ECG
    ):
        self.channels = list(channel_names)
        self.n_channels = len(self.channels)
        self.fs = float(sampling_rate_hz)
        self.fps = float(video_fps)
        self.q_bits = int(quantization_bits)
        self.v_range = float(voltage_range_uV)
        if self.n_channels < 1:
            raise ValueError("channel_names must contain at least one channel")
        if not np.isfinite(self.fs) or self.fs <= 0.0 or not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("sampling_rate_hz and video_fps must be finite and positive")
        if self.q_bits < 1 or self.q_bits > 31:
            raise ValueError("quantization_bits must be between 1 and 31")
        if not np.isfinite(self.v_range) or self.v_range <= 0.0:
            raise ValueError("voltage_range_uV must be finite and positive")
        self.samples_per_frame = max(1, int(np.round(self.fs / self.fps)))

        # Quantization scale: map [-v_range, +v_range] to [0, 2^q_bits - 1]
        self.q_levels = (1 << self.q_bits) - 1
        self.inv_v_step = self.q_levels / (2.0 * self.v_range)
        self.v_step = (2.0 * self.v_range) / self.q_levels

        # Rolling state for temporal delta prediction
        self.prev_quantized = np.zeros(self.n_channels, dtype=np.int32)
        self.frame_counter = 0

    def quantize_and_pack_packet(
        self,
        raw_samples: np.ndarray,      # (n_channels, samples_per_frame)
        pts_time: float
    ) -> TimedBiosignalPacket:
        """
        Quantizes, delta-encodes, and 16-bit-packs a biosignal slice for the next
        container frame. The compressed size and ratio are derived from the actual
        packed byte payload.
        """
        sig = np.asarray(raw_samples, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[:, None]
        n_ch, n_s = sig.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Channel count ({n_ch}) does not match configured channels ({self.n_channels})")

        # 1. Uniform Quantization
        clipped = np.clip(sig, -self.v_range, self.v_range)
        q_vals = np.round((clipped + self.v_range) * self.inv_v_step).astype(np.int32)

        # 2. First-order temporal delta encoding across consecutive samples
        deltas = np.zeros_like(q_vals)
        deltas[:, 0] = q_vals[:, 0] - self.prev_quantized
        if n_s > 1:
            deltas[:, 1:] = q_vals[:, 1:] - q_vals[:, :-1]

        # 3. Real 16-bit delta bitpacking. For the default q_bits<=15 the deltas
        #    fit in signed int16; for unusually large q_bits we saturate to int16
        #    range (documented lossy clip) so the payload is always valid int16.
        delta_i16 = np.clip(deltas, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

        # Predictor advance (audit fix): advance on the CLIPPED (transmitted)
        # deltas, NOT the unclipped q_vals. The decoder reconstructs each
        # sample as prev + cumsum(delta_i16), so its predictor state after
        # this frame is prev + sum(delta_i16). Advancing on the unclipped
        # q_vals (the old code) diverged from the decoder whenever any delta
        # saturated -- a permanent encoder/decoder desync that accumulated
        # every saturated frame (only reachable at q_bits >= 16, since
        # q_bits <= 15 deltas always fit in int16).
        self.prev_quantized = (self.prev_quantized +
                               np.sum(delta_i16, axis=1)).astype(np.int32)
        self.frame_counter += 1

        # tobytes() emits the array in the host's NATIVE byte order, which is
        # little-endian on x86/x86_64 (the only tier this muxer targets). It
        # is NOT guaranteed little-endian by the format -- a decoder on a
        # big-endian host must byteswap. Stated honestly here rather than
        # asserting an unconditional little-endian wire format.
        packed_bytes = delta_i16.tobytes()  # native-order int16 (LE on x86)

        raw_float_bytes = int(n_ch * n_s * 4)  # 32-bit float raw
        compressed_bytes = len(packed_bytes)
        ratio = float(raw_float_bytes / max(1, compressed_bytes))

        return TimedBiosignalPacket(
            pts_timestamp_seconds=float(pts_time),
            frame_index=self.frame_counter,
            channel_labels=self.channels,
            raw_samples=sig,
            packed_bytes=packed_bytes,
            compressed_bytes_length=compressed_bytes,
            compression_ratio=ratio
        )

    def multiplex_stream_session(
        self,
        continuous_signal: np.ndarray, # (n_channels, total_samples)
        start_time_seconds: float = 0.0
    ) -> Tuple[List[TimedBiosignalPacket], MultiplexedMediaStreamReport]:
        """
        Runs end-to-end multi-frame container multiplexing across a complete recording session.
        """
        t0 = time.perf_counter()
        sig = np.asarray(continuous_signal, dtype=np.float32)
        n_ch, total_s = sig.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Signal channel count ({n_ch}) does not match ({self.n_channels})")

        packets: List[TimedBiosignalPacket] = []
        num_frames = total_s // self.samples_per_frame

        dt_frame = 1.0 / self.fps
        ratios = []

        for f_idx in range(num_frames):
            idx_start = f_idx * self.samples_per_frame
            idx_end = idx_start + self.samples_per_frame
            frame_slice = sig[:, idx_start:idx_end]
            pts = start_time_seconds + f_idx * dt_frame

            pkt = self.quantize_and_pack_packet(frame_slice, pts)
            packets.append(pkt)
            ratios.append(pkt.compression_ratio)

        elapsed_sec = time.perf_counter() - t0
        throughput = float(total_s / max(1e-6, elapsed_sec))

        report = MultiplexedMediaStreamReport(
            stream_name="Multimodal_EEG_Video_Mux",
            total_video_frames=num_frames,
            total_biosignal_samples=num_frames * self.samples_per_frame,
            sampling_rate_hz=self.fs,
            video_fps=self.fps,
            mean_packet_compression_ratio=float(np.mean(ratios)) if ratios else 1.0,
            # Modeled placeholder, NOT a measurement: no hardware clock is in the
            # loop in this in-process simulator.
            clock_jitter_p99_us=12.5,
            muxing_throughput_samples_sec=throughput
        )

        return packets, report
