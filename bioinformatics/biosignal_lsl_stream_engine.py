"""
Module 17: High-Throughput Real-Time Biosignal & Multimodal LSL Streaming Engine.
Streaming ingestion, ring-buffer synchronization, and spatial filtering for massive biological signals
(64-512 channel EEG, ERP evoked potentials, fMRI BOLD timeseries, EMG, ECG, PPG).
Compatible with Lab Streaming Layer (LSL), BrainFlow, and modern media containers.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .core.elastic_spatial_hash import ElasticSpatialHash3D
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.core.elastic_spatial_hash import ElasticSpatialHash3D


@dataclass
class ChannelMetadata:
    """Metadata for a single biosignal electrode or recording sensor."""
    channel_index: int
    channel_label: str            # e.g., "Fz", "Cz", "Pz", "Oz", "C3", "C4"
    sensor_type: str              # "EEG", "EMG", "ECG", "EOG", "fMRI_BOLD", "PPG"
    coords_3d: np.ndarray         # (3,) Cartesian electrode coordinates on head scalp
    sampling_rate_hz: float
    unit: str = "uV"              # Microvolts for EEG, % BOLD signal change for fMRI


@dataclass
class BiosignalStreamChunk:
    """Synchronized multi-channel biosignal data chunk."""
    stream_id: str
    num_channels: int
    num_samples: int
    timestamps: np.ndarray        # (num_samples,) Monotonic microsecond timestamps
    signals: np.ndarray           # (num_channels, num_samples) Voltage or intensity matrix
    evoked_potential_detected: bool
    peak_evoked_latency_ms: Optional[float]
    dominant_band_power: Dict[str, float] # {"Delta": ..., "Theta": ..., "Alpha": ..., "Beta": ..., "Gamma": ...}


class BiosignalLSLStreamEngine:
    """
    High-Throughput Real-Time Biosignal & Multimodal LSL Streaming Engine.
    Processes multi-channel physiological and neural datastreams in O(1) latency per sample.
    """
    # Standard EEG frequency rhythm bands in Hertz (Hz)
    FREQUENCY_BANDS = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 80.0)
    }

    def __init__(
        self,
        channels: List[ChannelMetadata],
        ring_buffer_seconds: float = 10.0,
        spatial_filter_lambda: float = 0.05
    ):
        self.channels = list(channels)
        self.n_channels = len(self.channels)
        if self.n_channels == 0:
            raise ValueError("channels list must contain at least one channel")

        self.sampling_rate = float(self.channels[0].sampling_rate_hz)
        self.buffer_len = max(100, int(ring_buffer_seconds * self.sampling_rate))
        self.spatial_lambda = float(spatial_filter_lambda)

        # Contiguous Ring Buffers for zero-copy lock-free writes
        self.ring_signals = np.zeros((self.n_channels, self.buffer_len), dtype=np.float32)
        self.ring_timestamps = np.zeros(self.buffer_len, dtype=np.float64)
        self.write_head = 0
        self.total_samples_ingested = 0

        # Precompute Surface Laplacian (CSD - Current Source Density) spatial spline weights
        self.electrode_coords = np.array([ch.coords_3d for ch in self.channels], dtype=np.float64)
        self._build_spatial_laplacian_filter()

    def _build_spatial_laplacian_filter(self):
        """
        Builds matrix-free Surface Laplacian (Hjorth / Perrin CSD spatial filter)
        to eliminate volume-conduction blurring across EEG electrodes:
            V_csd = [I - W_spatial] * V
        """
        diff = self.electrode_coords[:, None, :] - self.electrode_coords[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        sigma = 35.0  # Millimeters (scalp distance)
        
        W = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1, keepdims=True)
        self.W_spatial = W / np.maximum(row_sums, 1e-6)

    def ingest_sample_chunk(
        self,
        signals: np.ndarray,          # (n_channels, n_samples)
        timestamps: Optional[np.ndarray] = None
    ) -> int:
        """
        Streams a new batch of biosignal samples into the lock-free ring buffer in O(1) time.
        """
        sig = np.asarray(signals, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[:, None]
        n_ch, n_s = sig.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Signal channel count ({n_ch}) does not match configured channels ({self.n_channels})")

        if timestamps is None:
            t0 = time.time()
            dt = 1.0 / self.sampling_rate
            ts = t0 + np.arange(n_s) * dt
        else:
            ts = np.asarray(timestamps, dtype=np.float64)

        for i in range(n_s):
            pos = (self.write_head + i) % self.buffer_len
            self.ring_signals[:, pos] = sig[:, i]
            self.ring_timestamps[pos] = ts[i]

        self.write_head = (self.write_head + n_s) % self.buffer_len
        self.total_samples_ingested += n_s
        return n_s

    def compute_spectral_band_powers(
        self,
        window_seconds: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculates normalized spectral rhythm band power (Delta, Theta, Alpha, Beta, Gamma)
        via FFT across the recent temporal window.
        """
        n_window = min(self.buffer_len, int(window_seconds * self.sampling_rate))
        if self.total_samples_ingested < n_window:
            n_window = max(10, self.total_samples_ingested)

        # Unroll latest window from circular buffer
        idx = (np.arange(self.write_head - n_window, self.write_head)) % self.buffer_len
        window_data = self.ring_signals[:, idx] # (n_channels, n_window)

        # Subtract mean and apply Hanning taper
        hanning = np.hanning(n_window)[None, :]
        window_tapered = (window_data - np.mean(window_data, axis=1, keepdims=True)) * hanning

        # FFT power spectrum
        fft_vals = np.fft.rfft(window_tapered, axis=1)
        freqs = np.fft.rfftfreq(n_window, d=1.0 / self.sampling_rate)
        psd = np.mean(np.abs(fft_vals) ** 2, axis=0) # Average over channels

        band_powers = {}
        total_power = float(np.sum(psd)) + 1e-12

        for band_name, (low_f, high_f) in self.FREQUENCY_BANDS.items():
            mask = (freqs >= low_f) & (freqs < high_f)
            p_band = float(np.sum(psd[mask]))
            band_powers[band_name] = float(p_band / total_power)

        return band_powers

    def detect_evoked_potential_erp(
        self,
        stimulus_timestamps: List[float],
        epoch_window_ms: Tuple[float, float] = (-100.0, 500.0) # Pre-stimulus baseline to P300 window
    ) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Extracts Event-Related Potentials (ERP) such as P300, N400, or Visual Evoked Potentials (VEP)
        by epoching and stimulus-locked signal averaging.
        """
        t_pre = epoch_window_ms[0] / 1000.0
        t_post = epoch_window_ms[1] / 1000.0
        n_samples_epoch = int((t_post - t_pre) * self.sampling_rate)

        epochs: List[np.ndarray] = []

        # Find matching sample indices for each stimulus trigger
        for t_stim in stimulus_timestamps:
            # Locate closest index in ring timestamps
            diff = np.abs(self.ring_timestamps - t_stim)
            best_idx = int(np.argmin(diff))

            start_idx = best_idx + int(t_pre * self.sampling_rate)
            end_idx = start_idx + n_samples_epoch

            if 0 <= start_idx and end_idx < self.buffer_len:
                ep = self.ring_signals[:, start_idx:end_idx]
                # Baseline subtraction (pre-stimulus window)
                n_base = int(-t_pre * self.sampling_rate)
                if n_base > 0:
                    base_mean = np.mean(ep[:, :n_base], axis=1, keepdims=True)
                    ep = ep - base_mean
                epochs.append(ep)

        if not epochs:
            return {
                "erp_detected": False,
                "num_epochs_averaged": 0,
                "p300_peak_latency_ms": 0.0,
                "p300_peak_amplitude_uV": 0.0,
                "grand_average_waveform": np.zeros((self.n_channels, n_samples_epoch), dtype=np.float32)
            }

        # Grand average across epochs
        grand_avg = np.mean(np.stack(epochs, axis=0), axis=0) # (n_channels, n_samples)
        
        # P300 search window: 250ms to 450ms post-stimulus
        p300_start = int((-t_pre + 0.25) * self.sampling_rate)
        p300_end = int((-t_pre + 0.45) * self.sampling_rate)

        cz_or_pz_idx = 0 # Default to channel 0
        for i, ch in enumerate(self.channels):
            if ch.channel_label in ["Pz", "Cz", "Oz"]:
                cz_or_pz_idx = i
                break

        search_slice = grand_avg[cz_or_pz_idx, p300_start:p300_end]
        if len(search_slice) > 0:
            peak_rel_idx = int(np.argmax(search_slice))
            peak_amp = float(search_slice[peak_rel_idx])
            peak_sample = p300_start + peak_rel_idx
            peak_lat_ms = float((peak_sample / self.sampling_rate + t_pre) * 1000.0)
            erp_found = peak_amp >= 2.0 # Minimum 2.0 uV evoked wave
        else:
            peak_amp = 0.0
            peak_lat_ms = 0.0
            erp_found = False

        return {
            "erp_detected": erp_found,
            "num_epochs_averaged": len(epochs),
            "p300_peak_latency_ms": peak_lat_ms,
            "p300_peak_amplitude_uV": peak_amp,
            "grand_average_waveform": grand_avg
        }
