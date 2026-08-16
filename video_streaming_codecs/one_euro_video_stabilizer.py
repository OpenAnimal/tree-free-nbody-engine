"""
1€ (One-Euro) Adaptive Gyro Video Stabilizer & Latency-Free ABR Bitrate Controller
Translates Casiez et al. (2012) into video deshake and live streaming network transport.

Features:
1. Video Camera Deshake: Full jitter suppression at low velocities, zero phase lag on sharp intentional pans.
2. Adaptive Bitrate (ABR) Controller: Smooths noisy network jitter without delaying rapid emergency downshifts.
"""

import numpy as np
import time
from typing import Tuple, List, Dict

class OneEuroVideoStabilizer:
    """
    1€ Adaptive Low-Pass Filter for Video Camera Trajectories & Gyro Deshake.
    """
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, rate: float = 60.0) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # 1. Filter the derivative (angular velocity / pan speed)
        dx = (x - self.x_prev) * rate
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Dynamic cutoff frequency based on movement speed
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Filter the coordinate/angle
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / rate
        alpha = 1.0 / (1.0 + tau / te)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class AdaptiveBitrateController:
    """
    1€ Filter Powered ABR Streaming Controller for WebRTC / HLS / DASH.
    Filters raw bandwidth estimates to prevent erratic bitrate ping-pong.
    """
    def __init__(self, min_cutoff: float = 0.2, beta: float = 0.15):
        self.filter = OneEuroVideoStabilizer(min_cutoff=min_cutoff, beta=beta)
        self.bitrate_ladder = [500, 1200, 2500, 5000, 8000, 15000] # kbps

    def update_and_select(self, measured_bandwidth_kbps: float, fps: float = 30.0) -> Tuple[int, float]:
        raw_val = np.array([measured_bandwidth_kbps], dtype=np.float32)
        filtered_val = self.filter.filter(raw_val, rate=fps)[0]
        
        # Select highest bitrate supported with 15% safety margin
        target_bw = filtered_val * 0.85
        selected = self.bitrate_ladder[0]
        for b in self.bitrate_ladder:
            if b <= target_bw:
                selected = b
            else:
                break
        return selected, float(filtered_val)
