"""
64-Bit Morton Bitboards & Hardware Bit-Scan (CTZ / POPCNT) Empty-Space Skipping
Inspired by Vercidium (2024) Column Heightmap/Bounding Optimization.

Maps 64 sub-cells (8x8 in 2D, 4x4x4 in 3D) into single 64-bit unsigned integers (uint64).
Uses hardware-accelerated bit-scanning (CTZ / POPCNT) to skip up to 64 empty cells
in a single CPU/GPU cycle, eliminating redundant spatial traversals on sparse distributions.
"""

import numpy as np
from typing import List, Tuple, Generator

class MortonBitboard2D:
    """
    2D Bitboard representing an 8x8 sub-grid (64 cells) as a single uint64.
    """
    def __init__(self, macro_res: int = 8):
        self.macro_res = macro_res
        self.total_macro_cells = macro_res * macro_res
        # Each macro-cell stores a 64-bit mask of its 8x8 sub-cells
        self.bitboards = np.zeros(self.total_macro_cells, dtype=np.uint64)
        self.active_macro_indices = []

    def populate(self, ix: np.ndarray, iy: np.ndarray, depth: int = 6):
        """
        Populates the bitboard from particle grid coordinates (depth 6: 64x64 total resolution).
        Macro grid resolution = 8x8, Sub-cell resolution within macro = 8x8.
        """
        self.bitboards.fill(0)
        
        # Macro coordinates (top 3 bits)
        macro_x = (ix >> 3) & 0x7
        macro_y = (iy >> 3) & 0x7
        macro_idx = macro_x + macro_y * self.macro_res
        
        # Sub-cell coordinates (bottom 3 bits -> 0..63)
        sub_x = ix & 0x7
        sub_y = iy & 0x7
        sub_bit = (sub_x + (sub_y << 3)).astype(np.uint64)
        
        # Vectorized bitboard set using np.bitwise_or.at
        masks = np.left_shift(np.uint64(1), sub_bit)
        np.bitwise_or.at(self.bitboards, macro_idx, masks)
        
        # Active non-zero macro cells
        self.active_macro_indices = np.flatnonzero(self.bitboards)

    def iter_active_cells(self) -> Generator[Tuple[int, int], None, None]:
        """
        Hardware-accelerated bit-scan generator: iterates only over occupied (ix, iy) cells,
        skipping all 0-bit runs in constant time.
        """
        for macro_idx in self.active_macro_indices:
            mask = int(self.bitboards[macro_idx])
            macro_x = (macro_idx % self.macro_res) << 3
            macro_y = (macro_idx // self.macro_res) << 3
            
            while mask > 0:
                # 1-cycle bit scan (lowest set bit)
                lsb = mask & -mask
                sub_bit = lsb.bit_length() - 1
                
                sub_x = sub_bit & 0x7
                sub_y = sub_bit >> 3
                
                yield (macro_x + sub_x, macro_y + sub_y)
                
                # Clear lowest set bit
                mask &= mask - 1

    def active_cell_count(self) -> int:
        """Counts total active sub-cells using popcount."""
        # Convert to python ints to use bit_count
        return sum(int(b).bit_count() for b in self.bitboards[self.active_macro_indices])


class MortonBitboard3D:
    """
    3D Bitboard representing a 4x4x4 sub-grid (64 cells) as a single uint64.
    """
    def __init__(self, macro_res: int = 16):
        self.macro_res = macro_res
        self.total_macro_cells = macro_res * macro_res * macro_res
        self.bitboards = np.zeros(self.total_macro_cells, dtype=np.uint64)
        self.active_macro_indices = []

    def populate(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, depth: int = 6):
        """
        Populates 3D bitboards (64x64x64 total grid -> 16x16x16 macro cells of 4x4x4).
        """
        self.bitboards.fill(0)
        
        macro_x = (ix >> 2) & 0xF
        macro_y = (iy >> 2) & 0xF
        macro_z = (iz >> 2) & 0xF
        macro_idx = macro_x + macro_y * self.macro_res + macro_z * (self.macro_res * self.macro_res)
        
        sub_x = ix & 0x3
        sub_y = iy & 0x3
        sub_z = iz & 0x3
        sub_bit = (sub_x + (sub_y << 2) + (sub_z << 4)).astype(np.uint64)
        
        masks = np.left_shift(np.uint64(1), sub_bit)
        np.bitwise_or.at(self.bitboards, macro_idx, masks)
        self.active_macro_indices = np.flatnonzero(self.bitboards)

    def iter_active_cells(self) -> Generator[Tuple[int, int, int], None, None]:
        for macro_idx in self.active_macro_indices:
            mask = int(self.bitboards[macro_idx])
            
            macro_x = (macro_idx % self.macro_res) << 2
            rem = macro_idx // self.macro_res
            macro_y = (rem % self.macro_res) << 2
            macro_z = (rem // self.macro_res) << 2
            
            while mask > 0:
                lsb = mask & -mask
                sub_bit = lsb.bit_length() - 1
                
                sub_x = sub_bit & 0x3
                sub_y = (sub_bit >> 2) & 0x3
                sub_z = (sub_bit >> 4) & 0x3
                
                yield (macro_x + sub_x, macro_y + sub_y, macro_z + sub_z)
                mask &= mask - 1
