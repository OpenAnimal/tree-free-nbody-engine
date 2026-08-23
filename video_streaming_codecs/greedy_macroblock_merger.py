"""
Vercidium-Style Run-Length Greedy Macroblock Aggregation for Video Encoders (AV1/VVC/HEVC)
Translates Vercidium's (2024) run-length meshing into video block partitioning.

Merges contiguous flat/static macroblocks along 1D Morton space-filling curves
into consolidated rectangular super-macroblocks in a single pass. Edge rows/columns
whose width/height is not a multiple of `block_size` are emitted as their own
(possibly partial) edge blocks rather than silently dropped.
"""

import numpy as np
import time
from typing import Tuple, List, Dict

class GreedyMacroblockMerger:
    """
    Greedy Macroblock Compressor for video frames & depth maps.
    Compresses 2D/3D video frames into run-length merged macroblock primitives.
    """
    def __init__(self, block_size: int = 16, variance_threshold: float = 12.0):
        self.block_size = block_size
        self.variance_threshold = variance_threshold

    def merge_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Processes frame into merged macroblock rectangles.

        The frame is tiled into a grid of `block_size` cells. When the frame
        width/height is not a multiple of `block_size`, the trailing edge cells
        are kept as partial-width/height blocks (their actual pixel extent) so no
        edge rows/columns are silently dropped. Only the full-size interior cells
        participate in run-length merging; partial edge cells are always emitted
        as standalone (complex) blocks.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        bs = self.block_size
        # Full-size interior block counts; the trailing edge cells (if any) are
        # handled separately so partial rows/columns are not dropped.
        bx_count = w // bs
        by_count = h // bs
        edge_w = w - bx_count * bs  # leftover columns (0 <= edge_w < bs)
        edge_h = h - by_count * bs  # leftover rows    (0 <= edge_h < bs)

        # 1. Compute block-level statistics (mean color + spatial variance)
        # Grid of block types: 0 = Complex (skip merge), 1 = Flat Static (merge candidate)
        block_types = np.zeros((by_count, bx_count), dtype=np.uint8)
        block_means = np.zeros((by_count, bx_count), dtype=np.float32)

        for by in range(by_count):
            for bx in range(bx_count):
                y = by * bs
                x = bx * bs
                blk = frame[y:y+bs, x:x+bs]

                var = np.var(blk)
                mean_val = np.mean(blk)
                block_means[by, bx] = mean_val

                if var < self.variance_threshold:
                    block_types[by, bx] = 1  # Merge candidate
                    
        # 2. Vercidium-Style 2D Greedy Run-Length Merging
        visited = np.zeros((by_count, bx_count), dtype=bool)
        merged_blocks = []
        
        for by in range(by_count):
            for bx in range(bx_count):
                if visited[by, bx]:
                    continue
                    
                b_type = block_types[by, bx]
                b_mean = block_means[by, bx]
                
                if b_type == 0:
                    # Complex individual block
                    visited[by, bx] = True
                    merged_blocks.append({
                        "x": bx * self.block_size,
                        "y": by * self.block_size,
                        "w": self.block_size,
                        "h": self.block_size,
                        "type": "complex",
                        "mean": b_mean
                    })
                    continue
                    
                # Extend horizontally along the row
                run_w = 1
                while (bx + run_w < bx_count) and (not visited[by, bx + run_w]) and \
                      (block_types[by, bx + run_w] == 1) and \
                      (abs(block_means[by, bx + run_w] - b_mean) < 4.0):
                    run_w += 1
                    
                # Extend vertically down the columns
                run_h = 1
                can_expand = True
                while (by + run_h < by_count) and can_expand:
                    # Check if entire row span can merge
                    for k in range(run_w):
                        if visited[by + run_h, bx + k] or (block_types[by + run_h, bx + k] != 1) or \
                           (abs(block_means[by + run_h, bx + k] - b_mean) >= 4.0):
                            can_expand = False
                            break
                    if can_expand:
                        run_h += 1
                        
                # Mark region as visited
                visited[by:by+run_h, bx:bx+run_w] = True
                
                merged_blocks.append({
                    "x": bx * self.block_size,
                    "y": by * self.block_size,
                    "w": run_w * self.block_size,
                    "h": run_h * self.block_size,
                    "type": "merged_flat" if (run_w > 1 or run_h > 1) else "single_flat",
                    "mean": b_mean,
                    "merged_count": run_w * run_h
                })

        # 3. Emit partial edge blocks for non-multiple-of-block-size frames so no
        #    trailing rows/columns are silently dropped. These are always treated
        #    as standalone (complex) blocks; they do not participate in merging.
        #
        #    The frame's trailing edge has up to THREE disjoint regions that must
        #    all be emitted independently whenever the corresponding dimension is
        #    not a multiple of `block_size`:
        #      (a) bottom strip  -- rows [by*bs : h], cols [0 : bx*bs]
        #      (b) right strip   -- rows [0 : by*bs], cols [bx*bs : w]
        #      (c) bottom-right corner -- rows [by*bs : h], cols [bx*bs : w]
        #    The previous code nested (c) inside the `if edge_h > 0:` branch and
        #    only emitted (b) under `elif edge_w > 0:`, so a frame with BOTH edge
        #    residuals dropped the right-edge interior strip (b): e.g. 100x60 @
        #    bs=16 left 192 px (cols 96:100, rows 0:48) uncovered; 60x100 left
        #    1152. The three regions are now emitted unconditionally on their own
        #    predicates so every pixel is covered exactly once.
        edge_block_count = 0
        if edge_h > 0:
            # (a) Bottom edge strip across the interior columns.
            y0 = by_count * bs
            for bx in range(bx_count):
                x0 = bx * bs
                blk = frame[y0:y0+edge_h, x0:x0+bs]
                merged_blocks.append({
                    "x": x0, "y": y0, "w": bs, "h": int(edge_h),
                    "type": "edge_partial", "mean": float(np.mean(blk)),
                    "merged_count": 1
                })
                edge_block_count += 1
        if edge_w > 0:
            # (b) Right edge strip across the interior rows.
            x0 = bx_count * bs
            for by in range(by_count):
                y0 = by * bs
                blk = frame[y0:y0+bs, x0:w]
                merged_blocks.append({
                    "x": x0, "y": y0, "w": int(edge_w), "h": bs,
                    "type": "edge_partial", "mean": float(np.mean(blk)),
                    "merged_count": 1
                })
                edge_block_count += 1
        if edge_h > 0 and edge_w > 0:
            # (c) Bottom-right corner (the intersection of the two strips).
            y0 = by_count * bs
            x0 = bx_count * bs
            blk = frame[y0:y0+edge_h, x0:w]
            merged_blocks.append({
                "x": x0, "y": y0, "w": int(edge_w), "h": int(edge_h),
                "type": "edge_partial", "mean": float(np.mean(blk)),
                "merged_count": 1
            })
            edge_block_count += 1

        t_elapsed = (time.perf_counter() - t0) * 1000.0

        # raw_blocks counts the ideal full-size tiling plus the partial edge cells
        # so the compression ratio reflects the true frame coverage.
        raw_block_count = bx_count * by_count + edge_block_count
        compressed_count = len(merged_blocks)
        reduction_ratio = raw_block_count / max(1, compressed_count)

        stats = {
            "elapsed_ms": t_elapsed,
            "raw_blocks": raw_block_count,
            "merged_blocks": compressed_count,
            "compression_ratio": reduction_ratio,
            "dct_operations_saved": raw_block_count - compressed_count,
            "edge_blocks": edge_block_count
        }
        return merged_blocks, stats
