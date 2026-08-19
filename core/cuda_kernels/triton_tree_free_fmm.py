"""
Direct O(N^2) N-Body Reference Solver (OpenAI Triton GPU Kernel)
================================================================
NOTE: this is NOT a Fast Multipole Method. It is a direct all-pairs
Coulomb P2P solver with block tiling, kept as a GPU reference baseline
for cross-validating approximate FMM engines. Despite the historic
filename, no multipole operators are implemented here.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_p2p_coulomb_kernel(
        coords_ptr,      # (N, 3) float32
        charges_ptr,     # (N,) float32
        out_pot_ptr,     # (N,) float32
        out_forces_ptr,  # (N, 3) float32
        num_particles,
        softening_sq: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Block-tiled Triton GPU kernel computing all-pairs Coulomb potential & vector forces
        in fast on-chip SRAM with zero intermediate global memory writes.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_particles

        # Load target particle coordinates and charges
        x_i = tl.load(coords_ptr + offsets * 3 + 0, mask=mask, other=0.0)
        y_i = tl.load(coords_ptr + offsets * 3 + 1, mask=mask, other=0.0)
        z_i = tl.load(coords_ptr + offsets * 3 + 2, mask=mask, other=0.0)
        q_i = tl.load(charges_ptr + offsets, mask=mask, other=0.0)

        acc_phi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_fx = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_fy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc_fz = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Loop over source particle blocks
        for block_j in range(0, tl.cdiv(num_particles, BLOCK_SIZE)):
            offsets_j = block_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_j = offsets_j < num_particles

            x_j = tl.load(coords_ptr + offsets_j * 3 + 0, mask=mask_j, other=0.0)
            y_j = tl.load(coords_ptr + offsets_j * 3 + 1, mask=mask_j, other=0.0)
            z_j = tl.load(coords_ptr + offsets_j * 3 + 2, mask=mask_j, other=0.0)
            q_j = tl.load(charges_ptr + offsets_j, mask=mask_j, other=0.0)

            # Pairwise distance: (BLOCK_SIZE, BLOCK_SIZE)
            dx = x_i[:, None] - x_j[None, :]
            dy = y_i[:, None] - y_j[None, :]
            dz = z_i[:, None] - z_j[None, :]

            r_sq = dx * dx + dy * dy + dz * dz + softening_sq
            inv_r = tl.rsqrt(r_sq)
            inv_r3 = inv_r * inv_r * inv_r

            # Exclude padded lanes and self-interactions, matching the CPU/JAX references.
            pair_mask = mask[:, None] & mask_j[None, :] & (offsets[:, None] != offsets_j[None, :])

            # Accumulate potential and forces
            phi_term = tl.where(pair_mask, q_j[None, :] * inv_r, 0.0)
            acc_phi += tl.sum(phi_term, axis=1)

            f_scalar = tl.where(pair_mask, q_i[:, None] * q_j[None, :] * inv_r3, 0.0)
            acc_fx += tl.sum(f_scalar * dx, axis=1)
            acc_fy += tl.sum(f_scalar * dy, axis=1)
            acc_fz += tl.sum(f_scalar * dz, axis=1)

        # Store results
        tl.store(out_pot_ptr + offsets, acc_phi, mask=mask)
        tl.store(out_forces_ptr + offsets * 3 + 0, acc_fx, mask=mask)
        tl.store(out_forces_ptr + offsets * 3 + 1, acc_fy, mask=mask)
        tl.store(out_forces_ptr + offsets * 3 + 2, acc_fz, mask=mask)


    def triton_direct_nbody(
        coords: torch.Tensor,
        charges: torch.Tensor,
        softening: float = 1e-3,
        block_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch entrypoint for the direct O(N^2) all-pairs Triton N-body
        reference solver (potentials + forces).
        """
        assert (coords.is_cuda and charges.is_cuda) or (coords.device.type in ('cuda', 'hip') and charges.device.type in ('cuda', 'hip')), (
            "Inputs must be on a CUDA or AMD ROCm/HIP GPU device"
        )
        N = coords.shape[0]
        out_pot = torch.empty(N, device=coords.device, dtype=torch.float32)
        out_forces = torch.empty((N, 3), device=coords.device, dtype=torch.float32)

        grid = (triton.cdiv(N, block_size),)
        _triton_p2p_coulomb_kernel[grid](
            coords,
            charges,
            out_pot,
            out_forces,
            N,
            softening_sq=softening ** 2,
            BLOCK_SIZE=block_size
        )
        return out_pot, out_forces


    # Backwards-compatible alias (historic, misleading name kept for callers)
    triton_tree_free_nbody = triton_direct_nbody


def check_triton_availability():
    if not HAS_TRITON:
        print("[INFO] PyTorch & Triton are not available. Install via `pip install torch triton` for GPU kernel execution.")
    else:
        print("[SUCCESS] OpenAI Triton & PyTorch are loaded and GPU kernel is ready.")


if __name__ == "__main__":
    check_triton_availability()
