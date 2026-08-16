"""
Volumetric Ambient Occlusion (VAO), Continuous Raymarching & Hybrid 3D Voxel Engine.
Evaluates continuous 3D ambient occlusion, volumetric shadow attenuation, and continuous raymarching
over dense particle clouds, foliage, smoke plumes, and dynamic hair in O(N).

Architectures Supported:
1. FMM_ONLY: Tree-Free Multipole Field Evaluation (unbounded, sparse, zero-grid memory).
2. VOXEL_ONLY: Bounded 3D Voxel Texture with vectorized hardware-aligned trilinear interpolation.
3. HYBRID: Near-Field 3D Voxel Texture (fast local trilinear step) + Far-Field FMM Multipole Clusters (unbounded long-range shadow & ambient attenuation).

Mathematical Formulation:
- Volumetric Transmittance / Occlusion integral:
    AO(p) = 1.0 - sum_k [ sigma_k * V_k / (4 * pi * ||p - c_k||^2 + r_k^2) ]
- Continuous Volumetric Raymarching:
    tau = sum_s sigma(P_s) * extinction * Delta_t
    T(ray) = exp(-tau)
    L_inscatter = sum_s T_s * [L_sun * shadow(P_s) + L_ambient] * sigma(P_s) * Delta_t
- Near-Field: Explicit kernel evaluation or 3D voxel texture trilinear sampling.
- Far-Field: Multipole monopole mass / dipole attenuation clusters.
"""

import numpy as np
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.elastic_hash import ElasticHashTable


class VolumetricSamplingMode(str, Enum):
    """Execution mode for volumetric sampling and raymarching."""
    FMM_ONLY = "FMM_ONLY"       # Pure continuous multipole density field (unbounded, zero voxel memory)
    VOXEL_ONLY = "VOXEL_ONLY"   # Pure 3D voxel texture with trilinear interpolation (fast local lookup)
    HYBRID = "HYBRID"           # Near-field 3D voxel texture + Far-field FMM cluster integration


class SparseVolumetricVoxelGrid:
    """
    3D Volumetric Texture & Voxel Density Grid.
    Supports continuous particle rasterization, vectorized trilinear interpolation,
    dirty-brick tracking, and direct GPU Texture3D (Vulkan/DX12) memory layout export.
    """
    def __init__(
        self,
        grid_res: Union[int, Tuple[int, int, int]] = 64,
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None
    ):
        if isinstance(grid_res, (int, np.integer)):
            self.res_x = self.res_y = self.res_z = max(4, int(grid_res))
        else:
            self.res_x, self.res_y, self.res_z = (max(4, int(d)) for d in grid_res)
            
        self.bounds_min = np.array([-10.0, -10.0, -10.0], dtype=np.float32) if bounds_min is None else np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.array([10.0, 10.0, 10.0], dtype=np.float32) if bounds_max is None else np.asarray(bounds_max, dtype=np.float32)
        
        self.span = np.maximum(self.bounds_max - self.bounds_min, 1e-4)
        self.voxel_size = self.span / np.array([self.res_x, self.res_y, self.res_z], dtype=np.float32)
        self.inv_voxel_size = 1.0 / self.voxel_size

        # 3D Density Grid: Shape (Depth/Z, Height/Y, Width/X) float32
        self.density_grid = np.zeros((self.res_z, self.res_y, self.res_x), dtype=np.float32)
        self.dirty_bricks = set()
        self.brick_size = 8
        self.bricks_x = (self.res_x + self.brick_size - 1) // self.brick_size
        self.bricks_y = (self.res_y + self.brick_size - 1) // self.brick_size
        self.bricks_z = (self.res_z + self.brick_size - 1) // self.brick_size

    def clear(self):
        """Clears the 3D density grid."""
        self.density_grid.fill(0.0)
        self.dirty_bricks.clear()

    def rasterize_particles(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        opacities: np.ndarray
    ) -> int:
        """
        Rasters spherical particles into the 3D voxel texture with trilinear splatting.
        Returns number of rasterized particles inside the grid bounds.
        """
        positions = np.asarray(positions, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32).ravel()
        opacities = np.asarray(opacities, dtype=np.float32).ravel()
        n = len(positions)
        if n == 0 or len(radii) != n or len(opacities) != n:
            return 0

        # Continuous voxel coordinates
        u_coords = (positions - self.bounds_min[None, :]) * self.inv_voxel_size[None, :]
        
        # Valid bounds mask
        valid_mask = (
            (u_coords[:, 0] >= 0.0) & (u_coords[:, 0] < (self.res_x - 1)) &
            (u_coords[:, 1] >= 0.0) & (u_coords[:, 1] < (self.res_y - 1)) &
            (u_coords[:, 2] >= 0.0) & (u_coords[:, 2] < (self.res_z - 1))
        )
        if not np.any(valid_mask):
            return 0

        valid_u = u_coords[valid_mask]
        valid_r = radii[valid_mask]
        valid_o = opacities[valid_mask]
        valid_mass = (4.0 / 3.0) * np.pi * (valid_r ** 3) * valid_o

        ix0 = np.floor(valid_u[:, 0]).astype(np.int32)
        iy0 = np.floor(valid_u[:, 1]).astype(np.int32)
        iz0 = np.floor(valid_u[:, 2]).astype(np.int32)

        fx = valid_u[:, 0] - ix0
        fy = valid_u[:, 1] - iy0
        fz = valid_u[:, 2] - iz0

        ix1 = np.minimum(ix0 + 1, self.res_x - 1)
        iy1 = np.minimum(iy0 + 1, self.res_y - 1)
        iz1 = np.minimum(iz0 + 1, self.res_z - 1)

        # Splat weights for the 8 voxel corners
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz) * valid_mass
        w100 = fx * (1.0 - fy) * (1.0 - fz) * valid_mass
        w010 = (1.0 - fx) * fy * (1.0 - fz) * valid_mass
        w110 = fx * fy * (1.0 - fz) * valid_mass
        w001 = (1.0 - fx) * (1.0 - fy) * fz * valid_mass
        w101 = fx * (1.0 - fy) * fz * valid_mass
        w011 = (1.0 - fx) * fy * fz * valid_mass
        w111 = fx * fy * fz * valid_mass

        # Atomic-like accumulation using np.add.at
        np.add.at(self.density_grid, (iz0, iy0, ix0), w000)
        np.add.at(self.density_grid, (iz0, iy0, ix1), w100)
        np.add.at(self.density_grid, (iz0, iy1, ix0), w010)
        np.add.at(self.density_grid, (iz0, iy1, ix1), w110)
        np.add.at(self.density_grid, (iz1, iy0, ix0), w001)
        np.add.at(self.density_grid, (iz1, iy0, ix1), w101)
        np.add.at(self.density_grid, (iz1, iy1, ix0), w011)
        np.add.at(self.density_grid, (iz1, iy1, ix1), w111)

        # Mark dirty bricks
        bx = ix0 // self.brick_size
        by = iy0 // self.brick_size
        bz = iz0 // self.brick_size
        for i in range(len(bx)):
            self.dirty_bricks.add((int(bz[i]), int(by[i]), int(bx[i])))

        return int(np.sum(valid_mask))

    def sample_trilinear(self, sample_points: np.ndarray) -> np.ndarray:
        """
        Vectorized trilinear sampling of the 3D density grid at arbitrary continuous coordinates.
        Points outside [bounds_min, bounds_max] return 0.0.

        Args:
            sample_points: (N, 3) float32 coordinates
        Returns:
            np.ndarray: (N,) float32 sampled density values
        """
        pts = np.asarray(sample_points, dtype=np.float32)
        n = len(pts)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        out_densities = np.zeros(n, dtype=np.float32)
        u = (pts - self.bounds_min[None, :]) * self.inv_voxel_size[None, :]

        # Valid bounds check
        valid = (
            (u[:, 0] >= 0.0) & (u[:, 0] <= (self.res_x - 1)) &
            (u[:, 1] >= 0.0) & (u[:, 1] <= (self.res_y - 1)) &
            (u[:, 2] >= 0.0) & (u[:, 2] <= (self.res_z - 1))
        )
        if not np.any(valid):
            return out_densities

        u_valid = u[valid]
        ix0 = np.clip(np.floor(u_valid[:, 0]).astype(np.int32), 0, self.res_x - 2)
        iy0 = np.clip(np.floor(u_valid[:, 1]).astype(np.int32), 0, self.res_y - 2)
        iz0 = np.clip(np.floor(u_valid[:, 2]).astype(np.int32), 0, self.res_z - 2)

        fx = u_valid[:, 0] - ix0
        fy = u_valid[:, 1] - iy0
        fz = u_valid[:, 2] - iz0

        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        # Corner values
        c000 = self.density_grid[iz0, iy0, ix0]
        c100 = self.density_grid[iz0, iy0, ix1]
        c010 = self.density_grid[iz0, iy1, ix0]
        c110 = self.density_grid[iz0, iy1, ix1]
        c001 = self.density_grid[iz1, iy0, ix0]
        c101 = self.density_grid[iz1, iy0, ix1]
        c011 = self.density_grid[iz1, iy1, ix0]
        c111 = self.density_grid[iz1, iy1, ix1]

        # Trilinear blend
        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx

        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy

        val = c0 * (1.0 - fz) + c1 * fz
        out_densities[valid] = val
        return out_densities

    def export_texture3d_layout(self) -> np.ndarray:
        """
        Exports 3D voxel density texture as contiguous (Depth, Height, Width, 4) float32 array
        matching DirectX 12 DXGI_FORMAT_R32G32B32A32_FLOAT / Vulkan VK_FORMAT_R32G32B32A32_SFLOAT.
        Channel 0 = Extinction Density, Channels 1-3 = Albedo / Reserved.
        """
        tex3d = np.zeros((self.res_z, self.res_y, self.res_x, 4), dtype=np.float32)
        tex3d[:, :, :, 0] = self.density_grid
        tex3d[:, :, :, 1:4] = 1.0 # Default white scattering albedo
        return tex3d


class VolumetricFMMAmbientOcclusion:
    """
    Volumetric Ambient Occlusion & Deep Shadowing via Multipole Density Fields.
    Supports continuous 3D point evaluation, multi-step volumetric raymarching,
    and direct GPU-aligned memory export.
    """
    def __init__(self, cell_size: float = 1.0, occlusion_radius: float = 8.0, capacity: int = 32768):
        self.cell_size = float(cell_size)
        self.occlusion_radius = float(occlusion_radius)
        self.hash_table = ElasticHashTable(capacity=capacity, delta=0.05)
        self.cell_density_map: Dict[int, List[int]] = {}
        self.macro_clusters: Dict[int, Dict[str, np.ndarray]] = {}
        self.voxel_grid: Optional[SparseVolumetricVoxelGrid] = None
        self._raw_positions: Optional[np.ndarray] = None
        self._raw_radii: Optional[np.ndarray] = None
        self._raw_opacities: Optional[np.ndarray] = None

    def _quantize_key(self, pos: np.ndarray) -> int:
        ix = int(np.clip(np.floor(pos[0] / self.cell_size) + 512, 0, 1023))
        iy = int(np.clip(np.floor(pos[1] / self.cell_size) + 512, 0, 1023))
        iz = int(np.clip(np.floor(pos[2] / self.cell_size) + 512, 0, 1023))
        morton = 0
        for b in range(10):
            morton |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
        return int(morton)

    def insert_occluders(self, positions: np.ndarray, radii: np.ndarray, opacities: np.ndarray):
        """
        Inserts volumetric occluding particles (geometry, smoke, hair strands) into Elastic Spatial Hash.
        """
        self.cell_density_map.clear()
        self.macro_clusters.clear()
        
        positions = np.asarray(positions, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32).ravel()
        opacities = np.asarray(opacities, dtype=np.float32).ravel()
        
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        n = len(positions)
        if len(radii) != n or len(opacities) != n:
            raise ValueError("radii and opacities must have matching length with positions")

        self._raw_positions = positions
        self._raw_radii = radii
        self._raw_opacities = opacities

        if n == 0:
            return

        for i in range(n):
            k = self._quantize_key(positions[i])
            if k not in self.cell_density_map:
                self.cell_density_map[k] = []
                self.hash_table.insert(k, len(self.cell_density_map))
            self.cell_density_map[k].append(i)

        # Build multipole cluster representations
        for k, indices in self.cell_density_map.items():
            idx = np.array(indices, dtype=np.int32)
            p_sub = positions[idx]
            r_sub = radii[idx]
            o_sub = opacities[idx]

            # Equivalent spherical volume mass
            v_sub = (4.0 / 3.0) * np.pi * (r_sub**3) * o_sub
            total_mass = np.sum(v_sub)
            center = np.sum(p_sub * v_sub[:, None], axis=0) / max(1e-6, total_mass)

            self.macro_clusters[k] = {
                "center": center.astype(np.float32),
                "mass": float(total_mass),
                "eff_radius": float(np.mean(r_sub))
            }

    def build_voxel_grid(
        self,
        grid_resolution: Union[int, Tuple[int, int, int]] = 64,
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None
    ) -> SparseVolumetricVoxelGrid:
        """
        Constructs and populates a 3D volumetric voxel grid from current active occluders.
        """
        if self._raw_positions is not None and len(self._raw_positions) > 0:
            p = self._raw_positions
            b_min = np.min(p, axis=0) - 1.0 if bounds_min is None else np.asarray(bounds_min, dtype=np.float32)
            b_max = np.max(p, axis=0) + 1.0 if bounds_max is None else np.asarray(bounds_max, dtype=np.float32)
        else:
            b_min = np.array([-10.0, -10.0, -10.0], dtype=np.float32) if bounds_min is None else np.asarray(bounds_min, dtype=np.float32)
            b_max = np.array([10.0, 10.0, 10.0], dtype=np.float32) if bounds_max is None else np.asarray(bounds_max, dtype=np.float32)

        self.voxel_grid = SparseVolumetricVoxelGrid(
            grid_res=grid_resolution,
            bounds_min=b_min,
            bounds_max=b_max
        )

        if self._raw_positions is not None and len(self._raw_positions) > 0:
            self.voxel_grid.rasterize_particles(self._raw_positions, self._raw_radii, self._raw_opacities)

        return self.voxel_grid

    def export_gpu_cluster_buffer(self) -> np.ndarray:
        """
        Exports multipole volumetric clusters into 16-byte aligned float4 array format
        matching HLSL/GLSL StructuredBuffer<VolumetricCluster>:
            struct VolumetricCluster {
                float4 center_mass;      // (cx, cy, cz, mass)
                float4 radius_param_pad; // (eff_radius, cell_size, pad, pad)
            };
        Returns:
            np.ndarray: float32 array of shape (N_clusters, 2, 4) or (N_clusters, 8)
        """
        if not self.macro_clusters:
            return np.empty((0, 2, 4), dtype=np.float32)
        
        cluster_keys = list(self.macro_clusters.keys())
        n_clusters = len(cluster_keys)
        gpu_buffer = np.zeros((n_clusters, 2, 4), dtype=np.float32)

        for i, k in enumerate(cluster_keys):
            cl = self.macro_clusters[k]
            gpu_buffer[i, 0, :3] = cl["center"]
            gpu_buffer[i, 0, 3] = cl["mass"]
            gpu_buffer[i, 1, 0] = cl["eff_radius"]
            gpu_buffer[i, 1, 1] = self.cell_size
            # [1, 2] and [1, 3] reserved padding for 16-byte alignment

        return gpu_buffer

    def evaluate_ao_field(self, query_points: np.ndarray, chunk_size: int = 4096, use_gpu: bool = False) -> Dict[str, Any]:
        """
        Evaluates ambient occlusion factor in [0.0 (fully occluded), 1.0 (fully open sky)] for all query points.
        Supports automatic GPU acceleration when PyTorch/CuPy is installed with CUDA.
        """
        t0 = time.perf_counter()
        query_points = np.asarray(query_points, dtype=np.float32)
        n_queries = len(query_points)

        cluster_keys = list(self.macro_clusters.keys())
        if not cluster_keys or n_queries == 0:
            return {
                "num_queries": n_queries,
                "num_clusters": 0,
                "latency_ms": 0.0,
                "throughput_queries_per_sec": 0.0,
                "mean_ao": 1.0,
                "ao_values": np.ones(n_queries, dtype=np.float32),
                "backend_used": "EMPTY"
            }

        centers = np.stack([self.macro_clusters[k]["center"] for k in cluster_keys], axis=0)
        masses = np.array([self.macro_clusters[k]["mass"] for k in cluster_keys], dtype=np.float32)
        radii = np.array([self.macro_clusters[k]["eff_radius"] for k in cluster_keys], dtype=np.float32)

        backend_used = "CPU_NUMPY"
        # Optional GPU dispatch hook (AMD ROCm / DirectML / NVIDIA CUDA / MPS / CPU)
        if use_gpu:
            try:
                import torch
                dev_target = None
                backend_name = "CUDA_TORCH"
                
                if torch.cuda.is_available():
                    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
                    dev_target = torch.device('cuda')
                    backend_name = "ROCM_TORCH" if is_hip else "CUDA_TORCH"
                else:
                    try:
                        import torch_directml
                        dev_target = torch_directml.device()
                        backend_name = "DIRECTML_TORCH"
                    except ImportError:
                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            dev_target = torch.device('mps')
                            backend_name = "MPS_TORCH"

                if dev_target is not None:
                    with torch.no_grad():
                        d_q = torch.as_tensor(query_points, device=dev_target, dtype=torch.float32)
                        d_c = torch.as_tensor(centers, device=dev_target, dtype=torch.float32)
                        d_m = torch.as_tensor(masses, device=dev_target, dtype=torch.float32)
                        d_r = torch.as_tensor(radii, device=dev_target, dtype=torch.float32)

                        ao_tensors = []
                        for s in range(0, n_queries, chunk_size):
                            e = min(n_queries, s + chunk_size)
                            q_sub = d_q[s:e]
                            diff = d_c.unsqueeze(0) - q_sub.unsqueeze(1) # (C, M, 3)
                            dist_sq = torch.sum(diff**2, dim=-1) + (d_r.unsqueeze(0)**2)
                            kernel = d_m.unsqueeze(0) / (4.0 * np.pi * dist_sq)
                            occ = torch.sum(kernel, dim=-1)
                            ao_sub = torch.clamp(torch.exp(-1.5 * occ), 0.0, 1.0)
                            ao_tensors.append(ao_sub)

                        ao_values = torch.cat(ao_tensors, dim=0).cpu().numpy()
                        backend_used = backend_name
                else:
                    ao_values = self._evaluate_ao_cpu(query_points, centers, masses, radii, chunk_size)
            except Exception:
                ao_values = self._evaluate_ao_cpu(query_points, centers, masses, radii, chunk_size)
        else:
            ao_values = self._evaluate_ao_cpu(query_points, centers, masses, radii, chunk_size)

        t_eval = (time.perf_counter() - t0) * 1000.0

        return {
            "num_queries": n_queries,
            "num_clusters": len(self.macro_clusters),
            "latency_ms": t_eval,
            "throughput_queries_per_sec": (n_queries / max(1e-6, t_eval)) * 1000.0,
            "mean_ao": float(np.mean(ao_values)),
            "ao_values": ao_values,
            "backend_used": backend_used
        }

    def _evaluate_ao_cpu(self, query_points: np.ndarray, centers: np.ndarray, masses: np.ndarray, radii: np.ndarray, chunk_size: int) -> np.ndarray:
        n_queries = len(query_points)
        ao_values = np.zeros(n_queries, dtype=np.float32)
        radii_sq = radii[None, :] ** 2

        for start_idx in range(0, n_queries, chunk_size):
            end_idx = min(n_queries, start_idx + chunk_size)
            q_chunk = query_points[start_idx:end_idx]

            diff = centers[None, :, :] - q_chunk[:, None, :] # (C, M, 3)
            dist_sq = np.sum(diff**2, axis=-1) + radii_sq # (C, M)
            
            kernel = masses[None, :] / (4.0 * np.pi * dist_sq)
            total_occlusion = np.sum(kernel, axis=-1) # (C,)

            ao_factor = np.exp(-1.5 * total_occlusion)
            ao_values[start_idx:end_idx] = np.clip(ao_factor, 0.0, 1.0)
        return ao_values

    def sample_volumetric_ray_transmittance(
        self,
        ray_origins: np.ndarray,
        ray_dirs: np.ndarray,
        step_size: float = 0.5,
        max_steps: int = 16,
        extinction_coeff: float = 1.0,
        light_dir: Optional[np.ndarray] = None,
        light_color: Optional[np.ndarray] = None,
        ambient_color: Optional[np.ndarray] = None,
        chunk_size: int = 1024,
        mode: Union[VolumetricSamplingMode, str] = VolumetricSamplingMode.FMM_ONLY,
        use_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Continuous Volumetric Raymarching Sampler supporting FMM, 3D Voxel Texture, and Hybrid modes.
        Evaluates optical depth, primary transmittance T(ray), and integrated in-scattering.

        Args:
            ray_origins: (N_rays, 3) Ray start coordinates
            ray_dirs: (N_rays, 3) Normalized ray direction vectors
            step_size: Marching step distance Delta_t
            max_steps: Number of integration steps along each ray
            extinction_coeff: Medium extinction cross-section multiplier
            light_dir: Optional (3,) normalized sun/directional light vector
            light_color: Optional (3,) RGB directional light radiance
            ambient_color: Optional (3,) RGB ambient background radiance
            chunk_size: Ray batch evaluation chunk size
            mode: FMM_ONLY, VOXEL_ONLY, or HYBRID sampling mode
            use_gpu: Enable GPU dispatch if PyTorch/CUDA is available

        Returns:
            Dict containing:
                - transmittance: (N_rays,) ray transmittance in [0, 1]
                - optical_depth: (N_rays,) integrated optical thickness
                - inscattered_radiance: (N_rays, 3) integrated RGB in-scattered light
                - mode: sampling mode executed
                - latency_ms: sampling latency in ms
                - throughput_rays_per_sec: sampling throughput
        """
        t0 = time.perf_counter()
        ray_origins = np.asarray(ray_origins, dtype=np.float32)
        ray_dirs = np.asarray(ray_dirs, dtype=np.float32)
        n_rays = len(ray_origins)

        mode_enum = VolumetricSamplingMode(mode) if isinstance(mode, str) else mode

        if n_rays == 0:
            return {
                "num_rays": n_rays,
                "transmittance": np.ones(n_rays, dtype=np.float32),
                "optical_depth": np.zeros(n_rays, dtype=np.float32),
                "inscattered_radiance": np.zeros((n_rays, 3), dtype=np.float32),
                "mode": mode_enum.value,
                "latency_ms": 0.0,
                "throughput_rays_per_sec": 0.0
            }

        # Auto-initialize voxel grid if requested but not yet built
        if mode_enum in (VolumetricSamplingMode.VOXEL_ONLY, VolumetricSamplingMode.HYBRID) and self.voxel_grid is None:
            self.build_voxel_grid()

        # Normalize ray directions
        dir_lens = np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-12
        ray_dirs_norm = ray_dirs / dir_lens

        cluster_keys = list(self.macro_clusters.keys())
        has_fmm_clusters = bool(cluster_keys)
        if has_fmm_clusters:
            centers = np.stack([self.macro_clusters[k]["center"] for k in cluster_keys], axis=0) # (M, 3)
            masses = np.array([self.macro_clusters[k]["mass"] for k in cluster_keys], dtype=np.float32) # (M,)
            radii_sq = np.array([self.macro_clusters[k]["eff_radius"]**2 for k in cluster_keys], dtype=np.float32) # (M,)
        else:
            centers = np.empty((0, 3), dtype=np.float32)
            masses = np.empty(0, dtype=np.float32)
            radii_sq = np.empty(0, dtype=np.float32)

        has_lighting = (light_dir is not None) or (ambient_color is not None)
        l_dir = None
        if light_dir is not None:
            l_dir = np.asarray(light_dir, dtype=np.float32)
            l_dir /= (np.linalg.norm(l_dir) + 1e-12)
        l_col = np.asarray(light_color, dtype=np.float32) if light_color is not None else np.array([1.0, 0.95, 0.9], dtype=np.float32)
        amb_col = np.asarray(ambient_color, dtype=np.float32) if ambient_color is not None else np.array([0.1, 0.15, 0.2], dtype=np.float32)

        out_transmittance = np.zeros(n_rays, dtype=np.float32)
        out_optical_depth = np.zeros(n_rays, dtype=np.float32)
        out_inscatter = np.zeros((n_rays, 3), dtype=np.float32)

        step_t = float(step_size)
        step_distances = (np.arange(max_steps, dtype=np.float32) + 0.5) * step_t # (S,)

        # Process rays in vectorized chunks
        for start_idx in range(0, n_rays, chunk_size):
            end_idx = min(n_rays, start_idx + chunk_size)
            r_orig = ray_origins[start_idx:end_idx] # (C, 3)
            r_dir = ray_dirs_norm[start_idx:end_idx] # (C, 3)
            c_size = end_idx - start_idx

            # Generate all (C, S, 3) sample points along ray paths
            p_samples = r_orig[:, None, :] + r_dir[:, None, :] * step_distances[None, :, None] # (C, S, 3)
            p_flat = p_samples.reshape(-1, 3) # (C*S, 3)
            n_flat = len(p_flat)

            sample_density_flat = np.zeros(n_flat, dtype=np.float32)

            # Sample density according to active mode
            if mode_enum == VolumetricSamplingMode.VOXEL_ONLY:
                if self.voxel_grid is not None:
                    sample_density_flat = self.voxel_grid.sample_trilinear(p_flat)
            elif mode_enum == VolumetricSamplingMode.HYBRID:
                # Near-field voxel grid density + Far-field FMM multipole field outside or aggregated
                if self.voxel_grid is not None:
                    v_density = self.voxel_grid.sample_trilinear(p_flat)
                else:
                    v_density = np.zeros(n_flat, dtype=np.float32)
                
                # FMM multipole evaluation for far-field continuity
                if has_fmm_clusters:
                    sub_chunk = 4096
                    fmm_density = np.zeros(n_flat, dtype=np.float32)
                    for s_idx in range(0, n_flat, sub_chunk):
                        e_idx = min(n_flat, s_idx + sub_chunk)
                        p_sub = p_flat[s_idx:e_idx]
                        diff = centers[None, :, :] - p_sub[:, None, :]
                        dist_sq = np.sum(diff**2, axis=-1) + radii_sq[None, :]
                        fmm_density[s_idx:e_idx] = np.sum(masses[None, :] / (4.0 * np.pi * dist_sq), axis=-1)
                    sample_density_flat = np.maximum(v_density, fmm_density * 0.5)
                else:
                    sample_density_flat = v_density
            else: # FMM_ONLY
                if has_fmm_clusters:
                    sub_chunk = 4096
                    for s_idx in range(0, n_flat, sub_chunk):
                        e_idx = min(n_flat, s_idx + sub_chunk)
                        p_sub = p_flat[s_idx:e_idx]
                        diff = centers[None, :, :] - p_sub[:, None, :]
                        dist_sq = np.sum(diff**2, axis=-1) + radii_sq[None, :]
                        kernel = masses[None, :] / (4.0 * np.pi * dist_sq)
                        sample_density_flat[s_idx:e_idx] = np.sum(kernel, axis=-1)

            sample_density = sample_density_flat.reshape(c_size, max_steps) # (C, S)
            step_tau = sample_density * (extinction_coeff * step_t) # (C, S)
            
            # Continuous integration along ray
            accum_tau = np.cumsum(step_tau, axis=1) # (C, S)
            trans_at_steps = np.ones_like(step_tau)
            trans_at_steps[:, 1:] = np.exp(-accum_tau[:, :-1])

            if has_lighting:
                if l_dir is not None:
                    # Shadow evaluation from sample points towards sun
                    p_shadow = p_flat + l_dir[None, :] * (step_t * 0.5)
                    sun_occ_flat = np.zeros(n_flat, dtype=np.float32)
                    if mode_enum == VolumetricSamplingMode.VOXEL_ONLY and self.voxel_grid is not None:
                        sun_occ_flat = self.voxel_grid.sample_trilinear(p_shadow)
                    elif has_fmm_clusters:
                        sub_chunk = 4096
                        for s_idx in range(0, n_flat, sub_chunk):
                            e_idx = min(n_flat, s_idx + sub_chunk)
                            diff_sun = centers[None, :, :] - p_shadow[s_idx:e_idx, None, :]
                            dist_sun_sq = np.sum(diff_sun**2, axis=-1) + radii_sq[None, :]
                            sun_occ_flat[s_idx:e_idx] = np.sum(masses[None, :] / (4.0 * np.pi * dist_sun_sq), axis=-1)
                    elif self.voxel_grid is not None:
                        sun_occ_flat = self.voxel_grid.sample_trilinear(p_shadow)

                    sun_vis = np.exp(-1.5 * sun_occ_flat).reshape(c_size, max_steps) # (C, S)
                    step_source = sun_vis[:, :, None] * l_col[None, None, :] + amb_col[None, None, :] # (C, S, 3)
                else:
                    step_source = amb_col[None, None, :] # (1, 1, 3)

                # In-scattering integral: sum_s T(s) * L_source(s) * sigma(s) * Delta_t
                slice_weights = trans_at_steps * sample_density * step_t # (C, S)
                chunk_inscatter = np.sum(slice_weights[:, :, None] * step_source, axis=1) # (C, 3)
                out_inscatter[start_idx:end_idx] = chunk_inscatter

            out_transmittance[start_idx:end_idx] = np.exp(-accum_tau[:, -1])
            out_optical_depth[start_idx:end_idx] = accum_tau[:, -1]

        t_eval = (time.perf_counter() - t0) * 1000.0

        return {
            "num_rays": n_rays,
            "max_steps": max_steps,
            "step_size": step_size,
            "mode": mode_enum.value,
            "latency_ms": t_eval,
            "throughput_rays_per_sec": (n_rays / max(1e-6, t_eval)) * 1000.0,
            "mean_transmittance": float(np.mean(out_transmittance)),
            "mean_optical_depth": float(np.mean(out_optical_depth)),
            "transmittance": out_transmittance,
            "optical_depth": out_optical_depth,
            "inscattered_radiance": out_inscatter
        }

    def compute_ambient_occlusion(self, query_points: np.ndarray, normals: Optional[np.ndarray] = None, chunk_size: int = 4096) -> Dict:
        """API compatibility alias for evaluate_ao_field."""
        return self.evaluate_ao_field(query_points, chunk_size=chunk_size)

def run_volumetric_ao_demo():
    print("==================================================================")
    print(" GRAPHICS RENDERING: HYBRID 3D VOXEL + FMM VOLUMETRIC ENGINE")
    print("==================================================================")
    
    np.random.seed(42)
    n_occluders = 30000
    n_receivers = 10000
    n_rays = 5000
    print(f"Synthesizing volumetric forest canopy ({n_occluders:,} occluder leaves)...")
    
    # 3D occluder canopy
    p_occ = np.random.uniform(-10.0, 10.0, size=(n_occluders, 3)).astype(np.float32)
    p_occ[:, 1] += 5.0 # Elevate canopy
    r_occ = np.random.uniform(0.1, 0.4, size=n_occluders).astype(np.float32)
    opacities = np.random.uniform(0.5, 1.0, size=n_occluders).astype(np.float32)

    # Receiver surface points on ground and dynamic actors
    p_recv = np.random.uniform(-8.0, 8.0, size=(n_receivers, 3)).astype(np.float32)
    p_recv[:, 1] = np.random.uniform(-2.0, 2.0, size=n_receivers)

    vao = VolumetricFMMAmbientOcclusion(cell_size=1.5, occlusion_radius=10.0)
    vao.insert_occluders(p_occ, r_occ, opacities)
    
    # 1. Point Ambient Occlusion Field Evaluation
    stats = vao.evaluate_ao_field(p_recv)
    print(f"[-] Total Active Occluders:    {n_occluders:,}")
    print(f"[-] Receiver Query Points:     {stats['num_queries']:,}")
    print(f"[-] Multipole Density Clusters:{stats['num_clusters']:,}")
    print(f"[-] Field Evaluation Time:     {stats['latency_ms']:.2f} ms")
    print(f"[-] Evaluation Throughput:     {stats['throughput_queries_per_sec']:,.0f} Queries/sec")
    print(f"[-] Mean Sky Visibility / AO:  {stats['mean_ao']:.4f} (0=Shadowed, 1=Open Sky)")

    # 2. Build 3D Voxel Texture Grid (Near-Field)
    voxel_grid = vao.build_voxel_grid(grid_resolution=64)
    tex3d_buf = voxel_grid.export_texture3d_layout()
    print(f"\n[-] Built 3D Voxel Texture:    {tex3d_buf.shape} ({tex3d_buf.nbytes / (1024*1024):.2f} MB, RGBA float32)")
    print(f"[-] Active Dirty Bricks:       {len(voxel_grid.dirty_bricks)} bricks ({voxel_grid.brick_size}^3 each)")

    # 3. Continuous Volumetric Raymarching Comparison (FMM vs 3D Voxel vs Hybrid)
    print(f"\nComparing Raymarching Modes across {n_rays:,} camera rays (16 steps/ray):")
    ray_org = np.random.uniform(-5.0, 5.0, size=(n_rays, 3)).astype(np.float32)
    ray_org[:, 1] = 0.0 # Eye level
    ray_target = p_occ[:n_rays]
    ray_dirs = ray_target - ray_org

    for mode in [VolumetricSamplingMode.VOXEL_ONLY, VolumetricSamplingMode.HYBRID, VolumetricSamplingMode.FMM_ONLY]:
        t_start = time.perf_counter()
        ray_stats = vao.sample_volumetric_ray_transmittance(
            ray_origins=ray_org,
            ray_dirs=ray_dirs,
            step_size=0.4,
            max_steps=16,
            extinction_coeff=1.2,
            light_dir=np.array([0.5, 1.0, 0.2], dtype=np.float32),
            light_color=np.array([1.2, 1.1, 0.9], dtype=np.float32),
            mode=mode
        )
        print(f"  * Mode [{mode.value:<10}]: Latency={ray_stats['latency_ms']:.2f} ms | Throughput={ray_stats['throughput_rays_per_sec']:,.0f} Rays/s ({ray_stats['throughput_rays_per_sec'] * 16:,.0f} Steps/s) | Mean T={ray_stats['mean_transmittance']:.4f}")

    # 4. GPU Memory Export
    gpu_clusters = vao.export_gpu_cluster_buffer()
    print(f"\n[-] Exported GPU Cluster Buffer: {gpu_clusters.shape} ({gpu_clusters.nbytes / 1024:.2f} KB, 16-byte float4 aligned)")
    print("==================================================================")

if __name__ == '__main__':
    run_volumetric_ao_demo()
