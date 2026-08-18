"""
Tree-Free Fast Multipole Method (FMM) in JAX & Python
Powered by Elastic Non-Reordering Spatial Hash Table (Farach-Colton et al. 2025)
and Carrier, Greengard, & Rokhlin (1988) / Greengard & Rokhlin (1987) mathematical formulations.

Implements:
1. Spatial Morton 2D/3D z-order encoding
2. P2M: Particle to Multipole Expansion (CGR88 Eq. 2.1 - 2.2)
3. M2M: Multipole to Multipole translation across spatial scales (CGR88 Theorem 2.2)
4. M2L: Multipole to Local translation (CGR88 Theorem 2.3)
5. L2L: Local to Local translation (CGR88 Theorem 2.4)
6. L2P & P2P: Local to Particle evaluation and Direct Near-Field summation
7. TreeFreeFMM: Non-reordering elastic spatial hash FMM
8. CGR88AdaptiveFMM: Exact adaptive quadtree FMM
9. GreengardRokhlin87RegularFMM: Exact uniform quadtree FMM
"""

from __future__ import annotations
import numpy as np
import math
import cmath
import time
from typing import Tuple, List, Dict, Optional, Union

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

try:
    from .elastic_hash import ElasticHashTable
    from .cgr88_adaptive_fmm import (
        CGR88AdaptiveFMM,
        TreeFreeElasticAdaptiveFMM,
        GreengardRokhlin87RegularFMM,
        AdaptiveQuadTree,
        morton_encode_box,
        decode_morton_box,
        exact_direct_nbody_2d,
        exact_direct_nbody_forces_2d,
        p2m as cgr88_p2m,
        m2m as cgr88_m2m,
        m2l as cgr88_m2l,
        l2l as cgr88_l2l,
        l2p as cgr88_l2p,
        l2p_force as cgr88_l2p_force,
        p2l as cgr88_p2l,
        m2p as cgr88_m2p,
        p2p_potential_and_force
    )
except ImportError:
    from elastic_hash import ElasticHashTable
    from cgr88_adaptive_fmm import (
        CGR88AdaptiveFMM,
        TreeFreeElasticAdaptiveFMM,
        GreengardRokhlin87RegularFMM,
        AdaptiveQuadTree,
        morton_encode_box,
        decode_morton_box,
        exact_direct_nbody_2d,
        exact_direct_nbody_forces_2d,
        p2m as cgr88_p2m,
        m2m as cgr88_m2m,
        m2l as cgr88_m2l,
        l2l as cgr88_l2l,
        l2p as cgr88_l2p,
        l2p_force as cgr88_l2p_force,
        p2l as cgr88_p2l,
        m2p as cgr88_m2p,
        p2p_potential_and_force
    )

ORDER = 8  # Default expansion order


# -------------------------------------------------------------
# 1. Morton / Spatial Indexing
# -------------------------------------------------------------
def morton_encode_2d(x: float, y: float, depth: int = 4) -> int:
    """Encodes normalized [0, 1] coordinates into a Morton z-order integer."""
    grid_res = 1 << depth
    ix = min(grid_res - 1, max(0, int(x * grid_res)))
    iy = min(grid_res - 1, max(0, int(y * grid_res)))
    
    key = 0
    for i in range(depth):
        key |= ((ix >> i) & 1) << (2 * i)
        key |= ((iy >> i) & 1) << (2 * i + 1)
    return (depth << 24) | key


def decode_morton_2d(code: int) -> Tuple[int, int, int]:
    depth = code >> 24
    raw = code & 0xFFFFFF
    ix, iy = 0, 0
    for i in range(depth):
        ix |= ((raw >> (2 * i)) & 1) << i
        iy |= ((raw >> (2 * i + 1)) & 1) << i
    return depth, ix, iy


def get_box_center_2d(depth: int, ix: int, iy: int) -> Tuple[float, float]:
    box_size = 1.0 / (1 << depth)
    cx = (ix + 0.5) * box_size
    cy = (iy + 0.5) * box_size
    return cx, cy


def morton_key_from_indices(depth: int, ix: int, iy: int) -> int:
    """Canonical level-tagged Morton key for integer box coordinates."""
    return morton_encode_2d((ix + 0.5) / (1 << depth),
                            (iy + 0.5) / (1 << depth), depth)


# -------------------------------------------------------------
# 2. Multipole & Field Kernels (CGR88 Complex Logarithmic Series)
# -------------------------------------------------------------
def p2m(points: np.ndarray, charges: np.ndarray, center: complex, order: int = ORDER) -> np.ndarray:
    return cgr88_p2m(points, charges, center, p=order)


def m2m(m_coeffs: np.ndarray, child_center: complex, parent_center: complex, order: int = ORDER) -> np.ndarray:
    return cgr88_m2m(m_coeffs, child_center, parent_center, p=order)


def l2l(l_coeffs: np.ndarray, parent_center: complex, child_center: complex, order: int = ORDER) -> np.ndarray:
    return cgr88_l2l(l_coeffs, parent_center, child_center, p=order)


def m2l(m_coeffs: np.ndarray, src_center: complex, dst_center: complex, order: int = ORDER) -> np.ndarray:
    return cgr88_m2l(m_coeffs, src_center, dst_center, p=order)


def eval_local(l_coeffs: np.ndarray, target_pt: complex, center: complex, order: int = ORDER) -> float:
    return cgr88_l2p(l_coeffs, target_pt, center, p=order)


def eval_local_force(l_coeffs: np.ndarray, target_pt: complex, center: complex, order: int = ORDER) -> Tuple[float, float]:
    return cgr88_l2p_force(l_coeffs, target_pt, center, p=order)


# -------------------------------------------------------------
# 3. Complete Tree-Free FMM using Elastic Non-Reordering Hash
# -------------------------------------------------------------
class TreeFreeFMM:
    def __init__(self, depth: int = 4, order: int = ORDER, softening: float = 0.0):
        if depth < 0 or order < 0:
            raise ValueError("depth and order must be non-negative")
        self.depth = depth
        self.order = order
        self.softening = softening
        self.hash_table = ElasticHashTable(capacity=(1 << (2 * depth)) * 2, delta=0.05)
        self.boxes: Dict[int, Dict] = {}
        self.levels: Dict[int, Dict[int, Dict]] = {}
        self.far_field = np.empty(0)
        self.near_field = np.empty(0)

    def build_hash_octree(self, positions: np.ndarray, charges: np.ndarray):
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        if charges.ndim != 1 or len(charges) != len(positions):
            raise ValueError("charges must have shape (N,) matching positions")
            
        self.hash_table = ElasticHashTable(capacity=(1 << (2 * self.depth)) * 2, delta=0.05)
        self.boxes.clear()
        self.levels.clear()
        
        leaf_map: Dict[int, List[int]] = {}
        for i, point in enumerate(positions):
            key = morton_encode_2d(point[0], point[1], self.depth)
            leaf_map.setdefault(key, []).append(i)
            
        for leaf_key, indices in leaf_map.items():
            _, ix, iy = decode_morton_2d(leaf_key)
            for level in range(self.depth + 1):
                shift = self.depth - level
                px, py = ix >> shift, iy >> shift
                key = morton_key_from_indices(level, px, py)
                node = self.levels.setdefault(level, {}).get(key)
                if node is None:
                    center = complex(*get_box_center_2d(level, px, py))
                    node = {
                        'key': key,
                        'level': level,
                        'ix': px,
                        'iy': py,
                        'center': center,
                        'indices': [],
                        'm_coeffs': np.zeros(self.order + 1, dtype=np.complex128),
                        'l_coeffs': np.zeros(self.order + 1, dtype=np.complex128)
                    }
                    self.levels[level][key] = node
                node['indices'].extend(indices)
                
        # Leaves use P2M; parents use M2M
        for node in self.levels.get(self.depth, {}).values():
            node['m_coeffs'] = p2m(positions[node['indices']], charges[node['indices']], node['center'], self.order)
            self.boxes[node['key']] = node
            self.hash_table.insert(node['key'], node['key'])
            
        for level in range(self.depth - 1, -1, -1):
            for node in self.levels[level].values():
                node['m_coeffs'].fill(0.0)
                for child in self.levels[level + 1].values():
                    if (child['ix'] >> 1) == node['ix'] and (child['iy'] >> 1) == node['iy']:
                        node['m_coeffs'] += m2m(child['m_coeffs'], child['center'], node['center'], self.order)

    def compute_field_contributions(self, positions: np.ndarray, charges: np.ndarray):
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        if len(positions) == 0:
            self.far_field = self.near_field = np.empty(0, dtype=np.float64)
            return self.far_field.copy(), self.near_field.copy()
            
        self.build_hash_octree(positions, charges)
        N = len(positions)
        far = np.zeros(N, dtype=np.float64)
        near = np.zeros(N, dtype=np.float64)
        
        # M2L translation for well-separated boxes at each level
        for level in range(1, self.depth + 1):
            nodes = list(self.levels[level].values())
            for target in nodes:
                for source in nodes:
                    if max(abs(target['ix'] - source['ix']), abs(target['iy'] - source['iy'])) <= 1:
                        continue
                    tp = self.levels[level - 1].get(morton_key_from_indices(level - 1, target['ix'] >> 1, target['iy'] >> 1))
                    sp = self.levels[level - 1].get(morton_key_from_indices(level - 1, source['ix'] >> 1, source['iy'] >> 1))
                    if tp is not None and sp is not None and max(abs(tp['ix'] - sp['ix']), abs(tp['iy'] - sp['iy'])) <= 1:
                        target['l_coeffs'] += m2l(source['m_coeffs'], source['center'], target['center'], self.order)
                        
        # Downward L2L propagation
        for level in range(1, self.depth + 1):
            for child in self.levels[level].values():
                parent_key = morton_key_from_indices(level - 1, child['ix'] >> 1, child['iy'] >> 1)
                parent = self.levels[level - 1].get(parent_key)
                if parent is not None:
                    child['l_coeffs'] += l2l(parent['l_coeffs'], parent['center'], child['center'], self.order)
                    
        grid = 1 << self.depth
        eps2 = self.softening * self.softening
        for target in self.boxes.values():
            for ti in target['indices']:
                far[ti] = eval_local(target['l_coeffs'], complex(*positions[ti]), target['center'], self.order)
                tx, ty = target['ix'], target['iy']
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = tx + dx, ty + dy
                        key = morton_encode_2d((nx + 0.5) / grid, (ny + 0.5) / grid, self.depth) if 0 <= nx < grid and 0 <= ny < grid else None
                        source = self.boxes.get(key)
                        if source is not None:
                            for si in source['indices']:
                                if si != ti:
                                    r2 = np.sum((positions[ti] - positions[si]) ** 2) + eps2
                                    near[ti] += charges[si] * 0.5 * np.log(max(r2, 1e-28))
                                    
        self.far_field, self.near_field = far, near
        return far.copy(), near.copy()

    def compute_far_and_near_field(self, positions: np.ndarray, charges: np.ndarray, return_components: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        far, near = self.compute_field_contributions(positions, charges)
        return (far, near) if return_components else far + near

    def evaluate(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        return self.compute_far_and_near_field(positions, charges, return_components=False)


# -------------------------------------------------------------
# 4. Exact O(N^2) Direct Evaluators
# -------------------------------------------------------------
def exact_direct_nbody(positions: np.ndarray, charges: np.ndarray, softening: float = 0.0) -> np.ndarray:
    return exact_direct_nbody_2d(positions, charges, softening=softening)


def exact_direct_forces(positions: np.ndarray, charges: np.ndarray, softening: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    return exact_direct_nbody_forces_2d(positions, charges, softening=softening)


if __name__ == '__main__':
    np.random.seed(42)
    N_PARTICLES = 1000
    print("=" * 70)
    print(" CGR88 ADAPTIVE & REGULAR FAST MULTIPOLE METHOD (FMM) TEST SUITE")
    print(" Carrier, Greengard, & Rokhlin (1988) / Greengard & Rokhlin (1987)")
    print("=" * 70)
    
    pos = np.random.uniform(0.05, 0.95, size=(N_PARTICLES, 2))
    charges = np.random.uniform(-1.0, 1.0, size=N_PARTICLES)
    
    # 1. Exact Reference
    t0 = time.perf_counter()
    exact_pot = exact_direct_nbody_2d(pos, charges)
    fx_exact, fy_exact = exact_direct_nbody_forces_2d(pos, charges)
    t_exact = time.perf_counter() - t0
    print(f"[-] Exact O(N^2) Direct Summation Time: {t_exact*1000:.2f} ms")
    
    # 2. CGR88 Adaptive FMM
    t0 = time.perf_counter()
    cgr_fmm = CGR88AdaptiveFMM(max_leaf_particles=20, max_depth=6, p=10)
    cgr_pot, cgr_fx, cgr_fy = cgr_fmm.evaluate(pos, charges, compute_forces=True)
    t_cgr = time.perf_counter() - t0
    print(f"[-] CGR88 Adaptive FMM Time:           {t_cgr*1000:.2f} ms")
    
    # 3. Regular FMM
    t0 = time.perf_counter()
    reg_fmm = GreengardRokhlin87RegularFMM(depth=4, p=10)
    reg_pot, reg_fx, reg_fy = reg_fmm.evaluate(pos, charges, compute_forces=True)
    t_reg = time.perf_counter() - t0
    print(f"[-] Greengard-Rokhlin 87 Regular FMM Time: {t_reg*1000:.2f} ms")
    
    # 4. Tree-Free Hash FMM
    t0 = time.perf_counter()
    tf_fmm = TreeFreeFMM(depth=4, order=8)
    tf_pot = tf_fmm.evaluate(pos, charges)
    t_tf = time.perf_counter() - t0
    print(f"[-] Tree-Free Elastic Hash FMM Time:   {t_tf*1000:.2f} ms")
    
    # Accuracy Metrics
    err_cgr_pot = np.linalg.norm(cgr_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_cgr_f = np.linalg.norm(cgr_fx - fx_exact) / np.linalg.norm(fx_exact)
    err_reg_pot = np.linalg.norm(reg_pot - exact_pot) / np.linalg.norm(exact_pot)
    err_tf_pot = np.linalg.norm(tf_pot - exact_pot) / np.linalg.norm(exact_pot)
    
    print("\n[Cross-Validation Accuracy]")
    print(f"[-] CGR88 Adaptive FMM  -> Rel Pot Error: {err_cgr_pot:.3e}, Rel Force Error: {err_cgr_f:.3e}")
    print(f"[-] Regular FMM (1987)  -> Rel Pot Error: {err_reg_pot:.3e}")
    print(f"[-] Tree-Free Hash FMM  -> Rel Pot Error: {err_tf_pot:.3e}")
    print("=" * 70)
