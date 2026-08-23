"""
Carrier, Greengard, & Rokhlin (1988) 2D Adaptive Fast Multipole Method (CGR88)
and Greengard & Rokhlin (1987) Regular Fast Multipole Method.

Reference:
- J. Carrier, L. Greengard, and V. Rokhlin,
  "A Fast Adaptive Multipole Algorithm for Particle Simulations",
  SIAM Journal on Scientific and Statistical Computing, 9(4):669-686, 1988.
- L. Greengard and V. Rokhlin,
  "A Fast Algorithm for Particle Simulations",
  Journal of Computational Physics, 73(2):325-348, 1987.

Provides:
1. Exact multipole and local expansion operators in the complex plane (P2M, M2M, M2L, L2L, L2P, P2L, M2P, P2P).
2. Complete 4-interaction-list construction for arbitrary adaptive quadtrees (List 1, List 2, List 3, List 4).
3. Exact O(N) Adaptive FMM with potential and vector force calculation.
4. Regular (Uniform) FMM for fixed-depth quadtrees.
5. Tree-Free Hash-Indexed FMM: adaptive FMM operators on a non-reordering funnel-hash
   cell index (core.elastic_hash.ElasticHashTable, Farach-Colton, Krapivin, & Kuszmaul, 2025).
6. Exact O(N^2) direct Coulomb / logarithmic N-body ground-truth evaluator for potentials and forces.
"""

from __future__ import annotations
import math
import cmath
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np


# =============================================================================
# 1. MATHEMATICAL EXPANSION OPERATORS (CGR88 THEOREMS 2.1 - 2.4)
# =============================================================================

def p2m(points: np.ndarray, charges: np.ndarray, center: complex, p: int = 8) -> np.ndarray:
    """
    Particle to Multipole (P2M) Expansion (CGR88 Eq. 2.1 - 2.2).
    
    Given m particles with charges q_i at locations z_i with |z_i - z0| < r,
    the potential for |z - z0| > r is:
        Phi(z) = a_0 * ln(z - z0) + sum_{k=1}^p a_k / (z - z0)^k
    where:
        a_0 = sum(q_i)
        a_k = -sum(q_i * (z_i - z0)^k) / k
    """
    coeffs = np.zeros(p + 1, dtype=np.complex128)
    if len(points) == 0:
        return coeffs
    
    z_pts = points[:, 0] + 1j * points[:, 1]
    dz = z_pts - center
    
    coeffs[0] = np.sum(charges)
    for k in range(1, p + 1):
        coeffs[k] = -np.sum(charges * (dz ** k)) / k
    return coeffs


def m2m(m_coeffs: np.ndarray, src_center: complex, dst_center: complex, p: int = 8) -> np.ndarray:
    """
    Multipole to Multipole (M2M) Translation (CGR88 Theorem 2.2).
    
    Translates a multipole expansion centered at z0 to a new center z1 (where |z0 - z1| = d).
    With delta = z0 - z1:
        b_0 = a_0
        b_l = - (a_0 * delta^l) / l + sum_{k=1}^l a_k * binom(l-1, k-1) * delta^(l-k)
    """
    delta = src_center - dst_center
    b = np.zeros(p + 1, dtype=np.complex128)
    b[0] = m_coeffs[0]
    
    for l in range(1, p + 1):
        term = -m_coeffs[0] * (delta ** l) / l
        for k in range(1, l + 1):
            term += m_coeffs[k] * math.comb(l - 1, k - 1) * (delta ** (l - k))
        b[l] = term
    return b


def m2l(m_coeffs: np.ndarray, src_center: complex, dst_center: complex, p: int = 8) -> np.ndarray:
    """
    Multipole to Local (M2L) Translation (CGR88 Theorem 2.3).
    
    Translates a multipole expansion centered at z0 into a local Taylor series
    expansion centered at z1 (where z0 and z1 are well-separated, |z1 - z0| > r0 + r1).
    Let delta = z1 - z0:
        c_0 = a_0 * ln(delta) + sum_{k=1}^p a_k / (delta^k)
        c_l = (a_0 * (-1)^(l-1)) / (l * delta^l) + sum_{k=1}^p [ (-1)^l * binom(k+l-1, l) * a_k ] / (delta^(k+l))
    """
    delta = dst_center - src_center
    if abs(delta) == 0.0:
        raise ValueError("Source and destination centers must be well-separated (nonzero distance)")
    
    c = np.zeros(p + 1, dtype=np.complex128)
    
    # c_0
    c[0] = m_coeffs[0] * cmath.log(delta)
    for k in range(1, p + 1):
        c[0] += m_coeffs[k] / (delta ** k)
        
    # c_l for l >= 1
    for l in range(1, p + 1):
        term = m_coeffs[0] * ((-1) ** (l - 1)) / (l * (delta ** l))
        for k in range(1, p + 1):
            factor = ((-1) ** l) * math.comb(k + l - 1, l)
            term += factor * m_coeffs[k] / (delta ** (k + l))
        c[l] = term
    return c


def l2l(l_coeffs: np.ndarray, src_center: complex, dst_center: complex, p: int = 8) -> np.ndarray:
    """
    Local to Local (L2L) Translation (CGR88 Theorem 2.4).
    
    Translates a local expansion centered at z0 to a child center z1.
    Let delta = z1 - z0:
        d_l = sum_{k=l}^p c_k * binom(k, l) * delta^(k-l)
    """
    delta = dst_center - src_center
    d = np.zeros(p + 1, dtype=np.complex128)
    for l in range(p + 1):
        term = 0.0 + 0.0j
        for k in range(l, p + 1):
            term += l_coeffs[k] * math.comb(k, l) * (delta ** (k - l))
        d[l] = term
    return d


def p2l(points: np.ndarray, charges: np.ndarray, center: complex, p: int = 8) -> np.ndarray:
    """
    Particle to Local (P2L) Expansion (for CGR88 List 4 interactions).
    
    Converts distant particles directly into a local Taylor expansion around center z0:
        c_0 = sum q_i * ln(z0 - z_i)
        c_l = sum q_i * (-1)^(l-1) / [ l * (z0 - z_i)^l ]
    """
    c = np.zeros(p + 1, dtype=np.complex128)
    if len(points) == 0:
        return c
        
    z_pts = points[:, 0] + 1j * points[:, 1]
    delta = center - z_pts  # z0 - z_i
    
    for i in range(len(points)):
        q = charges[i]
        d = delta[i]
        if abs(d) == 0.0:
            continue
        c[0] += q * cmath.log(d)
        for l in range(1, p + 1):
            c[l] += q * ((-1) ** (l - 1)) / (l * (d ** l))
    return c


def m2p(m_coeffs: np.ndarray, center: complex, target_pt: complex, p: int = 8) -> Tuple[float, complex]:
    """
    Multipole to Particle (M2P) Evaluation (for CGR88 List 3 interactions).
    
    Evaluates potential and complex field from multipole expansion at target point:
        Phi(z) = a_0 * ln(z - z0) + sum_{k=1}^p a_k / (z - z0)^k
        dPhi/dz = a_0 / (z - z0) - sum_{k=1}^p k * a_k / (z - z0)^(k+1)
    Returns:
        (potential_real, complex_field = dPhi/dz)
    """
    dz = target_pt - center
    if abs(dz) == 0.0:
        return 0.0, 0.0 + 0.0j
    
    val = m_coeffs[0] * cmath.log(dz)
    deriv = m_coeffs[0] / dz
    for k in range(1, p + 1):
        val += m_coeffs[k] / (dz ** k)
        deriv -= (k * m_coeffs[k]) / (dz ** (k + 1))
        
    pot = val.real
    return pot, deriv


def l2p(l_coeffs: np.ndarray, target_pt: complex, center: complex, p: int = 8) -> float:
    """
    Local to Particle (L2P) Potential Evaluation.
    
    Phi(z) = Re( sum_{l=0}^p c_l * (z - z0)^l )
    """
    dz = target_pt - center
    val = l_coeffs[0]
    dz_k = dz
    for l in range(1, p + 1):
        val += l_coeffs[l] * dz_k
        dz_k *= dz
    return float(val.real)


def l2p_force(l_coeffs: np.ndarray, target_pt: complex, center: complex, p: int = 8) -> Tuple[float, float]:
    """
    Local to Particle (L2P) Vector Force Evaluation.
    
    The potential is phi(x, y) = Re( Psi(z) ).
    Gradient:
        grad phi = ( d(Re(Psi))/dx, d(Re(Psi))/dy ) = ( Re(Psi'), -Im(Psi') )
    Force F = - grad phi = ( -Re(Psi'), Im(Psi') )
    where Psi'(z) = sum_{l=1}^p l * c_l * (z - z0)^(l-1)
    """
    dz = target_pt - center
    deriv = 0.0 + 0.0j
    dz_k = 1.0 + 0.0j
    for l in range(1, p + 1):
        deriv += l * l_coeffs[l] * dz_k
        dz_k *= dz
    fx = -float(deriv.real)
    fy = float(deriv.imag)
    return fx, fy


def p2p_potential_and_force(
    target_pts: np.ndarray,
    src_pts: np.ndarray,
    src_charges: np.ndarray,
    softening: float = 0.0,
    exclude_self: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct Particle to Particle (P2P) Potential and Vector Force calculation.
    
    Potential: phi(r) = sum q_j * ln( sqrt(r^2 + eps^2) )
    Force: F_i = - sum q_i * q_j * (r_i - r_j) / (r^2 + eps^2)
    (or acceleration / field: E_i = - sum q_j * (r_i - r_j) / (r^2 + eps^2))
    """
    Nt = len(target_pts)
    Ns = len(src_pts)
    pot = np.zeros(Nt, dtype=np.float64)
    fx = np.zeros(Nt, dtype=np.float64)
    fy = np.zeros(Nt, dtype=np.float64)
    if Nt == 0 or Ns == 0:
        return pot, fx, fy
    
    eps2 = softening * softening
    for i in range(Nt):
        dx = target_pts[i, 0] - src_pts[:, 0]
        dy = target_pts[i, 1] - src_pts[:, 1]
        r2 = dx * dx + dy * dy + eps2

        if exclude_self:
            # Mask out the self pair by INDEX (i == j), not by distance.
            # exclude_self is only set when target_pts and src_pts are the
            # same particle set in the same order (the cell's own
            # particle_indices), so target index i corresponds to source
            # index i. Distance-based masking (r2 >= 1e-28) leaks the self
            # pair in when softening > 0 because r2 = dx^2 + dy^2 + eps^2
            # is eps^2 > 1e-28 for the self pair (e.g. softening 0.05 ->
            # self-potential leak q_i * 0.5 * ln(eps^2) = q_i * ln(0.05)
            # ~ -2.996 * q_i instead of 0). Mirrors
            # exact_direct_nbody_2d below (:1196-1197).
            r2_safe = np.where(r2 < 1e-28, 1.0, r2)
            mask = np.ones(Ns, dtype=bool)
            if i < Ns:
                mask[i] = False
            q_masked = np.where(mask, src_charges, 0.0)
            pot[i] = np.sum(q_masked * 0.5 * np.log(r2_safe))
            inv_r2 = np.where(mask, 1.0 / r2_safe, 0.0)
            fx[i] = -np.sum(q_masked * dx * inv_r2)
            fy[i] = -np.sum(q_masked * dy * inv_r2)
        else:
            r2_safe = np.where(r2 < 1e-28, 1.0, r2)
            pot[i] = np.sum(src_charges * 0.5 * np.log(r2_safe))
            inv_r2 = 1.0 / r2_safe
            fx[i] = -np.sum(src_charges * dx * inv_r2)
            fy[i] = -np.sum(src_charges * dy * inv_r2)
            
    return pot, fx, fy


# =============================================================================
# 2. ADAPTIVE QUADTREE DATA STRUCTURE (CGR88 SECTION 3)
# =============================================================================

@dataclass
class QuadBox:
    box_id: int
    level: int
    ix: int
    iy: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center: complex
    particle_indices: List[int] = field(default_factory=list)
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    is_leaf: bool = True
    
    # Adaptive FMM Interaction Lists
    list1: List[int] = field(default_factory=list)  # Leaf neighbors (P2P)
    list2: List[int] = field(default_factory=list)  # Same-level well-separated (M2L)
    list3: List[int] = field(default_factory=list)  # Distant small descendants of colleagues (M2L/M2P)
    list4: List[int] = field(default_factory=list)  # Dual large leaves adjacent to ancestors (P2L)
    colleagues: List[int] = field(default_factory=list)  # Same-level adjacent boxes
    
    # Expansions
    m_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.complex128))
    l_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.complex128))


class AdaptiveQuadTree:
    """
    2D Adaptive Quadtree with full adaptive FMM interaction lists (Lists 1, 2, 3, 4).
    """
    def __init__(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        max_leaf_particles: int = 20,
        max_depth: int = 10,
        p: int = 8,
        domain_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.charges = np.asarray(charges, dtype=np.float64)
        self.N = len(positions)
        self.max_leaf_particles = max_leaf_particles
        self.max_depth = max_depth
        self.p = p
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds
        
        self.boxes: Dict[int, QuadBox] = {}
        self.level_boxes: Dict[int, List[int]] = {}
        self.leaves: List[int] = []
        self._next_box_id = 0
        
        self._build_tree()
        self._build_colleagues()
        self._build_adaptive_fmm_interaction_lists()

    def _create_box(
        self,
        level: int,
        ix: int,
        iy: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        parent_id: Optional[int]
    ) -> QuadBox:
        bid = self._next_box_id
        self._next_box_id += 1
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        box = QuadBox(
            box_id=bid,
            level=level,
            ix=ix,
            iy=iy,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            center=cx + 1j * cy,
            parent_id=parent_id,
            m_coeffs=np.zeros(self.p + 1, dtype=np.complex128),
            l_coeffs=np.zeros(self.p + 1, dtype=np.complex128)
        )
        self.boxes[bid] = box
        self.level_boxes.setdefault(level, []).append(bid)
        return box

    def _build_tree(self):
        root = self._create_box(
            level=0, ix=0, iy=0,
            x_min=self.x_min, x_max=self.x_max,
            y_min=self.y_min, y_max=self.y_max,
            parent_id=None
        )
        root.particle_indices = list(range(self.N))
        
        queue = [root.box_id]
        while queue:
            bid = queue.pop(0)
            box = self.boxes[bid]
            
            if len(box.particle_indices) <= self.max_leaf_particles or box.level >= self.max_depth:
                box.is_leaf = True
                self.leaves.append(bid)
                continue
            
            # Subdivide box into 4 children (quadrants)
            box.is_leaf = False
            cx = 0.5 * (box.x_min + box.x_max)
            cy = 0.5 * (box.y_min + box.y_max)
            
            quad_bounds = [
                (box.x_min, cx, box.y_min, cy, 2 * box.ix, 2 * box.iy),         # Q0: Bottom-Left
                (cx, box.x_max, box.y_min, cy, 2 * box.ix + 1, 2 * box.iy),     # Q1: Bottom-Right
                (box.x_min, cx, cy, box.y_max, 2 * box.ix, 2 * box.iy + 1),     # Q2: Top-Left
                (cx, box.x_max, cy, box.y_max, 2 * box.ix + 1, 2 * box.iy + 1)  # Q3: Top-Right
            ]
            
            child_indices = [[] for _ in range(4)]
            for p_idx in box.particle_indices:
                px, py = self.positions[p_idx]
                q_idx = (0 if px < cx else 1) + (0 if py < cy else 2)
                child_indices[q_idx].append(p_idx)
                
            for q_idx, (xmin, xmax, ymin, ymax, c_ix, c_iy) in enumerate(quad_bounds):
                child = self._create_box(
                    level=box.level + 1,
                    ix=c_ix,
                    iy=c_iy,
                    x_min=xmin,
                    x_max=xmax,
                    y_min=ymin,
                    y_max=ymax,
                    parent_id=bid
                )
                child.particle_indices = child_indices[q_idx]
                box.children_ids.append(child.box_id)
                queue.append(child.box_id)

    def _are_adjacent(self, b1: QuadBox, b2: QuadBox) -> bool:
        """Check if two boxes touch / overlap in spatial coordinates."""
        return not (
            b1.x_max < b2.x_min - 1e-12 or
            b2.x_max < b1.x_min - 1e-12 or
            b1.y_max < b2.y_min - 1e-12 or
            b2.y_max < b1.y_min - 1e-12
        )

    def _build_colleagues(self):
        """Build colleagues (adjacent boxes at the exact same tree level)."""
        for level, b_list in self.level_boxes.items():
            level_map = {(self.boxes[bid].ix, self.boxes[bid].iy): bid for bid in b_list}
            for bid in b_list:
                box = self.boxes[bid]
                colls = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neigh_key = (box.ix + dx, box.iy + dy)
                        if neigh_key in level_map:
                            colls.append(level_map[neigh_key])
                box.colleagues = colls

    def _build_adaptive_fmm_interaction_lists(self):
        """
        Constructs the 4 adaptive FMM Interaction Lists:
        - List 1(b): If b is leaf, all leaf boxes adjacent to b (including b).
        - List 2(b): Children of colleagues of parent(b) that are well-separated from b.
        - List 3(b): If b is leaf, descendants of colleagues(b) that are separated from b, but parent adjacent to b.
        - List 4(b): Leaf boxes c adjacent to parent(b) where b is in List 3(c).
        """
        # 1. List 1: Leaf-to-Leaf adjacent neighbors
        # Restructured: instead of recursing from the ROOT for every leaf
        # (O(leaves x total_boxes)), start from the leaf's colleagues and
        # the leaf's ancestors' colleagues (bounded 3x3 neighborhoods at
        # each level from the leaf's level up to the root). This captures
        # all adjacent leaves -- same-level colleagues, finer descendants
        # of non-leaf colleagues, and coarser leaves that are colleagues
        # of the leaf's ancestors -- without scanning the whole tree.
        # Produces the identical List-1 SET as the previous root recursion
        # (verified by set-equality snapshot on a 2000-particle clustered
        # scene). The order of entries may differ (level-order vs DFS),
        # which only affects floating-point accumulation order in the P2P
        # sum (well within the 1e-5 test tolerance).
        for bid in self.leaves:
            b = self.boxes[bid]
            l1 = [bid]  # b itself is always in List-1
            visited = {bid}

            def find_adjacent_leaves_from(node_id: int):
                if node_id in visited:
                    return
                visited.add(node_id)
                node = self.boxes[node_id]
                if not self._are_adjacent(b, node):
                    return
                if node.is_leaf:
                    l1.append(node_id)
                else:
                    for ch_id in node.children_ids:
                        find_adjacent_leaves_from(ch_id)

            # Walk from b's level up to the root. At each level, check the
            # 3x3 colleagues of b (or b's ancestor at that level). A
            # colleague that is a leaf and adjacent to b is a coarser or
            # same-level adjacent leaf; a non-leaf colleague's descendant
            # adjacent leaves are found by recursion. The visited set
            # prevents duplicates across levels.
            ancestor_id = bid
            for level in range(b.level, -1, -1):
                if level == b.level:
                    node = b
                else:
                    node = self.boxes[ancestor_id]
                for coll_id in node.colleagues:
                    if coll_id not in visited:
                        find_adjacent_leaves_from(coll_id)
                # Move up to parent for the next (coarser) level.
                if level > 0:
                    ancestor_id = node.parent_id
            b.list1 = l1

        # 2. List 2: Same level well-separated boxes (children of parent's colleagues not adjacent to b)
        for bid, b in self.boxes.items():
            if b.parent_id is None:
                continue
            parent = self.boxes[b.parent_id]
            for p_coll_id in parent.colleagues:
                p_coll = self.boxes[p_coll_id]
                for ch_id in p_coll.children_ids:
                    ch = self.boxes[ch_id]
                    if not self._are_adjacent(b, ch):
                        b.list2.append(ch_id)

        # 3. List 3 & List 4:
        # For leaf b, List 3 contains all descendants of colleagues(b) not adjacent to b,
        # but whose parent is adjacent to b.
        # By dual reciprocity, if d in List 3(b), then b in List 4(d).
        for bid in self.leaves:
            b = self.boxes[bid]
            for coll_id in b.colleagues:
                coll = self.boxes[coll_id]
                if coll.is_leaf:
                    continue  # Leaves at same level are handled in List 1 / List 2
                
                # Recursively inspect children of coll
                def inspect_descendant(curr_id: int):
                    curr = self.boxes[curr_id]
                    if self._are_adjacent(b, curr):
                        if not curr.is_leaf:
                            for ch_id in curr.children_ids:
                                inspect_descendant(ch_id)
                    else:
                        # curr is well-separated from b, but its parent was adjacent to b!
                        b.list3.append(curr_id)
                        self.boxes[curr_id].list4.append(bid)
                
                for ch_id in coll.children_ids:
                    inspect_descendant(ch_id)


# =============================================================================
# 3. CARRIER, GREENGARD, & ROKHLIN (1988) ADAPTIVE FMM ENGINE
# =============================================================================

class AdaptiveFMM:
    """
    Exact implementation of the Carrier, Greengard, & Rokhlin (1988)
    Adaptive Fast Multipole Method in 2D.
    """
    def __init__(
        self,
        max_leaf_particles: int = 20,
        max_depth: int = 8,
        p: int = 8,
        softening: float = 0.0
    ):
        self.max_leaf_particles = max_leaf_particles
        self.max_depth = max_depth
        self.p = p
        self.softening = softening
        self.tree: Optional[AdaptiveQuadTree] = None

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Evaluates potential (and optionally vector forces) for all N particles.
        
        Returns:
            potentials (N,)
            forces_x (N,) [if compute_forces is True]
            forces_y (N,) [if compute_forces is True]
        """
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        # Domain bounds
        x_min, x_max = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
        y_min, y_max = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
        margin = max(1e-4, 0.02 * max(x_max - x_min, y_max - y_min, 1e-3))
        bounds = (x_min - margin, x_max + margin, y_min - margin, y_max + margin)

        # 1. Build Adaptive QuadTree & Interaction Lists
        self.tree = AdaptiveQuadTree(
            positions=positions,
            charges=charges,
            max_leaf_particles=self.max_leaf_particles,
            max_depth=self.max_depth,
            p=self.p,
            domain_bounds=bounds
        )
        tree = self.tree

        # ---------------------------------------------------------------------
        # 2. UPWARD PASS: P2M at leaves and M2M upwards to root
        # ---------------------------------------------------------------------
        # Leaf P2M
        for bid in tree.leaves:
            box = tree.boxes[bid]
            if len(box.particle_indices) > 0:
                pts = positions[box.particle_indices]
                q = charges[box.particle_indices]
                box.m_coeffs = p2m(pts, q, box.center, self.p)

        # Bottom-up M2M
        max_lvl = max(tree.level_boxes.keys())
        for lvl in range(max_lvl, -1, -1):
            for bid in tree.level_boxes.get(lvl, []):
                box = tree.boxes[bid]
                if not box.is_leaf:
                    box.m_coeffs.fill(0.0)
                    for ch_id in box.children_ids:
                        ch = tree.boxes[ch_id]
                        box.m_coeffs += m2m(ch.m_coeffs, ch.center, box.center, self.p)

        # ---------------------------------------------------------------------
        # 3. DOWNWARD PASS: L2L + List 2 M2L + List 4 P2L
        # ---------------------------------------------------------------------
        for lvl in range(1, max_lvl + 1):
            for bid in tree.level_boxes.get(lvl, []):
                box = tree.boxes[bid]
                
                # Shift local expansion from parent via L2L
                if box.parent_id is not None:
                    parent = tree.boxes[box.parent_id]
                    box.l_coeffs += l2l(parent.l_coeffs, parent.center, box.center, self.p)
                
                # List 2: Same level well-separated boxes -> M2L
                for src_id in box.list2:
                    src_box = tree.boxes[src_id]
                    box.l_coeffs += m2l(src_box.m_coeffs, src_box.center, box.center, self.p)
                
                # List 4: Distant large leaf boxes -> P2L into box local expansion
                for c_id in box.list4:
                    c_box = tree.boxes[c_id]
                    if len(c_box.particle_indices) > 0:
                        c_pts = positions[c_box.particle_indices]
                        c_q = charges[c_box.particle_indices]
                        box.l_coeffs += p2l(c_pts, c_q, box.center, self.p)

        # ---------------------------------------------------------------------
        # 4. PARTICLE EVALUATION: Leaf L2P + List 3 M2P + List 1 Direct P2P
        # ---------------------------------------------------------------------
        potentials = np.zeros(N, dtype=np.float64)
        forces_x = np.zeros(N, dtype=np.float64)
        forces_y = np.zeros(N, dtype=np.float64)

        for bid in tree.leaves:
            box = tree.boxes[bid]
            if len(box.particle_indices) == 0:
                continue

            # Local far-field evaluation (L2P)
            for idx in box.particle_indices:
                pt_complex = complex(positions[idx, 0], positions[idx, 1])
                potentials[idx] += l2p(box.l_coeffs, pt_complex, box.center, self.p)
                if compute_forces:
                    fx, fy = l2p_force(box.l_coeffs, pt_complex, box.center, self.p)
                    forces_x[idx] += fx
                    forces_y[idx] += fy

            # List 3: Distant small descendants of colleagues -> evaluate multipole directly at leaf particles (M2P)
            for d_id in box.list3:
                d_box = tree.boxes[d_id]
                for idx in box.particle_indices:
                    pt_complex = complex(positions[idx, 0], positions[idx, 1])
                    pot_d, deriv_d = m2p(d_box.m_coeffs, d_box.center, pt_complex, self.p)
                    potentials[idx] += pot_d
                    if compute_forces:
                        forces_x[idx] += -deriv_d.real
                        forces_y[idx] += deriv_d.imag

            # Near-field direct evaluation (List 1 P2P)
            tgt_pts = positions[box.particle_indices]
            for n_id in box.list1:
                n_box = tree.boxes[n_id]
                if len(n_box.particle_indices) == 0:
                    continue
                src_pts = positions[n_box.particle_indices]
                src_q = charges[n_box.particle_indices]
                
                is_self = (n_id == bid)
                p_near, fx_near, fy_near = p2p_potential_and_force(
                    tgt_pts, src_pts, src_q,
                    softening=self.softening,
                    exclude_self=is_self
                )
                potentials[box.particle_indices] += p_near
                if compute_forces:
                    forces_x[box.particle_indices] += fx_near
                    forces_y[box.particle_indices] += fy_near

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials


# =============================================================================
# 4. GREENGARD & ROKHLIN (1987) REGULAR (UNIFORM) FMM ENGINE
# =============================================================================

class GreengardRokhlin87RegularFMM:
    """
    Exact implementation of the original Greengard & Rokhlin (1987)
    Regular / Uniform Fast Multipole Method on a fixed-depth quadtree.
    """
    def __init__(self, depth: int = 4, p: int = 8, softening: float = 0.0):
        self.depth = depth
        self.p = p
        self.softening = softening

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        grid_res = 1 << self.depth
        box_size = 1.0 / grid_res

        # Bin particles into uniform leaf grid
        leaf_particles: Dict[Tuple[int, int], List[int]] = {}
        ix = np.clip((positions[:, 0] * grid_res).astype(np.int32), 0, grid_res - 1)
        iy = np.clip((positions[:, 1] * grid_res).astype(np.int32), 0, grid_res - 1)
        for i in range(N):
            leaf_particles.setdefault((int(ix[i]), int(iy[i])), []).append(i)

        # Multi-level data structure
        # levels[lvl][(ix, iy)] = {'center': complex, 'm': np.ndarray, 'l': np.ndarray}
        levels: Dict[int, Dict[Tuple[int, int], Dict]] = {lvl: {} for lvl in range(self.depth + 1)}

        def get_or_create_node(lvl: int, gx: int, gy: int) -> Dict:
            if (gx, gy) not in levels[lvl]:
                res = 1 << lvl
                bs = 1.0 / res
                cx = (gx + 0.5) * bs
                cy = (gy + 0.5) * bs
                levels[lvl][(gx, gy)] = {
                    'center': complex(cx, cy),
                    'm': np.zeros(self.p + 1, dtype=np.complex128),
                    'l': np.zeros(self.p + 1, dtype=np.complex128),
                    'indices': []
                }
            return levels[lvl][(gx, gy)]

        # 1. P2M at uniform leaf level
        for (gx, gy), indices in leaf_particles.items():
            node = get_or_create_node(self.depth, gx, gy)
            node['indices'] = indices
            pts = positions[indices]
            q = charges[indices]
            node['m'] = p2m(pts, q, node['center'], self.p)

        # 2. Upward M2M pass
        for lvl in range(self.depth - 1, -1, -1):
            for (gx, gy), child_node in levels[lvl + 1].items():
                px, py = gx >> 1, gy >> 1
                parent_node = get_or_create_node(lvl, px, py)
                parent_node['m'] += m2m(child_node['m'], child_node['center'], parent_node['center'], self.p)

        # 3. Downward Pass: M2L at each level from List 2 (colleagues' children not adjacent)
        for lvl in range(2, self.depth + 1):
            nodes = list(levels[lvl].items())
            for (tx, ty), target in nodes:
                # Parent coordinates
                px, py = tx >> 1, ty >> 1
                for pdx in (-1, 0, 1):
                    for pdy in (-1, 0, 1):
                        spx, spy = px + pdx, py + pdy
                        for cdx in (0, 1):
                            for cdy in (0, 1):
                                sx = (spx << 1) + cdx
                                sy = (spy << 1) + cdy
                                # Well-separated if not in 3x3 adjacent neighborhood of (tx, ty)
                                if abs(sx - tx) > 1 or abs(sy - ty) > 1:
                                    src = levels[lvl].get((sx, sy))
                                    if src is not None:
                                        target['l'] += m2l(src['m'], src['center'], target['center'], self.p)

            # Downward L2L pass from parent to child
            for (gx, gy), child_node in levels[lvl].items():
                px, py = gx >> 1, gy >> 1
                parent_node = levels[lvl - 1].get((px, py))
                if parent_node is not None:
                    child_node['l'] += l2l(parent_node['l'], parent_node['center'], child_node['center'], self.p)

        # 4. Evaluation at particles: Leaf L2P + 3x3 Adjacent Leaf P2P
        potentials = np.zeros(N, dtype=np.float64)
        forces_x = np.zeros(N, dtype=np.float64)
        forces_y = np.zeros(N, dtype=np.float64)

        for (tx, ty), target in levels[self.depth].items():
            t_indices = target['indices']
            if not t_indices:
                continue

            # L2P far field
            for idx in t_indices:
                pt_complex = complex(positions[idx, 0], positions[idx, 1])
                potentials[idx] += l2p(target['l'], pt_complex, target['center'], self.p)
                if compute_forces:
                    fx, fy = l2p_force(target['l'], pt_complex, target['center'], self.p)
                    forces_x[idx] += fx
                    forces_y[idx] += fy

            # 3x3 P2P near field
            tgt_pts = positions[t_indices]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    sx, sy = tx + dx, ty + dy
                    src = levels[self.depth].get((sx, sy))
                    if src is not None and src['indices']:
                        s_indices = src['indices']
                        src_pts = positions[s_indices]
                        src_q = charges[s_indices]
                        is_self = (dx == 0 and dy == 0)
                        p_near, fx_near, fy_near = p2p_potential_and_force(
                            tgt_pts, src_pts, src_q,
                            softening=self.softening,
                            exclude_self=is_self
                        )
                        potentials[t_indices] += p_near
                        if compute_forces:
                            forces_x[t_indices] += fx_near
                            forces_y[t_indices] += fy_near

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials


try:
    from .elastic_hash import ElasticHashTable
except ImportError:
    from elastic_hash import ElasticHashTable


# =============================================================================
# 5. TREE-FREE ELASTIC ADAPTIVE FMM (NON-REORDERING HASH ACCELERATED)
# =============================================================================

def morton_encode_box(level: int, ix: int, iy: int) -> int:
    """Interleaves coordinates into a level-tagged Morton key."""
    key = 0
    for i in range(level):
        key |= ((ix >> i) & 1) << (2 * i)
        key |= ((iy >> i) & 1) << (2 * i + 1)
    return (level << 24) | key


def decode_morton_box(code: int) -> Tuple[int, int, int]:
    """Decodes level-tagged Morton key to (level, ix, iy)."""
    level = code >> 24
    raw = code & 0xFFFFFF
    ix, iy = 0, 0
    for i in range(level):
        ix |= ((raw >> (2 * i)) & 1) << i
        iy |= ((raw >> (2 * i + 1)) & 1) << i
    return level, ix, iy


class TreeFreeElasticAdaptiveFMM:
    """
    Tree-Free Adaptive FMM: Carrier, Greengard, & Rokhlin (1988) 4-list
    adaptive expansion mathematics indexed by a non-reordering funnel hash.

    The AUTHORITATIVE cell index is core.elastic_hash.ElasticHashTable
    (funnel hashing, Farach-Colton, Krapivin, & Kuszmaul, 2025, arXiv:2501.02305):
    the level-tagged Morton key of every cell maps, through the funnel
    probe sequence (alpha slabs of beta-slot sub-arrays plus the B/C
    overflow region), to that cell's record. There is no pointer-based
    tree and no auxiliary dict index: parent, child, colleague, and
    interaction-list resolution go exclusively through funnel hash
    lookups. After each build a sorted snapshot of the cell keys
    (`self.cell_keys`) is taken once so the linear passes iterate cells in
    a deterministic cache-friendly order; membership and data access still
    route through the hash. AdaptiveFMM above is the classical
    dict/tree reference implementation; the two engines agree numerically
    to <1e-12 relative (tests/core/test_adaptive_fmm_cross_validation.py).
    """
    def __init__(
        self,
        max_leaf_particles: int = 20,
        base_depth: int = 2,
        max_depth: int = 7,
        p: int = 8,
        softening: float = 0.0
    ):
        self.max_leaf_particles = max_leaf_particles
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.p = p
        self.softening = softening
        # Funnel hash: authoritative cell index (Morton cell key -> cell record).
        self.hash_table = ElasticHashTable(capacity=16384, delta=0.05)
        # Sorted snapshot of cell keys, refreshed once per build for
        # deterministic cache-friendly iteration (the hash remains the
        # authoritative membership/index structure).
        self.cell_keys: List[int] = []

    def _get_box_center(self, level: int, ix: int, iy: int, bounds: Tuple[float, float, float, float]) -> complex:
        xmin, xmax, ymin, ymax = bounds
        sx = (xmax - xmin) / (1 << level)
        sy = (ymax - ymin) / (1 << level)
        cx = xmin + (ix + 0.5) * sx
        cy = ymin + (iy + 0.5) * sy
        return complex(cx, cy)

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        xmin, xmax = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
        ymin, ymax = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
        margin = max(1e-4, 0.02 * max(xmax - xmin, ymax - ymin, 1e-3))
        bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)

        # ---------------------------------------------------------------------
        # 1. TREE-FREE MULTI-LEVEL FUNNEL HASH PARTITIONING
        # ---------------------------------------------------------------------
        # Rebuild the funnel hash as the authoritative cell index.
        self.hash_table = ElasticHashTable(capacity=max(16384, N * 4), delta=0.05)
        ht = self.hash_table
        cell_keys_log: List[int] = []  # insertion-order log of created keys

        # Initial binning at base level
        b_res = 1 << self.base_depth
        scale_x = b_res / (bounds[1] - bounds[0])
        scale_y = b_res / (bounds[3] - bounds[2])

        ix_base = np.clip(((positions[:, 0] - bounds[0]) * scale_x).astype(np.int32), 0, b_res - 1)
        iy_base = np.clip(((positions[:, 1] - bounds[2]) * scale_y).astype(np.int32), 0, b_res - 1)

        def make_cell(lvl: int, gx: int, gy: int) -> Dict:
            # Create-or-fetch a cell record through the funnel hash index.
            key = morton_encode_box(lvl, gx, gy)
            cell = ht.get(key)
            if cell is None:
                cell = {
                    'key': key, 'level': lvl, 'ix': gx, 'iy': gy,
                    'center': self._get_box_center(lvl, gx, gy, bounds),
                    'indices': [], 'is_leaf': True,
                    'm': np.zeros(self.p + 1, dtype=np.complex128),
                    'l': np.zeros(self.p + 1, dtype=np.complex128),
                    'list1': [], 'list2': [], 'list3': [], 'list4': []
                }
                ok, _ = ht.insert(key, cell)
                if not ok:
                    raise RuntimeError(f"funnel hash rejected cell key {key}")
                cell_keys_log.append(key)
            return cell

        # Populate base cells
        base_leaves = set()
        for i in range(N):
            cell = make_cell(self.base_depth, int(ix_base[i]), int(iy_base[i]))
            cell['indices'].append(i)
            base_leaves.add(cell['key'])

        # Adaptive subdivision via elastic hash splitting
        active_leaves = set(base_leaves)
        for lvl in range(self.base_depth, self.max_depth):
            current_leaves = [c for c in (ht.get(k) for k in active_leaves) if c['level'] == lvl]
            for leaf in current_leaves:
                if len(leaf['indices']) > self.max_leaf_particles:
                    leaf['is_leaf'] = False
                    active_leaves.remove(leaf['key'])

                    # Split into 4 child keys
                    c_res = 1 << (lvl + 1)
                    c_scale_x = c_res / (bounds[1] - bounds[0])
                    c_scale_y = c_res / (bounds[3] - bounds[2])

                    for p_idx in leaf['indices']:
                        c_ix = int(np.clip((positions[p_idx, 0] - bounds[0]) * c_scale_x, 0, c_res - 1))
                        c_iy = int(np.clip((positions[p_idx, 1] - bounds[2]) * c_scale_y, 0, c_res - 1))
                        child_cell = make_cell(lvl + 1, c_ix, c_iy)
                        child_cell['indices'].append(p_idx)
                        active_leaves.add(child_cell['key'])

        # Create ancestor keys up to root
        for key in list(cell_keys_log):
            lvl, gx, gy = decode_morton_box(key)
            while lvl > 0:
                lvl -= 1
                gx >>= 1
                gy >>= 1
                p_key = morton_encode_box(lvl, gx, gy)
                if p_key not in ht:
                    p_cell = make_cell(lvl, gx, gy)
                    p_cell['is_leaf'] = False
                else:
                    ht.get(p_key)['is_leaf'] = False

        # One-time cache-friendly iteration snapshot (hash stays authoritative).
        self.cell_keys = sorted(cell_keys_log)

        # ---------------------------------------------------------------------
        # 2. TREE-FREE 4-LIST RESOLUTION VIA HASH LOOKUPS
        # ---------------------------------------------------------------------
        # Pre-group active cells by level
        level_cells: Dict[int, List[Dict]] = {}
        for key in self.cell_keys:
            cell = ht.get(key)
            level_cells.setdefault(cell['level'], []).append(cell)

        # Build List 2 (colleagues of parent's children that are separated)
        for lvl in range(1, self.max_depth + 1):
            grid_p = 1 << (lvl - 1)
            grid_c = 1 << lvl
            for cell in level_cells.get(lvl, []):
                px, py = cell['ix'] >> 1, cell['iy'] >> 1
                for pdx in (-1, 0, 1):
                    for pdy in (-1, 0, 1):
                        spx, spy = px + pdx, py + pdy
                        if 0 <= spx < grid_p and 0 <= spy < grid_p:
                            p_coll_key = morton_encode_box(lvl - 1, spx, spy)
                            if p_coll_key in ht:
                                for cdx in (0, 1):
                                    for cdy in (0, 1):
                                        sx = (spx << 1) + cdx
                                        sy = (spy << 1) + cdy
                                        if 0 <= sx < grid_c and 0 <= sy < grid_c:
                                            if abs(sx - cell['ix']) > 1 or abs(sy - cell['iy']) > 1:
                                                c_key = morton_encode_box(lvl, sx, sy)
                                                if c_key in ht:
                                                    cell['list2'].append(c_key)

        # Build List 1, List 3, and List 4
        leaf_cells = []
        for key in self.cell_keys:
            cell = ht.get(key)
            if cell['is_leaf'] and len(cell['indices']) > 0:
                leaf_cells.append(cell)
        
        def are_boxes_adjacent(b1_lvl, b1_x, b1_y, b2_lvl, b2_x, b2_y) -> bool:
            s1_x = (bounds[1] - bounds[0]) / (1 << b1_lvl)
            s1_y = (bounds[3] - bounds[2]) / (1 << b1_lvl)
            s2_x = (bounds[1] - bounds[0]) / (1 << b2_lvl)
            s2_y = (bounds[3] - bounds[2]) / (1 << b2_lvl)
            
            x1_min, x1_max = bounds[0] + b1_x * s1_x, bounds[0] + (b1_x + 1) * s1_x
            y1_min, y1_max = bounds[2] + b1_y * s1_y, bounds[2] + (b1_y + 1) * s1_y
            
            x2_min, x2_max = bounds[0] + b2_x * s2_x, bounds[0] + (b2_x + 1) * s2_x
            y2_min, y2_max = bounds[2] + b2_y * s2_y, bounds[2] + (b2_y + 1) * s2_y
            
            return not (x1_max < x2_min - 1e-12 or x2_max < x1_min - 1e-12 or
                        y1_max < y2_min - 1e-12 or y2_max < y1_min - 1e-12)

        for leaf in leaf_cells:
            lvl, lx, ly = leaf['level'], leaf['ix'], leaf['iy']

            # Find all List 1 adjacent leaves by traversing active hash cells from root
            def find_adjacent_leaves_hash(c_key: int):
                c_cell = ht.get(c_key)
                if not are_boxes_adjacent(lvl, lx, ly, c_cell['level'], c_cell['ix'], c_cell['iy']):
                    return
                if c_cell['is_leaf']:
                    leaf['list1'].append(c_key)
                else:
                    c_lvl, cx, cy = c_cell['level'], c_cell['ix'], c_cell['iy']
                    for cdx in (0, 1):
                        for cdy in (0, 1):
                            ch_k = morton_encode_box(c_lvl + 1, (cx << 1) + cdx, (cy << 1) + cdy)
                            if ch_k in ht:
                                find_adjacent_leaves_hash(ch_k)

            find_adjacent_leaves_hash(morton_encode_box(0, 0, 0))

            # Find List 3 & List 4: descendants of colleagues of leaf that are separated but parent touches
            grid_lvl = 1 << lvl
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < grid_lvl and 0 <= ny < grid_lvl:
                        coll_key = morton_encode_box(lvl, nx, ny)
                        if coll_key in ht:
                            coll = ht.get(coll_key)
                            if not coll['is_leaf']:
                                def trace_list3_descendants(curr_key: int):
                                    c_cell = ht.get(curr_key)
                                    if are_boxes_adjacent(lvl, lx, ly, c_cell['level'], c_cell['ix'], c_cell['iy']):
                                        if not c_cell['is_leaf']:
                                            c_lvl, cx, cy = c_cell['level'], c_cell['ix'], c_cell['iy']
                                            for cdx in (0, 1):
                                                for cdy in (0, 1):
                                                    ch_k = morton_encode_box(c_lvl + 1, (cx << 1) + cdx, (cy << 1) + cdy)
                                                    if ch_k in ht:
                                                        trace_list3_descendants(ch_k)
                                    else:
                                        # Separated from leaf, but parent was adjacent
                                        leaf['list3'].append(curr_key)
                                        c_cell['list4'].append(leaf['key'])

                                for cdx in (0, 1):
                                    for cdy in (0, 1):
                                        ch_k = morton_encode_box(lvl + 1, (nx << 1) + cdx, (ny << 1) + cdy)
                                        if ch_k in ht:
                                            trace_list3_descendants(ch_k)

        # ---------------------------------------------------------------------
        # 3. TREE-FREE UPWARD PASS (P2M -> M2M via Hash Lookup)
        # ---------------------------------------------------------------------
        for leaf in leaf_cells:
            pts = positions[leaf['indices']]
            q = charges[leaf['indices']]
            leaf['m'] = p2m(pts, q, leaf['center'], self.p)

        for lvl in range(self.max_depth - 1, -1, -1):
            for p_cell in level_cells.get(lvl, []):
                if not p_cell['is_leaf']:
                    p_cell['m'].fill(0.0)
                    for cdx in (0, 1):
                        for cdy in (0, 1):
                            ch_k = morton_encode_box(lvl + 1, (p_cell['ix'] << 1) + cdx, (p_cell['iy'] << 1) + cdy)
                            ch_cell = ht.get(ch_k)
                            if ch_cell is not None:
                                p_cell['m'] += m2m(ch_cell['m'], ch_cell['center'], p_cell['center'], self.p)

        # ---------------------------------------------------------------------
        # 4. TREE-FREE DOWNWARD PASS (L2L + List 2 M2L + List 4 P2L)
        # ---------------------------------------------------------------------
        for lvl in range(1, self.max_depth + 1):
            for cell in level_cells.get(lvl, []):
                # Shift parent local expansion via L2L
                p_key = morton_encode_box(lvl - 1, cell['ix'] >> 1, cell['iy'] >> 1)
                p_cell = ht.get(p_key)
                if p_cell is not None:
                    cell['l'] += l2l(p_cell['l'], p_cell['center'], cell['center'], self.p)

                # List 2 M2L
                for src_k in cell['list2']:
                    src_cell = ht.get(src_k)
                    cell['l'] += m2l(src_cell['m'], src_cell['center'], cell['center'], self.p)

                # List 4 P2L
                for c_k in cell['list4']:
                    c_cell = ht.get(c_k)
                    if len(c_cell['indices']) > 0:
                        c_pts = positions[c_cell['indices']]
                        c_q = charges[c_cell['indices']]
                        cell['l'] += p2l(c_pts, c_q, cell['center'], self.p)

        # ---------------------------------------------------------------------
        # 5. TREE-FREE EVALUATION PASS (L2P + List 3 M2P + List 1 P2P)
        # ---------------------------------------------------------------------
        potentials = np.zeros(N, dtype=np.float64)
        forces_x = np.zeros(N, dtype=np.float64)
        forces_y = np.zeros(N, dtype=np.float64)

        for leaf in leaf_cells:
            t_indices = leaf['indices']
            if not t_indices:
                continue

            # L2P far field
            for idx in t_indices:
                pt_complex = complex(positions[idx, 0], positions[idx, 1])
                potentials[idx] += l2p(leaf['l'], pt_complex, leaf['center'], self.p)
                if compute_forces:
                    fx, fy = l2p_force(leaf['l'], pt_complex, leaf['center'], self.p)
                    forces_x[idx] += fx
                    forces_y[idx] += fy

            # List 3 M2P
            for d_k in leaf['list3']:
                d_cell = ht.get(d_k)
                for idx in t_indices:
                    pt_complex = complex(positions[idx, 0], positions[idx, 1])
                    pot_d, deriv_d = m2p(d_cell['m'], d_cell['center'], pt_complex, self.p)
                    potentials[idx] += pot_d
                    if compute_forces:
                        forces_x[idx] += -deriv_d.real
                        forces_y[idx] += deriv_d.imag

            # List 1 P2P
            tgt_pts = positions[t_indices]
            for n_k in leaf['list1']:
                n_cell = ht.get(n_k)
                if n_cell['indices']:
                    src_pts = positions[n_cell['indices']]
                    src_q = charges[n_cell['indices']]
                    is_self = (n_k == leaf['key'])
                    p_near, fx_near, fy_near = p2p_potential_and_force(
                        tgt_pts, src_pts, src_q,
                        softening=self.softening,
                        exclude_self=is_self
                    )
                    potentials[t_indices] += p_near
                    if compute_forces:
                        forces_x[t_indices] += fx_near
                        forces_y[t_indices] += fy_near

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials


# =============================================================================
# 6. EXACT DIRECT O(N^2) GROUND-TRUTH EVALUATOR
# =============================================================================

def exact_direct_nbody_2d(
    positions: np.ndarray,
    charges: np.ndarray,
    softening: float = 0.0
) -> np.ndarray:
    """Exact O(N^2) direct summation of 2D logarithmic potential: phi_i = sum_{j!=i} q_j * ln(|r_i - r_j|)."""
    N = len(positions)
    pot = np.zeros(N, dtype=np.float64)
    eps2 = softening * softening
    for i in range(N):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        r2 = dx * dx + dy * dy + eps2
        r2[i] = 1.0  # avoid log(0) for self
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        pot[i] = np.sum(charges[mask] * 0.5 * np.log(r2[mask]))
    return pot


def exact_direct_nbody_forces_2d(
    positions: np.ndarray,
    charges: np.ndarray,
    softening: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact O(N^2) direct summation of 2D forces: F_i = - sum_{j!=i} q_j * (r_i - r_j) / |r_i - r_j|^2."""
    N = len(positions)
    fx = np.zeros(N, dtype=np.float64)
    fy = np.zeros(N, dtype=np.float64)
    eps2 = softening * softening
    for i in range(N):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        r2 = dx * dx + dy * dy + eps2
        r2[i] = 1.0
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        inv_r2 = 1.0 / r2[mask]
        fx[i] = -np.sum(charges[mask] * dx[mask] * inv_r2)
        fy[i] = -np.sum(charges[mask] * dy[mask] * inv_r2)
    return fx, fy
