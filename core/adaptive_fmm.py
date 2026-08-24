"""
Carrier, Greengard, & Rokhlin (1988) 2D Adaptive Fast Multipole Method (CGR88)
and Greengard & Rokhlin (1987) Regular Fast Multipole Method.

THE CANONICAL ADAPTIVE ENGINE in this module is ``AdaptiveFMM`` (alias
``FastAdaptiveFMM``): the level-batched, 2:1-balanced, vectorized CGR88
engine (section 6 below). The classical per-box Python-loop implementations
(``ClassicalAdaptiveFMM`` section 3, ``TreeFreeElasticAdaptiveFMM``
section 5) are retained ONLY as slow cross-validation references; they agree
with the canonical engine to truncation level and are what the
cross-validation tests check the fast engine against.

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
3. Canonical AdaptiveFMM / FastAdaptiveFMM: level-batched vectorized 2:1-balanced
   CGR88 engine (potential + vector forces), ~25-35x faster than the classical loops.
4. ClassicalAdaptiveFMM: per-box loop reference implementation (slow validator).
5. Regular (Uniform) FMM for fixed-depth quadtrees (GreengardRokhlin87RegularFMM).
6. Tree-Free Hash-Indexed FMM: adaptive FMM operators on a non-reordering funnel-hash
   cell index (core.elastic_hash.ElasticHashTable, Farach-Colton, Krapivin, & Kuszmaul, 2025).
7. Exact O(N^2) direct Coulomb / logarithmic N-body ground-truth evaluator for potentials and forces.
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
# 3. CLASSICAL PER-BOX CGR88 ENGINE (SLOW CROSS-VALIDATION REFERENCE)
# =============================================================================

class ClassicalAdaptiveFMM:
    """
    SLOW REFERENCE VALIDATOR: exact per-box Python-loop implementation of the
    Carrier, Greengard, & Rokhlin (1988) Adaptive Fast Multipole Method in 2D.

    This is NOT the engine to use for workloads -- it is the pedagogical
    reference kept for cross-validation (it predates the canonical
    vectorized engine and computes the identical CGR88 sum one box at a
    time; agreement tests live in tests/core/test_adaptive_fmm_fast.py).
    For actual evaluations use the canonical ``AdaptiveFMM`` (= alias
    ``FastAdaptiveFMM``) in section 6 of this module, which computes the
    same four-list scheme level-batched and is ~25-35x faster.
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
    route through the hash. ClassicalAdaptiveFMM above is the classical
    dict/tree reference implementation; the two engines agree numerically
    to <1e-12 relative (tests/core/test_adaptive_fmm_cross_validation.py).
    Both are slow cross-validation references for the canonical
    level-batched AdaptiveFMM (= FastAdaptiveFMM) in section 6.
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
# 6. CANONICAL ENGINE: LEVEL-BATCHED VECTORIZED ADAPTIVE FMM (CGR88)
# =============================================================================
#
# High-throughput canonical sibling of ``TreeFreeElasticAdaptiveFMM`` /
# ``ClassicalAdaptiveFMM`` (the classical engines stay above as the
# pedagogical / cross-validation references). Same mathematics -- Carrier,
# Greengard, & Rokhlin (1988) adaptive multipole expansions with the exact
# four interaction lists -- but every pass is batched per tree level into
# dense ``(n_boxes, p+1)`` complex arrays, following the standard
# optimization lineage of production FMM codes:
#
# - **2:1 level-balanced quadtree** so adjacent leaves never differ by more
#   than one level; this bounds all four interaction lists to constant size
#   (Sundar, Sampath, & Biros, 2008; Ying, Biros, & Zorin, 2004).
# - **Interaction lists built from the bounded colleague ring**, never by root
#   recursion (Carrier, Greengard, & Rokhlin, 1988, Section 3; Yokota, 2012).
#   List-2 sources live on the FMMLIB2D-style ``(-3..3)^2 \\ 3x3`` stencil
#   with a parity-dependent colleague-ring test; List-3/4 separated children
#   sit at per-axis offsets ``{-2, +3}`` relative to the target leaf box.
# - **M2L operators precomputed per (level, relative offset)** as dense
#   ``(p+1, p+1)`` matrices, so each offset class collapses to one BLAS
#   matmul (Gimbutas & Greengard, 2012, FMMLIB2D ``itable(-3:3,-3:3)``;
#   exafmm-t ``M2L_setup``).
# - **Vectorized List-4 P2L**: the per-particle P2L of CGR88 is kept
#   deliberately (M2L from the coarse leaf's multipole would converge at
#   ratio (sqrt(2)+sqrt(2)/2)/2.5 ~ 0.85 per term and lose an order of
#   accuracy); it is batched into ragged (particle, cell) rows with sorted
#   ``np.add.reduceat`` segment sums instead of Python loops.
# - **CSR near-field P2P** with per-leaf concatenated source blocks (Lashuk et
#   al., 2012: sorted/concatenated particle arrays).
#
# Cell index: as everywhere in this repo, the authoritative cell index is the
# funnel hash (Farach-Colton, Krapivin, & Kuszmaul, 2025,
# ``core.elastic_hash.ElasticHashTable``) mapping each level-tagged Morton cell
# key to its dense cell id. The hot passes work on per-level dense occupancy
# grids -- the "implicit lattice" -- which are the vectorizable O(1) equivalent
# of hash membership probes (the same hybrid ``FastVectorizedFMM`` documents).


def _m2m_batch(a: np.ndarray, delta: np.ndarray, p: int) -> np.ndarray:
    """M2M (CGR88 Theorem 2.2) for n pairs at once.

    a: (n, p+1) source multipoles; delta: (n,) complex = src - dst centers.
        b_0 = a_0
        b_l = -a_0 delta^l / l + sum_{k=1..l} binom(l-1, k-1) a_k delta^(l-k)
    """
    n = len(delta)
    dpow = np.empty((n, p + 1), dtype=np.complex128)
    dpow[:, 0] = 1.0
    for l in range(1, p + 1):
        dpow[:, l] = dpow[:, l - 1] * delta
    b = np.empty((n, p + 1), dtype=np.complex128)
    b[:, 0] = a[:, 0]
    for l in range(1, p + 1):
        term = -a[:, 0] * dpow[:, l] / l
        for k in range(1, l + 1):
            term = term + math.comb(l - 1, k - 1) * a[:, k] * dpow[:, l - k]
        b[:, l] = term
    return b


def _l2l_batch(c: np.ndarray, delta: np.ndarray, p: int) -> np.ndarray:
    """L2L (CGR88 Theorem 2.4) for n pairs at once.

    c: (n, p+1) source locals; delta: (n,) = dst - src centers.
        d_l = sum_{k=l..p} binom(k, l) c_k delta^(k-l)
    """
    n = len(delta)
    dpow = np.empty((n, p + 1), dtype=np.complex128)
    dpow[:, 0] = 1.0
    for l in range(1, p + 1):
        dpow[:, l] = dpow[:, l - 1] * delta
    d = np.empty((n, p + 1), dtype=np.complex128)
    for l in range(p + 1):
        term = c[:, l].copy()
        for k in range(l + 1, p + 1):
            term = term + math.comb(k, l) * c[:, k] * dpow[:, k - l]
        d[:, l] = term
    return d


_M2L_TABLES: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _m2l_tables(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-p cached tables for the M2L matrix: (signs, binomials, k+l power
    indices), all (p+1, p+1) with row index l, column index k."""
    if p not in _M2L_TABLES:
        l_idx, k_idx = np.mgrid[0:p + 1, 0:p + 1]
        signs = np.ones((p + 1, p + 1), dtype=np.float64)
        signs[1:, :] = (-1.0) ** l_idx[1:, :]         # (-1)^l for l >= 1
        signs[1:, 0] = (-1.0) ** (l_idx[1:, 0] - 1)   # (-1)^(l-1) in column 0
        binoms = np.ones((p + 1, p + 1), dtype=np.float64)
        for l in range(1, p + 1):
            for k in range(1, p + 1):
                binoms[l, k] = math.comb(k + l - 1, l)
        pow_idx = (l_idx + k_idx).astype(np.int64)
        _M2L_TABLES[p] = (signs, binoms, pow_idx)
    return _M2L_TABLES[p]


def _m2l_matrix(delta: complex, p: int) -> np.ndarray:
    """Dense (p+1, p+1) M2L matrix for fixed separation delta = dst - src,
    valid for |delta| > r_src + r_dst. local = multipole @ M.T:
        c_0 = a_0 ln(delta) + sum_{k>=1} a_k delta^{-k}
        c_l = a_0 (-1)^{l-1}/(l delta^l)
              + sum_{k>=1} (-1)^l binom(k+l-1, l) a_k delta^{-(k+l)}
    """
    signs, binoms, pow_idx = _m2l_tables(p)
    dp = (1.0 / complex(delta)) ** np.arange(2 * p + 1)
    M = signs * binoms * dp[pow_idx]
    M[1:, 0] /= np.arange(1, p + 1)   # the 1/l factor in c_l, l >= 1
    M[0, 0] = np.log(complex(delta))
    return M


class AdaptiveFMM:
    """CANONICAL ENGINE: vectorized Carrier, Greengard, & Rokhlin (1988)
    adaptive FMM on a 2:1-balanced, funnel-hash-indexed quadtree.

    Every pass is batched per tree level into dense (n_boxes, p+1) complex
    arrays; M2L operators are precomputed per (level, relative offset) as
    dense matrices in the style of FMMLIB2D's ``itable`` (Gimbutas &
    Greengard, 2012); the per-particle CGR88 List-4 P2L is kept deliberately
    and batched via ``np.add.reduceat`` segment sums; the near field is a
    CSR-concatenated per-leaf P2P (full provenance in the section-6 comment
    block above). Parameters mirror ``TreeFreeElasticAdaptiveFMM`` so the
    fast and classical engines are drop-in comparable in the benchmark
    table; ``FastAdaptiveFMM`` is a backward-compatible alias for this
    class (its historical name before it became the canonical engine).

    Cross-validation: agrees with exact direct O(N^2) summation and with
    the slow classical references (ClassicalAdaptiveFMM /
    TreeFreeElasticAdaptiveFMM) at truncation level -- see
    tests/core/test_adaptive_fmm_fast.py and
    tests/core/test_adaptive_fmm_reference.py.
    """

    def __init__(
        self,
        max_leaf_particles: int = 24,
        base_depth: int = 2,
        max_depth: int = 9,
        p: int = 10,
        softening: float = 0.0,
    ):
        if max_depth > 12:
            raise ValueError(
                "max_depth > 12 allocates dense per-level occupancy grids "
                f"((2^max_depth)^2 int64 each); got {max_depth}")
        self.max_leaf_particles = max_leaf_particles
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.p = p
        self.softening = softening
        # Funnel hash: authoritative cell index (level-tagged Morton key -> id).
        self.hash_table: Optional[ElasticHashTable] = None
        self.cell_keys: List[int] = []

    # ------------------------------------------------------------------ build

    def _grow(self, k: int) -> int:
        """Extend cell storage by k slots; returns first new id."""
        old = self.n_cells
        need = old + k
        if need > self._cap:
            new_cap = max(self._cap * 2, need)
            for name, dt in (("lvl", np.int64), ("cx", np.int64),
                             ("cy", np.int64), ("par", np.int64),
                             ("cnt", np.int64), ("cen", np.complex128),
                             ("leaf", np.bool_), ("chb", np.int64)):
                arr = getattr(self, "_" + name)
                grown = np.empty(new_cap, dtype=dt)
                grown[:old] = arr[:old]
                setattr(self, "_" + name, grown)
            self._cap = new_cap
        self.n_cells = need
        return old

    def _add_level_grid(self, lvl: int) -> np.ndarray:
        if lvl not in self._occ:
            self._occ[lvl] = np.full((1 << lvl, 1 << lvl), -1, dtype=np.int64)
        return self._occ[lvl]

    def _split_cells(self, parents: np.ndarray, positions: np.ndarray) -> None:
        """Split leaf cells `parents` (same level) into 4 children each and
        rebin their particles one level deeper. Fully vectorized."""
        lvl = int(self._lvl[parents[0]])
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0
        npar = len(parents)
        base = self._grow(4 * npar)
        pix = self._cx[parents]
        piy = self._cy[parents]

        q = np.arange(4)
        qx = (q & 1)[None, :] + 2 * pix[:, None]      # (npar, 4)
        qy = (q >> 1)[None, :] + 2 * piy[:, None]
        ids = base + 4 * np.arange(npar)[:, None] + q[None, :]

        self._lvl[ids.ravel()] = lvl + 1
        self._cx[ids.ravel()] = qx.ravel()
        self._cy[ids.ravel()] = qy.ravel()
        self._par[ids.ravel()] = np.repeat(parents, 4)
        self._leaf[ids.ravel()] = True
        self._cnt[ids.ravel()] = 0
        self._chb[ids.ravel()] = -1
        hx = Wx / (1 << (lvl + 1))
        hy = Wy / (1 << (lvl + 1))
        self._cen[ids.ravel()] = (bx0 + (qx.ravel() + 0.5) * hx) + \
            1j * (by0 + (qy.ravel() + 0.5) * hy)
        self._leaf[parents] = False
        self._chb[parents] = base + 4 * np.arange(npar)

        grid = self._add_level_grid(lvl + 1)
        grid[qx.ravel(), qy.ravel()] = ids.ravel()

        # rebin particles of the split parents
        member = np.isin(self.pcell, parents)
        if member.any():
            sub = np.nonzero(member)[0]
            sc = self.pcell[sub]
            six = self._cx[sc]
            siy = self._cy[sc]
            cix = np.clip(((positions[sub, 0] - bx0) / Wx * (1 << (lvl + 1)))
                          .astype(np.int64), 2 * six, 2 * six + 1)
            ciy = np.clip(((positions[sub, 1] - by0) / Wy * (1 << (lvl + 1)))
                          .astype(np.int64), 2 * siy, 2 * siy + 1)
            cid = self._chb[sc] + (cix - 2 * six) + 2 * (ciy - 2 * siy)
            self.pcell[sub] = cid
            np.add.at(self._cnt, cid, 1)
            self._cnt[sc] = 0

    def _build(self, positions: np.ndarray, N: int) -> None:
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0

        cap = 1024
        self._cap = cap
        self.n_cells = 0
        self._lvl = np.zeros(cap, dtype=np.int64)
        self._cx = np.zeros(cap, dtype=np.int64)
        self._cy = np.zeros(cap, dtype=np.int64)
        self._par = np.full(cap, -1, dtype=np.int64)
        self._cnt = np.zeros(cap, dtype=np.int64)
        self._cen = np.zeros(cap, dtype=np.complex128)
        self._leaf = np.zeros(cap, dtype=bool)
        self._chb = np.full(cap, -1, dtype=np.int64)
        self._occ: Dict[int, np.ndarray] = {}

        # base-level binning
        b = self.base_depth
        grid = self._add_level_grid(b)
        fx = (positions[:, 0] - bx0) / Wx * (1 << b)
        fy = (positions[:, 1] - by0) / Wy * (1 << b)
        ix_b = np.clip(fx.astype(np.int64), 0, (1 << b) - 1)
        iy_b = np.clip(fy.astype(np.int64), 0, (1 << b) - 1)
        keys = ix_b * (1 << b) + iy_b
        uniq, inv = np.unique(keys, return_inverse=True)
        n0 = len(uniq)
        base = self._grow(n0)
        uix = uniq // (1 << b)
        uiy = uniq % (1 << b)
        ids = base + np.arange(n0)
        self._lvl[ids] = b
        self._cx[ids] = uix
        self._cy[ids] = uiy
        self._par[ids] = -1
        self._leaf[ids] = True
        self._chb[ids] = -1
        hbx = Wx / (1 << b)
        hby = Wy / (1 << b)
        self._cen[ids] = (bx0 + (uix + 0.5) * hbx) + 1j * (by0 + (uiy + 0.5) * hby)
        grid[uix, uiy] = ids
        self.pcell = ids[inv]
        self._cnt[ids] = np.bincount(inv, minlength=n0)

        # occupancy-driven splitting (level by level until quiescent)
        while True:
            did = False
            for lvl in range(self.base_depth, self.max_depth):
                g = self._occ.get(lvl)
                if g is None:
                    continue
                ids = g[g >= 0]
                over = ids[self._leaf[ids] & (self._cnt[ids] > self.max_leaf_particles)]
                if len(over):
                    self._split_cells(over, positions)
                    did = True
            if not did:
                break

        # 2:1 balance is REQUIRED, not optional: the bounded interaction
        # list construction (colleague rings, one-level descent) is only
        # complete for level-balanced trees.
        self._balance(positions)

        self._build_hash_index()

    def _balance(self, positions: np.ndarray) -> None:
        """Enforce 2:1 balance: split any leaf >= 2 levels coarser than an
        adjacent leaf, via 3x3 grid dilations projected onto coarser levels."""
        for _ in range(2 * self.max_depth + 4):
            to_split: List[np.ndarray] = []
            for lvl in range(self.max_depth, self.base_depth, -1):
                grid = self._occ.get(lvl)
                if grid is None:
                    continue
                leaf_cells = grid[grid >= 0]
                leaf_cells = leaf_cells[self._leaf[leaf_cells]]
                if len(leaf_cells) == 0:
                    continue
                mask = np.zeros_like(grid, dtype=bool)
                mask[self._cx[leaf_cells], self._cy[leaf_cells]] = True
                # Chebyshev 3x3 dilation = cross dilation applied twice
                dil = mask.copy()
                for _ in range(2):
                    nxt = dil.copy()
                    for ax in (0, 1):
                        for off in (1, -1):
                            a = [slice(None)] * 2
                            b_ = [slice(None)] * 2
                            if off == 1:
                                a[ax] = slice(off, None)
                                b_[ax] = slice(0, -off)
                            else:
                                a[ax] = slice(0, off)
                                b_[ax] = slice(-off, None)
                            shifted = np.zeros_like(dil)
                            shifted[tuple(a)] = dil[tuple(b_)]
                            nxt |= shifted
                    dil = nxt

                cur = dil
                cur_lvl = lvl
                for lc in range(lvl - 2, self.base_depth - 1, -1):
                    h = 1 << (cur_lvl - lc)
                    coarse = cur.reshape(1 << lc, h, 1 << lc, h).any(axis=(1, 3))
                    g = self._occ.get(lc)
                    if g is not None:
                        cc = g[g >= 0]
                        if len(cc):
                            marked = coarse[self._cx[cc], self._cy[cc]]
                            bad = cc[marked & self._leaf[cc]]
                            if len(bad):
                                to_split.append(bad)
                    cur = coarse
                    cur_lvl = lc
            if not to_split:
                return
            # A cell can be marked from several level projections in one
            # round; deduplicate before splitting (splitting a cell twice
            # would orphan its first children).
            allbad = np.unique(np.concatenate(to_split))
            for lvl in np.unique(self._lvl[allbad]):
                self._split_cells(allbad[self._lvl[allbad] == lvl], positions)

    def _build_hash_index(self) -> None:
        n = self.n_cells
        self.hash_table = ElasticHashTable(capacity=max(16384, 4 * n), delta=0.05)
        self.cell_keys = []
        for c in range(n):
            key = morton_encode_box(int(self._lvl[c]), int(self._cx[c]),
                                    int(self._cy[c]))
            ok, _ = self.hash_table.insert(key, int(c))
            if ok:
                self.cell_keys.append(key)
        self._list3_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ utils

    def _lookup(self, lvl: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Occupancy lookup at `lvl`; -1 for out-of-bounds/unoccupied."""
        g = self._occ.get(lvl)
        n = 1 << lvl
        ix = np.asarray(ix, dtype=np.int64)
        iy = np.asarray(iy, dtype=np.int64)
        valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        out = np.full(len(ix), -1, dtype=np.int64)
        if g is not None and valid.any():
            out[valid] = g[ix[valid], iy[valid]]
        return out

    def _children_matrix(self) -> np.ndarray:
        """(n_cells, 4) child ids (-1 for unsplit)."""
        n = self.n_cells
        ch = np.full((n, 4), -1, dtype=np.int64)
        has = np.nonzero(self._chb[:n] >= 0)[0]
        if len(has):
            ch[has] = self._chb[has][:, None] + np.arange(4)[None, :]
        return ch

    # ------------------------------------------------------------------ main

    def evaluate(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        compute_forces: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty, empty) if compute_forces else empty

        p = self.p
        xmin, xmax = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
        ymin, ymax = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
        margin = max(1e-4, 0.02 * max(xmax - xmin, ymax - ymin, 1e-3))
        self.bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)
        bx0, bx1, by0, by1 = self.bounds
        Wx, Wy = bx1 - bx0, by1 - by0

        self._build(positions, N)
        n_cells = self.n_cells
        lvl = self._lvl[:n_cells]
        cix_all = self._cx[:n_cells]
        ciy_all = self._cy[:n_cells]
        leaf = self._leaf[:n_cells]
        centers = self._cen[:n_cells]
        children = self._children_matrix()
        max_lvl = int(lvl.max())

        # leaf CSR over particles
        leaves = np.nonzero(leaf)[0]
        leaf_slot = np.full(n_cells, -1, dtype=np.int64)
        leaf_slot[leaves] = np.arange(len(leaves))
        leaf_of = self.pcell
        counts = np.bincount(leaf_slot[leaf_of], minlength=len(leaves))
        cell_start = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(counts, out=cell_start[1:])
        cell_particles = np.argsort(leaf_slot[leaf_of], kind="stable")

        m = np.zeros((n_cells, p + 1), dtype=np.complex128)
        lcl = np.zeros((n_cells, p + 1), dtype=np.complex128)

        # ---- P2M ------------------------------------------------------------
        z = positions[:, 0] + 1j * positions[:, 1]
        m[:, 0] = np.bincount(leaf_of, weights=charges, minlength=n_cells)
        dz_leaf = z - centers[leaf_of]
        dz_pow = np.ones(N, dtype=np.complex128)
        for k in range(1, p + 1):
            dz_pow *= dz_leaf
            w = charges * dz_pow / (-k)
            m[:, k] = (np.bincount(leaf_of, weights=w.real, minlength=n_cells)
                       + 1j * np.bincount(leaf_of, weights=w.imag,
                                          minlength=n_cells))

        # ---- M2M upward -----------------------------------------------------
        for l in range(max_lvl - 1, -1, -1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            parents = cells[~leaf[cells]]
            if len(parents) == 0:
                continue
            ch = children[parents]                    # (n_par, 4)
            a = m[ch]                                 # (n_par, 4, p+1)
            d = centers[ch] - centers[parents][:, None]
            b = _m2m_batch(a.reshape(-1, p + 1), d.reshape(-1), p)
            m[parents] = b.reshape(len(parents), 4, p + 1).sum(axis=1)

        # ---- downward: L2L + List-2 M2L + List-3/4 ---------------------------
        m2l_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        list3_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        for l in range(1, max_lvl + 1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            if len(cells) == 0:
                continue
            cix = cix_all[cells]
            ciy = ciy_all[cells]
            cpar = self._par[cells]

            # L2L from parent (parent local finalized at level l-1). Base
            # cells have no parent cell (the ancestors above base_depth do
            # not exist as cells; their far field enters via List-2 M2L at
            # the base level, the flat-scheme geometry).
            cpar = self._par[cells]
            has_par = cpar >= 0
            if has_par.any():
                hc = cells[has_par]
                hp = cpar[has_par]
                lcl[hc] += _l2l_batch(lcl[hp], centers[hc] - centers[hp], p)

            # List 2: children of the parent's colleagues that are not
            # adjacent. Ring test is parity-dependent, so it is applied per
            # cell rather than baked into the offset enumeration.
            ppx = cix >> 1
            ppy = ciy >> 1
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        continue
                    sx = cix + dx
                    sy = ciy + dy
                    src = self._lookup(l, sx, sy)
                    in_ring = (src >= 0) & \
                        (np.abs((sx >> 1) - ppx) <= 1) & \
                        (np.abs((sy >> 1) - ppy) <= 1)
                    if in_ring.any():
                        s = src[in_ring]
                        key = (l, dx, dy)
                        M = m2l_cache.get(key)
                        if M is None:
                            # delta = dst_center - src_center; the source sits
                            # at (+dx, +dy) relative to the target.
                            M = _m2l_matrix(
                                complex(-dx * Wx / (1 << l),
                                        -dy * Wy / (1 << l)), p)
                            m2l_cache[key] = M
                        lcl[cells[in_ring]] += m[s] @ M.T

            # List 3 (M2P at evaluation) + List 4 (P2L into the separated
            # child's local from the coarse leaf's particles). Child offsets
            # are expressed in level-(l+1) units relative to 2*cix (the
            # target box origin at l+1): adjacent children occupy
            # [-1, 2]^2; separated ones have at least one axis in {-2, +3}.
            # The colleague of the target is at cix + (vx2 >> 1)
            # (exact: 2*cix is even). List 4 uses true per-particle P2L --
            # M2L from the coarse leaf's multipole would converge at ratio
            # (sqrt(2)+sqrt(2)/2)/2.5 ~ 0.85 and lose ~an order of accuracy.
            leaf_mask = leaf[cells]
            for vx2 in range(-2, 4):
                for vy2 in range(-2, 4):
                    if -1 <= vx2 <= 2 and -1 <= vy2 <= 2:
                        continue  # adjacent child: belongs to List 1, not 3
                    coll = self._lookup(l, cix + (vx2 >> 1),
                                        ciy + (vy2 >> 1))
                    child = self._lookup(l + 1, 2 * cix + vx2, 2 * ciy + vy2)
                    ok = (coll >= 0) & (~leaf[coll.clip(0)]) & (child >= 0) \
                        & leaf_mask
                    if not ok.any():
                        continue
                    tgt = cells[ok]
                    src3 = child[ok]
                    list3_pairs.append((tgt, src3))
                    # List 4 P2L, vectorized over (leaf particle, dst cell)
                    slot_t = leaf_slot[tgt]
                    n_b = counts[slot_t]
                    T = int(n_b.sum())
                    if T == 0:
                        continue
                    reps = np.repeat(np.arange(len(tgt)), n_b)
                    prev = np.concatenate(([0], np.cumsum(n_b)[:-1]))
                    within = np.arange(T) - np.repeat(prev, n_b)
                    pids = cell_particles[cell_start[slot_t][reps] + within]
                    dcells = src3[reps]
                    dzz = centers[dcells] - z[pids]  # z0_D - z_i
                    w = 1.0 / dzz
                    wpow = np.empty((T, p + 1), dtype=np.complex128)
                    wpow[:, 0] = 1.0
                    for k in range(1, p + 1):
                        wpow[:, k] = wpow[:, k - 1] * w
                    qw = charges[pids][:, None] * wpow  # (T, p+1)
                    order = np.argsort(dcells, kind="stable")
                    d_sorted = dcells[order]
                    starts = np.concatenate(
                        ([0], np.nonzero(d_sorted[1:] != d_sorted[:-1])[0] + 1))
                    dsts = d_sorted[starts]
                    qw_s = qw[order]
                    # c_0 = sum q ln(z0 - z_i); no contribution from the
                    # w^0 column (classical P2L has no constant term).
                    log_term = np.add.reduceat(
                        (charges[pids] * np.log(dzz)).real[order], starts)
                    lcl[dsts, 0] += log_term
                    for kl in range(1, p + 1):
                        coef = ((-1.0) ** (kl - 1)) / kl
                        lcl[dsts, kl] += coef * (
                            np.add.reduceat(qw_s[:, kl].real, starts)
                            + 1j * np.add.reduceat(qw_s[:, kl].imag, starts))

        # ---- List 1 (adjacent leaves) for near-field P2P ----------------------
        l1_tgt_parts: List[np.ndarray] = []
        l1_src_parts: List[np.ndarray] = []
        l1_counts = np.zeros(len(leaves), dtype=np.int64)

        def add_l1(tgt: np.ndarray, src: np.ndarray) -> None:
            ok = counts[leaf_slot[src]] > 0  # drop empty neighbor leaves
            if not ok.any():
                return
            t = tgt[ok]
            s = src[ok]
            l1_tgt_parts.append(leaf_slot[t])
            l1_src_parts.append(s)
            l1_counts[:] += np.bincount(leaf_slot[t], minlength=len(leaves))

        # (a) each leaf's own cell (self pairs masked in the P2P kernel)
        add_l1(leaves, leaves)
        for l in range(0, max_lvl + 1):
            g = self._occ.get(l)
            if g is None:
                continue
            cells = g[g >= 0]
            if len(cells) == 0:
                continue
            cix = cix_all[cells]
            ciy = ciy_all[cells]
            leaf_mask = leaf[cells]

            # (b) same-level leaf colleagues (includes siblings)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    src = self._lookup(l, cix + dx, ciy + dy)
                    ok = (src >= 0) & leaf_mask & leaf[src.clip(0)]
                    if ok.any():
                        add_l1(cells[ok], src[ok])

            # (c) coarser adjacent leaves: 3x3 ring around the parent, with
            # the mixed-level touch test (a parent-colleague cell only
            # touches the child if the child's quadrant faces it).
            if l >= 1:
                px = cix >> 1
                py = ciy >> 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        sx = px + dx
                        sy = py + dy
                        src = self._lookup(l - 1, sx, sy)
                        touch = (2 * sx <= cix + 1) & (2 * sx + 2 >= cix) & \
                                (2 * sy <= ciy + 1) & (2 * sy + 2 >= ciy)
                        ok = (src >= 0) & touch & leaf_mask & leaf[src.clip(0)]
                        if ok.any():
                            add_l1(cells[ok], src[ok])

            # (d) adjacent children (qx,qy in [-1,2]^2) of split same-level
            # colleagues. The (0,0) colleague is the target itself (a leaf,
            # never split), so own-box children cannot leak in.
            for qx in range(-1, 3):
                for qy in range(-1, 3):
                    if 0 <= qx <= 1 and 0 <= qy <= 1:
                        continue  # own parent's quadrant block: tgt is a leaf
                    coll = self._lookup(l, cix + (qx >> 1), ciy + (qy >> 1))
                    child = self._lookup(l + 1, 2 * cix + qx, 2 * ciy + qy)
                    ok = ((coll >= 0) & (~leaf[coll.clip(0)]) & (child >= 0)
                          & leaf_mask & leaf[child.clip(0)])
                    if ok.any():
                        add_l1(cells[ok], child[ok])

        l1_start = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(l1_counts, out=l1_start[1:])
        l1_tgt = (np.concatenate(l1_tgt_parts) if l1_tgt_parts
                  else np.empty(0, dtype=np.int64))
        l1_src = (np.concatenate(l1_src_parts) if l1_src_parts
                  else np.empty(0, dtype=np.int64))
        order = np.argsort(l1_tgt, kind="stable")
        l1_src = l1_src[order]

        # per-leaf concatenated near-field source particles. `l1_src` is
        # sorted by target leaf slot, so the flat layout (leaf-major, then
        # entry order within the leaf) means destination offsets are simply
        # sequential.
        n_src_cells = counts[leaf_slot[l1_src]]
        flat_total = int(n_src_cells.sum())
        flat_sources = np.empty(flat_total, dtype=np.int64)
        if flat_total:
            entry_of_row = np.repeat(np.arange(len(l1_src)), n_src_cells)
            within_entry = np.arange(flat_total) - np.repeat(
                np.concatenate(([0], np.cumsum(n_src_cells)[:-1])), n_src_cells)
            src_pid = cell_particles[cell_start[leaf_slot[l1_src[entry_of_row]]]
                                     + within_entry]
            flat_sources[:] = src_pid  # dest == arange(flat_total)
        src_off = np.zeros(len(leaves) + 1, dtype=np.int64)
        np.cumsum(np.bincount(
            np.repeat(np.arange(len(leaves)), l1_counts.astype(np.int64)),
            weights=n_src_cells, minlength=len(leaves)), out=src_off[1:])

        # ---- evaluation -------------------------------------------------------
        potentials = np.zeros(N, dtype=np.float64)
        forces_x = np.zeros(N, dtype=np.float64)
        forces_y = np.zeros(N, dtype=np.float64)

        # L2P (+ forces), vectorized over all particles
        c_loc = lcl[leaf_of]
        dzp = z - centers[leaf_of]
        val = c_loc[:, 0].copy()
        deriv = np.zeros(N, dtype=np.complex128)
        zp = np.ones(N, dtype=np.complex128)
        for l in range(1, p + 1):
            deriv += l * c_loc[:, l] * zp
            zp *= dzp
            val += c_loc[:, l] * zp
        potentials += val.real
        if compute_forces:
            forces_x += -deriv.real
            forces_y += deriv.imag

        # M2P over List-3 pairs (leaf particles x separated source boxes)
        for tgt, src3 in list3_pairs:
            slot_t = leaf_slot[tgt]
            n_b = counts[slot_t]
            reps = np.repeat(np.arange(len(tgt)), n_b)
            prev = np.concatenate(([0], np.cumsum(n_b)[:-1]))
            within = np.arange(int(n_b.sum())) - np.repeat(prev, n_b)
            pids = cell_particles[cell_start[slot_t][reps] + within]
            m_src = m[src3][reps]
            dzs = z[pids] - centers[src3][reps]
            a0 = m_src[:, 0]
            dinv = 1.0 / dzs
            pot = a0 * np.log(dzs)
            dsum = a0 * dinv.copy()
            dinv_pow = dinv.copy()
            for k in range(1, p + 1):
                pot += m_src[:, k] * dinv_pow
                if compute_forces:
                    dsum -= k * m_src[:, k] * (dinv_pow * dinv)
                if k < p:
                    dinv_pow = dinv_pow * dinv
            np.add.at(potentials, pids, pot.real)
            if compute_forces:
                np.add.at(forces_x, pids, -dsum.real)
                np.add.at(forces_y, pids, dsum.imag)

        # near-field P2P per leaf block
        eps2 = self.softening * self.softening
        for s in range(len(leaves)):
            t_ids = cell_particles[cell_start[s]:cell_start[s + 1]]
            if len(t_ids) == 0:
                continue
            lo, hi = src_off[s], src_off[s + 1]
            if hi <= lo:
                continue
            s_ids = flat_sources[lo:hi]
            xt = positions[t_ids]
            xs = positions[s_ids]
            qs = charges[s_ids]
            ddx = xt[:, 0][:, None] - xs[None, :, 0]
            ddy = xt[:, 1][:, None] - xs[None, :, 1]
            r2 = ddx * ddx + ddy * ddy + eps2
            r2_safe = np.where(r2 < 1e-28, 1.0, r2)
            g = 0.5 * np.log(r2_safe)
            self_mask = t_ids[:, None] == s_ids[None, :]
            g = np.where(self_mask, 0.0, g)
            potentials[t_ids] += g @ qs
            if compute_forces:
                inv = np.where(self_mask, 0.0, 1.0 / r2_safe)
                forces_x[t_ids] -= (ddx * inv) @ qs
                forces_y[t_ids] -= (ddy * inv) @ qs

        if compute_forces:
            return potentials, forces_x, forces_y
        return potentials


#: Backward-compatible alias: the canonical level-batched engine was
#: originally shipped as ``core.adaptive_fmm_fast.FastAdaptiveFMM`` before
#: the two adaptive modules were consolidated; both names now refer to the
#: same canonical class in this module.
FastAdaptiveFMM = AdaptiveFMM


# =============================================================================
# 7. EXACT DIRECT O(N^2) GROUND-TRUTH EVALUATOR
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


__all__ = [
    # canonical engine (+ historical alias)
    "AdaptiveFMM",
    "FastAdaptiveFMM",
    # slow cross-validation reference engines
    "ClassicalAdaptiveFMM",
    "TreeFreeElasticAdaptiveFMM",
    "GreengardRokhlin87RegularFMM",
    # tree data structure
    "AdaptiveQuadTree",
    "QuadBox",
    # expansion operators (CGR88 Theorems 2.1-2.4)
    "p2m", "m2m", "m2l", "l2l", "p2l", "m2p",
    "l2p", "l2p_force", "p2p_potential_and_force",
    # cell keys
    "morton_encode_box", "decode_morton_box",
    # direct ground truth
    "exact_direct_nbody_2d", "exact_direct_nbody_forces_2d",
]
