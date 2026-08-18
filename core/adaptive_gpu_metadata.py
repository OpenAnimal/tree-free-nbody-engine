"""
Flat adaptive metadata for the hybrid WebGPU CGR88 backend.

The GPU performs numerical FMM kernels; this module builds the control metadata
that is awkward to construct safely inside a WebGPU dispatch: terminal adaptive
nodes and compact Lists 1--4. It deliberately contains no numerical evaluation.
The resulting arrays are suitable for direct upload to storage buffers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .cgr88_adaptive_fmm import AdaptiveQuadTree


MAX_INTERACTIONS_PER_NODE = 64
INVALID = np.uint32(0xFFFFFFFF)


@dataclass
class FlatAdaptiveMetadata:
    """Upload-ready flat adaptive hierarchy and interaction-list metadata."""

    node_center_size: np.ndarray       # (nodes, 4): cx, cy, width, depth
    node_parent: np.ndarray            # (nodes,)
    node_children: np.ndarray          # (nodes, 4)
    node_particle_range: np.ndarray    # (nodes, 2): offset, count into particle_indices
    node_flags: np.ndarray              # (nodes,): bit 0 terminal, bit 1 active
    particle_indices: np.ndarray       # particles grouped by terminal leaf
    list_offsets: np.ndarray            # (nodes, 4)
    list_counts: np.ndarray             # (nodes, 4)
    list_data: np.ndarray               # (nodes * MAX_INTERACTIONS_PER_NODE,)
    leaf_node_for_particle: np.ndarray
    bounds: Tuple[float, float, float, float]

    @property
    def node_count(self) -> int:
        return int(self.node_center_size.shape[0])

    @property
    def list_capacity(self) -> int:
        return int(self.list_data.size)

    def validate(self) -> None:
        """Validate indices, ranges, and List 3/List 4 reciprocity."""
        n = self.node_count
        if self.node_parent.shape != (n,):
            raise ValueError("node_parent shape mismatch")
        if self.node_children.shape != (n, 4):
            raise ValueError("node_children shape mismatch")
        if self.list_offsets.shape != (n, 4) or self.list_counts.shape != (n, 4):
            raise ValueError("interaction-list metadata shape mismatch")
        if np.any((self.node_parent != INVALID) & (self.node_parent >= n)):
            raise ValueError("invalid parent index")
        if np.any((self.node_children != INVALID) & (self.node_children >= n)): 
            raise ValueError("invalid child index")
        for node in range(n):
            for list_id in range(4):
                start = int(self.list_offsets[node, list_id])
                count = int(self.list_counts[node, list_id])
                if start + count > self.list_data.size:
                    raise ValueError("interaction-list range exceeds list_data")
        if np.any(self.leaf_node_for_particle == INVALID):
            raise ValueError("some particles have no terminal leaf")

    def list_for(self, node: int, list_id: int) -> np.ndarray:
        start = int(self.list_offsets[node, list_id])
        count = int(self.list_counts[node, list_id])
        return self.list_data[start:start + count]


def build_flat_adaptive_metadata(
    positions: np.ndarray,
    charges: np.ndarray | None = None,
    *,
    max_leaf_particles: int = 20,
    max_depth: int = 6,
    max_interactions_per_node: int = MAX_INTERACTIONS_PER_NODE,
) -> FlatAdaptiveMetadata:
    """Build upload-ready flat metadata from the validated CGR88 adaptive tree."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if charges is None:
        charges = np.ones(len(positions), dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    if charges.shape != (len(positions),):
        raise ValueError("charges must have shape (N,)")

    xmin, xmax = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
    ymin, ymax = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
    margin = max(1e-4, 0.02 * max(xmax - xmin, ymax - ymin, 1e-3))
    bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)

    tree = AdaptiveQuadTree(
        positions,
        charges,
        max_leaf_particles=max_leaf_particles,
        max_depth=max_depth,
        p=1,
        domain_bounds=bounds,
    )
    ids = sorted(tree.boxes)
    remap = {old: new for new, old in enumerate(ids)}
    n = len(ids)

    center_size = np.zeros((n, 4), dtype=np.float32)
    parent = np.full(n, INVALID, dtype=np.uint32)
    children = np.full((n, 4), INVALID, dtype=np.uint32)
    flags = np.zeros(n, dtype=np.uint32)
    particle_ranges = np.zeros((n, 2), dtype=np.uint32)
    leaf_for_particle = np.full(len(positions), INVALID, dtype=np.uint32)
    flat_particles = []

    for old_id in ids:
        new_id = remap[old_id]
        box = tree.boxes[old_id]
        center_size[new_id] = (box.center.real, box.center.imag, box.x_max - box.x_min, box.level)
        if box.parent_id is not None:
            parent[new_id] = remap[box.parent_id]
        flags[new_id] = np.uint32((1 if box.is_leaf else 0) | 2)
        for slot, child_id in enumerate(box.children_ids[:4]):
            children[new_id, slot] = remap[child_id]
        if box.is_leaf:
            start = len(flat_particles)
            flat_particles.extend(box.particle_indices)
            particle_ranges[new_id] = (start, len(box.particle_indices))
            for particle_id in box.particle_indices:
                leaf_for_particle[particle_id] = new_id

    list_offsets = np.zeros((n, 4), dtype=np.uint32)
    list_counts = np.zeros((n, 4), dtype=np.uint32)
    list_data = np.full(n * 4 * max_interactions_per_node, INVALID, dtype=np.uint32)

    for old_id in ids:
        new_id = remap[old_id]
        box = tree.boxes[old_id]
        lists = (box.list1, box.list2, box.list3, box.list4)
        for list_id, old_entries in enumerate(lists):
            entries = [remap[x] for x in old_entries]
            if len(entries) > max_interactions_per_node:
                raise ValueError(
                    f"node {new_id} List {list_id + 1} has {len(entries)} entries; "
                    f"increase max_interactions_per_node"
                )
            base = (new_id * 4 + list_id) * max_interactions_per_node
            list_offsets[new_id, list_id] = base
            list_counts[new_id, list_id] = len(entries)
            list_data[base:base + len(entries)] = entries

    metadata = FlatAdaptiveMetadata(
        node_center_size=center_size,
        node_parent=parent,
        node_children=children,
        node_particle_range=particle_ranges,
        node_flags=flags,
        particle_indices=np.asarray(flat_particles, dtype=np.uint32),
        list_offsets=list_offsets,
        list_counts=list_counts,
        list_data=list_data,
        leaf_node_for_particle=leaf_for_particle,
        bounds=bounds,
    )
    metadata.validate()
    return metadata
