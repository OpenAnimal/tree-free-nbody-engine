"""
PPF Contact Solver FMM Module
=============================
Matrix-free Incremental Potential Contact (IPC) & shell elasticity solver,
inspired by ZOZO ppf-contact-solver.

Honesty note: despite the historical folder name, there is NO Fast Multipole
Method here. The production broadphase in matrix_free_ipc.py is a vectorized
canonical-half-offset scheme that produces the same ring-1 candidate set as
``CellIndex`` from ``core/spatial_index.py`` (kept as the reference
implementation there); tetrahedral_surgical_soft_robotics.py is a
broadphase-only scaffold that does use CellIndex directly. A true FMM would target the far-field of the
barrier kernel — an unexplored idea, not an implemented one.
"""

from .matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes
)

__all__ = [
    "MatrixFreeIPCSolver",
    "ClothMesh",
    "create_cloth_grid",
    "combine_cloth_meshes"
]
