"""
PPF Contact Solver FMM Module
=============================
Matrix-free Incremental Potential Contact (IPC) & shell elasticity solver,
inspired by ZOZO ppf-contact-solver.

Honesty note: despite the historical folder name, there is NO Fast Multipole
Method here. Broadphase uses Morton-binned spatial hashing (sort-based in
matrix_free_ipc.py, elastic-hash in tetrahedral_surgical_soft_robotics.py,
which is a broadphase-only scaffold). A true FMM would target the far-field
of the barrier kernel — an unexplored idea, not an implemented one.
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
