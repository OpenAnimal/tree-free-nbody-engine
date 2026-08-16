"""
PPF Contact Solver FMM Module
=============================
Matrix-free Incremental Potential Contact (IPC) & shell elasticity solver,
inspired by ZOZO ppf-contact-solver with Fast Multipole Method spatial acceleration.
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
