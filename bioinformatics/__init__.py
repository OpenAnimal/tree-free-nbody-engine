"""
bioinformatics - Tree-Free Fast Multipole Engine & Elastic Hashing Suite for Computational Biology.

Modules & Capabilities:
1. Biophysics & FMM Potential Solvers:
   - solvation_free_energy: Generalized Born & Debye-Hückel implicit solvent free energy engine (App A).
   - gnn_long_range_layer: Differentiable O(N) long-range physical prior layer for Equivariant GNNs (App B).
   - non_periodic_md_engine: Linear-time Molecular Dynamics without 3D-FFT bottlenecks (App C).
   - constant_ph_titration: Fast Monte Carlo protonation state and pKa shift evaluator (App D).
   - core.fast_multipole_kernel: O(N) screened Coulomb, Debye-Hückel, and Born multipole kernel.

2. Pure Elastic Hashing & Structural Indexing:
   - kmer_elastic_hash: Lock-free genomic k-mer counting & De Bruijn graph construction.
   - binding_pocket_detector: Grid-free pocket & catalytic cavity detector for drug discovery.
   - contact_map_graph: O(N) residue contact networks and allosteric hub centrality builder.
   - core.elastic_spatial_hash: 3D Morton-indexed non-reordering open-addressing spatial hash.

3. Molecular Structures:
   - pdb_loader: PDB/mmCIF parsing and synthetic protein / viral capsid builders.
"""

from .pdb_loader import MolecularSystem, generate_synthetic_protein, generate_viral_capsid, parse_pdb
from .core.elastic_spatial_hash import ElasticSpatialHash3D
from .core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
from .solvation_free_energy import SolvationFreeEnergyEngine
from .gnn_long_range_layer import FMMLongRangeGNNLayer
from .non_periodic_md_engine import MacromolecularMDEngine
from .constant_ph_titration import ConstantPHTitrationEngine
from .kmer_elastic_hash import KmerElasticHashTable
from .binding_pocket_detector import BindingPocketDetector
from .contact_map_graph import ContactMapGraphBuilder

__all__ = [
    "MolecularSystem",
    "generate_synthetic_protein",
    "generate_viral_capsid",
    "parse_pdb",
    "ElasticSpatialHash3D",
    "TreeFreeBioFMM",
    "ScreenedKernelType",
    "SolvationFreeEnergyEngine",
    "FMMLongRangeGNNLayer",
    "MacromolecularMDEngine",
    "ConstantPHTitrationEngine",
    "KmerElasticHashTable",
    "BindingPocketDetector",
    "ContactMapGraphBuilder",
]
