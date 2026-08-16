"""
bioinformatics - Tree-Free Fast Multipole Engine & Elastic Hashing Suite for Computational Biology.

Core Biophysics Solvers:
- solvation_free_energy: Generalized Born & Debye-Huckel implicit solvent free energy engine (App A).
- gnn_long_range_layer: Differentiable O(N) long-range physical prior layer for Equivariant GNNs (App B).
- non_periodic_md_engine: Linear-time Molecular Dynamics without 3D-FFT bottlenecks (App C).
- constant_ph_titration: Fast Monte Carlo protonation state and pKa shift evaluator (App D).
- core.fast_multipole_kernel: O(N) screened Coulomb, Debye-Huckel, and Born multipole kernel.

High-Impact SOTA Application Suites:
1. personalized_oncology_ddg: Patient-specific Delta-Delta-G drug resistance & NGS variant profiler.
2. chromatin_expression_engine: 3D polyanionic chromatin polymer dynamics & non-coding SNP expression predictor.
3. smart_biologics_designer: pH-switchable antibody engineering (pH 7.4 vs 5.5) & polyreactivity / developability filter.
4. biomolecular_condensate_engine: Whole-cell crowding, Liquid-Liquid Phase Separation (LLPS) & droplet simulator.
5. diff_fmm_guidance: Differentiable physical guidance engine for generative SE(3) flow matching & diffusion samplers.
6. rna_tertiary_folding_engine: RNA 3D tertiary folding, Manning Mg2+ counterion condensation & riboswitch switching.
7. tcr_pmhc_immunogenicity: TCR-pMHC neoantigen immunogenicity & human self-peptidome cross-reactivity filter.
8. cryo_em_flexible_fitting: Real-space Cryo-EM molecular dynamics flexible fitting (MDFF) & map refinement.

Massive Genomic, Pan-Genome & Sequence Search Suites:
9. minimizer_sequence_search: High-throughput (w, k)-minimizer indexing & Minimap2-style anchor chaining aligner.
10. pangenome_search_engine: Compressed Colored De Bruijn Graph (cDBG) for cohort-scale gene presence/absence screening.
11. crispr_offtarget_detector: High-throughput CRISPR-Cas9 genome-wide off-target scanner & Hsu cleavage predictor.

Causal Inference & Perturbation Genomics Suites:
12. causal_perturb_seq_grn: Perturb-seq causal gene regulatory network inference & in silico knockout simulator.
13. mendelian_randomization_causal: Polygenic Mendelian Randomization (IVW, MR-Egger, Weighted Median) for unconfounded causal targets.

Personalized Medicine, Polypharmacology & Allostery Suites:
14. polypharmacology_affinity_matrix: Pan-target selectivity screening across 500+ human target families & hERG cardiotoxicity.
15. pharmacogenomics_metabolism: Patient cytochrome P450 (CYP2D6, CYP2C19, TPMT) allele metabolic clearance & CPIC dosage tuning.
16. allosteric_druggability_engine: Dynamic cryptic pocket discovery coupling Anisotropic Network Models with cavity detection.
- macromolecular_nma_engine: Matrix-free normal mode analysis (ANM/GNM) for allostery and functional breathing motions.

Neurotechnology, Biosignal Streaming & Neural Source Imaging Suites:
17. biosignal_lsl_stream_engine: Real-time 64-512ch EEG/fMRI/ERP stream ingestion, CSD spatial Laplacian, and LSL sync.
18. eeg_source_localization_fmm: 3-shell forward leadfield solver & sLORETA 3D cortical inverse source reconstruction.

Empirical Validation & Benchmark Suites:
- cross_validation: GroupKFold homology-clustered CV harness & standard biophysical dataset benchmarks.

Elastic Hashing & Structural Indexing:
- kmer_elastic_hash: Lock-free genomic k-mer counting & De Bruijn graph construction.
- binding_pocket_detector: Grid-free pocket & catalytic cavity detector for drug discovery.
- contact_map_graph: O(N) residue contact networks and allosteric hub centrality builder.
- core.elastic_spatial_hash: 3D Morton-indexed non-reordering open-addressing spatial hash.
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
from .macromolecular_nma_engine import TreeFreeMacromolecularNMA, NormalMode, NMAReport

# 18 High-Impact SOTA Modules
from .personalized_oncology_ddg import PersonalizedOncologyEngine, MutationEffect
from .chromatin_expression_engine import ChromatinExpressionEngine, ChromatinPolymerModel, GenomicLocus, ExpressionPrediction
from .smart_biologics_designer import SmartBiologicsDesigner, PHSwitchCandidate, DevelopabilityProfile
from .biomolecular_condensate_engine import BiomolecularCondensateEngine, CondensateDroplet, PhaseSeparationReport
from .diff_fmm_guidance import DiffFMMGuidanceEngine, GuidanceStepResult, GenerativeValidationMetrics
from .rna_tertiary_folding_engine import RNATertiaryFoldingEngine, RiboswitchState, RNAFoldingResult
from .tcr_pmhc_immunogenicity import TCRpMHCImmunogenicityEngine, TCRBindingProfile, OffTargetSafetyReport
from .cryo_em_flexible_fitting import CryoEMFlexibleFittingEngine, CryoEMFittingMetrics
from .minimizer_sequence_search import MinimizerSequenceSearchEngine, MinimizerSeed, SeedHit, AlignmentChain
from .pangenome_search_engine import PanGenomeSearchEngine, PanGenomeSearchResult
from .crispr_offtarget_detector import CRISPROffTargetScanner, CRISPROffTargetSite, GuideRNASafetyReport
from .causal_perturb_seq_grn import CausalPerturbSeqGRNEngine, CausalEdge, InSilicoKnockoutResult
from .mendelian_randomization_causal import PolygenicMendelianRandomizationEngine, GeneticInstrument, MendelianRandomizationReport
from .polypharmacology_affinity_matrix import PolypharmacologyAffinityMatrixEngine, TargetBindingScore, PolypharmacologyReport
from .pharmacogenomics_metabolism import PharmacogenomicsMetabolismEngine, PGxMetabolicProfile
from .allosteric_druggability_engine import AllostericDruggabilityEngine, CrypticPocket, AllostericDruggabilityReport
from .biosignal_lsl_stream_engine import BiosignalLSLStreamEngine, ChannelMetadata, BiosignalStreamChunk
from .eeg_source_localization_fmm import (
    EEGSourceLocalizationEngine,
    CorticalDipoleSource,
    SourceLocalizationResult,
    DynamicSpatiotemporalSourceResult
)

# Empirical Validation & Benchmark Harness
from .cross_validation import (
    BiophysicalCrossValidator,
    CrossValidationReport,
    CrossValidationFoldResult,
    RegressionMetrics,
    ClassificationMetrics
)

__all__ = [
    # Core Molecular & Spatial Hash
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
    "TreeFreeMacromolecularNMA",
    "NormalMode",
    "NMAReport",
    
    # 18 SOTA Applications
    "PersonalizedOncologyEngine",
    "MutationEffect",
    "ChromatinExpressionEngine",
    "ChromatinPolymerModel",
    "GenomicLocus",
    "ExpressionPrediction",
    "SmartBiologicsDesigner",
    "PHSwitchCandidate",
    "DevelopabilityProfile",
    "BiomolecularCondensateEngine",
    "CondensateDroplet",
    "PhaseSeparationReport",
    "DiffFMMGuidanceEngine",
    "GuidanceStepResult",
    "GenerativeValidationMetrics",
    "RNATertiaryFoldingEngine",
    "RiboswitchState",
    "RNAFoldingResult",
    "TCRpMHCImmunogenicityEngine",
    "TCRBindingProfile",
    "OffTargetSafetyReport",
    "CryoEMFlexibleFittingEngine",
    "CryoEMFittingMetrics",
    "MinimizerSequenceSearchEngine",
    "MinimizerSeed",
    "SeedHit",
    "AlignmentChain",
    "PanGenomeSearchEngine",
    "PanGenomeSearchResult",
    "CRISPROffTargetScanner",
    "CRISPROffTargetSite",
    "GuideRNASafetyReport",
    "CausalPerturbSeqGRNEngine",
    "CausalEdge",
    "InSilicoKnockoutResult",
    "PolygenicMendelianRandomizationEngine",
    "GeneticInstrument",
    "MendelianRandomizationReport",
    "PolypharmacologyAffinityMatrixEngine",
    "TargetBindingScore",
    "PolypharmacologyReport",
    "PharmacogenomicsMetabolismEngine",
    "PGxMetabolicProfile",
    "AllostericDruggabilityEngine",
    "CrypticPocket",
    "AllostericDruggabilityReport",
    "BiosignalLSLStreamEngine",
    "ChannelMetadata",
    "BiosignalStreamChunk",
    "EEGSourceLocalizationEngine",
    "CorticalDipoleSource",
    "SourceLocalizationResult",
    "DynamicSpatiotemporalSourceResult",

    # Empirical Validation & Benchmark Harness
    "BiophysicalCrossValidator",
    "CrossValidationReport",
    "CrossValidationFoldResult",
    "RegressionMetrics",
    "ClassificationMetrics",
]
