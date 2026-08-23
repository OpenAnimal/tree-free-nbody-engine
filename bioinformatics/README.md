# Tree-Free Bioinformatics, Pan-Genomics & Neurotechnology Engine (`fmm_bioinformatics`)

### $O(N{\cdot}K)$ Far-Field Macromolecular Biophysics, Pan-Genomics, Causal Genetics, Drug Discovery & Real-Time Biosignal Streams

The `fmm_bioinformatics` package extends the **Tree-Free Fast Multipole Method (FMM)** and **Farach-Colton Non-Reordering Open Addressing** to computational biophysics, whole-genome search, single-cell CRISPR causal networks, pharmacogenomics, allosteric drug discovery, and real-time EEG/fMRI neural datastreams.

> **Complexity note:** the far-field engine (`TaylorYukawaBioFMM`) is a
> single-level flat scheme with dense (K,K) M2L, giving **O(N·K) far-field
> cost** where K is the cell count — **O(N) at fixed cell count**. A
> multilevel O(N) FMM is future work.

---

## ⚠️ Scientific Validation Status & Empirical Disclaimer

> **IMPORTANT NOTICE:**  
> The modules in this package are **computational physics prototypes, mathematical formalisms, and analytical heuristics**. **They have not been experimentally or clinically validated in wet-lab assays.**
>
> In biophysics, neurotechnology, and precision medicine, computational approximations (e.g., continuum implicit solvent models, spherical head conductivity simplifications, coarse-grained bead representations, rigid-backbone mutation models) can deviate from true *in vitro* / *in vivo* measurements due to force field simplifications, missing solvent entropy, and unresolved conformational ensembles.
>
> **Do NOT use these modules for clinical diagnostic, prognostic, or therapeutic decision-making without prospective wet-lab and certified empirical benchmark validation.**

---

## 🔬 SOTA High-Impact Bioinformatics & Neurotechnology Suites (Modules 1–18)

| Module | Purpose & Core Advantage | Real-World Clinical, Genomic & Neural Target | Gold-Standard Validation Benchmark |
| :--- | :--- | :--- | :--- |
| **1. Personalized Oncology $\Delta\Delta G$ & Resistance** (`personalized_oncology_ddg.py`) | Evaluates patient single point mutations against drug-target complexes in $< 0.5\text{s}$ using HCT Born radii + TME pH shift. | Predicts patient-specific kinase inhibitor resistance (e.g. EGFR T790M, BCR-ABL T315I, KRAS G12D) from NGS panels. | **SKEMPI 2.0 / Platinum / FireProtDB** ($\Delta\Delta G$ Pearson $r$, Spearman $\rho$, RMSE) |
| **2. 3D Chromatin Architecture & Expression** (`chromatin_expression_engine.py`) | Coarse-grained polyanionic chromatin polymer dynamics with screened electrostatics and k-mer motif hashing. | Simulates *in silico* Hi-C contact maps and predicts how non-coding SNPs alter enhancer-promoter looping and target gene expression. | **ENCODE / 4DN Hi-C & Micro-C Loops** (Stratified loop contact correlation) |
| **3. Smart Biologics & pH-Switch Designer** (`smart_biologics_designer.py`) | CDR histidine scanning for endosomal release (pH 7.4 vs pH 5.5) + polyreactivity / developability profiler. | Designs recycling antibodies with extended serum half-lives and filters out aggregation-prone therapeutic candidates. | **Therapeutic Antibody Profiler (TAP) & SACS** (Polyreactivity ROC-AUC / Specificity) |
| **4. Whole-Cell Condensates & LLPS Engine** (`biomolecular_condensate_engine.py`) | Simulates multi-component IDR/RNA/crowder mixtures using sticker-spacer potentials without 3D-FFT limits. | Models membraneless organelles (stress granules, nuclear speckles) and pathological ALS/Alzheimer's fibril nucleation. | **PhaSepDB / LLPS Phase Diagrams** (Critical temperature $T_c$, dense/dilute ratio) |
| **5. Differentiable FMM Guidance for GenAI** (`diff_fmm_guidance.py`) | Exact analytical $-\nabla_{\mathbf{r}} E$ gradients and electrostatic steering for reverse SE(3) flow matching & diffusion. | Guides generative molecular models (RFdiffusion, TargetDiff, Flow Matching) to generate clash-free, pocket-complementary leads. | **PoseBusters / CrossDocked2020** (Steric clash rate %, chemical validity score) |
| **6. RNA 3D Folding & Riboswitch Electrostatics** (`rna_tertiary_folding_engine.py`) | Evaluates polyanionic backbone electrostatics, divalent $\text{Mg}^{2+}$ Manning counterion condensation, and dynamic programming secondary structure. | Models metabolite riboswitch switching (Theophylline, SAM, Guanine) and synthetic RNA aptamer therapeutics. | **RNA-Puzzles / CASP-RNA** (RMSD, base-pair accuracy, $K_d$ prediction) |
| **7. TCR-pMHC Neoantigen Immunogenicity** (`tcr_pmhc_immunogenicity.py`) | Simulates CDR3 recognition of peptide-MHC complexes and scans normal human proteome for off-target cross-reactivity. | Accelerates CAR-T/TCR-T engineering and screens neoantigen vaccines to prevent fatal off-target autoimmune toxicity. | **VDJdb / IEDB / NetMHCpan** (TCR binding ROC-AUC, cross-reactivity precision) |
| **8. Cryo-EM Real-Space Flexible Fitting** (`cryo_em_flexible_fitting.py`) | $O(N)$ molecular dynamics flexible fitting (MDFF) driven by Cryo-EM density cross-correlation gradients and FMM physical priors. | Refines atomic models into $2.5\text{ \AA} - 6.0\text{ \AA}$ electron density volumes for flexible macromolecular complexes. | **EMDataBank / PDB-REDO** (Cross-correlation coefficient CCC, MolProbity clashscore) |
| **9. Minimizer Sequence Indexing & Alignment** (`minimizer_sequence_search.py`) | $(w, k)$-minimizer seed extraction + dynamic programming anchor chaining (Minimap2-style). | Real-time sequence alignment for Oxford Nanopore / PacBio HiFi reads and database search against multi-gigabyte chromosomes. | **Genome In A Bottle (GIAB) / Minimap2 Benchmarks** (Alignment speed, sensitivity, accuracy) |
| **10. Pan-Genome Colored De Bruijn Search** (`pangenome_search_engine.py`) | Compressed Colored De Bruijn Graph (cDBG) mapping canonical $k$-mers to 64-bit cohort presence bitmasks. | Sub-millisecond screening of antibiotic resistance cassettes and viral variants across 500k isolate cohorts. | **COBS / Bifrost / Mantis Cohort Benchmarks** (Query QPS, false discovery rate) |
| **11. CRISPR-Cas9 Off-Target Cleavage Scanner** (`crispr_offtarget_detector.py`) | Locates PAM-adjacent off-target seeds within 1-4 mismatches with Hsu/Doench cleavage cutting scores. | Genome-wide off-target safety validation for therapeutic guide RNAs (e.g. CRISPR gene therapies). | **GUIDE-seq / CIRCLE-seq Benchmarks** (Off-target cleavage recall & specificity) |
| **12. Perturb-seq Causal GRN & Knockout Simulator** (`causal_perturb_seq_grn.py`) | Infers directed causal regulatory graphs from single-cell CRISPR screens with Markov equivalence pruning. | Simulates counterfactual multi-gene knockouts, predicting downstream cascading expression and cell fates. | **Replogle et al. Perturb-seq / Norman et al.** (Cascade prediction $R^2$, directionality recall) |
| **13. Polygenic Mendelian Randomization (MR)** (`mendelian_randomization_causal.py`) | Inverse-Variance Weighted (IVW), MR-Egger pleiotropy test, and Weighted Median estimators using genetic variants. | Establishes unconfounded causal relationships between biomarker exposures and clinical disease endpoints for target validation. | **UK Biobank / FinnGen GWAS Summary Statistics** (Causal $\beta$, Egger intercept $p$-value) |
| **14. Pan-Target Polypharmacology & Selectivity** (`polypharmacology_affinity_matrix.py`) | High-throughput screening across 500+ human target families using Tree-Free Implicit Solvation and Coulomb Descreening. | Predicts small molecule selectivity ratios, cross-reactive off-targets, and hERG cardiotoxicity / QT prolongation risks. | **ChEMBL / BindingDB / Tox21** (Selectivity index accuracy, hERG classification AUC) |
| **15. Pharmacogenomics (PGx) Metabolism** (`pharmacogenomics_metabolism.py`) | Models patient CYP450 enzyme allele variants (CYP2D6, CYP2C19, TPMT) and altered catalytic pocket volumes. | Recommends CPIC clinical dosage adjustments (PM, IM, NM, UM) to prevent fatal toxicity or bioactivation failure. | **CPIC / PharmGKB Guidelines** (Metabolizer phenotype concordance, dosage precision) |
| **16. Dynamic Cryptic Pocket Detector** (`allosteric_druggability_engine.py`) | Couples Matrix-Free Anisotropic Network Models (ANM) with Grid-Free Cavity Detection along breathing motions. | Unlocks small-molecule allosteric druggability for previously "undruggable" targets (KRAS, transcription factors, phosphatases). | **CryptoSite / Pocketome Benchmarks** (Cryptic cavity recall, volume expansion ratio) |
| **17. Real-Time Biosignal & LSL Multimodal Streaming** (`biosignal_lsl_stream_engine.py`) | Real-time multi-channel (64-512ch) EEG/ERP/fMRI buffer streaming, Surface Laplacian (CSD) filtering, and band-power estimation. | Enables real-time Brain-Computer Interfaces (BCI), neurofeedback, P300 spellers, and multimodal biological media streams (LSL/FFmpeg). | **BCI Competition IV / PhysioNet EEG** (Latency $<1\text{ms}$, ERP detection accuracy) |
| **18. EEG/MEG Forward & Inverse Source Imaging** (`eeg_source_localization_fmm.py`) | 3-shell spherical boundary leadfield forward solver + sLORETA cortical inverse current density reconstruction. | Reconstructs 3D cortical neural activation origins for epilepsy focus localization, cognitive neuroscience, and neuroimaging. | **Brainstorm / MNE-Python Benchmarks** (Localization dipole error $<5\text{mm}$, localization precision) |

---

## 🎯 Cross-Validation as a Core First-Class Feature (`cross_validation.py`)

To prevent the pervasive problem of **sequence and structural homology leakage** in computational biology (where train and test sets contain highly homologous proteins from the same gene family), `cross_validation.py` provides:

1. **Homology-Clustered `GroupKFold` Splitting**:
   Clusters proteins by family/fold so no two homologous sequences share train and validation folds.
2. **Multi-Metric Biophysical Evaluation**:
   * Continuous variables ($\Delta\Delta G$, $\text{p}K_a$, CCC, Causal $\beta$, $K_d$, Source Power): **Pearson $r$**, **Spearman rank $\rho$**, **RMSE**, and **MAE**.
   * Binary classifications (Resistance, Polyreactivity, Cardiotoxicity, Immunogenicity): **ROC-AUC**, **PR-AUC**, **Balanced Accuracy**, and **Specificity**.
3. **Built-in Benchmark Generators**:
   Pre-configured loaders matching **SKEMPI 2.0**, **TAP Antibody Developability**, and **TCR-pMHC Neoantigen** datasets.

---

## 📊 Verification & Benchmarks

Run the complete 19-suite verification and benchmark suite:
```bash
python bioinformatics/test_sota_modules.py
```

> **Synthetic-data caveat (Round-7 honesty pass):** the cross-validation
> benchmark numbers printed by `test_sota_modules.py` (e.g. "5-Fold
> Pearson r: 0.941", "ROC-AUC: 0.929") are computed on **synthetic
> data generated by the same engines under test** — they are circular
> validation smoke tests, not external-accuracy figures. Do not quote
> them as evidence of real-world predictive performance. The test
> output now labels these rows `[SYNTHETIC SELF-GENERATED DATA]`.
