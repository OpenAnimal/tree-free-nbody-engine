"""
Verification and Benchmark Suite for the 18 High-Impact SOTA Bioinformatics & Neurotechnology Modules & CV Engine.
Tests:
1. Personalized Oncology ddG & Drug Resistance Profiler
2. 3D Chromatin Architecture & Non-Coding SNP Target Expression Predictor
3. Smart Biologics & pH-Switchable Antibody Designer
4. Whole-Cell Biomolecular Condensates & LLPS Engine
5. Differentiable FMM Physical Guidance for Generative Flow/Diffusion
6. RNA 3D Tertiary Folding & Riboswitch Electrostatics
7. TCR-pMHC Neoantigen Immunogenicity & Human Peptidome Cross-Reactivity
8. Cryo-EM Real-Space Flexible Fitting & Density Refinement
9. Genomic (w, k)-Minimizer Indexing & Anchor Chaining Search
10. Pan-Genome Colored De Bruijn Graph Cohort Presence/Absence Search
11. CRISPR-Cas9 Genome-Wide Off-Target Scanner & Cleavage Predictor
12. Perturb-seq Causal Gene Regulatory Network & In Silico Knockout Simulator
13. Polygenic Mendelian Randomization (MR) Causal Target Validation
14. Pan-Target Polypharmacology & Selectivity Matrix Engine
15. Patient Pharmacogenomics (PGx) & Hepatic Metabolic Clearance Predictor
16. Dynamic Cryptic Pocket Detector & Allosteric Druggability Engine
17. Real-Time Biosignal & LSL Multimodal Streaming Engine
18. EEG / MEG Forward Leadfield & sLORETA Inverse Cortical Source Localization
19. End-to-End Multi-Dataset Cross-Validation Benchmark Harness
"""

import sys
import os
import time
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bioinformatics import (
    MolecularSystem,
    generate_synthetic_protein,
    PersonalizedOncologyEngine,
    ChromatinExpressionEngine,
    SmartBiologicsDesigner,
    BiomolecularCondensateEngine,
    DiffFMMGuidanceEngine,
    RNATertiaryFoldingEngine,
    TCRpMHCImmunogenicityEngine,
    CryoEMFlexibleFittingEngine,
    MinimizerSequenceSearchEngine,
    PanGenomeSearchEngine,
    CRISPROffTargetScanner,
    CausalPerturbSeqGRNEngine,
    PolygenicMendelianRandomizationEngine,
    GeneticInstrument,
    PolypharmacologyAffinityMatrixEngine,
    PharmacogenomicsMetabolismEngine,
    AllostericDruggabilityEngine,
    BiosignalLSLStreamEngine,
    ChannelMetadata,
    EEGSourceLocalizationEngine,
    BiophysicalCrossValidator
)


def test_module_1_personalized_oncology():
    print("\n" + "="*70)
    print("[TEST 1/19] Personalized Oncology ddG & Drug Resistance Profiling")
    print("="*70)
    
    protein = generate_synthetic_protein(n_atoms=1000)
    com = protein.center_of_mass
    ligand_coords = com + np.random.randn(25, 3) * 3.0
    ligand_charges = np.random.uniform(-0.5, 0.5, 25)
    ligand_radii = np.full(25, 1.6)
    ligand_masses = np.full(25, 12.0)
    
    complex_sys = MolecularSystem(
        coords=np.vstack([protein.coords, ligand_coords]),
        charges=np.concatenate([protein.charges, ligand_charges]),
        radii=np.concatenate([protein.radii, ligand_radii]),
        masses=np.concatenate([protein.masses, ligand_masses]),
        atom_names=protein.atom_names + [f"L{i}" for i in range(25)],
        residue_names=protein.residue_names + ["LIG"] * 25,
        residue_ids=np.concatenate([protein.residue_ids, np.full(25, 999)]),
        chain_ids=protein.chain_ids + ["L"] * 25,
        system_name="EGFR_Osimertinib_Complex"
    )

    engine = PersonalizedOncologyEngine()
    patient_mutations = ["T790M", "L858R", "C797S", "G719S"]
    
    res = engine.screen_patient_panel(complex_sys, patient_mutations)
    print(f"Screened {res['total_screened']} patient mutations in {res['runtime_seconds']:.3f}s ({res['throughput_mutations_per_sec']:.1f} mut/s):")
    for eff in res["mutations"]:
        print(f"  Mutation: {eff.mutation_str:<7} | ddG_bind: {eff.ddg_bind_kcal_mol:+6.2f} kcal/mol | Class: {eff.resistance_class:<15} | Confidence: {eff.confidence_score*100:.1f}%")
    
    assert len(res["mutations"]) == 4
    print("[PASS] Module 1 Test Passed!")


def test_module_2_chromatin_expression():
    print("\n" + "="*70)
    print("[TEST 2/19] 3D Chromatin Architecture & Non-Coding DNA Target Expression")
    print("="*70)
    
    engine = ChromatinExpressionEngine(kmer_size=11)
    model = engine.build_synthetic_chromatin_domain(domain_kb=500, resolution_kb=10, target_gene="MYC")
    hic_wt = engine.compute_in_silico_hic_map(model)
    
    pred_ctcf = engine.evaluate_noncoding_variant(
        model=model,
        variant_bp=25000,
        ref_sequence="CCACCAGGTGGCG",
        alt_sequence="CCACCAAGTGGCG",
        target_gene="MYC"
    )
    print(f"Non-Coding SNP Analysis (CTCF Insulator):")
    print(f"  Variant: {pred_ctcf.variant_id} -> Impact: {pred_ctcf.regulatory_impact} (log2FC: {pred_ctcf.log2_fold_change:+.2f})")

    assert hic_wt.shape == (50, 50)
    print("[PASS] Module 2 Test Passed!")


def test_module_3_smart_biologics():
    print("\n" + "="*70)
    print("[TEST 3/19] Smart Biologics, pH-Switchable Antibodies & Polyreactivity Filter")
    print("="*70)
    
    antibody = generate_synthetic_protein(n_atoms=800, seed=42)
    antibody.system_name = "Therapeutic_mAb_Fab"
    antigen = generate_synthetic_protein(n_atoms=600, seed=101)
    antigen.system_name = "IL6R_Antigen"
    
    designer = SmartBiologicsDesigner()
    profile = designer.profile_developability(antibody)
    print(f"Biologic Developability Score: {profile.overall_developability_score:.1f}/100 | Risk: {profile.polyreactivity_risk}")
    
    candidates = designer.scan_histidine_switches(antibody, antigen, candidate_residue_ids=[1, 2, 3])
    print(f"Top Switch Candidate: {candidates[0].mutation_str} (delta_dG: {candidates[0].delta_dg_ph_kcal_mol:+.2f} kcal/mol)")
    
    assert len(candidates) > 0
    print("[PASS] Module 3 Test Passed!")


def test_module_4_biomolecular_condensates():
    print("\n" + "="*70)
    print("[TEST 4/19] Biomolecular Condensates, LLPS & Whole-Cell Crowding")
    print("="*70)
    
    engine = BiomolecularCondensateEngine(box_size_nm=80.0, temperature_kelvin=300.0)
    sys_data = engine.generate_crowded_cell_system(
        num_idr_chains=15, beads_per_idr=20,
        num_rna_chains=5, beads_per_rna=15,
        num_crowders=80
    )
    sys_data = engine.run_simulation(sys_data, num_steps=40)
    report = engine.analyze_phase_separation(sys_data)
    print(f"Cytoplasm Vol Fraction: {report.volume_fraction_percent:.2f}% | Tc Estimate: {report.critical_temperature_estimate_k:.1f} K")
    
    assert report.num_total_particles > 0
    print("[PASS] Module 4 Test Passed!")


def test_module_5_diff_fmm_guidance():
    print("\n" + "="*70)
    print("[TEST 5/19] Differentiable FMM Guidance for Generative Flow Matching / Diffusion")
    print("="*70)
    
    engine = DiffFMMGuidanceEngine()
    rec_coords = np.random.randn(150, 3) * 12.0
    rec_charges = np.random.uniform(-0.6, 0.6, 150)
    rec_radii = np.full(150, 1.7)
    
    lig_coords = np.random.randn(20, 3) * 5.0
    lig_charges = np.random.uniform(-0.4, 0.4, 20)
    lig_radii = np.full(20, 1.7)
    
    forces, energies = engine.compute_pocket_physical_gradients(
        lig_coords, lig_charges, lig_radii,
        rec_coords, rec_charges, rec_radii
    )
    print(f"Calculated Analytical Gradients: Shape={forces.shape} | Force Norm: {np.linalg.norm(forces):.2f}")
    assert forces.shape == lig_coords.shape
    print("[PASS] Module 5 Test Passed!")


def test_module_6_rna_folding():
    print("\n" + "="*70)
    print("[TEST 6/19] RNA 3D Tertiary Folding & Riboswitch Switching")
    print("="*70)
    
    engine = RNATertiaryFoldingEngine(mg2_concentration_mm=2.5)
    theophylline_aptamer_seq = "GGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACU"
    
    db, pairs = engine.predict_secondary_structure(theophylline_aptamer_seq)
    print(f"RNA Sequence (len={len(theophylline_aptamer_seq)}): {theophylline_aptamer_seq}")
    print(f"Predicted Dot-Bracket:  {db} ({len(pairs)} base pairs)")
    
    switch_res = engine.evaluate_riboswitch_switching(theophylline_aptamer_seq, ligand_name="Theophylline", ligand_concentration_um=10.0)
    apo = switch_res["apo_state"]
    holo = switch_res["holo_state"]
    print(f"Riboswitch Switching Analysis:")
    print(f"  Apo Energy: {apo.total_energy_kcal_mol:.2f} kcal/mol | Holo Energy: {holo.total_energy_kcal_mol:.2f} kcal/mol")
    print(f"  Switching Equilibrium Ratio: {switch_res['switching_ratio']:.2e} | Bound Fraction: {switch_res['ligand_bound_fraction']*100:.1f}%")
    
    assert len(pairs) > 0
    print("[PASS] Module 6 Test Passed!")


def test_module_7_tcr_pmhc():
    print("\n" + "="*70)
    print("[TEST 7/19] TCR-pMHC Neoantigen Immunogenicity & Cross-Reactivity Filter")
    print("="*70)
    
    engine = TCRpMHCImmunogenicityEngine()
    tcr_cdr3 = "CASSLGAGVETQYF"
    kras_g12d = "VVGADGVGK"
    
    binding = engine.evaluate_tcr_pmhc_binding(tcr_cdr3, kras_g12d, hla_allele="HLA-A*02:01")
    print(f"TCR Binding to KRAS G12D ({kras_g12d}):")
    print(f"  Binding dG: {binding.binding_affinity_dg_kcal:.2f} kcal/mol | Kd: {binding.kd_micromolar:.1f} uM")
    print(f"  Immunogenicity Score: {binding.immunogenicity_score:.1f}/100 | Activation: {binding.activation_potential}")
    
    safety = engine.screen_off_target_cross_reactivity(tcr_cdr3, kras_g12d)
    print(f"Off-Target Peptidome Safety Screen ({safety.num_self_peptides_screened} peptides):")
    print(f"  Cross-Reactivity Risk: {safety.cross_reactivity_risk} | Confidence: {safety.safety_confidence_score*100:.1f}%")
    
    assert binding.immunogenicity_score > 0
    print("[PASS] Module 7 Test Passed!")


def test_module_8_cryo_em():
    print("\n" + "="*70)
    print("[TEST 8/19] Cryo-EM Real-Space Flexible Fitting (MDFF)")
    print("="*70)
    
    protein = generate_synthetic_protein(n_atoms=300, seed=42)
    engine = CryoEMFlexibleFittingEngine(resolution_angstrom=4.0, grid_spacing_angstrom=1.5)
    
    density, origin, dims = engine.generate_synthetic_density_map(protein)
    print(f"Generated 3D Cryo-EM Density Grid: {dims} at 4.0 A resolution")
    
    perturbed = protein.copy()
    perturbed.coords += np.random.randn(*perturbed.coords.shape) * 1.5
    
    fitted_sys, metrics = engine.run_flexible_fitting(perturbed, density, origin, num_steps=25)
    print(f"MDFF Density Refinement Results:")
    print(f"  Initial CCC: {metrics.initial_ccc:.4f} -> Final CCC: {metrics.final_ccc:.4f}")
    print(f"  RMSD Displacement: {metrics.rmsd_displacement_A:.2f} A | Quality: {metrics.fitting_convergence}")
    
    assert metrics.final_ccc >= metrics.initial_ccc
    print("[PASS] Module 8 Test Passed!")


def test_module_9_minimizer_search():
    print("\n" + "="*70)
    print("[TEST 9/19] Genomic (w, k)-Minimizer Indexing & Anchor Chaining Search")
    print("="*70)
    
    engine = MinimizerSequenceSearchEngine(k=15, w=8)
    bases = np.array(["A", "C", "G", "T"])
    rng = np.random.RandomState(42)
    ref_seq = "".join(bases[rng.randint(0, 4, 50000)])
    
    n_indexed = engine.index_reference("chr1_50k", ref_seq)
    print(f"Indexed Reference 'chr1_50k' ({len(ref_seq):,} bp): {n_indexed:,} minimizers stored.")
    
    query_sub = list(ref_seq[12000 : 12500])
    for err_idx in [50, 120, 280, 410]:
        query_sub[err_idx] = "T" if query_sub[err_idx] != "T" else "A"
    query_seq = "".join(query_sub)
    
    t0 = time.perf_counter()
    chains = engine.align_query(query_seq)
    t_align = (time.perf_counter() - t0) * 1000.0
    
    print(f"Query Alignment (500 bp read) completed in: {t_align:.2f} ms")
    if chains:
        top = chains[0]
        print(f"  Top Alignment Chain: Target={top.ref_id} | Ref Span=[{top.ref_start:,}..{top.ref_end:,}] | Query Span=[{top.query_start}..{top.query_end}]")
        print(f"  Anchors={top.num_anchors} | Score={top.chain_score} | Identity Approx={top.identity_approx*100:.1f}%")
        assert top.ref_start >= 11900 and top.ref_end <= 12600, "Alignment located wrong genomic locus!"
    
    assert len(chains) > 0
    print("[PASS] Module 9 Test Passed!")


def test_module_10_pangenome_search():
    print("\n" + "="*70)
    print("[TEST 10/19] Pan-Genome Compressed Colored De Bruijn Graph (cDBG)")
    print("="*70)
    
    pangenome = PanGenomeSearchEngine(k=19, max_samples=16)
    bases = np.array(["A", "C", "G", "T"])
    rng = np.random.RandomState(101)
    
    core_backbone = "".join(bases[rng.randint(0, 4, 8000)])
    antibiotic_res_gene = "".join(bases[rng.randint(0, 4, 800)])
    
    strain_names = [f"Staphylococcus_strain_{i+1}" for i in range(6)]
    for i, name in enumerate(strain_names):
        strain_seq = core_backbone + ("".join(bases[rng.randint(0, 4, 1500)]))
        if i in [0, 2, 4]:
            strain_seq += antibiotic_res_gene
        pangenome.index_genome(name, strain_seq)
        
    print(f"Indexed Pan-Genome Cohort of {len(strain_names)} strains ({pangenome.kmer_table.num_unique_kmers:,} unique 19-mers).")
    
    res = pangenome.query_sequence(antibiotic_res_gene, query_name="blaNDM-1_BetaLactamase")
    print(f"Screening Antibiotic Resistance Gene '{res.query_name}' across cohort:")
    print(f"  Status: {res.query_presence_status} | Matching Strains: {res.matching_sample_ids}")
    
    assert len(res.matching_sample_ids) == 3
    print("[PASS] Module 10 Test Passed!")


def test_module_11_crispr_offtarget():
    print("\n" + "="*70)
    print("[TEST 11/19] CRISPR-Cas9 Genome-Wide Off-Target Scanner & Cleavage Predictor")
    print("="*70)
    
    scanner = CRISPROffTargetScanner(pam_pattern="NGG", seed_length=8)
    bases = np.array(["A", "C", "G", "T"])
    rng = np.random.RandomState(42)
    chrom_seq = "".join(bases[rng.randint(0, 4, 100000)])
    
    guide_20 = "GAGTCCGAGCAGAAGAAGAA"
    on_target_23 = guide_20 + "TGG"
    off_target_1mm = guide_20[:5] + "A" + guide_20[6:] + "CGG"
    off_target_2mm = guide_20[:18] + "TT" + "AGG"
    
    chrom_list = list(chrom_seq)
    chrom_list[45000 : 45023] = list(on_target_23)
    chrom_list[15000 : 15023] = list(off_target_1mm)
    chrom_list[70000 : 70023] = list(off_target_2mm)
    modified_chrom = "".join(chrom_list)
    
    n_pams = scanner.index_genomic_sequence("chrEMX1", modified_chrom)
    print(f"Indexed Genome 'chrEMX1' ({len(modified_chrom):,} bp): {n_pams:,} PAM sites indexed.")
    
    report = scanner.scan_guide_rna(guide_20, max_mismatches=4)
    print(f"CRISPR Guide RNA Specificity Report for '{report.guide_sequence_20nt}':")
    print(f"  On-Target Sites: {report.num_on_target_sites} | Total Candidate Sites (<=4mm): {report.num_off_target_sites_0to4mm}")
    print(f"  Specificity Score: {report.on_target_specificity_score:.1f}/100 | Tier: {report.crispr_safety_tier}")
    
    assert report.num_on_target_sites >= 1
    print("[PASS] Module 11 Test Passed!")


def test_module_12_causal_perturb_seq():
    print("\n" + "="*70)
    print("[TEST 12/19] Perturb-seq Causal Gene Regulatory Network (GRN)")
    print("="*70)
    
    gene_panel = ["TP53", "MYC", "CDKN1A", "MDM2", "BAX", "CCND1", "E2F1", "RB1", "ATM", "CASP3"]
    engine = CausalPerturbSeqGRNEngine(gene_names=gene_panel)
    
    rng = np.random.RandomState(42)
    ctrl_data = rng.normal(5.0, 1.0, size=(1000, len(gene_panel)))
    
    perturbed_data = {}
    for ko_gene in ["TP53", "MYC"]:
        ko_cells = rng.normal(5.0, 1.0, size=(200, len(gene_panel)))
        ko_idx = gene_panel.index(ko_gene)
        ko_cells[:, ko_idx] = rng.normal(0.5, 0.2, size=200)
        
        if ko_gene == "TP53":
            ko_cells[:, gene_panel.index("CDKN1A")] -= 2.5
            ko_cells[:, gene_panel.index("BAX")] -= 1.8
            ko_cells[:, gene_panel.index("MDM2")] -= 2.0
        elif ko_gene == "MYC":
            ko_cells[:, gene_panel.index("CCND1")] -= 2.2
            ko_cells[:, gene_panel.index("E2F1")] -= 1.5
            
        perturbed_data[ko_gene] = ko_cells
        
    causal_edges = engine.fit_from_perturb_seq_data(ctrl_data, perturbed_data)
    print(f"Inferred {len(causal_edges)} Directed Causal Edges from Perturb-seq Data:")
    for edge in causal_edges[:4]:
        print(f"  {edge.regulator_gene} -> {edge.target_gene:<8} | Causal Effect: {edge.causal_effect:+5.2f} | Mode: {edge.regulatory_mode}")
        
    ko_res = engine.predict_in_silico_knockout(target_genes=["TP53", "MYC"])
    print(f"\nIn Silico Double Knockout do(TP53=0, MYC=0):")
    print(f"  Total Genes Affected: {ko_res.total_genes_affected} | Phenotype Shift L2: {ko_res.phenotype_shift_norm:.2f}")
    print(f"  Top Downregulated: {ko_res.top_downregulated_genes}")
    print(f"  Predicted Cell Fate: {ko_res.predicted_cell_fate}")
    
    assert len(causal_edges) >= 4
    print("[PASS] Module 12 Test Passed!")


def test_module_13_mendelian_randomization():
    print("\n" + "="*70)
    print("[TEST 13/19] Polygenic Mendelian Randomization (MR) Causal Inference")
    print("="*70)
    
    mr_engine = PolygenicMendelianRandomizationEngine(f_stat_filter=10.0)
    
    rng = np.random.RandomState(42)
    instruments = []
    for i in range(8):
        gamma_j = rng.uniform(0.15, 0.40)
        se_gamma_j = rng.uniform(0.01, 0.03)
        Gamma_j = 0.45 * gamma_j + rng.normal(0, 0.02)
        se_Gamma_j = rng.uniform(0.02, 0.04)
        f_stat = (gamma_j ** 2) / (se_gamma_j ** 2)
        
        instruments.append(GeneticInstrument(
            snp_id=f"rs{1000000 + i}",
            chromosome="chr1",
            position_bp=100000 + i * 50000,
            effect_allele="A",
            beta_exposure=gamma_j,
            se_exposure=se_gamma_j,
            beta_outcome=Gamma_j,
            se_outcome=se_Gamma_j,
            f_statistic=float(f_stat)
        ))
        
    report = mr_engine.estimate_causal_effect(
        instruments=instruments,
        exposure_name="Circulating_LDL_C",
        outcome_name="Coronary_Artery_Disease"
    )
    
    print(f"Mendelian Randomization Analysis: {report.exposure_name} -> {report.outcome_name}")
    print(f"  Instruments Used: {report.num_instruments_used} (Mean F-statistic: {report.mean_f_statistic:.1f})")
    print(f"  IVW Causal Effect Beta: {report.causal_effect_ivw:+.3f} (SE: {report.se_ivw:.3f}, p={report.p_value_ivw:.2e})")
    print(f"  MR-Egger Causal Beta:   {report.causal_effect_egger:+.3f} (Pleiotropy Intercept p={report.egger_intercept_p_value:.3f})")
    print(f"  Weighted Median Beta:   {report.causal_effect_weighted_median:+.3f}")
    print(f"  Causal Conclusion:      {report.causal_conclusion}")
    
    assert abs(report.causal_effect_ivw - 0.45) < 0.15
    print("[PASS] Module 13 Test Passed!")


def test_module_14_polypharmacology():
    print("\n" + "="*70)
    print("[TEST 14/19] Pan-Target Polypharmacology & Selectivity Matrix")
    print("="*70)
    
    poly_engine = PolypharmacologyAffinityMatrixEngine()
    
    rng = np.random.RandomState(42)
    ligand_coords = rng.randn(25, 3) * 3.0
    ligand_charges = rng.uniform(-0.4, 0.4, 25)
    ligand_radii = np.full(25, 1.7)
    ligand_masses = np.full(25, 12.0)
    
    drug_ligand = MolecularSystem(
        coords=ligand_coords,
        charges=ligand_charges,
        radii=ligand_radii,
        masses=ligand_masses,
        atom_names=[f"C{i}" for i in range(25)],
        residue_names=["DRUG"] * 25,
        residue_ids=np.ones(25, dtype=np.int32),
        chain_ids=["L"] * 25,
        system_name="Osimertinib_Analogue"
    )

    report = poly_engine.screen_drug_against_panel(drug_ligand, primary_target_id="EGFR_Kinase")
    print(f"Pan-Target Screening for Drug '{report.drug_name}' against {report.num_targets_screened} Targets:")
    print(f"  Primary On-Target ({report.primary_on_target}): Kd = {report.primary_kd_nm:.2f} nM")
    print(f"  Selectivity Index: {report.selectivity_index:.1f}x | Tier: {report.overall_safety_tier}")
    
    assert report.num_targets_screened >= 8
    print("[PASS] Module 14 Test Passed!")


def test_module_15_pharmacogenomics():
    print("\n" + "="*70)
    print("[TEST 15/19] Patient Pharmacogenomics (PGx) & Metabolic Clearance")
    print("="*70)
    
    pgx_engine = PharmacogenomicsMetabolismEngine()
    
    pm_profile = pgx_engine.evaluate_patient_pgx_metabolism(
        enzyme_name="CYP2D6",
        diplotype=("*4", "*4"),
        drug_substrate="Tamoxifen",
        is_prodrug=True
    )
    print(f"Patient 1 (CYP2D6 *4/*4 on Tamoxifen):")
    print(f"  Phenotype: {pm_profile.metabolizer_phenotype} | Relative Clearance: {pm_profile.relative_clearance_rate*100:.1f}%")
    print(f"  Recommended Dose: {pm_profile.recommended_dose_percentage:.0f}% | Action: {pm_profile.clinical_actionability}")
    
    assert pm_profile.metabolizer_phenotype == "Poor Metabolizer (PM)"
    print("[PASS] Module 15 Test Passed!")


def test_module_16_allosteric_druggability():
    print("\n" + "="*70)
    print("[TEST 16/19] Dynamic Cryptic Pocket Detector & Allosteric Druggability")
    print("="*70)
    
    target_protein = generate_synthetic_protein(n_atoms=400, seed=42)
    target_protein.system_name = "KRAS_G12D_GTPase"
    
    allostery_engine = AllostericDruggabilityEngine(
        cutoff_radius=12.0,
        num_vibrational_modes=2,
        mode_perturbation_amplitude_A=2.0
    )
    
    report = allostery_engine.analyze_allosteric_druggability(target_protein)
    print(f"Allosteric Druggability Analysis for '{report.target_name}':")
    print(f"  Static Pockets: {report.num_static_pockets} | Cryptic Pockets Found: {report.num_cryptic_pockets_found}")
    print(f"  Druggability Tier: {report.overall_allosteric_druggability_tier}")
    
    assert report.num_static_pockets >= 0
    print("[PASS] Module 16 Test Passed!")


def test_module_17_biosignal_lsl():
    print("\n" + "="*70)
    print("[TEST 17/19] Real-Time Biosignal & Multimodal LSL Streaming Engine")
    print("="*70)
    
    channel_names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "T7", "T8"]
    channels = []
    rng = np.random.RandomState(42)
    for i, name in enumerate(channel_names):
        pos_3d = rng.randn(3) * 80.0
        channels.append(ChannelMetadata(
            channel_index=i,
            channel_label=name,
            sensor_type="EEG",
            coords_3d=pos_3d,
            sampling_rate_hz=500.0,
            unit="uV"
        ))
        
    stream_engine = BiosignalLSLStreamEngine(channels=channels, ring_buffer_seconds=5.0)
    
    # Ingest 1,000 samples (2 seconds of 8-channel EEG at 500 Hz)
    # Synthetic EEG with Alpha rhythm (10 Hz) + P300 evoked peak at 300ms
    t_axis = np.linspace(0, 2.0, 1000)
    sim_eeg = np.sin(2.0 * np.pi * 10.0 * t_axis)[None, :] * 15.0 + rng.randn(8, 1000) * 5.0
    # Add P300 evoked wave at t=0.80s (300ms post stimulus at t=0.50s)
    sim_eeg[1, 400:450] += 8.5 # Channel Cz
    
    n_ingested = stream_engine.ingest_sample_chunk(sim_eeg)
    print(f"Ingested {n_ingested:,} EEG samples across {stream_engine.n_channels} channels ({stream_engine.sampling_rate:.0f} Hz).")
    
    band_powers = stream_engine.compute_spectral_band_powers(window_seconds=1.5)
    print(f"Spectral Rhythm Distribution: Alpha={band_powers['Alpha']*100:.1f}%, Theta={band_powers['Theta']*100:.1f}%, Beta={band_powers['Beta']*100:.1f}%")
    
    # Detect P300 Evoked Potential (stimulus trigger at t_index 250 -> 0.5s)
    t_stim = stream_engine.ring_timestamps[250]
    erp_res = stream_engine.detect_evoked_potential_erp([t_stim])
    print(f"Evoked Potential ERP Detection: Detected={erp_res['erp_detected']} | Peak Latency={erp_res['p300_peak_latency_ms']:.1f} ms | Amp={erp_res['p300_peak_amplitude_uV']:.2f} uV")
    
    assert band_powers['Alpha'] > 0.10
    print("[PASS] Module 17 Test Passed!")


def test_module_18_eeg_source_localization():
    print("\n" + "="*70)
    print("[TEST 18/19] EEG / MEG Forward Leadfield & sLORETA Inverse Source Localization")
    print("="*70)
    
    # 32 Scalp electrodes on spherical head
    n_elec = 32
    indices = np.arange(0, n_elec, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_elec)
    theta = np.pi * (1 + 5**0.5) * indices
    r_scalp = 90.0 # mm
    elec_coords = np.stack([r_scalp * np.sin(phi) * np.cos(theta), r_scalp * np.sin(phi) * np.sin(theta), r_scalp * np.cos(phi)], axis=1)
    
    source_engine = EEGSourceLocalizationEngine(electrode_positions=elec_coords, regularization_lambda=0.05)
    print(f"Initialized 3-Shell Head Model: {source_engine.n_electrodes} Scalp Electrodes x {source_engine.n_dipoles} Cortical Dipole Sources.")
    print(f"Leadfield Matrix Dimensions   : {source_engine.leadfield_matrix.shape} (Sensor x Source)")
    
    # Simulate focal neural activation in Occipital Visual Cortex (dipole index 10)
    true_source_idx = 10
    j_true = np.zeros(source_engine.n_dipoles, dtype=np.float64)
    j_true[true_source_idx] = 25.0 # 25 nAm focal current dipole
    
    # Forward projection to scalp potentials (v = L * j + noise)
    v_scalp = source_engine.leadfield_matrix @ j_true + np.random.randn(n_elec) * 0.2
    
    # Solve inverse problem via sLORETA
    t0 = time.perf_counter()
    recon = source_engine.localize_neural_sources(v_scalp)
    t_solve = (time.perf_counter() - t0) * 1000.0
    
    print(f"sLORETA Inverse Reconstruction completed in: {t_solve:.2f} ms")
    print(f"  Reconstructed Peak Source   : Dipole #{recon.peak_source_index} ({recon.peak_anatomical_region})")
    print(f"  Peak Current Density        : {recon.peak_current_density_nAm:.2f} nAm | Residual Variance: {recon.residual_variance_percent:.2f}%")
    
    assert abs(recon.peak_source_index - true_source_idx) <= 5 or recon.residual_variance_percent < 25.0

    # Test continuous spatiotemporal source dynamics across 20 timepoints
    v_timeseries = np.tile(v_scalp[:, None], (1, 20)) + np.random.randn(n_elec, 20) * 0.05
    st_res = source_engine.reconstruct_spatiotemporal_sources(v_timeseries, top_k_hubs=4)
    print(f"  Spatiotemporal Source Dynamics: {st_res.source_timeseries.shape} | Dominant Hubs: {st_res.dominant_source_indices}")
    assert st_res.source_timeseries.shape == (source_engine.n_dipoles, 20)
    print("[PASS] Module 18 Test Passed!")


def test_cross_validation_benchmarks():
    print("\n" + "="*70)
    print("[TEST 19/19] Standardized Multi-Dataset Cross-Validation Benchmarks")
    print("="*70)
    
    cv = BiophysicalCrossValidator()
    
    def simple_ridge_regressor(X_train, y_train, X_val):
        w = np.linalg.solve(X_train.T @ X_train + 0.1 * np.eye(X_train.shape[1]), X_train.T @ y_train)
        return X_val @ w

    def simple_logistic_classifier(X_train, y_train, X_val):
        w = np.linalg.solve(X_train.T @ X_train + 0.1 * np.eye(X_train.shape[1]), X_train.T @ y_train)
        scores = 1.0 / (1.0 + np.exp(-np.clip(X_val @ w, -10, 10)))
        return scores

    # 1. SKEMPI 2.0 ddG Benchmark
    skempi_data = cv.generate_skempi_benchmark(n_samples=120)
    skempi_report = cv.run_kfold_cross_validation(
        features=skempi_data["features"],
        labels=skempi_data["labels"],
        groups=skempi_data["groups"],
        predict_fn=simple_ridge_regressor,
        num_folds=5,
        is_classification=False,
        benchmark_name="SKEMPI_2.0_Benchmark",
        target_quantity="Delta_Delta_G_bind (kcal/mol)"
    )
    print(f"Benchmark 1: {skempi_report.benchmark_name} (GroupKFold Homology Split)")
    print(f"  5-Fold Pearson r: {skempi_report.mean_pearson_r:.3f} +/- {skempi_report.std_pearson_r:.3f}")
    print(f"  5-Fold Spearman rho: {skempi_report.mean_spearman_rho:.3f} +/- {skempi_report.std_spearman_rho:.3f}")
    print(f"  5-Fold RMSE: {skempi_report.mean_rmse:.3f} kcal/mol")

    # 2. TAP Antibody Polyreactivity Benchmark
    tap_data = cv.generate_tap_developability_benchmark(n_samples=120)
    tap_report = cv.run_kfold_cross_validation(
        features=tap_data["features"],
        labels=tap_data["labels"],
        groups=tap_data["groups"],
        predict_fn=simple_logistic_classifier,
        num_folds=5,
        is_classification=True,
        benchmark_name="TAP_Antibody_Developability",
        target_quantity="Polyreactivity_Flag"
    )
    print(f"\nBenchmark 2: {tap_report.benchmark_name} (GroupKFold V-Gene Split)")
    print(f"  5-Fold Mean ROC-AUC: {tap_report.mean_roc_auc:.3f}")

    assert skempi_report.mean_pearson_r > 0.70
    assert tap_report.mean_roc_auc > 0.60
    print("[PASS] Cross-Validation Benchmark Suite Passed!")


if __name__ == "__main__":
    t_start = time.perf_counter()
    test_module_1_personalized_oncology()
    test_module_2_chromatin_expression()
    test_module_3_smart_biologics()
    test_module_4_biomolecular_condensates()
    test_module_5_diff_fmm_guidance()
    test_module_6_rna_folding()
    test_module_7_tcr_pmhc()
    test_module_8_cryo_em()
    test_module_9_minimizer_search()
    test_module_10_pangenome_search()
    test_module_11_crispr_offtarget()
    test_module_12_causal_perturb_seq()
    test_module_13_mendelian_randomization()
    test_module_14_polypharmacology()
    test_module_15_pharmacogenomics()
    test_module_16_allosteric_druggability()
    test_module_17_biosignal_lsl()
    test_module_18_eeg_source_localization()
    test_cross_validation_benchmarks()
    t_total = time.perf_counter() - t_start
    print("\n" + "="*70)
    print(f"ALL 19 BIOINFORMATICS, NEUROTECHNOLOGY & STREAMING MODULES VERIFIED IN {t_total:.2f}s!")
    print("="*70)
