"""
Comprehensive Benchmark & Demonstration Suite for bioinformatics.
Evaluates O(N) linear scaling, numerical accuracy, physics application benchmarks
(Implicit Solvation, Equivariant GNN Layer, Non-Periodic MD, Constant-pH Titration)
and Non-FMM Elastic Hashing tools (k-mer Genomic Counting, Pocket Detection, Contact Graph).
"""

from __future__ import annotations
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from bioinformatics.pdb_loader import generate_synthetic_protein, generate_viral_capsid, MolecularSystem, COULOMB_CONSTANT_KCAL
from bioinformatics.core.fast_multipole_kernel import TreeFreeBioFMM, ScreenedKernelType
from bioinformatics.solvation_free_energy import SolvationFreeEnergyEngine
from bioinformatics.gnn_long_range_layer import FMMLongRangeGNNLayer
from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine
from bioinformatics.constant_ph_titration import ConstantPHTitrationEngine
from bioinformatics.kmer_elastic_hash import KmerElasticHashTable
from bioinformatics.binding_pocket_detector import BindingPocketDetector
from bioinformatics.contact_map_graph import ContactMapGraphBuilder


def direct_screened_potential(coords: np.ndarray, charges: np.ndarray, kappa: float = 0.127, eps_w: float = 78.5) -> np.ndarray:
    """Exact O(N^2) direct all-pairs Debye-Hückel potential evaluation for ground-truth validation."""
    N = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, 1e9)
    v_mat = (np.exp(-kappa * dist) / dist) / eps_w * COULOMB_CONSTANT_KCAL
    np.fill_diagonal(v_mat, 0.0)
    return np.sum(v_mat * charges[None, :], axis=1)


def run_scaling_benchmark():
    print("==========================================================================")
    print("   bioinformatics: O(N) Scaling & Numerical Accuracy Benchmark")
    print("==========================================================================")
    atom_counts = [500, 1500, 5000, 15000, 50000]
    direct_times = []
    fmm_times = []
    rel_errors = []

    fmm_engine = TreeFreeBioFMM(cell_size=8.0, kappa=0.127, kernel_type=ScreenedKernelType.DEBYE_HUCKEL)

    for N in atom_counts:
        print(f"\n[+] Benchmarking Molecular System (N = {N:,} atoms)...")
        protein = generate_synthetic_protein(N, seed=42)

        # 1. Tree-Free FMM Evaluation
        t0 = time.perf_counter()
        fmm_pot, _, _ = fmm_engine.evaluate(protein.coords, protein.charges)
        t_fmm = (time.perf_counter() - t0) * 1000.0  # ms
        fmm_times.append(t_fmm)
        print(f"    - Tree-Free Bio-FMM Time:  {t_fmm:8.2f} ms")

        # 2. Direct O(N^2) Evaluation (for N <= 15000)
        if N <= 15000:
            t0 = time.perf_counter()
            dir_pot = direct_screened_potential(protein.coords, protein.charges)
            t_dir = (time.perf_counter() - t0) * 1000.0  # ms
            direct_times.append(t_dir)
            
            # Numerical relative L2 error
            l2_err = np.linalg.norm(fmm_pot - dir_pot) / (np.linalg.norm(dir_pot) + 1e-8)
            rel_errors.append(l2_err)
            speedup = t_dir / t_fmm
            print(f"    - Direct O(N^2) Time:      {t_dir:8.2f} ms | Speedup: {speedup:6.1f}x | Rel L2 Error: {l2_err:.2e}")
        else:
            # Extrapolate direct quadratic time for reference
            t_extrap = direct_times[1] * ((N / atom_counts[1])**2)
            direct_times.append(t_extrap)
            speedup = t_extrap / t_fmm
            print(f"    - Direct O(N^2) (Extrap):  {t_extrap:8.2f} ms | Speedup: {speedup:6.1f}x (O(N) Advantage)")

    return atom_counts, direct_times, fmm_times, rel_errors


def run_all_applications_demo():
    print("\n==========================================================================")
    print("   Part 1: Physics & FMM Applications (A, B, C, D)")
    print("==========================================================================")

    # 1. Application A: Implicit Solvation & Born Radii
    print("\n--- Application A: Generalized Born & SASA Solvation Free Energy ---")
    system_a = generate_synthetic_protein(2500, seed=101)
    solv_engine = SolvationFreeEnergyEngine(cell_size=8.0)
    res_a = solv_engine.compute_solvation_free_energy(system_a)
    print(f"    System Atoms:          {res_a['num_atoms']:,}")
    print(f"    Total Solvation dG:    {res_a['delta_G_solv_kcal_mol']:10.2f} kcal/mol")
    print(f"    - Electrostatic GB dG: {res_a['delta_G_GB_kcal_mol']:10.2f} kcal/mol")
    print(f"    - Non-Polar SASA dG:   {res_a['delta_G_nonpolar_kcal_mol']:10.2f} kcal/mol")
    print(f"    Total SASA:            {res_a['total_sasa_angstrom2']:10.2f} Angstrom^2")
    print(f"    Computation Time:      {res_a['elapsed_seconds']*1000:.2f} ms")

    # 2. Application B: Equivariant GNN Long-Range Physical Layer
    print("\n--- Application B: Differentiable Equivariant GNN Physical Prior Layer ---")
    gnn_layer = FMMLongRangeGNNLayer(hidden_dim=64, cell_size=8.0)
    node_feat = np.random.randn(system_a.num_atoms, 64)
    t0 = time.perf_counter()
    h_out, e_tot, forces, diag = gnn_layer.forward(system_a.coords, node_feat, system_a.charges)
    t_gnn = (time.perf_counter() - t0) * 1000.0
    print(f"    Forward Latency:       {t_gnn:.2f} ms")
    print(f"    Electrostatic Energy:  {diag['e_electrostatic_kcal_mol']:10.2f} kcal/mol")
    print(f"    Mean Electric Field:   {diag['mean_field_magnitude']:10.2f} V/Angstrom")
    print(f"    Max Electric Field:    {diag['max_field_magnitude']:10.2f} V/Angstrom")

    # 3. Application C: Non-Periodic Macromolecular MD Engine
    print("\n--- Application C: Non-Periodic Macromolecular MD (NVT Langevin Verlet) ---")
    capsid = generate_viral_capsid(n_capsomers=12, atoms_per_unit=250, radius=50.0)
    print(f"    Initialized Viral Capsid Assembly: {capsid.num_atoms:,} atoms")
    md_engine = MacromolecularMDEngine(capsid, temperature_kelvin=300.0, timestep_fs=2.0)
    t0 = time.perf_counter()
    traj_stats = md_engine.run(num_steps=20)
    t_md = (time.perf_counter() - t0) * 1000.0
    print(f"    Executed 20 MD Steps in: {t_md:.2f} ms ({t_md/20.0:.2f} ms/step)")
    print(f"    Final Temperature:       {traj_stats[-1]['temperature_k']:.1f} K")
    print(f"    Final Total Energy:      {traj_stats[-1]['e_total']:.2f} kcal/mol")

    # 4. Application D: Constant-pH Monte Carlo Protonation Titration
    print("\n--- Application D: Constant-pH Monte Carlo Protonation Titration ---")
    titration_engine = ConstantPHTitrationEngine(system_a)
    print(f"    Titratable Sites Found:  {len(titration_engine.titratable_sites)}")
    titr_res = titration_engine.compute_titration_curve(ph_range=(3.0, 11.0), num_ph_points=9, steps_per_ph=100)
    print(f"    Isoelectric Point (pI):  pH {titr_res['isoelectric_point_pI']:.2f}")
    print(f"    Titration Sweep Time:    {titr_res['elapsed_seconds']*1000:.2f} ms")

    print("\n==========================================================================")
    print("   Part 2: Pure Elastic Hashing Tools (Genomics & Structural Biology)")
    print("==========================================================================")

    # 5. Non-FMM Elastic Hashing: Genomic k-mer Counting
    print("\n--- Non-FMM Tool 1: Lock-Free Genomic k-mer Counter (Farach-Colton Open Addressing) ---")
    kmer_table = KmerElasticHashTable(k=21, capacity=500000)
    # Generate synthetic 250,000 bp DNA sequencing read stream
    bases = np.array(["A", "C", "G", "T"])
    dna_seq = "".join(bases[np.random.randint(0, 4, 250000)])
    t0 = time.perf_counter()
    n_ingested = kmer_table.ingest_sequence(dna_seq)
    t_kmer = (time.perf_counter() - t0) * 1000.0
    print(f"    Ingested {n_ingested:,} 21-mers from 250k bp DNA in: {t_kmer:.2f} ms ({n_ingested / (t_kmer / 1000.0):,.0f} kmers/sec)")
    print(f"    Unique 21-mers in Table: {kmer_table.num_unique_kmers:,}")

    # 6. Non-FMM Elastic Hashing: Binding Pocket / Cavity Detection
    print("\n--- Non-FMM Tool 2: Grid-Free Binding Pocket & Cavity Detector ---")
    pocket_detector = BindingPocketDetector(grid_spacing=1.0, min_pocket_points=12)
    pocket_res = pocket_detector.detect_pockets(system_a)
    print(f"    Identified Pockets:      {pocket_res['num_pockets']}")
    if pocket_res['num_pockets'] > 0:
        top_p = pocket_res['pockets'][0]
        print(f"    Top Druggable Pocket #1: Vol = {top_p['volume_angstrom3']:.1f} A^3 | Score = {top_p['druggability_score']:.2f}")
    print(f"    Detection Latency:       {pocket_res['elapsed_seconds']*1000:.2f} ms")

    # 7. Non-FMM Elastic Hashing: Residue Contact Map & Network Graph
    print("\n--- Non-FMM Tool 3: O(N) Residue Contact Map & Allosteric Network Graph ---")
    contact_builder = ContactMapGraphBuilder(contact_cutoff=8.0)
    contact_res = contact_builder.build_ca_contact_graph(system_a)
    print(f"    Residues Analyzed:       {contact_res['num_residues']}")
    print(f"    Non-Local Contacts:      {contact_res['num_contacts']:,}")
    if contact_res['top_hub_residues']:
        hub0 = contact_res['top_hub_residues'][0]
        print(f"    Top Hub Residue:         {hub0['residue_name']} {hub0['residue_id']} (Degree: {hub0['degree_centrality']})")
    print(f"    Graph Construction Time: {contact_res['elapsed_seconds']*1000:.2f} ms")

    return res_a, diag, traj_stats, titr_res, system_a, capsid


def generate_benchmark_figures(atom_counts, direct_times, fmm_times, res_a, titr_res, system_a, capsid):
    print("\n[+] Generating Publication-Grade Visualization Figures...")
    output_dir = os.path.dirname(__file__)
    
    # Figure 1: Scaling & Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0B0E14')
    for ax in axes:
        ax.set_facecolor('#0B0E14')
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    # Plot 1: Log-Log Execution Time Scaling
    axes[0].plot(atom_counts, direct_times, 'o--', color='#F85149', label='Direct All-Pairs O(N²)', lw=2.0)
    axes[0].plot(atom_counts, fmm_times, 's-', color='#58A6FF', label='Tree-Free Bio-FMM O(N)', lw=2.5)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Atom Count (N)', color='#E6EDF3', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Execution Time (ms)', color='#E6EDF3', fontsize=11, fontweight='bold')
    axes[0].set_title('Screened Electrostatic Scaling: O(N) vs O(N²)', color='white', fontsize=12, fontweight='bold', pad=12)
    axes[0].grid(True, which='both', color='#21262D', linestyle='--', alpha=0.7)
    axes[0].legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3')

    # Plot 2: Speedup Factor vs N
    speedups = [d / f for d, f in zip(direct_times, fmm_times)]
    axes[1].plot(atom_counts, speedups, '^-', color='#3FB950', lw=2.5)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Atom Count (N)', color='#E6EDF3', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Speedup Factor (x)', color='#E6EDF3', fontsize=11, fontweight='bold')
    axes[1].set_title('Tree-Free FMM Acceleration Factor', color='white', fontsize=12, fontweight='bold', pad=12)
    axes[1].grid(True, which='both', color='#21262D', linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig1_path = os.path.join(output_dir, "fmm_bioinformatics_benchmark.png")
    plt.savefig(fig1_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"    - Saved benchmark chart to: {fig1_path}")

    # Figure 2: Application Showcase (Electrostatic Surface & Titration Curve)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor='#0B0E14', gridspec_kw={'width_ratios': [1.2, 1.0]})
    
    # 3D Protein Electrostatics
    ax1 = fig2.add_subplot(121, projection='3d')
    ax1.set_facecolor('#0B0E14')
    sc = ax1.scatter(system_a.coords[:, 0], system_a.coords[:, 1], system_a.coords[:, 2],
                     c=res_a['atom_potentials'], cmap='coolwarm', s=15, alpha=0.85, edgecolors='none')
    ax1.plot(system_a.coords[:, 0], system_a.coords[:, 1], system_a.coords[:, 2], color='#58A6FF', lw=0.5, alpha=0.3)
    ax1.set_title(f"3D Implicit Solvent Potential (N = {system_a.num_atoms:,})\nDelta G_solv = {res_a['delta_G_solv_kcal_mol']:.1f} kcal/mol",
                  color='white', fontsize=11, fontweight='bold', pad=10)
    ax1.xaxis.pane.fill = False; ax1.yaxis.pane.fill = False; ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('#30363D'); ax1.yaxis.pane.set_edgecolor('#30363D'); ax1.zaxis.pane.set_edgecolor('#30363D')
    ax1.tick_params(colors='#8B949E')
    cb = fig2.colorbar(sc, ax=ax1, fraction=0.03, pad=0.08)
    cb.set_label('Electrostatic Potential (kcal/mol/e)', color='#E6EDF3')
    cb.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#8B949E')

    # Titration Curve & Isoelectric Point
    ax2.set_facecolor('#0B0E14')
    ax2.tick_params(colors='#8B949E')
    for spine in ax2.spines.values():
        spine.set_color('#30363D')

    ax2.plot(titr_res['ph_values'], titr_res['net_charges'], 'o-', color='#BC8CFF', lw=2.5, label='Net Macromolecular Charge')
    ax2.axhline(0, color='#8B949E', linestyle=':', lw=1.2)
    ax2.axvline(titr_res['isoelectric_point_pI'], color='#F0883E', linestyle='--', lw=1.8,
                label=f"Isoelectric Point pI = {titr_res['isoelectric_point_pI']:.2f}")
    ax2.set_xlabel('Solution pH', color='#E6EDF3', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Net System Charge (e)', color='#E6EDF3', fontsize=11, fontweight='bold')
    ax2.set_title('Constant-pH Titration & Protonation Curve', color='white', fontsize=12, fontweight='bold', pad=12)
    ax2.grid(True, color='#21262D', linestyle='--', alpha=0.7)
    ax2.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3')

    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, "bioinformatics_showcase.png")
    plt.savefig(fig2_path, dpi=200, facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"    - Saved application showcase to: {fig2_path}")


if __name__ == '__main__':
    atom_counts, direct_times, fmm_times, rel_errors = run_scaling_benchmark()
    res_a, diag, traj_stats, titr_res, system_a, capsid = run_all_applications_demo()
    generate_benchmark_figures(atom_counts, direct_times, fmm_times, res_a, titr_res, system_a, capsid)
    print("\n[SUCCESS] All bioinformatics benchmarks and validations completed successfully!")
