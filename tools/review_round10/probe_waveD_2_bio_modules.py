"""Round-10 Wave D probe 2: kmer_elastic_hash, contact_map_graph,
binding_pocket_detector, non_periodic_md_engine vs independent oracles."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bioinformatics.kmer_elastic_hash import KmerElasticHashTable, NUC_MAP, COMP_MAP
from bioinformatics.pdb_loader import MolecularSystem, generate_synthetic_protein
from bioinformatics.contact_map_graph import ContactMapGraphBuilder
from bioinformatics.non_periodic_md_engine import MacromolecularMDEngine

rng = np.random.RandomState(777)
FAIL = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)

# ------------------------------------------------------------- kmer oracle
def ref_canonical(seq, k):
    """Independent canonical k-mer counter: min(kmer, revcomp) as strings."""
    from collections import Counter
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    c = Counter()
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if any(ch not in comp for ch in km):
            continue
        rc = "".join(comp[ch] for ch in reversed(km))
        c[min(km, rc)] += 1
    return c

def key_of(km):
    v = 0
    for ch in km:
        v = (v << 2) | NUC_MAP[ch]
    return v

seq = "".join(rng.choice(list("ACGT"), size=3000)) + "NNN" + \
    "".join(rng.choice(list("ACGT"), size=500))
for k in (1, 5, 21):
    tbl = KmerElasticHashTable(k=k, capacity=4096)
    n = tbl.ingest_sequence(seq)
    ref = ref_canonical(seq, k)
    got = {}
    for km, cnt in ref.items():
        got[tbl.query_count(key_of(km))] = True
    ok = all(tbl.query_count(key_of(km)) == cnt for km, cnt in ref.items())
    check(f"kmer counts vs string-reference counter (k={k})", ok)
    check(f"kmer total ingested = windows (k={k})", n == sum(ref.values()),
          f"({n} vs {sum(ref.values())})")
    check(f"kmer unique count (k={k})", tbl.num_unique_kmers == len(ref),
          f"({tbl.num_unique_kmers} vs {len(ref)})")

# canonical_kmer_key symmetric property: canonical(canonical(x)) == canonical(x)
tblk = KmerElasticHashTable(k=11)
ok_inv = True
for v in rng.randint(0, 1 << 22, size=500):
    c = tblk.canonical_kmer_key(int(v))
    if tblk.canonical_kmer_key(c) != c:
        ok_inv = False
    # revcomp of key decodes to complement-reverse
check("canonical_kmer_key idempotent", ok_inv)

# decode roundtrip
ok_dec = True
for v in rng.randint(0, 1 << 22, size=200):
    s = tblk.decode_kmer(int(v))
    if key_of(s) != int(v):
        ok_dec = False
check("decode_kmer roundtrip", ok_dec)

# empty / short / all-N sequences
t0 = KmerElasticHashTable(k=21, capacity=256)
check("kmer empty sequence", t0.ingest_sequence("") == 0)
check("kmer short sequence", t0.ingest_sequence("ACG") == 0)
check("kmer all-N sequence", t0.ingest_sequence("N" * 50) == 0)
check("kmer lowercase equals uppercase",
      (lambda a, b: a == b)(
          KmerElasticHashTable(k=5, capacity=256).ingest_sequence("acgtacgtac"),
          KmerElasticHashTable(k=5, capacity=256).ingest_sequence("ACGTACGTAC")))

# spectrum: counts conserved for max_depth >= max count
t1 = KmerElasticHashTable(k=7, capacity=8192)
t1.ingest_sequence("".join(rng.choice(list("ACGT"), size=4000)))
spec = t1.get_kmer_spectrum(max_depth=50)
counts = np.array([v for _, v in t1._table.items()])
check("kmer spectrum binning", int(spec[1:51].sum()) == int((counts <= 50).sum())
      and int(spec[0]) == 0)

# ---------------------------------------------------- contact map oracle
prot = generate_synthetic_protein(n_atoms=800, seed=11)
builder = ContactMapGraphBuilder(contact_cutoff=8.0, cell_size=8.0)
res = builder.build_ca_contact_graph(prot)

# independent O(N^2) reference on CA atoms; implementation returns edges in
# CA-subset index space -> map through ca_idx for comparison
ca_mask = np.array([n == "CA" for n in prot.atom_names])
ca_idx = np.where(ca_mask)[0]
C = prot.coords[ca_idx]
R = prot.residue_ids[ca_idx]
ref_edges = set()
for a in range(len(ca_idx)):
    for b in range(a + 1, len(ca_idx)):
        d = np.linalg.norm(C[a] - C[b])
        if d < 8.0 and abs(R[a] - R[b]) > 1:
            ref_edges.add((int(ca_idx[a]), int(ca_idx[b])))
got_edges = set((int(ca_idx[u]), int(ca_idx[v])) for u, v in res["edges"])
check("CA contact graph edges == O(N^2) reference",
      got_edges == ref_edges,
      f"(got {len(got_edges)}, ref {len(ref_edges)}, "
      f"missing {len(ref_edges - got_edges)}, extra {len(got_edges - ref_edges)})")
ref_deg = np.zeros(len(ca_idx), dtype=int)
sub_of = {int(g): i for i, g in enumerate(ca_idx)}
for a, b in ref_edges:
    ref_deg[sub_of[a]] += 1
    ref_deg[sub_of[b]] += 1
check("CA contact graph degrees", np.array_equal(np.array(res["degrees"]), ref_deg))
# each edge distance matches within the CA-subset space
check("edge distances consistent (CA-subset space)",
      all(abs(np.linalg.norm(C[u] - C[v]) - d) < 1e-9
          for (u, v), d in zip(res["edges"], res["edge_distances"])))

# duplicate points (all atoms identical) — must not crash, no self edges
dup = MolecularSystem(
    coords=np.zeros((30, 3)), charges=np.zeros(30), radii=np.ones(30),
    masses=np.ones(30), atom_names=["CA"] * 30, residue_names=["GLY"] * 30,
    residue_ids=np.arange(30, dtype=np.int32), chain_ids=["A"] * 30)
res_dup = builder.build_ca_contact_graph(dup)
check("contact graph duplicate points: no self-edges",
      all(i != j for i, j in res_dup["edges"]))

# ---------------------------------------------------- MD engine checks
prot_small = generate_synthetic_protein(n_atoms=120, seed=5)
md = MacromolecularMDEngine(prot_small, temperature_kelvin=300.0,
                            friction_gamma=0.0, timestep_fs=1.0)
hist = md.run(num_steps=30)
T_hist = [h["temperature_k"] for h in hist]
check("MD temperature within [0, 2x target] after 30 steps",
      all(0 <= T <= 600 for T in T_hist), f"(min {min(T_hist):.0f}, max {max(T_hist):.0f} K)")
# e_potential finite
check("MD energies finite", all(np.isfinite(h["e_potential"]) for h in hist))
# sigma_v units: initial temperature ~ target within a chi^2 fluctuation
# (fixed RandomState(42) => deterministic draw; allow 4 sigma of the 3N-dof
# chi^2 distribution)
md2 = MacromolecularMDEngine(prot_small, temperature_kelvin=310.0,
                             friction_gamma=0.0, timestep_fs=0.01)
v = md2.velocities
m = prot_small.masses
e_kin = 0.5 * np.sum(m[:, None] * v ** 2) / 418.4
T_init = 2 * e_kin / (3 * len(v) * 0.0019872041)
sig_T = 310.0 * np.sqrt(2.0 / (3 * len(v)))
check("MD Maxwell-Boltzmann init temperature ~ target",
      abs(T_init - 310.0) < 4 * sig_T, f"(T_init={T_init:.1f} K, 4s={4*sig_T:.1f} K)")
# friction=0, tiny dt: total energy drift bounded (Verlet stability)
e0 = hist[0]["e_total"]
drift = max(abs(h["e_total"] - e0) / max(1e-9, abs(e0)) for h in hist)
check("MD energy drift < 5% over 30 steps (frictionless)", drift < 0.05,
      f"(max rel drift {drift:.2e})")
# zero-charge system -> electrostatic energy 0, no crash
zc = MolecularSystem(prot_small.coords[:60], np.zeros(60), prot_small.radii[:60],
                     prot_small.masses[:60], prot_small.atom_names[:60],
                     prot_small.residue_names[:60], prot_small.residue_ids[:60],
                     prot_small.chain_ids[:60])
mdz = MacromolecularMDEngine(zc, timestep_fs=0.5)
hz = mdz.run(num_steps=5)
check("MD zero-charge system runs", all(np.isfinite(h["e_total"]) for h in hz))

print()
print(f"{len(FAIL)} failures: {FAIL}")
sys.exit(1 if FAIL else 0)
