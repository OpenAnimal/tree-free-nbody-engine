"""Check whether near+far covers all source particles for each target."""
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apps.app5_benchmark_variants import _protein, _direct_debye_huckel
from core.spatial_index import CellIndex

coords, charges = _protein(n_atoms=2000, seed=42)
depth = 6
kappa = 2.0
ci = CellIndex(dims=3, grid_res=depth)
unique_keys, inverse = ci.build(coords)
inverse = np.asarray(inverse, dtype=np.int64)
N = len(coords)

# For each target particle, count how many source particles are in near,
# how many in far, and check near+far == N-1 (all other particles).
cell_ints = np.array([ci.key_ints(int(k)) for k in unique_keys], dtype=np.int64)
K = len(unique_keys)
dci = cell_ints[:, None, :] - cell_ints[None, :, :]
cheb = np.max(np.abs(dci), axis=-1)
# cell-pair far mask
far_cell_mask = cheb > 2

# For each particle, near_count = sum of bucket sizes of near cells (minus self).
# far_count = sum of bucket sizes of far cells.
bucket_sizes = np.array([len(ci.bucket(int(k))) for k in unique_keys])
cell_id_of_key = {int(k): c for c, k in enumerate(unique_keys)}

near_count = np.zeros(N, dtype=np.int64)
far_count = np.zeros(N, dtype=np.int64)
for i in range(N):
    cid = inverse[i]
    # near: all cells within ring 2 of this particle's cell
    near_cells = ci.neighbor_keys(int(unique_keys[cid]), ring=2)
    nc = sum(len(ci.bucket(int(nk))) for nk in near_cells)
    near_count[i] = nc - 1  # exclude self
    # far: all occupied cells outside ring 2
    far_count[i] = N - nc  # all particles not in near cells

print(f"N={N}")
print(f"near_count: min={near_count.min()} max={near_count.max()} mean={near_count.mean():.1f}")
print(f"far_count:  min={far_count.min()} max={far_count.max()} mean={far_count.mean():.1f}")
print(f"near+far == N-1 for all? {np.all(near_count + far_count == N - 1)}")

# Now compute near-only potential and far-only potential, check sum vs direct.
# near-only: exact direct over near neighborhood
pot_near = np.zeros(N)
for cid, key in enumerate(unique_keys):
    idx_t = ci.bucket(int(key))
    if len(idx_t) == 0:
        continue
    near_idx = ci.neighborhood_indices(int(key), ring=2)
    xt = coords[idx_t]
    xs = coords[near_idx]
    qs = charges[near_idx]
    diff = xt[:, None, :] - xs[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    r_safe = np.where(r < 1e-30, 1.0, r)
    g = np.exp(-kappa * r_safe) / r_safe
    id_t = idx_t[:, None]
    id_s = near_idx[None, :]
    g = np.where(id_t == id_s, 0.0, g)
    pot_near[idx_t] += np.sum(qs[None, :] * g, axis=1)

# far-only: exact direct over far cells (the reference for the far field)
pot_far_exact = np.zeros(N)
for cid, key in enumerate(unique_keys):
    idx_t = ci.bucket(int(key))
    if len(idx_t) == 0:
        continue
    # far particles: all NOT in near cells
    near_keys_set = set(ci.neighbor_keys(int(key), ring=2))
    far_idx = np.array([j for j in range(N) if int(unique_keys[inverse[j]]) not in near_keys_set],
                       dtype=np.int64)
    if len(far_idx) == 0:
        continue
    xt = coords[idx_t]
    xs = coords[far_idx]
    qs = charges[far_idx]
    diff = xt[:, None, :] - xs[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-30
    g = np.exp(-kappa * r) / r
    pot_far_exact[idx_t] += np.sum(qs[None, :] * g, axis=1)

pot_direct = _direct_debye_huckel(coords, charges, kappa=kappa)
print(f"\n||pot_near + pot_far_exact - pot_direct|| / ||pot_direct|| = "
      f"{np.linalg.norm(pot_near + pot_far_exact - pot_direct) / np.linalg.norm(pot_direct):.3e}")
print(f"||pot_far_exact|| / ||pot_direct|| = {np.linalg.norm(pot_far_exact) / np.linalg.norm(pot_direct):.3e}")
print(f"||pot_near|| / ||pot_direct|| = {np.linalg.norm(pot_near) / np.linalg.norm(pot_direct):.3e}")

# Now compare FMM far field vs exact far field at various p.
from core import Yukawa3DFMM
print(f"\n{'p':>4} {'||far_fmm - far_exact||/||far_exact||':>40}")
for p in (4, 6, 8, 10, 12):
    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
    full = fmm.evaluate(coords, charges)
    far_fmm = full - pot_near
    rel = np.linalg.norm(far_fmm - pot_far_exact) / np.linalg.norm(pot_far_exact)
    print(f"{p:>4} {rel:>40.4e}")
