# Numerical probe for the two suspected root causes of the grid artifacts
# reported against index.html's FMM modes:
#
#   A. "Multipole order p" truncates the LOCAL expansion too: at p = 0 the
#      l = 1 coefficient (the monopole's force term) is dropped, so the whole
#      List-2 M2L chain contributes ZERO force. Only List-1 P2P + List-3 M2P
#      monopoles survive -> particles feel only a ~7x7-cell neighborhood ->
#      "collapse into their respective squares".
#   B. The adaptive far field (M2L operators, P2L, List-3 M2P) uses the
#      EXACT log kernel while near-field P2P uses the softened kernel
#      (eps = 0.0063, demo P2P_EPS2 = 4e-5). In a collapsed core the
#      quadtree refines to depth 10 (cell ~ 1e-3 << eps), so List-2/3
#      sources sit at d ~ 2-3e-3 << eps and their force is overestimated
#      by ~(d^2 + eps^2)/d^2, up to ~11x.
#
# Uses the repo's own canonical engine (core.adaptive_fmm.AdaptiveFMM), whose
# far field is unsoftened and whose l2p_force starts at l = 1 -- the same
# semantics as both WGSL kernels in index.html.
#
# Run:  python tools/probe_order_softening.py
import sys
import numpy as np

sys.path.insert(0, ".")
from core.adaptive_fmm import AdaptiveFMM  # noqa: E402

EPS = 0.0063  # demo softening (P2P_EPS2 = 4e-5 -> eps = 0.00632)


def direct_forces(pos, q, eps):
    """Exact all-pairs softened log-kernel forces (demo Direct mode)."""
    n = len(pos)
    dx = pos[:, 0][:, None] - pos[None, :, 0]
    dy = pos[:, 1][:, None] - pos[None, :, 1]
    r2 = dx * dx + dy * dy + eps * eps
    np.fill_diagonal(r2, 1.0)
    inv = 1.0 / r2
    np.fill_diagonal(inv, 0.0)
    fx = -((dx * inv) @ q)
    fy = -((dy * inv) @ q)
    return fx, fy


def rel_err(f, fx_ref, fy_ref):
    ex = fx_ref - f[0]
    ey = fy_ref - f[1]
    num = np.sqrt(np.mean(ex * ex + ey * ey))
    den = np.sqrt(np.mean(fx_ref * fx_ref + fy_ref * fy_ref))
    return num / den


def report(tag, f_fmm, fx_ref, fy_ref):
    e = rel_err(f_fmm, fx_ref, fy_ref)
    mag_f = np.hypot(f_fmm[0], f_fmm[1])
    mag_r = np.hypot(fx_ref, fy_ref)
    print(f"    {tag:34s} rel-L2 force err = {e:8.4f}   "
          f"|F| median ratio = {np.median(mag_f) / np.median(mag_r):.3f}")
    return e


rng = np.random.default_rng(42)

print("=" * 78)
print("EXP A: expansion-order scan on a DIFFUSE uniform disk")
print("       (N=1500, R=0.3, softened eps=0.0063 everywhere in references)")
print("=" * 78)
n = 1500
r = 0.3 * np.sqrt(rng.random(n))
th = 2 * np.pi * rng.random(n)
pos = np.column_stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)])
q = np.full(n, 1.0 / n)
fx_s, fy_s = direct_forces(pos, q, EPS)      # exact softened (demo Direct)
fx_u, fy_u = direct_forces(pos, q, 0.0)      # exact unsoftened
for p in (0, 1, 2, 4):
    eng = AdaptiveFMM(max_leaf_particles=16, max_depth=10, p=p, softening=EPS)
    _, fx, fy = eng.evaluate(pos, q)
    report(f"p={p}  vs softened direct", (fx, fy), fx_s, fy_s)

print()
print("=" * 78)
print("EXP B: dense collapsed core (N=4000 Gaussian, p=4 fixed)")
print("       is the residual error truncation (would track either law)")
print("       or a SOFTENING MISMATCH (tracks the UNsoftened law)?")
print("=" * 78)
n = 4000
for sigma in (0.05, 0.02, 0.01, 0.005):
    pos = rng.normal([0.5, 0.5], sigma, size=(n, 2))
    q = np.full(n, 1.0 / n)
    fx_s, fy_s = direct_forces(pos, q, EPS)
    fx_u, fy_u = direct_forces(pos, q, 0.0)
    eng = AdaptiveFMM(max_leaf_particles=16, max_depth=10, p=4, softening=EPS)
    _, fx, fy = eng.evaluate(pos, q)
    lvl = eng._lvl[:eng.n_cells]
    print(f"  sigma = {sigma:.3f} (tree max level {int(lvl.max())}, "
          f"cells {eng.n_cells}):")
    report("AFMM p=4  vs SOFTENED direct", (fx, fy), fx_s, fy_s)
    report("AFMM p=4  vs UNsoftened direct", (fx, fy), fx_u, fy_u)
