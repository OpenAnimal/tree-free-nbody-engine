"""Drop-in guarantees for neural_ops (round 9).

Pins the properties that make ``neural_ops/`` copyable into another
codebase without the rest of the repository:

1. no module-scope ``core.*`` imports outside the dependency shim;
2. the standalone fallbacks in ``_core_deps`` produce outputs matching
   the canonical ``core/`` implementations (CellIndex identical; FGT
   direct fallback is the accuracy reference of the fast engine);
3. the [0, 1)^dims coordinate contract warns instead of silently
   clipping out-of-range inputs, on the flagship forwards;
4. a temp-copy of ``neural_ops/`` imports and runs without the repo
   (executed as a subprocess so the parent process cannot leak the repo
   onto sys.path).
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "neural_ops"


# ------------------------------------------------------------------ 1
def test_no_module_scope_core_imports_outside_shim():
    """Only try/except-guarded imports (which degrade gracefully when
    core/ is absent) are allowed outside the dependency shim."""
    offenders = []
    for p in sorted(PKG.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ImportFrom):
                        guarded.add(sub.lineno)
                    elif isinstance(sub, ast.Import):
                        guarded.add(sub.lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".")[0]
                if root == "core" and p.name != "_core_deps.py" and node.lineno not in guarded:
                    offenders.append(f"{p.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "core" and p.name != "_core_deps.py" and node.lineno not in guarded:
                        offenders.append(f"{p.name}:{node.lineno}")
    assert offenders == [], f"unguarded module-scope core imports: {offenders}"


# ------------------------------------------------------------------ 2
def test_cellindex_fallback_matches_canonical():
    from core.spatial_index import CellIndex as Canonical
    from neural_ops._core_deps import _FallbackCellIndex

    rng = np.random.default_rng(7)
    for dims, res in [(1, 64), (2, 16), (3, 8)]:
        pts = rng.random((500, dims))
        a, b = Canonical(dims=dims, grid_res=res), _FallbackCellIndex(dims=dims, grid_res=res)
        ka, ia = a.build(pts)
        kb, ib = b.build(pts)
        np.testing.assert_array_equal(ka, kb)
        np.testing.assert_array_equal(ia, ib)
        for key in a.occupied_keys():
            np.testing.assert_array_equal(a.bucket(key), b.bucket(key))
            np.testing.assert_array_equal(
                a.neighborhood_indices(key), b.neighborhood_indices(key))
            assert a.cell_id(key) == b.cell_id(key)
        keys_a, inv_a, cnt_a, cen_a, tot_a = a.moments(pts, rng.random(500))
        keys_b, inv_b, cnt_b, cen_b, tot_b = b.moments(pts, None)  # weights optional
        assert keys_a == keys_b
        np.testing.assert_allclose(cnt_a, cnt_b)
        _, _, _, cen_a_u, _ = a.moments(pts, None)
        np.testing.assert_allclose(cen_a_u, cen_b, atol=1e-12)


def test_fgt_direct_fallback_is_accuracy_reference():
    """The canonical fast FGT must match the direct sum to truncation."""
    from core.gaussian2d_fgt import Gaussian2DFGT as CoreFGT
    from neural_ops._core_deps import _Gaussian2DFGTDirect, _Gaussian3DFGTDirect
    from core.gaussian2d_fgt import Gaussian3DFGT as CoreFGT3

    rng = np.random.default_rng(11)
    pos2 = rng.random((400, 2))
    q2 = rng.standard_normal(400)
    fast = CoreFGT(depth=5, p=8, h=0.25).evaluate(pos2, q2)
    direct = _Gaussian2DFGTDirect(h=0.25).evaluate(pos2, q2)
    rel2 = np.linalg.norm(fast - direct) / np.linalg.norm(direct)
    assert rel2 < 5e-3, rel2

    pos3 = rng.random((300, 3))
    q3 = rng.standard_normal(300)
    fast3 = CoreFGT3(depth=6, p=8, h=0.4).evaluate(pos3, q3)
    direct3 = _Gaussian3DFGTDirect(h=0.4).evaluate(pos3, q3)
    rel3 = np.linalg.norm(fast3 - direct3) / np.linalg.norm(direct3)
    assert rel3 < 5e-2, rel3


def test_fgt_direct_fallback_build_prebuilt_matches_evaluate():
    from neural_ops._core_deps import _Gaussian2DFGTDirect
    rng = np.random.default_rng(13)
    pos = rng.random((100, 2))
    q = rng.standard_normal(100)
    eng = _Gaussian2DFGTDirect(h=0.3)
    built = eng.build_operator(pos)
    np.testing.assert_allclose(eng.evaluate_prebuilt(built, q), eng.evaluate(pos, q))


# ------------------------------------------------------------------ 3
@pytest.mark.parametrize("cls", [
    "TreeFreeMultipoleAttention",
    "ContinuousMeshfreeGNNLayer",
    "FlashMultipoleAttentionEngine",
])
def test_coord_contract_warns_on_out_of_range(cls):
    import importlib
    mod_name = {
        "TreeFreeMultipoleAttention": "multipole_attention",
        "ContinuousMeshfreeGNNLayer": "continuous_meshfree_gnn",
        "FlashMultipoleAttentionEngine": "flash_multipole_kernel",
    }[cls]
    mod = importlib.import_module(f"neural_ops.{mod_name}")
    klass = getattr(mod, cls)
    rng = np.random.default_rng(3)
    N, D = 96, 8
    Q, K, V = (rng.standard_normal((N, D)).astype(np.float32) for _ in range(3))

    if cls == "ContinuousMeshfreeGNNLayer":
        coords = rng.random((N, 3)).astype(np.float32)
        obj = klass(in_features=D, out_features=D, spatial_dim=3)

        def call(c):
            return obj.forward(Q, c)[0]
    else:
        # both attention engines quantize 3D coordinates by default
        coords = rng.random((N, 3)).astype(np.float32)
        obj = klass(embed_dim=D, spatial_dim=3) if cls == "TreeFreeMultipoleAttention" else klass()

        def call(c):
            return obj.forward(Q, K, V, c)[0]

    # in-range: no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = call(coords)
        assert np.isfinite(out).all()
        assert not any("unit domain" in str(x.message) for x in w)
    # out-of-range: exactly the contract warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        call(coords * 10.0)
        hits = [x for x in w if "unit domain" in str(x.message)]
        assert hits, f"{cls}: no unit-domain warning for coords in [0,10)"


# ------------------------------------------------------------------ 4
def test_standalone_copy_imports_and_runs():
    """Copy neural_ops/ to a temp dir; import + forward without the repo."""
    src = PKG
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "neural_ops"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        code = """
import sys, warnings
assert not any('FMM_Repos' in p for p in sys.path), sys.path
import numpy as np
import neural_ops
from neural_ops import taylor_fgt_attention
from neural_ops._core_deps import USING_CORE
assert USING_CORE is False, USING_CORE
rng = np.random.default_rng(0)
N, D = 64, 8
Q, K, V = (rng.standard_normal((N, D)).astype(np.float32) for _ in range(3))
coords = rng.random((N, 2)).astype(np.float32)
att = neural_ops.TreeFreeMultipoleAttention(embed_dim=D, spatial_dim=2, grid_depth=4)
out, _ = att.forward(Q, K, V, coords)
assert np.isfinite(out).all()
fgt = taylor_fgt_attention.TaylorFGTAttention(spatial_dim=2)
out2, _ = fgt.forward(Q, K, V, coords)
assert np.isfinite(out2).all()
print('STANDALONE-OK')
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=180, cwd=tmp)
        assert r.returncode == 0, r.stderr[-800:]
        assert "STANDALONE-OK" in r.stdout


def test_script_mode_execution_still_works():
    """python neural_ops/taylor_fgt_attention.py (root not preloaded)."""
    r = subprocess.run([sys.executable, "neural_ops/taylor_fgt_attention.py"],
                       capture_output=True, text=True, timeout=240, cwd=REPO)
    assert r.returncode == 0, r.stderr[-800:]
