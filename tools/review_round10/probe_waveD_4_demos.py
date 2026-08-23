"""Round-10 Wave D probe 4: run every bioinformatics module's __main__ demo
as a subprocess with a per-module timeout (CPU only)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "bioinformatics")

SKIP = {
    "benchmark_bioinformatics.py",  # long-running full benchmark; sampled separately
}

results = []
for fn in sorted(os.listdir(PKG)):
    if not fn.endswith(".py") or fn in SKIP:
        continue
    mod = os.path.join(PKG, fn)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, "-X", "utf8", mod],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        ok = (p.returncode == 0)
        err = p.stderr.strip().splitlines()[-1][:220] if p.stderr.strip() else ""
    except subprocess.TimeoutExpired:
        ok, err = False, "TIMEOUT after 600s"
    results.append((fn, ok, err))
    print(f"[{'OK  ' if ok else 'FAIL'}] {fn} {err}", flush=True)

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{n_fail} failures out of {len(results)} modules")
sys.exit(1 if n_fail else 0)
