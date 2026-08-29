# One-shot build step: splice verbatim WGSL/JS blocks out of index.html into
# validate.template.html -> validate.html. Not part of the test suite.
import io, re, sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
src = (root / "index.html").read_text(encoding="utf-8")
tpl = (root / "validate.template.html").read_text(encoding="utf-8")

def template_literal(name):
    start_marker = "const " + name + " = `"
    i = src.index(start_marker)
    j = src.index("`", i + len(start_marker))
    return src[i + len(start_marker): j]

def between(start_marker, end_marker, strip_lines=()):
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    block = src[i:j].rstrip() + "\n"
    lines = block.split("\n")
    out = [ln for ln in lines if ln.strip() not in strip_lines]
    return "\n".join(out)

repl = {}

wgsl_compute = template_literal("wgslComputeSource")
wgsl_fmm = template_literal("wgslFmmSource")
wgsl_adaptive = template_literal("wgslAdaptiveFmmSource")
repl["/*__WGSL_COMPUTE__*/"] = "const wgslComputeSource = `\n" + wgsl_compute + "`;"
repl["/*__WGSL_FMM__*/"] = "const wgslFmmSource = `\n" + wgsl_fmm + "`;"
repl["/*__WGSL_ADAPTIVE__*/"] = "const wgslAdaptiveFmmSource = `\n" + wgsl_adaptive + "`;"

# computeFunnelGeometry .. FunnelTable class (ends before the uniform staging).
repl["/*__JS_FUNNEL__*/"] = between(
    "        function computeFunnelGeometry(capacity, delta) {",
    "        // Preallocated per-frame uniform staging",
)
# computeAdaptiveDepth / computeAdaptiveLeafTarget / cellsTouch /
# buildFarOperatorTable / buildAdaptiveMetadata (ends before initEngine).
repl["/*__JS_ADAPTIVE_META__*/"] = between(
    "        function computeAdaptiveDepth(n, leafTarget) {",
    "        async function initEngine() {",
)
# updateGalaxyCores (ends before the DYNAMIC OVERLAY LEGEND banner).
repl["/*__JS_UPDATE_CORES__*/"] = between(
    "        function updateGalaxyCores(dt) {",
    "        // ====================================================================\n        // DYNAMIC OVERLAY LEGEND",
)
# Seeded-RNG block (seedParam / seededState / resetSeededRandom + the
# Math.random override). generateParticles calls resetSeededRandom() at
# the top of every IC branch so the validator gets deterministic ICs.
# Stop before `let N` / `let scenario` — the rig config block declares
# its own `const N` and the spliced generateParticles block sets
# `scenario` from the URL params.
repl["/*__JS_SEEDED_RANDOM__*/"] = between(
    "        // Optional deterministic IC stream for controlled browser benchmarks.",
    "        const MAX_P = 10000000;",
)
# Standard self-gravity IC constants (galaxyIC + SELF_GRAV_MU/A +
# COLD_COLLAPSE_R0 + SELF_GRAV_DIAG_MAX_N) and generateParticles galaxy ICs
# (ends before the WebGPU COMPUTE banner); the two DOM/HUD lines are dropped
# (this page has no HUD) - the only edit.
repl["/*__JS_GENERATE__*/"] = between(
    "        // ====================================================================\n        // STANDARD SELF-GRAVITY SCENARIOS",
    "        // ====================================================================\n        // WebGPU COMPUTE",
    strip_lines=(
        "document.getElementById('valActiveParticles').innerText = N.toLocaleString();",
        "updateOverlayLegend();",
    ),
)

out = tpl
for k, v in repl.items():
    assert k in out, f"placeholder missing in template: {k}"
    out = out.replace(k, v)

assert "__WGSL_" not in out and "__JS_" not in out, "unreplaced placeholder"
for needed in ("fn main(", "fn direct(", "fn far_gather(", "fn energy_phi(",
               "class FunnelTable", "function buildAdaptiveMetadata",
               "function updateGalaxyCores", "function generateParticles",
               "let galaxyIC", "const SELF_GRAV_MU"):
    assert needed in out, f"missing expected block: {needed}"

(root / "validate.html").write_text(out, encoding="utf-8", newline="\n")
print(f"validate.html written: {len(out.splitlines())} lines, {len(out)} chars")
print(f"  wgslComputeSource:  {len(wgsl_compute.splitlines())} lines")
print(f"  wgslFmmSource:      {len(wgsl_fmm.splitlines())} lines")
print(f"  wgslAdaptiveFmmSource: {len(wgsl_adaptive.splitlines())} lines")
