// Emits the adaptive-FMM metadata built by index.html's buildAdaptiveMetadata
// for a synthetic scene, as a flat binary file for the Python cross-validator.
//
// Usage: node emit_adaptive_meta.mjs <scene> <N> <depth> <out.bin>
// Scenes: uniform | gaussian | clusters | hardedge | single
//
// The REAL shipped source is used: the function bodies are sliced verbatim
// out of index.html and eval'd here, so this harness cannot drift from the
// page. Only the FunnelTable dependency is stubbed (it is not consumed by
// the Python emulator; the nodeHash arrays are emitted as zeros of the right
// shape so byte layouts stay stable).
//
// Round 13: the emitted binary carries the materialized far-field CSR
// (nodeMeta packing + farStart/farCount/farEntries + the farOps operator
// table) so tools/validate_adaptive_js.py can validate the materialized
// far_gather path, not just the legacy chain.
import { readFileSync, writeFileSync } from 'node:fs';

const [scene, nStr, depthStr, outPath] = process.argv.slice(2);
const N = parseInt(nStr, 10);
const depth = parseInt(depthStr, 10);

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const startMark = 'function computeAdaptiveDepth';
const start = html.indexOf(startMark);
if (start < 0) throw new Error('computeAdaptiveDepth not found in index.html');
const endMark = 'async function initEngine';
const end = html.indexOf(endMark, start);
if (end < 0) throw new Error('initEngine marker not found after builder');
const source = html.slice(start, end);

// Deterministic PRNG (mulberry32) so scenes are reproducible across runs.
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const rnd = mulberry32(42);
function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

const pos = new Float32Array(N * 4);
for (let i = 0; i < N; i++) {
    let x, y;
    if (scene === 'uniform') { x = rnd(); y = rnd(); }
    else if (scene === 'gaussian') { x = 0.5 + 0.12 * gauss(); y = 0.5 + 0.12 * gauss(); }
    else if (scene === 'clusters') {
        const c = i % 4;
        const cx = [0.25, 0.75, 0.25, 0.75][c], cy = [0.25, 0.75, 0.75, 0.25][c];
        const s = [0.015, 0.010, 0.025, 0.008][c];
        x = cx + s * gauss(); y = cy + s * gauss();
    } else if (scene === 'hardedge') {
        if (i % 8 === 0) { x = 0.5001 + 0.45 * rnd(); y = rnd(); }
        else { x = 0.35 + 0.1499 * rnd(); y = 0.35 + 0.3 * rnd(); }
    } else if (scene === 'single') { x = 0.5 + 1e-3 * gauss(); y = 0.5 + 1e-3 * gauss(); }
    else throw new Error('unknown scene ' + scene);
    pos[i * 4] = Math.min(0.999999, Math.max(0.000001, x));
    pos[i * 4 + 1] = Math.min(0.999999, Math.max(0.000001, y));
    pos[i * 4 + 2] = 0; pos[i * 4 + 3] = 0;
}

// Stub FunnelTable: shape-compatible (keys/values/geom), Map-backed.
class FunnelTable {
    constructor(capacity) {
        this._map = new Map();
        const total = Math.max(16, capacity);
        this.keys = new Uint32Array(total).fill(0xFFFFFFFF);
        this.values = new Uint32Array(total);
        this.geom = { packed: new Uint32Array(16), totalSize: total };
    }
    insert(key, value) { this._map.set(key >>> 0, value); return 0; }
    probe(key) { return this._map.has(key >>> 0) ? this._map.get(key >>> 0) : -1; }
}

const initialPositions = pos;
const jsModule = new Function(
    'N', 'initialPositions', 'FunnelTable', 'p2pBudgetWarned', 'adaptiveNodeHashWarned',
    'useAdaptiveNodeHash', 'p2pBudgetOverride', 'adaptiveLeafTargetOverride', 'adaptiveDepthOverride',
    source + '\nreturn { buildAdaptiveMetadata, computeAdaptiveDepth, computeAdaptiveLeafTarget, cellsTouch, buildFarOperatorTable };'
);
// useAdaptiveNodeHash=false: the page-level `let useAdaptiveNodeHash` (index.html)
// gates the FunnelTable occupied-node directory build inside buildAdaptiveMetadata;
// the Python emulator never consumes nodeHash and the emitted binary layout has
// no nodeHash section, so the harness stubs the flag OFF (same pattern as the
// p2pBudgetWarned / adaptiveNodeHashWarned / p2pBudgetOverride stubs here).
const api = jsModule(N, pos, FunnelTable, false, false, false, 0, 0, 0);
const md = api.buildAdaptiveMetadata(depth, pos);

// Binary layout (little-endian):
//   u32 magic 'ADPM', u32 N, u32 nodeCount, u32 numLevels, u32 depth,
//   u32 leafCount, u32 farEntryCount
//   f32[N*2] positions (x,y)
//   f32[nodeCount*4] nodeCenterSize
//   u32[nodeCount*2] nodeMeta (parent | terminal-flag interleaved)
//   u32[nodeCount*4] nodeChildren
//   u32[nodeCount*2] nodeParticleRange
//   u32[nodeCount*4] listOffsets
//   u32[nodeCount*4] listCounts
//   u32[listData.length] listData
//   u32[nodeCount] farStart            (materialized far CSR, round 13)
//   u32[nodeCount] farCount
//   u32[farEntryCount] farEntries      (srcIdx | row << 22)
//   f32[26950] farOps                  (per-(level,offset) M2L operator table)
//   u32[N] leafForParticle
//   u32[N] particleIndices
const header = new Uint32Array([0x4D504441, N, md.totalNodes, md.numLevels, md.depth, md.leafCount,
    md.farEntries.length]);
const pos2 = new Float32Array(N * 2);
for (let i = 0; i < N; i++) { pos2[i * 2] = pos[i * 4]; pos2[i * 2 + 1] = pos[i * 4 + 1]; }
const parts = [
    header.buffer, pos2.buffer, md.nodeCenterSize.buffer, md.nodeMeta.buffer,
    md.nodeChildren.buffer, md.nodeParticleRange.buffer,
    md.listOffsets.buffer, md.listCounts.buffer, md.listData.buffer,
    md.farStart.buffer, md.farCount.buffer, md.farEntries.buffer, md.farOps.buffer,
    md.leafForParticle.buffer, md.particleIndices.buffer,
];
const totalLen = parts.reduce((s, b) => s + b.byteLength, 0);
const out = new Uint8Array(totalLen);
let off = 0;
for (const b of parts) { out.set(new Uint8Array(b), off); off += b.byteLength; }
writeFileSync(outPath, out);
console.log(JSON.stringify({
    scene, N, depth: md.depth, leafTarget: md.leafTarget, nodes: md.totalNodes,
    leaves: md.leafCount, maxLeaf: md.maxLeafParticles, listEntries: md.listData.length,
    farEntries: md.farEntries.length,
}));
