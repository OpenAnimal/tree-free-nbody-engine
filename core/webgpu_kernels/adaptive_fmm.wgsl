// Flat adaptive FMM WebGPU kernel — full pipeline with M2M, L2L, M2L, P2L.
//
// COVERAGE CONTRACT (read before feeding adaptive metadata): the near-field
// P2P below iterates a UNIFORM 3x3 grid-overlay neighborhood, NOT the
// quadtree's mixed-size List 1. Every (i, j) pair is counted exactly once
// ONLY when the overlay grid is at least as fine as the tree's DEEPEST leaf
// AND every leaf sits at that deepest level (a uniform-depth tree, where
// leaf == overlay cell). With a genuinely occupancy-adaptive tree, two
// particles inside one COARSE leaf can sit more than 3 overlay cells apart:
// the tree excludes that pair from M2L (same leaf) and the overlay P2P never
// sees it — the pair is silently missed. The occupancy-adaptive reference
// (mixed-depth List-1 P2P with Lists 1-4) is the inline kernel in
// index.html, cross-validated by tools/validate_adaptive_js.py.
//
// Cell lists for the near-field P2P are COUNTING-SORTED every frame in four
// passes (T-E1), overlaying a uniform grid on the simulation domain:
//   A) clear_cells:   cellCount[c] = 0
//   B) count_cells:   slot = p2pCellIndex(pos); atomicAdd(cellCount[slot], 1)
//   C) scan_cells:    exclusive prefix sum -> cellStart[] (+cellCursor[])
//   D) scatter_cells: idx = atomicAdd(cellCursor[cell], 1);
//                     sortedIndex[idx] = particleId
// The l2p P2P section then iterates CONTIGUOUS ranges of sortedIndex over the
// 3x3 neighborhood — no precomputed List-1 budgeted sampling, no pointer
// chasing. Particles of cell c occupy sortedIndex[cellStart[c] ..
// cellStart[c] + cellCount[c]] (CSR layout). The far-field multipole passes
// (p2m/m2m/l2l/m2l) and the L2P local-expansion math are unchanged.
struct Particle { pos: vec2<f32>, vel: vec2<f32> };
// levelBase is the flat node offset of the level being dispatched. Nodes are
// laid out with levelStart[l] = sum_{i<l} 4^i, so a dispatch of levelCount[l]
// threads must address nodes as levelBase + id.x — plain id.x would cover the
// WRONG level range (leaves missing M2L/L2L, other nodes accumulated 2-4x).
struct FmmParams {
    numParticles: u32,
    nodeCount: u32,
    expansionOrder: u32,
    levelCount: u32,
    levelBase: u32,
    zeroNearP2P: u32,  // probe flag: 1 = skip near-field P2P (measure multipole path only)
    farEntryCount: u32, // materialized far CSR entry total (farOps tail base)
    _pad2: u32,
};

// T-E1: uniform-grid parameters for the P2P counting-sort CSR cell list.
// The grid is overlaid on the simulation domain at a resolution matching the
// finest leaf cell size; it is independent of the adaptive quadtree used for
// the far-field multipole passes.
struct GridParams {
    gridDim: u32,       // cells per side of the uniform P2P grid
    _pad: u32,
    gridOrigin: vec2<f32>, // world-space origin of cell (0,0)
    cellSize: f32,      // width of one P2P grid cell (world units)
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> nodeCenterSize: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> nodeParticleRange: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> leafForParticle: array<u32>;
@group(0) @binding(4) var<storage, read> particleIndices: array<u32>;
@group(0) @binding(5) var<storage, read> listOffsets: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> listCounts: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> listData: array<u32>;
@group(0) @binding(8) var<storage, read_write> multipoles: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> locals: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> fields: array<vec4<f32>>;
@group(0) @binding(11) var<uniform> params: FmmParams;
// nodeMeta packs parent (.x) and flags (.y) into one vec2<u32> buffer: with
// nodeParent/nodeFlags as separate arrays plus the T-E1 cell-list buffers
// the module would declare 19 storage buffers, over the
// maxStorageBuffersPerShaderStage default of 16 (the same overflow disabled
// the demo's adaptive shader when its hash buffers were added).
@group(0) @binding(12) var<storage, read> nodeMeta: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> nodeChildren: array<vec4<u32>>;
// Materialized per-target far-field CSR + per-(level, offset) M2L operator
// table packed into ONE u32 buffer (see the farCSR accessors below the
// coefficient helpers): [per-node start/count header | packed entries |
// farOps f32 bits]. Built per metadata refresh (round 13).
@group(0) @binding(14) var<storage, read> farCSR: array<u32>;
@group(0) @binding(15) var<storage, read> charges: array<f32>;  // T-D7: per-particle charge
// T-E1 counting-sort CSR cell-list buffers (dispatched before l2p).
// cellArrays packs [0..nc) count | [nc..2nc) cursor | [2nc..3nc) start into
// ONE buffer — same packing as the demo's fixed-grid cellArrays buffer —
// keeping the module at 16 storage bindings.
@group(0) @binding(16) var<uniform> gridParams: GridParams;
@group(0) @binding(17) var<storage, read_write> cellArrays: array<atomic<u32>>;
@group(0) @binding(20) var<storage, read_write> sortedIndex: array<u32>;

fn cellCountAt(c: u32) -> u32 { return atomicLoad(&cellArrays[c]); }
fn cellStartAt(c: u32, nc: u32) -> u32 { return atomicLoad(&cellArrays[2u * nc + c]); }

const INVALID: u32 = 0xFFFFFFFFu;
const FLAG_TERMINAL: u32 = 1u;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = max(dot(b,b), 1e-20);
    return vec2<f32>((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d);
}
fn clog(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5 * log(max(dot(a,a), 1e-20)), atan2(a.y, a.x));
}
// Complex-coefficient accessors for the packed per-node coefficient slots
// (3 vec4 = 5 complex per node: k = 0..4). `which` selects the module-scope
// buffer: 0 = multipoles, 1 = locals. (Passing read_write storage pointers
// as function parameters is rejected by naga/wgpu-native, so the file
// kernel selects the buffer by index instead — the browser demo's inline
// copy keeps the pointer-parameter form, which Tint accepts.)
fn readc(which: u32, node: u32, k: u32) -> vec2<f32> {
    let b = node * 3u;
    if (which == 0u) {
        if (k == 0u) { return multipoles[b].xy; }
        if (k == 1u) { return multipoles[b].zw; }
        if (k == 2u) { return multipoles[b+1u].xy; }
        if (k == 3u) { return multipoles[b+1u].zw; }
        return multipoles[b+2u].xy;
    }
    if (k == 0u) { return locals[b].xy; }
    if (k == 1u) { return locals[b].zw; }
    if (k == 2u) { return locals[b+1u].xy; }
    if (k == 3u) { return locals[b+1u].zw; }
    return locals[b+2u].xy;
}
fn writec(which: u32, node: u32, k: u32, z: vec2<f32>) {
    let b = node * 3u;
    if (which == 0u) {
        if (k == 0u) {
            let v = multipoles[b];
            multipoles[b] = vec4<f32>(z.x, z.y, v.z, v.w);
        } else if (k == 1u) {
            let v = multipoles[b];
            multipoles[b] = vec4<f32>(v.x, v.y, z.x, z.y);
        } else if (k == 2u) {
            let v = multipoles[b+1u];
            multipoles[b+1u] = vec4<f32>(z.x, z.y, v.z, v.w);
        } else if (k == 3u) {
            let v = multipoles[b+1u];
            multipoles[b+1u] = vec4<f32>(v.x, v.y, z.x, z.y);
        } else {
            let v = multipoles[b+2u];
            multipoles[b+2u] = vec4<f32>(z.x, z.y, v.z, v.w);
        }
    } else {
        if (k == 0u) {
            let v = locals[b];
            locals[b] = vec4<f32>(z.x, z.y, v.z, v.w);
        } else if (k == 1u) {
            let v = locals[b];
            locals[b] = vec4<f32>(v.x, v.y, z.x, z.y);
        } else if (k == 2u) {
            let v = locals[b+1u];
            locals[b+1u] = vec4<f32>(z.x, z.y, v.z, v.w);
        } else if (k == 3u) {
            let v = locals[b+1u];
            locals[b+1u] = vec4<f32>(v.x, v.y, z.x, z.y);
        } else {
            let v = locals[b+2u];
            locals[b+2u] = vec4<f32>(z.x, z.y, v.z, v.w);
        }
    }
}
fn isTerminal(node: u32) -> bool {
    return (nodeMeta[node].y & FLAG_TERMINAL) != 0u;
}

// ---- Materialized far-field CSR accessors + helpers (round 13) ----
// farCSR layout (one u32 storage buffer, binding 14):
//   [0 .. 2*nodeCount)              per-node header: start (2n) | count (2n+1)
//   [2n .. 2n + farEntryCount)      packed entries: source node index (low
//                                   22 bits) | operator row index (top 10)
//   [2n + farEntryCount ..)         farOps: per-(level, offset) M2L operator
//                                   table, 50 f32 (25 complex) per row, stored
//                                   as bitcast u32 words.
fn farStartOf(node: u32) -> u32 { return farCSR[2u * node]; }
fn farCountOf(node: u32) -> u32 { return farCSR[2u * node + 1u]; }
fn farEntryAt(i: u32) -> u32 { return farCSR[2u * params.nodeCount + i]; }
fn farOpsBase() -> u32 { return 2u * params.nodeCount + params.farEntryCount; }
fn farOpAt(row: u32, m: u32, k: u32) -> vec2<f32> {
    let base = farOpsBase() + row * 50u + (m * 5u + k) * 2u;
    return vec2<f32>(bitcast<f32>(farCSR[base]), bitcast<f32>(farCSR[base + 1u]));
}
fn farZero() -> array<vec2<f32>, 5> {
    return array<vec2<f32>, 5>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
}
fn farAdd(a: array<vec2<f32>, 5>, b: array<vec2<f32>, 5>) -> array<vec2<f32>, 5> {
    var out = farZero();
    for (var k = 0u; k < 5u; k = k + 1u) { out[k] = a[k] + b[k]; }
    return out;
}
// One-shot L2L recentering of a 5-coefficient local expansion by d: the
// exact polynomial shift the per-level l2l kernel applies one level at a
// time; composing exact shifts equals the single long shift, so gathering
// each ancestor level's contribution and shifting it straight to the leaf
// reproduces the chain result (same operator math as l2l below).
fn farShift(c: array<vec2<f32>, 5>, d: vec2<f32>) -> array<vec2<f32>, 5> {
    var out = farZero();
    var dpow = farZero();
    dpow[0] = vec2<f32>(1.0, 0.0);
    for (var j = 1u; j <= 4u; j = j + 1u) { dpow[j] = cmul(dpow[j - 1u], d); }
    for (var m = 0u; m <= 4u; m = m + 1u) {
        var acc = vec2<f32>(0.0);
        for (var k = m; k <= 4u; k = k + 1u) {
            var binom = 1.0;
            for (var b = 1u; b <= m; b = b + 1u) { binom *= f32(k - b + 1u) / f32(b); }
            acc += cmul(c[k], dpow[k - m]) * binom;
        }
        out[m] = acc;
    }
    return out;
}

// ---- T-E1 counting-sort CSR cell-list passes (uniform grid overlay) ----
// Hash a world-space position to a uniform-grid P2P cell index.
fn p2pCellIndex(pos: vec2<f32>) -> u32 {
    let dim = gridParams.gridDim;
    let inv_cell = 1.0 / gridParams.cellSize;
    let cx = min(u32(clamp((pos.x - gridParams.gridOrigin.x) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
    let cy = min(u32(clamp((pos.y - gridParams.gridOrigin.y) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
    return cy * dim + cx;
}

@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let nc = gridParams.gridDim * gridParams.gridDim;
    if (id.x < nc) { atomicStore(&cellArrays[id.x], 0u); }
}

@compute @workgroup_size(256)
fn count_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.numParticles) { return; }
    atomicAdd(&cellArrays[p2pCellIndex(particles[id.x].pos)], 1u);
}

var<workgroup> scan_partial: array<u32, 256>;
var<workgroup> scan_base: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_cells(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Single-workgroup exclusive prefix sum over the packed cell counts:
    // each thread sequentially scans one CONTIGUOUS chunk of cells
    // (chunk-local exclusive prefix + chunk sum), thread 0 scans the 256
    // chunk sums, then every thread offsets its chunk by the totals of all
    // earlier chunks. Writes cellStart[c] (exclusive prefix, at
    // cellArrays[2*nc + c]) and initializes cellCursor[c] = cellStart[c]
    // (at cellArrays[nc + c]) for the scatter pass.
    let tid = lid.x;
    let nc = gridParams.gridDim * gridParams.gridDim;
    let chunk = (nc + 255u) / 256u;
    let lo = tid * chunk;
    let hi = min(lo + chunk, nc);
    var run: u32 = 0u;
    var i = lo;
    loop {
        if (i >= hi) { break; }
        let c = cellCountAt(i);
        atomicStore(&cellArrays[2u * nc + i], run);
        run = run + c;
        i = i + 1u;
    }
    scan_partial[tid] = run;
    workgroupBarrier();
    if (tid == 0u) {
        var acc: u32 = 0u;
        for (var t = 0u; t < 256u; t = t + 1u) {
            scan_base[t] = acc;
            acc = acc + scan_partial[t];
        }
    }
    workgroupBarrier();
    let base = scan_base[tid];
    var j = lo;
    loop {
        if (j >= hi) { break; }
        let s = cellStartAt(j, nc) + base;
        atomicStore(&cellArrays[2u * nc + j], s);
        atomicStore(&cellArrays[nc + j], s);
        j = j + 1u;
    }
}

@compute @workgroup_size(256)
fn scatter_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.numParticles) { return; }
    let nc = gridParams.gridDim * gridParams.gridDim;
    let cell = p2pCellIndex(particles[id.x].pos);
    let slot = atomicAdd(&cellArrays[nc + cell], 1u);
    sortedIndex[slot] = id.x;
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.nodeCount * 3u) { return; }
    multipoles[id.x] = vec4<f32>(0.0);
    locals[id.x] = vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn p2m(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if (node >= params.nodeCount) { return; }
    if (!isTerminal(node)) { return; }
    let range = nodeParticleRange[node];
    let center = nodeCenterSize[node].xy;
    var coeff = array<vec2<f32>, 5>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
    for (var n = 0u; n < range.y; n = n + 1u) {
        let particle = particleIndices[range.x + n];
        let dz = particles[particle].pos - center;
        var power = vec2<f32>(1.0, 0.0);
        coeff[0].x += charges[particle];  // T-D7: use per-particle charge
        for (var k = 1u; k <= 4u; k = k + 1u) {
            power = cmul(power, dz);
            // P18-2: the T-D7 per-particle charge fix only applied to the
            // monopole (k=0); the higher-order coefficients (k>=1) were still
            // using unit charge, inconsistent with the Python reference
            // (core/adaptive_fmm.py:54 `coeffs[k] = -sum(charges * dz^k)/k`).
            if (k <= params.expansionOrder) { coeff[k] -= charges[particle] * power / f32(k); }
        }
    }
    for (var k = 0u; k <= 4u; k = k + 1u) { writec(0u, node, k, coeff[k]); }
}

@compute @workgroup_size(64)
fn m2m(@builtin(global_invocation_id) id: vec3<u32>) {
    // Upward pass, dispatched per level: node = levelBase + id.x so the
    // workgroup range covers exactly this level's nodes.
    let node = params.levelBase + id.x;
    if (node >= params.nodeCount) { return; }
    if (isTerminal(node)) { return; }
    let center = nodeCenterSize[node].xy;
    let children = nodeChildren[node];
    for (var k = 0u; k <= 4u; k = k + 1u) {
        var acc = vec2<f32>(0.0);
        for (var slot = 0u; slot < 4u; slot = slot + 1u) {
            let child = children[slot];
            if (child == INVALID) { continue; }
            let childCenter = nodeCenterSize[child].xy;
            if (k == 0u) {
                acc += readc(0u, child, 0u);
            } else {
                let delta = childCenter - center;
                var deltaPower = vec2<f32>(1.0, 0.0);
                for (var n = 0u; n < k; n = n + 1u) { deltaPower = cmul(deltaPower, delta); }
                let a0 = readc(0u, child, 0u);
                acc -= cmul(a0, deltaPower) / f32(k);
                for (var j = 1u; j <= k; j = j + 1u) {
                    var bj = 1.0;
                    for (var b = 1u; b <= (j - 1u); b = b + 1u) { bj *= f32(k - b) / f32(b); }
                    var dp = vec2<f32>(1.0, 0.0);
                    for (var n = 0u; n < (k - j); n = n + 1u) { dp = cmul(dp, delta); }
                    acc += cmul(readc(0u, child, j), dp) * bj;
                }
            }
        }
        writec(0u, node, k, acc);
    }
}

@compute @workgroup_size(64)
fn l2l(@builtin(global_invocation_id) id: vec3<u32>) {
    // Downward pass, dispatched per level with exact level ranges.
    let node = params.levelBase + id.x;
    if (node >= params.nodeCount) { return; }
    let parent = nodeMeta[node].x;
    if (parent == INVALID) { return; }
    let center = nodeCenterSize[node].xy;
    let parentCenter = nodeCenterSize[parent].xy;
    let delta = center - parentCenter;
    for (var l = 0u; l <= 4u; l = l + 1u) {
        if (l > params.expansionOrder) { break; }
        var acc = vec2<f32>(0.0);
        for (var k = l; k <= 4u; k = k + 1u) {
            if (k > params.expansionOrder) { break; }
            var binom = 1.0;
            for (var b = 1u; b <= l; b = b + 1u) { binom *= f32(k - b + 1u) / f32(b); }
            var dp = vec2<f32>(1.0, 0.0);
            for (var n = 0u; n < (k - l); n = n + 1u) { dp = cmul(dp, delta); }
            acc += cmul(readc(1u, parent, k), dp) * binom;
        }
        let existing = readc(1u, node, l);
        writec(1u, node, l, existing + acc);
    }
}

@compute @workgroup_size(64)
fn m2l(@builtin(global_invocation_id) id: vec3<u32>) {
    // M2L, dispatched per level with exact level ranges. Each node is
    // processed exactly once per frame (locals were cleared by the clear
    // pass), so the existing + accumulate semantics are correct.
    let targetNode = params.levelBase + id.x;
    if (targetNode >= params.nodeCount) { return; }
    let targetCenter = nodeCenterSize[targetNode].xy;

    let l2Offset = listOffsets[targetNode].y;
    let l2Count = listCounts[targetNode].y;
    for (var q = 0u; q < l2Count; q = q + 1u) {
        let source = listData[l2Offset + q];
        let delta = targetCenter - nodeCenterSize[source].xy;
        for (var l = 0u; l <= 4u; l = l + 1u) {
            if (l > params.expansionOrder) { break; }
            var local = vec2<f32>(0.0);
            let a0 = readc(0u, source, 0u);
            if (l == 0u) {
                local += cmul(a0, clog(delta));
                for (var k = 1u; k <= 4u; k = k + 1u) {
                    if (k > params.expansionOrder) { break; }
                    var power = vec2<f32>(1.0, 0.0);
                    for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
                    local += cdiv(readc(0u, source, k), power);
                }
            } else {
                var deltaPower = vec2<f32>(1.0, 0.0);
                for (var n = 0u; n < l; n = n + 1u) { deltaPower = cmul(deltaPower, delta); }
                let sign_a0 = select(1.0, -1.0, (l & 1u) == 0u);
                let sign_ak = -sign_a0;
                local += cdiv(a0 * (sign_a0 / f32(l)), deltaPower);
                for (var k = 1u; k <= 4u; k = k + 1u) {
                    if (k > params.expansionOrder) { break; }
                    var power = deltaPower;
                    for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
                    var binom = 1.0;
                    for (var b = 1u; b <= l; b = b + 1u) { binom *= f32(k + b - 1u) / f32(b); }
                    local += cdiv(readc(0u, source, k) * (sign_ak * binom), power);
                }
            }
            let existing = readc(1u, targetNode, l);
            writec(1u, targetNode, l, existing + local);
        }
    }

    let l4Offset = listOffsets[targetNode].w;
    let l4Count = listCounts[targetNode].w;
    for (var q = 0u; q < l4Count; q = q + 1u) {
        let sourceNode = listData[l4Offset + q];
        let range = nodeParticleRange[sourceNode];
        for (var n = 0u; n < range.y; n = n + 1u) {
            let pIdx = particleIndices[range.x + n];
            let d = targetCenter - particles[pIdx].pos;
            // P18-3: the P2L (List 4) translation was missing the per-particle
            // charge factor, inconsistent with the Python reference
            // (core/adaptive_fmm.py:149 `c[0] += q * log(d)`).
            let qj = charges[pIdx];
            let existing0 = readc(1u, targetNode, 0u);
            writec(1u, targetNode, 0u, existing0 + qj * clog(d));
            for (var l = 1u; l <= 4u; l = l + 1u) {
                if (l > params.expansionOrder) { break; }
                let sign = select(1.0, -1.0, (l & 1u) == 0u);
                var dp = vec2<f32>(1.0, 0.0);
                for (var n2 = 0u; n2 < l; n2 = n2 + 1u) { dp = cmul(dp, d); }
                let existing = readc(1u, targetNode, l);
                writec(1u, targetNode, l, existing + qj * cdiv(vec2<f32>(sign / f32(l), 0.0), dp));
            }
        }
    }
}

@compute @workgroup_size(64)
fn p2l(@builtin(global_invocation_id) id: vec3<u32>) {
    // List-4 P2L pass of the materialized far-field path (round 13): the
    // List-4 block of the per-level m2l kernel, dispatched once over ALL
    // nodes (P2L has no inter-level data dependency). Writes ONLY each
    // node's own P2L contribution into locals; far_gather then folds every
    // ancestor's part into each leaf with one-shot L2L shifts, so the P2L
    // work stays shared per (node, source leaf) pair exactly as before.
    if (id.x >= params.nodeCount) { return; }
    let targetNode = id.x;
    let targetCenter = nodeCenterSize[targetNode].xy;
    let l4Offset = listOffsets[targetNode].w;
    let l4Count = listCounts[targetNode].w;
    for (var q = 0u; q < l4Count; q = q + 1u) {
        let sourceNode = listData[l4Offset + q];
        let range = nodeParticleRange[sourceNode];
        if (range.y == 0u) { continue; }
        // Bound the serial P2L loop: a saturated max-depth leaf can hold
        // thousands of particles and this kernel runs one thread per target
        // node — unbounded, that is a collapse-triggered stall. Stride-sample
        // with a (target, source)-rotated offset and reweight by range.y/cnt.
        let cnt = min(range.y, 128u);
        let skip = (targetNode * 2654435761u + q * 40503u) % range.y;
        let w = f32(range.y) / f32(cnt);
        for (var n = 0u; n < cnt; n = n + 1u) {
            let pIdx = particleIndices[range.x + ((skip + n) % range.y)];
            let d = targetCenter - particles[pIdx].pos;
            let qj = charges[pIdx] * w;
            let existing0 = readc(1u, targetNode, 0u);
            writec(1u, targetNode, 0u, existing0 + qj * clog(d));
            for (var l = 1u; l <= 4u; l = l + 1u) {
                if (l > params.expansionOrder) { break; }
                let sign = select(1.0, -1.0, (l & 1u) == 0u);
                var dp = vec2<f32>(1.0, 0.0);
                for (var n2 = 0u; n2 < l; n2 = n2 + 1u) { dp = cmul(dp, d); }
                let existing = readc(1u, targetNode, l);
                writec(1u, targetNode, l, existing + qj * cdiv(vec2<f32>(sign / f32(l), 0.0), dp));
            }
        }
    }
}

@compute @workgroup_size(64)
fn far_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    // Materialized far-field evaluation (round 13): one thread per LEAF
    // gathers the leaf's CSR of List-2 M2L sources across ALL levels of its
    // ancestor chain, applying each source through the precomputed
    // per-(level, offset) operator row (two loads + a 5x5 complex matvec —
    // no clog/cdiv chains) and recentering each level's partial expansion
    // onto the leaf with one exact L2L shift per level run. This replaces
    // the per-level m2l (List-2 part) + l2l chain of the legacy path. The
    // ancestors' List-4 P2L locals (written by the p2l pass) are folded in
    // by the same shifts; a strict ancestor of a leaf is always internal,
    // so no thread reads a leaf slot another thread writes.
    if (id.x >= params.nodeCount) { return; }
    let t = id.x;
    if (!isTerminal(t)) { return; }
    let tCenter = nodeCenterSize[t].xy;
    let tLevel = u32(nodeCenterSize[t].w);
    // Ancestor node index + center per level (entry [tLevel] is t itself).
    var ancIdx: array<u32, 11>;
    var ancCenter: array<vec2<f32>, 11>;
    var a = t;
    for (var l = tLevel; l > 0u; l = l - 1u) {
        ancIdx[l] = a;
        ancCenter[l] = nodeCenterSize[a].xy;
        a = nodeMeta[a].x;
    }
    var acc = farZero();
    // CSR run accumulator: same-level entries accumulate about that
    // ancestor's center; on a level change the run is shifted to the leaf.
    var lvl = farZero();
    var runLevel = 0xFFFFFFFFu;
    let cnt = farCountOf(t);
    let start = farStartOf(t);
    for (var e = 0u; e < cnt; e = e + 1u) {
        let packed = farEntryAt(start + e);
        let src = packed & 0x3FFFFFu;
        let row = packed >> 22u;
        let l = row / 49u;
        if (l != runLevel) {
            if (runLevel != 0xFFFFFFFFu) {
                acc = farAdd(acc, farShift(lvl, tCenter - ancCenter[runLevel]));
            }
            lvl = farZero();
            runLevel = l;
        }
        var mom = farZero();
        for (var k = 0u; k <= params.expansionOrder; k = k + 1u) {
            mom[k] = readc(0u, src, k);
        }
        for (var m = 0u; m <= params.expansionOrder; m = m + 1u) {
            var v = vec2<f32>(0.0);
            for (var k = 0u; k <= params.expansionOrder; k = k + 1u) {
                v += cmul(farOpAt(row, m, k), mom[k]);
            }
            lvl[m] += v;
        }
    }
    if (runLevel != 0xFFFFFFFFu) {
        acc = farAdd(acc, farShift(lvl, tCenter - ancCenter[runLevel]));
    }
    // Fold the ancestors' P2L locals (List-4 parts written by p2l) with the
    // same one-shot shifts; index tLevel is t itself, picking up t's own
    // P2L entries with a zero shift.
    for (var l = 1u; l <= tLevel; l = l + 1u) {
        var pl = farZero();
        for (var k = 0u; k <= params.expansionOrder; k = k + 1u) {
            pl[k] = readc(1u, ancIdx[l], k);
        }
        acc = farAdd(acc, farShift(pl, tCenter - ancCenter[l]));
    }
    for (var k = 0u; k <= 4u; k = k + 1u) { writec(1u, t, k, acc[k]); }
}

@compute @workgroup_size(256)
fn l2p(@builtin(global_invocation_id) id: vec3<u32>) {
    let particle = id.x;
    if (particle >= params.numParticles) { return; }
    let targetNode = leafForParticle[particle];
    let center = nodeCenterSize[targetNode].xy;
    let dz = particles[particle].pos - center;
    var potential = 0.0;
    var derivative = vec2<f32>(0.0);

    for (var l = 0u; l <= 4u; l = l + 1u) {
        if (l > params.expansionOrder) { break; }
        let c = readc(1u, targetNode, l);
        var power = vec2<f32>(1.0, 0.0);
        for (var n = 0u; n < l; n = n + 1u) { power = cmul(power, dz); }
        // Potential is the REAL part of the complex product, not a
        // vector dot product.
        potential += cmul(c, power).x;
        if (l > 0u) {
            var derivativePower = vec2<f32>(1.0, 0.0);
            for (var n = 1u; n < l; n = n + 1u) { derivativePower = cmul(derivativePower, dz); }
            derivative += cmul(c * f32(l), derivativePower);
        }
    }

    let list3Offset = listOffsets[targetNode].z;
    let list3Count = listCounts[targetNode].z;
    for (var q = 0u; q < list3Count; q = q + 1u) {
        let source = listData[list3Offset + q];
        let delta = particles[particle].pos - nodeCenterSize[source].xy;
        let a0 = readc(0u, source, 0u);
        let inv = cdiv(vec2<f32>(1.0, 0.0), delta);
        potential += cmul(a0, clog(delta)).x;
        derivative += cmul(a0, inv);
        for (var k = 1u; k <= 4u; k = k + 1u) {
            if (k > params.expansionOrder) { break; }
            var power = vec2<f32>(1.0, 0.0);
            for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
            let ak = readc(0u, source, k);
            potential += cdiv(ak, power).x;
            derivative -= f32(k) * cmul(ak, cdiv(vec2<f32>(1.0, 0.0), cmul(power, delta)));
        }
    }

    // Near-field direct P2P via CSR cell lists (T-E1).
    // Each particle iterates the 3x3 neighborhood of its uniform-grid P2P
    // cell using the counting-sort CSR ranges built by
    // clear_cells/count_cells/scan_cells/scatter_cells. Particles in
    // sortedIndex[cellStart[c] .. cellStart[c]+cellCount[c]] are read
    // directly — no List-1 budgeted random sampling, full neighbor
    // coverage with contiguous (semi-coalesced) reads.
    // Uses per-particle charges[j] (binding 15) and the near-field
    // log-potential term, matching the FMM far-field softening.
    // When params.zeroNearP2P == 1 (probe mode), the P2P accumulation
    // is skipped so the probe measures the multipole (M2L/L2P) path
    // only, without conflating P2P with far-field truncation error.
    var p2pForce = vec2<f32>(0.0);
    if (params.zeroNearP2P == 0u) {
        let dim = gridParams.gridDim;
        let nc = dim * dim;
        let inv_cell = 1.0 / gridParams.cellSize;
        let myPos = particles[particle].pos;
        let my_cx = min(u32(clamp((myPos.x - gridParams.gridOrigin.x) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
        let my_cy = min(u32(clamp((myPos.y - gridParams.gridOrigin.y) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);

        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nx = i32(my_cx) + dx;
                let ny = i32(my_cy) + dy;
                if (nx < 0 || nx >= i32(dim) || ny < 0 || ny >= i32(dim)) { continue; }
                let cell = u32(ny) * dim + u32(nx);
                let start = cellStartAt(cell, nc);
                let cnt = cellCountAt(cell);
                for (var k = 0u; k < cnt; k = k + 1u) {
                    let j = sortedIndex[start + k];
                    if (j == particle) { continue; }
                    let d = myPos - particles[j].pos;
                    let r2 = dot(d, d) + 0.00004;
                    let inv2 = inverseSqrt(r2);
                    let qj = charges[j];
                    // Near-field log-potential + force with per-particle charge.
                    potential += qj * 0.5 * log(r2);
                    p2pForce -= d * (qj * inv2 * inv2);
                }
            }
        }
    }

    fields[particle] = vec4<f32>(-derivative.x + p2pForce.x, derivative.y + p2pForce.y, potential, 0.0);
}
