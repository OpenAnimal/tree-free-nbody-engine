// Flat adaptive CGR88 WebGPU kernel interface.
// Metadata (nodes, terminal leaves, Lists 1-4) is supplied by
// core.adaptive_gpu_metadata. GPU passes evaluate the supplied schedule.
//
// Coefficients use three vec4 slots per node:
// slot 0 = (c0.re,c0.im,c1.re,c1.im)
// slot 1 = (c2.re,c2.im,c3.re,c3.im)
// slot 2 = (c4.re,c4.im,unused,unused)
//
// Full CGR88 pipeline:
//   1. clear  – zero multipoles and locals
//   2. p2m    – P2M at terminal leaves
//   3. m2m    – M2M upward pass (dispatched deepest-to-shallowest, level barriers)
//   4. l2l    – L2L downward pass (dispatched shallowest-to-deepest, level barriers)
//   5. m2l    – M2L for List 2 + P2L for List 4 (accumulates on top of L2L)
//   6. l2p    – L2P + List 3 M2P + List 1 P2P at particles

struct Particle { pos: vec2<f32>, vel: vec2<f32> };
struct FmmParams {
    numParticles: u32,
    nodeCount: u32,
    expansionOrder: u32,
    levelCount: u32,
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
@group(0) @binding(12) var<storage, read> nodeParent: array<u32>;
@group(0) @binding(13) var<storage, read> nodeChildren: array<vec4<u32>>;
@group(0) @binding(14) var<storage, read> nodeFlags: array<u32>;

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
fn readc(buf: ptr<storage, array<vec4<f32>>, read_write>, node: u32, k: u32) -> vec2<f32> {
    let b = node * 3u;
    if (k == 0u) { return (*buf)[b].xy; }
    if (k == 1u) { return (*buf)[b].zw; }
    if (k == 2u) { return (*buf)[b+1u].xy; }
    if (k == 3u) { return (*buf)[b+1u].zw; }
    return (*buf)[b+2u].xy;
}
fn writec(buf: ptr<storage, array<vec4<f32>>, read_write>, node: u32, k: u32, z: vec2<f32>) {
    let b = node * 3u;
    if (k == 0u) {
        let v = (*buf)[b];
        (*buf)[b] = vec4<f32>(z.x, z.y, v.z, v.w);
    } else if (k == 1u) {
        let v = (*buf)[b];
        (*buf)[b] = vec4<f32>(v.x, v.y, z.x, z.y);
    } else if (k == 2u) {
        let v = (*buf)[b+1u];
        (*buf)[b+1u] = vec4<f32>(z.x, z.y, v.z, v.w);
    } else if (k == 3u) {
        let v = (*buf)[b+1u];
        (*buf)[b+1u] = vec4<f32>(v.x, v.y, z.x, z.y);
    } else {
        let v = (*buf)[b+2u];
        (*buf)[b+2u] = vec4<f32>(z.x, z.y, v.z, v.w);
    }
}
fn isTerminal(node: u32) -> bool {
    return (nodeFlags[node] & FLAG_TERMINAL) != 0u;
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.nodeCount * 3u) { return; }
    multipoles[id.x] = vec4<f32>(0.0);
    locals[id.x] = vec4<f32>(0.0);
}

// P2M for terminal nodes. Source charge is one because the browser particle
// representation has no independent charge channel; this can be replaced by a
// charge buffer without changing the expansion operators.
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
        coeff[0].x += 1.0;
        for (var k = 1u; k <= 4u; k = k + 1u) {
            power = cmul(power, dz);
            if (k <= params.expansionOrder) { coeff[k] -= power / f32(k); }
        }
    }
    for (var k = 0u; k <= 4u; k = k + 1u) { writec(&multipoles, node, k, coeff[k]); }
}

// M2M upward pass: for each non-terminal node, accumulate multipoles from children.
// Dispatched level-by-level from deepest to shallowest with compute pass barriers.
@compute @workgroup_size(64)
fn m2m(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
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
                acc += readc(&multipoles, child, 0u);
            } else {
                // M2M translation: b_l = -a_0 * delta^l / l + sum_{j=1}^{l} a_j * C(l-1,j-1) * delta^(l-j)
                let delta = childCenter - center;
                var deltaPower = vec2<f32>(1.0, 0.0);
                for (var n = 0u; n < k; n = n + 1u) { deltaPower = cmul(deltaPower, delta); }
                let a0 = readc(&multipoles, child, 0u);
                acc -= a0 * deltaPower / f32(k);
                for (var j = 1u; j <= k; j = j + 1u) {
                    // C(k-1, j-1) computed iteratively
                    var bj = 1.0;
                    for (var b = 1u; b <= (j - 1u); b = b + 1u) { bj *= f32(k - b) / f32(b); }
                    var dp = vec2<f32>(1.0, 0.0);
                    for (var n = 0u; n < (k - j); n = n + 1u) { dp = cmul(dp, delta); }
                    acc += readc(&multipoles, child, j) * bj * dp;
                }
            }
        }
        writec(&multipoles, node, k, acc);
    }
}

// L2L downward pass: for each node with a parent, shift parent's local expansion
// to this node and accumulate. Dispatched level-by-level from level 1 to max.
@compute @workgroup_size(64)
fn l2l(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if (node >= params.nodeCount) { return; }
    let parent = nodeParent[node];
    if (parent == INVALID) { return; }
    let center = nodeCenterSize[node].xy;
    let parentCenter = nodeCenterSize[parent].xy;
    let delta = center - parentCenter;
    // d_l = sum_{k=l}^{p} c_k * C(k,l) * delta^(k-l)
    for (var l = 0u; l <= 4u; l = l + 1u) {
        if (l > params.expansionOrder) { break; }
        var acc = vec2<f32>(0.0);
        for (var k = l; k <= 4u; k = k + 1u) {
            if (k > params.expansionOrder) { break; }
            var binom = 1.0;
            for (var b = 1u; b <= l; b = b + 1u) { binom *= f32(k - b + 1u) / f32(b); }
            var dp = vec2<f32>(1.0, 0.0);
            for (var n = 0u; n < (k - l); n = n + 1u) { dp = cmul(dp, delta); }
            acc += readc(&locals, parent, k) * binom * dp;
        }
        // Accumulate on top of existing locals (which may have been zeroed by clear)
        let existing = readc(&locals, node, l);
        writec(&locals, node, l, existing + acc);
    }
}

// M2L for List 2 + P2L for List 4. Accumulates on top of L2L-shifted locals.
@compute @workgroup_size(64)
fn m2l(@builtin(global_invocation_id) id: vec3<u32>) {
    let targetNode = id.x;
    if (targetNode >= params.nodeCount) { return; }
    let targetCenter = nodeCenterSize[targetNode].xy;

    // List 2: M2L
    let l2Offset = listOffsets[targetNode].y;
    let l2Count = listCounts[targetNode].y;
    for (var q = 0u; q < l2Count; q = q + 1u) {
        let source = listData[l2Offset + q];
        let delta = targetCenter - nodeCenterSize[source].xy;
        for (var l = 0u; l <= 4u; l = l + 1u) {
            if (l > params.expansionOrder) { break; }
            var local = vec2<f32>(0.0);
            let a0 = readc(&multipoles, source, 0u);
            if (l == 0u) {
                local += cmul(a0, clog(delta));
                for (var k = 1u; k <= 4u; k = k + 1u) {
                    if (k > params.expansionOrder) { break; }
                    var power = vec2<f32>(1.0, 0.0);
                    for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
                    local += cdiv(readc(&multipoles, source, k), power);
                }
            } else {
                var deltaPower = vec2<f32>(1.0, 0.0);
                for (var n = 0u; n < l; n = n + 1u) { deltaPower = cmul(deltaPower, delta); }
                // a0 term uses (-1)^(l-1), ak term uses (-1)^l
                let sign_a0 = select(1.0, -1.0, (l & 1u) == 0u);  // (-1)^(l-1)
                let sign_ak = -sign_a0;                            // (-1)^l
                local += cdiv(a0 * (sign_a0 / f32(l)), deltaPower);
                for (var k = 1u; k <= 4u; k = k + 1u) {
                    if (k > params.expansionOrder) { break; }
                    var power = deltaPower;
                    for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
                    var binom = 1.0;
                    for (var b = 1u; b <= l; b = b + 1u) { binom *= f32(k + b - 1u) / f32(b); }
                    local += cdiv(readc(&multipoles, source, k) * (sign_ak * binom), power);
                }
            }
            // Accumulate
            let existing = readc(&locals, targetNode, l);
            writec(&locals, targetNode, l, existing + local);
        }
    }

    // List 4: P2L (particles in distant large leaves -> local expansion)
    let l4Offset = listOffsets[targetNode].w;
    let l4Count = listCounts[targetNode].w;
    for (var q = 0u; q < l4Count; q = q + 1u) {
        let sourceNode = listData[l4Offset + q];
        let range = nodeParticleRange[sourceNode];
        for (var n = 0u; n < range.y; n = n + 1u) {
            let pIdx = particleIndices[range.x + n];
            let d = targetCenter - particles[pIdx].pos;
            // c_0 += q * log(d), c_l += q * (-1)^(l-1) / (l * d^l)
            let existing0 = readc(&locals, targetNode, 0u);
            writec(&locals, targetNode, 0u, existing0 + clog(d));
            for (var l = 1u; l <= 4u; l = l + 1u) {
                if (l > params.expansionOrder) { break; }
                let sign = select(1.0, -1.0, (l & 1u) == 0u);
                var dp = vec2<f32>(1.0, 0.0);
                for (var n2 = 0u; n2 < l; n2 = n2 + 1u) { dp = cmul(dp, d); }
                let existing = readc(&locals, targetNode, l);
                writec(&locals, targetNode, l, existing + cdiv(vec2<f32>(sign / f32(l), 0.0), dp));
            }
        }
    }
}

// L2P plus List 1 P2P and List 3 M2P direct particle/multipole evaluation.
@compute @workgroup_size(256)
fn l2p(@builtin(global_invocation_id) id: vec3<u32>) {
    let particle = id.x;
    if (particle >= params.numParticles) { return; }
    let targetNode = leafForParticle[particle];
    let center = nodeCenterSize[targetNode].xy;
    let dz = particles[particle].pos - center;
    var potential = 0.0;
    var derivative = vec2<f32>(0.0);

    // L2P: evaluate local expansion
    for (var l = 0u; l <= 4u; l = l + 1u) {
        if (l > params.expansionOrder) { break; }
        let c = readc(&locals, targetNode, l);
        var power = vec2<f32>(1.0, 0.0);
        for (var n = 0u; n < l; n = n + 1u) { power = cmul(power, dz); }
        potential += dot(c, power);
        if (l > 0u) {
            var derivativePower = vec2<f32>(1.0, 0.0);
            for (var n = 1u; n < l; n = n + 1u) { derivativePower = cmul(derivativePower, dz); }
            derivative += cmul(c * f32(l), derivativePower);
        }
    }

    // List 3: direct M2P from smaller well-separated descendants.
    let list3Offset = listOffsets[targetNode].z;
    let list3Count = listCounts[targetNode].z;
    for (var q = 0u; q < list3Count; q = q + 1u) {
        let source = listData[list3Offset + q];
        let delta = particles[particle].pos - nodeCenterSize[source].xy;
        let a0 = readc(&multipoles, source, 0u);
        let inv = cdiv(vec2<f32>(1.0, 0.0), delta);
        potential += dot(a0, clog(delta));
        derivative += cmul(a0, inv);
        for (var k = 1u; k <= 4u; k = k + 1u) {
            if (k > params.expansionOrder) { break; }
            var power = vec2<f32>(1.0, 0.0);
            for (var n = 0u; n < k; n = n + 1u) { power = cmul(power, delta); }
            let ak = readc(&multipoles, source, k);
            potential += dot(ak, cdiv(vec2<f32>(1.0, 0.0), power));
            derivative -= f32(k) * cmul(ak, cdiv(vec2<f32>(1.0, 0.0), cmul(power, delta)));
        }
    }

    // List 1: direct near-field particle interactions.
    // NOTE: In the browser visualization path, near-field P2P is handled by the
    // main compute shader's budget-limited loop. This kernel outputs far-field only.
    fields[particle] = vec4<f32>(-derivative.x, derivative.y, potential, 0.0);
}
