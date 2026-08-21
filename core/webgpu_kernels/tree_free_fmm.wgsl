// Tree-Free Fast Multipole Method (FMM) WebGPU WGSL Compute Shader
// ====================================================================
// Runs across all WebGPU implementations (Chrome, Edge, Safari, Firefox, wgpu-py, Dawn)
// executing on AMD Radeon (Vulkan/DX12), NVIDIA CUDA, Intel Arc, and Apple Metal.
// Implements Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole Kernels (P2M, M2L, L2P, P2P).
//
// Cell lists are COUNTING-SORTED every frame in four passes (T-E1):
//   A) clear_cells:   cellCount[c] = 0
//   B) count_cells:   slot = cellIndex(pos); atomicAdd(cellCount[slot], 1)
//   C) scan_cells:    exclusive prefix sum -> cellStart[] (+cellCursor[])
//   D) scatter_cells: idx = atomicAdd(cellCursor[cell], 1);
//                     sortedIndex[idx] = particleId
// All P2P consumers then iterate CONTIGUOUS ranges of sortedIndex — no
// masked all-tile scan, no linked list, no pointer chasing, semi-coalesced
// reads. Particles of cell c occupy sortedIndex[cellStart[c] ..
// cellStart[c] + cellCount[c]] (CSR layout).

struct Particle2D {
    pos: vec2<f32>,
    q: f32,
    cluster_id: u32,
};

struct OutputForce2D {
    force: vec2<f32>,
    _pad: vec2<f32>,
};

struct ComplexNum {
    r: f32,
    i: f32,
};

struct SimulationParams {
    num_particles: u32,
    num_clusters: u32,
    order: u32,
    softening_sq: f32,
    cell_size: f32, // width of one uniform-grid leaf cell (world units)
    grid_dim: u32,  // cells per side of the uniform leaf grid
    _pad0: u32,
    grid_origin: vec2<f32>, // world-space origin of cell (0,0)
};

@group(0) @binding(0) var<storage, read> particles: array<Particle2D>;
@group(0) @binding(1) var<storage, read_write> potentials: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces: array<OutputForce2D>;
@group(0) @binding(3) var<storage, read> cluster_centers: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> cluster_local_coeffs: array<ComplexNum>; // [num_clusters * (order + 1)]
@group(0) @binding(5) var<uniform> params: SimulationParams;
// T-E1 counting-sort CSR cell-list buffers (dispatched before fmm_compute_main):
@group(0) @binding(6) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> cellCursor: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(9) var<storage, read_write> sortedIndex: array<u32>;

fn c_add(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r + b.r, a.i + b.i);
}

fn c_sub(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r - b.r, a.i - b.i);
}

fn c_mul(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

// ---- T-E1 counting-sort CSR cell-list passes ----
// Hash a world-space position to a uniform-grid leaf cell index.
fn cellIndex(pos: vec2<f32>) -> u32 {
    let dim = params.grid_dim;
    let inv_cell = 1.0 / params.cell_size;
    let cx = min(u32(clamp((pos.x - params.grid_origin.x) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
    let cy = min(u32(clamp((pos.y - params.grid_origin.y) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
    return cy * dim + cx;
}

@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let nc = params.grid_dim * params.grid_dim;
    if (id.x < nc) { atomicStore(&cellCount[id.x], 0u); }
}

@compute @workgroup_size(256)
fn count_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) { return; }
    atomicAdd(&cellCount[cellIndex(particles[id.x].pos)], 1u);
}

var<workgroup> scan_partial: array<u32, 256>;
var<workgroup> scan_base: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_cells(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Single-workgroup exclusive prefix sum over cellCount: each thread
    // sequentially scans one CONTIGUOUS chunk of cells (chunk-local
    // exclusive prefix + chunk sum), thread 0 scans the 256 chunk sums,
    // then every thread offsets its chunk by the totals of all earlier
    // chunks. Writes cellStart[c] (exclusive prefix) and initializes
    // cellCursor[c] = cellStart[c] for the scatter pass.
    let tid = lid.x;
    let nc = params.grid_dim * params.grid_dim;
    let chunk = (nc + 255u) / 256u;
    let lo = tid * chunk;
    let hi = min(lo + chunk, nc);
    var run: u32 = 0u;
    var i = lo;
    loop {
        if (i >= hi) { break; }
        let c = atomicLoad(&cellCount[i]);
        cellStart[i] = run;
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
        let s = cellStart[j] + base;
        cellStart[j] = s;
        atomicStore(&cellCursor[j], s);
        j = j + 1u;
    }
}

@compute @workgroup_size(256)
fn scatter_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) { return; }
    let cell = cellIndex(particles[id.x].pos);
    let slot = atomicAdd(&cellCursor[cell], 1u);
    sortedIndex[slot] = id.x;
}

@compute @workgroup_size(128, 1, 1)
fn fmm_compute_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let n = params.num_particles;
    let p_order = params.order;

    var my_p: Particle2D;
    if (gid < n) {
        my_p = particles[gid];
    } else {
        my_p = Particle2D(vec2<f32>(0.0, 0.0), 0.0, 0u);
    }

    var acc_phi: f32 = 0.0;
    var acc_f: vec2<f32> = vec2<f32>(0.0, 0.0);

    // 1. Far-field CGR88 L2P Local Expansion Evaluation
    if (gid < n) {
        let cid = my_p.cluster_id;
        let c_center = cluster_centers[cid];
        let dz = ComplexNum(my_p.pos.x - c_center.x, my_p.pos.y - c_center.y);
        var dz_k = ComplexNum(1.0, 0.0);

        let l0 = cluster_local_coeffs[cid * (p_order + 1u) + 0u];
        var pot_comp = l0;
        var deriv_comp = ComplexNum(0.0, 0.0);

        for (var l = 1u; l <= p_order; l = l + 1u) {
            let coeff = cluster_local_coeffs[cid * (p_order + 1u) + l];
            let term_deriv = c_mul(ComplexNum(f32(l), 0.0), c_mul(coeff, dz_k));
            deriv_comp = c_add(deriv_comp, term_deriv);
            dz_k = c_mul(dz_k, dz);
            pot_comp = c_add(pot_comp, c_mul(coeff, dz_k));
        }

        acc_phi = acc_phi + pot_comp.r;
        acc_f = acc_f + vec2<f32>(-deriv_comp.r, deriv_comp.i);
    }

    // 2. Near-field Direct P2P via CSR cell lists (T-E1)
    // Each particle iterates the 3x3 neighborhood of its leaf cell using
    // the counting-sort CSR ranges built by clear/count/scan/scatter_cells.
    // Particles in sortedIndex[cellStart[c] .. cellStart[c]+cellCount[c]]
    // are read directly — no masked all-tile scan, O(N * neighbors) work.
    if (gid < n) {
        let dim = params.grid_dim;
        let inv_cell = 1.0 / params.cell_size;
        let my_cx = min(u32(clamp((my_p.pos.x - params.grid_origin.x) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);
        let my_cy = min(u32(clamp((my_p.pos.y - params.grid_origin.y) * inv_cell, 0.0, f32(dim) - 1.0)), dim - 1u);

        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nx = i32(my_cx) + dx;
                let ny = i32(my_cy) + dy;
                if (nx < 0 || nx >= i32(dim) || ny < 0 || ny >= i32(dim)) { continue; }
                let cell = u32(ny) * dim + u32(nx);
                let start = cellStart[cell];
                let cnt = atomicLoad(&cellCount[cell]);
                for (var k: u32 = 0u; k < cnt; k = k + 1u) {
                    let j = sortedIndex[start + k];
                    if (j != gid) {
                        let pj = particles[j];
                        let diff = my_p.pos - pj.pos;
                        let r_sq = dot(diff, diff) + params.softening_sq;
                        let inv_r = inverseSqrt(r_sq);
                        let inv_r2 = inv_r * inv_r;
                        acc_phi = acc_phi + pj.q * 0.5 * log(r_sq);
                        acc_f = acc_f - pj.q * diff * inv_r2;
                    }
                }
            }
        }
    }

    if (gid < n) {
        potentials[gid] = acc_phi;
        forces[gid].force = acc_f;
    }
}
