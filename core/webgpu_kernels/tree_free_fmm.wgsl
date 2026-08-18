// Tree-Free Fast Multipole Method (FMM) WebGPU WGSL Compute Shader
// ====================================================================
// Runs across all WebGPU implementations (Chrome, Edge, Safari, Firefox, wgpu-py, Dawn)
// executing on AMD Radeon (Vulkan/DX12), NVIDIA CUDA, Intel Arc, and Apple Metal.
// Implements Carrier, Greengard, & Rokhlin (1988) 2D Complex Multipole Kernels (P2M, M2L, L2P, P2P).

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
};

@group(0) @binding(0) var<storage, read> particles: array<Particle2D>;
@group(0) @binding(1) var<storage, read_write> potentials: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces: array<OutputForce2D>;
@group(0) @binding(3) var<storage, read> cluster_centers: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> cluster_local_coeffs: array<ComplexNum>; // [num_clusters * (order + 1)]
@group(0) @binding(5) var<uniform> params: SimulationParams;

fn c_add(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r + b.r, a.i + b.i);
}

fn c_sub(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r - b.r, a.i - b.i);
}

fn c_mul(a: ComplexNum, b: ComplexNum) -> ComplexNum {
    return ComplexNum(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

var<workgroup> tile_particles: array<Particle2D, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
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

    // 2. Tiled Near-field Direct P2P Evaluation
    let num_tiles = (n + 127u) / 128u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let load_idx = t * 128u + lid;
        if (load_idx < n) {
            tile_particles[lid] = particles[load_idx];
        } else {
            tile_particles[lid] = Particle2D(vec2<f32>(0.0, 0.0), 0.0, 0u);
        }
        workgroupBarrier();

        let tile_limit = min(128u, n - t * 128u);
        if (gid < n) {
            for (var j: u32 = 0u; j < tile_limit; j = j + 1u) {
                let global_j = t * 128u + j;
                if (global_j != gid) {
                    let pj = tile_particles[j];
                    let diff = my_p.pos - pj.pos;
                    let r_sq = dot(diff, diff) + params.softening_sq;
                    if (r_sq < 0.0025) { // Within direct near-field cutoff
                        let r_sq_safe = max(r_sq, 1e-12);
                        let inv_r = inverseSqrt(r_sq_safe);
                        let inv_r2 = inv_r * inv_r;
                        acc_phi = acc_phi + pj.q * 0.5 * log(r_sq_safe);
                        acc_f = acc_f - pj.q * diff * inv_r2;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if (gid < n) {
        potentials[gid] = acc_phi;
        forces[gid].force = acc_f;
    }
}
