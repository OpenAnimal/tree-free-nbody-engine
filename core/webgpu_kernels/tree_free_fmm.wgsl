// Tree-Free Fast Multipole Method (FMM) WebGPU WGSL Compute Shader
// ====================================================================
// Runs across all WebGPU implementations (Chrome, Edge, Safari, Firefox, wgpu-py, Dawn)
// executing on AMD Radeon (Vulkan/DX12), NVIDIA CUDA, Intel Arc, and Apple Metal.

struct Particle {
    pos: vec3<f32>,
    q: f32,
};

struct OutputForce {
    force: vec3<f32>,
    _pad: f32,
};

struct SimulationParams {
    num_particles: u32,
    softening_sq: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> potentials: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces: array<OutputForce>;
@group(0) @binding(3) var<uniform> params: SimulationParams;

var<workgroup> tile_particles: array<Particle, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let n = params.num_particles;

    var my_particle: Particle;
    if (gid < n) {
        my_particle = particles[gid];
    } else {
        my_particle = Particle(vec3<f32>(0.0, 0.0, 0.0), 0.0);
    }

    var acc_phi: f32 = 0.0;
    var acc_f: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let num_tiles = (n + 127u) / 128u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let load_idx = t * 128u + lid;
        if (load_idx < n) {
            tile_particles[lid] = particles[load_idx];
        } else {
            tile_particles[lid] = Particle(vec3<f32>(0.0, 0.0, 0.0), 0.0);
        }
        workgroupBarrier();

        let tile_limit = min(128u, n - t * 128u);
        if (gid < n) {
            for (var j: u32 = 0u; j < tile_limit; j = j + 1u) {
                let global_j = t * 128u + j;
                if (global_j != gid) {
                    let pj = tile_particles[j];
                    let diff = my_particle.pos - pj.pos;
                    let r_sq = dot(diff, diff) + params.softening_sq;
                    let inv_r = inverseSqrt(r_sq);
                    let inv_r3 = inv_r * inv_r * inv_r;

                    acc_phi = acc_phi + pj.q * inv_r;
                    let f_scalar = my_particle.q * pj.q * inv_r3;
                    acc_f = acc_f + f_scalar * diff;
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
