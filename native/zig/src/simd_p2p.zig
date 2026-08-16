const std = @import("std");

pub const VecWidth = 8;
pub const F32Vec = @Vector(VecWidth, f32);

/// Fast reciprocal square root approximation with Newton-Raphson refinement
pub inline fn fastRsqrtVec(v: F32Vec) F32Vec {
    const one: F32Vec = @splat(1.0);
    return one / @sqrt(v);
}

/// Computes Direct N-Body P2P Gravitational / Coulomb Potentials via SIMD @Vector(8, f32)
pub fn computeDirectP2P(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    masses: [*]const f32,
    num_particles: usize,
    softening_sq: f32,
    out_potentials: [*]f32,
) void {
    const eps2_vec: F32Vec = @splat(softening_sq);

    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        const xi_v: F32Vec = @splat(px[i]);
        const yi_v: F32Vec = @splat(py[i]);
        const zi_v: F32Vec = @splat(pz[i]);

        var acc_vec: F32Vec = @splat(0.0);
        var j: usize = 0;

        // Vectorized SIMD loop (8 elements per iteration)
        while (j + VecWidth <= num_particles) : (j += VecWidth) {
            const xj_v: F32Vec = px[j..][0..VecWidth].*;
            const yj_v: F32Vec = py[j..][0..VecWidth].*;
            const zj_v: F32Vec = pz[j..][0..VecWidth].*;
            const mj_v: F32Vec = masses[j..][0..VecWidth].*;

            const dx = xj_v - xi_v;
            const dy = yj_v - yi_v;
            const dz = zj_v - zi_v;

            const dist_sq = dx * dx + dy * dy + dz * dz + eps2_vec;
            const inv_r = fastRsqrtVec(dist_sq);

            acc_vec += mj_v * inv_r;
        }

        // Horizontal sum across SIMD lanes
        var total_pot: f32 = @reduce(.Add, acc_vec);

        // Scalar remainder loop
        while (j < num_particles) : (j += 1) {
            const dx = px[j] - px[i];
            const dy = py[j] - py[i];
            const dz = pz[j] - pz[i];
            const dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            total_pot += masses[j] / @sqrt(dist_sq);
        }

        // Self-interaction correction (subtract self term if softening_sq was counted)
        // Self contribution at distance 0 is mass[i] / sqrt(softening_sq)
        const self_term = masses[i] / @sqrt(softening_sq);
        out_potentials[i] = total_pot - self_term;
    }
}

/// Computes Direct N-Body P2P Forces (Fx, Fy, Fz) via SIMD vectorization
pub fn computeDirectP2PForces(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    masses: [*]const f32,
    num_particles: usize,
    softening_sq: f32,
    out_fx: [*]f32,
    out_fy: [*]f32,
    out_fz: [*]f32,
) void {
    const eps2_vec: F32Vec = @splat(softening_sq);

    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        const xi_v: F32Vec = @splat(px[i]);
        const yi_v: F32Vec = @splat(py[i]);
        const zi_v: F32Vec = @splat(pz[i]);

        var acc_fx: F32Vec = @splat(0.0);
        var acc_fy: F32Vec = @splat(0.0);
        var acc_fz: F32Vec = @splat(0.0);

        var j: usize = 0;
        while (j + VecWidth <= num_particles) : (j += VecWidth) {
            const xj_v: F32Vec = px[j..][0..VecWidth].*;
            const yj_v: F32Vec = py[j..][0..VecWidth].*;
            const zj_v: F32Vec = pz[j..][0..VecWidth].*;
            const mj_v: F32Vec = masses[j..][0..VecWidth].*;

            const dx = xj_v - xi_v;
            const dy = yj_v - yi_v;
            const dz = zj_v - zi_v;

            const dist_sq = dx * dx + dy * dy + dz * dz + eps2_vec;
            const inv_r = fastRsqrtVec(dist_sq);
            const inv_r3 = inv_r * inv_r * inv_r;
            const m_inv_r3 = mj_v * inv_r3;

            acc_fx += dx * m_inv_r3;
            acc_fy += dy * m_inv_r3;
            acc_fz += dz * m_inv_r3;
        }

        var total_fx: f32 = @reduce(.Add, acc_fx);
        var total_fy: f32 = @reduce(.Add, acc_fy);
        var total_fz: f32 = @reduce(.Add, acc_fz);

        while (j < num_particles) : (j += 1) {
            if (i == j) continue;
            const dx = px[j] - px[i];
            const dy = py[j] - py[i];
            const dz = pz[j] - pz[i];
            const dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            const inv_r = 1.0 / @sqrt(dist_sq);
            const inv_r3 = inv_r * inv_r * inv_r;
            const factor = masses[j] * inv_r3;

            total_fx += dx * factor;
            total_fy += dy * factor;
            total_fz += dz * factor;
        }

        out_fx[i] = masses[i] * total_fx;
        out_fy[i] = masses[i] * total_fy;
        out_fz[i] = masses[i] * total_fz;
    }
}
