const std = @import("std");

/// 3D Cartesian Multipole Expansion Coefficients up to Quadrupole (Order 2)
/// Layout: [M0, Dx, Dy, Dz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] (10 floats)
pub const Multipole3D = extern struct {
    m0: f32 = 0.0,
    dx: f32 = 0.0,
    dy: f32 = 0.0,
    dz: f32 = 0.0,
    qxx: f32 = 0.0,
    qyy: f32 = 0.0,
    qzz: f32 = 0.0,
    qxy: f32 = 0.0,
    qxz: f32 = 0.0,
    qyz: f32 = 0.0,

    pub inline fn reset(self: *Multipole3D) void {
        self.* = .{};
    }

    pub inline fn accumulate(self: *Multipole3D, x: f32, y: f32, z: f32, mass: f32) void {
        self.m0 += mass;
        self.dx += mass * x;
        self.dy += mass * y;
        self.dz += mass * z;
        self.qxx += mass * x * x;
        self.qyy += mass * y * y;
        self.qzz += mass * z * z;
        self.qxy += mass * x * y;
        self.qxz += mass * x * z;
        self.qyz += mass * y * z;
    }
};

/// 3D Local Taylor Series Coefficients up to 2nd Order (10 floats)
/// Layout: [L0, Gx, Gy, Gz, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz]
pub const Local3D = extern struct {
    l0: f32 = 0.0,
    gx: f32 = 0.0,
    gy: f32 = 0.0,
    gz: f32 = 0.0,
    hxx: f32 = 0.0,
    hyy: f32 = 0.0,
    hzz: f32 = 0.0,
    hxy: f32 = 0.0,
    hxz: f32 = 0.0,
    hyz: f32 = 0.0,
};

/// P2M: Accumulates particles into 3D Cartesian multipole moments around center (cx, cy, cz)
pub fn computeP2M3D(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    masses: [*]const f32,
    num_particles: usize,
    cx: f32,
    cy: f32,
    cz: f32,
    out_moments: *Multipole3D,
) void {
    out_moments.reset();
    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        const dx = px[i] - cx;
        const dy = py[i] - cy;
        const dz = pz[i] - cz;
        out_moments.accumulate(dx, dy, dz, masses[i]);
    }
}

/// M2P: Directly evaluates Far-Field Gravitational / Coulomb Potential (1/r) from 3D Multipole
pub fn evaluateM2P3D(
    moments: *const Multipole3D,
    cx: f32,
    cy: f32,
    cz: f32,
    tx: [*]const f32,
    ty: [*]const f32,
    tz: [*]const f32,
    num_targets: usize,
    softening_sq: f32,
    out_potentials: [*]f32,
) void {
    var i: usize = 0;
    while (i < num_targets) : (i += 1) {
        const rx = tx[i] - cx;
        const ry = ty[i] - cy;
        const rz = tz[i] - cz;

        const r2 = rx * rx + ry * ry + rz * rz + softening_sq;
        const inv_r = 1.0 / @sqrt(r2);
        const inv_r2 = inv_r * inv_r;
        const inv_r3 = inv_r2 * inv_r;
        const inv_r5 = inv_r3 * inv_r2;

        // Monopole: M0 / r
        var pot: f32 = moments.m0 * inv_r;

        // Dipole: (R . D) / r^3
        const r_dot_d = rx * moments.dx + ry * moments.dy + rz * moments.dz;
        pot += r_dot_d * inv_r3;

        // Quadrupole: (3 * R^T Q R - Tr(Q) * r^2) / (2 * r^5)
        const tr_q = moments.qxx + moments.qyy + moments.qzz;
        const r_q_r = rx * (moments.qxx * rx + moments.qxy * ry + moments.qxz * rz) +
                      ry * (moments.qxy * rx + moments.qyy * ry + moments.qyz * rz) +
                      rz * (moments.qxz * rx + moments.qyz * ry + moments.qzz * rz);

        const quad_term = (3.0 * r_q_r - tr_q * r2) * (0.5 * inv_r5);
        pot += quad_term;

        out_potentials[i] = pot;
    }
}

/// M2L: Translates 3D Multipole from source cluster to Local Taylor expansion at target cluster center
pub fn computeM2L3D(
    moments: *const Multipole3D,
    src_cx: f32,
    src_cy: f32,
    src_cz: f32,
    dst_cx: f32,
    dst_cy: f32,
    dst_cz: f32,
    softening_sq: f32,
    out_local: *Local3D,
) void {
    const rx = dst_cx - src_cx;
    const ry = dst_cy - src_cy;
    const rz = dst_cz - src_cz;

    const r2 = rx * rx + ry * ry + rz * rz + softening_sq;
    const inv_r = 1.0 / @sqrt(r2);
    const inv_r3 = inv_r * inv_r * inv_r;
    const inv_r5 = inv_r3 * inv_r * inv_r;

    // Potential at dst center (L0)
    const r_dot_d = rx * moments.dx + ry * moments.dy + rz * moments.dz;
    const tr_q = moments.qxx + moments.qyy + moments.qzz;
    const r_q_r = rx * (moments.qxx * rx + moments.qxy * ry + moments.qxz * rz) +
                  ry * (moments.qxy * rx + moments.qyy * ry + moments.qyz * rz) +
                  rz * (moments.qxz * rx + moments.qyz * ry + moments.qzz * rz);

    out_local.l0 = moments.m0 * inv_r + r_dot_d * inv_r3 + (3.0 * r_q_r - tr_q * r2) * (0.5 * inv_r5);

    // Gradient at dst center (Gx, Gy, Gz) = d/dR (Phi)
    // Monopole grad: -M0 * R / r^3
    // Dipole grad: D / r^3 - 3 (R . D) R / r^5
    out_local.gx = -moments.m0 * rx * inv_r3 + (moments.dx * inv_r3 - 3.0 * r_dot_d * rx * inv_r5);
    out_local.gy = -moments.m0 * ry * inv_r3 + (moments.dy * inv_r3 - 3.0 * r_dot_d * ry * inv_r5);
    out_local.gz = -moments.m0 * rz * inv_r3 + (moments.dz * inv_r3 - 3.0 * r_dot_d * rz * inv_r5);

    // Hessian approximation (leading monopole curvature)
    out_local.hxx = -moments.m0 * (inv_r3 - 3.0 * rx * rx * inv_r5);
    out_local.hyy = -moments.m0 * (inv_r3 - 3.0 * ry * ry * inv_r5);
    out_local.hzz = -moments.m0 * (inv_r3 - 3.0 * rz * rz * inv_r5);
    out_local.hxy = moments.m0 * 3.0 * rx * ry * inv_r5;
    out_local.hxz = moments.m0 * 3.0 * rx * rz * inv_r5;
    out_local.hyz = moments.m0 * 3.0 * ry * rz * inv_r5;
}

/// L2P: Evaluates Local Taylor series potential at target particles
pub fn evaluateL2P3D(
    local: *const Local3D,
    cx: f32,
    cy: f32,
    cz: f32,
    tx: [*]const f32,
    ty: [*]const f32,
    tz: [*]const f32,
    num_targets: usize,
    out_potentials: [*]f32,
) void {
    var i: usize = 0;
    while (i < num_targets) : (i += 1) {
        const ux = tx[i] - cx;
        const uy = ty[i] - cy;
        const uz = tz[i] - cz;

        var pot: f32 = local.l0;
        pot += local.gx * ux + local.gy * uy + local.gz * uz;
        pot += 0.5 * (local.hxx * ux * ux + local.hyy * uy * uy + local.hzz * uz * uz +
                      2.0 * (local.hxy * ux * uy + local.hxz * ux * uz + local.hyz * uy * uz));

        out_potentials[i] = pot;
    }
}
