const std = @import("std");

pub const Complex = struct {
    re: f32,
    im: f32,

    pub inline fn init(re: f32, im: f32) Complex {
        return .{ .re = re, .im = im };
    }

    pub inline fn add(a: Complex, b: Complex) Complex {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }

    pub inline fn sub(a: Complex, b: Complex) Complex {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }

    pub inline fn mul(a: Complex, b: Complex) Complex {
        return .{
            .re = a.re * b.re - a.im * b.im,
            .im = a.re * b.im + a.im * b.re,
        };
    }

    pub inline fn scale(a: Complex, s: f32) Complex {
        return .{ .re = a.re * s, .im = a.im * s };
    }

    pub inline fn neg(a: Complex) Complex {
        return .{ .re = -a.re, .im = -a.im };
    }

    pub inline fn abs(a: Complex) f32 {
        return @sqrt(a.re * a.re + a.im * a.im);
    }

    pub inline fn absSq(a: Complex) f32 {
        return a.re * a.re + a.im * a.im;
    }

    pub inline fn log(a: Complex) Complex {
        const r = @sqrt(a.re * a.re + a.im * a.im + 1e-18);
        const theta = std.math.atan2(a.im, a.re);
        return .{ .re = @log(r), .im = theta };
    }

    pub inline fn inv(a: Complex) Complex {
        const d = a.re * a.re + a.im * a.im + 1e-18;
        return .{ .re = a.re / d, .im = -a.im / d };
    }
};

pub const MAX_ORDER_2D = 8;

/// P2M: Accumulates particle charges into 2D Laurent multipole moments around center
/// a_0 = sum(q_i)
/// a_k = - sum(q_i * (z_i - z0)^k) / k
pub fn computeP2M2D(
    px: [*]const f32,
    py: [*]const f32,
    charges: [*]const f32,
    num_particles: usize,
    cx: f32,
    cy: f32,
    order: usize,
    out_moments_re: [*]f32,
    out_moments_im: [*]f32,
) void {
    const p_order = @min(order, MAX_ORDER_2D);

    // Initialize moments to zero (index 0 to order)
    var k: usize = 0;
    while (k <= p_order) : (k += 1) {
        out_moments_re[k] = 0.0;
        out_moments_im[k] = 0.0;
    }

    const center = Complex.init(cx, cy);

    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        const q = charges[i];
        out_moments_re[0] += q;

        const dz = Complex.init(px[i] - center.re, py[i] - center.im);
        var dz_pow = dz;

        var m: usize = 1;
        while (m <= p_order) : (m += 1) {
            const factor: f32 = -q / @as(f32, @floatFromInt(m));
            const term = dz_pow.scale(factor);
            out_moments_re[m] += term.re;
            out_moments_im[m] += term.im;
            dz_pow = dz_pow.mul(dz);
        }
    }
}

/// M2L: Translates 2D multipole moments at src_center into local Taylor expansion at dst_center
/// Matches exact mathematical formulation in core.tree_free_fmm.m2l
pub fn computeM2L2D(
    src_moments_re: [*]const f32,
    src_moments_im: [*]const f32,
    src_cx: f32,
    src_cy: f32,
    dst_cx: f32,
    dst_cy: f32,
    order: usize,
    out_local_re: [*]f32,
    out_local_im: [*]f32,
) void {
    const p_order = @min(order, MAX_ORDER_2D);
    const src_center = Complex.init(src_cx, src_cy);
    const dst_center = Complex.init(dst_cx, dst_cy);
    const z0 = src_center.sub(dst_center);
    const neg_z0 = z0.neg();

    // l_0 = a_0 * log(-z0) + sum_{k=1}^P a_k / (-z0)^k
    const a0 = Complex.init(src_moments_re[0], src_moments_im[0]);
    var l0 = a0.mul(neg_z0.log());

    var inv_neg_z0_pow = neg_z0.inv();
    const inv_neg_z0 = inv_neg_z0_pow;

    var k: usize = 1;
    while (k <= p_order) : (k += 1) {
        const ak = Complex.init(src_moments_re[k], src_moments_im[k]);
        l0 = l0.add(ak.mul(inv_neg_z0_pow));
        inv_neg_z0_pow = inv_neg_z0_pow.mul(inv_neg_z0);
    }

    out_local_re[0] = l0.re;
    out_local_im[0] = l0.im;

    // Higher order local coefficients: l_m
    const inv_z0 = z0.inv();
    var inv_z0_pow = inv_z0;

    var l: usize = 1;
    while (l <= p_order) : (l += 1) {
        const sign: f32 = if (l % 2 == 1) -1.0 else 1.0;
        const scale_factor = sign / @as(f32, @floatFromInt(l));
        var lm = a0.mul(inv_z0_pow).scale(scale_factor);

        // Sum_{k=1}^P a_k / (-z0)^(k+l)
        // Note: inv_neg_z0^(k+l)
        var inv_neg_z0_kl = inv_neg_z0;
        var step: usize = 1;
        while (step < l) : (step += 1) {
            inv_neg_z0_kl = inv_neg_z0_kl.mul(inv_neg_z0);
        }

        k = 1;
        while (k <= p_order) : (k += 1) {
            inv_neg_z0_kl = inv_neg_z0_kl.mul(inv_neg_z0);
            const ak = Complex.init(src_moments_re[k], src_moments_im[k]);
            lm = lm.add(ak.mul(inv_neg_z0_kl));
        }

        out_local_re[l] = lm.re;
        out_local_im[l] = lm.im;
        inv_z0_pow = inv_z0_pow.mul(inv_z0);
    }
}

/// L2P: Evaluates local expansion potential at target particles
/// Phi(z) = Re[ l_0 + sum_{l=1}^P l_l * (z - z_center)^l ]
pub fn evaluateL2P2D(
    local_re: [*]const f32,
    local_im: [*]const f32,
    order: usize,
    cx: f32,
    cy: f32,
    tx: [*]const f32,
    ty: [*]const f32,
    num_targets: usize,
    out_potentials: [*]f32,
) void {
    const p_order = @min(order, MAX_ORDER_2D);
    const center = Complex.init(cx, cy);

    var i: usize = 0;
    while (i < num_targets) : (i += 1) {
        const dz = Complex.init(tx[i] - center.re, ty[i] - center.im);
        var dz_pow = dz;
        var pot: f32 = local_re[0];

        var l: usize = 1;
        while (l <= p_order) : (l += 1) {
            const ll = Complex.init(local_re[l], local_im[l]);
            const term = ll.mul(dz_pow);
            pot += term.re;
            dz_pow = dz_pow.mul(dz);
        }
        out_potentials[i] = pot;
    }
}
