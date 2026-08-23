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

/// Integer binomial coefficient C(n, k). Exact for the small arguments used by
/// M2L (n <= 2*MAX_ORDER_2D - 1 = 15, k <= MAX_ORDER_2D = 8; largest value
/// C(15, 8) = 6435): every intermediate result * (n - i) is divisible by (i + 1),
/// so the integer division below never truncates.
fn binomial(n: usize, k: usize) usize {
    if (k > n) return 0;
    var result: usize = 1;
    var i: usize = 0;
    while (i < k) : (i += 1) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

/// M2L: Translates 2D multipole moments at src_center into local Taylor expansion at dst_center.
/// Exact transcription of CGR88 Theorem 2.3 (core.adaptive_fmm.m2l, the
/// brute-force-validated Python reference), using the same direction convention
/// delta = dst_center - src_center as the Python code:
///   c_0 = a_0 * log(delta) + sum_{k=1}^{P} a_k / delta^k
///   c_l = (-1)^(l-1) * a_0 / (l * delta^l)
///       + sum_{k=1}^{P} (-1)^l * binom(k+l-1, l) * a_k / delta^(k+l),  l >= 1
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
    // delta = dst_center - src_center (same convention as Python m2l)
    const delta = dst_center.sub(src_center);
    const inv_delta = delta.inv();

    const a0 = Complex.init(src_moments_re[0], src_moments_im[0]);

    // c_0 = a_0 * log(delta) + sum_{k=1}^P a_k / delta^k
    var c0 = a0.mul(delta.log());
    var inv_delta_pow = inv_delta; // delta^(-k), starts at k = 1
    var k: usize = 1;
    while (k <= p_order) : (k += 1) {
        const ak = Complex.init(src_moments_re[k], src_moments_im[k]);
        c0 = c0.add(ak.mul(inv_delta_pow));
        inv_delta_pow = inv_delta_pow.mul(inv_delta);
    }
    out_local_re[0] = c0.re;
    out_local_im[0] = c0.im;

    // c_l = (-1)^(l-1) * a_0 / (l * delta^l) + sum_{k=1}^P (-1)^l * binom(k+l-1, l) * a_k / delta^(k+l)
    var inv_delta_l = inv_delta; // delta^(-l), starts at l = 1
    var l: usize = 1;
    while (l <= p_order) : (l += 1) {
        // (-1)^(l-1): +1 for odd l, -1 for even l (monopole term sign)
        const sign_mono: f32 = if (l % 2 == 1) 1.0 else -1.0;
        // (-1)^l: -1 for odd l, +1 for even l (a_k sum sign)
        const sign_ak: f32 = if (l % 2 == 1) -1.0 else 1.0;

        const mono_scale = sign_mono / @as(f32, @floatFromInt(l));
        var cl = a0.mul(inv_delta_l).scale(mono_scale);

        // delta^-(k+l), starts at k = 1
        var inv_delta_kl = inv_delta_l.mul(inv_delta);
        k = 1;
        while (k <= p_order) : (k += 1) {
            const ak = Complex.init(src_moments_re[k], src_moments_im[k]);
            const factor = sign_ak * @as(f32, @floatFromInt(binomial(k + l - 1, l)));
            cl = cl.add(ak.mul(inv_delta_kl).scale(factor));
            inv_delta_kl = inv_delta_kl.mul(inv_delta);
        }

        out_local_re[l] = cl.re;
        out_local_im[l] = cl.im;
        inv_delta_l = inv_delta_l.mul(inv_delta);
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
