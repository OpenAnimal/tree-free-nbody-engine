const std = @import("std");

/// Point-Point Contact / Barrier Potential (Incremental Potential Contact)
/// Zero-allocation, high-speed SIMD evaluation
pub const ContactConfig = struct {
    dhat: f32 = 1e-3, // Contact barrier activation distance
    kappa: f32 = 1e4, // Barrier stiffness
};

/// Log-barrier function: B(d) = -(d - dhat)^2 * ln(d / dhat) for d < dhat
pub inline fn evaluateBarrier(dist: f32, dhat: f32, kappa: f32) f32 {
    if (dist >= dhat or dist <= 1e-9) return 0.0;
    const diff = dist - dhat;
    const ratio = dist / dhat;
    return -kappa * diff * diff * @log(ratio);
}

/// Derivative of log-barrier function: dB/dd
pub inline fn evaluateBarrierGradient(dist: f32, dhat: f32, kappa: f32) f32 {
    if (dist >= dhat or dist <= 1e-9) return 0.0;
    const diff = dist - dhat;
    const ratio = dist / dhat;
    // -kappa * [ 2*(d - dhat)*ln(d/dhat) + (d - dhat)^2 / d ]
    return -kappa * (2.0 * diff * @log(ratio) + (diff * diff) / dist);
}

/// Batch evaluate matrix-free contact forces with zero heap allocation
pub fn evaluateContactForces(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    num_nodes: usize,
    dhat: f32,
    kappa: f32,
    out_fx: [*]f32,
    out_fy: [*]f32,
    out_fz: [*]f32,
) void {
    // Clear outputs
    var i: usize = 0;
    while (i < num_nodes) : (i += 1) {
        out_fx[i] = 0.0;
        out_fy[i] = 0.0;
        out_fz[i] = 0.0;
    }

    // Pairwise contact evaluation
    i = 0;
    while (i < num_nodes) : (i += 1) {
        var j = i + 1;
        while (j < num_nodes) : (j += 1) {
            const dx = px[i] - px[j];
            const dy = py[i] - py[j];
            const dz = pz[i] - pz[j];
            const dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < dhat * dhat and dist_sq > 1e-12) {
                const dist = @sqrt(dist_sq);
                const g = evaluateBarrierGradient(dist, dhat, kappa);
                const inv_dist = 1.0 / dist;
                const fx = -g * (dx * inv_dist);
                const fy = -g * (dy * inv_dist);
                const fz = -g * (dz * inv_dist);

                out_fx[i] += fx;
                out_fy[i] += fy;
                out_fz[i] += fz;

                out_fx[j] -= fx;
                out_fy[j] -= fy;
                out_fz[j] -= fz;
            }
        }
    }
}
