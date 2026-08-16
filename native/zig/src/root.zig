const std = @import("std");
pub const morton = @import("morton.zig");
pub const simd_p2p = @import("simd_p2p.zig");
pub const multipole_2d = @import("multipole_2d.zig");
pub const multipole_3d = @import("multipole_3d.zig");
pub const contact_ipc = @import("contact_ipc.zig");

// ============================================================================
// C-ABI EXPORTS (Unreal Engine 5, C++, Python ctypes, Robotics)
// ============================================================================

/// Direct N-body P2P Potential calculation via SIMD @Vector(8, f32)
pub export fn zig_fmm_p2p_potentials(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    masses: [*]const f32,
    num_particles: usize,
    softening_sq: f32,
    out_potentials: [*]f32,
) void {
    simd_p2p.computeDirectP2P(
        px, py, pz,
        masses,
        num_particles,
        softening_sq,
        out_potentials,
    );
}

/// Direct N-body P2P Force calculation (Fx, Fy, Fz) via SIMD vectorization
pub export fn zig_fmm_p2p_forces(
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
    simd_p2p.computeDirectP2PForces(
        px, py, pz,
        masses,
        num_particles,
        softening_sq,
        out_fx, out_fy, out_fz,
    );
}

/// 3D Morton-64 Quantization & Bit-Packing
pub export fn zig_fmm_encode_morton3d(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    num_particles: usize,
    min_bound: f32,
    max_bound: f32,
    depth: u32,
    out_morton: [*]u64,
) void {
    const d: u5 = @intCast(@min(depth, 21));
    morton.batchEncodeMorton3D(
        px, py, pz,
        num_particles,
        min_bound, max_bound,
        d,
        out_morton,
    );
}

/// 64-Bit Bitboard Occupancy Extraction
pub export fn zig_fmm_build_bitboard64(
    morton_codes: [*]const u64,
    num_particles: usize,
    shift: u32,
) u64 {
    if (shift >= 64) return 0;
    const s: u6 = @intCast(shift);
    return morton.buildMortonBitboard64(morton_codes, num_particles, s);
}

/// Zero-Allocation Matrix-Free IPC Contact Force Solver
pub export fn zig_fmm_contact_forces(
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
    contact_ipc.evaluateContactForces(
        px, py, pz,
        num_nodes,
        dhat, kappa,
        out_fx, out_fy, out_fz,
    );
}

// ----------------------------------------------------------------------------
// 2D Complex Multipole Exports (Greengard-Rokhlin Laurent series)
// ----------------------------------------------------------------------------

pub export fn zig_fmm_2d_p2m(
    px: [*]const f32,
    py: [*]const f32,
    charges: [*]const f32,
    num_particles: usize,
    cx: f32,
    cy: f32,
    order: u32,
    out_moments_re: [*]f32,
    out_moments_im: [*]f32,
) void {
    multipole_2d.computeP2M2D(
        px, py,
        charges,
        num_particles,
        cx, cy,
        order,
        out_moments_re,
        out_moments_im,
    );
}

pub export fn zig_fmm_2d_m2l(
    src_moments_re: [*]const f32,
    src_moments_im: [*]const f32,
    src_cx: f32,
    src_cy: f32,
    dst_cx: f32,
    dst_cy: f32,
    order: u32,
    out_local_re: [*]f32,
    out_local_im: [*]f32,
) void {
    multipole_2d.computeM2L2D(
        src_moments_re,
        src_moments_im,
        src_cx, src_cy,
        dst_cx, dst_cy,
        order,
        out_local_re,
        out_local_im,
    );
}

pub export fn zig_fmm_2d_l2p(
    local_re: [*]const f32,
    local_im: [*]const f32,
    order: u32,
    cx: f32,
    cy: f32,
    tx: [*]const f32,
    ty: [*]const f32,
    num_targets: usize,
    out_potentials: [*]f32,
) void {
    multipole_2d.evaluateL2P2D(
        local_re,
        local_im,
        order,
        cx, cy,
        tx, ty,
        num_targets,
        out_potentials,
    );
}

// ----------------------------------------------------------------------------
// 3D Cartesian Multipole Exports (1/r Gravitational / Coulomb / UE5)
// ----------------------------------------------------------------------------

pub export fn zig_fmm_3d_p2m(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    masses: [*]const f32,
    num_particles: usize,
    cx: f32,
    cy: f32,
    cz: f32,
    out_moments: *multipole_3d.Multipole3D,
) void {
    multipole_3d.computeP2M3D(
        px, py, pz,
        masses,
        num_particles,
        cx, cy, cz,
        out_moments,
    );
}

pub export fn zig_fmm_3d_m2p(
    moments: *const multipole_3d.Multipole3D,
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
    multipole_3d.evaluateM2P3D(
        moments,
        cx, cy, cz,
        tx, ty, tz,
        num_targets,
        softening_sq,
        out_potentials,
    );
}

pub export fn zig_fmm_3d_m2l(
    moments: *const multipole_3d.Multipole3D,
    src_cx: f32,
    src_cy: f32,
    src_cz: f32,
    dst_cx: f32,
    dst_cy: f32,
    dst_cz: f32,
    softening_sq: f32,
    out_local: *multipole_3d.Local3D,
) void {
    multipole_3d.computeM2L3D(
        moments,
        src_cx, src_cy, src_cz,
        dst_cx, dst_cy, dst_cz,
        softening_sq,
        out_local,
    );
}

pub export fn zig_fmm_3d_l2p(
    local: *const multipole_3d.Local3D,
    cx: f32,
    cy: f32,
    cz: f32,
    tx: [*]const f32,
    ty: [*]const f32,
    tz: [*]const f32,
    num_targets: usize,
    out_potentials: [*]f32,
) void {
    multipole_3d.evaluateL2P3D(
        local,
        cx, cy, cz,
        tx, ty, tz,
        num_targets,
        out_potentials,
    );
}

/// Version / Health Check
pub export fn zig_fmm_version() u32 {
    return 110; // v1.1.0 (with 2D & 3D Multipoles)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

test "morton 3D roundtrip" {
    const code = morton.encodeMorton3D(0.5, 0.25, 0.75, 0.0, 1.0, 6);
    const decoded = morton.decodeMorton3D(code, 0.0, 1.0, 6);
    try std.testing.expectApproxEqAbs(decoded[0], 0.5, 0.02);
    try std.testing.expectApproxEqAbs(decoded[1], 0.25, 0.02);
    try std.testing.expectApproxEqAbs(decoded[2], 0.75, 0.02);
}

test "simd p2p evaluation" {
    const px = [_]f32{ 0.0, 1.0, 0.0 };
    const py = [_]f32{ 0.0, 0.0, 1.0 };
    const pz = [_]f32{ 0.0, 0.0, 0.0 };
    const masses = [_]f32{ 1.0, 1.0, 1.0 };
    var pot = [_]f32{ 0.0, 0.0, 0.0 };

    simd_p2p.computeDirectP2P(&px, &py, &pz, &masses, 3, 1e-4, &pot);
    try std.testing.expect(pot[0] > 0.0);
}

test "2d multipole p2m and l2p" {
    const px = [_]f32{ 0.1, -0.1 };
    const py = [_]f32{ 0.05, -0.05 };
    const q = [_]f32{ 1.0, 2.0 };
    var m_re = [_]f32{0.0} ** 5;
    var m_im = [_]f32{0.0} ** 5;

    multipole_2d.computeP2M2D(&px, &py, &q, 2, 0.0, 0.0, 4, &m_re, &m_im);
    try std.testing.expectApproxEqAbs(m_re[0], 3.0, 1e-5);
}

test "3d cartesian multipole p2m and m2p" {
    const px = [_]f32{ 0.1, -0.1 };
    const py = [_]f32{ 0.0, 0.0 };
    const pz = [_]f32{ 0.0, 0.0 };
    const m = [_]f32{ 1.0, 1.0 };
    var moments: multipole_3d.Multipole3D = .{};

    multipole_3d.computeP2M3D(&px, &py, &pz, &m, 2, 0.0, 0.0, 0.0, &moments);
    try std.testing.expectApproxEqAbs(moments.m0, 2.0, 1e-5);
    try std.testing.expectApproxEqAbs(moments.dx, 0.0, 1e-5); // Dipole cancels for symmetric mass
}
