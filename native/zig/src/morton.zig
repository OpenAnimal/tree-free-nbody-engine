const std = @import("std");

/// Interleave bits for 2D Morton code (x, y: 16-bit -> 32-bit Morton)
pub inline fn expandBits2D(v: u32) u32 {
    var x = v & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

/// Compact bits for 2D Morton decode
pub inline fn compactBits2D(v: u32) u32 {
    var x = v & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

/// Interleave bits for 3D Morton code (x, y, z: 21-bit -> 63-bit Morton)
pub inline fn expandBits3D(v: u64) u64 {
    var x = v & 0x1FFFFF; // 21 bits
    x = (x | (x << 32)) & 0x001F00000000FFFF;
    x = (x | (x << 16)) & 0x001F0000FF0000FF;
    x = (x | (x << 8))  & 0x010F00F00F00F00F;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

pub inline fn compactBits3D(v: u64) u64 {
    var x = v & 0x1249249249249249;
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3;
    x = (x | (x >> 4))  & 0x010F00F00F00F00F;
    x = (x | (x >> 8))  & 0x001F0000FF0000FF;
    x = (x | (x >> 16)) & 0x001F00000000FFFF;
    x = (x | (x >> 32)) & 0x1FFFFF;
    return x;
}

/// Encode 3D floating-point coordinates into 64-bit Morton code with bounds
pub inline fn encodeMorton3D(x: f32, y: f32, z: f32, min_bound: f32, max_bound: f32, depth: u5) u64 {
    const span = max_bound - min_bound;
    const inv_span = if (span > 1e-8) 1.0 / span else 1.0;
    const grid_res: f32 = @floatFromInt(@as(u32, 1) << depth);
    const max_idx: f32 = grid_res - 1.0;

    const norm_x = std.math.clamp((x - min_bound) * inv_span * grid_res, 0.0, max_idx);
    const norm_y = std.math.clamp((y - min_bound) * inv_span * grid_res, 0.0, max_idx);
    const norm_z = std.math.clamp((z - min_bound) * inv_span * grid_res, 0.0, max_idx);

    const ix: u64 = @intFromFloat(norm_x);
    const iy: u64 = @intFromFloat(norm_y);
    const iz: u64 = @intFromFloat(norm_z);

    return expandBits3D(ix) | (expandBits3D(iy) << 1) | (expandBits3D(iz) << 2);
}

/// Decode 64-bit Morton code back to 3D continuous coordinates
pub inline fn decodeMorton3D(code: u64, min_bound: f32, max_bound: f32, depth: u5) [3]f32 {
    const ix = compactBits3D(code);
    const iy = compactBits3D(code >> 1);
    const iz = compactBits3D(code >> 2);

    const span = max_bound - min_bound;
    const grid_res: f32 = @floatFromInt(@as(u32, 1) << depth);
    const cell_size = span / grid_res;

    return [3]f32{
        min_bound + (@as(f32, @floatFromInt(ix)) + 0.5) * cell_size,
        min_bound + (@as(f32, @floatFromInt(iy)) + 0.5) * cell_size,
        min_bound + (@as(f32, @floatFromInt(iz)) + 0.5) * cell_size,
    };
}

/// Batch encode 3D particles to Morton-64 array
pub fn batchEncodeMorton3D(
    px: [*]const f32,
    py: [*]const f32,
    pz: [*]const f32,
    num_particles: usize,
    min_bound: f32,
    max_bound: f32,
    depth: u5,
    out_morton: [*]u64,
) void {
    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        out_morton[i] = encodeMorton3D(px[i], py[i], pz[i], min_bound, max_bound, depth);
    }
}

/// Bitboard 64-bit occupancy accumulator: populates a fast 64-bit bitboard
pub fn buildMortonBitboard64(morton_codes: [*]const u64, num_particles: usize, shift: u6) u64 {
    var bitboard: u64 = 0;
    var i: usize = 0;
    while (i < num_particles) : (i += 1) {
        const bit_idx: u6 = @truncate((morton_codes[i] >> shift) & 0x3F);
        bitboard |= (@as(u64, 1) << bit_idx);
    }
    return bitboard;
}
