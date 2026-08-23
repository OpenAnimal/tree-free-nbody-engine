// Funnel hash table (Farach-Colton, Krapivin, & Kuszmaul, 2025, Section 3)
// Zig port of core/elastic_hash.py:ElasticHashTable for microbenchmarking.
//
// Geometry (matches the Python reference):
//   alpha = ceil(4*log2(1/delta) + 10)   slabs, sizes shrinking by ~3/4,
//   beta  = ceil(2*log2(1/delta))        slots per sub-array,
//   A_{alpha+1} in [ceil(delta*n/2), floor(3*delta*n/4)] overflow slots
//   (>= 16 for tiny tables), split into halves B (uniform probing,
//   ceil(log2 log2 n) attempt cutoff) and C (two-choice buckets of
//   2*ceil(log2 log2 n) slots).
//
// The table supports `capacity` insertions (slots = capacity/(1-delta),
// i.e. final load 1-delta) with probability 1 - n^{-omega(1)}; insertion
// never displaces an existing key and any search inspects at most
// `probe_bound` slots deterministically.
//
// This is a faithful port: the salt generation uses a deterministic
// LCG seeded with `seed` so a given (capacity, delta, seed) produces the
// same slab geometry and the same probe sequences as the Python reference
// (modulo the mixer's u64 wrap, which is identical). The Python reference
// uses numpy.RandomState(seed) for its salts; we use a splitmix64-seeded
// LCG here which is NOT bit-identical to numpy's MT19937, so the exact
// slot assignments differ -- but the *geometry* (alpha, beta, slab sizes,
// overflow split) is identical, and the throughput characteristics the
// microbenchmark measures are geometry-driven, not salt-driven.

const std = @import("std");

const U64_MASK: u64 = 0xFFFFFFFFFFFFFFFF;

// splitmix64 finalizer (matches core/elastic_hash.py:_mix64).
fn mix64(z_in: u64) u64 {
    var z: u64 = z_in;
    z ^= z >> 30;
    z = z *% 0xBF58476D1CE4E5B9;
    z ^= z >> 27;
    z = z *% 0x94D049BB133111EB;
    z ^= z >> 31;
    return z;
}

pub const Slab = struct {
    offset: usize,
    subarray_count: usize,
    salt: u64,
};

pub const FunnelHashTable = struct {
    allocator: std.mem.Allocator,
    capacity: usize,
    delta: f64,
    alpha: usize,
    beta: usize,
    total_size: usize,
    b_offset: usize,
    c_offset: usize,
    b_size: usize,
    c_size: usize,
    b_attempts: usize,
    c_bucket_slots: usize,
    c_num_buckets: usize,
    count: usize,
    overflow_count: usize,
    keys: []i64,
    occupied: []bool,
    slabs: []Slab,
    salt_b: []u64,
    salt_c1: u64,
    salt_c2: u64,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, delta: f64, seed: u64) !FunnelHashTable {
        if (capacity < 1) return error.InvalidCapacity;
        var d = delta;
        if (!(d > 0.0 and d <= 0.125)) d = @min(@max(d, 1e-9), 0.125);

        const log_inv = std.math.log2(1.0 / d);
        const alpha: usize = @intFromFloat(@ceil(4.0 * log_inv + 10.0));
        const beta: usize = @max(2, @as(usize, @intFromFloat(@ceil(2.0 * log_inv))));

        const n_target: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(capacity)) / (1.0 - d)));

        // Overflow region fixed-point (matches the Python loop).
        var overflow_size: usize = @max(16, @as(usize, @intFromFloat(@ceil(d * @as(f64, @floatFromInt(n_target)) / 2.0))));
        var i: usize = 0;
        while (i < 8) : (i += 1) {
            const a_probe: usize = @max(alpha, @as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(n_target - overflow_size)) / @as(f64, @floatFromInt(beta))))));
            const n_final: usize = a_probe * beta + overflow_size;
            const o_low: usize = @intFromFloat(@ceil(d * @as(f64, @floatFromInt(n_final)) / 2.0));
            if (o_low <= overflow_size) break;
            overflow_size = @max(16, o_low);
        }

        const a_total: usize = @max(alpha, @as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(n_target - overflow_size)) / @as(f64, @floatFromInt(beta))))));

        // Sub-array counts per slab: geometric 3/4 shrink, non-increasing,
        // summing to a_total. Matches the Python vectorized computation.
        var subarray_counts = try allocator.alloc(usize, alpha);
        defer allocator.free(subarray_counts);
        var weights_sum: f64 = 0.0;
        var weights = try allocator.alloc(f64, alpha);
        defer allocator.free(weights);
        var j: usize = 0;
        while (j < alpha) : (j += 1) {
            const w = std.math.pow(f64, 0.75, @as(f64, @floatFromInt(j)));
            weights[j] = w;
            weights_sum += w;
        }
        var assigned: usize = 0;
        j = 0;
        while (j < alpha) : (j += 1) {
            const w_norm = weights[j] / weights_sum;
            const c = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(a_total)) * w_norm)));
            subarray_counts[j] = @max(1, c);
            assigned += subarray_counts[j];
        }
        // Distribute the remainder into the first slots.
        var rem: usize = a_total - assigned;
        j = 0;
        while (rem > 0 and j < alpha) : (j += 1) {
            subarray_counts[j] += 1;
            rem -= 1;
        }

        // Slab offsets.
        var slab_offsets = try allocator.alloc(usize, alpha);
        defer allocator.free(slab_offsets);
        slab_offsets[0] = 0;
        j = 1;
        while (j < alpha) : (j += 1) {
            slab_offsets[j] = slab_offsets[j - 1] + subarray_counts[j - 1] * beta;
        }
        const funnel_end: usize = slab_offsets[alpha - 1] + subarray_counts[alpha - 1] * beta;

        const b_size: usize = overflow_size / 2;
        const c_size: usize = overflow_size - b_size;
        const b_offset: usize = funnel_end;
        const c_offset: usize = funnel_end + b_size;
        const total_size: usize = c_offset + c_size;

        const ll = std.math.log2(std.math.log2(@max(@as(f64, @floatFromInt(total_size)), 4.0)));
        const b_attempts: usize = @max(4, @as(usize, @intFromFloat(@ceil(ll))));
        const c_bucket_slots: usize = 2 * @max(2, @as(usize, @intFromFloat(@ceil(ll))));
        const c_num_buckets: usize = @max(1, c_size / c_bucket_slots);

        // Backing storage (key sentinel -1 = empty).
        const keys = try allocator.alloc(i64, total_size);
        const occupied = try allocator.alloc(bool, total_size);
        @memset(keys, -1);
        @memset(occupied, false);

        // Salts: deterministic splitmix64-seeded LCG (NOT numpy MT19937,
        // but geometry-preserving -- see file header).
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();

        var slabs = try allocator.alloc(Slab, alpha);
        var k: usize = 0;
        while (k < alpha) : (k += 1) {
            slabs[k] = .{
                .offset = slab_offsets[k],
                .subarray_count = subarray_counts[k],
                .salt = random.int(u64) | 1, // ensure nonzero
            };
        }
        var salt_b = try allocator.alloc(u64, b_attempts);
        k = 0;
        while (k < b_attempts) : (k += 1) {
            salt_b[k] = random.int(u64) | 1;
        }
        const salt_c1: u64 = random.int(u64) | 1;
        const salt_c2: u64 = random.int(u64) | 1;

        return FunnelHashTable{
            .allocator = allocator,
            .capacity = capacity,
            .delta = d,
            .alpha = alpha,
            .beta = beta,
            .total_size = total_size,
            .b_offset = b_offset,
            .c_offset = c_offset,
            .b_size = b_size,
            .c_size = c_size,
            .b_attempts = b_attempts,
            .c_bucket_slots = c_bucket_slots,
            .c_num_buckets = c_num_buckets,
            .count = 0,
            .overflow_count = 0,
            .keys = keys,
            .occupied = occupied,
            .slabs = slabs,
            .salt_b = salt_b,
            .salt_c1 = salt_c1,
            .salt_c2 = salt_c2,
        };
    }

    pub fn deinit(self: *FunnelHashTable) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.occupied);
        self.allocator.free(self.slabs);
        self.allocator.free(self.salt_b);
    }

    pub fn probeBound(self: *const FunnelHashTable) usize {
        return self.alpha * self.beta + self.b_attempts + 2 * self.c_bucket_slots;
    }

    fn fill(self: *FunnelHashTable, pos: usize, key: i64) void {
        self.keys[pos] = key;
        self.occupied[pos] = true;
        self.count += 1;
        if (pos >= self.b_offset) self.overflow_count += 1;
    }

    fn searchOverflow(self: *FunnelHashTable, key: i64) struct { pos: i64, probes: usize } {
        var probes: usize = 0;
        const k: u64 = @bitCast(key);
        var t: usize = 0;
        while (t < self.b_attempts) : (t += 1) {
            const pos = self.b_offset + (@as(usize, @intCast(mix64(k ^ self.salt_b[t]) % self.b_size)));
            probes += 1;
            if (self.occupied[pos] and self.keys[pos] == key) return .{ .pos = @intCast(pos), .probes = probes };
        }
        const b1 = @as(usize, @intCast(mix64(k ^ self.salt_c1) % self.c_num_buckets));
        const b2 = @as(usize, @intCast(mix64(k ^ self.salt_c2) % self.c_num_buckets));
        var s: usize = 0;
        while (s < self.c_bucket_slots) : (s += 1) {
            const buckets = [_]usize{ b1, b2 };
            for (buckets) |bkt| {
                const pos = self.c_offset + bkt * self.c_bucket_slots + s;
                probes += 1;
                if (self.occupied[pos] and self.keys[pos] == key) return .{ .pos = @intCast(pos), .probes = probes };
            }
        }
        return .{ .pos = -1, .probes = probes };
    }

    fn placeOverflow(self: *FunnelHashTable, key: i64) struct { pos: i64, probes: usize } {
        var probes: usize = 0;
        const k: u64 = @bitCast(key);
        var t: usize = 0;
        while (t < self.b_attempts) : (t += 1) {
            const pos = self.b_offset + (@as(usize, @intCast(mix64(k ^ self.salt_b[t]) % self.b_size)));
            probes += 1;
            if (!self.occupied[pos]) {
                self.fill(pos, key);
                return .{ .pos = @intCast(pos), .probes = probes };
            }
        }
        const b1 = @as(usize, @intCast(mix64(k ^ self.salt_c1) % self.c_num_buckets));
        const b2 = @as(usize, @intCast(mix64(k ^ self.salt_c2) % self.c_num_buckets));
        var s: usize = 0;
        while (s < self.c_bucket_slots) : (s += 1) {
            const buckets = [_]usize{ b1, b2 };
            for (buckets) |bkt| {
                const pos = self.c_offset + bkt * self.c_bucket_slots + s;
                probes += 1;
                if (!self.occupied[pos]) {
                    self.fill(pos, key);
                    return .{ .pos = @intCast(pos), .probes = probes };
                }
            }
        }
        return .{ .pos = -1, .probes = probes };
    }

    /// Insert (key) without displacing any resident key. Returns (success, probes).
    pub fn insert(self: *FunnelHashTable, key: i64) struct { ok: bool, probes: usize } {
        if (self.count >= self.capacity) return .{ .ok = false, .probes = 0 };
        const k: u64 = @bitCast(key);
        var probes: usize = 0;
        const overflow_was_empty = (self.overflow_count == 0);
        var first_free: i64 = -1;

        for (self.slabs) |slab| {
            const base = slab.offset + (@as(usize, @intCast(mix64(k ^ slab.salt) % slab.subarray_count))) * self.beta;
            var sub_first_free: i64 = -1;
            var s: usize = 0;
            while (s < self.beta) : (s += 1) {
                const pos = base + s;
                probes += 1;
                if (self.occupied[pos]) {
                    if (self.keys[pos] == key) {
                        // Update in place (no value channel in this microbench).
                        return .{ .ok = true, .probes = probes };
                    }
                } else if (sub_first_free < 0) {
                    sub_first_free = @intCast(pos);
                }
            }
            if (sub_first_free >= 0) {
                if (overflow_was_empty) {
                    self.fill(@intCast(sub_first_free), key);
                    return .{ .ok = @as(usize, @intCast(sub_first_free)) >= 0, .probes = probes };
                }
                if (first_free < 0) first_free = sub_first_free;
            }
        }

        // Not found in any slab: check overflow for the key.
        const ov = self.searchOverflow(key);
        probes += ov.probes;
        if (ov.pos >= 0) return .{ .ok = true, .probes = probes };

        // Genuinely new key.
        if (first_free >= 0) {
            self.fill(@intCast(first_free), key);
            return .{ .ok = true, .probes = probes };
        }
        const placed = self.placeOverflow(key);
        probes += placed.probes;
        if (placed.pos >= 0) return .{ .ok = true, .probes = probes };
        return .{ .ok = false, .probes = probes };
    }

    /// Lookup key. Returns (found, probes).
    pub fn lookup(self: *FunnelHashTable, key: i64) struct { found: bool, probes: usize } {
        const k: u64 = @bitCast(key);
        var probes: usize = 0;
        for (self.slabs) |slab| {
            const base = slab.offset + (@as(usize, @intCast(mix64(k ^ slab.salt) % slab.subarray_count))) * self.beta;
            var s: usize = 0;
            while (s < self.beta) : (s += 1) {
                const pos = base + s;
                probes += 1;
                if (self.occupied[pos] and self.keys[pos] == key) return .{ .found = true, .probes = probes };
            }
        }
        const ov = self.searchOverflow(key);
        probes += ov.probes;
        return .{ .found = ov.pos >= 0, .probes = probes };
    }
};
