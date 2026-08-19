// Microbenchmark for the Zig funnel hash port (zig/funnel_hash.zig).
//
// Mirrors the Python reference (core/elastic_hash.py:ElasticHashTable) on
// the same (capacity, delta) geometry and reports:
//   - insert throughput (M keys/s) and mean probes/insert
//   - lookup throughput (M keys/s) and mean probes/lookup, hit rate
//   - the deterministic probe_bound (alpha*beta + O(log log n))
//
// The workload is N distinct u64 keys derived from a splitmix64 sequence
// (deterministic, no libc rand dependency). The keys are inserted at the
// table's rated capacity (load 1 - delta), then looked up in a shuffled
// order (every key is queried once, hits expected = 100%).
//
// Run with:  zig build run --release=fast
// (or)       zig build run --release=fast -- --capacity 1000000 --delta 0.05

const std = @import("std");
const funnel = @import("funnel_hash.zig");

const Args = struct {
    capacity: usize = 1_000_000,
    delta: f64 = 0.05,
    seed: u64 = 42,
};

fn parseArgs(argv: []const []const u8) Args {
    var a = Args{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--capacity") and i + 1 < argv.len) {
            i += 1;
            a.capacity = std.fmt.parseInt(usize, argv[i], 10) catch a.capacity;
        } else if (std.mem.eql(u8, arg, "--delta") and i + 1 < argv.len) {
            i += 1;
            a.delta = std.fmt.parseFloat(f64, argv[i]) catch a.delta;
        } else if (std.mem.eql(u8, arg, "--seed") and i + 1 < argv.len) {
            i += 1;
            a.seed = std.fmt.parseInt(u64, argv[i], 10) catch a.seed;
        }
    }
    return a;
}

// splitmix64 to generate distinct keys.
fn nextKey(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z: u64 = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z = z ^ (z >> 31);
    return z;
}

// Portable monotonic timer (Zig 0.16 removed std.time.Timer).
// On Windows uses RtlQueryPerformanceCounter; on POSIX uses clock_gettime.
const Timer = struct {
    freq_ns_per_tick: f64,
    start_ticks: u64,

    pub fn start() Timer {
        if (@import("builtin").os.tag == .windows) {
            const ntdll = std.os.windows.ntdll;
            var freq: std.os.windows.LARGE_INTEGER = 0;
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            var pc: std.os.windows.LARGE_INTEGER = 0;
            _ = ntdll.RtlQueryPerformanceCounter(&pc);
            return .{
                .freq_ns_per_tick = 1e9 / @as(f64, @floatFromInt(freq)),
                .start_ticks = @intCast(pc),
            };
        } else {
            // POSIX fallback: clock_gettime(CLOCK_MONOTONIC).
            var ts: std.posix.timespec = undefined;
            std.posix.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) catch return .{ .freq_ns_per_tick = 1.0, .start_ticks = 0 };
            const ticks = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
            return .{ .freq_ns_per_tick = 1.0, .start_ticks = ticks };
        }
    }

    pub fn reset(self: *Timer) void {
        if (@import("builtin").os.tag == .windows) {
            var pc: std.os.windows.LARGE_INTEGER = 0;
            _ = std.os.windows.ntdll.RtlQueryPerformanceCounter(&pc);
            self.start_ticks = @intCast(pc);
        } else {
            var ts: std.posix.timespec = undefined;
            std.posix.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) catch return;
            self.start_ticks = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
        }
    }

    pub fn readNs(self: Timer) u64 {
        if (@import("builtin").os.tag == .windows) {
            var pc: std.os.windows.LARGE_INTEGER = 0;
            _ = std.os.windows.ntdll.RtlQueryPerformanceCounter(&pc);
            const now_ticks: u64 = @intCast(pc);
            const delta_ticks = now_ticks - self.start_ticks;
            return @intFromFloat(@as(f64, @floatFromInt(delta_ticks)) * self.freq_ns_per_tick);
        } else {
            var ts: std.posix.timespec = undefined;
            std.posix.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts) catch return 0;
            const now_ticks = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
            return now_ticks - self.start_ticks;
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    // Zig 0.16 args: use Args.toSlice with an arena for cross-platform
    // (Windows WTF-16 -> UTF-8) arg parsing.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();
    const argv_slice = try std.process.Args.toSlice(init.minimal.args, arena_alloc);
    // argv_slice[0] is the program name; parse the rest.
    var arg_buf: [16][]const u8 = undefined;
    var arg_count: usize = 0;
    for (argv_slice[1..]) |a| {
        if (arg_count >= arg_buf.len) break;
        arg_buf[arg_count] = a;
        arg_count += 1;
    }
    const args = parseArgs(arg_buf[0..arg_count]);

    std.debug.print("Zig funnel-hash microbenchmark\n", .{});
    std.debug.print("  capacity = {d}\n", .{args.capacity});
    std.debug.print("  delta    = {d:.4}\n", .{args.delta});
    std.debug.print("  seed     = {d}\n", .{args.seed});

    var table = try funnel.FunnelHashTable.init(
        allocator, args.capacity, args.delta, args.seed,
    );
    defer table.deinit();

    std.debug.print("  geometry: alpha={d} beta={d} total_slots={d} probe_bound={d}\n", .{
        table.alpha, table.beta, table.total_size, table.probeBound(),
    });
    std.debug.print("  load factor at capacity = {d:.4}\n", .{
        @as(f64, @floatFromInt(args.capacity)) / @as(f64, @floatFromInt(table.total_size)),
    });

    // Generate distinct keys.
    const keys = try allocator.alloc(u64, args.capacity);
    defer allocator.free(keys);
    var ks: u64 = args.seed;
    for (keys) |*k| k.* = nextKey(&ks);

    // ---- Insert benchmark ----
    var insert_probes: u64 = 0;
    var insert_ok: usize = 0;
    var timer = Timer.start();
    for (keys) |k| {
        const r = table.insert(@bitCast(k));
        insert_probes += r.probes;
        if (r.ok) insert_ok += 1;
    }
    const insert_ns = timer.readNs();
    const insert_s = @as(f64, @floatFromInt(insert_ns)) / 1e9;
    const insert_mps = @as(f64, @floatFromInt(insert_ok)) / insert_s / 1e6;
    const insert_mean_probes = @as(f64, @floatFromInt(insert_probes)) / @as(f64, @floatFromInt(args.capacity));

    std.debug.print("\n--- INSERT ---\n", .{});
    std.debug.print("  inserted      = {d} / {d}\n", .{ insert_ok, args.capacity });
    std.debug.print("  time          = {d:.4} s\n", .{insert_s});
    std.debug.print("  throughput    = {d:.2} M keys/s\n", .{insert_mps});
    std.debug.print("  mean probes   = {d:.4}\n", .{insert_mean_probes});
    std.debug.print("  final load    = {d:.4}\n", .{
        @as(f64, @floatFromInt(table.count)) / @as(f64, @floatFromInt(table.total_size)),
    });
    std.debug.print("  overflow used = {d}\n", .{table.overflow_count});

    // ---- Lookup benchmark (shuffled order) ----
    // Fisher-Yates shuffle the keys for the lookup pass.
    var shuffle = try allocator.alloc(u64, args.capacity);
    defer allocator.free(shuffle);
    @memcpy(shuffle, keys);
    var prng = std.Random.DefaultPrng.init(args.seed ^ 0xDEADBEEF);
    const rnd = prng.random();
    var i: usize = args.capacity;
    while (i > 1) {
        i -= 1;
        const j = rnd.intRangeLessThan(usize, 0, i + 1);
        const tmp = shuffle[i];
        shuffle[i] = shuffle[j];
        shuffle[j] = tmp;
    }

    var lookup_probes: u64 = 0;
    var lookup_hits: usize = 0;
    timer.reset();
    for (shuffle) |k| {
        const r = table.lookup(@bitCast(k));
        lookup_probes += r.probes;
        if (r.found) lookup_hits += 1;
    }
    const lookup_ns = timer.readNs();
    const lookup_s = @as(f64, @floatFromInt(lookup_ns)) / 1e9;
    const lookup_mps = @as(f64, @floatFromInt(args.capacity)) / lookup_s / 1e6;
    const lookup_mean_probes = @as(f64, @floatFromInt(lookup_probes)) / @as(f64, @floatFromInt(args.capacity));
    const hit_rate = @as(f64, @floatFromInt(lookup_hits)) / @as(f64, @floatFromInt(args.capacity));

    std.debug.print("\n--- LOOKUP (shuffled, all keys queried once) ---\n", .{});
    std.debug.print("  hits          = {d} / {d}  (hit rate = {d:.6})\n", .{
        lookup_hits, args.capacity, hit_rate,
    });
    std.debug.print("  time          = {d:.4} s\n", .{lookup_s});
    std.debug.print("  throughput    = {d:.2} M keys/s\n", .{lookup_mps});
    std.debug.print("  mean probes   = {d:.4}\n", .{lookup_mean_probes});

    // ---- Absent-key lookup benchmark (keys never inserted) ----
    var absent_probes: u64 = 0;
    var absent_false_hits: usize = 0;
    timer.reset();
    var as: u64 = args.seed ^ 0xCAFEBABE;
    for (0..args.capacity) |_| {
        const k = nextKey(&as) | 0x8000000000000000; // high bit set: distinct space
        const r = table.lookup(@bitCast(k));
        absent_probes += r.probes;
        if (r.found) absent_false_hits += 1;
    }
    const absent_ns = timer.readNs();
    const absent_s = @as(f64, @floatFromInt(absent_ns)) / 1e9;
    const absent_mps = @as(f64, @floatFromInt(args.capacity)) / absent_s / 1e6;
    const absent_mean_probes = @as(f64, @floatFromInt(absent_probes)) / @as(f64, @floatFromInt(args.capacity));

    std.debug.print("\n--- LOOKUP (absent keys, high-bit space) ---\n", .{});
    std.debug.print("  false hits    = {d}  (must be 0)\n", .{absent_false_hits});
    std.debug.print("  time          = {d:.4} s\n", .{absent_s});
    std.debug.print("  throughput    = {d:.2} M keys/s\n", .{absent_mps});
    std.debug.print("  mean probes   = {d:.4}  (worst case = probe_bound {d})\n", .{
        absent_mean_probes, table.probeBound(),
    });

    // ---- Verdict ----
    const ok = (insert_ok == args.capacity) and (lookup_hits == args.capacity) and (absent_false_hits == 0);
    std.debug.print("\n=== {s} (insert_ok={d}, lookup_hits={d}, absent_false_hits={d}) ===\n", .{
        if (ok) "PASS" else "FAIL",
        insert_ok, lookup_hits, absent_false_hits,
    });
    if (!ok) std.process.exit(1);
}
