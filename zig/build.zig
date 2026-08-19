const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bench_exe = b.addExecutable(.{
        .name = "funnel_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(bench_exe);

    const run_cmd = b.addRunArtifact(bench_exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the funnel-hash microbenchmark");
    run_step.dependOn(&run_cmd.step);

    // Tests (none yet -- correctness parity vs Python is checked separately).
    const t = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("funnel_hash.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(t);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
