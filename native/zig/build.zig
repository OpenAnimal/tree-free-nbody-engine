const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Root module
    const mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Shared / dynamic library (.dll on Windows, .so on Linux)
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "tree_free_fmm_native",
        .root_module = mod,
    });
    b.installArtifact(lib);

    // Static library (.lib / .a for UE5 / embedded C++ linking)
    const static_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "tree_free_fmm_static",
        .root_module = mod,
    });
    b.installArtifact(static_lib);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run native unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
