const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const testExe = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
    });

    b.default_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    b.step("run", "Run the app").dependOn(&b.addRunArtifact(exe).step);

    b.step("check", "Semantic analysis").dependOn(&testExe.step);

    b.step("test", "Run tests").dependOn(&b.addRunArtifact(testExe).step);
}
