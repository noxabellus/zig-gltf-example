const std = @import("std");
const sdl = @import("sdl");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgl = b.dependency("zgl", .{
        .target = target,
        .optimize = optimize,
    });

    const sdlSdk = sdl.init(b, .{});

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zgl", zgl.module("zgl"));

    sdlSdk.link(exe, .dynamic, sdl.Library.SDL2);

    exe.root_module.addImport("sdl", sdlSdk.getWrapperModule());

    const testExe = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    testExe.root_module.addImport("zgl", zgl.module("zgl"));

    sdlSdk.link(testExe, .dynamic, sdl.Library.SDL2);

    testExe.root_module.addImport("sdl", sdlSdk.getWrapperModule());

    b.default_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    b.step("run", "Run the app").dependOn(&b.addRunArtifact(exe).step);

    b.step("check", "Semantic analysis").dependOn(&testExe.step);

    b.step("test", "Run tests").dependOn(&b.addRunArtifact(testExe).step);
}
