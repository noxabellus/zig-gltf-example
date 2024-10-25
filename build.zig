const std = @import("std");
const sdl = @import("sdl");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgl = b.dependency("zgl", .{
        .target = target,
        .optimize = optimize,
    });

    const zgltf = b.dependency("zgltf", .{
        .target = target,
        .optimize = optimize,
    });

    const zigUtils = b.dependency("ZigUtils", .{
        .target = target,
        .optimize = optimize,
    });

    const sdlSdk = sdl.init(b, .{});

    const zmath = b.addModule("zmath", .{
        .root_source_file = b.path("extern/zmath.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zgl", zgl.module("zgl"));
    exe.root_module.addImport("zgltf", zgltf.module("zgltf"));
    exe.root_module.addImport("ZigUtils", zigUtils.module("ZigUtils"));
    exe.root_module.addImport("zmath", zmath);

    sdlSdk.link(exe, .dynamic, sdl.Library.SDL2);

    exe.root_module.addImport("sdl", sdlSdk.getWrapperModule());

    const testExe = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    testExe.root_module.addImport("zgl", zgl.module("zgl"));
    testExe.root_module.addImport("zgltf", zgltf.module("zgltf"));
    testExe.root_module.addImport("ZigUtils", zigUtils.module("ZigUtils"));
    testExe.root_module.addImport("zmath", zmath);

    sdlSdk.link(testExe, .dynamic, sdl.Library.SDL2);

    testExe.root_module.addImport("sdl", sdlSdk.getWrapperModule());

    b.default_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    b.step("run", "Run the app").dependOn(&b.addRunArtifact(exe).step);

    b.step("check", "Semantic analysis").dependOn(&testExe.step);

    b.step("test", "Run tests").dependOn(&b.addRunArtifact(testExe).step);
}
