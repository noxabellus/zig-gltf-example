const std = @import("std");
const zgl = @import("zgl");
const sdl = @import("sdl");

pub fn main() !void {
    try sdl.init(.{
        .video = true,
        .events = true,
    });
    defer sdl.quit();

    var window = try sdl.createWindow(
        "zig-gltf",
        .centered, .centered,
        640, 480,
        .{.vis = .shown, .context = .opengl },
    );
    defer window.destroy();

    try sdl.gl.setAttribute(.{ .context_major_version = 3 });
    try sdl.gl.setAttribute(.{ .context_minor_version = 1 });
    try sdl.gl.setAttribute(.{ .context_profile_mask = .core });

    const glContext = try sdl.gl.createContext(window);
    defer glContext.delete();

    try zgl.loadExtensions({}, sdlGlGetProcAddressWrapper);

    try sdl.gl.setSwapInterval(.adaptive_vsync);

    mainLoop: while (true) {
        while (sdl.pollEvent()) |ev| {
            switch (ev) {
                .quit => break :mainLoop,
                else => {},
            }
        }

        zgl.clearColor(1.0, 0.0, 1.0, 1.0);
        zgl.clear(.{ .color = true, .depth = true });

        sdl.gl.swapWindow(window);
    }
}

fn sdlGlGetProcAddressWrapper(_: void, symbolName: [:0]const u8) ?zgl.binding.FunctionPointer {
    return sdl.gl.getProcAddress(symbolName);
}

test {
    std.testing.refAllDeclsRecursive(@This());
}