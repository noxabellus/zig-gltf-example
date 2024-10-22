const std = @import("std");
const zgl = @import("zgl");
const zgltf = @import("zgltf");
const sdl = @import("sdl");
const MiscUtils = @import("ZigUtils").Misc;

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

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();

    const dir = try std.fs.cwd().openDir("test-samples/rigged_simple/", .{});

    const gltfJSON = try dir.readFileAllocOptions(
        allocator,
        "RiggedSimple.gltf",
        512_000,
        null,
        4,
        null
    );
    defer allocator.free(gltfJSON);

    var gltf = zgltf.init(allocator);
    defer gltf.deinit();

    try gltf.parse(gltfJSON);

    gltf.debugPrint();

    for (gltf.data.nodes.items) |node| {
        const message =
            \\Node's name: {s}
            \\Children count: {}
            \\Have skin: {}
            \\
        ;

        std.debug.print(message, .{
            node.name,
            node.children.items.len,
            node.skin != null,
        });
    }


    // TODO: this doesnt need to be a hashmap
    var bufferMap = std.ArrayHashMap(usize, []const align(4) u8, MiscUtils.SimpleHashContext, false).init(allocator);
    defer bufferMap.deinit();

    for (gltf.data.buffers.items, 0..) |buffer, i| {
        const uri = buffer.uri.?;
        const DATA_URI_PREFIX = "data:application/gltf-buffer;base64,";

        const bin = if (std.mem.startsWith(u8, uri, DATA_URI_PREFIX)) base64: {
            const dataUri = uri[DATA_URI_PREFIX.len..];

            break :base64 MiscUtils.todo(noreturn, dataUri); // TODO: decode base64
        } else bin: {
            break :bin try dir.readFileAllocOptions(
                allocator,
                uri,
                5_000_000,
                null,
                4,
                null
            );
        };

        try bufferMap.put(i, bin);
    }

    const Vertex = struct {
        pos: struct { f32, f32, f32 },
        normal: struct { f32, f32, f32 },
        joints: struct { u16, u16, u16, u16 },
        weights: struct { f32, f32, f32, f32 },
    };

    var vertices = std.ArrayList(Vertex).init(allocator);
    defer vertices.deinit();

    var indices = std.ArrayList(u16).init(allocator);
    defer indices.deinit();

    const mesh = gltf.data.meshes.items[0];
    std.debug.print("Mesh name: {s}\n", .{mesh.name});

    for (mesh.primitives.items) |primitive| {
        {
            const indicesAccessor = gltf.data.accessors.items[primitive.indices.?];

            std.debug.print("indicesAccessor: {}\n", .{indicesAccessor});

            const bufferView = gltf.data.buffer_views.items[indicesAccessor.buffer_view.?];

            const indicesView = try indices.addManyAt(indices.items.len, @intCast(indicesAccessor.count));

            var it = IndexedAccessorIterator(u16).init(&gltf, indicesAccessor, bufferMap.get(bufferView.buffer).?);
            while (it.next()) |x| {
                const k, const i = x;
                indicesView[i] = k[0];
            }
        }

        const verticesView = view: {
            const firstAccessor = gltf.data.accessors.items[extractAttributeIndex(primitive.attributes.items[0])];
            
            std.debug.print("Vertices input count: {}\n", .{firstAccessor.count});
            
            break :view try vertices.addManyAt(vertices.items.len, @intCast(firstAccessor.count));
        };

        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    
                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap.get(bufferView.buffer).?);
                    while (it.next()) |x| {
                        const v, const i = x;
                        verticesView[i].pos = .{ v[0], v[1], v[2] };
                    }
                },
                .normal => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    
                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap.get(bufferView.buffer).?);
                    while (it.next()) |x| {
                        const n, const i = x;
                        verticesView[i].normal = .{ n[0], n[1], n[2] };
                    }
                },
                .joints => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    
                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(u16).init(&gltf, accessor, bufferMap.get(bufferView.buffer).?);
                    while (it.next()) |x| {
                        const j, const i = x;
                        verticesView[i].joints = .{ j[0], j[1], j[2], j[3] };
                    }
                },
                .weights => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    
                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap.get(bufferView.buffer).?);
                    while (it.next()) |x| {
                        const w, const i = x;
                        verticesView[i].weights = .{ w[0], w[1], w[2], w[3] };
                    }
                },
                else => {
                    std.debug.print("Unhandled attribute: {}\n", .{attribute});
                },
            }
        }
    }

    std.debug.print("Vertices count: {}\n", .{vertices.items.len});
    std.debug.print("Indices count: {}\n", .{indices.items.len});


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

fn IndexedAccessorIterator(comptime T: type) type {
    return struct {
        inner: IndexedIterator(zgltf.AccessorIterator(T)),

        const Self = @This();

        pub fn init(gltf: *const zgltf, accessor: zgltf.Accessor, bin: []align(4) const u8) Self {
            return .{ .inner = IndexedIterator(zgltf.AccessorIterator(T)).from(accessor.iterator(T, gltf, bin)) };
        }

        pub inline fn next(self: *Self) ?struct { []const T, usize } {
            return self.inner.next();
        }

        pub inline fn peek(self: *const Self) ?struct { []const T, usize } {
            return self.inner.peek();
        }

        pub inline fn reset(self: *Self) void {
            self.inner.reset();
        }
    };
}

fn IndexedIterator(comptime T: type) type {
    return struct {
        inner: T,
        index: usize = 0,

        const Self = @This();
        const InnerResult = @typeInfo(@typeInfo(@TypeOf(T.next)).@"fn".return_type.?).optional.child;
        const Result = struct { InnerResult, usize };

        pub fn from(inner: T) Self {
            return .{ .inner = inner };
        }

        pub fn next(self: *Self) ?Result {
            const innerNext = self.inner.next() orelse return null;

            const out = Result { innerNext, self.index };
            
            self.index += 1;

            return out;
        }

        pub fn peek(self: *const Self) ?Result {
            const innerPeek = self.inner.peek() orelse return null;

            return Result { innerPeek, self.index };
        }

        pub fn reset(self: *Self) void {
            self.inner.reset();
            self.index = 0;
        }
    };
}

fn extractAttributeIndex(attribute: zgltf.Attribute) zgltf.Index {
    return switch (attribute) {
        .position => |idx| idx,
        .normal => |idx| idx,
        .tangent => |idx| idx,
        .texcoord => |idx| idx,
        .color => |idx| idx,
        .joints => |idx| idx,
        .weights => |idx| idx,
    };
}

fn sdlGlGetProcAddressWrapper(_: void, symbolName: [:0]const u8) ?zgl.binding.FunctionPointer {
    return sdl.gl.getProcAddress(symbolName);
}

test {
    std.testing.refAllDeclsRecursive(@This());
}