const std = @import("std");
const math = std.math;
const zgl = @import("zgl");
const zgltf = @import("zgltf");
const zmath = @import("zmath.zig");
const sdl = @import("sdl");
const MiscUtils = @import("ZigUtils").Misc;

const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;
const ASPECT_RATIO = @as(f32, WINDOW_WIDTH) / @as(f32, WINDOW_HEIGHT);

pub fn main() !void {
    try sdl.init(.{
        .video = true,
        .events = true,
    });
    defer sdl.quit();

    var window = try sdl.createWindow(
        "zig-gltf",
        .centered, .centered,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        .{.vis = .shown, .context = .opengl },
    );
    defer window.destroy();

    try sdl.gl.setAttribute(.{ .context_major_version = 3 });
    try sdl.gl.setAttribute(.{ .context_minor_version = 3 });
    try sdl.gl.setAttribute(.{ .context_profile_mask = .core });

    const glContext = try sdl.gl.createContext(window);
    defer glContext.delete();

    try zgl.loadExtensions({}, sdlGlGetProcAddressWrapper);

    try sdl.gl.setSwapInterval(.adaptive_vsync);

    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();

    const samplesDir = try std.fs.cwd().openDir("test-samples", .{});
    const modelDir = try samplesDir.openDir("khronos", .{});
    const shaderDir = try samplesDir.openDir("shaders", .{});

    const gltfJSON = try modelDir.readFileAllocOptions(
        allocator,
        "khronos.gltf",
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


    var bufferMap = try allocator.alloc([]align(4) const u8, gltf.data.buffers.items.len);
    defer {
        for (bufferMap) |buffer| allocator.free(buffer);
        allocator.free(bufferMap);
    }

    for (gltf.data.buffers.items, 0..) |buffer, i| {
        const uri = buffer.uri.?;
        const DATA_URI_PREFIX = "data:application/gltf-buffer;base64,";

        const bin = if (std.mem.startsWith(u8, uri, DATA_URI_PREFIX)) base64: {
            const dataUri = uri[DATA_URI_PREFIX.len..];

            const decoder = std.base64.standard.Decoder;
            const upperBound = try decoder.calcSizeUpperBound(dataUri.len);
            const bin = try allocator.allocWithOptions(u8, upperBound, 4, null);
            try decoder.decode(bin, dataUri);
            break :base64 bin;
        } else bin: {
            break :bin try modelDir.readFileAllocOptions(
                allocator,
                uri,
                5_000_000,
                null,
                4,
                null
            );
        };

        bufferMap[i] = bin;
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
    std.debug.assert(gltf.data.meshes.items.len == 1);
    std.debug.print("Mesh name: {s}\n", .{mesh.name});

    for (mesh.primitives.items) |primitive| {
        {
            const indicesAccessor = gltf.data.accessors.items[primitive.indices.?];

            std.debug.print("indicesAccessor: {}\n", .{indicesAccessor});

            const bufferView = gltf.data.buffer_views.items[indicesAccessor.buffer_view.?];

            const indicesView = try indices.addManyAt(indices.items.len, @intCast(indicesAccessor.count));

            var it = IndexedAccessorIterator(u16).init(&gltf, indicesAccessor, bufferMap[bufferView.buffer]);
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

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const v, const i = x;
                        verticesView[i].pos = .{ v[0], v[1], v[2] };
                    }
                },
                .normal => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const n, const i = x;
                        verticesView[i].normal = .{ n[0], n[1], n[2] };
                    }
                },
                .joints => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(u16).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const j, const i = x;
                        verticesView[i].joints = .{ j[0], j[1], j[2], j[3] };
                    }
                },
                .weights => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
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


    const vertexSource = try shaderDir.readFileAlloc(allocator, "basic.vert", math.maxInt(u16));
    defer allocator.free(vertexSource);

    std.debug.print("loaded vert src: ```\n{s}\n```\n", .{vertexSource});

    const fragmentSource = try shaderDir.readFileAlloc(allocator, "basic.frag", math.maxInt(u16));
    defer allocator.free(fragmentSource);

    std.debug.print("loaded frag src: ```\n{s}\n```\n", .{fragmentSource});


    const vertexShader = zgl.createShader(.vertex);
    vertexShader.source(1, &[1][]const u8 { vertexSource });
    vertexShader.compile();
    defer vertexShader.delete();

    if (vertexShader.get(.compile_status) == 0) {
        const info = try vertexShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Vertex shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const fragmentShader = zgl.createShader(.fragment);
    fragmentShader.source(1, &[1][]const u8 { fragmentSource });
    fragmentShader.compile();
    defer fragmentShader.delete();

    if (fragmentShader.get(.compile_status) == 0) {
        const info = try fragmentShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Fragment shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const shaderProgram = zgl.createProgram();
    shaderProgram.attach(vertexShader);
    shaderProgram.attach(fragmentShader);
    shaderProgram.link();
    defer shaderProgram.delete();

    if (shaderProgram.get(.link_status) == 0) {
        const info = try shaderProgram.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Shader program info log: ```\n{s}\n```\n", .{info});

        return error.ShaderLinkingFailed;
    }

    shaderProgram.use();

    const matrixLocation = shaderProgram.uniformLocation("modelToClip") orelse return error.MissingUniform;


    const vao = zgl.createVertexArray();
    vao.bind();

    const vbo = zgl.createBuffer();
    vbo.bind(.array_buffer);
    vbo.data(Vertex, vertices.items, .static_draw);

    const ebo = zgl.createBuffer();
    ebo.bind(.element_array_buffer);
    ebo.data(u16, indices.items, .static_draw);

    // // position attribute
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);
    // // normal attribute
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3* sizeof(float)));
    // glEnableVertexAttribArray(1);


    const modelToWorld = zmath.rotationY(0);
    const worldToView = zmath.lookAtRh(
        zmath.f32x4(3.0, 3.0, 3.0, 1.0), // eye position
        zmath.f32x4(0.0, 0.0, 0.0, 1.0), // focus point
        zmath.f32x4(0.0, 1.0, 0.0, 0.0), // up direction ('w' coord is zero because this is a vector not a point)
    );

    const viewToClip = zmath.perspectiveFovRhGl(0.25 * math.pi, ASPECT_RATIO, 0.1, 20.0);

    const modelToView = zmath.mul(modelToWorld, worldToView);
    const modelToClip = zmath.mul(modelToView, viewToClip);

    // Transposition is needed because GLSL uses column-major matrices by default
    shaderProgram.uniformMatrix4(matrixLocation, true, @as([*]const [4][4]f32, @ptrCast(zmath.arrNPtr(&modelToClip)))[0..1]);


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
