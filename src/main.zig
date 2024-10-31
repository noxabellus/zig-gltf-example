const std = @import("std");
const math = std.math;
const zgl = @import("zgl");
const zgltf = @import("zgltf");
const zmath = @import("zmath");
const ZigUtils = @import("ZigUtils");
const sdl = @import("sdl");

const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;
const ASPECT_RATIO = @as(f32, WINDOW_WIDTH) / @as(f32, WINDOW_HEIGHT);

const Pos = [3]f32;

const Vertex = extern struct {
    pos: Pos,
    normal: Normal,
    joints: Joints,
    weights: Weights,

    pub const Normal = [3]f32;
    pub const Joints = [4]u16;
    pub const Weights = [4]f32;
};

const Node = struct {
    pos: Pos,
    rot: zmath.Quat,
    scl: Scale,
    inv: InverseBindMatrix,

    // TODO: hierarchy

    pub const Scale = [3]f32;
    pub const InverseBindMatrix = [16]f32;
};


pub fn main() !void {
    var samplesDir = try std.fs.cwd().openDir("test-samples", .{});
    defer samplesDir.close();


    // setup allocator //
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();


    // setup sdl //
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


    // setup opengl //
    const glContext = try sdl.gl.createContext(window);
    defer glContext.delete();

    try zgl.loadExtensions({}, sdlGlGetProcAddressWrapper);

    try sdl.gl.setSwapInterval(.adaptive_vsync);


    // load model file //
    var modelDir = try samplesDir.openDir("khronos", .{});
    defer modelDir.close();

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


    // load mesh data //
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


    // load skin data //
    var nodes = std.ArrayList(Node).init(allocator);
    defer nodes.deinit();

    const skin = gltf.data.skins.items[0];
    std.debug.assert(gltf.data.skins.items.len == 1);
    std.debug.print("skin name: {s}\n", .{skin.name});

    { // bind matrices
        const accessor = gltf.data.accessors.items[skin.inverse_bind_matrices.?];

        _ = try nodes.addManyAt(0, @intCast(accessor.count));

        const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

        var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
        while (it.next()) |x| {
            const b, const i = x;

            // TODO: is this the right format? or do we need to transpose?
            nodes.items[i].inv = .{
                b[0],  b[1],  b[2],  b[3],
                b[4],  b[5],  b[6],  b[7],
                b[8],  b[9],  b[10], b[11],
                b[12], b[13], b[14], b[15],
            };
        }
    }

    { // nodes
        const jointIndices = skin.joints.items;

        std.debug.assert(jointIndices.len == nodes.items.len);

        const nodesView = gltf.data.nodes.items;

        // TODO: we need to save the hierarchy, but
        // if there are children that are not in jointIndices, ... error?

        for (jointIndices, 0..) |j, i| {
            const node = nodesView[j];

            nodes.items[i].pos = node.translation;
            nodes.items[i].rot = .{ node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3] };
            nodes.items[i].scl = node.scale;
        }
    }


    // load animation data //
    const animation = gltf.data.animations.items[0];
    std.debug.assert(gltf.data.animations.items.len == 1);

    { // samplers
        const samplers = animation.samplers.items;

        for (samplers, 0..) |sampler, s| {
            std.debug.print("sampler {}: {}\n", .{s, sampler});

            std.debug.assert(sampler.interpolation == .linear);

            const inputAccessor = gltf.data.accessors.items[sampler.input];
            const outputAccessor = gltf.data.accessors.items[sampler.output];

            const inputBufferView = gltf.data.buffer_views.items[inputAccessor.buffer_view.?];
            const outputBufferView = gltf.data.buffer_views.items[outputAccessor.buffer_view.?];

            var inputIt = IndexedAccessorIterator(f32).init(&gltf, inputAccessor, bufferMap[inputBufferView.buffer]);
            var outputIt = outputAccessor.iterator(f32, &gltf, bufferMap[outputBufferView.buffer]);

            while (inputIt.next()) |x| {
                const input, const i = x;

                const output = outputIt.next() orelse return error.MissingOutput;

                std.debug.print("\tindex {}\n", .{i});
                std.debug.print("\t\tinput: {any}\n", .{input});
                std.debug.print("\t\toutput: {any}\n", .{output});
            }
        }
    }

    // TODO: create in memory animation format
    { // channels
        const channels = animation.channels.items;

        for (channels, 0..) |channel, c| {
            std.debug.print("channel {}: {}\n", .{c, channel});
        }
    }


    // load shaders //
    var shaderDir = try samplesDir.openDir("shaders", .{});
    defer shaderDir.close();

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

    const uniforms = .{
        .model = shaderProgram.uniformLocation("model") orelse return error.MissingUniform,
        .view = shaderProgram.uniformLocation("view") orelse return error.MissingUniform,
        .projection = shaderProgram.uniformLocation("projection") orelse return error.MissingUniform,
        .color = shaderProgram.uniformLocation("color") orelse return error.MissingUniform,
    };

    shaderProgram.use();


    // bind mesh data //
    const meshVao = zgl.createVertexArray();
    {
        const vbo = zgl.createBuffer();
        meshVao.vertexBuffer(0, vbo, 0, @sizeOf(Vertex));
        vbo.data(Vertex, vertices.items, .static_draw);

        const ebo = zgl.createBuffer();
        meshVao.elementBuffer(ebo);
        ebo.data(u16, indices.items, .static_draw);

        meshVao.attribFormat(0, 3, .float, true, @offsetOf(Vertex, "pos"));
        meshVao.attribBinding(0, 0);
        meshVao.enableVertexAttribute(0);
    }


    // bind skin data //
    var skinVertices = std.ArrayList(Pos).init(allocator);
    defer skinVertices.deinit();
    {
        var parentMatrix = zmath.identity();

        for (nodes.items) |node| {
            const pos = zmath.translation(node.pos[0], node.pos[1], node.pos[2]);
            const rot = zmath.quatToMat(node.rot);
            const scl = zmath.scaling(node.scl[0], node.scl[1], node.scl[2]);

            const localMatrix = zmath.mul(zmath.mul(pos, rot), scl);
            const worldMatrix = zmath.mul(parentMatrix, localMatrix);

            const worldPos = zmath.mul(zmath.f32x4(0.0, 0.0, 0.0, 1.0), worldMatrix);

            try skinVertices.append(.{ worldPos[0], worldPos[1], worldPos[2] });

            parentMatrix = worldMatrix;
        }
    }

    const skinVao = zgl.createVertexArray();
    {
        const vbo = zgl.createBuffer();
        skinVao.vertexBuffer(0, vbo, 0, @sizeOf(Pos));
        vbo.data(Pos, skinVertices.items, .static_draw);

        skinVao.attribFormat(0, 3, .float, false, 0);
        skinVao.attribBinding(0, 0);
        skinVao.enableVertexAttribute(0);
    }


    // bind matrix data //
    const matrices = .{
        .model = zmath.identity(),
        .view = zmath.lookAtLh(
            zmath.f32x4(0.0, 1.0, 3.0, 1.0), // eye position
            zmath.f32x4(0.0, 1.0, 0.0, 1.0), // focus point
            zmath.f32x4(0.0, 1.0, 0.0, 0.0), // up direction ('w' coord is zero because this is a vector not a point)
        ),
        .projection = zmath.perspectiveFovLhGl(0.25 * math.pi, ASPECT_RATIO, 0.1, 20.0),
    };

    shaderProgram.uniformMatrix4(uniforms.model, false, @as([*]const [4][4]f32, @ptrCast(zmath.arrNPtr(&matrices.model)))[0..1]);
    shaderProgram.uniformMatrix4(uniforms.view, false, @as([*]const [4][4]f32, @ptrCast(zmath.arrNPtr(&matrices.view)))[0..1]);
    shaderProgram.uniformMatrix4(uniforms.projection, false, @as([*]const [4][4]f32, @ptrCast(zmath.arrNPtr(&matrices.projection)))[0..1]);


    // run //
    mainLoop: while (true) {
        while (sdl.pollEvent()) |ev| {
            switch (ev) {
                .quit => break :mainLoop,
                else => {},
            }
        }

        zgl.clearColor(0.0, 0.0, 0.0, 1.0);
        zgl.clear(.{ .color = true, .depth = true });

        zgl.pointSize(10.0);
        zgl.polygonMode(.front_and_back, .line);

        meshVao.bind();
        shaderProgram.uniform3f(uniforms.color, 1.0, 0.5, 0.2);
        zgl.drawElements(.triangles, indices.items.len, .unsigned_short, 0);

        skinVao.bind();
        shaderProgram.uniform3f(uniforms.color, 0.0, 1.0, 0.0);
        zgl.drawArrays(.lines, 0, skinVertices.items.len);

        shaderProgram.uniform3f(uniforms.color, 1.0, 1.0, 1.0);
        zgl.drawArrays(.points, 0, skinVertices.items.len);

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
