// config //
const WINDOW_WIDTH = 1024;
const WINDOW_HEIGHT = 768;
const ASPECT_RATIO = @as(f32, WINDOW_WIDTH) / @as(f32, WINDOW_HEIGHT);
const USE_WIREFRAME = false;
const RENDER_BONE_POSITIONS = false;
const COLOR = .{ 0.761, 0.506, 0.184 };
const MESH_FRAGMENT: enum { basic, visualize_weights, lit } = .lit;
const EXAMPLE: enum { khronos, rigged_simple } = .rigged_simple;


// imports //
const std = @import("std");
const math = std.math;
const zgl = @import("zgl");
const zgltf = @import("zgltf");
const zmath = @import("zmath");
const ZigUtils = @import("ZigUtils");
const sdl = @import("sdl");


// example data //
const NUM_EXAMPLES = std.meta.fields(@TypeOf(EXAMPLE)).len;

const FOLDERS = [NUM_EXAMPLES][]const u8 {
    "khronos",
    "rigged_simple",
};

const ROOT_FILES = [NUM_EXAMPLES][]const u8 {
    "khronos.gltf",
    "RiggedSimple.gltf",
};

var VIEWS = [NUM_EXAMPLES]Mat4 {
    zmath.lookAtLh(
        zmath.f32x4(0.0, 1.0, -6.0, 1.0), // eye position
        zmath.f32x4(0.0, 1.0,  0.0, 1.0), // focus point
        zmath.f32x4(0.0, 1.0,  0.0, 0.0), // up direction ('w' coord is zero because this is a vector not a point)
    ),
    zmath.lookAtLh(
        zmath.f32x4(20.0, 0.0, 5.0, 1.0),
        zmath.f32x4( 0.0, 0.0, 5.0, 1.0),
        zmath.f32x4( 0.0, 0.0, 1.0, 0.0),
    ),
};

var LIGHT_DIRS = [NUM_EXAMPLES]Vec3Arr {
    .{0.0, 0.0, 1.0},
    .{-20.0, 20.0, 10.0},
};


// types //
const Mat4 = zmath.Mat4;
const Mat4Arr = [4][4]f32;
const Vec4 = zmath.Vec4;
const Vec4Arr = [4]f32;
const Quat = zmath.Quat;
const QuatArr = [4]f32;
const Vec3 = @Vector(3, f32);
const Vec3Arr = [3]f32;

const Vertex = extern struct {
    pos: Pos = .{0.0, 0.0, 0.0},
    normal: Normal = .{0.0, 0.0, 0.0},
    weights: Weights = .{0.0, 0.0, 0.0, 0.0},
    joints: Joints = .{0, 0, 0, 0},

    pub const Pos = [3]f32;
    pub const Normal = [3]f32;
    pub const Weights = [4]f32;
    pub const Joints = [4]BoneIndex;
};

const Transform = struct {
    pos: Vec3,
    rot: Quat,
    scl: Vec3,

    fn computeMatrix(self: *const Transform) Mat4 {
        const pos = zmath.translation(self.pos[0], self.pos[1], self.pos[2]);
        const rot = zmath.quatToMat(self.rot);
        const scl = zmath.scaling(self.scl[0], self.scl[1], self.scl[2]);

        return zmath.mul(zmath.mul(scl, rot), pos);
    }
};

const Bone = struct {
    tran: Transform,
    inv: zmath.Mat4,
    parent: ?BoneIndex,
    children: [MAX_BONES]BoneIndex,
    num_children: BoneIndex,
};

const BoneIndex = u8;
const MAX_BONES = std.math.maxInt(BoneIndex);

const Skeleton = struct {
    bones: [MAX_BONES]Bone,
    num_bones: BoneIndex,

    fn getBone(self: *const Skeleton, boneIndex: BoneIndex) ?*const Bone {
        return if (boneIndex < self.num_bones) &self.bones[boneIndex] else null;
    }

    fn addBones(self: *Skeleton, map: *NodeBoneMap, nodes: []const zgltf.Node, nodeIndex: usize, parentBoneIndex: ?BoneIndex) !BoneIndex {
        if (map.contains(nodeIndex)) return error.DuplicateBone;

        const boneIndex = self.num_bones;
        self.num_bones += 1;

        try map.put(nodeIndex, boneIndex);

        const node = &nodes[nodeIndex];
        const bone = &self.bones[boneIndex];

        bone.tran = .{
            .pos = node.translation,
            .rot = .{ node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3] },
            .scl = node.scale,
        };

        bone.parent = parentBoneIndex;
        bone.num_children = 0;

        for (nodes[nodeIndex].children.items) |childNodeIndex| {
            const childBoneIndex = try self.addBones(map, nodes, childNodeIndex, boneIndex);
            const childLocalIndex = bone.num_children;
            bone.num_children += 1;
            bone.children[childLocalIndex] = childBoneIndex;
        }

        return boneIndex;
    }

    fn generateBoneVertices(self: *const Skeleton, output: *std.ArrayList([3]f32), parentMatrix: Mat4, boneIndex: BoneIndex) !void {
        const bone = self.bones[boneIndex];

        const localMatrix = bone.tran.computeMatrix();

        const worldMatrix = zmath.mul(parentMatrix, localMatrix);

        const worldPos = zmath.mul(zmath.f32x4(0.0, 0.0, 0.0, 1.0), worldMatrix);

        try output.append(.{ worldPos[0], worldPos[1], worldPos[2] });

        for (bone.children[0..bone.num_children]) |childBoneIndex| {
            try self.generateBoneVertices(output, worldMatrix, childBoneIndex);
        }
    }
};

const NodeBoneMap = std.ArrayHashMap(usize, BoneIndex, ZigUtils.Misc.SimpleHashContext, false);

const Animation = struct {
    bones: [MAX_BONES]?BoneAnimation = [1]?BoneAnimation { null } ** MAX_BONES,

    fn deinit(self: *Animation, allocator: std.mem.Allocator) void {
        for (0..MAX_BONES) |boneIndex| {
            if (self.bones[boneIndex]) |*b| b.deinit(allocator);
        }
    }

    fn getBone(self: *const Animation, boneIndex: BoneIndex) ?*const BoneAnimation {
        return if (self.bones[boneIndex]) |*bone| bone else null;
    }

    fn getOrInitBone(self: *Animation, boneIndex: BoneIndex) *BoneAnimation {
        if (self.bones[boneIndex]) |*bone| return bone;

        self.bones[boneIndex] = BoneAnimation {};
        return &self.bones[boneIndex].?;
    }

    fn computeBoneTransform(self: *const Animation, skeleton: *const Skeleton, boneIndex: BoneIndex, time: f32) !Transform {
        var base = (skeleton.getBone(boneIndex) orelse return error.InvalidBoneIndex).tran;

        if (self.getBone(boneIndex)) |boneAnim| {
            if (boneAnim.getPos(time)) |pos| base.pos = pos;
            if (boneAnim.getRot(time)) |rot| base.rot = rot;
            if (boneAnim.getScl(time)) |scl| base.scl = scl;
        }

        return base;
    }

    fn generateBoneMatrices(self: *const Animation, skeleton: *const Skeleton, output: *std.ArrayList(Mat4), parentMatrix: Mat4, boneIndex: BoneIndex, time: f32) !void {
        const invBind = skeleton.bones[boneIndex].inv;
        const tran = try self.computeBoneTransform(skeleton, boneIndex, time);

        const localMatrix = tran.computeMatrix();

        const worldMatrix = zmath.mul(parentMatrix, localMatrix);

        const offsetMatrix = zmath.mul(invBind, worldMatrix);

        try output.append(offsetMatrix);

        for (skeleton.bones[boneIndex].children[0..skeleton.bones[boneIndex].num_children]) |childBoneIndex| {
            try self.generateBoneMatrices(skeleton, output, worldMatrix, childBoneIndex, time);
        }
    }

    fn generateBoneVertices(self: *const Animation, skeleton: *const Skeleton, output: *std.ArrayList([3]f32), parentMatrix: Mat4, boneIndex: BoneIndex, time: f32) !void {
        const tran = try self.computeBoneTransform(skeleton, boneIndex, time);

        const localMatrix = tran.computeMatrix();

        const worldMatrix = zmath.mul(parentMatrix, localMatrix);

        const worldPos = zmath.mul(zmath.f32x4(0.0, 0.0, 0.0, 1.0), worldMatrix);

        try output.append(.{ worldPos[0], worldPos[1], worldPos[2] });

        for (skeleton.bones[boneIndex].children[0..skeleton.bones[boneIndex].num_children]) |childBoneIndex| {
            try self.generateBoneVertices(skeleton, output, worldMatrix, childBoneIndex, time);
        }
    }
};

const BoneAnimation = struct {
    positions: Channel(Vec3) = .{},
    rotations: Channel(Quat) = .{},
    scales: Channel(Vec3) = .{},

    fn deinit(self: *BoneAnimation, allocator: std.mem.Allocator) void {
        self.positions.deinit(allocator);
        self.rotations.deinit(allocator);
        self.scales.deinit(allocator);
    }

    fn getPos(self: *const BoneAnimation, time: f32) ?Vec3 {
        return if (self.positions.hasFrames()) self.positions.getInterpolation(.lerp, time) else null;
    }

    fn getRot(self: *const BoneAnimation, time: f32) ?Quat {
        return if (self.rotations.hasFrames()) self.rotations.getInterpolation(.slerp, time) else null;
    }

    fn getScl(self: *const BoneAnimation, time: f32) ?Vec3 {
        return if (self.scales.hasFrames()) self.scales.getInterpolation(.lerp, time) else null;
    }
};

fn Channel (comptime T: type) type {
    return struct {
        length: f32 = 0.0,
        keyframes: []Keyframe(T) = &[0]Keyframe(T) {},

        const Self = @This();

        fn init(self: *Self, allocator: std.mem.Allocator, numFrames: usize) !void {
            self.keyframes = try allocator.alloc(Keyframe(T), numFrames);
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.keyframes);
            self.length = 0.0;
            self.keyframes = &[0]Keyframe(T) {};
        }

        fn hasFrames(self: *const Self) bool {
            return self.keyframes.len > 0;
        }

        fn computeRelativeTime(self: *const Self, absoluteTime: f32) f32 {
            return @mod(absoluteTime, self.length);
        }

        fn findKeyframes(self: *const Self, relativeTime: f32) struct { *const Keyframe(T), *const Keyframe(T) } {
            var i: usize = 0;

            while (i < self.keyframes.len) : (i += 1) {
                if (self.keyframes[i].time >= relativeTime) break;
            }

            return
                if (i == 0) .{ &self.keyframes[0], &self.keyframes[self.keyframes.len - 1] }
                else .{ &self.keyframes[i - 1], &self.keyframes[i] };
        }

        fn getInterpolation(self: *const Channel(T), comptime method: enum { lerp, slerp }, time: f32) T {
            const relativeTime = self.computeRelativeTime(time);
            const a, const b = self.findKeyframes(relativeTime);
            const t = (relativeTime - a.time) / (b.time - a.time);
            return switch (method) {
                .lerp => zmath.lerp(a.value, b.value, t),
                .slerp => zmath.slerp(a.value, b.value, t),
            };
        }
    };
}

fn Keyframe (comptime T: type) type {
    return struct {
        time: f32,
        value: T,
    };
}

fn computeNormals(vertices: []Vertex, indices: []const u16, indexOffset: u16) void {
    for (vertices) |*vert| vert.normal = .{0.0, 0.0, 0.0};

    for (0..indices.len / 3) |i| {
        const x0 = indices[i * 3 + 0] - indexOffset;
        const x1 = indices[i * 3 + 1] - indexOffset;
        const x2 = indices[i * 3 + 2] - indexOffset;

        const v0 = zmath.loadArr3w(vertices[x0].pos, 1.0);
        const v1 = zmath.loadArr3w(vertices[x1].pos, 1.0);
        const v2 = zmath.loadArr3w(vertices[x2].pos, 1.0);

        const n = zmath.normalize3(zmath.cross3(v1 - v0, v2 - v0));

        vertices[x0].normal = zmath.vecToArr3(zmath.loadArr3(vertices[x0].normal) + n);
        vertices[x1].normal = zmath.vecToArr3(zmath.loadArr3(vertices[x1].normal) + n);
        vertices[x2].normal = zmath.vecToArr3(zmath.loadArr3(vertices[x2].normal) + n);
    }

    for (vertices) |*vert| vert.normal = zmath.vecToArr3(zmath.normalize3(zmath.loadArr3(vert.normal)));
}


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
    var modelDir = try samplesDir.openDir(FOLDERS[@intFromEnum(EXAMPLE)], .{});
    defer modelDir.close();

    const gltfJSON = try modelDir.readFileAllocOptions(
        allocator,
        ROOT_FILES[@intFromEnum(EXAMPLE)],
        512_000,
        null,
        4,
        null
    );
    defer allocator.free(gltfJSON);

    var gltf = zgltf.init(allocator);
    defer gltf.deinit();

    try gltf.parse(gltfJSON);

    // print raw gltf data
    // gltf.debugPrint();
    // for (gltf.data.nodes.items) |node| {
    //     const message =
    //         \\Node's name: {s}
    //         \\Children count: {}
    //         \\Have skin: {}
    //         \\
    //     ;

    //     std.debug.print(message, .{
    //         node.name,
    //         node.children.items.len,
    //         node.skin != null,
    //     });
    // }


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




    // load skin data //
    var skeleton = Skeleton {
        .bones = undefined,
        .num_bones = 0,
    };

    var nodeBoneMap = NodeBoneMap.init(allocator);
    defer nodeBoneMap.deinit();

    const skin = gltf.data.skins.items[0];
    std.debug.assert(gltf.data.skins.items.len == 1);

    { // hierarchy
        const jointNodeIndices = skin.joints.items;
        std.debug.assert(jointNodeIndices.len <= MAX_BONES and jointNodeIndices.len > 0);

        const rootNodeIndex = skin.skeleton orelse skin.joints.items[0];

        const nodes = gltf.data.nodes.items;

        _ = try skeleton.addBones(&nodeBoneMap, nodes, rootNodeIndex, null);

        for (jointNodeIndices) |jointNodeIndex| {
            std.debug.assert(nodeBoneMap.contains(jointNodeIndex));
        }
    }

    { // bind matrices
        const accessor = gltf.data.accessors.items[skin.inverse_bind_matrices.?];

        std.debug.assert(skeleton.num_bones == @as(usize, @intCast(accessor.count)));

        const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

        var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
        while (it.next()) |x| {
            const b, const i = x;

            const j = nodeBoneMap.get(skin.joints.items[i]) orelse return error.InvalidBindMatrix;

            // TODO: is this the right format? or do we need to transpose?
            skeleton.bones[j].inv = .{
                .{ b[0],  b[1],  b[2],  b[3]  },
                .{ b[4],  b[5],  b[6],  b[7]  },
                .{ b[8],  b[9],  b[10], b[11] },
                .{ b[12], b[13], b[14], b[15] },
            };
        }
    }


    // load mesh data //
    var vertices = std.ArrayList(Vertex).init(allocator);
    defer vertices.deinit();

    var indices = std.ArrayList(u16).init(allocator);
    defer indices.deinit();

    const mesh = gltf.data.meshes.items[0];
    std.debug.assert(gltf.data.meshes.items.len == 1);

    for (mesh.primitives.items) |primitive| {
        const indicesAccessor = gltf.data.accessors.items[primitive.indices.?];

        const indexBufferView = gltf.data.buffer_views.items[indicesAccessor.buffer_view.?];

        const indexOffset: u16 = @intCast(indices.items.len);
        const indicesView = try indices.addManyAt(indices.items.len, @intCast(indicesAccessor.count));

        var indexIt = IndexedAccessorIterator(u16).init(&gltf, indicesAccessor, bufferMap[indexBufferView.buffer]);
        while (indexIt.next()) |x| {
            const k, const i = x;
            indicesView[i] = k[0] + indexOffset;
        }

        const verticesView = view: {
            const firstAccessor = gltf.data.accessors.items[extractAttributeIndex(primitive.attributes.items[0])];
            const view = try vertices.addManyAt(vertices.items.len, @intCast(firstAccessor.count));

            // for (view) |*vert| vert.* = Vertex {};
            break :view view;
        };

        var haveNormals = false;

        for (primitive.attributes.items) |attribute| {
            switch (attribute) {
                .position => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);
                    std.debug.assert(accessor.component_type == .float);

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
                    std.debug.assert(accessor.component_type == .float);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const n, const i = x;
                        verticesView[i].normal = .{ n[0], n[1], n[2] };
                    }

                    haveNormals = true;
                },
                .joints => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);
                    std.debug.assert(accessor.component_type == .unsigned_short);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(u16).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const j, const i = x;
                        verticesView[i].joints = .{
                            nodeBoneMap.get(skin.joints.items[j[0]]) orelse return error.InvalidJointIndex,
                            nodeBoneMap.get(skin.joints.items[j[1]]) orelse return error.InvalidJointIndex,
                            nodeBoneMap.get(skin.joints.items[j[2]]) orelse return error.InvalidJointIndex,
                            nodeBoneMap.get(skin.joints.items[j[3]]) orelse return error.InvalidJointIndex,
                        };
                    }
                },
                .weights => |idx| {
                    const accessor = gltf.data.accessors.items[idx];

                    std.debug.assert(accessor.count == verticesView.len);
                    std.debug.assert(accessor.component_type == .float);

                    const bufferView = gltf.data.buffer_views.items[accessor.buffer_view.?];

                    var it = IndexedAccessorIterator(f32).init(&gltf, accessor, bufferMap[bufferView.buffer]);
                    while (it.next()) |x| {
                        const w, const i = x;
                        std.debug.assert(std.math.approxEqAbs(f32, w[0] + w[1] + w[2] + w[3], 1.0, std.math.floatEps(f32)));
                        verticesView[i].weights = .{ w[0], w[1], w[2], w[3] };
                    }
                },
                else => {
                    std.debug.print("Unhandled attribute: {}\n", .{attribute});
                },
            }
        }

        if (!haveNormals) {
            std.debug.print("Note: Mesh file does not contain normals, computing them instead", .{});
            computeNormals(verticesView, indicesView, indexOffset);
        }
    }

    // print mesh data
    // std.debug.print("Vertices count: {}\n", .{vertices.items.len});
    // for (vertices.items, 0..) |vert, i| {
    //     std.debug.print(
    //         \\{}:
    //         \\  Pos: [{d:.3}, {d:.3}, {d:.3}]
    //         \\  Normal: [{d:.3}, {d:.3}, {d:.3}]
    //         \\  Joints: [{}, {}, {}, {}]
    //         \\  Weights: [{d:.3}, {d:.3}, {d:.3}, {d:.3}]
    //         \\
    //         , .{
    //             i,
    //             vert.pos[0], vert.pos[1], vert.pos[2],
    //             vert.normal[0], vert.normal[1], vert.normal[2],
    //             vert.joints[0], vert.joints[1], vert.joints[2], vert.joints[3],
    //             vert.weights[0], vert.weights[1], vert.weights[2], vert.weights[3],
    //         }
    //     );
    // }
    // std.debug.print("Indices count: {}\n", .{indices.items.len});


    // load animation data //
    const glAnimation = gltf.data.animations.items[0];
    std.debug.assert(gltf.data.animations.items.len == 1);

    var animation = Animation {};
    defer animation.deinit(allocator);

    { // channels
        const channels = glAnimation.channels.items;

        for (channels) |channel| {
            const sampler: zgltf.AnimationSampler = glAnimation.samplers.items[channel.sampler];

            const inputAccessor = gltf.data.accessors.items[sampler.input];
            const outputAccessor = gltf.data.accessors.items[sampler.output];

            std.debug.assert(inputAccessor.count == outputAccessor.count);

            const boneIndex = nodeBoneMap.get(channel.target.node) orelse return error.InvalidBoneIndex;
            const boneAnim = animation.getOrInitBone(boneIndex);

            const inputBufferView = gltf.data.buffer_views.items[inputAccessor.buffer_view.?];
            const outputBufferView = gltf.data.buffer_views.items[outputAccessor.buffer_view.?];

            var inputIt = IndexedAccessorIterator(f32).init(&gltf, inputAccessor, bufferMap[inputBufferView.buffer]);
            var outputIt = outputAccessor.iterator(f32, &gltf, bufferMap[outputBufferView.buffer]);

            switch (channel.target.property) {
                .translation => {
                    std.debug.assert(!boneAnim.positions.hasFrames());

                    try boneAnim.positions.init(allocator, @intCast(inputAccessor.count));

                    while (inputIt.next()) |x| {
                        const input, const i = x;

                        const output = outputIt.next() orelse return error.MissingOutput;

                        boneAnim.positions.length = @max(boneAnim.positions.length, input[0]);

                        boneAnim.positions.keyframes[i] = .{
                            .time = input[0],
                            .value = .{ output[0], output[1], output[2] },
                        };
                    }
                },
                .rotation => {
                    std.debug.assert(!boneAnim.rotations.hasFrames());

                    try boneAnim.rotations.init(allocator, @intCast(inputAccessor.count));

                    while (inputIt.next()) |x| {
                        const input, const i = x;

                        const output = outputIt.next() orelse return error.MissingOutput;

                        boneAnim.rotations.length = @max(boneAnim.rotations.length, input[0]);

                        boneAnim.rotations.keyframes[i] = .{
                            .time = input[0],
                            .value = .{ output[0], output[1], output[2], output[3] },
                        };
                    }
                },
                .scale => {
                    std.debug.assert(!boneAnim.scales.hasFrames());

                    try boneAnim.scales.init(allocator, @intCast(inputAccessor.count));

                    while (inputIt.next()) |x| {
                        const input, const i = x;

                        const output = outputIt.next() orelse return error.MissingOutput;

                        boneAnim.scales.length = @max(boneAnim.scales.length, input[0]);

                        boneAnim.scales.keyframes[i] = .{
                            .time = input[0],
                            .value = .{ output[0], output[1], output[2] },
                        };
                    }
                },
                else => std.debug.print("Unhandled channel property: {s}\n", .{@tagName(channel.target.property)}),
            }
        }
    }

    // print animation data
    // for (animation.bones, 0..) |boneAnim, boneIndex| {
    //     if (boneAnim) |*b| {
    //         std.debug.print("Bone {} animation:\n", .{boneIndex});

    //         if (b.positions.hasFrames()) {
    //             std.debug.print("\tpositions ({d:.3}):", .{b.positions.length});

    //             for (b.positions.keyframes) |keyframe| {
    //                 std.debug.print(" ({d:.3})[{d:.3}, {d:.3}, {d:.3}];", .{keyframe.time, keyframe.value[0], keyframe.value[1], keyframe.value[2]});
    //             }

    //             std.debug.print("\n", .{});
    //         }

    //         if (b.rotations.hasFrames()) {
    //             std.debug.print("\trotations ({d:.3}):", .{b.rotations.length});

    //             for (b.rotations.keyframes) |keyframe| {
    //                 std.debug.print(" ({d:.3})[{d:.3}, {d:.3}, {d:.3}, {d:.3}];", .{keyframe.time, keyframe.value[0], keyframe.value[1], keyframe.value[2], keyframe.value[3]});
    //             }

    //             std.debug.print("\n", .{});
    //         }

    //         if (b.scales.hasFrames()) {
    //             std.debug.print("\tscales ({d:.3}):", .{b.scales.length});

    //             for (b.scales.keyframes) |keyframe| {
    //                 std.debug.print(" ({d:.3})[{d:.3}, {d:.3}, {d:.3}];", .{keyframe.time, keyframe.value[0], keyframe.value[1], keyframe.value[2]});
    //             }

    //             std.debug.print("\n", .{});
    //         }
    //     }
    // }


    // load shaders //
    var shaderDir = try samplesDir.openDir("shaders", .{});
    defer shaderDir.close();

    const basicVertexSource = try shaderDir.readFileAlloc(allocator, "basic.vert", math.maxInt(u16));
    defer allocator.free(basicVertexSource);

    const skinVertexSource = try shaderDir.readFileAlloc(allocator, "skin.vert", math.maxInt(u16));
    defer allocator.free(skinVertexSource);

    const basicFragmentSource = try shaderDir.readFileAlloc(allocator, "basic.frag", math.maxInt(u16));
    defer allocator.free(basicFragmentSource);

    const skinFragmentSource = try shaderDir.readFileAlloc(allocator, "skin.frag", math.maxInt(u16));
    defer allocator.free(skinFragmentSource);

    const litFragmentSource = try shaderDir.readFileAlloc(allocator, "lit.frag", math.maxInt(u16));
    defer allocator.free(litFragmentSource);

    // print shader sources
    // std.debug.print("loaded vert src: ```\n{s}\n```\n", .{basicVertexSource});
    // std.debug.print("loaded vert src: ```\n{s}\n```\n", .{skinVertexSource});
    // std.debug.print("loaded frag src: ```\n{s}\n```\n", .{basicFragmentSource});
    // std.debug.print("loaded frag src: ```\n{s}\n```\n", .{skinFragmentSource});

    const basicVertexShader = zgl.createShader(.vertex);
    basicVertexShader.source(1, &[1][]const u8 { basicVertexSource });
    basicVertexShader.compile();
    defer basicVertexShader.delete();

    if (basicVertexShader.get(.compile_status) == 0) {
        const info = try basicVertexShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Basic vertex shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const skinVertexShader = zgl.createShader(.vertex);
    skinVertexShader.source(1, &[1][]const u8 { skinVertexSource });
    skinVertexShader.compile();
    defer skinVertexShader.delete();

    if (skinVertexShader.get(.compile_status) == 0) {
        const info = try skinVertexShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Mesh vertex shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const basicFragmentShader = zgl.createShader(.fragment);
    basicFragmentShader.source(1, &[1][]const u8 { basicFragmentSource });
    basicFragmentShader.compile();
    defer basicFragmentShader.delete();

    if (basicFragmentShader.get(.compile_status) == 0) {
        const info = try basicFragmentShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Fragment shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const skinFragmentShader = zgl.createShader(.fragment);
    skinFragmentShader.source(1, &[1][]const u8 { skinFragmentSource });
    skinFragmentShader.compile();
    defer skinFragmentShader.delete();

    if (skinFragmentShader.get(.compile_status) == 0) {
        const info = try skinFragmentShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Fragment shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const litFragmentShader = zgl.createShader(.fragment);
    litFragmentShader.source(1, &[1][]const u8 { litFragmentSource });
    litFragmentShader.compile();
    defer litFragmentShader.delete();

    if (litFragmentShader.get(.compile_status) == 0) {
        const info = try litFragmentShader.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Fragment shader info log: ```\n{s}\n```\n", .{info});

        return error.ShaderCompilationFailed;
    }

    const basicShaderProgram = zgl.createProgram();
    basicShaderProgram.attach(basicVertexShader);
    basicShaderProgram.attach(basicFragmentShader);
    basicShaderProgram.link();
    defer basicShaderProgram.delete();

    if (basicShaderProgram.get(.link_status) == 0) {
        const info = try basicShaderProgram.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Basic shader program info log: ```\n{s}\n```\n", .{info});

        return error.ShaderLinkingFailed;
    }

    const basicUniforms = .{
        .model = basicShaderProgram.uniformLocation("model") orelse return error.MissingUniform,
        .view = basicShaderProgram.uniformLocation("view") orelse return error.MissingUniform,
        .projection = basicShaderProgram.uniformLocation("projection") orelse return error.MissingUniform,
        .color = basicShaderProgram.uniformLocation("color") orelse return error.MissingUniform,
    };

    const meshShaderProgram = zgl.createProgram();
    meshShaderProgram.attach(skinVertexShader);
    meshShaderProgram.attach(switch (MESH_FRAGMENT) { .visualize_weights => skinFragmentShader, .basic => basicFragmentShader, .lit => litFragmentShader });
    meshShaderProgram.link();
    defer meshShaderProgram.delete();

    if (meshShaderProgram.get(.link_status) == 0) {
        const info = try meshShaderProgram.getCompileLog(allocator);
        defer allocator.free(info);

        std.debug.print("Mesh shader program info log: ```\n{s}\n```\n", .{info});

        return error.ShaderLinkingFailed;
    }

    const meshUniforms = .{
        .model = meshShaderProgram.uniformLocation("model") orelse return error.MissingUniform,
        .view = meshShaderProgram.uniformLocation("view") orelse return error.MissingUniform,
        .projection = meshShaderProgram.uniformLocation("projection") orelse return error.MissingUniform,
        .skinMatrices = meshShaderProgram.uniformLocation("skinMatrices") orelse return error.MissingUniform,
        .color = meshShaderProgram.uniformLocation("color"),
        .lightDir = meshShaderProgram.uniformLocation("lightDir"),
    };


    // bind mesh data //
    const meshVao = zgl.createVertexArray();
    {
        const vbo = zgl.createBuffer();
        meshVao.vertexBuffer(0, vbo, 0, @sizeOf(Vertex));
        vbo.data(Vertex, vertices.items, .static_draw);

        const ebo = zgl.createBuffer();
        meshVao.elementBuffer(ebo);
        ebo.data(u16, indices.items, .static_draw);

        const aPosLoc = zgl.getAttribLocation(meshShaderProgram, "aPos") orelse return error.MissingAttribute;
        meshVao.attribFormat(aPosLoc, componentCount(Vertex.Pos), nativeToGlType(componentType(Vertex.Pos)), true, @offsetOf(Vertex, "pos"));
        meshVao.attribBinding(aPosLoc, 0);
        meshVao.enableVertexAttribute(aPosLoc);

        if (zgl.getAttribLocation(meshShaderProgram, "aNormal")) |aNormalLoc| {
            meshVao.attribFormat(aNormalLoc, componentCount(Vertex.Normal), nativeToGlType(componentType(Vertex.Normal)), true, @offsetOf(Vertex, "normal"));
            meshVao.attribBinding(aNormalLoc, 0);
            meshVao.enableVertexAttribute(aNormalLoc);
        } else {
            std.debug.print("Note: no aNormal attribute in shader program; skipping", .{});
        }

        const aWeightsLoc = zgl.getAttribLocation(meshShaderProgram, "aWeights") orelse return error.MissingAttribute;
        meshVao.attribFormat(aWeightsLoc, componentCount(Vertex.Weights), nativeToGlType(componentType(Vertex.Weights)), true, @offsetOf(Vertex, "weights"));
        meshVao.attribBinding(aWeightsLoc, 0);
        meshVao.enableVertexAttribute(aWeightsLoc);

        const aJointsLoc = zgl.getAttribLocation(meshShaderProgram, "aJoints") orelse return error.MissingAttribute;
        meshVao.attribIFormat(aJointsLoc, componentCount(Vertex.Joints), nativeToGlType(componentType(Vertex.Joints)), @offsetOf(Vertex, "joints"));
        meshVao.attribBinding(aJointsLoc, 0);
        meshVao.enableVertexAttribute(aJointsLoc);
    }


    // bind skin data //
    var skinMatrices = std.ArrayList(Mat4).init(allocator);
    defer skinMatrices.deinit();
    {
        try animation.generateBoneMatrices(&skeleton, &skinMatrices, zmath.identity(), 0, 0.0);
    }

    var skinVertices = std.ArrayList(Vertex.Pos).init(allocator);
    defer skinVertices.deinit();
    {
        try skeleton.generateBoneVertices(&skinVertices, zmath.identity(), 0);
    }

    const skinVao = zgl.createVertexArray();
    const skinVbo = zgl.createBuffer();
    {
        const aPosLoc = zgl.getAttribLocation(basicShaderProgram, "aPos") orelse return error.MissingAttribute;
        skinVao.vertexBuffer(0, skinVbo, 0, @sizeOf(Vertex.Pos));
        skinVbo.data(Vertex.Pos, skinVertices.items, .dynamic_draw);

        skinVao.attribFormat(aPosLoc, 3, .float, false, 0);
        skinVao.attribBinding(aPosLoc, 0);
        skinVao.enableVertexAttribute(aPosLoc);
    }


    // init matrix data //
    const matrices = .{
        .model = zmath.identity(),
        .view = VIEWS[@intFromEnum(EXAMPLE)],
        .projection = zmath.perspectiveFovLhGl(0.25 * math.pi, ASPECT_RATIO, 0.1, 100.0),
    };


    // init uniforms //

    basicShaderProgram.uniformMatrix4(basicUniforms.model, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.model)))[0..1]);
    basicShaderProgram.uniformMatrix4(basicUniforms.view, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.view)))[0..1]);
    basicShaderProgram.uniformMatrix4(basicUniforms.projection, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.projection)))[0..1]);
    basicShaderProgram.uniform3f(basicUniforms.color, 1.0, 1.0, 1.0);

    meshShaderProgram.uniformMatrix4(meshUniforms.model, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.model)))[0..1]);
    meshShaderProgram.uniformMatrix4(meshUniforms.view, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.view)))[0..1]);
    meshShaderProgram.uniformMatrix4(meshUniforms.projection, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(&matrices.projection)))[0..1]);
    meshShaderProgram.uniformMatrix4(meshUniforms.skinMatrices, false, @as([*]const Mat4Arr, @ptrCast(zmath.arrNPtr(skinMatrices.items)))[0..skinMatrices.items.len]);
    if (meshUniforms.color) |loc| meshShaderProgram.uniform3f(loc, COLOR[0], COLOR[1], COLOR[2]);
    if (meshUniforms.lightDir) |loc| {
        const lightDir = LIGHT_DIRS[@intFromEnum(EXAMPLE)];
        meshShaderProgram.uniform3f(loc, lightDir[0], lightDir[1], lightDir[2]);
    }


    var runTimer = try std.time.Timer.start();
    // var frameTimer = try std.time.Timer.start();


    // rendering basics //
    zgl.clearColor(0.0, 0.0, 0.0, 1.0);

    zgl.pointSize(10.0);
    if (USE_WIREFRAME) {
        zgl.polygonMode(.front_and_back, .line);
    } else {
        zgl.enable(.cull_face);
        zgl.frontFace(.ccw);
    }


    // run //
    mainLoop: while (true) {
        while (sdl.pollEvent()) |ev| {
            switch (ev) {
                .quit => break :mainLoop,
                else => {},
            }
        }

        const runNs = runTimer.read();
        const runTime = @as(f32, @floatFromInt(runNs)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        // const deltaNs = frameTimer.lap();
        // var deltaTime = @as(f32, @floatFromInt(deltaNs)) / @as(f32, @floatFromInt(std.time.ns_per_s));

        skinVertices.clearRetainingCapacity();
        try animation.generateBoneVertices(&skeleton, &skinVertices, zmath.identity(), 0, runTime);
        skinVbo.subData(0, Vertex.Pos, skinVertices.items);

        skinMatrices.clearRetainingCapacity();
        try animation.generateBoneMatrices(&skeleton, &skinMatrices, zmath.identity(), 0, runTime);
        // std.debug.print("transferring skin matrices: {any}\n", .{skinMatrices.items});
        meshShaderProgram.uniformMatrix4(meshUniforms.skinMatrices, false, @as([*]const [4][4]f32, @ptrCast(zmath.arrNPtr(skinMatrices.items.ptr)))[0..skinMatrices.items.len]);

        zgl.clear(.{ .color = true, .depth = true });

        meshShaderProgram.use();
        meshVao.bind();
        zgl.drawElements(.triangles, indices.items.len, .unsigned_short, 0);

        if (RENDER_BONE_POSITIONS) {
            basicShaderProgram.use();
            skinVao.bind();
            zgl.drawArrays(.points, 0, skinVertices.items.len);
        }

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

fn componentType (comptime T: type) type {
    return switch (@typeInfo(T)) {
        .array => |info| info.child,
        .vector => |info| info.child,
        else => @compileError("Unsupported type `" ++ @typeName(T) ++ "`"),
    };
}

fn componentCount (comptime T: type) u32 {
    return switch (@typeInfo(T)) {
        .array => |info| info.len,
        .vector => |info| info.len,
        else => @compileError("Unsupported type `" ++ @typeName(T) ++ "`"),
    };
}

fn nativeToGlType (comptime T: type) zgl.Type {
    return switch (T) {
        i8 => .byte,
        i16 => .short,
        i32 => .int,
        f32 => .float,
        f16 => .half_float,
        f64 => .double,
        u8 => .unsigned_byte,
        u16 => .unsigned_short,
        u32 => .unsigned_int,
        else => @compileError("Unsupported type `" ++ @typeName(T) ++ "`"),
    };
}

fn sdlGlGetProcAddressWrapper(_: void, symbolName: [:0]const u8) ?zgl.binding.FunctionPointer {
    return sdl.gl.getProcAddress(symbolName);
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
