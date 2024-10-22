const std = @import("std");

pub fn main() !void {
    std.debug.print("Hello world", .{});
}

test {
    std.testing.refAllDeclsRecursive(@This());
}