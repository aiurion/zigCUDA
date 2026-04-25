// examples/05_ergonomic_launch.zig
// Preferred low-level style: owned buffers, typed copies, packed params, and launch defaults.

const std = @import("std");
const zigcuda = @import("zigcuda");
const cuda = zigcuda.bindings;

pub fn main() !void {
    try cuda.load();
    try cuda.init(0);

    const device = try cuda.getDevice(0);
    const ctx = try cuda.createContext(0, device);
    defer cuda.destroyContext(ctx) catch {};

    const n: u32 = 1024;
    const allocator = std.heap.page_allocator;

    const a = try allocator.alloc(f32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, n);
    defer allocator.free(c);

    for (a, 0..) |*value, i| value.* = @floatFromInt(i);
    for (b, 0..) |*value, i| value.* = @floatFromInt(i * 2);
    @memset(c, 0);

    var d_a = try zigcuda.DeviceBuffer.alloc(std.mem.sliceAsBytes(a).len);
    defer d_a.deinit();
    var d_b = try zigcuda.DeviceBuffer.alloc(std.mem.sliceAsBytes(b).len);
    defer d_b.deinit();
    var d_c = try zigcuda.DeviceBuffer.alloc(std.mem.sliceAsBytes(c).len);
    defer d_c.deinit();

    try d_a.copyFromTyped(f32, a);
    try d_b.copyFromTyped(f32, b);

    var module = try zigcuda.Module.loadFirst(allocator, &.{
        "examples/kernels/vector_add.cubin",
        "examples/kernels/vector_add.ptx",
    });
    defer module.deinit();

    const kernel = try module.kernel("vector_add");

    var params = zigcuda.Params.init();
    try params.devicePtr(d_a.ptr);
    try params.devicePtr(d_b.ptr);
    try params.devicePtr(d_c.ptr);
    try params.value(u32, n);

    try kernel.launch(.{
        .grid = zigcuda.Dim3.init((n + 255) / 256),
        .block = .{ .x = 256 },
        .sync_after = true,
    }, params.slice());

    try d_c.copyToTyped(f32, c);

    var mismatches: usize = 0;
    for (a, b, c) |left, right, actual| {
        if (actual != left + right) {
            mismatches += 1;
        }
    }

    if (mismatches != 0) {
        return error.VerificationFailed;
    }

    std.debug.print("Vector add completed with ergonomic zigCUDA API ({d} elements).\n", .{n});
}
