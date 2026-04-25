const std = @import("std");
const testing = std.testing;
const zigcuda = @import("zigcuda");
const cuda = zigcuda.bindings;

test "Dim3 and LaunchConfig provide CUDA launch defaults" {
    const dim = zigcuda.Dim3{ .x = 4, .y = 2 };
    try testing.expectEqual(@as(cuda.c_uint, 4), dim.x);
    try testing.expectEqual(@as(cuda.c_uint, 2), dim.y);
    try testing.expectEqual(@as(cuda.c_uint, 1), dim.z);

    const cfg = zigcuda.LaunchConfig{
        .grid = .{ .x = 8 },
        .block = .{ .x = 128 },
    };
    try testing.expectEqual(@as(cuda.c_uint, 8), cfg.grid.x);
    try testing.expectEqual(@as(cuda.c_uint, 1), cfg.grid.y);
    try testing.expectEqual(@as(cuda.c_uint, 1), cfg.grid.z);
    try testing.expectEqual(@as(cuda.c_uint, 128), cfg.block.x);
    try testing.expectEqual(@as(cuda.c_uint, 1), cfg.block.y);
    try testing.expectEqual(@as(cuda.c_uint, 1), cfg.block.z);
    try testing.expectEqual(@as(cuda.c_uint, 0), cfg.shared_mem_bytes);
    try testing.expectEqual(@as(?*cuda.CUstream, null), cfg.stream);
    try testing.expect(!cfg.sync_after);
}

test "LaunchConfig.forElements computes a one-dimensional grid" {
    const cfg = zigcuda.LaunchConfig.forElementsWithBlock(1025, 256);
    try testing.expectEqual(@as(cuda.c_uint, 5), cfg.grid.x);
    try testing.expectEqual(@as(cuda.c_uint, 256), cfg.block.x);
}

test "Params packs values, device pointers, and raw arguments" {
    var params = zigcuda.Params.init();
    var raw_value: i32 = 11;

    try params.value(i32, 7);
    try params.devicePtr(@as(cuda.CUdeviceptr, 0x1234));
    try params.raw(@ptrCast(&raw_value));

    try testing.expectEqual(@as(usize, 3), params.len());
    try testing.expectEqual(@as(usize, 3), params.slice().len);
}

test "Params reports parameter count and storage overflow" {
    var params = zigcuda.Params.init();
    for (0..zigcuda.Params.max_params) |i| {
        try params.value(usize, i);
    }
    try testing.expectError(error.TooManyParams, params.value(usize, 99));

    var large_params = zigcuda.Params.init();
    const large_value: [zigcuda.Params.storage_bytes + 1]u8 = undefined;
    try testing.expectError(error.ParamStorageOverflow, large_params.value(@TypeOf(large_value), large_value));
}

test "Params reports unsupported alignment" {
    const TooAligned = struct {
        value: u8 align(32),
    };
    var params = zigcuda.Params.init();
    try testing.expectError(error.ParamAlignmentUnsupported, params.value(TooAligned, .{ .value = 1 }));
}

test "typed copy helper signatures compile" {
    const dst: cuda.CUdeviceptr = 0;
    const src: []const f16 = &.{};
    var out: [0]f32 = .{};

    const copy_to_fn = zigcuda.copyToDeviceTyped;
    const copy_from_fn = zigcuda.copyFromDeviceTyped;

    if (false) {
        try copy_to_fn(f16, dst, src);
        try copy_from_fn(f32, out[0..], dst);
    }
}

test "DeviceBuffer allocation and typed copies round trip when CUDA is available" {
    const ctx = try requireCudaContext();
    defer cuda.destroyContext(ctx) catch {};

    const input = [_]f32{ 1.0, 2.5, 3.25, 4.75 };
    var output: [input.len]f32 = undefined;

    var buf = zigcuda.DeviceBuffer.alloc(@sizeOf(@TypeOf(input))) catch |err| return skipCuda(err);
    defer buf.deinit();

    try buf.copyFromTyped(f32, input[0..]);
    try buf.copyToTyped(f32, output[0..]);

    try testing.expectEqualSlices(f32, input[0..], output[0..]);

    var raw_output: [input.len]f32 = undefined;
    try zigcuda.copyToDeviceTyped(f32, buf.ptr, input[0..]);
    try zigcuda.copyFromDeviceTyped(f32, raw_output[0..], buf.ptr);
    try testing.expectEqualSlices(f32, input[0..], raw_output[0..]);
}

test "Module and Kernel wrappers load PTX and launch through LaunchConfig when CUDA is available" {
    const ctx = try requireCudaContext();
    defer cuda.destroyContext(ctx) catch {};

    var module = zigcuda.Module.loadFirst(testing.allocator, &.{
        "test/test.ptx",
        "test.ptx",
    }) catch |err| return skipCuda(err);
    defer module.deinit();

    const kernel = module.kernel("add_arrays") catch |err| return skipCuda(err);

    const input_a = [_]f32{ 1, 2, 3, 4 };
    const input_b = [_]f32{ 10, 20, 30, 40 };
    var output: [input_a.len]f32 = .{0} ** input_a.len;

    var dev_a = zigcuda.DeviceBuffer.alloc(@sizeOf(@TypeOf(input_a))) catch |err| return skipCuda(err);
    defer dev_a.deinit();
    var dev_b = zigcuda.DeviceBuffer.alloc(@sizeOf(@TypeOf(input_b))) catch |err| return skipCuda(err);
    defer dev_b.deinit();
    var dev_out = zigcuda.DeviceBuffer.alloc(@sizeOf(@TypeOf(output))) catch |err| return skipCuda(err);
    defer dev_out.deinit();

    try dev_a.copyFromTyped(f32, input_a[0..]);
    try dev_b.copyFromTyped(f32, input_b[0..]);

    var params = zigcuda.Params.init();
    try params.devicePtr(dev_a.ptr);
    try params.devicePtr(dev_b.ptr);
    try params.devicePtr(dev_out.ptr);
    try params.value(u32, @as(u32, input_a.len));

    try kernel.launch(.{
        .grid = .{ .x = 1 },
        .block = .{ .x = 32 },
        .sync_after = true,
    }, params.slice());

    try dev_out.copyToTyped(f32, output[0..]);
    try testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, output[0..]);
}

fn requireCudaContext() !*cuda.CUcontext {
    cuda.load() catch |err| return skipCuda(err);
    cuda.init(0) catch |err| return skipCuda(err);

    const count = cuda.getDeviceCount() catch |err| return skipCuda(err);
    if (count == 0) {
        return error.SkipZigTest;
    }

    const device = cuda.getDevice(0) catch |err| return skipCuda(err);
    return cuda.createContext(0, device) catch |err| return skipCuda(err);
}

fn skipCuda(_: anyerror) error{SkipZigTest} {
    return error.SkipZigTest;
}
