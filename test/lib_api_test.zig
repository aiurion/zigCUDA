// test/lib_api_test.zig
const std = @import("std");
const testing = std.testing;
const zigcuda = @import("zigcuda");

test "library initialization" {
    var ctx = zigcuda.init() catch |err| {
        if (err == error.NoCudaDevice or err == error.CudaInitFailed or err == error.CudaLoadFailed) {
            return;
        }
        return err;
    };
    defer ctx.deinit();

    try testing.expect(ctx.getDeviceCount() > 0);

    if (ctx.getDeviceCount() > 0) {
        const props = try ctx.getDeviceProperties(0);
        // Valid major version is > 0
        try testing.expect(props.major > 0);
    }
}

test "library context management" {
    var ctx1 = zigcuda.init() catch return;
    defer ctx1.deinit();
    var ctx2 = zigcuda.init() catch return;
    defer ctx2.deinit();

    try testing.expect(ctx1.isAvailable());
    try testing.expect(ctx2.isAvailable());
}
