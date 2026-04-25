// test/cublas_integration_test_simple.zig
// Lightweight cuBLAS smoke tests for the dedicated test-cublas-simple step.

const std = @import("std");
const cuda = @import("cuda");
const integrations = @import("integrations");

test "cuBLAS simple: CUDA bindings initialize" {
    try cuda.load();
    try cuda.init(0);

    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;
}

test "cuBLAS simple: create and destroy handle" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping simple test\n", .{});
        return error.SkipZigTest;
    }

    var cublas = try init_result;
    defer cublas.deinit() catch {};

    try std.testing.expect(cublas.handle != null);
}
