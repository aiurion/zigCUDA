// test/v2_memory_bindings_test.zig
// Test suite specifically for CUDA 13+ / Blackwell _v2 Memory APIs
// Verifies that v2 symbols load and function correctly

const std = @import("std");
const testing = std.testing;

// Import CUDA bindings using module name (set up in build.zig)
const cuda = @import("cuda");

// =============================================================================
// V2 MEMORY API BINDING TESTS
// These tests verify _v2 symbol loading (init required first)
// =============================================================================

test "cuMemAlloc_v2 - symbol loads correctly" {
    // Initialize CUDA first to load symbols
    try cuda.init(0);

    if (cuda.cuMemAlloc_v2 != null) {
        std.debug.print("✓ cuMemAlloc_v2 symbol loaded successfully\n", .{});
    } else {
        return error.SkipZigTest; // v2 not available on this system
    }
}

test "cuMemFree_v2 - symbol loads correctly" {
    if (cuda.cuMemFree_v2 != null) {
        std.debug.print("✓ cuMemFree_v2 symbol loaded successfully\n", .{});
    } else {
        return error.SkipZigTest;
    }
}

test "cuMemGetInfo_v2 - symbol loads correctly" {
    if (cuda.cuMemGetInfo_v2 != null) {
        std.debug.print("✓ cuMemGetInfo_v2 symbol loaded successfully\n", .{});
    } else {
        return error.SkipZigTest;
    }
}

// =============================================================================
// V2 MEMORY FUNCTIONAL TESTS
// These tests verify _v2 APIs work with actual memory operations
// CRITICAL: v2 functions need PRIMARY CONTEXT retained via cuDevicePrimaryCtxRetain
// and made current via cuCtxSetCurrent (see cublas.zig:71-95 for reference pattern)
// =============================================================================

test "cuMemAlloc_v2 - allocate and free 1MB" {
    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Skip if no v2 available
    if (cuda.cuMemAlloc_v2 == null) return error.SkipZigTest;

    const dev = try cuda.getDevice(0);

    // v2 functions need PRIMARY CONTEXT retained and set current (see cublas.zig:71-95)
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuDevicePrimaryCtxRetain) |f| {
        const ctx_result = f(&ctx, dev);
        if (ctx_result != 0 or ctx == null) return error.SkipZigTest;
    } else {
        return error.SkipZigTest;
    }

    if (cuda.cuCtxSetCurrent) |f| {
        const set_result = f(ctx.?);
        if (set_result != 0) return error.SkipZigTest;
    }

    defer {
        if (cuda.cuDevicePrimaryCtxRelease) |f| _ = f(dev);
    }

    const size = 1024 * 1024; // 1MB
    var dptr: cuda.CUdeviceptr = undefined;

    const res = cuda.cuMemAlloc_v2.?(&dptr, size);
    try testing.expectEqual(@as(i32, 0), res);

    std.debug.print("✓ cuMemAlloc_v2 allocated {d} bytes at ptr={x}\n", .{ size, dptr });

    if (cuda.cuMemFree_v2) |free_fn| {
        const free_res = free_fn(dptr);
        try testing.expectEqual(@as(i32, 0), free_res);
    }
}

test "cuMemGetInfo_v2 - query memory info" {
    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Skip if no v2 available
    if (cuda.cuMemGetInfo_v2 == null) return error.SkipZigTest;

    const dev = try cuda.getDevice(0);

    // v2 functions need PRIMARY CONTEXT retained and set current (see cublas.zig:71-95)
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuDevicePrimaryCtxRetain) |f| {
        const ctx_result = f(&ctx, dev);
        if (ctx_result != 0 or ctx == null) return error.SkipZigTest;
    } else {
        return error.SkipZigTest;
    }

    if (cuda.cuCtxSetCurrent) |f| {
        const set_result = f(ctx.?);
        if (set_result != 0) return error.SkipZigTest;
    }

    defer {
        if (cuda.cuDevicePrimaryCtxRelease) |f| _ = f(dev);
    }

    var free: u64 = undefined;
    var total: u64 = undefined;

    const res = cuda.cuMemGetInfo_v2.?(&free, &total);
    try testing.expectEqual(@as(i32, 0), res);

    std.debug.print("✓ cuMemGetInfo_v2: Free={d:.2} GB, Total={d:.2} GB\n", .{
        @as(f64, @floatFromInt(free)) / (1024 * 1024 * 1024),
        @as(f64, @floatFromInt(total)) / (1024 * 1024 * 1024),
    });

    try testing.expect(total > 0);
    try testing.expect(free <= total);
}

test "v2 symbols available on system" {
    // Initialize CUDA first (loads symbols)
    try cuda.init(0);

    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Get device properties
    const dev = try cuda.getDevice(0);

    var major: i32 = undefined;
    var minor: i32 = undefined;
    if (cuda.cuDeviceComputeCapability) |f| {
        _ = f(&major, &minor, dev);
        std.debug.print("✓ Device compute capability: {d}.{d}\n", .{ major, minor });
    }

    // Verify v2 symbols are available
    const has_v2 = cuda.cuMemAlloc_v2 != null and
        cuda.cuMemFree_v2 != null and
        cuda.cuMemGetInfo_v2 != null;

    try testing.expect(has_v2);

    if (has_v2) {
        std.debug.print("✓ All v2 memory symbols available\n", .{});
    } else {
        std.debug.print("INFO: Some v2 symbols not available - may be older driver\n", .{});
    }
}
