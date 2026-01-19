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
// Using same pattern as comprehensive_bindings_test.zig
// =============================================================================

test "cuMemAlloc_v2 - allocate and free 1MB" {
    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Skip if no v2 available  
    if (cuda.cuMemAlloc_v2 == null) return error.SkipZigTest;

    const dev = try cuda.getDevice(0);
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuCtxCreate) |f| _ = f(&ctx, 0, dev);

    // Make context current
    if (cuda.cuCtxSetCurrent) |setCtx| {
        if (ctx) |c| _ = setCtx(c);
    }
    defer {
        if (ctx) |c| {
            if (cuda.cuCtxDestroy) |f| _ = f(c);
        }
    }

    const size = 1024 * 1024; // 1MB
    var dptr: cuda.CUdeviceptr = undefined;

    if (cuda.cuMemAlloc_v2) |alloc_fn| {
        const res = alloc_fn(&dptr, size);
        try testing.expectEqual(@as(i32, 0), res);

        std.debug.print("✓ cuMemAlloc_v2 allocated {d} bytes at ptr={x}\n", .{size, dptr});

        if (cuda.cuMemFree_v2) |free_fn| {
            const free_res = free_fn(dptr);
            try testing.expectEqual(@as(i32, 0), free_res);
        }
    } else {
        return error.SkipZigTest;
    }
}

test "cuMemGetInfo_v2 - query memory info" {
    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Skip if no v2 available
    if (cuda.cuMemGetInfo_v2 == null) return error.SkipZigTest;

    const dev = try cuda.getDevice(0);
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuCtxCreate) |f| _ = f(&ctx, 0, dev);

    // Make context current
    if (cuda.cuCtxSetCurrent) |setCtx| {
        if (ctx) |c| _ = setCtx(c);
    }
    defer {
        if (ctx) |c| {
            if (cuda.cuCtxDestroy) |f| _ = f(c);
        }
    }

    var free: u64 = undefined;
    var total: u64 = undefined;

    if (cuda.cuMemGetInfo_v2) |info_fn| {
        const res = info_fn(&free, &total);
        try testing.expectEqual(@as(i32, 0), res);

        std.debug.print("✓ cuMemGetInfo_v2: Free={d:.2} GB, Total={d:.2} GB\n", .{
            @as(f64, @floatFromInt(free)) / (1024*1024*1024),
            @as(f64, @floatFromInt(total)) / (1024*1024*1024)
        });

        try testing.expect(total > 0);
        try testing.expect(free <= total);
    } else {
        return error.SkipZigTest;
    }
}

test "cuMemGetInfo - v1 vs v2 consistency check" {
    const count = try cuda.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    // Need both to compare
    if (cuda.cuMemGetInfo_v2 == null or cuda.cuMemGetInfo == null) {
        std.debug.print("INFO: Only one version available, skipping consistency check\n", .{});
        return error.SkipZigTest;
    }

    const dev = try cuda.getDevice(0);
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuCtxCreate) |f| _ = f(&ctx, 0, dev);

    // Make context current
    if (cuda.cuCtxSetCurrent) |setCtx| {
        if (ctx) |c| _ = setCtx(c);
    }
    defer {
        if (ctx) |c| {
            if (cuda.cuCtxDestroy) |f| _ = f(c);
        }
    }

    var free_v2: u64 = undefined;
    var total_v2: u64 = undefined;

    var free_v1: u64 = undefined;
    var total_v1: u64 = undefined;

    const res_v2 = cuda.cuMemGetInfo_v2.?(&free_v2, &total_v2);
    try testing.expectEqual(@as(i32, 0), res_v2);

    const res_v1 = cuda.cuMemGetInfo.?(&free_v1, &total_v1);
    try testing.expectEqual(@as(i32, 0), res_v1);

    // Values should be identical
    try testing.expectEqual(free_v1, free_v2);
    try testing.expectEqual(total_v1, total_v2);

    std.debug.print("✓ v1 and v2 return consistent values\n", .{});
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
        std.debug.print("✓ Device compute capability: {d}.{d}\n", .{major, minor});
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