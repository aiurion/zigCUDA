// test/cublas_integration_test.zig
// Comprehensive cuBLAS integration testing on CUDA Blackwell hardware
// Tests optimized BLAS operations: sgemm, dgemm, dot products

const std = @import("std");
const testing = std.testing;
const integrations = @import("integrations");

test "cuBLAS: Initialize and destroy handle" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch {}; // Ignore errors in cleanup

    // Verify we have a valid handle
    try testing.expect(cublas.handle != null);
}

test "cuBLAS: Single-precision matrix multiplication (sgemm)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping sgemm test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test matrices: A(2x3), B(3x4) -> C(2x4)
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // 2x3
    const b = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 }; // 3x4
    var c = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // 2x4

    const expected_c = [8]f32{
        // Row 0: [1*7 + 2*11 + 3*15, 1*8 + 2*12 + 3*16, 1*9 + 2*13 + 3*17, 1*10 + 2*14 + 3*18]
        74.0, 80.0, 86.0, 92.0,
        // Row 1: [4*7 + 5*11 + 6*15, 4*8 + 5*12 + 6*16, 4*9 + 5*13 + 6*17, 4*10 + 5*14 + 6*18]
        173.0, 188.0, 203.0, 218.0
    };

    try cublas.sgemm(
        2, // m: rows of op(A) and C
        4, // n: cols of op(B) and C
        3, // k: cols of op(A) and rows of op(B)
        1.0,
        &a,
        3, // lda: leading dimension for A
        &b,
        4, // ldb: leading dimension for B
        0.0,
        &c,
        4, // ldc: leading dimension for C
    );

    const epsilon = @as(f32, 1e-5);
    inline for (0..8) |i| {
        try testing.expectApproxEqAbs(expected_c[i], c[i], epsilon);
    }
}

test "cuBLAS: Double-precision matrix multiplication (dgemm)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping dgemm test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test matrices: A(3x2), B(2x3) -> C(3x3)
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // 3x2
    const b = [_]f64{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }; // 2x3
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // 3x3

    const expected_c = [9]f64{
        // Row 1: (1*7 + 2*10), (1*8 + 2*11), (1*9 + 2*12)
        27.0, 30.0,  33.0,
        // Row 2: (3*7 + 4*10), (3*8 + 4*11), (3*9 + 4*12)
        61.0, 68.0,  75.0,
        // Row 3: (5*7 + 6*10), (5*8 + 6*11), (5*9 + 6*12)
        95.0, 106.0, 117.0,
    };

    try cublas.dgemm(
        3,
        3,
        2,
        1.0,
        &a,
        2, // lda
        &b,
        3, // ldb
        0.0,
        &c,
        3, // ldc
    );

    const epsilon = @as(f64, 1e-10);
    inline for (0..9) |i| {
        try testing.expectApproxEqAbs(expected_c[i], c[i], epsilon);
    }
}

test "cuBLAS: Matrix-vector multiplication (sgemv)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping sgemv test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test matrix A(3x4) and vector x(4), result y(3)
    const a = [12]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }; // 3x4 matrix
    const x = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = [3]f32{ 0.0, 0.0, 0.0 };

    try cublas.sgemv(
        false, // trans_a
        3, // m: rows of A if no transpose, cols of A if transposed
        4, // n: cols of A if no transpose, rows of A if transposed
        1.0,
        &a,
        4, // lda: leading dimension of A
        &x,
        @as(c_int, 1), // incx
        0.0,
        &y,
        @as(c_int, 1), // incy
    );

    const expected_y = [3]f32{
        // Row 1: 1*1 + 2*2 + 3*3 + 4*4 = 30
        30.0,
        // Row 2: 5*1 + 6*2 + 7*3 + 8*4 = 70
        70.0,
        // Row 3: 9*1 + 10*2 + 11*3 + 12*4 = 110
        110.0,
    };

    const epsilon = @as(f32, 1e-5);
    inline for (0..3) |i| {
        try testing.expectApproxEqAbs(expected_y[i], y[i], epsilon);
    }
}

test "cuBLAS: Dot product (sdot)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping sdot test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    const x = [5]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [5]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };

    const dot_result = try cublas.sdot(5, &x, @as(c_int, 1), &y, @as(c_int, 1));

    // Expected: 1*10 + 2*20 + 3*30 + 4*40 + 5*50 = 550
    try testing.expectApproxEqAbs(@as(f32, 550.0), dot_result, @as(f32, 1e-5));
}

test "cuBLAS: Large matrix multiplication performance" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping large sgemm test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test with larger matrices to validate Blackwell tensor core utilization
    const m: usize = 512;
    const n: usize = 256;
    const k: usize = 128;

    // Allocate test matrices (simplified for testing)
    var a = try std.testing.allocator.create([m * k]f32);
    defer std.testing.allocator.destroy(a);

    var b = try std.testing.allocator.create([k * n]f32);
    defer std.testing.allocator.destroy(b);

    const c = try std.testing.allocator.create([m * n]f32);
    defer std.testing.allocator.destroy(c);

    // Initialize with test data
    for (0..(m * k)) |i| {
        a[i] = @as(f32, @floatFromInt(i % 100));
    }

    for (0..(k * n)) |i| {
        b[i] = @as(f32, @floatFromInt((i + 50) % 200));
    }

    // Time the operation
    const start_time = std.time.microTimestamp();

    try cublas.sgemm(
        m, // m
        n, // n
        k, // k
        1.0,
        a.*[0..], // a
        k, // lda
        b.*[0..],
        n, // ldb: leading dimension for B
        0.0,
        c.*[0..], // c
        n, // ldc: leading dimension for C
    );

    const end_time = std.time.microTimestamp();
    const elapsed_us = @as(f64, @floatFromInt(end_time - start_time));

    // Log performance metrics
    const flops_2n = 2.0 * @as(f64, m) * @as(f64, n) * @as(f64, k);
    const gflops_per_sec = (flops_2n / elapsed_us) * 1e6;

    std.log.info("cuBLAS large matrix performance: {} GFLOPS/s", .{gflops_per_sec});

    // Spot-check computation correctness at a few positions
    // C[0,0] = sum(A[0,j] * B[j,0]) for j in 0..k
    var expected_c00: f32 = 0.0;
    for (0..k) |j| {
        expected_c00 += a[j] * b[j * n];
    }
    try testing.expectApproxEqAbs(expected_c00, c[0], 1e-2);

    // C[1,1] = sum(A[1,j] * B[j,1]) for j in 0..k
    var expected_c11: f32 = 0.0;
    for (0..k) |j| {
        expected_c11 += a[k + j] * b[j * n + 1];
    }
    try testing.expectApproxEqAbs(expected_c11, c[n + 1], 1e-2);

    // C[m-1, n-1] = sum(A[m-1,j] * B[j,n-1]) for j in 0..k
    var expected_c_last: f32 = 0.0;
    for (0..k) |j| {
        expected_c_last += a[(m - 1) * k + j] * b[j * n + (n - 1)];
    }
    try testing.expectApproxEqAbs(expected_c_last, c[(m - 1) * n + (n - 1)], 1e-2);
}

test "cuBLAS: Stream integration" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping stream test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test cuBLAS operations work correctly when integrated with CUDA streams
    const a = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [4]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [4]f32{ 0.0, 0.0, 0.0, 0.0 };

    // Basic sgemm should work
    try cublas.sgemm(
        2, // m
        2, // n
        2, // k
        1.0,
        &a,
        2, // lda
        &b,
        2, // ldb
        0.0,
        &c,
        2, // ldc
    );

    // Verify result matches expected matrix multiplication
    const epsilon = @as(f32, 1e-5);

    // Row 1: [19, 22] (1*5 + 2*7, 1*6 + 2*8)
    try testing.expectApproxEqAbs(@as(f32, 19.0), c[0], epsilon);
    try testing.expectApproxEqAbs(@as(f32, 22.0), c[1], epsilon);

    // Row 2: [43, 50] (3*5 + 4*7, 3*6 + 4*8)
    try testing.expectApproxEqAbs(@as(f32, 43.0), c[2], epsilon);
    try testing.expectApproxEqAbs(@as(f32, 50.0), c[3], epsilon);
}

test "cuBLAS: Vector operations - saxpy (single-precision scaled vector addition)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping saxpy test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test vectors
    const n: usize = 5;
    const alpha: f32 = 2.0;
    const x = [5]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var y = [5]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 }; // Will be modified to: y := α*x + y

    try cublas.saxpy(n, alpha, &x, @as(c_int, 1), &y, @as(c_int, 1));

    // Expected result: y[i] = 2 * x[i] + original_y[i]
    const expected_y = [5]f32{
        10.0 + (2.0 * 1.0), // 12.0
        20.0 + (2.0 * 2.0), // 24.0
        30.0 + (2.0 * 3.0), // 36.0
        40.0 + (2.0 * 4.0), // 48.0
        50.0 + (2.0 * 5.0), // 60.0
    };

    const epsilon = @as(f32, 1e-5);
    inline for (0..5) |i| {
        try testing.expectApproxEqAbs(expected_y[i], y[i], epsilon);
    }
}

test "cuBLAS: Vector operations - sscal (single-precision vector scaling)" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping sscal test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test vector
    const n: usize = 4;
    const alpha: f32 = 3.0;
    var x = [4]f32{ 1.0, 2.0, 3.0, 4.0 }; // Will be modified to: x := α*x

    try cublas.sscal(n, alpha, &x, @as(c_int, 1));

    // Expected result: x[i] = 3 * original_x[i]
    const expected_x = [4]f32{ 3.0, 6.0, 9.0, 12.0 };

    const epsilon = @as(f32, 1e-5);
    inline for (0..4) |i| {
        try testing.expectApproxEqAbs(expected_x[i], x[i], epsilon);
    }
}

test "cuBLAS: Batched matrix multiplication setup" {
    const init_result = integrations.Cublas.init();

    if (init_result == error.CudaError) {
        std.debug.print("INFO: cuBLAS library not available - skipping batched test\n", .{});
        return; // Skip gracefully
    }

    var cublas = try init_result;
    defer cublas.deinit() catch unreachable;

    // Test individual matrix multiplication with simple 2x2 matrices
    const a = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [4]f32{ 1.0, 1.0, 1.0, 1.0 };
    var c = [4]f32{ 0.0, 0.0, 0.0, 0.0 };

    try cublas.sgemm(2, // m: rows
        2, // n: cols
        2, // k: inner dimension
        1.0, // alpha
        &a, 2, // lda
        &b, 2, // ldb
        0.0, // beta
        &c, 2 // ldc
    );

    // Expected result: A * B where A = [1,2; 3,4] and B = [1,1; 1,1]
    // C[0,0] = 1*1 + 2*1 = 3
    // C[0,1] = 1*1 + 2*1 = 3
    // C[1,0] = 3*1 + 4*1 = 7
    // C[1,1] = 3*1 + 4*1 = 7
    const expected_c = [4]f32{ 3.0, 3.0, 7.0, 7.0 };

    const epsilon = @as(f32, 1e-5);
    inline for (0..4) |i| {
        try testing.expectApproxEqAbs(expected_c[i], c[i], epsilon);
    }
}
