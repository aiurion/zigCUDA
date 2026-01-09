// src/integrations/cublas.zig
// cuBLAS integration for optimized BLAS operations
// TODO: Implement cuBLAS integration

const std = @import("std");
const bindings = @import("../bindings/cublas.zig");
const tensor = @import("../ops/tensor.zig");

/// cuBLAS error types mapped from cublasStatus_t
pub const CUBLASError = error{
    InvalidValue,
    NotSupported,
    AllocationFailed,
    InternalError,
    InvalidHandle,
    DriverVersion,
    RuntimeWarning,
    RuntimePending,
    UnknownCUBLASError,
};

pub const Cublas = struct {
    handle: bindings.cublasHandle_t,

    pub fn init() !Cublas {
        // Load cuBLAS dynamic library first
        bindings.load() catch |err| {
            std.log.err("Failed to load cuBLAS library: {}", .{err});
            return CUBLASError.AllocationFailed; // Use closest error type
        };

        var handle: bindings.cublasHandle_t = null;

        std.log.info("Calling cublasCreate with handle ptr: {*}, value: {any}", .{ &handle, handle });

        const result = bindings.cublasCreate(&handle);

        // Log the actual status for debugging
        std.log.info("cublasCreate returned status: {}, handle now: {any}", .{ result, handle });

        if (result != .success) {
            std.log.err("cublasCreate failed with status: {}", .{result});
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                .allocation_failed => CUBLASError.AllocationFailed,
                .internal_error => CUBLASError.InternalError,
                .not_supported => CUBLASError.NotSupported,
                .invalid_value => CUBLASError.InvalidValue,
                else => CUBLASError.UnknownCUBLASError,
            };
        }

        return Cublas{
            .handle = handle,
        };
    }

    pub fn deinit(self: *Cublas) !void {
        const result = bindings.cublasDestroy(self.handle);
        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    pub fn setStream(self: *Cublas, stream: bindings.CUstream) CUBLASError!void {
        const result = bindings.cublasSetStream(self.handle, stream);
        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    // ============================================================================
    // SINGLE-PRECISION MATRIX OPERATIONS (4 functions)
    // ============================================================================

    /// Single-precision matrix multiplication: C = alpha*A*B + beta*C
    pub fn sgemm(
        self: *Cublas,
        m: usize, // rows of op(A) and C
        n: usize, // cols of op(B) and C
        k: usize, // cols of op(A) and rows of op(B)
        alpha: f32,
        a: []const f32, // matrix A (size m*k if no transpose, k*m if transposed)
        lda: usize, // leading dimension of A
        b: []const f32, // matrix B (size k*n if no transpose, n*k if transposed)
        ldb: usize, // leading dimension of B
        beta: f32,
        c: []f32, // output matrix C (size m*n)
        ldc: usize, // leading dimension of C
    ) !void {
        const result = bindings.cublasSgemm(self.handle, 0, // trans_a - assume no transpose for now
            0, // trans_b - assume no transpose for now
            @intCast(m), @intCast(n), @intCast(k), &alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), &beta, c.ptr, @intCast(ldc));

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                .not_supported => CUBLASError.NotSupported,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    /// Double-precision matrix multiplication: C = alpha*A*B + beta*C
    pub fn dgemm(
        self: *Cublas,
        m: usize, // rows of op(A) and C
        n: usize, // cols of op(B) and C
        k: usize, // cols of op(A) and rows of op(B)
        alpha: f64,
        a: []const f64, // matrix A (size m*k if no transpose, k*m if transposed)
        lda: usize, // leading dimension of A
        b: []const f64, // matrix B (size k*n if no transpose, n*k if transposed)
        ldb: usize, // leading dimension of B
        beta: f64,
        c: []f64, // output matrix C (size m*n)
        ldc: usize, // leading dimension of C
    ) !void {
        const result = bindings.cublasDgemm(self.handle, 0, // trans_a - assume no transpose for now
            0, // trans_b - assume no transpose for now
            @intCast(m), @intCast(n), @intCast(k), &alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), &beta, c.ptr, @intCast(ldc));

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                .not_supported => CUBLASError.NotSupported,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    /// Single-precision matrix-vector multiplication: y = alpha*A*x + beta*y
    pub fn sgemv(
        self: *Cublas,
        trans_a: bool, // whether to transpose A (0=no transpose, 1=transpose)
        m: usize, // rows of A if no transpose, cols of A if transposed
        n: usize, // cols of A if no transpose, rows of A if transposed
        alpha: f32,
        a: []const f32, // matrix A (size m*n)
        lda: usize, // leading dimension of A
        x: []const f32, // input vector X (length n or m depending on transpose)
        incx: c_int, // increment between elements in X (usually 1)
        beta: f32,
        y: []f32, // output/input vector Y (length m or n depending on transpose)
        incy: c_int, // increment between elements in Y (usually 1)
    ) !void {
        const result = bindings.cublasSgemv(self.handle, if (trans_a) 1 else 0, // trans_a
            @intCast(m), @intCast(n), &alpha, a.ptr, @intCast(lda), x.ptr, incx, &beta, y.ptr, incy);

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                .not_supported => CUBLASError.NotSupported,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    /// Double-precision matrix-vector multiplication: y = alpha*A*x + beta*y
    pub fn dgemv(
        self: *Cublas,
        trans_a: bool, // whether to transpose A (0=no transpose, 1=transpose)
        m: usize, // rows of A if no transpose, cols of A if transposed
        n: usize, // cols of A if no transpose, rows of A if transposed
        alpha: f64,
        a: []const f64, // matrix A (size m*n)
        lda: usize, // leading dimension of A
        x: []const f64, // input vector X (length n or m depending on transpose)
        incx: c_int, // increment between elements in X (usually 1)
        beta: f64,
        y: []f64, // output/input vector Y (length m or n depending on transpose)
        incy: c_int, // increment between elements in Y (usually 1)
    ) !void {
        const result = bindings.cublasDgemv(self.handle, if (trans_a) 1 else 0, // trans_a
            @intCast(m), @intCast(n), &alpha, a.ptr, @intCast(lda), x.ptr, incx, &beta, y.ptr, incy);

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                .not_supported => CUBLASError.NotSupported,
                else => CUBLASError.UnknownCUBLASError,
            };
        }
    }

    /// Single-precision dot product: result = sum(x[i] * y[i])
    pub fn sdot(
        self: *Cublas,
        n: usize, // number of elements to multiply and sum
        x: []const f32, // first input vector X
        incx: c_int, // increment between elements in X (usually 1)
        y: []const f32, // second input vector Y
        incy: c_int, // increment between elements in Y (usually 1)
    ) !f32 {
        var result_f32: f32 = undefined;

        const result = bindings.cublasSdot(self.handle, @intCast(n), x.ptr, incx, y.ptr, incy, &result_f32);

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                else => CUBLASError.UnknownCUBLASError,
            };
        }

        return result_f32;
    }

    /// Double-precision dot product: result = sum(x[i] * y[i])
    pub fn ddot(
        self: *Cublas,
        n: usize, // number of elements to multiply and sum
        x: []const f64, // first input vector X (double precision)
        incx: c_int, // increment between elements in X (usually 1)
        y: []const f64, // second input vector Y (double precision)
        incy: c_int, // increment between elements in Y (usually 1)
    ) !f64 {
        var result_f64: f64 = undefined;

        const result = bindings.cublasDdot(self.handle, @intCast(n), x.ptr, incx, y.ptr, incy, &result_f64);

        if (result != .success) {
            return switch (result) {
                .invalid_handle => CUBLASError.InvalidHandle,
                else => CUBLASError.UnknownCUBLASError,
            };
        }

        return result_f64;
    }

    // ============================================================================
    // LEGACY FUNCTIONS - DEPRECATED
    // ============================================================================

};

// TODO: Add more cuBLAS operations
