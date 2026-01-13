// src/bindings/cublas.zig
// cuBLAS API bindings (Dynamic Loading)

const std = @import("std");

/// Debug mode flag - set to true to enable all debug output, false for quiet operation  
const DEBUG_MODE_ENABLED = false;

/// Check if debug mode is enabled
fn isDebugEnabled() bool {
    return DEBUG_MODE_ENABLED;
}

pub const cublasStatus_t = enum(c_int) {
    success = 0,
    not_initialized = 1,
    allocation_failed = 3,
    invalid_value = 7,
    arch_mismatch = 8,
    mapping_error = 11,
    execution_failed = 13,
    internal_error = 14,
    insufficient_resources = 15,
};

pub const cublasHandle_t = ?*anyopaque;
pub const cublasLtMatDescriptor_t = ?*anyopaque;
pub const cublasLtMatrixLayout_t = ?*anyopaque;

// Library state
pub var lib: ?std.DynLib = null;

// Function Pointers
pub var cublasCreate: *const fn (handle: *?*anyopaque) callconv(.c) cublasStatus_t = undefined;
pub var cublasDestroy: *const fn (handle: ?*anyopaque) callconv(.c) cublasStatus_t = undefined;
pub var cublasSetStream: *const fn (handle: ?*anyopaque, stream: ?*anyopaque) callconv(.c) cublasStatus_t = undefined;

// Matrix multiplication pointers
pub var cublasSgemm: *const fn (handle: ?*anyopaque, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: *const f32, c: [*]f32, ldc: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasDgemm: *const fn (handle: ?*anyopaque, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: *const f64, c: [*]f64, ldc: c_int) callconv(.c) cublasStatus_t = undefined;

// Matrix-vector multiplication pointers
pub var cublasSgemv: *const fn (handle: ?*anyopaque, trans_a: c_int, m: c_int, n: c_int, alpha: *const f32, a: [*]const f32, lda: c_int, x: [*]const f32, incx: c_int, beta: *const f32, y: [*]f32, incy: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasDgemv: *const fn (handle: ?*anyopaque, trans_a: c_int, m: c_int, n: c_int, alpha: *const f64, a: [*]const f64, lda: c_int, x: [*]const f64, incx: c_int, beta: *const f64, y: [*]f64, incy: c_int) callconv(.c) cublasStatus_t = undefined;

// Dot product pointers
pub var cublasSdot: *const fn (handle: ?*anyopaque, n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int, result: *f32) callconv(.c) cublasStatus_t = undefined;

pub var cublasDdot: *const fn (handle: ?*anyopaque, n: c_int, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int, result: *f64) callconv(.c) cublasStatus_t = undefined;

// Vector operations - Functions 59-62
pub var cublasSaxpy: *const fn (handle: ?*anyopaque, n: c_int, alpha: *const f32, x: [*]const f32, incx: c_int, y: [*]f32, incy: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasDaxpy: *const fn (handle: ?*anyopaque, n: c_int, alpha: *const f64, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasSscal: *const fn (handle: ?*anyopaque, n: c_int, alpha: *const f32, x: [*]f32, incx: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasDscal: *const fn (handle: ?*anyopaque, n: c_int, alpha: *const f64, x: [*]f64, incx: c_int) callconv(.c) cublasStatus_t = undefined;

// Batched matrix multiplication - Functions 63-64
pub var cublasSgemmBatched: *const fn (handle: ?*anyopaque, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f32, a_array: [*]const [*]const f32, lda: c_int, b_array: [*]const [*]const f32, ldb: c_int, beta: *const f32, c_array: [*][*]f32, ldc: c_int, batch_count: c_int) callconv(.c) cublasStatus_t = undefined;

pub var cublasDgemmBatched: *const fn (handle: ?*anyopaque, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f64, a_array: [*]const [*]const f64, lda: c_int, b_array: [*]const [*]const f64, ldb: c_int, beta: *const f64, c_array: [*][*]f64, ldc: c_int, batch_count: c_int) callconv(.c) cublasStatus_t = undefined;

// Pointer mode configuration - Function 65
pub const CUBLAS_POINTER_MODE_HOST = 0;
pub const CUBLAS_POINTER_MODE_DEVICE = 1;

// Operation types for matrix operations
pub const CUBLAS_OP_N: c_int = 0; // Normal (no transpose)
pub const CUBLAS_OP_T: c_int = 1; // Transpose

pub var cublasSetPointerMode: *const fn (handle: ?*anyopaque, pointer_mode: c_int) callconv(.c) cublasStatus_t = undefined;

pub fn load() !void {
    if (lib != null) return;

    // Try common library names first
    const lib_names = [_][]const u8{ "libcublas.so", "libcublas.so.12", "libcublas.so.11", "libcublas.so.10" };

    for (lib_names) |name| {
        lib = std.DynLib.open(name) catch continue;
        if (lib != null) {
            std.debug.print("SUCCESS: Loaded cuBLAS from: {s}\n", .{name});
            break;
        }
    }

    // Try common library names first
    const linux_lib_names = [_][]const u8{ "libcublas.so", "libcublas.so.12", "libcublas.so.11" };

    for (linux_lib_names) |name| {
        if (isDebugEnabled()) {
            std.debug.print("DEBUG: Trying Linux path: {s}\n", .{name});
        }
        lib = std.DynLib.open(name) catch continue;
        if (lib != null) {
            if (isDebugEnabled()) {
                std.debug.print("SUCCESS: Loaded cuBLAS from standard paths!\n", .{});
            }
            break;
        }
    }

    // Try Windows partition path via WSL
    const windows_paths = [_][]const u8{
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64/cublas64_13.dll",
    };
    
    for (windows_paths) |path| {
        if (isDebugEnabled()) {
            std.debug.print("DEBUG: Trying Windows path: {s}\n", .{path});
        }
        lib = std.DynLib.open(path) catch continue;
        if (lib != null) {
            std.debug.print("SUCCESS: Loaded cuBLAS from Windows!\n", .{});
            break;
        }
    }

    // Try WSL paths
    const wsl_paths = [_][]const u8{
        "/usr/lib/wsl/lib/libcublas.so.13",
        "/usr/lib/wsl/lib/libcublas.so.12",
        "/usr/lib/wsl/lib/libcublas.so.11" 
    };
    
    for (wsl_paths) |path| {
        lib = std.DynLib.open(path) catch continue;
        if (lib != null) {
            std.debug.print("SUCCESS: Loaded cuBLAS from WSL!\n", .{});
            break;
        }
    }

    // Fallback - only try if library not yet loaded
    if (lib == null) {
        const default_path = "/usr/lib/wsl/lib/libcublas.so.13";

        lib = std.DynLib.open(default_path) catch {
            return error.CublasLibraryNotFound;
        };
    }

    // Create a reference to the loaded library for symbol lookup
    const l = &lib.?;
    
    // Load functions with 'v2' suffix first, fallback to non-suffixed versions
    if (isDebugEnabled()) {
        std.debug.print("DEBUG: Looking up cublasCreate symbol...\n", .{});
    }
    cublasCreate = (l.lookup(@TypeOf(cublasCreate), "cublasCreate_v2")) orelse
                 l.lookup(@TypeOf(cublasCreate), "cublasCreate") orelse {
        std.debug.print("ERROR: Could not find cublasCreate symbol\n", .{});
        return error.SymbolNotFound;
    };
    if (isDebugEnabled()) {
        std.debug.print("DEBUG: Found cublasCreate at: {*}\n", .{cublasCreate});
    }
    cublasDestroy = l.lookup(@TypeOf(cublasDestroy), "cublasDestroy_v2") orelse
                 l.lookup(@TypeOf(cublasDestroy), "cublasDestroy") orelse {
        std.debug.print("ERROR: Could not find cublasDestroy symbol\n", .{});
        return error.SymbolNotFound;
    };

    // Note: cublasSetStream_v2 is standard, fallback to non-suffixed
    if (l.lookup(@TypeOf(cublasSetStream), "cublasSetStream_v2")) |f| {
        cublasSetStream = f;
    } else if (l.lookup(@TypeOf(cublasSetStream), "cublasSetStream")) |f2| {
        cublasSetStream = f2;
    } else {
        std.debug.print("ERROR: Could not find cublasSetStream symbol\n", .{});
        return error.SymbolNotFound;
    }

    // Matrix multiplication functions - try v2 first, then fallback
    cublasSgemm = l.lookup(@TypeOf(cublasSgemm), "cublasSgemm_v2") orelse 
                l.lookup(@TypeOf(cublasSgemm), "cublasSgemm") orelse return error.SymbolNotFound;
    cublasDgemm = l.lookup(@TypeOf(cublasDgemm), "cublasDgemm_v2") orelse
                l.lookup(@TypeOf(cublasDgemm), "cublasDgemm") orelse return error.SymbolNotFound;

    // Matrix-vector multiplication - try v2 first, then fallback  
    cublasSgemv = l.lookup(@TypeOf(cublasSgemv), "cublasSgemv_v2") orelse
                l.lookup(@TypeOf(cublasSgemv), "cublasSgemv") orelse return error.SymbolNotFound;
    cublasDgemv = l.lookup(@TypeOf(cublasDgemv), "cublasDgemv_v2") orelse
                l.lookup(@TypeOf(cublasDgemv), "cublasDgemv") orelse return error.SymbolNotFound;

    // Dot product functions - try v2 first, then fallback
    cublasSdot = l.lookup(@TypeOf(cublasSdot), "cublasSdot_v2") orelse
                l.lookup(@TypeOf(cublasSdot), "cublasSdot") orelse return error.SymbolNotFound;
    cublasDdot = l.lookup(@TypeOf(cublasDdot), "cublasDdot_v2") orelse
               l.lookup(@TypeOf(cublasDdot), "cublasDdot") orelse return error.SymbolNotFound;

    // Vector operations - Functions 59-62 - try v2 first, then fallback
    cublasSaxpy = l.lookup(@TypeOf(cublasSaxpy), "cublasSaxpy_v2") orelse
                l.lookup(@TypeOf(cublasSaxpy), "cublasSaxpy") orelse return error.SymbolNotFound;
    cublasDaxpy = l.lookup(@TypeOf(cublasDaxpy), "cublasDaxpy_v2") orelse
               l.lookup(@TypeOf(cublasDaxpy), "cublasDaxpy") orelse return error.SymbolNotFound;
    cublasSscal = l.lookup(@TypeOf(cublasSscal), "cublasSscal_v2") orelse
                l.lookup(@TypeOf(cublasSscal), "cublasSscal") orelse return error.SymbolNotFound;
    cublasDscal = l.lookup(@TypeOf(cublasDscal), "cublasDscal_v2") orelse
               l.lookup(@TypeOf(cublasDscal), "cublasDscal") orelse return error.SymbolNotFound;

    // Batched matrix multiplication - Functions 63-64 - try v2 first, then fallback
    cublasSgemmBatched = l.lookup(@TypeOf(cublasSgemmBatched), "cublasSgemmBatched_v2") orelse
                      l.lookup(@TypeOf(cublasSgemmBatched), "cublasSgemmBatched") orelse return error.SymbolNotFound;
    cublasDgemmBatched = l.lookup(@TypeOf(cublasDgemmBatched), "cublasDgemmBatched_v2") orelse
                         l.lookup(@TypeOf(cublasDgemmBatched), "cublasDgemmBatched") orelse return error.SymbolNotFound;

    // Pointer mode configuration - Function 65 - try v2 first, then fallback
    cublasSetPointerMode = l.lookup(@TypeOf(cublasSetPointerMode), "cublasSetPointerMode_v2") orelse
                      l.lookup(@TypeOf(cublasSetPointerMode), "cublasSetPointerMode") orelse return error.SymbolNotFound;
}
