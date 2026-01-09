// src/bindings/cublas.zig
// cuBLAS API bindings (Dynamic Loading)

const std = @import("std");

pub const cublasStatus_t = enum(c_int) {
    success = 0,
    invalid_value = 1,
    not_supported = 2,
    allocation_failed = 3,
    internal_error = 4,
    invalid_handle = 5,
    driver_version = 6,
    runtime_warning = 7,
    runtime_pending = 8,
};

pub const cublasHandle_t = ?*anyopaque;
pub const cublasLtMatDescriptor_t = ?*anyopaque;
pub const cublasLtMatrixLayout_t = ?*anyopaque;

// Library state
var lib: ?std.DynLib = null;

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

pub fn load() !void {
    if (lib != null) return;

    // Try common library names
    // On Linux typically libcublas.so or libcublas.so.11 (or .12)
    const lib_names = [_][]const u8{ "libcublas.so", "libcublas.so.12", "libcublas.so.11", "libcublas.so.10" };

    for (lib_names) |name| {
        lib = std.DynLib.open(name) catch continue;
        break;
    }

    if (lib == null) {
        // Fallback for WSL default locations if not in LD_LIBRARY_PATH
        const wsl_path = "/usr/lib/wsl/lib/libcublas.so.12"; // V12 is common now
        lib = std.DynLib.open(wsl_path) catch blk: {
            // Try older version
            break :blk std.DynLib.open("/usr/lib/wsl/lib/libcublas.so.11") catch null;
        };
    }

    if (lib == null) return error.CublasLibraryNotFound;
    const l = &lib.?;

    // Load functions with 'v2' suffix which is standard for cuBLAS
    cublasCreate = l.lookup(@TypeOf(cublasCreate), "cublasCreate_v2") orelse return error.SymbolNotFound;
    cublasDestroy = l.lookup(@TypeOf(cublasDestroy), "cublasDestroy_v2") orelse return error.SymbolNotFound;

    // Note: cublasSetStream_v2 is standard
    cublasSetStream = l.lookup(@TypeOf(cublasSetStream), "cublasSetStream_v2") orelse return error.SymbolNotFound;

    cublasSgemm = l.lookup(@TypeOf(cublasSgemm), "cublasSgemm_v2") orelse return error.SymbolNotFound;
    cublasDgemm = l.lookup(@TypeOf(cublasDgemm), "cublasDgemm_v2") orelse return error.SymbolNotFound;

    cublasSgemv = l.lookup(@TypeOf(cublasSgemv), "cublasSgemv_v2") orelse return error.SymbolNotFound;
    cublasDgemv = l.lookup(@TypeOf(cublasDgemv), "cublasDgemv_v2") orelse return error.SymbolNotFound;

    cublasSdot = l.lookup(@TypeOf(cublasSdot), "cublasSdot_v2") orelse return error.SymbolNotFound;
    cublasDdot = l.lookup(@TypeOf(cublasDdot), "cublasDdot_v2") orelse return error.SymbolNotFound;
}
