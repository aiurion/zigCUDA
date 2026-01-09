// src/bindings/cublas.zig
// cuBLAS API bindings
// TODO: Implement cuBLAS bindings

pub const cublasStatus_t = extern enum(c_int) {
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

pub const cublasHandle_t = opaque {};
pub const cublasLtMatDescriptor_t = opaque {};
pub const cublasLtMatrixLayout_t = opaque {};

// cuBLAS function declarations
pub extern fn cublasCreate(handle: *cublasHandle_t) cublasStatus_t;
pub extern fn cublasDestroy(handle: cublasHandle_t) cublasStatus_t;
pub extern fn cublasSetStream(handle: cublasHandle_t, stream: anytype) cublasStatus_t;

// TODO: Add more cuBLAS API bindings