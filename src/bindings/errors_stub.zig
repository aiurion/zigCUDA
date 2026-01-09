// src/bindings/errors_stub.zig
// Stub error types for development without CUDA

// Simple error type for stub
pub const CUDAError = error{
    InitializationFailed,
    DeviceNotFound,
    ContextCreationFailed,
    MemoryAllocationFailed,
};