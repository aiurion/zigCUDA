// src/bindings/errors.zig
// Error code mappings and Zig error types
// Phase 0: Essential error handling implementation

const std = @import("std");

// Zig error union for CUDA operations
pub const CUDAError = error{
    /// Invalid device specification
    InvalidDevice,
    /// Device not in use
    DeviceNotInUse,
    /// Device already active
    DeviceAlreadyActive,
    /// Context already current
    ContextAlreadyCurrent,
    /// Context pop mismatch
    ContextPopMismatch,
    /// No context
    NoContext,
    /// Invalid context
    InvalidContext,
    /// Invalid context handle
    InvalidContextHandle,
    /// CUDA uninitialized
    Uninitialized,
    /// Invalid value
    InvalidValue,
    /// Memory allocation failed
    MemoryAllocation,
    /// Memory free failed
    MemoryFree,
    /// Out of memory
    OutOfMemory,
    /// Not ready
    NotReady,
    /// Unknown error
    Unknown,
    /// No device found
    NoDevice,
    /// Operating System Error
    OperatingSystemError,
    /// CUDA library not found
    CudaLibraryNotFound,
    /// Symbol not found in library
    SymbolNotFound,
    /// Feature not supported
    NotSupported,
    /// Resource or symbol not found
    NotFound,
};

/// Map CUDA result to Zig error (assumes result is not success)
pub fn cudaError(result: c_int) CUDAError {
    return switch (result) {
        0 => error.Unknown, // success shouldn't be here
        1 => error.InvalidValue,
        2 => error.OutOfMemory,
        3 => error.Uninitialized, // CUDA_ERROR_NOT_INITIALIZED
        4 => error.Uninitialized, // CUDA_ERROR_DEINITIALIZED

        100 => error.NoDevice,
        101 => error.InvalidDevice,

        200 => error.NotSupported, // Invalid Image - feature not supported for this image format
        201 => error.InvalidContext,
        202 => error.ContextAlreadyCurrent,

        210 => error.DeviceAlreadyActive,

        300 => error.InvalidValue, // Invalid Source
        301 => error.InvalidValue, // File not found (mapped to invalid value for now)
        304 => error.OperatingSystemError,

        400 => error.InvalidContextHandle, // Invalid Handle

        500 => error.NotFound, // Not Found - resource or symbol not found
        600 => error.NotReady,

        700 => error.MemoryAllocation, // Illegal Address
        719 => error.Unknown, // Launch Failed

        999 => error.Unknown,
        else => error.Unknown,
    };
}

/// Alias for cudaError to maintain compatibility with existing code
pub fn mapCudaError(result: c_int) CUDAError {
    return cudaError(result);
}

/// Convert CUresult to error string
pub fn resultToString(result: c_int) []const u8 {
    return switch (result) {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        304 => "CUDA_ERROR_OPERATING_SYSTEM",
        500 => "CUDA_ERROR_NOT_FOUND",
        999 => "CUDA_ERROR_UNKNOWN",
        else => "CUDA_ERROR_UNKNOWN_CODE",
    };
}

/// Check if result indicates success
pub fn isSuccess(result: c_int) bool {
    return result == 0;
}

/// Check if result indicates an error
pub fn isError(result: c_int) bool {
    return result != 0;
}

/// Convenience function to handle CUDA results
pub fn handleResult(result: c_int) CUDAError!void {
    if (result == 0) return;
    return cudaError(result);
}
