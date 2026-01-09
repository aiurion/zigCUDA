// src/bindings/errors.zig
// Error code mappings and Zig error types
// TODO: Implement error handling

pub const Error = error{
    InvalidDevice,
    DeviceNotInUse,
    DeviceAlreadyActive,
    ContextAlreadyCurrent,
    ContextPopMismatch,
    NoContext,
    InvalidContext,
    InvalidContextHandle,
    Uninitialized,
    InvalidValue,
    MemoryAllocation,
    MemoryFree,
    Unknown,
    VersionMismatch,
    NotInitialized,
    AllocationFailed,
    InvalidValue,
    InvalidDevice,
    InvalidHost,
    InvalidM,
    InvalidN,
    InvalidSeed,
    InvalidOffset,
    InvalidCount,
    InvalidSize,
};

pub fn mapCUresultToError(result: anytype) Error!void {
    switch (result) {
        .success => return,
        .invalid_device => return Error.InvalidDevice,
        .device_not_in_use => return Error.DeviceNotInUse,
        .device_already_active => return Error.DeviceAlreadyActive,
        .context_already_current => return Error.ContextAlreadyCurrent,
        .context_pop_mismatch => return Error.ContextPopMismatch,
        .no_context => return Error.NoContext,
        ._invalid_context => return Error.InvalidContext,
        .invalid_context_handle => return Error.InvalidContextHandle,
        .cuda_uninitialized => return Error.Uninitialized,
        .invalid_value => return Error.InvalidValue,
        .memory_allocation => return Error.MemoryAllocation,
        .memory_free => return Error.MemoryFree,
        .unknown => return Error.Unknown,
        else => return Error.Unknown,
    }
}

// TODO: Add more error mappings