// src/lib.zig - Unified entry point for ZigCUDA
const std = @import("std");
pub const bindings = @import("bindings/cuda.zig");
pub const errors = @import("bindings/errors.zig");

// Re-export core abstractions as the primary public API
pub const Device = @import("core/device.zig").Device;
pub const DeviceProperties = @import("core/device.zig").DeviceProperties;
pub const Context = @import("core/context.zig").Context;
pub const Module = @import("core/module.zig").Module;
pub const Stream = @import("core/stream.zig").Stream;
pub const Kernel = @import("core/kernel.zig").Kernel;

// Metadata
pub const version = std.SemanticVersion{ .major = 0, .minor = 0, .patch = 1 };
pub const version_string = "0.0.1";

/// Initialize the CUDA driver and load necessary symbols.
/// This must be called before using any other CUDA functionality.
pub fn init() !void {
    try bindings.load();
    try bindings.init(0);
}

// Convenience re-exports for common low-level operations
pub const allocDeviceMemory = bindings.allocDeviceMemory;
pub const freeDeviceMemory = bindings.freeDeviceMemory;
pub const copyHostToDevice = bindings.copyHostToDevice;
pub const copyDeviceToHost = bindings.copyDeviceToHost;
pub const launchKernel = bindings.launchKernel;

// Opaque handles for advanced usage
pub const CUdevice = bindings.CUdevice;
pub const CUcontext = bindings.CUcontext;
pub const CUstream = bindings.CUstream;
pub const CUevent = bindings.CUevent;
pub const CUmodule = bindings.CUmodule;
pub const CUfunction = bindings.CUfunction;
pub const CUdeviceptr = bindings.CUdeviceptr;

test {
    std.testing.refAllDecls(@This());
}
