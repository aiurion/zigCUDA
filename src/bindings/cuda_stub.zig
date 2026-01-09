// src/bindings/cuda_stub.zig
// Development stub for when CUDA is not available
// This allows the project to compile and test non-CUDA functionality

const std = @import("std");
const errors = @import("errors_stub.zig");

// Stub error types
pub const CUresult = enum(c_int) {
    success = 0,
    cuda_uninitialized = 999,
    unknown = 1000,
};

// Stub device type
pub const CUdevice = opaque {};

// Stub context type  
pub const CUcontext = opaque {};

// Stub module type
pub const CUmodule = opaque {};

// Stub function type
pub const CUfunction = opaque {};

// Stub functions that return success
pub fn cuInit(flags: c_uint) CUresult {
    _ = flags; // silence unused warning
    return .success;
}

pub fn cuDriverGetVersion(version: *c_int) CUresult {
    version.* = 1200; // CUDA 12.0
    return .success;
}

pub fn cuDeviceGetCount(count: *c_int) CUresult {
    count.* = 0; // No devices available
    return .success;
}

pub fn cuGetErrorString(result: CUresult, pstr: [*:0]c_char) CUresult {
    _ = result;
    _ = pstr;
    return .success;
}

pub fn cuModuleLoad(pmodule: *?*CUmodule, fname: [*:0]c_char) CUresult {
    _ = pmodule;
    _ = fname;
    return .success;
}

pub fn cuModuleLoadData(pmodule: *?*CUmodule, image: [*:0]c_char) CUresult {
    _ = pmodule;
    _ = image;
    return .success;
}

pub fn cuModuleGetFunction(pfunc: *?*CUfunction, module: *CUmodule, name: [*:0]c_char) CUresult {
    _ = pfunc;
    _ = module;
    _ = name;
    return .success;
}

pub fn cuGetErrorName(result: CUresult, pstr: [*:0]c_char) CUresult {
    _ = result;
    _ = pstr;
    return .success;
}

// Stub wrapper functions
pub fn init() !void {
    const result = cuInit(0);
    if (result != .success) {
        return errors.CUDAError.InitializationFailed;
    }
}

pub fn getVersion() ![2]c_int {
    var version: [2]c_int = undefined;
    const result = cuDriverGetVersion(&version[0]);
    if (result != .success) {
        return errors.CUDAError.InitializationFailed;
    }
    return version;
}

pub fn getDeviceCount() !c_int {
    var count: c_int = undefined;
    const result = cuDeviceGetCount(&count);
    if (result != .success) {
        return errors.CUDAError.InitializationFailed;
    }
    return count;
}