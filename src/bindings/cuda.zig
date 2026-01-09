// src/bindings/cuda.zig
// Core CUDA Driver API declarations
// TODO: Implement low-level CUDA bindings

pub const CUdriver = extern struct {
    major: c_int,
    minor: c_int,
};

pub const CUresult = extern enum(c_int) {
    success = 0,
    invalid_device = 1,
    device_not_in_use = 2,
    device_already_active = 3,
    context_already_current = 4,
    context_pop_mismatch = 5,
    no_context = 6,
   _invalid_context = 7,
    invalid_context_handle = 8,
    cuda_uninitialized = 9,
    invalid_value = 10,
    memory_allocation = 11,
    memory_free = 12,
    unknown = -1,
};

pub const CUdevice = opaque {};
pub const CUcontext = opaque {};
pub const CUstream = opaque {};
pub const CUevent = opaque {};

// Function declarations will be added here
pub extern fn cuInit(flags: c_uint) CUresult;
pub extern fn cuDriverGetVersion(version: *c_int) CUresult;
pub extern fn cuDeviceGetCount(count: *c_int) CUresult;
pub extern fn cuDeviceGetDevice(device: *CUdevice, ordinal: c_int) CUresult;

// TODO: Add more CUDA API bindings