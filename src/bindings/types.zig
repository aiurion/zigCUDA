// src/bindings/types.zig
// CUDA type definitions and constants
// TODO: Implement type definitions

pub const c_int = @import("std").c_int;
pub const c_uint = @import("std").c_uint;

// Device properties
pub const DeviceAttribute = enum(c_int) {
    max_threads_per_block = 0,
    max_threads_dim = 1,
    max_grid_size = 2,
    shared_mem_per_block = 3,
    warp_size = 4,
    shared_mem_per_multiprocessor = 5,
    processor_count = 6,
    multiprocessor_count = 7,
    is_integrated = 8,
    can_map_host_memory = 9,
    unified_address_space = 10,
    compute_verification_mode = 11,
    unified_address_space_reporting = 12,
    cooperative_kernel_launch = 13,
    cooperative_device_copy_launch = 14,
    cooperative_kernel_launch_reporting = 15,
    _,
};

// Memory types
pub const MemoryType = enum(c_int) {
    host = 0,
    device = 1,
    unified = 2,
    managed = 3,
};

// Compute modes
pub const ComputeMode = enum(c_int) {
    default = 0,
    exclusive = 1,
    prohibited = 2,
};

// Function cache configurations
pub const FuncCacheConfig = enum(c_int) {
    prefer_none = 0,
    prefer_shared_mem = 1,
    prefer_cache = 2,
    prefer_global_mem = 3,
};

// Stream capture modes  
pub const StreamCaptureMode = enum(c_int) {
    none = -1,        // Not capturing
    incremental = 0, // Incremental stream capture mode
    all = 1,         // Capture all operations in the stream
};

// Stream creation flags
pub const StreamFlags = enum(c_uint) {
    default = 0,
    non_blocking = 1,       // Operations don't block host thread
    high_priority = 2,   // High-priority for time-critical ops
};

// Cooperative kernel launch flags
pub const CooperativeLaunchFlags = enum(c_uint) {
    default = 0,
    multi_device = 1 << 0,
    device_function = 1 << 1,
};

// Stream callback function type
pub const CUstreamCallback = *const fn (stream: anyopaque, status: c_int, userdata: ?*anyopaque) callconv(.c) void;

// TODO: Add more type definitions