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

// TODO: Add more type definitions