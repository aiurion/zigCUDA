// src/bindings/curand.zig
// cuRAND API bindings
// TODO: Implement cuRAND bindings

pub const curandStatus_t = extern enum(c_int) {
    success = 0,
    version_mismatch = 1,
    not_initialized = 2,
    allocation_failed = 3,
    invalid_value = 4,
    invalid_device = 5,
    invalid_host = 6,
    invalid_m = 7,
    invalid_n = 8,
    invalid_seed = 9,
    invalid_offset = 10,
    invalid_count = 11,
    invalid_size = 12,
};

pub const curandGenerator = opaque {};

// cuRAND function declarations
pub extern fn curandCreate(generator: *curandGenerator) curandStatus_t;
pub extern fn curandDestroy(generator: curandGenerator) curandStatus_t;
pub extern fn curandSetStream(generator: curandGenerator, stream: anytype) curandStatus_t;

// TODO: Add more cuRAND API bindings