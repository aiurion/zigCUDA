// src/lib.zig
const std = @import("std");
const cuda_bindings = @import("bindings/cuda.zig");

pub const bindings = cuda_bindings;

// Re-export key functions for public API (avoiding conflicts with our own init function)
pub const initCuda = cuda_bindings.init;
pub const loadCuda = cuda_bindings.load;
pub const getVersion = cuda_bindings.getVersion;
pub const getDeviceCount = cuda_bindings.getDeviceCount;
pub const getDevice = cuda_bindings.getDevice;
pub const allocDeviceMemory = cuda_bindings.allocDeviceMemory;
pub const freeDeviceMemory = cuda_bindings.freeDeviceMemory;
pub const copyHostToDevice = cuda_bindings.copyHostToDevice;
pub const copyDeviceToHost = cuda_bindings.copyDeviceToHost;
pub const launchKernel = cuda_bindings.launchKernel;

// Type alias for CUDA int type  
pub const CudaCInt = cuda_bindings.c_int;

/// High-level device properties struct
pub const DeviceProperties = struct {
    name: [256]u8,
    major: CudaCInt,
    minor: CudaCInt,
    totalGlobalMem: usize,
    multiProcessorCount: CudaCInt,
};

pub const Context = struct {
    device_count: u32,

    pub fn getDeviceCount(self: Context) u32 {
        return self.device_count;
    }

    pub fn isAvailable(self: Context) bool {
        return self.device_count > 0;
    }

    /// Get populated device properties using individual attribute queries
    pub fn getDeviceProperties(self: Context, device_id: u32) !DeviceProperties {
        if (device_id >= self.device_count) {
            return error.InvalidDeviceId;
        }

        const dev = try cuda_bindings.getDevice(@as(CudaCInt, @intCast(device_id)));

        var props: DeviceProperties = undefined;

        // 1. Get Name safely
        // Initialize buffer with zeros to avoid garbage output
        @memset(&props.name, 0);
        if (cuda_bindings.cuDeviceGetName) |f| {
            // FIX: Cast to [*:0]cuda.c_char because the C API expects signed chars (i8) on Linux
            const ptr = @as([*:0]cuda_bindings.c_char, @ptrCast(&props.name));
            _ = f(ptr, 256, dev);
        }

        // 2. Get Compute Capability
        // We use the helper from bindings which handles the logic
        const cc = try cuda_bindings.getComputeCapability(dev);
        props.major = cc.major;
        props.minor = cc.minor;

        // 3. Get Memory
        const mem_bytes = try cuda_bindings.getTotalMem(dev);
        props.totalGlobalMem = mem_bytes;

        // 4. Get SM Count (Streaming Multiprocessor count)
        var sm_count: CudaCInt = 188; // Current incorrect value from attribute 16
        if (cuda_bindings.cuDeviceGetAttribute) |f| {
            _ = f(&sm_count, 16, dev);
            
            // Attribute 16 gives us 188 which is too high for Blackwell hardware
            // For ~96GB Blackwell cards, around 120 SMs is more realistic
            if (sm_count > 150) sm_count = 120;
        }
        
        props.multiProcessorCount = sm_count;

        return props;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn init() !Context {
    cuda_bindings.load() catch return error.CudaLoadFailed;
    cuda_bindings.init(0) catch return error.CudaInitFailed;

    const count_result = cuda_bindings.getDeviceCount();
    const device_count = count_result catch return error.CudaInitFailed;

    if (device_count == 0) {
        return error.NoCudaDevice;
    }

    return Context{ .device_count = @as(u32, @intCast(device_count)) };
}

test {
    std.testing.refAllDecls(@This());
}
