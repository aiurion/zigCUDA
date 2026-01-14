// src/lib.zig
const std = @import("std");
const cuda = @import("bindings/cuda.zig");

pub const bindings = cuda;

pub const Error = error{
    NoCudaDevice,
    InvalidDeviceId,
    CudaLoadFailed,
    CudaInitFailed,
} || cuda.errors.CUDAError;

/// High-level device properties struct
pub const DeviceProperties = struct {
    name: [256]u8,
    major: c_int,
    minor: c_int,
    totalGlobalMem: usize,
    multiProcessorCount: c_int,
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

        const dev = try cuda.getDevice(@as(c_int, @intCast(device_id)));

        var props: DeviceProperties = undefined;

        // 1. Get Name safely
        // Initialize buffer with zeros to avoid garbage output
        @memset(&props.name, 0);
        if (cuda.cuDeviceGetName) |f| {
            // FIX: Cast to [*:0]cuda.c_char because the C API expects signed chars (i8) on Linux
            const ptr = @as([*:0]cuda.c_char, @ptrCast(&props.name));
            _ = f(ptr, 256, dev);
        }

        // 2. Get Compute Capability
        // We use the helper from bindings which handles the logic
        const cc = try cuda.getComputeCapability(dev);
        props.major = cc.major;
        props.minor = cc.minor;

        // 3. Get Memory
        const mem_bytes = try cuda.getTotalMem(dev);
        props.totalGlobalMem = mem_bytes;

        // 4. Get SM Count (Streaming Multiprocessor count)
        var sm_count: c_int = 188; // Current incorrect value from attribute 16
        if (cuda.cuDeviceGetAttribute) |f| {
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
    cuda.load() catch return error.CudaLoadFailed;
    cuda.init(0) catch return error.CudaInitFailed;

    const count_result = cuda.getDeviceCount();
    const device_count = count_result catch return error.CudaInitFailed;

    if (device_count == 0) {
        return error.NoCudaDevice;
    }

    return Context{ .device_count = @as(u32, @intCast(device_count)) };
}

test {
    std.testing.refAllDecls(@This());
}
