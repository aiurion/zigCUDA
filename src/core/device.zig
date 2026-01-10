// src/core/device.zig
// GPU device enumeration and properties
// Phase 1.1: Device Management Implementation

const std = @import("std");
const bindings = @import("../bindings/cuda.zig");
const errors = @import("../bindings/errors.zig");

pub const DeviceProperties = struct {
    name: [256]u8,
    total_memory: usize,
    max_threads_per_block: u32,
    max_grid_size: [3]u32,
    shared_mem_per_block: u32,
    warp_size: u32,
    multiprocessor_count: u32,
    compute_capability: struct {
        major: u32,
        minor: u32,
    },
};

pub const Device = struct {
    handle: bindings.CUdevice,
    index: u32,
    properties: DeviceProperties,

    /// Initialize a device by ordinal number
    pub fn init(ordinal: u32) !Device {
        // First, ensure CUDA is initialized
        try bindings.init(0);

        const properties = try queryDeviceProperties(@intCast(ordinal));

        return Device{
            .handle = @intCast(ordinal),
            .index = ordinal,
            .properties = properties,
        };
    }

    /// Get total number of CUDA devices available
    pub fn count() errors.CUDAError!u32 {
        try bindings.init(0); // Ensure CUDA is initialized
        const device_count = try bindings.getDeviceCount();
        return @intCast(device_count);
    }

    /// Get a specific device by ordinal (0-based index)
    pub fn get(ordinal: u32) !Device {
        if (ordinal >= try count()) {
            return errors.CUDAError.InvalidDevice;
        }

        const properties = try queryDeviceProperties(@intCast(ordinal));

        return Device{
            .handle = @intCast(ordinal),
            .index = ordinal,
            .properties = properties,
        };
    }

    /// Get the best device (highest compute capability)
    pub fn getBest() !Device {
        const num_devices = try count();
        if (num_devices == 0) {
            return errors.CUDAError.NoDevice;
        }

        var best_device: Device = undefined;
        var best_score: u32 = 0;
        var has_valid_device = false;

        for (0..@intCast(num_devices)) |i| {
            const device = try get(@intCast(i));
            const score = device.properties.compute_capability.major * 100 +
                device.properties.compute_capability.minor;

            if (!has_valid_device or score > best_score) {
                best_score = score;
                best_device = device;
                has_valid_device = true;
            }
        }

        // If we get here, we should have at least one valid device
        return best_device;
    }

    /// Set this device as current
    pub fn setCurrent(self: Device) !void {
        // Create a context for the device and make it current
        var ctx: ?*bindings.CUcontext = null;

        // First try to get existing context
        const get_ctx_result = bindings.cuCtxGetCurrent(&ctx);
        if (get_ctx_result == bindings.CUDA_SUCCESS and ctx != null) {
            return; // Already current
        }

        // Create new context for this device
        const create_result = bindings.cuCtxCreate(&ctx, 0, self.handle);
        if (create_result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(create_result);
        }

        // Set as current context
        if (ctx != null) {
            const set_result = bindings.cuCtxSetCurrent(ctx.?);
            if (set_result != bindings.CUDA_SUCCESS) {
                _ = bindings.cuCtxDestroy(ctx.?); // Clean up on failure
                return errors.cudaError(set_result);
            }
        }
    }

    /// Get device properties
    pub fn getProperties(self: *const Device) DeviceProperties {
        return self.properties;
    }

    /// Query all properties for a given device handle (device_id is the ordinal number)
    fn queryDeviceProperties(device_handle: bindings.CUdevice) !DeviceProperties {
        // Initialize with zeros/empty values
        var name_buffer: [256]u8 = std.mem.zeroes([256]u8);

        // Get device name using proper function call style
        const name_result = if (bindings.cuDeviceGetName) |f|
            f(@as([*:0]bindings.c_char, @ptrCast(&name_buffer)), 256, device_handle)
        else
            bindings.CUDA_ERROR_UNKNOWN;

        if (name_result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(name_result);
        }

        // Get total memory
        var total_memory: usize = 0;
        if (bindings.cuDeviceTotalMem) |f| {
            var mem_bytes: u64 = 0;
            const res = f(&mem_bytes, device_handle);
            if (res == bindings.CUDA_SUCCESS) {
                total_memory = @intCast(mem_bytes);
            }
        }

        // Get compute capability
        var major: u32 = 0;
        var minor: u32 = 0;
        if (bindings.cuDeviceComputeCapability) |f| {
            var c_major: i32 = 0;
            var c_minor: i32 = 0;
            if (f(&c_major, &c_minor, device_handle) == bindings.CUDA_SUCCESS) {
                major = @intCast(c_major);
                minor = @intCast(c_minor);
            }
        } else if (bindings.cuDeviceGetAttribute) |f| {
            var c_major: i32 = 0;
            var c_minor: i32 = 0;
            _ = f(&c_major, 75, device_handle); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            _ = f(&c_minor, 76, device_handle); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            major = @intCast(c_major);
            minor = @intCast(c_minor);
        }

        // Get legacy properties structure for additional info
        var dev_prop: bindings.CUdevprop = std.mem.zeroes(bindings.CUdevprop);
        const prop_result = if (bindings.cuDeviceGetProperties) |f|
            f(&dev_prop, device_handle)
        else
            bindings.CUDA_ERROR_NOT_FOUND;

        return DeviceProperties{
            .name = name_buffer,
            .total_memory = total_memory,
            .max_threads_per_block = if (prop_result == bindings.CUDA_SUCCESS)
                @as(u32, @intCast(dev_prop.maxThreadsPerBlock))
            else
                0,
            .max_grid_size = if (prop_result == bindings.CUDA_SUCCESS)
                [3]u32{ @as(u32, @intCast(dev_prop.maxGridSize[0])), @as(u32, @intCast(dev_prop.maxGridSize[1])), @as(u32, @intCast(dev_prop.maxGridSize[2])) }
            else
                [3]u32{ 0, 0, 0 },
            .shared_mem_per_block = if (prop_result == bindings.CUDA_SUCCESS)
                @truncate(@as(u64, @intCast(dev_prop.sharedMemPerBlock)))
            else
                0,

            .warp_size = if (prop_result == bindings.CUDA_SUCCESS and dev_prop.warpSize > 0)
                @intCast(dev_prop.warpSize)
            else
                32,
            .multiprocessor_count = 1, // CUdevprop doesn't contain multiprocessor count info
            .compute_capability = .{
                .major = major,
                .minor = minor,
            },
        };
    }
};

/// Get selection of all available CUDA devices
pub fn getAllDevices(allocator: std.mem.Allocator) ![]Device {
    const count = try Device.count();
    var devices = try allocator.alloc(Device, count);

    for (0..count) |i| {
        devices[i] = try Device.get(@intCast(i));
    }

    return devices;
}
