// src/main.zig
// Main entry point for ZigCUDA testing

const std = @import("std");

// Use real CUDA bindings to test actual hardware
const cuda = @import("bindings/cuda.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("ZigCUDA Phase 0 Testing", .{});

    // 1. Initialize CUDA driver
    cuda.init(0) catch |err| {
        std.log.err("Failed to initialize CUDA: {}", .{err});
        return err;
    };
    std.log.info("CUDA initialized successfully!", .{});

    // 2. Get CUDA version
    const version = cuda.getVersion() catch |err| {
        std.log.err("Failed to get CUDA version: {}", .{err});
        return err;
    };
    std.log.info("CUDA Driver Version: {}.{}", .{ version[0], version[1] });

    // 3. Get device count
    const device_count = cuda.getDeviceCount() catch |err| {
        std.log.err("Failed to get device count: {}", .{err});
        return err;
    };
    std.log.info("Found {} CUDA device(s)", .{device_count});

    // Test each device
    var i: i32 = 0;
    while (i < device_count) : (i += 1) {
        std.log.info("Testing Device {}", .{i});

        // 4. Get device properties
        // const props = cuda.getDeviceProperties(i) catch |err| {
        //     std.log.err("Failed to get device properties: {}", .{err});
        //     continue;
        // };
        // std.log.info("  Max Threads per Block: {}", .{props.maxThreadsPerBlock});

        // 5. Get device name
        const name = cuda.getDeviceName(i, allocator) catch |err| {
            std.log.err("Failed to get device name: {}", .{err});
            continue;
        };
        defer allocator.free(name);
        std.log.info("  Name: {s}", .{name});

        // 6. Get compute capability
        const cc = cuda.getComputeCapability(i) catch |err| {
            std.log.err("Failed to get compute capability: {}", .{err});
            continue;
        };
        std.log.info("  Compute Capability: {}.{}", .{ cc.major, cc.minor });

        // 7. Get total memory
        const total_mem = cuda.getTotalMem(i) catch |err| {
            std.log.err("Failed to get total memory: {}", .{err});
            continue;
        };
        std.log.info("  Total Memory: {} bytes ({d:.2} MB)", .{ total_mem, @as(f64, @floatFromInt(total_mem)) / 1024.0 / 1024.0 });
    }

    // 8. Test getErrorString
    const err_str = cuda.getErrorString(0) catch |err| { // 0 is success
        std.log.err("Failed to get error string: {}", .{err});
        return err;
    };
    std.log.info("Error String for 0 (Success): {s}", .{err_str});

    // Test a specific error code
    const err_str_val = cuda.getErrorString(1) catch "Unknown"; // 1 is Invalid Value
    std.log.info("Error String for 1 (Invalid Value): {s}", .{err_str_val});

    std.log.info("Phase 0 All Tests completed successfully!", .{});
}
