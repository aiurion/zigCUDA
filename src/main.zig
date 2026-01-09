// src/main.zig
// Main entry point for ZigCUDA testing

const std = @import("std");

// Use real CUDA bindings to test actual hardware
const cuda = @import("bindings/cuda.zig");

pub fn main() !void {
    _ = std; // silence unused warning
    
    std.log.info("ZigCUDA Phase 0 Testing", .{});
    
    // Initialize CUDA driver
    cuda.init(0) catch |err| {
        std.log.err("Failed to initialize CUDA: {}", .{err});
        return err;
    };
    std.log.info("CUDA initialized successfully!", .{});
    
    // // Get CUDA version
    // const version = cuda.getVersion() catch |err| {
    //     std.log.err("Failed to get CUDA version: {}", .{err});
    //     return err;
    // };
    // std.log.info("CUDA Driver Version: {}.{}", .{ version[0], version[1] });
    
    // // Get device count
    // const device_count = cuda.getDeviceCount() catch |err| {
    //     std.log.err("Failed to get device count: {}", .{err});
    //     return err;
    // };
    // std.log.info("Found {} CUDA device(s)", .{device_count});
    
    std.log.info("Phase 0 cuInit test completed successfully!", .{});
}