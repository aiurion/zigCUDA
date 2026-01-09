// src/test_cuda_simple.zig
// Simple test to debug CUDA function calls

const std = @import("std");
const cuda = @import("bindings/cuda.zig");

// Import c_int type
const c_int = cuda.c_int;

pub fn main() !void {
    std.log.info("Testing CUDA functions directly...", .{});
    
    var device_count: c_int = undefined;
    const result = cuda.cudaGetDeviceCount(&device_count);
    std.log.info("cudaGetDeviceCount result: {}", .{@as(c_int, @intCast(result))});
    std.log.info("Device count: {}", .{device_count});
    
    var driver_version: c_int = undefined;
    const version_result = cuda.cudaDriverGetVersion(&driver_version);
    std.log.info("cudaDriverGetVersion result: {}", .{@as(c_int, @intCast(version_result))});
    std.log.info("Driver version: {}", .{driver_version});
}