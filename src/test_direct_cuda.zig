// src/test_direct_cuda.zig
// Direct test of CUDA functions

const std = @import("std");

// Direct CUDA function declarations
pub extern fn cudaGetDeviceCount(count: *std.c_int) std.c_int;
pub extern fn cudaDriverGetVersion(version: *std.c_int) std.c_int;

pub fn main() !void {
    std.log.info("Direct CUDA test...", .{});
    
    var device_count: std.c_int = undefined;
    const result = cudaGetDeviceCount(&device_count);
    std.log.info("cudaGetDeviceCount result: {}", .{result});
    std.log.info("Device count: {}", .{device_count});
    
    var driver_version: std.c_int = undefined;
    const version_result = cudaDriverGetVersion(&driver_version);
    std.log.info("cudaDriverGetVersion result: {}", .{version_result});
    std.log.info("Driver version: {}", .{driver_version});
}