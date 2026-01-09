// src/test_basic_cuda.zig
// Basic test to debug CUDA function calls

const std = @import("std");

// Direct CUDA function declarations - keep it simple
pub extern fn cudaGetDeviceCount(count: *std.c_int) std.c_int;

pub fn main() !void {
    std.log.info("Basic CUDA test...", .{});
    
    var device_count: std.c_int = undefined;
    const result = cudaGetDeviceCount(&device_count);
    
    if (result == 0) {
        std.log.info("SUCCESS: cudaGetDeviceCount returned 0");
        std.log.info("Device count: {}", .{device_count});
    } else {
        std.log.info("ERROR: cudaGetDeviceCount returned {}", .{result});
    }
}