// src/simple_cuda_test.zig
// Simple test to isolate the segfault issue

const std = @import("std");

pub extern fn cudaGetDeviceCount(count: *std.c_int) std.c_int;

pub fn main() !void {
    std.log.info("Simple CUDA test - device count only", .{});
    
    var device_count: std.c_int = undefined;
    const result = cudaGetDeviceCount(&device_count);
    
    if (result == 0) {
        std.log.info("SUCCESS: Found {} CUDA device(s)", .{device_count});
    } else {
        std.log.info("ERROR: cudaGetDeviceCount failed with code {}", .{result});
    }
    
    std.log.info("Test completed successfully!", .{});
}