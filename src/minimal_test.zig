// src/minimal_test.zig
// Minimal test to debug the segfault

const std = @import("std");

pub extern fn cudaGetDeviceCount(count: *std.c_int) std.c_int;

pub fn main() !void {
    std.log.info("Minimal CUDA test starting...", .{});
    
    var device_count: std.c_int = undefined;
    
    // Call the function directly
    const result = cudaGetDeviceCount(&device_count);
    
    std.log.info("Function returned: {}", .{result});
    std.log.info("Device count: {}", .{device_count});
    
    if (result == 0) {
        std.log.info("SUCCESS! Found {} CUDA device(s)", .{device_count});
    } else {
        std.log.info("FAILED with error code: {}", .{result});
    }
}