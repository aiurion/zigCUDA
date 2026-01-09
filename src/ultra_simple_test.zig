// src/ultra_simple_test.zig
// Ultra minimal test - just call one CUDA function

const std = @import("std");

// Direct extern declaration - no wrappers
pub extern fn cudaGetDeviceCount(count: *std.c_int) std.c_int;

pub fn main() !void {
    std.log.info("Ultra simple test...", .{});
    
    var count: std.c_int = undefined;
    const result = cudaGetDeviceCount(&count);
    
    if (result == 0) {
        std.log.info("SUCCESS: {} devices", .{count});
    } else {
        std.log.info("ERROR: {}", .{result});
    }
}