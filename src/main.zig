// src/main.zig

const std = @import("std");
const cuda = @import("bindings/cuda.zig");

pub fn main() void {
    // Simple, clean startup message
    std.debug.print("\n=== ZigCUDA: Native CUDA API for Zig ===\n", .{});

    // Try to detect and initialize CUDA
    const cuda_available = detectCuda();

    if (cuda_available) {
        std.debug.print("✓ CUDA detected and initialized successfully\n", .{});
    } else {
        std.debug.print("✗ No CUDA available - using stub mode\n", .{});
    }

    if (cuda_available) {
        std.debug.print("\n=== ZigCUDA Ready ===\n\n", .{});
    } else {
        std.debug.print("\n=== ZigCUDA Ready (Stub Mode) ===\n\n", .{});
    }
}

/// Detect if CUDA is available by trying to initialize
fn detectCuda() bool {
    // Try to load CUDA library
    cuda.load() catch |err| {
        std.debug.print("Failed to load CUDA library: {}\n", .{err});
        return false;
    };

    // Try to initialize CUDA
    cuda.init(0) catch |err| {
        std.debug.print("Failed to initialize CUDA: {}\n", .{err});
        return false;
    };

    // Try to get device count
    const count = cuda.getDeviceCount() catch |err| {
        std.debug.print("Failed to get device count: {}\n", .{err});
        return false;
    };

    std.debug.print("Found {d} CUDA device(s)\n", .{count});
    return count > 0;
}

/// Helper function
fn fileExists(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch return false;
    defer file.close();
    return true;
}
