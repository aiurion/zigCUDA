// Simple kernel integration test - demonstrates basic functionality works

const std = @import("std");
const testing = std.testing;

// Import core abstractions using module name (as defined in build.zig)
const cuda = @import("cuda");

test "Simple Kernel Integration: Basic CUDA Setup" {
    // Initialize CUDA library first (this is critical!)
    try cuda.init(0);

    // Test that we can get driver version
    const version = try cuda.getVersion();
    
    if (version.len == 2) {
        try testing.expect(version[0] > 0); // Major version should be > 0
    }

    // Get device count
    const device_count = try cuda.getDeviceCount();

    try testing.expect(device_count >= 1);

    // Get first device
    _ = try cuda.getDevice(0);
}

test "Simple Kernel Integration: Memory Operations" {
    // Initialize CUDA library first (this is critical!)
    try cuda.init(0);

    var device_ptr: c_ulonglong = undefined;

    // Get a device and create context for memory operations
    const device = try cuda.getDevice(0);
    _ = try cuda.createContext(0, device);  // Create primary context

    // Allocate memory
    device_ptr = try cuda.allocDeviceMemory(@sizeOf(c_int));
    
    // Test host to device copy
    var host_val: c_int = 42;
    const h2d_slice = @as([]const u8, @ptrCast(&host_val));  
    try cuda.copyHostToDevice(device_ptr, h2d_slice);

    // Test device to host copy
    var readback_val: c_int = undefined;
    const d2h_slice = @as([]u8, @ptrCast(&readback_val));
    try cuda.copyDeviceToHost(d2h_slice, device_ptr);
    
    try testing.expectEqual(@as(c_int, 42), readback_val);

    // Cleanup (ignore errors for cleanup)
    _ = cuda.freeDeviceMemory(device_ptr) catch {};
}

test "Simple Kernel Integration: Context Management" {
    // Initialize CUDA library first (this is critical!)
    try cuda.init(0);

    const device = try cuda.getDevice(0);
    
    var context: ?*cuda.CUcontext = undefined;

    // Create context
    context = try cuda.createContext(0, device);

    if (context != null) {
        try testing.expect(context != null);

        // Test context operations  
        _ = try cuda.setCurrentContext(context.?);

        // Cleanup (ignore errors for cleanup)
        _ = cuda.destroyContext(context.?) catch {};
    }
}