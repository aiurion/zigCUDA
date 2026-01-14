// examples/library_usage.zig - Example of using ZigCUDA as a library

const std = @import("std");
const zigcuda = @import("zigcuda"); // Import our library

pub fn main() !void {
    std.debug.print("\n=== Using ZigCUDA Library ===\n\n", .{});

    // Initialize the library
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    if (!ctx.isAvailable()) {
        std.debug.print("❗ CUDA not available on this system\n", .{});
        return;
    }

    const device_count = ctx.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n\n", .{device_count});

    // Show basic info for each device (up to 3)
    const max_devices = @min(device_count, @as(usize, 3));
    for (0..max_devices) |i| {
        const props = try ctx.getDeviceProperties(@as(u32, i));
        
        std.debug.print("Device {}:\n", .{i});
        // Extract device name from fixed-size array
        var name_end: usize = 0;
        while (name_end < props.deviceName.len and props.deviceName[name_end] != 0) {
            name_end += 1;
        }
        const device_name = props.deviceName[0..name_end];
        
        std.debug.print("  Name: {s}\n", .{device_name});
        std.debug.print("  Compute Capability: {}.{}\n", .{
            @as(u32, @intCast(props.major)), 
            @as(u32, @intCast(props.minor))
        });
        std.debug.print("  Total Memory: {} MB\n", .{
            @divTrunc(@as(usize, @intCast(props.totalGlobalMem)), 1024 * 1024)
        });
        std.debug.print("\n");
    }

    // Now you could use the library for:
    // - Allocating GPU memory with cuda.allocDeviceMemory()
    // - Copying data between host and device
    // - Loading CUDA modules and launching kernels
    // - Setting up streams for async operations
    
    std.debug.print("✅ Library ready for use!\n");
}