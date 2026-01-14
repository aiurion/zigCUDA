// Example: How Another Project Uses ZigCUDA
// Save this as examples/external_project_usage.zig to show the pattern

const std = @import("std");

pub fn main() !void {
    std.debug.print("\n=== External Project Using ZigCUDA ===\n\n", .{});

    // Step 1: Import the library (as external projects would)
    const zigcuda = @import("zigcuda"); 

    // Step 2: Initialize and use it
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    if (!ctx.isAvailable()) {
        std.debug.print("❗ CUDA not available\n", .{});
        return;
    }

    const device_count = ctx.getDeviceCount();
    std.debug.print("✅ External project successfully using ZigCUDA!\n");
    std.debug.print("Found {d} devices via library import\n\n", .{device_count});

    // Step 3: Use the clean API
    for (0..@min(device_count, @as(usize, 2))) |i| {
        const props = try ctx.getDeviceProperties(@as(u32, i));
        
        std.debug.print("External Project Device {}:\n", .{i});
        // Show we can access device info through the clean API
        std.debug.print("  Major: {}, Minor: {}\n", .{
            @as(u32, @intCast(props.major)),
            @as(u32, @intCast(props.minor))
        });
    }

    std.debug.print("\n🎉 External project integration complete!\n");
}