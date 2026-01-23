// examples/01_device_info.zig
// Basic CUDA device enumeration and property querying using high-level core API

const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    // Initialize CUDA driver
    try zigcuda.init();

    const device_count = try zigcuda.Device.count();
    std.debug.print("Found {d} CUDA device(s)\n\n", .{device_count});

    // Query properties for each device
    for (0..device_count) |i| {
        const device = try zigcuda.Device.init(@intCast(i));
        const props = device.getProperties();

        // Extract null-terminated name
        const device_name = std.mem.sliceTo(&props.name, 0);

        std.debug.print("Device {d}: {s}\n", .{ i, device_name });
        std.debug.print("  Compute Capability: {d}.{d}\n", .{ props.compute_capability.major, props.compute_capability.minor });
        std.debug.print("  Total Memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(props.total_memory)) / (1024.0 * 1024.0 * 1024.0)});
        std.debug.print("  Multiprocessors: {d}\n", .{props.multiprocessor_count});
        std.debug.print("\n", .{});
    }
}
