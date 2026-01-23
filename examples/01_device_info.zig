// examples/01_device_info.zig
// Basic CUDA device enumeration and property querying

const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    // Initialize CUDA and get context
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    const device_count = ctx.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n\n", .{device_count});

    // Query properties for each device
    var i: u32 = 0;
    while (i < device_count) : (i += 1) {
        const props = try ctx.getDeviceProperties(i);

        // Extract null-terminated name
        const name_len = std.mem.indexOfScalar(u8, &props.name, 0) orelse props.name.len;
        const device_name = props.name[0..name_len];

        std.debug.print("Device {d}: {s}\n", .{ i, device_name });
        std.debug.print("  Compute Capability: {d}.{d}\n", .{ props.major, props.minor });
        std.debug.print("  Total Memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(props.totalGlobalMem)) / (1024.0 * 1024.0 * 1024.0)});
        std.debug.print("  Multiprocessors: {d}\n", .{props.multiProcessorCount});
        std.debug.print("\n", .{});
    }
}
