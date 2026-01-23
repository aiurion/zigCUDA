// src/main.zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    std.debug.print("\n=== ZigCUDA CLI Diagnostic Tool ===\n", .{});

    zigcuda.init() catch |err| {
        switch (err) {
            error.CudaLibraryNotFound => {
                std.debug.print("✗  Error: Could not load libcuda.so. Is the driver installed?\n", .{});
                return;
            },
            else => {
                std.debug.print("✗  Fatal Error during init: {}\n", .{err});
                return;
            },
        }
    };

    const count = zigcuda.Device.count() catch |err| {
        std.debug.print("✗  Error getting device count: {}\n", .{err});
        return;
    };

    if (count == 0) {
        std.debug.print("⚠  Success: Library loaded, but no CUDA devices found.\n", .{});
        return;
    }

    std.debug.print("✓  CUDA Driver Initialized\n", .{});
    std.debug.print("✓  Device Count: {d}\n\n", .{count});

    for (0..count) |i| {
        const device = zigcuda.Device.init(@intCast(i)) catch |err| {
            std.debug.print("   [GPU {d}] (Failed to initialize: {})\n", .{ i, err });
            continue;
        };
        const props = device.getProperties();

        // Name is now a standard Zig array, slice it to the first null byte
        const name_slice = std.mem.sliceTo(&props.name, 0);

        std.debug.print("   [GPU {d}] {s}\n", .{ i, name_slice });
        std.debug.print("     ├─ Compute: {d}.{d}\n", .{ props.compute_capability.major, props.compute_capability.minor });
        std.debug.print("     ├─ SMs:     {d}\n", .{props.multiprocessor_count});

        const mem_gb = @as(f64, @floatFromInt(props.total_memory)) / (1024.0 * 1024.0 * 1024.0);
        std.debug.print("     └─ VRAM:    {d:.2} GB\n", .{mem_gb});
        std.debug.print("\n", .{});
    }
}
