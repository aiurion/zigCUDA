// examples/02_memory_transfer.zig
// Demonstrates host-to-device and device-to-host memory transfers

const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    // Initialize CUDA
    try zigcuda.loadCuda();
    try zigcuda.initCuda(0);

    std.debug.print("=== CUDA Memory Transfer Example ===\n\n", .{});

    // Allocate host memory and initialize with test data
    const array_size = 1024;
    const byte_size = array_size * @sizeOf(f32);

    const host_input = try std.heap.page_allocator.alloc(f32, array_size);
    defer std.heap.page_allocator.free(host_input);

    const host_output = try std.heap.page_allocator.alloc(f32, array_size);
    defer std.heap.page_allocator.free(host_output);

    // Initialize input data
    for (host_input, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }

    std.debug.print("Allocated {d} bytes on host\n", .{byte_size});
    std.debug.print("Input data: [{d:.1}, {d:.1}, {d:.1}, ..., {d:.1}]\n", .{
        host_input[0],
        host_input[1],
        host_input[2],
        host_input[array_size - 1],
    });

    // Allocate device memory
    const device_ptr = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(device_ptr) catch {};

    std.debug.print("Allocated {d} bytes on device\n\n", .{byte_size});

    // Transfer data from host to device
    std.debug.print("Copying data from host to device...\n", .{});
    try zigcuda.copyHostToDevice(device_ptr, host_input.ptr, byte_size);
    std.debug.print("✓ Host-to-device transfer complete\n\n", .{});

    // Transfer data back from device to host
    std.debug.print("Copying data from device to host...\n", .{});
    try zigcuda.copyDeviceToHost(host_output.ptr, device_ptr, byte_size);
    std.debug.print("✓ Device-to-host transfer complete\n\n", .{});

    // Verify data integrity
    var errors: usize = 0;
    for (host_input, host_output) |input, output| {
        if (input != output) {
            errors += 1;
        }
    }

    if (errors == 0) {
        std.debug.print("✓ SUCCESS: All {d} elements transferred correctly\n", .{array_size});
        std.debug.print("Output data: [{d:.1}, {d:.1}, {d:.1}, ..., {d:.1}]\n", .{
            host_output[0],
            host_output[1],
            host_output[2],
            host_output[array_size - 1],
        });
    } else {
        std.debug.print("✗ FAILED: {d} elements had errors\n", .{errors});
        return error.DataMismatch;
    }
}
