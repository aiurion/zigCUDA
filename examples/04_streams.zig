// examples/04_streams.zig
// Demonstrates asynchronous operations using CUDA streams and core abstractions

const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    // Initialize CUDA
    try zigcuda.init();

    std.debug.print("=== CUDA Streams Example ===\n\n", .{});

    // Initialize device and context
    var ctx = try zigcuda.Context.init(0);
    defer ctx.deinit();
    try ctx.makeCurrent();

    // Create multiple streams for concurrent operations
    const num_streams: usize = 3;
    var streams: [num_streams]zigcuda.Stream = undefined;

    // Create streams using core Stream abstraction
    std.debug.print("Creating {d} CUDA streams...\n", .{num_streams});
    for (&streams) |*stream| {
        stream.* = try zigcuda.Stream.createDefault();
    }

    defer {
        for (&streams) |*stream| {
            stream.destroy();
        }
    }

    std.debug.print("✓ {d} streams created\n\n", .{num_streams});

    // Allocate memory for each stream
    const array_size = 512;
    const byte_size = array_size * @sizeOf(f32);

    var host_buffers: [num_streams][]f32 = undefined;
    var device_buffers: [num_streams]zigcuda.CUdeviceptr = undefined;

    const allocator = std.heap.page_allocator;

    // Allocate host and device memory for each stream
    for (&host_buffers, 0..) |*buffer, i| {
        buffer.* = try allocator.alloc(f32, array_size);
        // Initialize with stream-specific data
        for (buffer.*, 0..) |*val, j| {
            val.* = @as(f32, @floatFromInt(i * 1000 + j));
        }
    }

    defer {
        for (host_buffers) |buffer| {
            allocator.free(buffer);
        }
    }

    for (&device_buffers) |*d_ptr| {
        d_ptr.* = try zigcuda.allocDeviceMemory(byte_size);
    }

    defer {
        for (device_buffers) |d_ptr| {
            _ = zigcuda.freeDeviceMemory(d_ptr) catch {};
        }
    }

    std.debug.print("Memory allocated for {d} streams\n", .{num_streams});
    std.debug.print("Each stream handles {d} elements ({d} bytes)\n\n", .{ array_size, byte_size });

    // Launch async operations on each stream
    std.debug.print("Launching async memory copies on all streams...\n", .{});

    const start_time = std.time.milliTimestamp();

    for (host_buffers, device_buffers, streams) |h_buffer, d_buffer, stream| {
        try zigcuda.bindings.copyHostToDeviceAsync(d_buffer, std.mem.sliceAsBytes(h_buffer), stream.handle);
    }

    // Synchronize all streams
    std.debug.print("Synchronizing all streams...\n", .{});
    for (&streams) |*stream| {
        // Wait for this stream to complete
        try stream.synchronize();
    }

    const end_time = std.time.milliTimestamp();
    const elapsed_ms = end_time - start_time;

    std.debug.print("✓ All streams synchronized\n", .{});
    std.debug.print("Time elapsed: {d} ms\n\n", .{elapsed_ms});

    // Query stream status
    std.debug.print("Querying stream status...\n", .{});
    for (&streams, 0..) |*stream, i| {
        if (try stream.query()) {
            std.debug.print("  Stream {d}: ✓ Complete\n", .{i});
        } else {
            std.debug.print("  Stream {d}: ⏳ Pending\n", .{i});
        }
    }

    std.debug.print("\n✓ SUCCESS: Streams example completed\n", .{});
    std.debug.print("Total data transferred: {d} KB across {d} streams\n", .{
        (byte_size * num_streams) / 1024,
        num_streams,
    });
}
