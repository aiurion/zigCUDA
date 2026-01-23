// examples/04_streams.zig
// Demonstrates asynchronous operations using CUDA streams

const std = @import("std");
const zigcuda = @import("zigcuda");
const cuda = zigcuda.bindings;

pub fn main() !void {
    // Initialize CUDA
    try cuda.load();
    try cuda.init(0);

    std.debug.print("=== CUDA Streams Example ===\n\n", .{});

    // Get device and create context
    const device = try cuda.getDevice(0);
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuCtxCreate) |f| {
        const result = f(&ctx, 0, device);
        if (result != 0) return error.ContextCreationFailed;
    } else return error.CudaFunctionNotAvailable;

    defer {
        if (ctx) |c| {
            if (cuda.cuCtxDestroy) |f| _ = f(c);
        }
    }

    // Create multiple streams for concurrent operations
    const num_streams: usize = 3;
    var streams: [num_streams]?*cuda.CUstream = undefined;

    // Create streams
    std.debug.print("Creating {d} CUDA streams...\n", .{num_streams});
    if (cuda.cuStreamCreate) |f| {
        for (&streams) |*stream| {
            const result = f(stream, 0);
            if (result != 0) return error.StreamCreationFailed;
        }
    } else return error.CudaFunctionNotAvailable;

    defer {
        if (cuda.cuStreamDestroy) |f| {
            for (streams) |stream| {
                if (stream) |s| _ = f(s);
            }
        }
    }

    std.debug.print("✓ {d} streams created\n\n", .{num_streams});

    // Allocate memory for each stream
    const array_size = 512;
    const byte_size = array_size * @sizeOf(f32);

    var host_buffers: [num_streams][]f32 = undefined;
    var device_buffers: [num_streams]cuda.CUdeviceptr = undefined;

    // Allocate host and device memory for each stream
    for (&host_buffers, 0..) |*buffer, i| {
        buffer.* = try std.heap.page_allocator.alloc(f32, array_size);
        // Initialize with stream-specific data
        for (buffer.*, 0..) |*val, j| {
            val.* = @as(f32, @floatFromInt(i * 1000 + j));
        }
    }

    defer {
        for (host_buffers) |buffer| {
            std.heap.page_allocator.free(buffer);
        }
    }

    for (&device_buffers) |*d_ptr| {
        d_ptr.* = try cuda.allocDeviceMemory(byte_size);
    }

    defer {
        for (device_buffers) |d_ptr| {
            _ = cuda.freeDeviceMemory(d_ptr) catch {};
        }
    }

    std.debug.print("Memory allocated for {d} streams\n", .{num_streams});
    std.debug.print("Each stream handles {d} elements ({d} bytes)\n\n", .{ array_size, byte_size });

    // Launch async operations on each stream
    std.debug.print("Launching async memory copies on all streams...\n", .{});

    const start_time = std.time.milliTimestamp();

    if (cuda.cuMemcpyHtoDAsync) |async_copy| {
        for (host_buffers, device_buffers, streams) |h_buffer, d_buffer, stream| {
            const result = async_copy(d_buffer, h_buffer.ptr, byte_size, stream);
            if (result != 0) return error.AsyncCopyFailed;
        }
    } else {
        // Fallback to synchronous copy if async not available
        std.debug.print("Warning: Async copy not available, using sync copy\n", .{});
        for (host_buffers, device_buffers) |h_buffer, d_buffer| {
            try cuda.copyHostToDevice(d_buffer, h_buffer.ptr, byte_size);
        }
    }

    // Synchronize all streams
    std.debug.print("Synchronizing all streams...\n", .{});
    if (cuda.cuStreamSynchronize) |sync_fn| {
        for (streams) |stream| {
            if (stream) |s| {
                const result = sync_fn(s);
                if (result != 0) return error.StreamSyncFailed;
            }
        }
    }

    const end_time = std.time.milliTimestamp();
    const elapsed_ms = end_time - start_time;

    std.debug.print("✓ All streams synchronized\n", .{});
    std.debug.print("Time elapsed: {d} ms\n\n", .{elapsed_ms});

    // Query stream status
    std.debug.print("Querying stream status...\n", .{});
    if (cuda.cuStreamQuery) |query_fn| {
        for (streams, 0..) |stream, i| {
            if (stream) |s| {
                const result = query_fn(s);
                if (result == 0) {
                    std.debug.print("  Stream {d}: ✓ Complete\n", .{i});
                } else {
                    std.debug.print("  Stream {d}: ⏳ Pending\n", .{i});
                }
            }
        }
    }

    std.debug.print("\n✓ SUCCESS: Streams example completed\n", .{});
    std.debug.print("Total data transferred: {d} KB across {d} streams\n", .{
        (byte_size * num_streams) / 1024,
        num_streams,
    });
}
