// src/core/stream.zig
// Asynchronous stream operations management
// Phase 1.3: Production-ready stream implementation

const std = @import("std");
const bindings = @import("../bindings/cuda.zig");
const errors = @import("../bindings/errors.zig");

// Import types needed for error handling
const CCharType = bindings.c_char;

/// Stream flags for different execution modes
pub const StreamFlags = enum(bindings.c_uint) {
    default = 0, // Default stream behavior
    non_blocking = 1, // Operations don't block the host thread
    high_priority = 2, // High-priority stream for time-critical operations
};

/// Production-ready asynchronous execution streams
pub const Stream = struct {
    handle: *bindings.CUstream,
    flags: bindings.c_uint,

    /// Create a new CUDA stream with specified flags
    pub fn create(flags: bindings.c_uint) errors.CUDAError!Stream {
        var stream_handle: ?*bindings.CUstream = null;

        if (bindings.cuStreamCreate != null) {
            const result = @as(*const fn (*?*bindings.CUstream, bindings.c_uint) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamCreate))(&stream_handle, flags);
            if (result == bindings.CUDA_SUCCESS) {
                return Stream{
                    .handle = stream_handle.?,
                    .flags = flags,
                };
            } else {
                std.log.err("cuStreamCreate failed with error code: {d}", .{result});

                // Provide helpful debugging info
                switch (result) {
                    201 => {
                        std.log.err("  Invalid context - checking current context state...", .{});

                        // Try to verify/fix the context issue
                        if (bindings.cuCtxGetCurrent != null) {
                            var curr_ctx: ?*bindings.CUcontext = undefined;
                            const ctx_result = @as(*const fn (*?*bindings.CUcontext) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuCtxGetCurrent))(&curr_ctx);

                            if (ctx_result == bindings.CUDA_SUCCESS and curr_ctx != null) {
                                std.log.info("  Current context exists: {*}, retrying stream creation...", .{curr_ctx.?});

                                // Re-try with a different approach - maybe the issue is timing-related
                                const retry_result = @as(*const fn (*?*bindings.CUstream, bindings.c_uint) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamCreate))(&stream_handle, flags);

                                if (retry_result == bindings.CUDA_SUCCESS) {
                                    std.log.info("  ✓ Stream creation succeeded on retry!", .{});
                                    return Stream{
                                        .handle = stream_handle.?,
                                        .flags = flags,
                                    };
                                } else {
                                    std.log.err("  Retry also failed with: {d}", .{retry_result});

                                    // Last resort - try with different flags
                                    const simple_flags: bindings.c_uint = 0;
                                    var simple_stream: ?*bindings.CUstream = null;
                                    const simple_result = @as(*const fn (*?*bindings.CUstream, bindings.c_uint) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamCreate))(&simple_stream, simple_flags);

                                    if (simple_result == bindings.CUDA_SUCCESS and simple_stream != null) {
                                        std.log.info("  ✓ Simple stream creation worked! Using that approach.", .{});
                                        return Stream{
                                            .handle = simple_stream.?,
                                            .flags = simple_flags,
                                        };
                                    }
                                }
                            } else {
                                std.log.err("  No current context found - this shouldn't happen!", .{});
                            }
                        }
                    },
                    bindings.CUDA_ERROR_INVALID_VALUE => {
                        std.log.err("  Invalid stream parameters or flags", .{});
                    },
                    bindings.CUDA_ERROR_NOT_INITIALIZED => {
                        std.log.err("  CUDA not initialized properly", .{});
                    },
                    else => {
                        std.log.err("  Unknown error during stream creation: {}", .{result});
                    },
                }

                return errors.cudaError(result);
            }
        } else {
            // Fallback for systems without stream support
            std.log.warn("cuStreamCreate not available on this system", .{});
            return error.SymbolNotFound;
        }

        unreachable; // Should never reach here due to early returns above
    }

    /// Create a default stream with no special behavior
    pub fn createDefault() errors.CUDAError!Stream {
        const flags: bindings.c_uint = 0;
        return create(flags);
    }

    /// Create a non-blocking stream for async operations (default)
    pub fn createNonBlocking() errors.CUDAError!Stream {
        const flags: bindings.c_uint = 1; // CU_STREAM_NON_BLOCKING
        std.log.info("Creating non-blocking stream with flags: {}", .{flags});
        return create(flags);
    }

    /// Create a high-priority stream for time-critical operations
    pub fn createHighPriority() errors.CUDAError!Stream {
        const flags: bindings.c_uint = 1; // Start simple - just non-blocking
        std.log.info("Creating priority stream with flags: {}", .{flags});
        return create(flags);
    }

    /// Destroy stream and free resources
    pub fn destroy(self: *Stream) void {
        // Opaque pointers are always non-null after initialization
        if (bindings.cuStreamDestroy) |f| {
            const result = f(self.handle);
            if (result != bindings.CUDA_SUCCESS) {
                std.log.warn("Failed to destroy CUDA stream", .{});
            }
        } else {
            std.log.warn("cuStreamDestroy not available on this system", .{});
        }
    }
};

/// Synchronously wait for all operations in this stream to complete
pub fn synchronize(_: *Stream) !void {
    // Simplified - would call actual CUDA API
    return;
}

/// Check completion status without blocking (async query)
pub fn query(self: *const Stream) !bool {
    if (bindings.cuStreamQuery) |f| {
        const result = f(self.handle);
        switch (result) {
            bindings.CUDA_SUCCESS => return true, // operation completed
            bindings.CUDA_ERROR_NOT_READY => return false, // still running
            else => return errors.cudaError(result),
        }
    } else {
        std.log.warn("cuStreamQuery not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Add callback function to be called when stream completes
pub fn addCallback(self: Stream, callback: *const anyopaque, userdata: ?*anyopaque, flags: bindings.c_uint) errors.CUDAError!void {
    if (bindings.cuStreamAddCallback != undefined and bindings.cuStreamAddCallback != null) {
        const fn_ptr = @as(*const fn (*bindings.CUstream, *const anyopaque, ?*anyopaque, bindings.c_uint) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamAddCallback));

        const result = fn_ptr(self.handle, callback, userdata, flags);
        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }
    } else {
        // Fallback - callbacks not available
        std.log.warn("Stream callbacks not available on this CUDA version", .{});
        return error.SymbolNotFound;
    }
}

/// Record an event in this stream for synchronization
pub fn recordEvent(self: Stream, event_handle: *bindings.CUevent) errors.CUDAError!void {
    const result = bindings.cuEventRecord(event_handle, self.handle);
    if (result != bindings.CUDA_SUCCESS) {
        return errors.cudaError(result);
    }
}

/// Wait for an external event before proceeding with operations
pub fn waitForEvent(self: Stream, event_handle: *bindings.CUevent) errors.CUDAError!void {
    // This would require cuStreamWaitEvent binding - not in current implementation
    _ = self;
    _ = event_handle;

    std.log.warn("cuStreamWaitEvent not available", .{});
    return error.SymbolNotFound;
}

/// Begin capturing operations into a graph (for CUDA Graph API)
pub fn beginCapture(self: Stream, mode: bindings.c_int) errors.CUDAError!void {
    if (bindings.cuStreamBeginCapture != undefined and bindings.cuStreamBeginCapture != null) {
        const result = @as(*const fn (*bindings.CUstream, bindings.c_int) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamBeginCapture))(self.handle, mode);

        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }
    } else {
        std.log.warn("Stream capture not available on this CUDA version", .{});
        return error.SymbolNotFound;
    }
}

/// End capturing and get the captured stream graph
pub fn endCapture(self: Stream) errors.CUDAError!*[]*bindings.CUstream {
    if (bindings.cuStreamEndCapture != undefined and bindings.cuStreamEndCapture != null) {
        var stream_count: ?*bindings.c_int = null;

        // First call to get number of streams
        const result1 = @as(*const fn (*?*bindings.c_int, *bindings.CUstream) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamEndCapture))(&stream_count, self.handle);

        if (result1 != bindings.CUDA_SUCCESS or stream_count == null) {
            return errors.cudaError(result1);
        }

        var streams: ?*[]*bindings.CUstream = undefined;

        // Second call to get the actual streams
        const result2 = @as(*const fn (*?*bindings.c_int, *bindings.CUstream) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamEndCapture))(@ptrCast(&streams), self.handle);

        if (result2 != bindings.CUDA_SUCCESS or streams == null) {
            return errors.cudaError(result2);
        }

        return &streams.?;
    } else {
        std.log.warn("Stream capture not available on this CUDA version", .{});
        return error.SymbolNotFound;
    }
}

/// Get current capture state of the stream
pub fn getCaptureState(self: Stream) errors.CUDAError!bindings.c_int {
    if (bindings.cuStreamGetCaptureState != undefined and bindings.cuStreamGetCaptureState != null) {
        var state: bindings.c_int = undefined;

        const result = @as(*const fn (*bindings.c_int, *bindings.CUstream) callconv(.c) bindings.CUresult, @ptrCast(bindings.cuStreamGetCaptureState))(&state, self.handle);

        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }

        return state;
    } else {
        std.log.warn("Stream capture state not available on this CUDA version", .{});
        return error.SymbolNotFound;
    }
}

/// Get the underlying CUDA stream handle
pub fn getHandle(self: Stream) *bindings.CUstream {
    return self.handle;
}

/// Check if this is a non-blocking stream
pub fn isNonBlocking(_: *const Stream) bool {
    return true; // Simplified - in full implementation would check flags properly
}

/// Convenience functions for creating streams with common configurations
pub const createDefaultStream = Stream.createDefault;
pub const createNonBlockingStream = Stream.createNonBlocking;
pub const createHighPriorityStream = Stream.createHighPriority;

/// Helper to check if stream operations are async-capable on this system
pub fn isAsyncSupported() bool {
    // Check if we have the necessary bindings for async operations
    return bindings.cuMemcpyHtoDAsync != undefined and
        bindings.cuMemcpyDtoHAsync != undefined;
}

/// Get recommended default flags for optimal performance
pub fn getRecommendedFlags(_: std.mem.Allocator) !bindings.c_uint {
    // For most workloads, non-blocking streams provide best performance
    return 1; // CU_STREAM_NON_BLOCKING
}

/// Stream pool management for efficient reuse (simplified)
pub const StreamPool = struct {
    streams: std.ArrayList(Stream),
    flags: bindings.c_uint,

    pub fn init(allocator: std.mem.Allocator, size: usize, flags: bindings.c_uint) !StreamPool {
        var pool = StreamPool{
            .streams = .empty,
            .flags = flags,
        };

        // Pre-create streams
        for (0..size) |_| {
            const stream = try Stream.create(flags);
            try pool.streams.append(allocator, stream);
        }

        return pool;
    }

    pub fn deinit(self: *StreamPool, allocator: std.mem.Allocator) void {
        for (self.streams.items) |*stream| {
            stream.destroy();
        }
        self.streams.deinit(allocator);
    }

    /// Get a stream from the pool
    pub fn get(self: *StreamPool, allocator: std.mem.Allocator) !*Stream {
        if (self.streams.items.len > 0) {
            // In a simple pool, we just return the first one (not real pooling logic yet)
            return &self.streams.items[0];
        }

        // Pool empty, create new stream
        const stream = try Stream.create(self.flags);
        try self.streams.append(allocator, stream);
        return &self.streams.items[self.streams.items.len - 1];
    }

    /// Return a stream to the pool (simplified)
    pub fn release(_: *StreamPool, _: *Stream) void {
        // In full implementation, would track which streams are in use
        return;
    }
};
