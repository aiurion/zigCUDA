// src/core/stream.zig
// Asynchronous stream operations

const std = @import("std");
const cuda = @import("../bindings/cuda.zig");

pub const Stream = struct {
    handle: *cuda.CUstream,
    
    pub fn init(flags: c_uint) errors.CUDAError!Stream {
        const stream_handle = try cuda.createStream(flags);
        return Stream{
            .handle = stream_handle,
        };
    }
    
    pub fn deinit(self: *Stream) !void {
        _ = self;
        // Note: In many cases, streams are automatically cleaned up when the context is destroyed
        // But we can explicitly destroy if needed:
        // try cuda.destroyStream(self.handle);
    }
    
    /// Wait for all operations in this stream to complete (synchronous)
    pub fn wait(self: *Stream) !void {
        return cuda.syncStream(self.handle);
    }
    
    /// Check if the stream is ready without blocking
    pub fn isDone(self: *const Stream) errors.CUDAError!bool {
        return cuda.queryStream(self.handle);
    }
    
    /// Create a default stream with no special flags
    pub fn create() errors.CUDAError!Stream {
        const default_flags: c_uint = 0;
        return init(default_flags);
    }
    
    /// Create an non-blocking stream (operations don't block the host)
    pub fn createNonBlocking() errors.CUDAError!Stream {
        // CU_STREAM_NON_BLOCKING flag is typically 1
        const non_blocking_flags: c_uint = 1; 
        return init(non_blocking_flags);
    }
    
    /// Create a high-priority stream for time-critical operations
    pub fn createHighPriority() errors.CUDAError!Stream {
        // CU_STREAM_HIGH_PRIORITY flag is typically 2  
        const priority_flags: c_uint = 2;
        return init(priority_flags);
    }
};

pub const StreamFlags = enum(c_uint) {
    default = 0,
    non_blocking = 1,      // Operations don't block the host thread
    high_priority = 2,   // High-priority stream for time-critical operations
};

// Convenience aliases for backward compatibility
pub const createDefaultStream = Stream.create;
pub const createNonBlockingStream = Stream.createNonBlocking; 
pub const createHighPriorityStream = Stream.createHighPriority;

const c_uint = @import("../bindings/cuda.zig").c_uint;
const errors = @import("../bindings/errors.zig");