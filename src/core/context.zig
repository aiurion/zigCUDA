// src/core/context.zig
// CUDA context management and lifecycle

const std = @import("std");
const errors = @import("../bindings/errors.zig");
const bindings = @import("../bindings/cuda.zig");

pub const Context = struct {
    handle: ?*bindings.CUcontext,
    
    /// Initialize a new CUDA context for the specified device
    pub fn init(device_index: u32) !Context {
        // Create context with default flags
        const flags: bindings.c_uint = 0;
        
        var ctx_handle = try bindings.createContext(flags, device_index);
        
        return Context{
            .handle = ctx_handle,
        };
    }
    
    /// Clean up the CUDA context
    pub fn deinit(self: *Context) void {
        if (self.handle != null) {
            // Destroy the context and ignore errors during cleanup
            bindings.destroyContext(self.handle.?).catch |err| {
                std.log.warn("Failed to destroy CUDA context during cleanup: {}", .{err});
            };
            self.handle = null;
        }
    }

    /// Make this context the current/active context
    pub fn makeCurrent(self: *const Context) !void {
        if (self.handle == null) {
            return errors.CUDAError.InvalidContext;
        }
        
        try bindings.setCurrentContext(self.handle.?);
    }
    
    /// Check if this context is valid
    pub fn isValid(self: *const Context) bool {
        return self.handle != null;
    }
};