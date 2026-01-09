// src/core/context.zig
// CUDA context management and lifecycle
// TODO: Implement context management

const std = @import("std");
const errors = @import("../bindings/errors.zig");
const bindings = @import("../bindings/cuda.zig");
const device = @import("device.zig");

pub const Context = struct {
    handle: *bindings.CUcontext,
    memory_pool: ?*MemoryPool,
    
    pub fn init(device_index: u32) !Context {
        const flags: bindings.c_uint = 0;
        
        var ctx_handle = try bindings.createContext(flags, device_index);
        
        return Context{
            .handle = ctx_handle,
            .memory_pool = null,
        };
    }
    
    
    
    pub fn deinit(self: *Context) void {
        if (self.handle != null) {
            bindings.destroyContext(self.handle.?).catch |err| {
                std.log.err("Failed to destroy CUDA context: {}", .{err});
                // Don't return error on cleanup failure, just log it
            };
        }
    }

/// Make this context the current/active context
pub fn makeCurrent(self: *Context) !void {
    if (self.handle == null) {
        // Return an error instead of panicking for better error handling
        return error.InvalidContext;
    }
    
    try bindings.setCurrentContext(self.handle.?);
}
    
    pub fn makeCurrent(self: *Context) !void {
        // TODO: Implement context activation
        _ = self;
    }
    
    
};

pub const MemoryPool = struct {
    // TODO: Implement memory pool
    pub fn init() !MemoryPool {
        return MemoryPool{};
    }
};

// TODO: Add more context functionality