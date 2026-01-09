// src/core/event.zig
// Synchronization events - Phase 0 Implementation  
// Event Management API bindings implementation

const std = @import("std");
const bindings = @import("../bindings/cuda.zig");
const errors = @import("../bindings/errors.zig");

pub const Event = struct {
    handle: *bindings.CUevent,
    
    /// Initialize a new CUDA event with default settings
    pub fn init() !Event {
        // Use the new binding: cuEventCreate
        const event_handle = try bindings.createDefaultTimingEvent();
        return Event{
            .handle = event_handle,
        };
    }
    
    /// Initialize a blocking synchronization event
    pub fn initBlocking() !Event {
        // Use the new binding: cuEventCreate with blocking flag
        const event_handle = try bindings.createBlockingEvent();
        return Event{
            .handle = event_handle,
        };
    }
    
    /// Clean up and destroy the CUDA event
    pub fn deinit(self: *Event) void {
        if (self.handle != null) {
            // Use the new binding: cuEventDestroy
            bindings.destroyEvent(self.handle).catch |err| {
                @panic("Failed to destroy event");
            };
        }
    }
    
    /// Record this event in a stream for synchronization
    pub fn record(self: *Event, stream: ?*bindings.CUstream) !void {
        // Use the new binding: cuEventRecord
        try bindings.recordEvent(self.handle, stream);
    }
    
    /// Wait synchronously for this event to complete
    pub fn wait(self: *Event) !void {
        // Use the new binding: cuEventSynchronize
        try bindings.syncEvent(self.handle);
    }
    
    /// Record in default stream (null pointer)
    pub fn recordInDefaultStream(self: *Event) !void {
        return self.record(null);
    }
    
    /// Check if event is done using stream query
    pub fn isDone(self: *const Event, stream: ?*bindings.CUstream) bool {
        if (stream == null) {
            // For default stream, we can't easily check without waiting
            // This would require additional API bindings for cuEventQuery
            return false;
        }
        
        // Note: Would need cuEventQuery binding for proper async checking
        // For now, assume it's done after being recorded and synced once
        _ = self;
        return true; // Simplified for Phase 0 implementation
    }
    
    /// Get the underlying CUDA event handle
    pub fn getHandle(self: *const Event) *bindings.CUevent {
        return self.handle;
    }
};

/// Convenience function to create a default timing event
pub fn createDefaultEvent() !*Event {
    var event = try std.heap.c_allocator.create(Event);
    event.* = try Event.init();
    return event;
}

/// Convenience function to create a blocking synchronization event  
pub fn createBlockingEvent() !*Event {
    var event = try std.heap.c_allocator.create(Event);
    event.* = try Event.initBlocking();
    return event;
}

const Stream = @import("stream.zig").Stream;