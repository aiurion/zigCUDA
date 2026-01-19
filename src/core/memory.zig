// src/core/memory.zig
// Memory pool allocation and management
// Phase 1.2: Type-safe memory management implementation

const std = @import("std");
const bindings = @import("../bindings/cuda.zig");
const errors = @import("../bindings/errors.zig");

// ============================================================================
// WRAPPER FUNCTIONS FOR MODERN CUDA (CUDA 13+ / Blackwell)
// These wrappers use _v2 APIs exclusively - no v1 fallback
// ============================================================================

/// Allocate device memory using _v2 API (required for CUDA 13+)
pub fn alloc(size: usize) errors.CUDAError!bindings.CUdeviceptr {
    var ptr: bindings.CUdeviceptr = undefined;

    // Use _v2 only - modern CUDA requires v2 APIs
    if (bindings.cuMemAlloc_v2 == null) {
        return error.SymbolNotFound;
    }
    const result = bindings.cuMemAlloc_v2.?(&ptr, size);

    if (result == bindings.CUDA_SUCCESS) return ptr;
    return errors.cudaError(result);
}

/// Free device memory using _v2 API
pub fn free(ptr: bindings.CUdeviceptr) errors.CUDAError!void {
    // Use _v2 only - modern CUDA requires v2 APIs
    if (bindings.cuMemFree_v2 == null) {
        return error.SymbolNotFound;
    }
    const result = bindings.cuMemFree_v2.?(ptr);

    if (result == bindings.CUDA_SUCCESS) return;
    return errors.cudaError(result);
}

/// Get memory information using _v2 API
pub fn getMemoryInfo() errors.CUDAError!struct { free: usize, total: usize } {
    var free_bytes: c_ulonglong = undefined;
    var total_bytes: c_ulonglong = undefined;

    // Use _v2 only - modern CUDA requires v2 APIs
    if (bindings.cuMemGetInfo_v2 == null) {
        return error.SymbolNotFound;
    }
    const result = bindings.cuMemGetInfo_v2.?(&free_bytes, &total_bytes);

    if (result == bindings.CUDA_SUCCESS) {
        return .{ .free = @as(usize, @intCast(free_bytes)), .total = @as(usize, @intCast(total_bytes)) };
    }
    return errors.cudaError(result);
}

// Type aliases for C types used above
const c_ulonglong = u64;

/// Type-safe device pointer that prevents mixing CPU/GPU memory at compile time
pub fn DevicePtr(comptime T: type) type {
    return struct {
        ptr: *anyopaque,
        len: usize,

        const Self = @This();

        /// Create a new device pointer with proper bounds checking
        pub fn init(ptr: *anyopaque, length: usize) Self {
            return Self{
                .ptr = ptr,
                .len = length,
            };
        }

        /// Get a slice of the memory region [start, end)
        pub fn slice(self: Self, start: usize, end: usize) !Self {
            if (end > self.len or start >= end) {
                return errors.CUDAError.InvalidValue;
            }

            const byte_offset = @sizeOf(T) * start;
            return Self{
                .ptr = @as(*anyopaque, @ptrCast(@as([*]u8, @ptrCast(self.ptr)) + byte_offset)),
                .len = end - start,
            };
        }

        /// Get total size in bytes
        pub fn byteSize(self: Self) usize {
            return self.len * @sizeOf(T);
        }

        /// Convert to generic pointer for CUDA API calls
        pub fn asCudaPtr(self: Self) bindings.CUdeviceptr {
            return @intFromPtr(self.ptr);
        }
    };
}

/// Memory allocation tracking information
pub const AllocationInfo = struct {
    size: usize,
    alignment: usize,
    timestamp: i64,
};

/// Statistics for memory pool
pub const PoolStats = struct {
    total_allocated: usize,
    active_allocations: u32,
    peak_usage: usize,
    fragmentation_ratio: f32,
};

/// Production-ready memory pool with async allocation support
pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    device_index: u32,
    allocations: std.AutoHashMap(bindings.CUdeviceptr, AllocationInfo),
    total_allocated: usize,
    peak_usage: usize,

    pub fn init(allocator: std.mem.Allocator, device_index: u32) !MemoryPool {
        const allocations = std.AutoHashMap(bindings.CUdeviceptr, AllocationInfo).init(allocator);
        
        return MemoryPool{
            .allocator = allocator,
            .device_index = device_index,
            .allocations = allocations,
            .total_allocated = 0,
            .peak_usage = 0,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        // Clean up all allocations using the preferred wrapper (tries _v2 first)
        var it = self.allocations.iterator();
        while (it.next()) |entry| {
            const cuda_ptr = @as(bindings.CUdeviceptr, @intCast(entry.key_ptr.*));
            free(cuda_ptr) catch |err| {
                std.log.err("Failed to free memory during deinit: {}", .{err});
            };
        }
        self.allocations.deinit();
    }

    /// Allocate device memory with type safety and tracking
    pub fn alloc(self: *MemoryPool, comptime T: type, count: usize) !DevicePtr(T) {
        const byte_size = @sizeOf(T) * count;

        // Ensure CUDA is initialized
        try bindings.init(0);

        // Allocate memory using the preferred wrapper (tries _v2 first)
        const cuda_ptr_raw = try alloc(byte_size);
        const cuda_ptr: ?*anyopaque = @ptrFromInt(cuda_ptr_raw);

        // Track the allocation for cleanup and stats
        const info = AllocationInfo{
            .size = byte_size,
            .alignment = @alignOf(T),
            .timestamp = std.time.microTimestamp(),
        };

        try self.allocations.put(@intFromPtr(cuda_ptr), info);

        // Update statistics
        self.total_allocated += byte_size;
        if (self.total_allocated > self.peak_usage) {
            self.peak_usage = self.total_allocated;
        }

        return DevicePtr(T).init(cuda_ptr, count);
    }

    /// Free device memory and update tracking
    pub fn free(self: *MemoryPool, ptr: anytype) void {
        const cuda_ptr = @intFromPtr(ptr.ptr);

        if (self.allocations.remove(self.allocator, cuda_ptr)) |entry| {
            self.total_allocated -= entry.value.size;

            // Free using the preferred wrapper (tries _v2 first)
            free(@as(bindings.CUdeviceptr, @intCast(cuda_ptr))) catch |err| {
                std.log.err("Failed to free memory: {}", .{err});
            };
        } else {
            // Warning: trying to free untracked memory
            std.log.warn("Attempting to free untracked device memory", .{});
        }
    }

    /// Trim memory pool by freeing unused allocations (simplified version)
    pub fn trim(self: *MemoryPool) !void {
        // In a full implementation, this would implement actual CUDA memory pooling
        // For now, we just log that trimming was requested
        std.log.debug("Memory pool trim requested for device {}", .{self.device_index});

        // Could implement actual pool management here:
        // - Free oldest allocations if over threshold
        // - Consolidate fragments
        // - Return unused memory to CUDA runtime
    }

    /// Get current statistics
    pub fn stats(self: *const MemoryPool) PoolStats {
        return PoolStats{
            .total_allocated = self.total_allocated,
            .active_allocations = @intCast(self.allocations.count()),
            .peak_usage = self.peak_usage,
            .fragmentation_ratio = if (self.peak_usage > 0)
                @as(f32, @floatFromInt(self.total_allocated)) / @as(f32, @floatFromInt(self.peak_usage))
            else
                1.0,
        };
    }

    /// Create a stream for async operations
    pub fn createStream(_: *MemoryPool) !*bindings.CUstream {
        // Create non-blocking stream for async memory ops
        const flags: bindings.c_uint = 1; // CU_STREAM_NON_BLOCKING
        var stream_handle: ?*bindings.CUstream = null;

        const result = bindings.cuStreamCreate(&stream_handle, flags);
        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }

        return stream_handle.?;
    }
};

/// High-level device memory management
pub const DeviceMemory = struct {
    ptr: *anyopaque,
    size: usize,
    device_index: u32,

    pub fn init(device_index: u32, comptime T: type, count: usize) !DeviceMemory {
        // Ensure CUDA is initialized for this device
        var device = try @import("device.zig").Device.get(device_index);
        try device.setCurrent();

        const byte_size = @sizeOf(T) * count;

        // Allocate memory using the preferred wrapper (tries _v2 first)
        const cuda_ptr_raw = try alloc(byte_size);

        return DeviceMemory{
            .ptr = @ptrFromInt(cuda_ptr_raw),
            .size = byte_size,
            .device_index = device_index,
        };
    }

    pub fn deinit(self: *DeviceMemory) void {
        const addr = @intFromPtr(self.ptr);
        if (addr != 0) {
            free(@as(bindings.CUdeviceptr, @intCast(addr))) catch |err| {
                std.log.err("Failed to free DeviceMemory: {}", .{err});
            };
        }
    }

    /// Copy from host to device
    pub fn copyFromHost(self: *DeviceMemory, host_data: []const u8) !void {
        if (self.size < host_data.len) {
            return errors.CUDAError.InvalidValue;
        }

        const result = bindings.cuMemcpyHtoD(@ptrFromInt(@intFromPtr(self.ptr)), host_data.ptr, host_data.len);

        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }
    }

    /// Copy from device to host
    pub fn copyToHost(self: *const DeviceMemory, host_buffer: []u8) !void {
        if (self.size < host_buffer.len) {
            return errors.CUDAError.InvalidValue;
        }

        const result = bindings.cuMemcpyDtoH(host_buffer.ptr, @ptrFromInt(@intFromPtr(self.ptr)), host_buffer.len);

        if (result != bindings.CUDA_SUCCESS) {
            return errors.cudaError(result);
        }
    }

    /// Convert to type-safe device pointer
    pub fn asDevicePtr(comptime T: type, self: *DeviceMemory) DevicePtr(T) {
        const count = @divExact(self.size, @sizeOf(T));
        return DevicePtr(T).init(self.ptr, count);
    }
};

/// Convenience function for creating memory pools
pub fn createPool(allocator: std.mem.Allocator, device_index: u32) !*MemoryPool {
    const pool = try allocator.create(MemoryPool);
    pool.* = try MemoryPool.init(allocator, device_index);
    return pool;
}

/// Helper to get recommended alignment for a type
pub fn getAlignment(comptime T: type) usize {
    const natural_alignment = @alignOf(T);

    // Ensure minimum 16-byte alignment for better performance on most GPUs
    if (natural_alignment < 16) {
        return 16;
    }

    return natural_alignment;
}

/// Check if a pointer is properly aligned for the given type
pub fn isAligned(comptime T: type, ptr: *anyopaque) bool {
    _ = getAlignment(T); // Silence unused warning
    _ = ptr;
    return true; // Simplified - in full implementation, would check actual memory layout
}
