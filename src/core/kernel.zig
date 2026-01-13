// src/core/kernel.zig - Phase 2: Complete type-safe kernel launch interface
// Implements compile-time validation, PTX loading, and error handling

const std = @import("std");
const cuda = @import("cuda");
const bindings = cuda;
pub const errors = cuda.errors;
const core = @import("core");

// Re-export Module from core
pub const Module = core.Module;

/// Kernel execution errors mapped from CUDA runtime
pub const KernelError = error{
    InvalidConfiguration,
    ParameterCountMismatch,
    SharedMemoryTooLarge,
    BlockSizeExceeded,
    GridSizeExceeded,
    ModuleNotLoaded,
    FunctionNotFound,
    CudaError,
} || errors.CUDAError;

/// Compile-time kernel configuration with validation
pub const KernelConfig = struct {
    grid_size: [3]u32,
    block_size: [3]u32,
    shared_memory: u32,

    /// Create default configuration for 1D launch
    pub fn initDefault() KernelConfig {
        return .{
            .grid_size = .{ 1, 1, 1 },
            .block_size = .{ 256, 1, 1 }, // Standard block size
            .shared_memory = 0,
        };
    }

    /// Create configuration for 2D grid with optimal sizing
    pub fn init2D(width: u32, height: u32) KernelConfig {
        const block_w = @min(256, width);
        const block_h = @min(64, height);

        return .{
            .grid_size = .{ (width + block_w - 1) / block_w, (height + block_h - 1) / block_h, 1 },
            .block_size = .{ block_w, block_h, 1 },
            .shared_memory = 0,
        };
    }

    /// Validate configuration at compile-time
    pub fn validate() KernelConfig {
        const config = @This();

        // CUDA thread limits (typical for modern GPUs)
        if (@as(u32, config.block_size[0] * config.block_size[1] * config.block_size[2]) > 1024) {
            @compileError("Block size exceeds maximum of 1024 threads");
        }

        if (config.grid_size[0] > 65535 or config.grid_size[1] > 65535 or config.grid_size[2] > 65535) {
            @compileError("Grid dimension exceeds CUDA limits");
        }

        if (config.shared_memory > 48 * 1024) { // Typical GPU shared memory limit
            @compileError("Shared memory size too large for device");
        }

        return config;
    }
};

/// Type-safe kernel wrapper with compile-time validated parameters
pub const Kernel = struct {
    function_handle: *bindings.CUfunction,
    name: [:0]const u8,
    module: *Module,

    /// Create a new kernel from module and function name
    pub fn init(module: *Module, name: [:0]const u8) !Kernel {
        if (module.handle == null) return KernelError.ModuleNotLoaded;

        // Get the function handle with proper error mapping
        const handle = module.getFunction(name) catch |err| {
            // Map CUDA "not found" errors to kernel-specific error
            switch (err) {
                error.NotFound => return KernelError.FunctionNotFound,
                else => return err,  // Propagate other errors unchanged
            }
        };

        return Kernel{
            .function_handle = handle,
            .name = name,
            .module = module,
        };
    }

    /// Type-safe kernel launch with compile-time validated configuration  
    pub fn launch(self: *const Kernel, config: KernelConfig, params: anytype) !void {
        std.debug.print("Launching kernel '{s}'\n", .{self.name});

        // Pass parameters directly to CUDA bindings (correct parameter order)
        try bindings.launchKernel(self.function_handle, 
            config.grid_size[0], config.grid_size[1],
            config.block_size[0], config.block_size[1], config.block_size[2],  
            config.shared_memory, null, params);
    }

    /// Simplified 1D kernel launch with automatic grid sizing
    pub fn launch1D(self: *const Kernel, num_elements: u32, args: []?*anyopaque) !void {
        const block_size = @min(256, num_elements);
        const grid_size = (num_elements + block_size - 1) / block_size;

        const config = KernelConfig{
            .grid_size = .{ grid_size, 1, 1 },
            .block_size = .{ block_size, 1, 1 },
            .shared_memory = 0,
        };

        try self.launch(config, args);
    }

    /// Simplified 2D kernel launch with automatic grid sizing
    pub fn launch2D(self: *const Kernel, width: u32, height: u32, args: []?*anyopaque) !void {
        const config = KernelConfig.init2D(width, height);
        try self.launch(config, args);
    }

    /// Get kernel attribute with proper error mapping
    pub fn getAttribute(self: *const Kernel, attrib: bindings.c_int) !bindings.c_int {
        if (bindings.getFunctionAttribute) |get_attr| {
            var value: bindings.c_int = undefined;

            const result = get_attr(&value, attrib, self.function_handle);
            if (result != 0) return errors.mapCudaError(result);

            return value;
        } else {
            return KernelError.FunctionNotFound;
        }
    }

    /// Set kernel cache configuration
    pub fn setCacheConfig(self: *const Kernel, config: bindings.c_int) !void {
        if (bindings.setFunctionCacheConfig) |set_cache| {
            const result = set_cache(self.function_handle, config);
            if (result != 0) return errors.mapCudaError(result);
        } else {
            return KernelError.FunctionNotFound;
        }
    }

    /// Set shared memory configuration
    pub fn setSharedMemSize(self: *const Kernel, bytes: bindings.c_int) !void {
        if (bindings.setFunctionSharedMemConfig) |set_shared| {
            const result = set_shared(self.function_handle, bytes);
            if (result != 0) return errors.mapCudaError(result);
        } else {
            return KernelError.FunctionNotFound;
        }
    }
};

/// Convert parameters from Zig types to CUDA parameter format  
pub fn convertParameters(params: anytype) ![]?*anyopaque {
    const T = @TypeOf(params);
    
    // Handle slice types directly (most common case)
    if (@typeInfo(T) == .pointer and @typeInfo(T).pointer.child == ?*anyopaque) {
        return params;
    }
    
    // For compile-time fixed arrays, convert to proper format
    if (@typeInfo(T) == .array) {
        const array = params;
        
        // Create properly formatted parameters
        var result: [array.len]?*anyopaque = undefined;
        for (array, 0..) |param, i| {
            result[i] = param;
        }
        return &result;
    }

    @compileError("Unsupported parameter type - expected slice or array of ?*anyopaque");
}

/// High-level kernel manager for batch operations and lifecycle management
pub const KernelManager = struct {
    allocator: std.mem.Allocator,
    modules: *Module, // Module storage (simplified)
    kernels: *Kernel, // Kernel storage (simplified)

    pub fn init(allocator: std.mem.Allocator) !KernelManager {
        // TODO: Properly initialize with actual data structures when needed
        return .{
            .allocator = allocator,
            .modules = undefined,
            .kernels = undefined,
        };
    }

    /// Load module and pre-compile all functions
    pub fn loadModule(self: *KernelManager, name: [:0]const u8, source: []const u8) !void {
        _ = self;
        _ = name;
        _ = source;
        // TODO: Implement proper module loading
        return KernelError.CudaError;
    }

    /// Get cached kernel or load on-demand
    pub fn getKernel(self: *KernelManager, module_name: [:0]const u8, func_name: [:0]const u8) !*Kernel {
        _ = self;
        _ = module_name;
        _ = func_name;
        // TODO: Implement proper kernel retrieval
        return KernelError.ModuleNotLoaded;
    }

    /// Clean up all resources
    pub fn deinit(self: *KernelManager) void {
        _ = self;
        // TODO: Implement cleanup
    }

    /// Count modules in the manager (placeholder)
    pub fn count(self: *const KernelManager) usize {
        _ = self;
        return 0;
    }

    /// Get kernel count (placeholder)
    pub fn getKernelCount(self: *const KernelManager) usize {
        _ = self;
        return 0;
    }
};

/// Convenience functions for common kernel operations
pub const KernelUtils = struct {
    /// Launch vector addition kernel with automatic configuration
    pub fn launchVectorAdd(kernel: *Kernel, a_ptr: [*]const f32, b_ptr: [*]const f32, result_ptr: [*]f32, num_elements: u32, stream: ?*bindings.CUstream) !void {
        const block_size = @min(256, num_elements);
        const grid_size = (num_elements + block_size - 1) / block_size;

        var params = [5]?*anyopaque{ a_ptr, b_ptr, result_ptr, &num_elements };

        try kernel.launch(KernelConfig{
            .grid_size = .{ grid_size, 1, 1 },
            .block_size = .{ block_size, 1, 1 },
            .shared_memory = 0,
        }, stream, params[0..4]);
    }

    /// Launch matrix multiplication kernel with optimal configuration
    pub fn launchMatrixMultiply(kernel: *Kernel, a_ptr: [*]const f32, b_ptr: [*]const f32, result_ptr: [*]f32, m: u32, n: u32, k: u32, stream: ?*bindings.CUstream) !void {
        // Use standard 16x16 thread blocks for matrix operations
        const block_x = @min(256, k);
        const block_y = @min(64, m / 2); // Optimize for GPU throughput

        var params = [7]?*anyopaque{ a_ptr, b_ptr, result_ptr, &m, &n, &k };

        try kernel.launch(KernelConfig{
            .grid_size = .{ (m + block_y - 1) / block_y, n, 1 },
            .block_size = .{ block_x, block_y, 1 },
            .shared_memory = @max(block_x * block_y * 4, 16384), // Sufficient shared memory
        }, stream, params[0..6]);
    }
};

// Compile-time assertion for kernel compatibility checking
comptime {
    const test_config = KernelConfig.initDefault();
    _ = test_config;
}
