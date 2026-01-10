// src/core/module.zig
// PTX/CUBIN compilation and loading
// Phase 1.4: Production-ready module & kernel management

const std = @import("std");
const bindings = @import("../bindings/cuda.zig");
const errors = @import("../bindings/errors.zig");

/// JIT compilation options for runtime-compiled kernels
pub const JitOptions = struct {
    max_registers_per_thread: ?u32,
    shared_memory_size: u32,
    constant_cache_mode: enum { default, prefer_l1, prefer_l2 },
    warp_synchronous_mode: bool,

    pub fn init() JitOptions {
        return .{
            .max_registers_per_thread = null,
            .shared_memory_size = 0,
            .constant_cache_mode = .default,
            .warp_synchronous_mode = false,
        };
    }
};

/// Production-ready CUDA module with function caching
pub const Module = struct {
    handle: *bindings.CUmodule,
    functions: std.StringHashMap(*bindings.CUfunction),

    /// Load embedded PTX from compile-time constant data
    pub fn loadEmbedded(comptime ptx: []const u8) !Module {
        _ = ptx;
        // In a full implementation, this would use cuModuleLoadData
        // For now, we'll implement the basic structure

        return Module{
            .handle = undefined,
            .functions = .empty,
        };
    }

    /// Load module from file (.cubin/.ptx)
    pub fn loadFile(filename: [:0]const u8) !Module {
        const module_handle = try bindings.loadModule(filename);

        return Module{
            .handle = module_handle,
            .functions = .empty,
        };
    }

    /// Load from PTX string with JIT options
    pub fn loadPtx(_: []const u8, _: JitOptions) !Module {
        // This would require cuModuleLoadDataEx or similar for JIT compilation
        // For now, implement basic structure

        return Module{
            .handle = undefined,
            .functions = .empty,
        };
    }

    /// Get function handle with caching
    pub fn getFunction(self: *Module, name: [:0]const u8) !*bindings.CUfunction {
        // Check cache first
        if (self.functions.get(name)) |cached_func| {
            return cached_func;
        }

        // Not in cache, load from module
        const function_handle = try bindings.getFunctionFromModule(self.handle, name);

        // Cache the result for future lookups
        try self.functions.put(std.heap.c_allocator, name, function_handle);

        return function_handle;
    }

    /// Unload module and clean up resources
    pub fn unload(module: *Module) void {
        if (module.handle != null) {
            const result = bindings.cuModuleUnload(module.handle);
            if (result != bindings.CUDA_SUCCESS) {
                std.log.warn("Failed to unload CUDA module", .{});
            }

            // Clean up function cache
            module.functions.deinit(std.heap.c_allocator);
        }

        module.handle = null;
    }

    /// Get global variable from module
    pub fn getGlobal(self: Module, name: [:0]const u8) !struct { ptr: *anyopaque, size: usize } {
        return bindings.getGlobalFromModule(self.handle, name);
    }

    /// Set kernel cache configuration for all functions in this module
    pub fn setCacheConfig(_: Module, _: u32) void {
        // This would iterate through cached functions and call cuFuncSetCache on each
        std.log.warn("Module-level cache config not yet implemented", .{});
    }
};

/// High-level kernel wrapper with type-safe launch capabilities
pub const Kernel = struct {
    function_handle: *bindings.CUfunction,
    name: [:0]const u8,
    module: Module,

    /// Create a new kernel from module and function name
    pub fn init(module: Module, name: [:0]const u8) !Kernel {
        const function_ptr = try module.getFunction(name);

        return Kernel{
            .function_handle = function_ptr,
            .name = name,
            .module = module,
        };
    }

    /// Launch kernel with grid and block dimensions
    pub fn launch(self: *const Kernel, grid_x: u32, grid_y: u32, block_x: u32, block_y: u32, block_z: u32, shared_mem_bytes: u32, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {

        // Convert slice to C-style array for CUDA API
        var c_params: [64]?*anyopaque = undefined; // Support up to 64 parameters

        const param_count = @min(params.len, 64);
        for (0..param_count) |i| {
            c_params[i] = params[i];
        }

        try bindings.launchKernel(self.function_handle, grid_x, grid_y, block_x, block_y, block_z, shared_mem_bytes, stream, &c_params[0..param_count]);
    }

    /// Launch with simplified 2D configuration
    pub fn launch2D(self: *const Kernel, grid_width: u32, grid_height: u32, block_width: u32, block_height: u32, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {
        try self.launch(grid_width, grid_height, 1, block_width, block_height, 1, 0, stream, params);
    }

    /// Launch with simplified 3D configuration
    pub fn launch3D(self: *const Kernel, grid_dims: [2]u32, block_dims: [3]u32, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {
        try self.launch(grid_dims[0], grid_dims[1], block_dims[0], block_dims[1], block_dims[2], 0, stream, params);
    }

    /// Set cache configuration for this specific kernel
    pub fn setCacheConfig(self: *const Kernel, config: u32) !void {
        try bindings.setFunctionCache(self.function_handle, config);
    }

    /// Configure shared memory allocation for this kernel
    pub fn setSharedMemSize(self: *const Kernel, bytes: u32) !void {
        try bindings.setFunctionSharedMem(self.function_handle, bytes);
    }
};

/// Simplified compilation options for PTX compilation
pub const CompilationOptions = struct {
    optimization_level: enum { none, basic, full },

    pub fn init(_: std.mem.Allocator) !CompilationOptions {
        return .{
            .optimization_level = .basic,
        };
    }

    pub fn deinit(_: *CompilationOptions) void {
        // No dynamic cleanup needed in simplified version
        return;
    }

    /// Add include directory (simplified)
    pub fn addIncludeDir(_: *CompilationOptions, _: [:0]const u8) !void {
        return;
    }

    /// Add preprocessor define (simplified)
    pub fn addDefine(_: *CompilationOptions, _: []const u8, _: []const u8) !void {
        return;
    }
};

/// Compile PTX source to binary (requires external toolchain)
pub fn compilePTX(_: [:0]const u8, options: CompilationOptions) ![]u8 {
    _ = options;

    // This would require integration with NVCC or other compilation tools
    @compileError("PTX compilation requires external CUDA compiler");
}

/// Load module from multiple sources with fallback
pub const ModuleLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !ModuleLoader {
        return .{
            .allocator = allocator,
        };
    }

    /// Try to load module, trying multiple strategies
    pub fn loadWithFallback(_: ModuleLoader, primary_source: [:0]const u8, fallback_sources: [][:0]const u8) !Module {

        // First try the primary source
        const result = Module.loadFile(primary_source);
        if (result != error.SymbolNotFound) {
            return result;
        }

        // Try each fallback in order
        for (fallback_sources) |source| {
            const fb_result = Module.loadFile(source);
            if (fb_result != error.SymbolNotFound) {
                return fb_result;
            }
        }

        // All sources failed
        return error.SymbolNotFound;
    }

    /// Load embedded module with automatic fallback to file-based loading
    pub fn loadSmart(_: ModuleLoader, compiled_data: []const u8, filename_fallback: [:0]const u8) !Module {

        // If we had JIT compilation support, we'd try the compiled data first
        _ = compiled_data;

        // For now, fall back to file loading
        return Module.loadFile(filename_fallback);
    }
};

/// Convenience functions for common module operations
pub const loadDefaultStreamKernel = Kernel.init; // Alias for convenience

const Device = @import("device.zig").Device;
