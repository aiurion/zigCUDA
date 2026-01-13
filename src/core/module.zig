// src/core/module.zig - Phase 2: Clean implementation
// PTX/CUBIN compilation and loading with proper error handling
const std = @import("std");
const bindings = @import("cuda");

/// JIT options for runtime-compiled kernels
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
    handle: ?*bindings.CUmodule,
    functions: std.StringHashMap(*bindings.CUfunction),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Module {
        // Ensure bindings are loaded before any module operations
        try bindings.load();

        return .{
            .handle = null, // Will be set by load methods
            .functions = std.StringHashMap(*bindings.CUfunction).init(allocator),
            .allocator = allocator,
        };
    }

    /// Load embedded PTX from compile-time constant data with proper error handling
    pub fn loadEmbedded(allocator: std.mem.Allocator, comptime ptx: []const u8) !Module {
        var module = try Module.init(allocator);

        // SAFEGUARD: Ensure the PTX string is null-terminated for the C API
        // @embedFile data is not guaranteed to be null-terminated when passed as slice
        const ptx_z = try allocator.dupeZ(u8, ptx);
        defer allocator.free(ptx_z);

        if (bindings.cuModuleLoadData) |cu_module_load_data| {
            var module_handle: ?*bindings.CUmodule = null;
            // Pass the explicitly null-terminated pointer
            const result = cu_module_load_data(&module_handle, @ptrCast(ptx_z));

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
            module.handle = module_handle.?;
        } else {
            return error.SymbolNotFound;
        }

        return module;
    }

    /// Load from file (.cubin/.ptx)
    pub fn loadFile(allocator: std.mem.Allocator, filename: [:0]const u8) !Module {
        var module = try Module.init(allocator);

        if (bindings.cuModuleLoad) |cu_module_load| {
            var module_handle: ?*bindings.CUmodule = null;
            const result = cu_module_load(&module_handle, filename);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
            module.handle = module_handle.?;
        } else {
            return error.SymbolNotFound;
        }

        return module;
    }

    /// Load from PTX string with JIT options
    pub fn loadPtx(allocator: std.mem.Allocator, ptx: []const u8, _: JitOptions) !Module {
        var module = try Module.init(allocator);

        // Safeguard null termination
        const ptx_z = try allocator.dupeZ(u8, ptx);
        defer allocator.free(ptx_z);

        if (bindings.cuModuleLoadData) |cu_module_load_data| {
            var module_handle: ?*bindings.CUmodule = null;
            const result = cu_module_load_data(&module_handle, @ptrCast(ptx_z));

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
            module.handle = module_handle.?;
        } else {
            return error.SymbolNotFound;
        }

        // Note: JIT options not fully supported yet
        return module;
    }

    /// Get function handle with caching and proper error handling
pub fn getFunction(self: *Module, name: [:0]const u8) !*bindings.CUfunction {
    // Check cache first
    if (self.functions.get(name)) |cached_func| {
        return cached_func;
    }

    // Not in cache, load from module using actual CUDA binding
    if (bindings.cuModuleGetFunction) |cu_module_get_function| {
        var func_handle: ?*bindings.CUfunction = null;
        const result = cu_module_get_function(&func_handle, self.handle.?, @ptrCast(name));

        if (result != 0) { // CUDA_SUCCESS
            return bindings.errors.cudaError(result);
        }

        // Cache the result for future lookups
        try self.functions.put(try self.allocator.dupe(u8, name), func_handle.?);

        return func_handle.?;
    } else {
        return error.SymbolNotFound;
    }
}

    /// Unload module and clean up resources properly
    pub fn unload(self: *Module) void {
        // Clean up function cache first (always, regardless of handle state)
        var it = self.functions.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.functions.deinit();

        // Then unload the CUDA module if it exists
        if (self.handle != null) {
            if (bindings.cuModuleUnload) |cu_module_unload| {
                const result = cu_module_unload(self.handle.?);

                if (result != 0) { // CUDA_SUCCESS
                    std.log.warn("Failed to unload CUDA module: {}", .{bindings.errors.cudaError(result)});
                }
            }
            self.handle = null;
        }
    }

    /// Get global variable from module with proper error handling
    pub fn getGlobal(self: Module, name: [:0]const u8) !struct { ptr: *anyopaque, size: usize } {
        if (bindings.cuModuleGetGlobal) |cu_module_get_global| {
            var global_ptr: ?*anyopaque = null;
            var bytesize: bindings.c_size_t = undefined;

            const result = cu_module_get_global(&global_ptr, &bytesize, self.handle.?, name);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }

            return .{ .ptr = global_ptr.?, .size = @as(usize, bytesize) };
        } else {
            return error.SymbolNotFound;
        }
    }

    /// Set kernel cache configuration for all functions in this module
    pub fn setCacheConfig(self: Module, config: u32) !void {
        if (bindings.cuFuncSetCacheConfig) |cu_func_set_cache_config| {
            var iterator = self.functions.iterator();
            while (iterator.next()) |entry| {
                const result = cu_func_set_cache_config(entry.value_ptr.*, @as(bindings.c_int, config));

                if (result != 0) { // CUDA_SUCCESS
                    std.log.warn("Failed to set cache config: {}", .{bindings.errors.cudaError(result)});
                }
            }
        } else {
            return error.SymbolNotFound;
        }
    }
};
/// High-level kernel wrapper with type-safe launch capabilities
pub const Kernel = struct {
    function_handle: *bindings.CUfunction,
    name: [:0]const u8,

    /// Create a new kernel from module and function name
    pub fn init(module: Module, name: [:0]const u8) !Kernel {
        // Get the function handle with proper error mapping  
        const function_ptr = module.getFunction(name) catch |err| {
            switch (err) {
                bindings.errors.CUDAError.NotFound => return @import("kernel").KernelError.FunctionNotFound,
                else => return err,  // Propagate other errors unchanged
            }
        };

        return Kernel{
            .function_handle = function_ptr,
            .name = name,
        };
    }

    /// Launch kernel with grid and block dimensions
    pub fn launch(self: *const Kernel, grid_x: bindings.c_uint, grid_y: bindings.c_uint, grid_z: bindings.c_uint, block_x: bindings.c_uint, block_y: bindings.c_uint, block_z: bindings.c_uint, shared_mem_bytes: bindings.c_uint, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {
        if (bindings.launchKernel) |launch_kernel| {
            // Convert slice to C-style array for CUDA API
            var c_params: [64]?*anyopaque = undefined; // Support up to 64 parameters

            const param_count = @min(params.len, 64);
            for (0..param_count) |i| {
                c_params[i] = params[i];
            }

            const result = launch_kernel(self.function_handle, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes, stream, &c_params[0..param_count]);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
        } else {
            return error.SymbolNotFound;
        }
    }

    /// Launch with simplified 2D configuration
    pub fn launch2D(self: *const Kernel, grid_width: u32, grid_height: u32, block_width: bindings.c_uint, block_height: u32, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {
        try self.launch(@as(bindings.c_uint, grid_width), @as(bindings.c_uint, grid_height), 1, block_width, block_height, 1, 0, stream, params);
    }

    /// Launch with simplified 3D configuration
    pub fn launch3D(self: *const Kernel, grid_dims: [2]u32, block_dims: [3]bindings.c_uint, stream: ?*bindings.CUstream, params: []?*anyopaque) !void {
        try self.launch(@as(bindings.c_uint, grid_dims[0]), @as(bindings.c_uint, grid_dims[1]), 1, block_dims[0], block_dims[1], block_dims[2], 0, stream, params);
    }

    /// Set cache configuration for this specific kernel
    pub fn setCacheConfig(self: *const Kernel, config: bindings.c_int) !void {
        if (bindings.setFunctionCacheConfig) |set_func_cache_config| {
            const result = set_func_cache_config(self.function_handle, config);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
        } else {
            return error.SymbolNotFound;
        }
    }

    /// Configure shared memory allocation for this kernel
    pub fn setSharedMemSize(self: *const Kernel, bytes: bindings.c_int) !void {
        if (bindings.setFunctionSharedMemConfig) |set_func_shared_mem_config| {
            const result = set_func_shared_mem_config(self.function_handle, bytes);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
        } else {
            return error.SymbolNotFound;
        }
    }

    /// Get kernel attribute
    pub fn getAttribute(self: *const Kernel, attrib: bindings.c_int) !bindings.c_int {
        if (bindings.getFunctionAttribute) |get_func_attr| {
            var value: bindings.c_int = undefined;

            const result = get_func_attr(&value, attrib, self.function_handle);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
            return value;
        } else {
            return error.SymbolNotFound;
        }
    }

    /// Set kernel attribute
    pub fn setAttribute(self: *const Kernel, attrib: bindings.c_int, value: bindings.c_int) !void {
        if (bindings.setFunctionAttribute) |set_func_attr| {
            const result = set_func_attr(self.function_handle, attrib, value);

            if (result != 0) { // CUDA_SUCCESS
                const err = bindings.errors.cudaError(result);
                
                // Return the original CUDA not found error
                return err;
            }
        } else {
            return error.SymbolNotFound;
        }
    }
};
/// Simplified compilation options for PTX compilation
pub const CompilationOptions = struct {
    optimization_level: enum { none, basic, full },

    pub fn init(allocator: std.mem.Allocator) !CompilationOptions {
        _ = allocator; // Unused in simplified version

        return .{
            .optimization_level = .basic,
        };
    }
};
/// Compile PTX source to binary (requires external toolchain)
pub fn compilePTX(_: [:0]const u8, options: CompilationOptions) ![]u8 {
    _ = options;

    // This would require integration with NVCC or other compilation tools
    @compileError("PTX compilation requires external CUDA compiler - integrate with nvcc or ptxas");
}
/// Load module from multiple sources with fallback strategies
pub const ModuleLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !ModuleLoader {
        return .{
            .allocator = allocator,
        };
    }

    /// Try to load module, trying multiple strategies with proper error handling
    pub fn loadWithFallback(self: ModuleLoader, primary_source: [:0]const u8, fallback_sources: [][:0]const u8) !Module {
        // First try the primary source using file loading
        var result = Module.loadFile(self.allocator, primary_source);

        if (result == error.SymbolNotFound or result == error.NotFound) {
            for (fallback_sources) |source| {
                result = Module.loadFile(self.allocator, source);
                if (result != error.SymbolNotFound and result != error.NotFound) {
                    return result;
                }
            }
        }

        return result;
    }

    /// Load embedded module with automatic fallback to file-based loading
    pub fn loadSmart(self: ModuleLoader, compiled_data: []const u8, filename_fallback: [:0]const u8) !Module {
        // If we had JIT compilation support, we'd try the compiled data first
        if (Module.loadEmbedded(self.allocator, compiled_data)) |module| {
            return module;
        } else |_| {
            // Fall back to file loading
            return Module.loadFile(self.allocator, filename_fallback);
        }
    }

    /// Load from multiple formats with format detection
    pub fn loadAutoDetect(self: ModuleLoader, source: [:0]const u8) !Module {
        const ext = std.mem.span(std.path.extension(source));

        if (std.mem.eql(u8, ext, ".ptx") or std.mem.eql(u8, ext, ".cubin")) {
            return Module.loadFile(self.allocator, source);
        } else {
            // Try as PTX string
            const ptx_data = try self.allocator.dupe(u8, source);

            defer self.allocator.free(ptx_data);

            const jit_options = JitOptions.init();
            return Module.loadPtx(self.allocator, ptx_data, jit_options);
        }
    }
};
/// Convenience functions for common module operations
pub const loadDefaultStreamKernel = Kernel.init; // Alias for compatibility
