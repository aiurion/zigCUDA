const std = @import("std");
const bindings = @import("bindings/cuda.zig");

pub const CUdeviceptr = bindings.CUdeviceptr;

pub const Dim3 = struct {
    x: bindings.c_uint,
    y: bindings.c_uint = 1,
    z: bindings.c_uint = 1,

    pub fn init(x: bindings.c_uint) Dim3 {
        return .{ .x = x };
    }

    pub fn init2D(x: bindings.c_uint, y: bindings.c_uint) Dim3 {
        return .{ .x = x, .y = y };
    }

    pub fn init3D(x: bindings.c_uint, y: bindings.c_uint, z: bindings.c_uint) Dim3 {
        return .{ .x = x, .y = y, .z = z };
    }
};

pub const LaunchConfig = struct {
    grid: Dim3,
    block: Dim3 = .{ .x = 256 },
    shared_mem_bytes: bindings.c_uint = 0,
    stream: ?*bindings.CUstream = null,
    sync_after: bool = false,

    pub fn init(grid: Dim3) LaunchConfig {
        return .{ .grid = grid };
    }

    pub fn forElements(element_count: usize) LaunchConfig {
        return forElementsWithBlock(element_count, 256);
    }

    pub fn forElementsWithBlock(element_count: usize, block_x: bindings.c_uint) LaunchConfig {
        const block_count = ceilDiv(element_count, block_x);
        return .{
            .grid = .{ .x = toCuint(block_count) },
            .block = .{ .x = block_x },
        };
    }
};

pub const ParamsError = error{
    TooManyParams,
    ParamStorageOverflow,
    ParamAlignmentUnsupported,
};

pub const Params = struct {
    pub const max_params = 32;
    pub const storage_bytes = 1024;
    pub const max_param_align = 16;

    items: [max_params]?*anyopaque = [_]?*anyopaque{null} ** max_params,
    count: usize = 0,
    storage: [storage_bytes]u8 align(max_param_align) = undefined,
    storage_len: usize = 0,

    pub fn init() Params {
        return .{};
    }

    pub fn value(self: *Params, comptime T: type, input: T) ParamsError!void {
        const alignment = @alignOf(T);
        if (alignment > max_param_align) {
            return error.ParamAlignmentUnsupported;
        }

        const offset = alignForward(self.storage_len, alignment);
        if (offset + @sizeOf(T) > storage_bytes) {
            return error.ParamStorageOverflow;
        }

        try self.ensureParamSlot();

        const typed_ptr: *T = @ptrCast(@alignCast(&self.storage[offset]));
        typed_ptr.* = input;
        self.items[self.count] = @ptrCast(typed_ptr);
        self.count += 1;
        self.storage_len = offset + @sizeOf(T);
    }

    pub fn devicePtr(self: *Params, ptr: bindings.CUdeviceptr) ParamsError!void {
        return self.value(bindings.CUdeviceptr, ptr);
    }

    pub fn raw(self: *Params, arg: *anyopaque) ParamsError!void {
        try self.ensureParamSlot();
        self.items[self.count] = arg;
        self.count += 1;
    }

    pub fn slice(self: *Params) []?*anyopaque {
        return self.items[0..self.count];
    }

    pub fn len(self: Params) usize {
        return self.count;
    }

    fn ensureParamSlot(self: Params) ParamsError!void {
        if (self.count >= max_params) {
            return error.TooManyParams;
        }
    }
};

pub const DeviceBufferError = error{BufferTooSmall};

pub const DeviceBuffer = struct {
    ptr: bindings.CUdeviceptr,
    len: usize,
    owned: bool,

    pub fn alloc(bytes: usize) bindings.errors.CUDAError!DeviceBuffer {
        return .{
            .ptr = try bindings.allocDeviceMemory(bytes),
            .len = bytes,
            .owned = true,
        };
    }

    pub fn fromOwned(ptr: bindings.CUdeviceptr, len: usize) DeviceBuffer {
        return .{ .ptr = ptr, .len = len, .owned = true };
    }

    pub fn fromBorrowed(ptr: bindings.CUdeviceptr, len: usize) DeviceBuffer {
        return .{ .ptr = ptr, .len = len, .owned = false };
    }

    pub fn free(self: *DeviceBuffer) bindings.errors.CUDAError!void {
        if (self.owned and self.ptr != 0) {
            try bindings.freeDeviceMemory(self.ptr);
        }
        self.ptr = 0;
        self.len = 0;
        self.owned = false;
    }

    pub fn deinit(self: *DeviceBuffer) void {
        self.free() catch |err| {
            std.log.warn("CUDA device buffer cleanup failed: {s}", .{@errorName(err)});
            self.ptr = 0;
            self.len = 0;
            self.owned = false;
        };
    }

    pub fn copyFrom(self: DeviceBuffer, host_src: []const u8) (bindings.errors.CUDAError || DeviceBufferError)!void {
        if (host_src.len > self.len) {
            return error.BufferTooSmall;
        }
        try bindings.copyHostToDevice(self.ptr, host_src);
    }

    pub fn copyTo(self: DeviceBuffer, host_dst: []u8) (bindings.errors.CUDAError || DeviceBufferError)!void {
        if (host_dst.len > self.len) {
            return error.BufferTooSmall;
        }
        try bindings.copyDeviceToHost(host_dst, self.ptr);
    }

    pub fn copyFromTyped(self: DeviceBuffer, comptime T: type, host_src: []const T) (bindings.errors.CUDAError || DeviceBufferError)!void {
        try self.copyFrom(std.mem.sliceAsBytes(host_src));
    }

    pub fn copyToTyped(self: DeviceBuffer, comptime T: type, host_dst: []T) (bindings.errors.CUDAError || DeviceBufferError)!void {
        try self.copyTo(std.mem.sliceAsBytes(host_dst));
    }
};

pub fn copyToDeviceTyped(comptime T: type, dst: bindings.CUdeviceptr, host_src: []const T) bindings.errors.CUDAError!void {
    try bindings.copyHostToDevice(dst, std.mem.sliceAsBytes(host_src));
}

pub fn copyFromDeviceTyped(comptime T: type, host_dst: []T, src: bindings.CUdeviceptr) bindings.errors.CUDAError!void {
    try bindings.copyDeviceToHost(std.mem.sliceAsBytes(host_dst), src);
}

pub fn launch(function: *bindings.CUfunction, config: LaunchConfig, params: []?*anyopaque) bindings.errors.CUDAError!void {
    try doLaunch(function, config, params);
}

pub const Module = struct {
    allocator: ?std.mem.Allocator,
    handle: ?*bindings.CUmodule,
    owned: bool,

    pub fn loadFile(allocator: std.mem.Allocator, path: []const u8) !Module {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        const handle = try bindings.loadModule(@ptrCast(path_z));
        return .{
            .allocator = allocator,
            .handle = handle,
            .owned = true,
        };
    }

    pub fn loadFirst(allocator: std.mem.Allocator, paths: []const []const u8) !Module {
        if (paths.len == 0) {
            return error.NoModulePath;
        }

        var last_error: ?anyerror = null;
        for (paths) |path| {
            return Module.loadFile(allocator, path) catch |err| {
                last_error = err;
                continue;
            };
        }

        return last_error orelse error.NoModulePath;
    }

    pub fn fromOwned(handle: *bindings.CUmodule) Module {
        return .{ .allocator = null, .handle = handle, .owned = true };
    }

    pub fn fromBorrowed(handle: *bindings.CUmodule) Module {
        return .{ .allocator = null, .handle = handle, .owned = false };
    }

    pub fn unload(self: *Module) bindings.errors.CUDAError!void {
        if (self.owned) {
            if (self.handle) |handle| {
                try bindings.unloadModule(handle);
            }
        }
        self.handle = null;
        self.owned = false;
    }

    pub fn deinit(self: *Module) void {
        self.unload() catch |err| {
            std.log.warn("CUDA module unload failed: {s}", .{@errorName(err)});
            self.handle = null;
            self.owned = false;
        };
    }

    pub fn kernel(self: Module, name: []const u8) !Kernel {
        const handle = self.handle orelse return error.InvalidModule;
        const allocator = self.allocator orelse std.heap.page_allocator;
        const name_z = try allocator.dupeZ(u8, name);
        defer allocator.free(name_z);

        return .{ .function = try bindings.getFunctionFromModule(handle, @ptrCast(name_z)) };
    }
};

pub const Kernel = struct {
    function: *bindings.CUfunction,

    pub fn launch(self: Kernel, config: LaunchConfig, params: []?*anyopaque) bindings.errors.CUDAError!void {
        try doLaunch(self.function, config, params);
    }
};

fn doLaunch(function: *bindings.CUfunction, config: LaunchConfig, params: []?*anyopaque) bindings.errors.CUDAError!void {
    try bindings.launchKernel(
        function,
        config.grid.x,
        config.grid.y,
        config.grid.z,
        config.block.x,
        config.block.y,
        config.block.z,
        config.shared_mem_bytes,
        config.stream,
        params,
    );

    if (config.sync_after) {
        try syncContext();
    }
}

fn syncContext() bindings.errors.CUDAError!void {
    const synchronize = bindings.cuCtxSynchronize orelse return error.SymbolNotFound;
    const result = synchronize();
    if (result == bindings.CUDA_SUCCESS) {
        return;
    }
    return bindings.errors.cudaError(result);
}

fn ceilDiv(numerator: usize, denominator: usize) usize {
    if (denominator == 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

fn toCuint(value: usize) bindings.c_uint {
    return @as(bindings.c_uint, @intCast(value));
}

fn alignForward(value: usize, alignment: usize) usize {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}
