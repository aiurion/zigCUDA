// src/bindings/cuda.zig
// Core CUDA Driver API declarations and bindings
// Phase 0: Essential Bindings Implementation

const std = @import("std");
const errors = @import("errors.zig");

pub const @"c_int" = c_int;
pub const @"c_uint" = c_uint;
pub const @"c_ulonglong" = c_ulonglong;
pub const @"c_char" = c_char;
pub const c_float = f32;
pub const c_double = f64;
pub const c_size_t = usize;

// CUDA version structure
pub const CUdriver = extern struct {
    major: c_int,
    minor: c_int,
};

// CUDA result/error codes
pub const CUresult = c_int;

pub const CUDA_SUCCESS = 0;
pub const CUDA_ERROR_INVALID_VALUE = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY = 2;
pub const CUDA_ERROR_NOT_INITIALIZED = 3;
pub const CUDA_ERROR_DEINITIALIZED = 4;
pub const CUDA_ERROR_PROFILER_DISABLED = 5;
pub const CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6;
pub const CUDA_ERROR_PROFILER_ALREADY_STARTED = 7;
pub const CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8;
pub const CUDA_ERROR_NO_DEVICE = 100;
pub const CUDA_ERROR_INVALID_DEVICE = 101;
pub const CUDA_ERROR_INVALID_IMAGE = 200;
pub const CUDA_ERROR_INVALID_CONTEXT = 201;
pub const CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202;
pub const CUDA_ERROR_MAP_FAILED = 205;
pub const CUDA_ERROR_UNMAP_FAILED = 206;
pub const CUDA_ERROR_ARRAY_IS_MAPPED = 207;
pub const CUDA_ERROR_ALREADY_MAPPED = 208;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU = 209;
pub const CUDA_ERROR_ALREADY_ACQUIRED = 210;
pub const CUDA_ERROR_NOT_MAPPED = 211;
pub const CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212;
pub const CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213;
pub const CUDA_ERROR_ECC_UNCORRECTABLE = 214;
pub const CUDA_ERROR_UNSUPPORTED_LIMIT = 215;
pub const CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216;
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217;
pub const CUDA_ERROR_INVALID_PTX = 218;
pub const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219;
pub const CUDA_ERROR_NVLINK_UNCORRECTABLE = 220;
pub const CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221;
pub const CUDA_ERROR_INVALID_SOURCE = 300;
pub const CUDA_ERROR_FILE_NOT_FOUND = 301;
pub const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302;
pub const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303;
pub const CUDA_ERROR_OPERATING_SYSTEM = 304;
pub const CUDA_ERROR_INVALID_HANDLE = 400;
pub const CUDA_ERROR_ILLEGAL_STATE = 401; // This is a common one
pub const CUDA_ERROR_NOT_FOUND = 500;
pub const CUDA_ERROR_NOT_READY = 600;
pub const CUDA_ERROR_ILLEGAL_ADDRESS = 700;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701;
pub const CUDA_ERROR_LAUNCH_TIMEOUT = 702;
pub const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703;
pub const CUDA_ERROR_LAUNCH_FAILED = 719;
pub const CUDA_ERROR_UNKNOWN = 999;

// Opaque CUDA handles
pub const CUdevice = c_int; // CUdevice is often c_int (handle)
pub const CUcontext = opaque {};
pub const CUstream = opaque {};
pub const CUevent = opaque {};
pub const CUmodule = opaque {};
pub const CUfunction = opaque {};
pub const CUarray = opaque {};
pub const CUarray3D = extern struct {
    pub const CUDA_3D = extern struct {
        Width: c_uint,
        Height: c_uint,
        Depth: c_uint,
        Format: c_uint,
        NumChannels: c_uint,
        Reserved: [4]c_uint,
    };
    ChannelDesc: CUDA_3D,
    ptr: ?*anyopaque,
    reserved: c_ulonglong,
};

// Device properties structure
pub const CUdevprop = extern struct {
    major: c_int,
    minor: c_int,
    totalGlobalMem: c_ulonglong,
    sharedMemPerBlock: c_ulonglong,
    totalConstMem: c_ulonglong,
    warpSize: c_int,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [3]c_int,
    maxGridSize: [3]c_int,
    regsPerBlock: c_int,
    memPitch: c_ulonglong,
    deviceName: [256]c_char,
};

// Memory copy directions
pub const CUmemcpyKind = enum(c_int) {
    host_to_host = 0,
    host_to_device = 1,
    device_to_host = 2,
    device_to_device = 3,
};

// Memory flags
pub const CUmemoryAdvise = enum(c_int) {
    preferred_location = 0,
    accessed_by = 1,
    last_touch_location = 2,
    accessed_by_host = 3,
};

// ============================================================================
// DYNAMIC LOADING & FUNCTION POINTERS
// ============================================================================

var lib: ?std.DynLib = null;

// Function Pointers
pub var cuInit: *const fn (flags: c_uint) callconv(.c) CUresult = undefined;
pub var cuDriverGetVersion: *const fn (driver_version: *c_int) callconv(.c) CUresult = undefined;
pub var cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) CUresult = undefined;
pub var cuDeviceGetProperties: *const fn (device: *CUdevprop, device_id: c_int) callconv(.c) CUresult = undefined;
pub var cuDeviceGetName: *const fn (name: [*:0]c_char, len: c_int, dev: CUdevice) callconv(.c) CUresult = undefined;
pub var cuDeviceComputeCapability: *const fn (major: *c_int, minor: *c_int, device: CUdevice) callconv(.c) CUresult = undefined;
pub var cuDeviceTotalMem: *const fn (bytes: *c_ulonglong, device: CUdevice) callconv(.c) CUresult = undefined; // device handle
pub var cuGetErrorName: *const fn (result: CUresult, pstr: *[*:0]const c_char) callconv(.c) CUresult = undefined;
pub var cuGetErrorString: *const fn (result: CUresult, pstr: *[*:0]const c_char) callconv(.c) CUresult = undefined;
// Note on error string getters: they usually take char**.

// Memory
pub var cuMemAllocHost: *const fn (pHost: *?*anyopaque, bytesize: c_size_t) callconv(.c) CUresult = undefined;
pub var cuMemFreeHost: *const fn (pHost: *anyopaque) callconv(.c) CUresult = undefined;
pub var cuMemAlloc: *const fn (pdDev: *?*anyopaque, bytesize: c_size_t) callconv(.c) CUresult = undefined;
pub var cuMemFree: *const fn (pdDev: *anyopaque) callconv(.c) CUresult = undefined;
pub var cuMemcpyHtoD: *const fn (dst: *anyopaque, hostSrc: *const anyopaque, ByteCount: c_size_t) callconv(.c) CUresult = undefined;
pub var cuMemcpyDtoH: *const fn (hostDst: *anyopaque, srcDev: *const anyopaque, ByteCount: c_size_t) callconv(.c) CUresult = undefined;
pub var cuMemcpyDtoD: *const fn (dst: *anyopaque, src: *const anyopaque, ByteCount: c_size_t) callconv(.c) CUresult = undefined;

// Async memory operations
pub var cuMemcpyHtoDAsync: *const fn (dst: *anyopaque, hostSrc: *const anyopaque, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = undefined;
pub var cuMemcpyDtoHAsync: *const fn (hostDst: *anyopaque, srcDev: *const anyopaque, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = undefined;
pub var cuMemcpyDtoDAsync: *const fn (dst: *anyopaque, src: *const anyopaque, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = undefined;

// Memory information and handle operations
pub var cuMemGetInfo: *const fn (free_bytes: *c_ulonglong, total_bytes: *c_ulonglong) callconv(.c) CUresult = undefined;
pub var cuMemGetHandle: *const fn (handle: ?*anyopaque, flags: c_uint, dev_ptr: *const anyopaque, size: c_size_t) callconv(.c) CUresult = undefined;

// Modules
pub var cuModuleLoad: *const fn (pmodule: *?*CUmodule, fname: [*:0]const c_char) callconv(.c) CUresult = undefined;
pub var cuModuleLoadData: *const fn (pmodule: *?*CUmodule, image: [*:0]const c_char) callconv(.c) CUresult = undefined;
pub var cuModuleUnload: *const fn (module: *CUmodule) callconv(.c) CUresult = undefined;
pub var cuCtxCreate: *const fn (pctx: *?*CUcontext, flags: c_uint, device: CUdevice) callconv(.c) CUresult = undefined;
pub var cuCtxDestroy: *const fn (ctx: *CUcontext) callconv(.c) CUresult = undefined;
pub var cuCtxSetCurrent: *const fn (ctx: *CUcontext) callconv(.c) CUresult = undefined;
pub var cuCtxGetCurrent: *const fn (pctx: *?*CUcontext) callconv(.c) CUresult = undefined;
pub var cuCtxPushCurrent: *const fn (ctx: *CUcontext) callconv(.c) CUresult = undefined;
pub var cuCtxPopCurrent: *const fn (pctx: *?*CUcontext, flags: c_uint) callconv(.c) CUresult = undefined;
pub var cuModuleGetFunction: *const fn (pfunc: *?*CUfunction, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = undefined;

// Additional Module & Kernel Management Functions
pub var cuModuleGetGlobal: *const fn (pglobal: *?*anyopaque, pbytesize: *c_size_t, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = undefined;
pub var cuModuleGetTexRef: *const fn (ptref: *?*anyopaque, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = undefined;
pub var cuModuleLaunch: *const fn (function: *CUfunction, gridDimX: c_uint, gridDimY: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: ?*CUstream, kernel_params: [*]?*anyopaque) callconv(.c) CUresult = undefined;
pub var cuModuleLaunchCooperative: *const fn (function: *CUfunction, gridDimX: c_uint, gridDimY: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: ?*CUstream, kernel_params: [*]?*anyopaque) callconv(.c) CUresult = undefined;
pub var cuFuncSetCache: *const fn (function: *CUfunction, cache_config: c_int) callconv(.c) CUresult = undefined;
pub var cuFuncSetSharedMem: *const fn (function: *CUfunction, bytes: c_uint) callconv(.c) CUresult = undefined;

// Stream Management Functions
pub var cuStreamCreate: *const fn (pstream: *?*CUstream, flags: c_uint) callconv(.c) CUresult = undefined;
pub var cuStreamDestroy: *const fn (stream: *CUstream) callconv(.c) CUresult = undefined;
pub var cuStreamQuery: *const fn (stream: *CUstream) callconv(.c) CUresult = undefined;
pub var cuStreamSynchronize: *const fn (stream: *CUstream) callconv(.c) CUresult = undefined;
pub var cuStreamAddCallback: *const fn (stream: *CUstream, callback: *anyopaque, userdata: ?*anyopaque, flags: c_uint) callconv(.c) CUresult = undefined;
pub var cuStreamBeginCapture: *const fn (stream: *CUstream, mode: c_int) callconv(.c) CUresult = undefined;
pub var cuStreamEndCapture: *const fn (pstream_count: *?*[]*CUstream, stream: *CUstream) callconv(.c) CUresult = undefined;
pub var cuStreamGetCaptureState: *const fn (state: *c_int, stream: *CUstream) callconv(.c) CUresult = undefined;

// Event Management Functions
pub var cuEventCreate: *const fn (pEvent: *?*CUevent, flags: c_uint) callconv(.c) CUresult = undefined;
pub var cuEventDestroy: *const fn (event: *CUevent) callconv(.c) CUresult = undefined;
pub var cuEventRecord: *const fn (event: *CUevent, stream: ?*CUstream) callconv(.c) CUresult = undefined;
pub var cuEventSynchronize: *const fn (event: *CUevent) callconv(.c) CUresult = undefined;

pub fn load() !void {
    if (lib != null) return;

    // Try standard names
    const lib_names = [_][]const u8{ "libcuda.so.1", "libcuda.so" };
    for (lib_names) |name| {
        lib = std.DynLib.open(name) catch continue;
        break;
    }
    if (lib == null) {
        // Fallback for WSL specifically if not in path (though usually it is)
        const wsl_path = "/usr/lib/wsl/lib/libcuda.so.1";
        lib = std.DynLib.open(wsl_path) catch null;
    }

    if (lib == null) return error.CudaLibraryNotFound;
    const l = &lib.?;

    // Helper to lookup
    cuInit = l.lookup(@TypeOf(cuInit), "cuInit") orelse return error.SymbolNotFound;
    cuDriverGetVersion = l.lookup(@TypeOf(cuDriverGetVersion), "cuDriverGetVersion") orelse return error.SymbolNotFound;
    cuDeviceGetCount = l.lookup(@TypeOf(cuDeviceGetCount), "cuDeviceGetCount") orelse return error.SymbolNotFound;
    cuDeviceGetProperties = l.lookup(@TypeOf(cuDeviceGetProperties), "cuDeviceGetProperties") orelse return error.SymbolNotFound;
    cuDeviceGetName = l.lookup(@TypeOf(cuDeviceGetName), "cuDeviceGetName") orelse return error.SymbolNotFound;

    // Check if cuDeviceComputeCapability exists (might be version dependent)
    if (l.lookup(@TypeOf(cuDeviceComputeCapability), "cuDeviceComputeCapability")) |fn_ptr| {
        cuDeviceComputeCapability = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuDeviceTotalMem), "cuDeviceTotalMem")) |fn_ptr| {
        cuDeviceTotalMem = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuDeviceTotalMem), "cuDeviceTotalMem_v2")) |fn_ptr| {
        cuDeviceTotalMem = fn_ptr;
    }

    // Error functions
    cuGetErrorName = l.lookup(@TypeOf(cuGetErrorName), "cuGetErrorName") orelse return error.SymbolNotFound;
    cuGetErrorString = l.lookup(@TypeOf(cuGetErrorString), "cuGetErrorString") orelse return error.SymbolNotFound;

    // Memory
    cuMemAllocHost = l.lookup(@TypeOf(cuMemAllocHost), "cuMemAllocHost") orelse
        l.lookup(@TypeOf(cuMemAllocHost), "cuMemAllocHost_v2") orelse return error.SymbolNotFound;
    cuMemFreeHost = l.lookup(@TypeOf(cuMemFreeHost), "cuMemFreeHost") orelse return error.SymbolNotFound;

    cuMemAlloc = l.lookup(@TypeOf(cuMemAlloc), "cuMemAlloc") orelse
        l.lookup(@TypeOf(cuMemAlloc), "cuMemAlloc_v2") orelse return error.SymbolNotFound;

    cuMemFree = l.lookup(@TypeOf(cuMemFree), "cuMemFree") orelse
        l.lookup(@TypeOf(cuMemFree), "cuMemFree_v2") orelse return error.SymbolNotFound;

    cuMemcpyHtoD = l.lookup(@TypeOf(cuMemcpyHtoD), "cuMemcpyHtoD") orelse
        l.lookup(@TypeOf(cuMemcpyHtoD), "cuMemcpyHtoD_v2") orelse return error.SymbolNotFound;

    cuMemcpyDtoH = l.lookup(@TypeOf(cuMemcpyDtoH), "cuMemcpyDtoH") orelse
        l.lookup(@TypeOf(cuMemcpyDtoH), "cuMemcpyDtoH_v2") orelse return error.SymbolNotFound;

    cuMemcpyDtoD = l.lookup(@TypeOf(cuMemcpyDtoD), "cuMemcpyDtoD") orelse
        l.lookup(@TypeOf(cuMemcpyDtoD), "cuMemcpyDtoD_v2") orelse return error.SymbolNotFound;

    // Async memory operations (optional - may not exist on older CUDA versions)
    if (l.lookup(@TypeOf(cuMemcpyHtoDAsync), "cuMemcpyHtoDAsync")) |fn_ptr| {
        cuMemcpyHtoDAsync = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuMemcpyDtoHAsync), "cuMemcpyDtoHAsync")) |fn_ptr| {
        cuMemcpyDtoHAsync = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuMemcpyDtoDAsync), "cuMemcpyDtoDAsync")) |fn_ptr| {
        cuMemcpyDtoDAsync = fn_ptr;
    }

    // Memory information and handle operations
    if (l.lookup(@TypeOf(cuMemGetInfo), "cuMemGetInfo")) |fn_ptr| {
        cuMemGetInfo = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuMemGetHandle), "cuMemGetHandle_v1") orelse 
         l.lookup(@TypeOf(cuMemGetHandle), "cuMemGetHandle_v2")) |fn_ptr| {
        cuMemGetHandle = fn_ptr;
    }

    // Context management
    cuCtxCreate = l.lookup(@TypeOf(cuCtxCreate), "cuCtxCreate") orelse return error.SymbolNotFound;
    cuCtxDestroy = l.lookup(@TypeOf(cuCtxDestroy), "cuCtxDestroy") orelse return error.SymbolNotFound;
    cuCtxSetCurrent = l.lookup(@TypeOf(cuCtxSetCurrent), "cuCtxSetCurrent") orelse return error.SymbolNotFound;
    cuCtxGetCurrent = l.lookup(@TypeOf(cuCtxGetCurrent), "cuCtxGetCurrent") orelse return error.SymbolNotFound;
    cuCtxPushCurrent = l.lookup(@TypeOf(cuCtxPushCurrent), "cuCtxPushCurrent") orelse return error.SymbolNotFound;
    cuCtxPopCurrent = l.lookup(@TypeOf(cuCtxPopCurrent), "cuCtxPopCurrent") orelse return error.SymbolNotFound;
    
    // Modules
    cuModuleLoad = l.lookup(@TypeOf(cuModuleLoad), "cuModuleLoad") orelse return error.SymbolNotFound;
    cuModuleLoadData = l.lookup(@TypeOf(cuModuleLoadData), "cuModuleLoadData") orelse return error.SymbolNotFound;
    cuModuleUnload = l.lookup(@TypeOf(cuModuleUnload), "cuModuleUnload") orelse return error.SymbolNotFound;
    cuModuleGetFunction = l.lookup(@TypeOf(cuModuleGetFunction), "cuModuleGetFunction") orelse return error.SymbolNotFound;

    // Additional Module & Kernel Management Functions
    if (l.lookup(@TypeOf(cuModuleGetGlobal), "cuModuleGetGlobal")) |fn_ptr| {
        cuModuleGetGlobal = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuModuleGetTexRef), "cuModuleGetTexRef")) |fn_ptr| {
        cuModuleGetTexRef = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuModuleLaunch), "cuModuleLaunch")) |fn_ptr| {
        cuModuleLaunch = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuModuleLaunchCooperative), "cuModuleLaunchCooperative")) |fn_ptr| {
        cuModuleLaunchCooperative = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuFuncSetCache), "cuFuncSetCache")) |fn_ptr| {
        cuFuncSetCache = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuFuncSetSharedMem), "cuFuncSetSharedMem")) |fn_ptr| {
        cuFuncSetSharedMem = fn_ptr;
    }

    // Stream Management Functions
    if (l.lookup(@TypeOf(cuStreamCreate), "cuStreamCreate") orelse l.lookup(@TypeOf(cuStreamCreate), "cuStreamCreate_v2")) |fn_ptr| {
        cuStreamCreate = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamDestroy), "cuStreamDestroy")) |fn_ptr| {
        cuStreamDestroy = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamQuery), "cuStreamQuery")) |fn_ptr| {
        cuStreamQuery = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamSynchronize), "cuStreamSynchronize") orelse l.lookup(@TypeOf(cuStreamSynchronize), "cuStreamSynchronize_v2")) |fn_ptr| {
        cuStreamSynchronize = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamAddCallback), "cuStreamAddCallback")) |fn_ptr| {
        cuStreamAddCallback = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamBeginCapture), "cuStreamBeginCapture_v2") orelse l.lookup(@TypeOf(cuStreamBeginCapture), "cuStreamBeginCapture")) |fn_ptr| {
        cuStreamBeginCapture = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamEndCapture), "cuStreamEndCapture_v2") orelse l.lookup(@TypeOf(cuStreamEndCapture), "cuStreamEndCapture")) |fn_ptr| {
        cuStreamEndCapture = fn_ptr;
    }
    if (l.lookup(@TypeOf(cuStreamGetCaptureState), "cuStreamGetCaptureState")) |fn_ptr| {
        cuStreamGetCaptureState = fn_ptr;
    }

    // Event Management Functions
    cuEventCreate = l.lookup(@TypeOf(cuEventCreate), "cuEventCreate") orelse return error.SymbolNotFound;
    if (l.lookup(@TypeOf(cuEventDestroy), "cuEventDestroy")) |fn_ptr| {
        cuEventDestroy = fn_ptr;
    } else {
        // Provide fallback - some CUDA versions might not have this
        std.log.warn("cuEventDestroy not found, event cleanup will be skipped", .{});
    }
    if (l.lookup(@TypeOf(cuEventRecord), "cuEventRecord")) |fn_ptr| {
        cuEventRecord = fn_ptr;
    } else {
        // Provide fallback - some CUDA versions might not have this
        std.log.warn("cuEventRecord not found, event recording will be skipped", .{});
    }
    if (l.lookup(@TypeOf(cuEventSynchronize), "cuEventSynchronize") orelse l.lookup(@TypeOf(cuEventSynchronize), "cuEventSynchronize_v2")) |fn_ptr| {
        cuEventSynchronize = fn_ptr;
    } else {
        // Provide fallback - some CUDA versions might not have this
        std.log.warn("cuEventSynchronize not found, event synchronization will be skipped", .{});
    }
}

// ============================================================================
// ZIG WRAPPER FUNCTIONS
// ============================================================================

pub fn init(flags: c_uint) errors.CUDAError!void {
    load() catch |err| {
        std.log.err("Failed to load CUDA library: {}", .{err});
        return errors.CUDAError.Unknown;
    };

    const result = cuInit(flags);
    if (result == CUDA_SUCCESS) {
        return;
    }

    // Log unexpected result to help debugging
    if (result != 0) {
        std.log.err("cuInit failed with code: {}", .{result});
    }

    return errors.cudaError(result);
}

/// Get CUDA driver version
pub fn getVersion() errors.CUDAError![2]c_int {
    var driver_version: c_int = undefined;
    const result = cuDriverGetVersion(&driver_version);
    if (result != CUDA_SUCCESS) {
        return errors.cudaError(result);
    }

    // Convert driver version to major.minor format
    const major = @divTrunc(driver_version, 1000);
    const minor = @mod(driver_version, 1000);

    return [2]c_int{ major, minor };
}

/// Get number of CUDA devices
pub fn getDeviceCount() errors.CUDAError!c_int {
    var count: c_int = undefined;
    const result = cuDeviceGetCount(&count);
    if (result == CUDA_SUCCESS) {
        return count;
    }
    return errors.cudaError(result);
}

/// Allocate device memory
pub fn allocDeviceMemory(size: c_size_t) errors.CUDAError!*anyopaque {
    var ptr: ?*anyopaque = null;
    const result = cuMemAlloc(&ptr, size);
    if (result == CUDA_SUCCESS) {
        return ptr.?;
    }
    return errors.cudaError(result);
}

/// Free device memory
pub fn freeDeviceMemory(ptr: *anyopaque) errors.CUDAError!void {
    const result = cuMemFree(ptr);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Allocate pinned host memory
pub fn allocHost(size: c_size_t) errors.CUDAError!*anyopaque {
    var ptr: ?*anyopaque = null;
    const result = cuMemAllocHost(&ptr, size);
    if (result == CUDA_SUCCESS) {
        return ptr.?;
    }
    return errors.cudaError(result);
}

/// Free pinned host memory
pub fn freeHost(ptr: *anyopaque) errors.CUDAError!void {
    const result = cuMemFreeHost(ptr);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Copy memory from host to device
pub fn copyHostToDevice(dst: *anyopaque, host_src: []const u8) errors.CUDAError!void {
    const result = cuMemcpyHtoD(dst, host_src.ptr, host_src.len);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Copy memory from device to host
pub fn copyDeviceToHost(host_dst: []u8, device_src: *const anyopaque) errors.CUDAError!void {
    const result = cuMemcpyDtoH(host_dst.ptr, device_src, host_dst.len);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Copy memory from device to device
pub fn copyDeviceToDevice(dst: *anyopaque, src: *const anyopaque, size: c_size_t) errors.CUDAError!void {
    const result = cuMemcpyDtoD(dst, src, size);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Copy memory from host to device asynchronously
pub fn copyHostToDeviceAsync(dst: *anyopaque, host_src: []const u8, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyHtoDAsync != undefined and cuMemcpyHtoDAsync != null) {
        const result = @as(*const fn (*anyopaque, *const anyopaque, usize, ?*CUstream) callconv(.c) CUresult, @ptrCast(cuMemcpyHtoDAsync))(dst, host_src.ptr, host_src.len, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        const result = cuMemcpyHtoD(dst, host_src.ptr, host_src.len);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
}

/// Copy memory from device to host asynchronously
pub fn copyDeviceToHostAsync(host_dst: []u8, device_src: *const anyopaque, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyDtoHAsync != undefined and cuMemcpyDtoHAsync != null) {
        const result = @as(*const fn (*anyopaque, *const anyopaque, usize, ?*CUstream) callconv(.c) CUresult, @ptrCast(cuMemcpyDtoHAsync))(host_dst.ptr, device_src, host_dst.len, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        const result = cuMemcpyDtoH(host_dst.ptr, device_src, host_dst.len);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
}

/// Copy memory from device to device asynchronously
pub fn copyDeviceToDeviceAsync(dst: *anyopaque, src: *const anyopaque, size: c_size_t, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyDtoDAsync != undefined and cuMemcpyDtoDAsync != null) {
        const result = @as(*const fn (*anyopaque, *const anyopaque, usize, ?*CUstream) callconv(.c) CUresult, @ptrCast(cuMemcpyDtoDAsync))(dst, src, size, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        const result = cuMemcpyDtoD(dst, src, size);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
}

/// Get device memory information
pub fn getDeviceMemoryInfo() errors.CUDAError!struct { free: c_ulonglong, total: c_ulonglong } {
    var free_bytes: c_ulonglong = undefined;
    var total_bytes: c_ulonglong = undefined;
    
    if (cuMemGetInfo != undefined and cuMemGetInfo != null) {
        const fn_ptr = @as(*const fn (*c_ulonglong, *c_ulonglong) callconv(.c) CUresult, @ptrCast(cuMemGetInfo));
        const result = fn_ptr(&free_bytes, &total_bytes);
        if (result == CUDA_SUCCESS) {
            return .{ .free = free_bytes, .total = total_bytes };
        }
        return errors.cudaError(result);
    } else {
        // Fallback - return zero values for unsupported
        std.log.warn("cuMemGetInfo not available on this system", .{});
        return .{ .free = 0, .total = 0 };
    }
}

/// Get memory handle for device pointer  
pub fn getMemoryHandle(dev_ptr: *const anyopaque, size: c_size_t) errors.CUDAError!*anyopaque {
    var handle: ?*anyopaque = null;
    
    if (cuMemGetHandle != undefined and cuMemGetHandle != null) {
        const flags: c_uint = 0; // Default flags
        const fn_ptr = @as(*const fn (*?*anyopaque, c_uint, *const anyopaque, usize) callconv(.c) CUresult, @ptrCast(cuMemGetHandle));
        const result = fn_ptr(&handle, flags, dev_ptr, size);
        if (result == CUDA_SUCCESS) {
            return handle.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - for unsupported systems, just cast away constness
        std.log.warn("cuMemGetHandle not available on this system", .{});
        @constCast(dev_ptr); // Use as-is
        return dev_ptr;
    }
}

// ============================================================================
// CONVENIENCE ALIASES FOR BACKWARD COMPATIBILITY
// ============================================================================

/// Backward compatibility alias for allocDeviceMemory
pub const alloc = allocDeviceMemory;

/// Backward compatibility alias for freeDeviceMemory  
pub const free = freeDeviceMemory;

/// Backward compatibility alias for copyHostToDevice
pub const copyHtoD = copyHostToDevice;

/// Backward compatibility alias for copyDeviceToHost
pub const copyDtoH = copyDeviceToHost;

/// Get device properties
pub fn getDeviceProperties(device: CUdevice) errors.CUDAError!CUdevprop {
    var prop: CUdevprop = undefined;
    const result = cuDeviceGetProperties(&prop, device);
    if (result == CUDA_SUCCESS) {
        return prop;
    }
    return errors.cudaError(result);
}

/// Get device name
pub fn getDeviceName(device: CUdevice, allocator: std.mem.Allocator) ![]u8 {
    var buffer: [256]u8 = undefined;
    // Cast buffer pointer to [*:0]c_char expected by C ABI
    const ptr = @as([*:0]c_char, @ptrCast(&buffer));
    const result = cuDeviceGetName(ptr, 256, device);

    if (result != CUDA_SUCCESS) {
        return errors.cudaError(result);
    }

    const len = std.mem.len(ptr);
    const slice = buffer[0..len];
    return allocator.dupe(u8, slice);
}

/// Get compute capability
pub fn getComputeCapability(device: CUdevice) errors.CUDAError!struct { major: c_int, minor: c_int } {
    var major: c_int = undefined;
    var minor: c_int = undefined;
    const result = cuDeviceComputeCapability(&major, &minor, device);
    if (result == CUDA_SUCCESS) {
        return .{ .major = major, .minor = minor };
    }
    return errors.cudaError(result);
}

/// Get total device memory
pub fn getTotalMem(device: CUdevice) errors.CUDAError!usize {
    var bytes: c_ulonglong = undefined;
    const result = cuDeviceTotalMem(&bytes, device);
    if (result == CUDA_SUCCESS) {
        return @as(usize, @intCast(bytes));
    }
    return errors.cudaError(result);
}

/// Get error name
pub fn getErrorName(error_code: CUresult) ![]const u8 {
    var ptr: [*:0]const c_char = undefined;
    const result = cuGetErrorName(error_code, &ptr);
    if (result == CUDA_SUCCESS) {
        const span = std.mem.span(ptr);
        return @ptrCast(span);
    }
    return errors.cudaError(result);
}

/// Get error string
pub fn getErrorString(error_code: CUresult) ![]const u8 {
    var ptr: [*:0]const c_char = undefined;
    const result = cuGetErrorString(error_code, &ptr);
    if (result == CUDA_SUCCESS) {
        const span = std.mem.span(ptr);
        return @ptrCast(span);
    }
    return errors.cudaError(result);
}

/// Create a new CUDA context
pub fn createContext(flags: c_uint, device: CUdevice) errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;
    const result = cuCtxCreate(&ctx_handle, flags, device);
    if (result == CUDA_SUCCESS) {
        return ctx_handle.?;
    }
    return errors.cudaError(result);
}

/// Destroy a CUDA context
pub fn destroyContext(ctx: *CUcontext) errors.CUDAError!void {
    const result = cuCtxDestroy(ctx);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Set the current CUDA context
pub fn setCurrentContext(ctx: *CUcontext) errors.CUDAError!void {
    const result = cuCtxSetCurrent(ctx);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Get the current CUDA context
pub fn getCurrentContext() errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;
    const result = cuCtxGetCurrent(&ctx_handle);
    if (result == CUDA_SUCCESS) {
        return ctx_handle.?;
    }
    return errors.cudaError(result);
}

/// Push context onto the stack
pub fn pushContext(ctx: *CUcontext) errors.CUDAError!void {
    const result = cuCtxPushCurrent(ctx);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Pop context from the stack  
pub fn popContext() errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;
    const flags: c_uint = 0;
    const result = cuCtxPopCurrent(&ctx_handle, flags);
    if (result == CUDA_SUCCESS) {
        return ctx_handle.?;
    }
    return errors.cudaError(result);
}

// ============================================================================
// MODULE & KERNEL MANAGEMENT WRAPPERS
// ============================================================================

/// Load a CUDA module from file (.cubin/.ptx)
pub fn loadModule(filename: [:0]const c_char) errors.CUDAError!*CUmodule {
    var module_handle: ?*CUmodule = null;
    const result = cuModuleLoad(&module_handle, filename);
    if (result == CUDA_SUCCESS) {
        return module_handle.?;
    }
    return errors.cudaError(result);
}

/// Load a CUDA module from memory
pub fn loadModuleFromData(image: [:0]const c_char) errors.CUDAError!*CUmodule {
    var module_handle: ?*CUmodule = null;
    const result = cuModuleLoadData(&module_handle, image);
    if (result == CUDA_SUCCESS) {
        return module_handle.?;
    }
    return errors.cudaError(result);
}

/// Unload a CUDA module
pub fn unloadModule(module: *CUmodule) errors.CUDAError!void {
    const result = cuModuleUnload(module);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Get function handle from module
pub fn getFunctionFromModule(module: *CUmodule, name: [:0]const c_char) errors.CUDAError!*CUfunction {
    var func_handle: ?*CUfunction = null;
    const result = cuModuleGetFunction(&func_handle, module, name);
    if (result == CUDA_SUCCESS) {
        return func_handle.?;
    }
    return errors.cudaError(result);
}

/// Get global variable from module
pub fn getGlobalFromModule(module: *CUmodule, name: [:0]const c_char) errors.CUDAError!struct { ptr: *anyopaque, size: c_size_t } {
    var global_ptr: ?*anyopaque = null;
    var bytesize: c_size_t = undefined;
    
    if (cuModuleGetGlobal != undefined and cuModuleGetGlobal != null) {
        const fn_ptr = @as(*const fn (*?*anyopaque, *c_size_t, *CUmodule, [*:0]const c_char) callconv(.c) CUresult, @ptrCast(cuModuleGetGlobal));
        const result = fn_ptr(&global_ptr, &bytesize, module, name);
        if (result == CUDA_SUCCESS) {
            return .{ .ptr = global_ptr.?, .size = bytesize };
        }
        return errors.cudaError(result);
    } else {
        // Fallback for systems that don't support this
        std.log.warn("cuModuleGetGlobal not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Get texture reference from module  
pub fn getTextureFromModule(module: *CUmodule, name: [:0]const c_char) errors.CUDAError!*anyopaque {
    if (cuModuleGetTexRef != undefined and cuModuleGetTexRef != null) {
        var tex_ref: ?*anyopaque = null;
        const fn_ptr = @as(*const fn (*?*anyopaque, *CUmodule, [*:0]const c_char) callconv(.c) CUresult, @ptrCast(cuModuleGetTexRef));
        const result = fn_ptr(&tex_ref, module, name);
        if (result == CUDA_SUCCESS) {
            return tex_ref.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback for systems that don't support this
        std.log.warn("cuModuleGetTexRef not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Launch kernel synchronously from module
pub fn launchKernel(
    function: *CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: ?*CUstream,
    kernel_params: []?*anyopaque
) errors.CUDAError!void {
    
    if (cuModuleLaunch != undefined and cuModuleLaunch != null) {
        const fn_ptr = @as(*const fn (*CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*CUstream, [*]?*anyopaque) callconv(.c) CUresult, @ptrCast(cuModuleLaunch));
        
        // Convert slice to C array
        var params_array: [32]?*anyopaque = undefined; // Max 32 parameters
        const param_count = @min(kernel_params.len, 32);
        for (0..param_count) |i| {
            params_array[i] = kernel_params[i];
        }
        
        const result = fn_ptr(
            function,
            grid_dim_x, grid_dim_y,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes,
            stream,
            &params_array
        );
        
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - synchronous execution not available
        std.log.warn("cuModuleLaunch not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Launch cooperative kernels from module
pub fn launchCooperativeKernel(
    function: *CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: ?*CUstream,
    kernel_params: []?*anyopaque
) errors.CUDAError!void {
    
    if (cuModuleLaunchCooperative != undefined and cuModuleLaunchCooperative != null) {
        const fn_ptr = @as(*const fn (*CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*CUstream, [*]?*anyopaque) callconv(.c) CUresult, @ptrCast(cuModuleLaunchCooperative));
        
        // Convert slice to C array
        var params_array: [32]?*anyopaque = undefined; // Max 32 parameters
        const param_count = @min(kernel_params.len, 32);
        for (0..param_count) |i| {
            params_array[i] = kernel_params[i];
        }
        
        const result = fn_ptr(
            function,
            grid_dim_x, grid_dim_y,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes,
            stream,
            &params_array
        );
        
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - cooperative execution not available
        std.log.warn("cuModuleLaunchCooperative not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Set kernel cache configuration
pub fn setFunctionCache(function: *CUfunction, cache_config: c_int) errors.CUDAError!void {
    if (cuFuncSetCache != undefined and cuFuncSetCache != null) {
        const result = @as(*const fn (*CUfunction, c_int) callconv(.c) CUresult, @ptrCast(cuFuncSetCache))(function, cache_config);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - function cache configuration not available
        std.log.warn("cuFuncSetCache not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Set shared memory configuration for kernel
pub fn setFunctionSharedMem(function: *CUfunction, bytes: c_uint) errors.CUDAError!void {
    if (cuFuncSetSharedMem != undefined and cuFuncSetSharedMem != null) {
        const result = @as(*const fn (*CUfunction, c_uint) callconv(.c) CUresult, @ptrCast(cuFuncSetSharedMem))(function, bytes);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - shared memory configuration not available
        std.log.warn("cuFuncSetSharedMem not available on this system", .{});
        return error.SymbolNotFound;
    }
}

// ============================================================================
// STREAM MANAGEMENT WRAPPERS (8 functions)
// ============================================================================

/// Create a new CUDA stream
pub fn createStream(flags: c_uint) errors.CUDAError!*CUstream {
    var stream_handle: ?*CUstream = null;
    const result = cuStreamCreate(&stream_handle, flags);
    if (result == CUDA_SUCCESS) {
        return stream_handle.?;
    }
    return errors.cudaError(result);
}

/// Destroy a CUDA stream
pub fn destroyStream(stream: *CUstream) errors.CUDAError!void {
    const result = cuStreamDestroy(stream);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Query stream status without blocking
pub fn queryStream(stream: *CUstream) errors.CUDAError!bool {
    const result = cuStreamQuery(stream);
    switch (result) {
        0 => return true, // Success - operation completed
        600 => return false, // CUDA_ERROR_NOT_READY - still running
        else => return errors.cudaError(result),
    }
}

/// Wait for stream to complete all operations synchronously
pub fn syncStream(stream: *CUstream) errors.CUDAError!void {
    const result = cuStreamSynchronize(stream);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Add callback function to be called when stream completes
pub fn addStreamCallback(
    stream: *CUstream,
    callback: *anyopaque, 
    userdata: ?*anyopaque,
    flags: c_uint
) errors.CUDAError!void {
    if (cuStreamAddCallback != undefined and cuStreamAddCallback != null) {
        const fn_ptr = @as(*const fn (*CUstream, *anyopaque, ?*anyopaque, c_uint) callconv(.c) CUresult, @ptrCast(cuStreamAddCallback));
        const result = fn_ptr(stream, callback, userdata, flags);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - stream callbacks not available
        std.log.warn("cuStreamAddCallback not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Begin capturing operations into a graph  
pub fn beginCapture(stream: *CUstream, mode: c_int) errors.CUDAError!void {
    if (cuStreamBeginCapture != undefined and cuStreamBeginCapture != null) {
        const result = @as(*const fn (*CUstream, c_int) callconv(.c) CUresult, @ptrCast(cuStreamBeginCapture))(stream, mode);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - stream capture not available
        std.log.warn("cuStreamBeginCapture not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// End capturing and get the captured streams  
pub fn endCapture(stream: *CUstream) errors.CUDAError!*[]*CUstream {
    if (cuStreamEndCapture != undefined and cuStreamEndCapture != null) {
        var stream_count: ?*c_int = null;
        
        // First call to get number of streams
        const result1 = @as(*const fn (*?*c_int, *CUstream) callconv(.c) CUresult, @ptrCast(cuStreamEndCapture))(&stream_count, stream);
        if (result1 == CUDA_SUCCESS and stream_count != null) {
            // Mark as used to avoid unused variable warning
            const count = stream_count.?;
            _ = count;
            
            var streams: ?*[]*CUstream = undefined;
            // Second call to get the actual streams
            const result2 = @as(*const fn (*?*c_int, *CUstream) callconv(.c) CUresult, @ptrCast(cuStreamEndCapture))(@ptrCast(&streams), stream);
            
            if (result2 == CUDA_SUCCESS and streams != null) {
                return &streams.?;
            }
        }
        
        // If we get here, there was an issue with the capture
        const final_result = cuStreamQuery(stream);
        return errors.cudaError(final_result);
    } else {
        // Fallback - stream capture not available
        std.log.warn("cuStreamEndCapture not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Get current capture state of a stream
pub fn getCaptureState(stream: *CUstream) errors.CUDAError!c_int {
    var state: c_int = undefined;
    
    if (cuStreamGetCaptureState != undefined and cuStreamGetCaptureState != null) {
        const result = @as(*const fn (*c_int, *CUstream) callconv(.c) CUresult, @ptrCast(cuStreamGetCaptureState))(&state, stream);
        if (result == CUDA_SUCCESS) {
            return state;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - capture state not available
        std.log.warn("cuStreamGetCaptureState not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Create a default stream (convenience function)
pub fn createDefaultStream() errors.CUDAError!*CUstream {
    const default_flags: c_uint = 0;
    return createStream(default_flags);
}

/// Create a non-blocking stream for async operations
pub fn createNonBlockingStream() errors.CUDAError!*CUstream {
    // CU_STREAM_NON_BLOCKING flag (typically 1)
    const flags: c_uint = 1; 
    return createStream(flags);
}

/// Create a high-priority stream for time-critical operations  
pub fn createHighPriorityStream() errors.CUDAError!*CUstream {
    // CU_STREAM_HIGH_PRIORITY flag (typically 2)
    const flags: c_uint = 2;
    return createStream(flags);
}

// ============================================================================
// EVENT MANAGEMENT WRAPPERS (4 functions)
// ============================================================================

/// Create a new CUDA event
pub fn createEvent(flags: c_uint) errors.CUDAError!*CUevent {
    var event_handle: ?*CUevent = null;
    const result = cuEventCreate(&event_handle, flags);
    if (result == CUDA_SUCCESS) {
        return event_handle.?;
    }
    return errors.cudaError(result);
}

/// Destroy a CUDA event
pub fn destroyEvent(event: *CUevent) errors.CUDAError!void {
    if (cuEventDestroy != undefined and cuEventDestroy != null) {
        const result = @as(*const fn (*CUevent) callconv(.c) CUresult, @ptrCast(cuEventDestroy))(event);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - event destruction not available
        std.log.warn("cuEventDestroy not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Record an event in a stream (for synchronization)
pub fn recordEvent(event: *CUevent, stream: ?*CUstream) errors.CUDAError!void {
    if (cuEventRecord != undefined and cuEventRecord != null) {
        const result = @as(*const fn (*CUevent, ?*CUstream) callconv(.c) CUresult, @ptrCast(cuEventRecord))(event, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - event recording not available
        std.log.warn("cuEventRecord not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Synchronously wait for an event to complete
pub fn syncEvent(event: *CUevent) errors.CUDAError!void {
    if (cuEventSynchronize != undefined and cuEventSynchronize != null) {
        const result = @as(*const fn (*CUevent) callconv(.c) CUresult, @ptrCast(cuEventSynchronize))(event);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - event synchronization not available
        std.log.warn("cuEventSynchronize not available on this system", .{});
        return error.SymbolNotFound;
    }
}

// ============================================================================
// CONVENIENCE EVENT FUNCTIONS
// ============================================================================

/// Create a default timing event (flags = 0)
pub fn createDefaultTimingEvent() errors.CUDAError!*CUevent {
    const default_flags: c_uint = 0;
    return createEvent(default_flags);
}

/// Create an event with blocking flag (blocks host thread until completion)
pub fn createBlockingEvent() errors.CUDAError!*CUevent {
    // CU_EVENT_BLOCKING_SYNC flag (typically 1)
    const flags: c_uint = 1;
    return createEvent(flags);
}

/// Record an event in the default stream
pub fn recordInDefaultStream(event: *CUevent) errors.CUDAError!void {
    return recordEvent(event, null);
}




