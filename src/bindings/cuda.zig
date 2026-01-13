// src/bindings/cuda.zig
// Core CUDA Driver API declarations and bindings
// Phase 0: Essential Bindings Implementation

const std = @import("std");
pub const errors = @import("errors.zig");

// Use C's dlopen instead of Zig's DynLib for WSL compatibility
const c = @cImport({
    @cInclude("dlfcn.h");
});

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
pub const CUgraph = opaque {};
pub const CUdeviceptr = c_ulonglong;
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

// Function attributes
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS = 4;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION = 5;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6;
pub const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9;

// Cache configuration
pub const CU_FUNC_CACHE_PREFER_NONE = 0x00;
pub const CU_FUNC_CACHE_PREFER_SHARED = 0x01;
pub const CU_FUNC_CACHE_PREFER_L1 = 0x02;
pub const CU_FUNC_CACHE_PREFER_EQUAL = 0x03;

// Shared memory configuration
pub const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00;
pub const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01;
pub const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02;

// ============================================================================
// DYNAMIC LOADING & FUNCTION POINTERS
// ============================================================================

var lib_handle: ?*anyopaque = null;

// Function Pointers (using optional types for safety)
// WSL might use different signatures - try multiple
pub var cuInit: ?*const fn (flags: c_int) callconv(.c) CUresult = null;
pub var cuDriverGetVersion: ?*const fn (driver_version: *c_int) callconv(.c) CUresult = null;
pub var cuDeviceGetCount: ?*const fn (count: *c_int) callconv(.c) CUresult = null;
pub var cuDeviceGet: ?*const fn (device: *CUdevice, ordinal: c_int) callconv(.c) CUresult = null;
pub var cuDeviceGetProperties: ?*const fn (device: *CUdevprop, device_id: c_int) callconv(.c) CUresult = null;
pub var cuDeviceGetName: ?*const fn (name: [*:0]c_char, len: c_int, dev: CUdevice) callconv(.c) CUresult = null;
pub var cuDeviceComputeCapability: ?*const fn (major: *c_int, minor: *c_int, device: CUdevice) callconv(.c) CUresult = null;
pub var cuDeviceTotalMem: ?*const fn (bytes: *c_ulonglong, device: CUdevice) callconv(.c) CUresult = null;
pub var cuDeviceGetAttribute: ?*const fn (pi: *c_int, attrib: c_int, dev: CUdevice) callconv(.c) CUresult = null;
pub var cuGetErrorName: ?*const fn (result: CUresult, pstr: *[*:0]const c_char) callconv(.c) CUresult = null;
pub var cuGetErrorString: ?*const fn (result: CUresult, pstr: *[*:0]const c_char) callconv(.c) CUresult = null;

// Memory - Note: CUdeviceptr is passed by value (it's a c_ulonglong)
pub var cuMemAllocHost: ?*const fn (pHost: *?*anyopaque, bytesize: c_size_t) callconv(.c) CUresult = null;
pub var cuMemFreeHost: ?*const fn (pHost: *anyopaque) callconv(.c) CUresult = null;
pub var cuMemAlloc: ?*const fn (pdDev: *CUdeviceptr, bytesize: c_size_t) callconv(.c) CUresult = null;
pub var cuMemFree: ?*const fn (dptr: CUdeviceptr) callconv(.c) CUresult = null;
pub var cuMemcpyHtoD: ?*const fn (dstDevice: CUdeviceptr, srcHost: *const anyopaque, ByteCount: c_size_t) callconv(.c) CUresult = null;
pub var cuMemcpyDtoH: ?*const fn (dstHost: *anyopaque, srcDevice: CUdeviceptr, ByteCount: c_size_t) callconv(.c) CUresult = null;
pub var cuMemcpyDtoD: ?*const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: c_size_t) callconv(.c) CUresult = null;

// Async memory operations
pub var cuMemcpyHtoDAsync: ?*const fn (dstDevice: CUdeviceptr, srcHost: *const anyopaque, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = null;
pub var cuMemcpyDtoHAsync: ?*const fn (dstHost: *anyopaque, srcDevice: CUdeviceptr, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = null;
pub var cuMemcpyDtoDAsync: ?*const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: c_size_t, stream: ?*CUstream) callconv(.c) CUresult = null;

// Memory information and handle operations
pub var cuMemGetInfo: ?*const fn (free_bytes: *c_ulonglong, total_bytes: *c_ulonglong) callconv(.c) CUresult = null;
pub var cuMemGetHandle: ?*const fn (handle: ?*anyopaque, flags: c_uint, dev_ptr: *const anyopaque, size: c_size_t) callconv(.c) CUresult = null;

// Modules
pub var cuModuleLoad: ?*const fn (pmodule: *?*CUmodule, fname: [*:0]const c_char) callconv(.c) CUresult = null;
pub var cuModuleLoadData: ?*const fn (pmodule: *?*CUmodule, image: [*:0]const c_char) callconv(.c) CUresult = null;
pub var cuModuleUnload: ?*const fn (module: *CUmodule) callconv(.c) CUresult = null;
pub var cuCtxCreate: ?*const fn (pctx: *?*CUcontext, flags: c_uint, device: CUdevice) callconv(.c) CUresult = null;
pub var cuCtxDestroy: ?*const fn (ctx: *CUcontext) callconv(.c) CUresult = null;
pub var cuCtxSetCurrent: ?*const fn (ctx: *CUcontext) callconv(.c) CUresult = null;
pub var cuCtxGetCurrent: ?*const fn (pctx: *?*CUcontext) callconv(.c) CUresult = null;
pub var cuCtxPushCurrent: ?*const fn (ctx: *CUcontext) callconv(.c) CUresult = null;
pub var cuCtxPopCurrent: ?*const fn (pctx: *?*CUcontext) callconv(.c) CUresult = null;

// Primary Context Management
pub var cuDevicePrimaryCtxRetain: ?*const fn (pctx: *?*CUcontext, device: CUdevice) callconv(.c) CUresult = null;
pub var cuDevicePrimaryCtxRelease: ?*const fn (device: CUdevice) callconv(.c) CUresult = null;
pub var cuDevicePrimaryCtxSetFlags: ?*const fn (device: CUdevice, flags: c_uint) callconv(.c) CUresult = null;
pub var cuModuleGetFunction: ?*const fn (pfunc: *?*CUfunction, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = null;

// Additional Module & Kernel Management Functions
pub var cuModuleGetGlobal: ?*const fn (pglobal: *?*anyopaque, pbytesize: *c_size_t, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = null;
pub var cuModuleGetTexRef: ?*const fn (ptref: *?*anyopaque, module: *CUmodule, name: [*:0]const c_char) callconv(.c) CUresult = null;
pub var cuModuleLaunch: ?*const fn (function: *CUfunction, gridDimX: c_uint, gridDimY: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: ?*CUstream, kernel_params: [*]?*anyopaque) callconv(.c) CUresult = null;
pub var cuLaunchKernel: ?*const fn (function: *CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: ?*CUstream, kernel_params: [*]?*anyopaque, extra: [*]?*anyopaque) callconv(.c) CUresult = null;
pub var cuModuleLaunchCooperative: ?*const fn (function: *CUfunction, gridDimX: c_uint, gridDimY: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: ?*CUstream, kernel_params: [*]?*anyopaque) callconv(.c) CUresult = null;

// Function/Kernel Configuration
pub var cuFuncGetAttribute: ?*const fn (pi: *c_int, attrib: c_int, func: *CUfunction) callconv(.c) CUresult = null;
pub var cuFuncSetAttribute: ?*const fn (func: *CUfunction, attrib: c_int, value: c_int) callconv(.c) CUresult = null;
pub var cuFuncSetCacheConfig: ?*const fn (func: *CUfunction, config: c_int) callconv(.c) CUresult = null;
pub var cuFuncSetSharedMemConfig: ?*const fn (func: *CUfunction, config: c_int) callconv(.c) CUresult = null;

// Stream Management Functions
pub var cuStreamCreate: ?*const fn (pstream: *?*CUstream, flags: c_uint) callconv(.c) CUresult = null;
pub var cuStreamDestroy: ?*const fn (stream: *CUstream) callconv(.c) CUresult = null;
pub var cuStreamQuery: ?*const fn (stream: *CUstream) callconv(.c) CUresult = null;
pub var cuStreamSynchronize: ?*const fn (stream: *CUstream) callconv(.c) CUresult = null;
pub var cuStreamAddCallback: ?*const fn (stream: *CUstream, callback: *anyopaque, userdata: ?*anyopaque, flags: c_uint) callconv(.c) CUresult = null;
pub var cuStreamBeginCapture: ?*const fn (stream: *CUstream, mode: c_int) callconv(.c) CUresult = null;
pub var cuStreamEndCapture: ?*const fn (stream: *CUstream, pgraph: *?*CUgraph) callconv(.c) CUresult = null;
pub var cuStreamGetCaptureState: ?*const fn (state: *c_int, stream: *CUstream) callconv(.c) CUresult = null;

// Event Management Functions
pub var cuEventCreate: ?*const fn (pEvent: *?*CUevent, flags: c_uint) callconv(.c) CUresult = null;
pub var cuEventDestroy: ?*const fn (event: *CUevent) callconv(.c) CUresult = null;
pub var cuEventRecord: ?*const fn (event: *CUevent, stream: ?*CUstream) callconv(.c) CUresult = null;
pub var cuEventSynchronize: ?*const fn (event: *CUevent) callconv(.c) CUresult = null;

// Event timing function
pub var cuEventElapsedTime: ?*const fn (ms: *f32, start: *CUevent, end: *CUevent) callconv(.c) CUresult = null;

/// Helper to lookup a symbol using C's dlsym
fn dlsym_lookup(comptime T: type, name: [*:0]const u8) ?T {
    if (lib_handle) |handle| {
        const sym = c.dlsym(handle, name);
        if (sym != null) {
            return @ptrCast(sym);
        }
    }
    return null;
}

pub fn load() !void {
    if (lib_handle != null) return;

    // Try standard paths using C's dlopen for WSL compatibility
    const lib_paths = [_][*:0]const u8{
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/wsl/lib/libcuda.so.1",
    };

    for (lib_paths) |path| {
        lib_handle = c.dlopen(path, c.RTLD_NOW);
        if (lib_handle != null) {
            std.debug.print("DEBUG: Loaded CUDA library from {s}\n", .{path});
            break;
        }
    }

    if (lib_handle == null) {
        const err = c.dlerror();
        if (err != null) {
            std.debug.print("ERROR: Failed to load CUDA library: {s}\n", .{err});
        }
        return error.CudaLibraryNotFound;
    }

    std.debug.print("DEBUG: Using library handle at address {*}\n", .{lib_handle});

    // Core functions (required)
    cuInit = dlsym_lookup(@TypeOf(cuInit.?), "cuInit") orelse return error.SymbolNotFound;
    std.debug.print("DEBUG: cuInit loaded at address {x}\n", .{@intFromPtr(cuInit.?)});

    cuDriverGetVersion = dlsym_lookup(@TypeOf(cuDriverGetVersion.?), "cuDriverGetVersion") orelse return error.SymbolNotFound;
    cuDeviceGetCount = dlsym_lookup(@TypeOf(cuDeviceGetCount.?), "cuDeviceGetCount") orelse return error.SymbolNotFound;
    cuDeviceGet = dlsym_lookup(@TypeOf(cuDeviceGet.?), "cuDeviceGet") orelse return error.SymbolNotFound;
    cuDeviceGetProperties = dlsym_lookup(@TypeOf(cuDeviceGetProperties.?), "cuDeviceGetProperties") orelse return error.SymbolNotFound;
    cuDeviceGetName = dlsym_lookup(@TypeOf(cuDeviceGetName.?), "cuDeviceGetName") orelse return error.SymbolNotFound;

    cuGetErrorName = dlsym_lookup(@TypeOf(cuGetErrorName.?), "cuGetErrorName") orelse return error.SymbolNotFound;
    cuGetErrorString = dlsym_lookup(@TypeOf(cuGetErrorString.?), "cuGetErrorString") orelse return error.SymbolNotFound;

    // Memory functions - try versioned names
    cuMemAllocHost = dlsym_lookup(@TypeOf(cuMemAllocHost.?), "cuMemAllocHost") orelse
        dlsym_lookup(@TypeOf(cuMemAllocHost.?), "cuMemAllocHost_v2") orelse return error.SymbolNotFound;
    cuMemFreeHost = dlsym_lookup(@TypeOf(cuMemFreeHost.?), "cuMemFreeHost") orelse return error.SymbolNotFound;

    cuMemAlloc = dlsym_lookup(@TypeOf(cuMemAlloc.?), "cuMemAlloc") orelse
        dlsym_lookup(@TypeOf(cuMemAlloc.?), "cuMemAlloc_v2");
    if (cuMemAlloc == null) {
        std.debug.print("ERROR: cuMemAlloc not found\n", .{});
        return error.SymbolNotFound;
    }
    std.debug.print("DEBUG: Found cuMemAlloc symbol at {x}\n", .{@intFromPtr(cuMemAlloc.?)});

    cuMemFree = dlsym_lookup(@TypeOf(cuMemFree.?), "cuMemFree") orelse
        dlsym_lookup(@TypeOf(cuMemFree.?), "cuMemFree_v2");
    if (cuMemFree == null) {
        std.debug.print("ERROR: cuMemFree not found\n", .{});
        return error.SymbolNotFound;
    }
    std.debug.print("DEBUG: Found cuMemFree symbol at {x}\n", .{@intFromPtr(cuMemFree.?)});

    cuMemcpyHtoD = dlsym_lookup(@TypeOf(cuMemcpyHtoD.?), "cuMemcpyHtoD") orelse
        dlsym_lookup(@TypeOf(cuMemcpyHtoD.?), "cuMemcpyHtoD_v2") orelse return error.SymbolNotFound;
    cuMemcpyDtoH = dlsym_lookup(@TypeOf(cuMemcpyDtoH.?), "cuMemcpyDtoH") orelse
        dlsym_lookup(@TypeOf(cuMemcpyDtoH.?), "cuMemcpyDtoH_v2") orelse return error.SymbolNotFound;
    cuMemcpyDtoD = dlsym_lookup(@TypeOf(cuMemcpyDtoD.?), "cuMemcpyDtoD") orelse
        dlsym_lookup(@TypeOf(cuMemcpyDtoD.?), "cuMemcpyDtoD_v2") orelse return error.SymbolNotFound;

    // Optional helper lookups
    cuDeviceComputeCapability = dlsym_lookup(@TypeOf(cuDeviceComputeCapability.?), "cuDeviceComputeCapability");
    cuDeviceGetAttribute = dlsym_lookup(@TypeOf(cuDeviceGetAttribute.?), "cuDeviceGetAttribute");
    cuDeviceTotalMem = dlsym_lookup(@TypeOf(cuDeviceTotalMem.?), "cuDeviceTotalMem") orelse
        dlsym_lookup(@TypeOf(cuDeviceTotalMem.?), "cuDeviceTotalMem_v2");

    // Async memory operations (optional)
    cuMemcpyHtoDAsync = dlsym_lookup(@TypeOf(cuMemcpyHtoDAsync.?), "cuMemcpyHtoDAsync");
    cuMemcpyDtoHAsync = dlsym_lookup(@TypeOf(cuMemcpyDtoHAsync.?), "cuMemcpyDtoHAsync");
    cuMemcpyDtoDAsync = dlsym_lookup(@TypeOf(cuMemcpyDtoDAsync.?), "cuMemcpyDtoDAsync");

    // Memory information and handle operations
    cuMemGetInfo = dlsym_lookup(@TypeOf(cuMemGetInfo.?), "cuMemGetInfo");
    cuMemGetHandle = dlsym_lookup(@TypeOf(cuMemGetHandle.?), "cuMemGetHandle_v1") orelse
        dlsym_lookup(@TypeOf(cuMemGetHandle.?), "cuMemGetHandle_v2");

    // Context management (required)
    cuCtxCreate = dlsym_lookup(@TypeOf(cuCtxCreate.?), "cuCtxCreate") orelse return error.SymbolNotFound;
    cuCtxDestroy = dlsym_lookup(@TypeOf(cuCtxDestroy.?), "cuCtxDestroy") orelse return error.SymbolNotFound;
    cuCtxSetCurrent = dlsym_lookup(@TypeOf(cuCtxSetCurrent.?), "cuCtxSetCurrent") orelse return error.SymbolNotFound;
    cuCtxGetCurrent = dlsym_lookup(@TypeOf(cuCtxGetCurrent.?), "cuCtxGetCurrent") orelse return error.SymbolNotFound;
    cuCtxPushCurrent = dlsym_lookup(@TypeOf(cuCtxPushCurrent.?), "cuCtxPushCurrent") orelse return error.SymbolNotFound;
    cuCtxPopCurrent = dlsym_lookup(@TypeOf(cuCtxPopCurrent.?), "cuCtxPopCurrent") orelse return error.SymbolNotFound;

    // Primary Context Management (recommended for libraries like cuBLAS)
    cuDevicePrimaryCtxRetain = dlsym_lookup(@TypeOf(cuDevicePrimaryCtxRetain.?), "cuDevicePrimaryCtxRetain") orelse return error.SymbolNotFound;
    cuDevicePrimaryCtxRelease = dlsym_lookup(@TypeOf(cuDevicePrimaryCtxRelease.?), "cuDevicePrimaryCtxRelease") orelse return error.SymbolNotFound;
    cuDevicePrimaryCtxSetFlags = dlsym_lookup(@TypeOf(cuDevicePrimaryCtxSetFlags.?), "cuDevicePrimaryCtxSetFlags") orelse return error.SymbolNotFound;

    // Modules (required)
    cuModuleLoad = dlsym_lookup(@TypeOf(cuModuleLoad.?), "cuModuleLoad") orelse return error.SymbolNotFound;
    cuModuleLoadData = dlsym_lookup(@TypeOf(cuModuleLoadData.?), "cuModuleLoadData") orelse return error.SymbolNotFound;
    cuModuleUnload = dlsym_lookup(@TypeOf(cuModuleUnload.?), "cuModuleUnload") orelse return error.SymbolNotFound;
    cuModuleGetFunction = dlsym_lookup(@TypeOf(cuModuleGetFunction.?), "cuModuleGetFunction") orelse return error.SymbolNotFound;

    // Additional Module & Kernel Management Functions (optional)
    cuModuleGetGlobal = dlsym_lookup(@TypeOf(cuModuleGetGlobal.?), "cuModuleGetGlobal");
    cuModuleGetTexRef = dlsym_lookup(@TypeOf(cuModuleGetTexRef.?), "cuModuleGetTexRef");
    cuModuleLaunch = dlsym_lookup(@TypeOf(cuModuleLaunch.?), "cuModuleLaunch");
    cuLaunchKernel = dlsym_lookup(@TypeOf(cuLaunchKernel.?), "cuLaunchKernel");
    cuModuleLaunchCooperative = dlsym_lookup(@TypeOf(cuModuleLaunchCooperative.?), "cuModuleLaunchCooperative");

    // Function/Kernel Configuration Functions (optional)
    cuFuncGetAttribute = dlsym_lookup(@TypeOf(cuFuncGetAttribute.?), "cuFuncGetAttribute");
    cuFuncSetAttribute = dlsym_lookup(@TypeOf(cuFuncSetAttribute.?), "cuFuncSetAttribute");
    cuFuncSetCacheConfig = dlsym_lookup(@TypeOf(cuFuncSetCacheConfig.?), "cuFuncSetCacheConfig");
    cuFuncSetSharedMemConfig = dlsym_lookup(@TypeOf(cuFuncSetSharedMemConfig.?), "cuFuncSetSharedMemConfig");

    // Stream Management Functions (optional)
    cuStreamCreate = dlsym_lookup(@TypeOf(cuStreamCreate.?), "cuStreamCreate") orelse
        dlsym_lookup(@TypeOf(cuStreamCreate.?), "cuStreamCreate_v2");
    cuStreamDestroy = dlsym_lookup(@TypeOf(cuStreamDestroy.?), "cuStreamDestroy");
    cuStreamQuery = dlsym_lookup(@TypeOf(cuStreamQuery.?), "cuStreamQuery");
    cuStreamSynchronize = dlsym_lookup(@TypeOf(cuStreamSynchronize.?), "cuStreamSynchronize") orelse
        dlsym_lookup(@TypeOf(cuStreamSynchronize.?), "cuStreamSynchronize_v2");
    cuStreamAddCallback = dlsym_lookup(@TypeOf(cuStreamAddCallback.?), "cuStreamAddCallback");
    cuStreamBeginCapture = dlsym_lookup(@TypeOf(cuStreamBeginCapture.?), "cuStreamBeginCapture_v2") orelse
        dlsym_lookup(@TypeOf(cuStreamBeginCapture.?), "cuStreamBeginCapture");
    cuStreamEndCapture = dlsym_lookup(@TypeOf(cuStreamEndCapture.?), "cuStreamEndCapture_v2") orelse
        dlsym_lookup(@TypeOf(cuStreamEndCapture.?), "cuStreamEndCapture");
    cuStreamGetCaptureState = dlsym_lookup(@TypeOf(cuStreamGetCaptureState.?), "cuStreamGetCaptureState");

    // Event Management Functions (optional)
    cuEventCreate = dlsym_lookup(@TypeOf(cuEventCreate.?), "cuEventCreate");
    cuEventDestroy = dlsym_lookup(@TypeOf(cuEventDestroy.?), "cuEventDestroy");
    cuEventRecord = dlsym_lookup(@TypeOf(cuEventRecord.?), "cuEventRecord");
    cuEventSynchronize = dlsym_lookup(@TypeOf(cuEventSynchronize.?), "cuEventSynchronize") orelse
        dlsym_lookup(@TypeOf(cuEventSynchronize.?), "cuEventSynchronize_v2");
    cuEventElapsedTime = dlsym_lookup(@TypeOf(cuEventElapsedTime.?), "cuEventElapsedTime");
}


pub fn init(flags: c_int) errors.CUDAError!void {
    try load();

    if (cuInit) |f| {
        std.debug.print("DEBUG: Calling cuInit with flags {d}...\n", .{flags});
        const result = f(flags);
        if (result != CUDA_SUCCESS) {
            std.debug.print("ERROR: cuInit failed with code {d}\n", .{result});
            return errors.cudaError(result);
        }
        std.debug.print("INFO: cuInit succeeded\n", .{});
    } else {
        std.debug.print("ERROR: cuInit not loaded\n", .{});
        return error.Uninitialized;
    }
}

/// Get CUDA driver version
pub fn getVersion() errors.CUDAError![2]c_int {
    if (cuDriverGetVersion) |f| {
        var driver_version: c_int = undefined;
        const result = f(&driver_version);
        if (result != CUDA_SUCCESS) {
            return errors.cudaError(result);
        }

        // Convert driver version to major.minor format
        const major = @divTrunc(driver_version, 1000);
        const minor = @mod(driver_version, 1000);

        return [2]c_int{ major, minor };
    } else {
        std.debug.print("ERROR: cuDriverGetVersion not loaded\n", .{});
        return error.Uninitialized;
    }
}

/// Get number of CUDA devices
pub fn getDeviceCount() errors.CUDAError!c_int {
    if (cuDeviceGetCount) |f| {
        var count: c_int = undefined;
        const result = f(&count);
        if (result == CUDA_SUCCESS) {
            return count;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Get device handle by index
pub fn getDevice(device_index: c_int) errors.CUDAError!CUdevice {
    if (cuDeviceGet) |f| {
        var device: CUdevice = undefined;
        const result = f(&device, device_index);
        if (result == CUDA_SUCCESS) {
            return device;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Allocate device memory
pub fn allocDeviceMemory(size: c_size_t) errors.CUDAError!CUdeviceptr {
    if (cuMemAlloc) |f| {
        var dev_ptr: CUdeviceptr = 0;
        const result = f(&dev_ptr, size);
        if (result == CUDA_SUCCESS) {
            return dev_ptr;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Free device memory
pub fn freeDeviceMemory(dev_ptr: CUdeviceptr) errors.CUDAError!void {
    if (cuMemFree) |f| {
        const result = f(dev_ptr);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Allocate pinned host memory
pub fn allocHost(size: c_size_t) errors.CUDAError!*anyopaque {
    var ptr: ?*anyopaque = null;
    const result = cuMemAllocHost.?(&ptr, size);
    if (result == CUDA_SUCCESS) {
        return ptr.?;
    }
    return errors.cudaError(result);
}

/// Free pinned host memory
pub fn freeHost(ptr: *anyopaque) errors.CUDAError!void {
    const result = cuMemFreeHost.?(ptr);
    if (result == CUDA_SUCCESS) {
        return;
    }
    return errors.cudaError(result);
}

/// Copy memory from host to device
pub fn copyHostToDevice(dst: CUdeviceptr, host_src: []const u8) errors.CUDAError!void {
    if (cuMemcpyHtoD) |f| {
        const result = f(dst, host_src.ptr, host_src.len);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Copy memory from device to host
pub fn copyDeviceToHost(host_dst: []u8, device_src: CUdeviceptr) errors.CUDAError!void {
    if (cuMemcpyDtoH) |f| {
        const result = f(host_dst.ptr, device_src, host_dst.len);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Copy memory from device to device
pub fn copyDeviceToDevice(dst: CUdeviceptr, src: CUdeviceptr, size: c_size_t) errors.CUDAError!void {
    if (cuMemcpyDtoD) |f| {
        const result = f(dst, src, size);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Copy memory from host to device asynchronously
pub fn copyHostToDeviceAsync(dst: CUdeviceptr, host_src: []const u8, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyHtoDAsync) |f| {
        const result = f(dst, host_src.ptr, host_src.len, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        return copyHostToDevice(dst, host_src);
    }
}

/// Copy memory from device to host asynchronously
pub fn copyDeviceToHostAsync(host_dst: []u8, device_src: CUdeviceptr, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyDtoHAsync) |f| {
        const result = f(host_dst.ptr, device_src, host_dst.len, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        return copyDeviceToHost(host_dst, device_src);
    }
}

/// Copy memory from device to device asynchronously
pub fn copyDeviceToDeviceAsync(dst: CUdeviceptr, src: CUdeviceptr, size: c_size_t, stream: ?*CUstream) errors.CUDAError!void {
    if (cuMemcpyDtoDAsync) |f| {
        const result = f(dst, src, size, stream);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback to synchronous copy
        std.log.warn("Async memory operations not available, falling back to synchronous", .{});
        return copyDeviceToDevice(dst, src, size);
    }
}

/// Get device memory information
pub fn getDeviceMemoryInfo() errors.CUDAError!struct { free: c_ulonglong, total: c_ulonglong } {
    var free_bytes: c_ulonglong = undefined;
    var total_bytes: c_ulonglong = undefined;

    if (cuMemGetInfo) |f| {
        const result = f(&free_bytes, &total_bytes);
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

    if (cuMemGetHandle) |f| {
        const flags: c_uint = 0; // Default flags
        const result = f(&handle, flags, dev_ptr, size);
        if (result == CUDA_SUCCESS) {
            return handle.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - for unsupported systems, just cast away constness
        std.log.warn("cuMemGetHandle not available on this system", .{});
        return @constCast(dev_ptr);
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
    const result = cuDeviceGetProperties.?(&prop, device);
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
    const result = cuDeviceGetName.?(ptr, 256, device);

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

    if (cuDeviceComputeCapability) |f| {
        const result = f(&major, &minor, device);
        if (result == CUDA_SUCCESS) {
            return .{ .major = major, .minor = minor };
        }
        return errors.cudaError(result);
    } else {
        // Fallback using attributes if capability function is missing
        major = 0;
        minor = 0;
        if (cuDeviceGetAttribute) |f| {
            _ = f(&major, 75, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            _ = f(&minor, 76, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        }
        return .{ .major = major, .minor = minor };
    }
}

/// Get total device memory
pub fn getTotalMem(device: CUdevice) errors.CUDAError!usize {
    var bytes: c_ulonglong = undefined;
    if (cuDeviceTotalMem) |f| {
        const result = f(&bytes, device);
        if (result == CUDA_SUCCESS) {
            return @as(usize, @intCast(bytes));
        }
        return errors.cudaError(result);
    }
    return error.NotSupported;
}

/// Get error name
pub fn getErrorName(error_code: CUresult) ![]const u8 {
    var ptr: [*:0]const c_char = undefined;
    const result = cuGetErrorName.?(error_code, &ptr);
    if (result == CUDA_SUCCESS) {
        const span = std.mem.span(ptr);
        return @ptrCast(span);
    }
    return errors.cudaError(result);
}

/// Get error string
pub fn getErrorString(error_code: CUresult) ![]const u8 {
    var ptr: [*:0]const c_char = undefined;
    const result = cuGetErrorString.?(error_code, &ptr);
    if (result == CUDA_SUCCESS) {
        const span = std.mem.span(ptr);
        return @ptrCast(span);
    }
    return errors.cudaError(result);
}

/// Create a new CUDA context
pub fn createContext(flags: c_uint, device: CUdevice) errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;

    if (cuCtxCreate != null) {
        const result = @as(*const fn (*?*CUcontext, c_uint, c_int) callconv(.c) CUresult, @ptrCast(cuCtxCreate))(&ctx_handle, flags, device);
        if (result == CUDA_SUCCESS) {
            return ctx_handle.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context creation not available
        std.log.warn("cuCtxCreate not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Destroy a CUDA context
pub fn destroyContext(ctx: *CUcontext) errors.CUDAError!void {
    if (cuCtxDestroy != null) {
        const result = @as(*const fn (*CUcontext) callconv(.c) CUresult, @ptrCast(cuCtxDestroy))(ctx);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context destruction not available
        std.log.warn("cuCtxDestroy not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Set the current CUDA context
pub fn setCurrentContext(ctx: *CUcontext) errors.CUDAError!void {
    if (cuCtxSetCurrent != null) {
        const result = @as(*const fn (*CUcontext) callconv(.c) CUresult, @ptrCast(cuCtxSetCurrent))(ctx);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context setting not available
        std.log.warn("cuCtxSetCurrent not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Get the current CUDA context
pub fn getCurrentContext() errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;

    if (cuCtxGetCurrent != null) {
        const result = @as(*const fn (*?*CUcontext) callconv(.c) CUresult, @ptrCast(cuCtxGetCurrent))(&ctx_handle);
        if (result == CUDA_SUCCESS) {
            return ctx_handle.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context query not available
        std.log.warn("cuCtxGetCurrent not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Push context onto the stack
pub fn pushContext(ctx: *CUcontext) errors.CUDAError!void {
    if (cuCtxPushCurrent != null) {
        const result = @as(*const fn (*CUcontext) callconv(.c) CUresult, @ptrCast(cuCtxPushCurrent))(ctx);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context push not available
        std.log.warn("cuCtxPushCurrent not available on this system", .{});
        return error.SymbolNotFound;
    }
}

/// Pop context from the stack
pub fn popContext() errors.CUDAError!*CUcontext {
    var ctx_handle: ?*CUcontext = null;

    if (cuCtxPopCurrent != null) {
        const result = @as(*const fn (*?*CUcontext) callconv(.c) CUresult, @ptrCast(cuCtxPopCurrent))(&ctx_handle);
        if (result == CUDA_SUCCESS) {
            return ctx_handle.?;
        }
        return errors.cudaError(result);
    } else {
        // Fallback - context pop not available
        std.log.warn("cuCtxPopCurrent not available on this system", .{});
        return error.SymbolNotFound;
    }
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
pub fn launchKernel(function: *CUfunction, grid_dim_x: c_uint, grid_dim_y: c_uint, block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint, shared_mem_bytes: c_uint, stream: ?*CUstream, kernel_params: []?*anyopaque) errors.CUDAError!void {
    // Convert slice to C array (CUDA expects non-optional pointers)
    var params_array: [32]*anyopaque = undefined; // Max 32 parameters
    const param_count = @min(kernel_params.len, 32);
    for (0..param_count) |i| {
        params_array[i] = kernel_params[i] orelse return error.InvalidValue;
    }

    // Try modern cuLaunchKernel first (CUDA 4.0+)
    if (cuLaunchKernel) |launch_fn| {
        const fn_ptr = @as(*const fn (*CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*CUstream, [*]*anyopaque, ?[*]*anyopaque) callconv(.c) CUresult, @ptrCast(launch_fn));

        const result = fn_ptr(function, grid_dim_x, grid_dim_y, 1, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, stream, &params_array, null);

        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }

    // Fallback to deprecated cuModuleLaunch (CUDA 2.0-3.x)
    if (cuModuleLaunch) |launch_fn| {
        const fn_ptr = @as(*const fn (*CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*CUstream, [*]*anyopaque) callconv(.c) CUresult, @ptrCast(launch_fn));

        const result = fn_ptr(function, grid_dim_x, grid_dim_y, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, stream, &params_array);

        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }

    // No kernel launch API available
    std.log.warn("Neither cuLaunchKernel nor cuModuleLaunch available on this system", .{});
    return error.SymbolNotFound;
}

/// Launch cooperative kernels from module
pub fn launchCooperativeKernel(function: *CUfunction, grid_dim_x: c_uint, grid_dim_y: c_uint, block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint, shared_mem_bytes: c_uint, stream: ?*CUstream, kernel_params: []?*anyopaque) errors.CUDAError!void {
    if (cuModuleLaunchCooperative != undefined and cuModuleLaunchCooperative != null) {
        const fn_ptr = @as(*const fn (*CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*CUstream, [*]*anyopaque) callconv(.c) CUresult, @ptrCast(cuModuleLaunchCooperative));

        // Convert slice to C array (CUDA expects non-optional pointers)
        var params_array: [32]*anyopaque = undefined; // Max 32 parameters
        const param_count = @min(kernel_params.len, 32);
        for (0..param_count) |i| {
            params_array[i] = kernel_params[i] orelse return error.InvalidValue;
        }

        const result = fn_ptr(function, grid_dim_x, grid_dim_y, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, stream, &params_array);

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
pub fn addStreamCallback(stream: *CUstream, callback: *anyopaque, userdata: ?*anyopaque, flags: c_uint) errors.CUDAError!void {
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

/// End capturing and get the captured graph
pub fn endCapture(stream: *CUstream) errors.CUDAError!*CUgraph {
    if (cuStreamEndCapture != undefined and cuStreamEndCapture != null) {
        var graph: ?*CUgraph = null;

        const result = @as(*const fn (*CUstream, *?*CUgraph) callconv(.c) CUresult, @ptrCast(cuStreamEndCapture))(stream, &graph);
        if (result == CUDA_SUCCESS and graph != null) {
            return graph.?;
        }
        return errors.cudaError(result);
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

// ============================================================================
// KERNEL CONFIGURATION FUNCTIONS
// ============================================================================

/// Get a function attribute
pub fn getFunctionAttribute(func: *CUfunction, attrib: c_int) errors.CUDAError!c_int {
    if (cuFuncGetAttribute) |f| {
        var value: c_int = undefined;
        const result = f(&value, attrib, func);
        if (result == CUDA_SUCCESS) {
            return value;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Set a function attribute
pub fn setFunctionAttribute(func: *CUfunction, attrib: c_int, value: c_int) errors.CUDAError!void {
    if (cuFuncSetAttribute) |f| {
        const result = f(func, attrib, value);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Set cache configuration for a function
pub fn setFunctionCacheConfig(func: *CUfunction, config: c_int) errors.CUDAError!void {
    if (cuFuncSetCacheConfig) |f| {
        const result = f(func, config);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Set shared memory configuration for a function
pub fn setFunctionSharedMemConfig(func: *CUfunction, config: c_int) errors.CUDAError!void {
    if (cuFuncSetSharedMemConfig) |f| {
        const result = f(func, config);
        if (result == CUDA_SUCCESS) {
            return;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

// ============================================================================
// MODULE RESOURCE ACCESS FUNCTIONS
// ============================================================================

/// Get pointer to global variable from module
pub fn getModuleGlobal(module: *CUmodule, name: [*:0]const u8) errors.CUDAError!struct { ptr: CUdeviceptr, size: usize } {
    if (cuModuleGetGlobal) |f| {
        var global_ptr: ?*anyopaque = null;
        var size: c_size_t = undefined;
        const result = f(&global_ptr, &size, module, name);
        if (result == CUDA_SUCCESS) {
            // Cast pointer to device pointer
            const dev_ptr = @intFromPtr(global_ptr);
            return .{ .ptr = @as(CUdeviceptr, dev_ptr), .size = size };
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}

/// Get texture reference from module
pub fn getModuleTexRef(module: *CUmodule, name: [*:0]const u8) errors.CUDAError!*anyopaque {
    if (cuModuleGetTexRef) |f| {
        var texref: ?*anyopaque = null;
        const result = f(&texref, module, name);
        if (result == CUDA_SUCCESS) {
            if (texref) |ref| {
                return ref;
            }
            return error.InvalidValue;
        }
        return errors.cudaError(result);
    }
    return error.Uninitialized;
}
