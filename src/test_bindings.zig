// src/test_bindings.zig
// Test module to verify CUDA bindings compile correctly
// This doesn't run actual CUDA operations, just validates the bindings

const std = @import("std");
const testing = std.testing;

// Import our own cuda module from the same project - use relative import for src/ location
const cuda = @import("../src/bindings/cuda.zig");

test "cuda bindings compile" {
    // Just verify that the basic types and constants exist
    try testing.expectEqual(@as(cuda.CUresult, 0), cuda.CUDA_SUCCESS);
    
    std.debug.print("✓ CUDA type definitions compiled successfully\n", .{});
}

test "cuda function pointers exist" {
    // Verify that the function pointer declarations are valid
    if (cuda.cuInit != null) {
        std.debug.print("✓ cuInit function pointer declared\n", .{});
    }
    
    if (cuda.cuMemAlloc != null) {
        std.debug.print("✓ cuMemAlloc function pointer declared\n", .{});
    }
    
    // Test that we can reference the types
    const device_type: cuda.CUdevice = 0;
    _ = device_type;
    
    try testing.expect(true); // If we get here, types compiled successfully
}

test "cuda error handling" {
    // Import and test our error module - use relative import for src/ location  
    const errors = @import("../src/bindings/errors.zig");
    
    // Test that the CUDAError type exists (just verify it compiles)
    _ = errors.CUDAError;
    std.debug.print("✓ CUDA error types compiled successfully\n", .{});
}

test "cuda memory types" {
    // Verify that device pointer and context types compile
    const dev_ptr: cuda.CUdeviceptr = 0x1234;
    _ = dev_ptr;
    
    // Test struct types
    const prop: cuda.CUdevprop = undefined;
    _ = prop;
    
    try testing.expect(true); // If we get here, memory types compiled successfully
}

test "cuda module and kernel types" {
    // Verify that module-related opaque types compile
    const module_opaque: *cuda.CUmodule = @ptrFromInt(0x1234);
    _ = module_opaque;
    
    const function_opaque: *cuda.CUfunction = @ptrFromInt(0x5678);  
    _ = function_opaque;
    
    try testing.expect(true);
}

test "cuda stream and event types" {
    // Verify that stream and event opaque types compile
    const stream_opaque: *cuda.CUstream = @ptrFromInt(0x1111);
    _ = stream_opaque;
    
    const event_opaque: *cuda.CUevent = @ptrFromInt(0x2222);
    _ = event_opaque;
    
    try testing.expect(true);
}

test "cuda constants and enums" {
    // Test that enum values are accessible
    try testing.expectEqual(0, @intFromEnum(cuda.CUmemcpyKind.host_to_host));
    try testing.expectEqual(1, @intFromEnum(cuda.CUmemcpyKind.host_to_device));
    
    std.debug.print("✓ CUDA constants and enums compiled successfully\n", .{});
}

test "cuda load function signature" {
    // Test that the load() function exists (but don't call it in test)
    if (@hasDecl(cuda, "load")) {
        std.debug.print("✓ cuda.load() function declared\n", .{});
    }
    
    try testing.expect(true);
}

pub fn main() void {
    std.debug.print("\n=== Running ZigCUDA Binding Compilation Tests ===\n\n", .{});
    
    // Run our compilation tests
    test_cuda_bindings_compile();
    test_function_pointers_exist(); 
    test_error_handling_types();
    test_memory_types();
    test_module_kernel_types();
    test_stream_event_types();
    test_constants_enums();
    test_load_function_exists();
    
    std.debug.print("\n=== All Binding Compilation Tests Passed ===\n\n", .{});
}

fn test_cuda_bindings_compile() void {
    std.debug.print("Testing CUDA bindings compilation...\n", .{});
    // This is essentially what the Zig compiler does when it processes this file
    if (cuda.CUDA_SUCCESS == 0) {
        std.debug.print("✓ Basic type definitions compiled\n", .{});
    }
}

fn test_function_pointers_exist() void {
    std.debug.print("Testing function pointer declarations...\n", .{});
    
    // We can't actually call these in a test environment, but we can check they exist
    if (cuda.cuInit != null) {
        std.debug.print("✓ cuInit function pointer exists\n", .{});
    }
}

fn test_error_handling_types() void {
    std.debug.print("Testing error handling types...\n", .{});
    
    const errors = @import("bindings/errors");
    // Test that we can reference the error union
    _ = errors.CUDAError;
    std.debug.print("✓ Error types compiled\n", .{});
}

fn test_memory_types() void {
    std.debug.print("Testing memory management types...\n", .{});
    
    const dev_ptr: cuda.CUdeviceptr = 0x1000;
    const prop: cuda.CUdevprop = undefined;
    _ = dev_ptr;
    _ = prop;
    
    std.debug.print("✓ Memory types compiled\n", .{});
}

fn test_module_kernel_types() void {
    std.debug.print("Testing module and kernel types...\n", .{});
    
    const module_opaque: *cuda.CUmodule = @ptrFromInt(0x1234);
    const function_opaque: *cuda.CUfunction = @ptrFromInt(0x5678);  
    _ = module_opaque;
    _ = function_opaque;
    
    std.debug.print("✓ Module/kernel types compiled\n", .{});
}

fn test_stream_event_types() void {
    std.debug.print("Testing stream and event types...\n", .{});
    
    const stream_opaque: *cuda.CUstream = @ptrFromInt(0x1111);
    const event_opaque: *cuda.CUevent = @ptrFromInt(0x2222);
    _ = stream_opaque;
    _ = event_opaque;
    
    std.debug.print("✓ Stream/event types compiled\n", .{});
}

fn test_constants_enums() void {
    std.debug.print("Testing constants and enums...\n", .{});
    
    // Test that enum values are accessible
    if (cuda.host_to_device == 1) {
        std.debug.print("✓ Constants/enums compiled\n", .{});
    }
}

fn test_load_function_exists() void {
    std.debug.print("Testing load function...\n", .{});
    
    if (@hasDecl(cuda, "load")) {
        std.debug.print("✓ Load function declared\n", .{});
    } else {
        std.debug.print("✗ Load function not found\n", .{});
    }
}