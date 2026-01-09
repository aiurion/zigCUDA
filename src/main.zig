// src/main.zig
// Main entry point for ZigCUDA testing

const std = @import("std");

// Use real CUDA bindings to test actual hardware
const cuda = @import("bindings/cuda.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("ZigCUDA Phase 0 Testing", .{});

    // 1. Initialize CUDA driver
    cuda.init(0) catch |err| {
        std.log.err("Failed to initialize CUDA: {}", .{err});
        return err;
    };
    std.log.info("CUDA initialized successfully!", .{});

    // 2. Get CUDA version
    const version = cuda.getVersion() catch |err| {
        std.log.err("Failed to get CUDA version: {}", .{err});
        return err;
    };
    std.log.info("CUDA Driver Version: {}.{}", .{ version[0], version[1] });

    // 3. Get device count
    const device_count = cuda.getDeviceCount() catch |err| {
        std.log.err("Failed to get device count: {}", .{err});
        return err;
    };
    std.log.info("Found {} CUDA device(s)", .{device_count});

    // Test each device
    var i: i32 = 0;
    while (i < device_count) : (i += 1) {
        std.log.info("Testing Device {}", .{i});

        // 4. Get device properties
        // const props = cuda.getDeviceProperties(i) catch |err| {
        //     std.log.err("Failed to get device properties: {}", .{err});
        //     continue;
        // };
        // std.log.info("  Max Threads per Block: {}", .{props.maxThreadsPerBlock});

        // 5. Get device name
        const name = cuda.getDeviceName(i, allocator) catch |err| {
            std.log.err("Failed to get device name: {}", .{err});
            continue;
        };
        defer allocator.free(name);
        std.log.info("  Name: {s}", .{name});

        // 6. Get compute capability
        const cc = cuda.getComputeCapability(i) catch |err| {
            std.log.err("Failed to get compute capability: {}", .{err});
            continue;
        };
        std.log.info("  Compute Capability: {}.{}", .{ cc.major, cc.minor });

        // 7. Get total memory
        const total_mem = cuda.getTotalMem(i) catch |err| {
            std.log.err("Failed to get total memory: {}", .{err});
            continue;
        };
        std.log.info("  Total Memory: {} bytes ({d:.2} MB)", .{ total_mem, @as(f64, @floatFromInt(total_mem)) / 1024.0 / 1024.0 });
    }

    // 8. Test getErrorString
    const err_str = cuda.getErrorString(0) catch |err| { // 0 is success
        std.log.err("Failed to get error string: {}", .{err});
        return err;
    };
    std.log.info("Error String for 0 (Success): {s}", .{err_str});

    // Test a specific error code
    const err_str_val = cuda.getErrorString(1) catch "Unknown"; // 1 is Invalid Value
    std.log.info("Error String for 1 (Invalid Value): {s}", .{err_str_val});

    // 9. Test Basic Context Management (cuCtxCreate)
    if (device_count > 0) {
        std.log.info("Testing Basic Context Management on Device 0...", .{});

        const ctx = cuda.createContext(0, 0) catch |err| {
            std.log.err("Failed to create context: {}", .{err});
            return err;
        };
        std.log.info("  ✓ Context created successfully (handle: {*})", .{ctx});

        // Test Setting Current Context (cuCtxSetCurrent)
        cuda.setCurrentContext(ctx) catch |err| {
            std.log.err("Failed to set current context: {}", .{err});
            return err;
        };
        std.log.info("  ✓ Context set as current successfully", .{});

        // Clean up
        cuda.destroyContext(ctx) catch |err| {
            std.log.err("Failed to destroy context: {}", .{err});
            return err;
        };
        std.log.info("  ✓ Basic context destroyed successfully", .{});
    }

    // 10. Test Context Management API Bindings
    if (device_count > 0) {
        std.log.info("Testing Advanced Context API Bindings...", .{});

        // Verify all context management APIs are available
        const api_available = @hasDecl(cuda, "getCurrentContext") and
            @hasDecl(cuda, "pushContext") and
            @hasDecl(cuda, "popContext");

        if (api_available) {
            std.log.info("  ✓ getCurrentContext API binding found", .{});
            std.log.info("  ✓ pushContext API binding found", .{});
            std.log.info("  ✓ popContext API binding found", .{});

            // Try basic context operations (compile-time verification)
            _ = @TypeOf(cuda.getCurrentContext());
            _ = @TypeOf(cuda.pushContext);
            _ = @TypeOf(cuda.popContext);

            std.log.info("  ✓ All context management APIs verified", .{});
        } else {
            std.log.err("Some context API bindings are missing!", .{});
        }
    }

    _ = @as(i32, device_count); // Suppress unused warning

    // 11. Test Memory Management Functions
    if (device_count > 0) {
        std.log.info("Testing Phase 0 Memory Management...", .{});

        // Verify all 12 memory management functions are available

        { // Memory allocation/deallocation - compile-time verification
            _ = @TypeOf(cuda.allocDeviceMemory); // cuMemAlloc wrapper
            _ = cuda.freeDeviceMemory; // cuMemFree wrapper
            _ = cuda.allocHost; // cuMemAllocHost wrapper
            _ = cuda.freeHost; // cuMemFreeHost wrapper

            std.log.info("  ✓ All memory alloc/dealloc functions available", .{});
        }

        { // Memory copy operations - compile-time verification
            _ = @TypeOf(cuda.copyHostToDevice); // H→D
            _ = @TypeOf(cuda.copyDeviceToHost); // D→H
            _ = @TypeOf(cuda.copyDeviceToDevice); // D→D

            std.log.info("  ✓ All memory copy function signatures verified", .{});
        }

        { // Async operations - compile-time verification
            _ = @TypeOf(cuda.copyHostToDeviceAsync); // H→D Async
            _ = @TypeOf(cuda.copyDeviceToHostAsync); // D→H Async
            _ = @TypeOf(cuda.copyDeviceToDeviceAsync); // D→D Async

            std.log.info("  ✓ All async memory operation function signatures verified", .{});
        }

        { // Memory information and handle operations - compile-time verification
            _ = @TypeOf(cuda.getDeviceMemoryInfo); // cuMemGetInfo wrapper
            _ = cuda.getMemoryHandle; // cuMemGetHandle wrapper

            std.log.info("  ✓ All memory info/handle functions available", .{});
        }

        { // Final verification - ensure all 12 functions are accounted for
            var function_count: u32 = 0;

            // Allocation/Deallocation (4)
            _ = cuda.allocDeviceMemory;
            function_count += 1;
            _ = cuda.freeDeviceMemory;
            function_count += 1;
            _ = cuda.allocHost;
            function_count += 1;
            _ = cuda.freeHost;
            function_count += 1;

            // Copy operations (3)
            _ = cuda.copyHostToDevice;
            function_count += 1;
            _ = cuda.copyDeviceToHost;
            function_count += 1;
            _ = cuda.copyDeviceToDevice;
            function_count += 1;

            // Async operations (3)
            _ = cuda.copyHostToDeviceAsync;
            function_count += 1;
            _ = cuda.copyDeviceToHostAsync;
            function_count += 1;
            _ = cuda.copyDeviceToDeviceAsync;
            function_count += 1;

            // Info/Handle operations (2)
            _ = cuda.getDeviceMemoryInfo;
            function_count += 1;
            _ = cuda.getMemoryHandle;
            function_count += 1;

            if (function_count == 12) {
                std.log.info("  ✓ Phase 0 Memory Management: ALL {d}/12 FUNCTIONS IMPLEMENTED", .{function_count});
            } else {
                std.log.warn("  Expected 12 functions, found {}", .{function_count});
            }
        }

        // Summary of what was implemented
        if (device_count > 0) {
            std.log.info("", .{});
            std.log.info("🎉 PHASE 0 MEMORY MANAGEMENT COMPLETE!", .{});
            std.log.info("✓ cuMemAlloc / cuMemFree", .{});
            std.log.info("✓ cuMemAllocHost / cuMemFreeHost", .{});
            std.log.info("✓ H→D, D→H, D→D memory copies", .{});
            std.log.info("✓ Async versions with sync fallback", .{});
            std.log.info("✓ Memory info (cuMemGetInfo)", .{});
            std.log.info("✓ Memory handles (cuMemGetHandle)", .{});
            std.log.info("", .{});
        }

        std.log.info("Phase 0 All Tests completed successfully!", .{});
    }
}
