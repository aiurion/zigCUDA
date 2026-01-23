// examples/05_cubin_launch.zig
// Load a CUBIN module and launch a vector addition kernel

const std = @import("std");
const zigcuda = @import("zigcuda");
const cuda = zigcuda.bindings;

pub fn main() !void {
    // Initialize CUDA
    _ = try zigcuda.init();

    std.debug.print("=== CUDA Kernel Launch Example (CUBIN) ===\n\n", .{});

    // Get device and create context
    const device = try cuda.getDevice(0);
    var ctx: ?*cuda.CUcontext = null;
    if (cuda.cuCtxCreate) |f| {
        const result = f(&ctx, 0, device);
        if (result != 0) return error.ContextCreationFailed;
    } else return error.CudaFunctionNotAvailable;

    defer {
        if (ctx) |c| {
            if (cuda.cuCtxDestroy) |f| _ = f(c);
        }
    }

    // Array size for vector addition
    const n: u32 = 1024;
    const byte_size = n * @sizeOf(f32);

    // Allocate and initialize host memory
    const h_a = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(h_a);
    const h_b = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(h_b);
    const h_c = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(h_c);

    // Initialize input vectors
    for (h_a, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i));
    for (h_b, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i * 2));

    std.debug.print("Input A: [{d:.1}, {d:.1}, {d:.1}, ..., {d:.1}]\n", .{ h_a[0], h_a[1], h_a[2], h_a[n - 1] });
    std.debug.print("Input B: [{d:.1}, {d:.1}, {d:.1}, ..., {d:.1}]\n\n", .{ h_b[0], h_b[1], h_b[2], h_b[n - 1] });

    // Allocate device memory
    const d_a = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_a) catch {};
    const d_b = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_b) catch {};
    const d_c = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_c) catch {};

    // Copy input data to device
    try zigcuda.copyHostToDevice(d_a, h_a.ptr, byte_size);
    try zigcuda.copyHostToDevice(d_b, h_b.ptr, byte_size);

    std.debug.print("✓ Memory allocated and initialized\n", .{});

    // Load CUBIN module from file (production approach)
    var module: ?*zigcuda.CUmodule = null;

    if (zigcuda.bindings.cuModuleLoad) |f| {
        const result = f(&module, "kernels/vector_add.cubin");
        if (result != 0 or module == null) {
            std.debug.print("Failed to load CUBIN module\n", .{});
            return error.ModuleLoadFailed;
        }
    } else return error.CudaFunctionNotAvailable;

    defer {
        if (module) |m| {
            if (zigcuda.bindings.cuModuleUnload) |f| _ = f(m);
        }
    }

    std.debug.print("✓ CUBIN module loaded from file\n", .{});

    // Get kernel function
    var kernel: ?*zigcuda.CUfunction = null;
    if (zigcuda.bindings.cuModuleGetFunction) |f| {
        const result = f(&kernel, module.?, @ptrCast("vector_add"));
        if (result != 0 or kernel == null) {
            std.debug.print("Failed to get kernel function\n", .{});
            return error.KernelGetFailed;
        }
    } else return error.CudaFunctionNotAvailable;

    std.debug.print("✓ Kernel function extracted\n\n", .{});

    // Prepare kernel parameters
    var params = [_]?*anyopaque{
        @ptrCast(&d_a),
        @ptrCast(&d_b), 
        @ptrCast(&d_c),
        @ptrCast(&n),
    };

    // Launch configuration
    const threads_per_block: u32 = 256;
    const blocks: u32 = (n + threads_per_block - 1) / threads_per_block;

    std.debug.print("Launching kernel with {d} blocks x {d} threads\n", .{ blocks, threads_per_block });

    // Launch kernel using high-level wrapper
    try zigcuda.launchKernel(
        kernel.?,
        blocks,
        1,
        1, // grid dimensions
        threads_per_block,
        1,
        1, // block dimensions
        0, // shared memory
        null, // stream
        &params,
    );

    std.debug.print("✓ Kernel launched\n\n", .{});

    // Synchronize
    if (zigcuda.bindings.cuCtxSynchronize) |f| {
        const result = f();
        if (result != 0) return error.SynchronizeFailed;
    }

    // Copy result back to host
    try zigcuda.copyDeviceToHost(h_c.ptr, d_c, byte_size);

    // Verify results
    var errors: usize = 0;
    for (h_a, h_b, h_c) |a, b, c| {
        const expected = a + b;
        if (c != expected) {
            errors += 1;
        }
    }

    if (errors == 0) {
        std.debug.print("✓ SUCCESS: Vector addition completed correctly\n", .{});
        std.debug.print("Result C: [{d:.1}, {d:.1}, {d:.1}, ..., {d:.1}]\n", .{ h_c[0], h_c[1], h_c[2], h_c[n - 1] });
        std.debug.print("Verified: C[i] = A[i] + B[i] for all {d} elements\n\n", .{n});
        
        // Show compilation info
        std.debug.print("=== Production Workflow ===\n", .{});
        std.debug.print("Step 1: Compile .cu → .cubin:\n", .{});
        std.debug.print("nvcc -arch=compute_80 --gpu-code=sm_90a --cubin vector_add.cu\n\n", .{});
        std.debug.print("Step 2: Load .cubin at runtime with cuModuleLoad()\n", .{});
        std.debug.print("This is the production-ready approach!\n", .{});
    } else {
        std.debug.print("✗ FAILED: {d} errors detected\n", .{errors});
        return error.VerificationFailed;
    }
}