// examples/05_cubin_launch.zig
// Load a CUBIN module and launch a vector addition kernel using core abstractions

const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    // Initialize CUDA driver
    try zigcuda.init();

    std.debug.print("=== CUDA Kernel Launch Example (CUBIN) ===\n\n", .{});

    // Initialize device and context
    var ctx = try zigcuda.Context.init(0);
    defer ctx.deinit();
    try ctx.makeCurrent();

    // Array size for vector addition
    const n: u32 = 1024;
    const byte_size = n * @sizeOf(f32);

    // Allocate and initialize host memory
    const allocator = std.heap.page_allocator;
    const h_a = try allocator.alloc(f32, n);
    defer allocator.free(h_a);
    const h_b = try allocator.alloc(f32, n);
    defer allocator.free(h_b);
    const h_c = try allocator.alloc(f32, n);
    defer allocator.free(h_c);

    // Initialize input vectors
    for (h_a, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i));
    for (h_b, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i * 2));

    std.debug.print("Input A: [{d:.1}, {d:.1}, ..., {d:.1}]\n", .{ h_a[0], h_a[1], h_a[n - 1] });
    std.debug.print("Input B: [{d:.1}, {d:.1}, ..., {d:.1}]\n\n", .{ h_b[0], h_b[1], h_b[n - 1] });

    // Allocate device memory
    const d_a = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_a) catch {};
    const d_b = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_b) catch {};
    const d_c = try zigcuda.allocDeviceMemory(byte_size);
    defer _ = zigcuda.freeDeviceMemory(d_c) catch {};

    // Copy input data to device
    try zigcuda.copyHostToDevice(d_a, std.mem.sliceAsBytes(h_a));
    try zigcuda.copyHostToDevice(d_b, std.mem.sliceAsBytes(h_b));

    std.debug.print("✓ Memory allocated and initialized\n", .{});

    // Load CUBIN module from file (production approach)
    var module = try zigcuda.Module.loadFile(allocator, "kernels/vector_add.cubin");
    defer module.unload();

    std.debug.print("✓ CUBIN module loaded from file\n", .{});

    // Get kernel function using core Kernel abstraction
    const kernel = try zigcuda.Kernel.init(&module, "vector_add");

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

    // Launch kernel using core Kernel abstraction
    try kernel.launch(.{
        .grid_size = .{ blocks, 1, 1 },
        .block_size = .{ threads_per_block, 1, 1 },
        .shared_memory = 0,
    }, &params);

    std.debug.print("✓ Kernel launched\n\n", .{});

    // Synchronize via context
    try ctx.synchronize();

    // Copy result back to host
    try zigcuda.copyDeviceToHost(std.mem.sliceAsBytes(h_c), d_c);

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
        std.debug.print("Verified: C[i] = A[i] + B[i] for all {d} elements\n\n", .{n});

        // Show compilation info
        std.debug.print("=== Production Workflow ===\n", .{});
        std.debug.print("Step 1: Compile .cu → .cubin:\n", .{});
        std.debug.print("nvcc -arch=compute_80 --gpu-code=sm_90a --cubin vector_add.cu\n\n", .{});
        std.debug.print("Step 2: Load .cubin at runtime with Module.loadFile()\n", .{});
        std.debug.print("This is the production-ready approach!\n", .{});
    } else {
        std.debug.print("✗ FAILED: {d} errors detected\n", .{errors});
        return error.VerificationFailed;
    }
}
