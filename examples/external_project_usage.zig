// Example: How Another Project Uses ZigCUDA
//
// This demonstrates the pattern for external projects that want to use ZigCUDA as a library.
// In your own project, you would add zigcuda as a dependency and import it like this:

const std = @import("std");

pub fn main() !void {
    std.debug.print("\n=== External Project Using ZigCUDA ===\n\n", .{});

    // Step 1: Import the library (as external projects would)
    // In your project:
    // const zigcuda = @import("zigcuda");
    
    std.debug.print("Step 1: Import the library\n", .{});
    std.debug.print("  const zigcuda = @import(\"zigcuda\");\n\n", .{});

    // Step 2: Initialize and use it
    // In your project:
    // var ctx = try zigcuda.init();
    // defer ctx.deinit();

    std.debug.print("Step 2: Initialize the library\n", .{});
    std.debug.print("  var ctx = try zigcuda.init();\n", .{});
    std.debug.print("  defer ctx.deinit();\n\n", .{});

    // In your project:
    // if (!ctx.isAvailable()) {
    //     return error.NoCudaDevice;
    // }

    std.debug.print("Step 3: Use the clean API\n", .{});
    std.debug.print("  if (!ctx.isAvailable()) return;\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  const device_count = ctx.getDeviceCount();\n", .{});
    std.debug.print("  ✅ External project successfully using ZigCUDA!\n\n", .{});

    // Step 4: Use device properties
    // In your project:
    // for (0..@min(device_count, @as(usize, 2))) |i| {
    //     const props = try ctx.getDeviceProperties(@as(u32, i));
    //     
    //     std.debug.print("External Project Device {}:\n", .{i});
    //     std.debug.print("  Major: {}, Minor: {}\n", .{
    //         @as(u32, @intCast(props.major)),
    //         @as(u32, @intCast(props.minor))
    //     });
    // }

    std.debug.print("Step 4: Access device information\n", .{});
    std.debug.print("  for (0..@min(device_count, @as(usize, 2))) |i| {\n", .{});
    std.debug.print("    const props = try ctx.getDeviceProperties(@as(u32, i));\n", .{});
    std.debug.print("    // Use device info...\n", .{});
    std.debug.print("  }\n\n", .{});

    // Show the complete integration pattern
    std.debug.print("\n🎉 Complete External Project Pattern:\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("const zigcuda = @import(\"zigcuda\");\n\n", .{});
    std.debug.print("pub fn main() !void {\n", .{});
    std.debug.print("  var ctx = try zigcuda.init();\n", .{});
    std.debug.print("  defer ctx.deinit();\n\n", .{});
    std.debug.print("  if (!ctx.isAvailable()) return error.NoCudaDevice;\n\n", .{});
    std.debug.print("  const device_count = ctx.getDeviceCount();\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  for (0..device_count) |i| {\n", .{});
    std.debug.print("    const props = try ctx.getDeviceProperties(@as(u32, i));\n", .{});
    std.debug.print("    // Use device info...\n", .{});
    std.debug.print("  }\n\n", .{});
    std.debug.print("  // Now you can use:\n", .{});
    std.debug.print("  // - GPU memory allocation\n", .{});
    std.debug.print("  // - Kernel launches\n", .{});
    std.debug.print("  // - cuBLAS operations\n", .{});
    std.debug.print("  // - Tensor computations\n", .{});
    std.debug.print("}\n\n", .{});

    std.debug.print("\n🎉 External project integration complete!\n", .{});
}