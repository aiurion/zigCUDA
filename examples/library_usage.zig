// examples/library_usage.zig - Example of using ZigCUDA as a library
//
// USAGE: This file demonstrates how to import and use the ZigCUDA library.
// When used in your own project, you would import it as:
//   const zigcuda = @import("zigcuda");

const std = @import("std");

pub fn main() !void {
    std.debug.print("\n=== Using ZigCUDA Library (Demo) ===\n\n", .{});

    // This is what external projects would write:
    // const zigcuda = @import("zigcuda"); 

    std.debug.print("EXTERNAL PROJECT USAGE PATTERN:\n\n", .{});
    std.debug.print("Step 1: Import the library\n", .{});
    std.debug.print("  const zigcuda = @import(\"zigcuda\");\n\n", .{});

    std.debug.print("Step 2: Initialize and use it\n", .{});
    std.debug.print("  var ctx = try zigcuda.init();\n", .{});
    std.debug.print("  defer ctx.deinit();\n\n", .{});

    std.debug.print("ZIGCUDA LIBRARY API FEATURES:\n", .{});
    std.debug.print("- Device enumeration and properties\n", .{});
    std.debug.print("- CUDA context management\n", .{}); 
    std.debug.print("- Memory allocation (host/device)\n", .{});
    std.debug.print("- Module loading and kernel launch\n", .{});
    std.debug.print("- Stream operations for async work\n", .{});
    std.debug.print("- cuBLAS integration\n", .{});
    std.debug.print("- Tensor operations (GEMM, attention)\n\n", .{});

    std.debug.print("TYPICAL USAGE PATTERN:\n\n", .{});
    std.debug.print("// In your main function:\n", .{});
    std.debug.print("const zigcuda = @import(\"zigcuda\");\n\n", .{});
    std.debug.print("pub fn main() !void {\n", .{});
    std.debug.print("  var ctx = try zigcuda.init();\n", .{});
    std.debug.print("  defer ctx.deinit();\n\n", .{});
    std.debug.print("  if (!ctx.isAvailable()) return;\n\n", .{});
    std.debug.print("  const device_count = ctx.getDeviceCount();\n\n", .{});
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

    std.debug.print("✅ Library API pattern demonstrated!\n", .{});
}