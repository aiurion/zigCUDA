# ZigCUDA - How Other Projects Use It

## Quick Start for External Projects

### 1. Add as Dependency in your `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) !void {
    // Import ZigCUDA library
    const zigcuda_lib = b.createModule(.{
        .root_source_file = "path/to/zigCuda/src/lib.zig",
        .target = target,
    });

    // Your executable that uses the library
    const your_exe = b.addExecutable(.{
        .name = "your-app",
        .root_module = b.path("src/main.zig"),
    });
    
    // Link against ZigCUDA
    your_exe.root_module.addImport("zigcuda", zigcuda_lib);
}
```

### 2. Use in Your Code:

```zig
const std = @import("std");
const zigcuda = @import("zigcuda"); // Import the library

pub fn main() !void {
    // Initialize ZigCUDA
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    if (ctx.isAvailable()) {
        const device_count = ctx.getDeviceCount();
        std.debug.print("Found {d} CUDA devices\n", .{device_count});
        
        // Get device properties
        for (0..device_count) |i| {
            const props = try ctx.getDeviceProperties(@as(u32, i));
            // Use device info...
        }
    }
}
```

## What You Get

- **Initialization**: `zigcuda.init()` - Sets up CUDA and returns a context
- **Device Enumeration**: `ctx.getDeviceCount()`, `ctx.getDeviceProperties()`
- **Error Handling**: Clean error types, graceful fallback when no CUDA available
- **Type Safety**: Compile-time checked API with Zig's strong typing

## No External Dependencies Required

Unlike Python-based CUDA solutions, your users only need:
1. NVIDIA driver (already installed for most systems)  
2. Your executable + this library
3. That's it - no CUDA toolkit installation needed!

## Library vs CLI Tool

- **Library (`src/lib.zig`)**: What external projects import and use
- **CLI Tool (`src/main.zig`)**: Development/testing tool that demonstrates the library in action