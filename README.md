[![Version: v0.0.1](https://img.shields.io/badge/Version-v0.0.1-blue)](#)
[![Tests: 113/113 Passing](https://img.shields.io/badge/Tests-113%2F113_Passing-brightgreen)](#)
[![Binary Size: ~8MB](https://img.shields.io/badge/Binary_Size-%7E8MB-success)](#)

# zigCUDA - CUDA Driver API for Zig

Blackwell ready, pure Zig (0.16.0+) bindings to the NVIDIA CUDA Driver API

Dynamic loading of libcuda.so, clean high-level wrappers, and graceful stubs for non-CUDA environments.

No static linking, no CUDA toolkit required at runtime.

> Tested on Blackwell (sm_120) — ready for low-level GPU programming, kernel launching, and basic BLAS in Zig.

## 🚀 Try It Now

```bash
git clone https://github.com/Aiurion/zigcuda.git && cd zigcuda
zig build run
```

Example output:

```
=== ZigCUDA CLI Diagnostic Tool ===
INFO: cuInit succeeded
✓  CUDA Driver Initialized
✓  Device Count: 1

   [GPU 0] NVIDIA RTX PRO 6000 Blackwell Workstation Edition
     ├─ Compute: 12.0
     ├─ SMs:     120
     └─ VRAM:    95.59 GB

```

## 🎯 Key Features (v0.0.1)

- **Dynamic Driver Loading** – Works on Linux native and WSL2, multiple symbol resolution paths
- **Clean Zig API** – Raw Driver API access plus low-level ergonomic wrappers for memory, params, modules, and launch
- **Graceful Stubs** – Compiles and runs basic checks without a GPU
- **Zero External Dependencies** – Only needs NVIDIA driver at runtime
- **Test Coverage** – 113 passing tests across core, bindings, ergonomics, and integrations
- **Easy Library Usage** – Single `@import("zigcuda")` with init/deinit pattern

## 📊 Status

| Component              | Status                  | Notes                                      |
|------------------------|-------------------------|--------------------------------------------|
| Driver Loading         | Complete                | Dynamic + extensive fallbacks              |
| Core API (memory, streams, contexts) | Complete           | Full wrappers, async support       |
| Kernel Launch          | Complete                | cuLaunchKernel + legacy fallback           |
| cuBLAS Integration     | Partial                 | Basic handle + common ops working           |


## 🛠️ Using in Your Project

### 1. Add dependency (`build.zig.zon`)

```zig
.dependencies = .{
    .zigcuda = .{
        .url = "git+https://github.com/Aiurion/zigcuda.git#v0.0.1",
        // Run `zig build` once to fill in hash
    },
},
```

### 2. In `build.zig`

```zig
const zigcuda_dep = b.dependency("zigcuda", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("zigcuda", zigcuda_dep.module("zigcuda"));
exe.root_module.linkSystemLibrary("c", .{});
```

### 3. Example usage

The raw Driver API wrappers remain available under `zigcuda.bindings.*`. For lower-boilerplate code, use the ergonomic layer exported from `zigcuda` directly.

### Low-level ergonomic API

```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn runKernel(allocator: std.mem.Allocator, input: []const f16, output: []f16) !void {
    var input_dev = try zigcuda.DeviceBuffer.alloc(std.mem.sliceAsBytes(input).len);
    defer input_dev.deinit();
    var output_dev = try zigcuda.DeviceBuffer.alloc(std.mem.sliceAsBytes(output).len);
    defer output_dev.deinit();

    try input_dev.copyFromTyped(f16, input);

    var module = try zigcuda.Module.loadFirst(allocator, &.{
        "build/kernels/lm_head_q6k_mmq.cubin",
        "kernels/lm_head_q6k_mmq.cubin",
    });
    defer module.deinit();

    const kernel = try module.kernel("lm_head_mmq_q6k_kernel");

    var params = zigcuda.Params.init();
    try params.devicePtr(output_dev.ptr);
    try params.devicePtr(input_dev.ptr);
    try params.value(i32, @intCast(input.len));

    try kernel.launch(.{
        .grid = .{ .x = @intCast((input.len + 255) / 256) },
        .block = .{ .x = 256 },
        .sync_after = true,
    }, params.slice());

    try output_dev.copyToTyped(f16, output);
}
```

Defaults keep common CUDA launch boilerplate out of the call site: `grid.z = 1`, `block.y = 1`, `block.z = 1`, `shared_mem_bytes = 0`, `stream = null`, and `sync_after = false`.

**Basic device enumeration:**
```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    try zigcuda.bindings.load();
    try zigcuda.bindings.init(0);

    const device_count = try zigcuda.bindings.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n", .{device_count});

    for (0..@min(device_count, 3)) |i| {
        const device = try zigcuda.bindings.getDevice(@intCast(i));
        const props = try zigcuda.bindings.getDeviceProperties(device);
        std.debug.print("Device {d}: {s}\n", .{
            i, @as([*:0]const u8, @ptrCast(&props.deviceName)),
        });
    }
}
```

**Kernel launch example:**
```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    try zigcuda.bindings.load();
    try zigcuda.bindings.init(0);
    
    // Load compiled CUDA binary (.cubin file)
    const filename: [:0]const zigcuda.bindings.c_char = @ptrCast("my_kernel.cubin");
    const module = try zigcuda.bindings.loadModule(filename);
    
    const c_kernel_name: [:0]const zigcuda.bindings.c_char = @ptrCast("my_kernel");
    const kernel_func = try zigcuda.bindings.getFunctionFromModule(module, c_kernel_name);

    // Launch with correct parameter count (grid_dim_z is required!)
    const empty_params: []?*anyopaque = &.{};
    
    try zigcuda.bindings.launchKernel(kernel_func,
        1,          // grid_dim_x
        1,          // grid_dim_y  
        1,          // FIXED: grid_dim_z (cannot be 0!)
        32,         // block_dim_x 
        1,          // block_dim_y
        1,          // block_dim_z
        0,           // shared_mem_bytes
        null,       // stream
        empty_params // kernel parameters
    );
    
    std.debug.print("Kernel launched successfully!\n", .{});
}
```

## Scope

**This IS:**
- A solid CUDA Driver API wrapper for Zig
- Ready for writing and launching kernels, memory management, streams/events
- Usable today for low-level GPU work and experimentation

**This is NOT:**
- A full ML framework
- Complete high-level tensor ops
- Optimized inference engine

## 🗺️ Roadmap

- **v0.0.x** – Core polish and further validation

## 🛠️ Development

```bash
zig build test      # Run full suite
zig build run       # Diagnostic tool
```

**Supported Platforms:**
- Linux (x86_64) – Fully tested
- WSL2 – Working with dual-context handling

## 🤝 Contributing

Open issues for bugs & in-scope features. 

## 📜 License

MIT (see LICENSE file)

---

ZigCUDA gives you real CUDA access in pure Zig with minimal overhead. The foundation is ready – start building GPU code today.
