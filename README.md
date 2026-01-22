# zigCUDA - CUDA Driver API for Zig

Pure-Zig bindings to the NVIDIA CUDA Driver API. Dynamic loading of `libcuda.so`, clean low level wrappers, and stubs for non-CUDA environments. No static linking, no CUDA toolkit required at runtime. Tested on NVIDIA Blackwell (sm_120).

[![Version: v0.0.1](https://img.shields.io/badge/Version-v0.0.1-blue)](#)
[![Tests: 95/97 Passing](https://img.shields.io/badge/Tests-95%2F97_Passing-brightgreen)](#)
[![Binary Size: ~8MB](https://img.shields.io/badge/Binary_Size-%7E8MB-success)](#)

> Core driver wrapper is tested. Ready for low-level GPU programming, kernel launching, and basic BLAS operations.

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
- **Clean Zig API** – Context, device, memory, streams, events, module loading, kernel launch
- **Graceful Stubs** – Compiles and runs basic checks without a GPU
- **Zero External Dependencies** – Only needs NVIDIA driver at runtime
- **Test Coverage** – 97 passing tests across core, bindings, and integrations
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
```

### 3. Example usage

**Basic device enumeration:**
```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    try zigcuda.bindings.init();

    const device_count = try zigcuda.bindings.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n", .{device_count});

    for (0..@min(device_count, 3)) |i| {
        const props = try zigcuda.bindings.getDeviceProperties(@intCast(i));
        std.debug.print("Device {d}: {s}\n", .{
            i, @as([:0]const u8, @ptrCast(&props.name)),
        });
    }
}
```

**Kernel launch example:**
```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    try zigcuda.bindings.init();
    
    // Load compiled CUDA binary (.cubin file)
    const filename: [:0]zigcuda.bindings.@"c_char" = "my_kernel.cubin";
    const module = try zigcuda.bindings.loadModule(filename);
    
    var kernel_name_buf = "my_kernel".*;
    const c_kernel_name: [:0]zigcuda.bindings.@"c_char" = @ptrCast(&kernel_name_buf);
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


## 🏗️ Project Structure

```
src/
├── bindings/     # Raw FFI + dynamic loading (cuda.zig is core)
├── core/         # High-level wrappers (context, device, memory, stream, kernel)
├── integrations/ # cuBLAS, FlashAttention prototype, etc.
├── ops/          # Future tensor operations (currently stubs)
├── examples/     # Demo programs
└── lib.zig       # Public root (re-exports API)
```

## 🎯 What This Is vs Isn't

**This IS:**
- A solid CUDA Driver API wrapper for Zig
- Ready for writing and launching kernels, memory management, streams/events
- Usable today for low-level GPU work and experimentation

**This is NOT:**
- A full ML framework
- Complete high-level tensor ops
- Optimized inference engine (FlashAttention is prototype only)

## 🗺️ Roadmap

- **v0.0.x** – Core polish and further validation

## 🛠️ Development

```bash
zig build run test      # Run full suite (97 tests)
zig build run       # Diagnostic tool
```

**Supported Platforms:**
- Linux (x86_64) – Fully tested
- WSL2 – Working with dual-context handling

## 🤝 Contributing

Open issues for bugs/features. PRs welcome if:
- Tests pass
- Core remains dependency-free
- Changes target low-level first

## 📜 License

MIT (see LICENSE file)

---

ZigCUDA gives you real CUDA access in pure Zig with minimal overhead. The foundation is ready – start building GPU code today.
