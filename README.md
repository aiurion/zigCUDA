[![Version: v0.0.1](https://img.shields.io/badge/Version-v0.0.1-blue)](#)
[![Tests: 101/101 Passing](https://img.shields.io/badge/Tests-101%2F101_Passing-brightgreen)](#)
[![Binary Size: ~8MB](https://img.shields.io/badge/Binary_Size-%7E8MB-success)](#)

# zigCUDA - CUDA Driver API for Zig

Blackwell ready, pure Zig (0.15.2+) bindings to the NVIDIA CUDA Driver API

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
    // High-level init returns a Context helper
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    const device_count = ctx.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n", .{device_count});

    for (0..device_count) |i| {
        const props = try ctx.getDeviceProperties(@intCast(i));
        std.debug.print("Device {d}: {s} (Compute {d}.{d})\n", .{
            i, std.mem.sliceTo(&props.name, 0), props.major, props.minor
        });
    }
}
```

**Kernel launch example:**
```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    _ = try zigcuda.init();
    
    // Load compiled CUDA binary (.cubin file)
    const module = try zigcuda.loadModule("my_kernel.cubin");
    defer _ = zigcuda.unloadModule(module) catch {};
    
    const kernel_func = try zigcuda.getFunctionFromModule(module, "my_kernel");

    // Launch with correct grid/block dimensions
    const empty_params: []?*anyopaque = &.{};
    
    try zigcuda.launchKernel(kernel_func,
        1, 1, 1,    // grid_dim (x, y, z)
        32, 1, 1,   // block_dim (x, y, z)
        0,          // shared_mem_bytes
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
zig build run test      # Run full suite (97 tests)
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
