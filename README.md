# ZigCUDA - CUDA Driver API for Zig

Pure-Zig bindings to the NVIDIA CUDA Driver API. Dynamic loading of `libcuda.so`, clean high-level wrappers, and stubs for non-CUDA environments. No static linking, no CUDA toolkit required at runtime.

[![Version: v0.0.1](https://img.shields.io/badge/Version-v0.0.1-blue)](#)
[![Tests: 95/97 Passing](https://img.shields.io/badge/Tests-97%2F97_Passing-brightgreen)](#)
[![Binary Size: ~8MB](https://img.shields.io/badge/Binary_Size-%7E8MB-success)](#)

> Core driver wrapper is stable and well-tested. Ready for low-level GPU programming, kernel launching, and basic BLAS operations.

## 🚀 Try It Now

```bash
git clone https://github.com/Aiurion/zigcuda.git && cd zigcuda
zig build run
```

Example output (on real hardware):
```
=== Using ZigCUDA Library ===

Found 1 CUDA device(s)

Device 0:
  Name: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9
  Total Memory: 24217 MB

✅ Library ready for use!
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
| FlashAttention Prototype | Early                 | Hardware detection + cuBLAS fallback       |


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

```zig
const std = @import("std");
const zigcuda = @import("zigcuda");

pub fn main() !void {
    var ctx = try zigcuda.init();
    defer ctx.deinit();

    if (!ctx.isAvailable()) {
        std.debug.print("No CUDA available\n", .{});
        return;
    }

    const device_count = ctx.getDeviceCount();
    std.debug.print("Found {d} CUDA device(s)\n", .{device_count});

    for (0..@min(device_count, 3)) |i| {
        const props = try ctx.getDeviceProperties(@intCast(i));
        const name_end = std.mem.indexOf(u8, &props.deviceName, &[_]u8{0}) orelse props.deviceName.len;
        const name = props.deviceName[0..name_end];
        std.debug.print("Device {d}: {s} (CC {d}.{d})\n", .{
            i, name, props.major, props.minor,
        });
    }
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

- **v0.0.x** – Core polish, more tests, Windows support
- **v0.1.0** – Stable API, basic tensor abstraction, expanded cuBLAS/cuRAND
- **Later** – Optimized kernels (full FlashAttention, Marlin), model loading, inference primitives

## 🛠️ Development

```bash
zig build test      # Run full suite (97 tests)
zig build run       # Diagnostic tool
zig build           # Production binary
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