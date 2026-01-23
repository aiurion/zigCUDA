# ZigCUDA Examples

This directory contains practical examples demonstrating how to use the ZigCUDA library.

## Prerequisites

- CUDA-capable GPU with driver installed
- Zig compiler (0.15.2 or later)
- CUDA Toolkit (for PTX compilation, optional)

## Examples Overview

### 01_device_info.zig
**Basic device enumeration and querying**

Demonstrates:
- Initializing CUDA context
- Enumerating available devices
- Querying device properties (name, compute capability, memory, SM count)

```bash
zig build-exe examples/01_device_info.zig --dep zigcuda --mod zigcuda::src/lib.zig
./01_device_info
```

### 02_memory_transfer.zig
**Host ↔ Device memory operations**

Demonstrates:
- Allocating device memory
- Copying data from host to device
- Copying data from device to host
- Memory cleanup and verification

```bash
zig build-exe examples/02_memory_transfer.zig --dep zigcuda --mod zigcuda::src/lib.zig
./02_memory_transfer
```

### 03_kernel_launch.zig
**Loading and launching CUDA kernels**

Demonstrates:
- Loading PTX modules
- Extracting kernel functions
- Setting up kernel parameters
- Launching kernels with grid/block configuration
- Result verification

```bash
zig build-exe examples/03_kernel_launch.zig --dep zigcuda --mod zigcuda::src/lib.zig
./03_kernel_launch
```

### 04_streams.zig
**Asynchronous operations with CUDA streams**

Demonstrates:
- Creating multiple CUDA streams
- Launching concurrent operations
- Stream synchronization
- Querying stream status

```bash
zig build-exe examples/04_streams.zig --dep zigcuda --mod zigcuda::src/lib.zig
./04_streams
```

### 05_cubin_launch.zig
**Production-ready kernel launching from binary files**

Demonstrates:
- Loading pre-compiled .cubin files
- Handling binary modules in production environments
- Grid and block configuration for high performance
- Memory management for kernel arguments

```bash
zig build-exe examples/05_cubin_launch.zig --dep zigcuda --mod zigcuda::src/lib.zig
./05_cubin_launch
```

## Building All Examples

You can add these examples to your `build.zig` to make them easy to build:

```zig
// In build.zig, add:
const examples = [_][]const u8{
    "01_device_info",
    "02_memory_transfer",
    "03_kernel_launch",
    "04_streams",
};

for (examples) |example_name| {
    const exe = b.addExecutable(.{
        .name = example_name,
        .root_source_file = b.path(b.fmt("examples/{s}.zig", .{example_name})),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zigcuda", lib_module);
    b.installArtifact(exe);
}
```

Then build with:
```bash
zig build
```

## Kernels Directory

The `kernels/` subdirectory contains PTX (Parallel Thread Execution) code for CUDA kernels:

- **vector_add.ptx**: Vector addition kernel used by example 03
- **vector_add.cubin**: Compiled vector addition kernel used by example 05

### Compiling Your Own Kernels


For compute capability 12.0 (Blackwell):
```bash
nvcc --cubin -arch=compute_120 -code=sm_120 your_kernel.cu -o your_kernel.cubin
```

## Expected Output

### Example 01 - Device Info
```
Found 1 CUDA device(s)

Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  Compute Capability: 12.0
  Total Memory: 95.59 GB
  Multiprocessors: 188
```

### Example 02 - Memory Transfer
```
=== CUDA Memory Transfer Example ===

Allocated 4096 bytes on host
Input data: [0.0, 1.0, 2.0, ..., 1023.0]
Allocated 4096 bytes on device

Copying data from host to device...
✓ Host-to-device transfer complete

Copying data from device to host...
✓ Device-to-host transfer complete

✓ SUCCESS: All 1024 elements transferred correctly
Output data: [0.0, 1.0, 2.0, ..., 1023.0]
```

### Example 03 - Kernel Launch
```
=== CUDA Kernel Launch Example ===

Input A: [0.0, 1.0, 2.0, ..., 1023.0]
Input B: [0.0, 2.0, 4.0, ..., 2046.0]

✓ PTX module loaded
✓ Kernel function extracted

Launching kernel with 4 blocks x 256 threads
✓ Kernel launched

✓ SUCCESS: Vector addition completed correctly
Result C: [0.0, 3.0, 6.0, ..., 3069.0]
Verified: C[i] = A[i] + B[i] for all 1024 elements
```

### Example 04 - Streams
```
=== CUDA Streams Example ===

Creating 3 CUDA streams...
✓ 3 streams created

Memory allocated for 3 streams
Each stream handles 512 elements (2048 bytes)

Launching async memory copies on all streams...
Synchronizing all streams...
✓ All streams synchronized
Time elapsed: 2 ms

Querying stream status...
  Stream 0: ✓ Complete
  Stream 1: ✓ Complete
  Stream 2: ✓ Complete

✓ SUCCESS: Streams example completed
Total data transferred: 6 KB across 3 streams
```

### Example 05 - CUBIN Launch
```
=== CUDA Kernel Launch Example (CUBIN) ===

✓ Memory allocated and initialized
✓ CUBIN module loaded from file
✓ Kernel function extracted

Launching kernel with 4 blocks x 256 threads
✓ Kernel launched

✓ SUCCESS: Vector addition completed correctly
Verified: C[i] = A[i] + B[i] for all 1024 elements

=== Production Workflow ===
Step 1: Compile .cu → .cubin:
nvcc -arch=compute_80 --gpu-code=sm_90a --cubin vector_add.cu

Step 2: Load .cubin at runtime with cuModuleLoad()
This is the production-ready approach!
```

## Troubleshooting

### "CUDA driver not found"
Ensure your NVIDIA driver is properly installed:
```bash
nvidia-smi
```

### "No CUDA devices found"
Check that your GPU is recognized:
```bash
lspci | grep -i nvidia
```

### "PTX module load failed"
Ensure the PTX file exists and the path is correct. PTX files are embedded at compile time using `@embedFile()`.

## Next Steps

- Modify the examples to experiment with different configurations
- Create your own CUDA kernels
- Explore the cuBLAS integration for matrix operations
- Check the test suite in `test/` for more advanced usage patterns
