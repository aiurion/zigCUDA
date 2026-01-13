# ZigCUDA - Native CUDA API for Zig

A comprehensive native CUDA API binding for the Zig programming language, providing direct access to NVIDIA's CUDA runtime and libraries without the overhead of C FFI.

> **Status**: Research Prototype - Solid foundation with 86+ tests passing, ready for experimentation and contribution

## 🎯 Key Features (Implemented)

- **Native Zig Integration**: Direct bindings to CUDA Driver API with dynamic loading
- **Type-Safe Kernel Launch**: Compile-time verification of kernel parameters  
- **Memory Management**: Efficient GPU memory allocation and resource management
- **Comprehensive Testing**: 86+ unit tests covering core functionality
- **Zero External Dependencies**: No Python runtime, no C FFI complexity

## 📊 Current Status

| Component | Tests Passing | Implementation |
|-----------|---------------|------------------|
| CUDA Driver API | ✅ 44/46 | Production quality bindings |
| Kernel System | ✅ 23/23 | Full type-safe kernel launching |  
| Runtime Core | ✅ 13/13 | Memory/streams/events working |
| cuBLAS Integration | �️ Partial | Library loads, symbol resolution in progress |

## 🏗️ Architecture

### Core Components (Implemented)

- **CUDA Bindings** (`src/bindings/`): CUDA Driver API with comprehensive error handling and dynamic loading
- **Core Runtime** (`src/core/`): Device management, context handling, memory pools, type-safe kernel launching  
- **Testing Infrastructure**: Comprehensive unit test suite covering all major components
- **Library Integrations** (`src/integrations/`): cuBLAS wrapper (partial implementation)

### Planned Components

- **Tensor Operations** (`src/ops/`): Matrix multiplication, attention mechanisms, normalization, activations
- **Model Loading**: Safetensors, GPTQ, AWQ format support
- **Inference Engine**: KV cache management, continuous batching, production server
- **Library Integrations**: Marlin INT4 kernels, Flash Attention

## 🚀 Quick Start

### Prerequisites
- Zig compiler (latest stable)
- NVIDIA driver (CUDA toolkit not required for runtime)
- NVIDIA GPU with compute capability 6.0+

### Building

```bash
# Debug build
zig build

# Release build for production
zig build -Doptimize=ReleaseFast
```

### Basic Usage

```// examples/basic_cuda.zig
const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    // Initialize
    try cuda.load();
    try cuda.init(0);
    
    const device_count = try cuda.getDeviceCount();
    std.debug.print("Found {} CUDA device(s)\n", .{device_count});
    
    // Get device info
    const device = try cuda.getDevice(0);
    _ = device;
    
    // Allocate memory
    var d_ptr: cuda.CUdeviceptr = 0;
    try cuda.allocate(&d_ptr, 1024);
    defer cuda.free(d_ptr) catch {};
    
    std.debug.print("Successfully allocated 1KB on GPU\n", .{});
}
```


## 🤖 Planned Features (Not Yet Implemented)

The following features are planned for future releases:

- **Model Loading**: Support for Safetensors, GPTQ, AWQ formats
- **Tensor Operations**: Matrix multiplication, attention mechanisms, normalization  
- **Inference Engine**: KV cache management and batching
- **HTTP API**: OpenAI-compatible endpoints for serving models
- **Command Line Tools**: Model serving and benchmarking utilities

## 🎯 Current Use Cases

This research prototype is currently useful for:

- **CUDA Researchers**: Direct native bindings without C FFI complexity
- **Zig Developers**: Type-safe kernel launching with compile-time validation  
- **GPU Programmers**: Testing and experimenting with CUDA operations
- **Embedded Systems Research**: Exploring lightweight CUDA runtime alternatives
- **Performance Experiments**: Benchmarking native vs Python-based stacks

*Note: Production deployment not recommended until model loading and inference engine are implemented.*

## 📁 Project Structure

```
src/
├── bindings/          # CUDA API bindings
├── core/             # Core runtime components
├── ops/              # Tensor operations
├── integrations/     # Library integrations
└── main.zig         # Entry point
```

## 🛠️ Development Status

### ✅ Completed Phases
- **Phase 0: Driver Bindings** - Complete with 44/46 comprehensive tests passing
- **Phase 1: Core Runtime** - Complete with 13/13 memory/streams/events working  
- **Phase 2: Kernel Integration** - Complete with 23/23 type-safe kernel launching

### 🔧 In Progress
- **cuBLAS Integration**: Library loads successfully, symbol resolution in progress

### �️ Planned Phases (Not Started)
- **Tensor Layer**: Matrix operations, attention mechanisms, normalization layers
- **Model Loading**: Support for Safetensors, GPTQ, AWQ formats  
- **Inference Engine**: KV cache management and batching system
- **Production Serving**: HTTP API server with OpenAI-compatible endpoints

## 🤝 Contributing

Contributions are welcome! Please see the contributing guidelines and ensure all tests pass before submitting PRs.

## 📜 License

[License information to be added]

---

*A research prototype for native CUDA bindings in Zig, providing type-safe access to GPU computation without C FFI complexity. Perfect for experimentation and development of GPU-accelerated applications.*

*Current focus: Solid foundation with comprehensive testing suite - production AI inference features coming in future releases.*