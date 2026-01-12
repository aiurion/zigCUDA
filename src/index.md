## Overview

zigCuda is a native CUDA API binding for Zig that provides type-safe interfaces to CUDA operations. The codebase is organized into four main layers: low-level bindings, core abstractions, high-level integrations, and tensor operations.

## Root Files

- [main.zig](../src/main.zig) - Application entry point with clean startup message

## Directory Structure

### bindings/ - Low-Level CUDA API Bindings

Core CUDA library interfaces and type definitions:

- [errors_stub.zig](bindings/errors_stub.zig) - Stub error types for development without CUDA
- [cuda_stub.zig](bindings/cuda_stub.zig) - Development stub when CUDA is not available
- [cublas.zig](bindings/cublas.zig) - cuBLAS API bindings with dynamic loading support
- [curand.zig](bindings/curand.zig) - cuRAND API bindings for random number generation
- [cuda.zig](bindings/cuda.zig) - Core CUDA Driver API declarations and essential bindings
- [types.zig](bindings/types.zig) - CUDA type definitions and constants
- [ffi.zig](bindings/ffi.zig) - Foreign function interface declarations
- [errors.zig](bindings/errors.zig) - Error code mappings and Zig error types

### core/ - Core CUDA Abstractions

High-level CUDA resource management and operations:

- [kernel.zig](core/kernel.zig) - Type-safe kernel launch interface
- [stream.zig](core/stream.zig) - Asynchronous stream operations management
- [event.zig](core/event.zig) - Synchronization events for GPU operations
- [memory.zig](core/memory.zig) - Memory pool allocation and management
- [device.zig](core/device.zig) - GPU device enumeration and properties
- [context.zig](core/context.zig) - CUDA context management and lifecycle
- [module.zig](core/module.zig) - PTX/CUBIN compilation and loading
- [config.zig](core/config.zig) - Runtime configuration management

### integrations/ - High-Level Integrations

Optimized third-party library integrations:

- [cublas.zig](integrations/cublas.zig) - cuBLAS integration for optimized BLAS operations
- [cublaslt.zig](integrations/cublaslt.zig) - cuBLASLt integration for custom operations
- [marlin.zig](integrations/marlin.zig) - Marlin INT4 kernels for quantized operations

### ops/ - Tensor Operations

Core tensor types and neural network operations:

- [tensor.zig](ops/tensor.zig) - Core tensor type and fundamental tensor operations
- [gemm.zig](ops/gemm.zig) - General matrix multiplication operations
- [conv.zig](ops/conv.zig) - Convolution operations for neural networks
- [attention.zig](ops/attention.zig) - Attention mechanisms (Flash, Multi-head)
- [reduce.zig](ops/reduce.zig) - Reduction operations (sum, mean, max)
- [norm.zig](ops/norm.zig) - Normalization layers (LayerNorm, RMSNorm)
- [activations.zig](ops/activations.zig) - Activation functions for neural networks
- [embedding.zig](ops/embedding.zig) - Embedding table operations

## Directory Tree View

```
src/
├── main.zig
├── bindings/
│   ├── errors_stub.zig
│   ├── cuda_stub.zig
│   ├── cublas.zig
│   ├── curand.zig
│   ├── cuda.zig
│   ├── types.zig
│   ├── ffi.zig
│   └── errors.zig
├── core/
│   ├── kernel.zig
│   ├── stream.zig
│   ├── event.zig
│   ├── memory.zig
│   ├── device.zig
│   ├── context.zig
│   ├── module.zig
│   └── config.zig
├── integrations/
│   ├── cublas.zig
│   ├── cublaslt.zig
│   └── marlin.zig
└── ops/
    ├── tensor.zig
    ├── gemm.zig
    ├── conv.zig
    ├── attention.zig
    ├── reduce.zig
    ├── norm.zig
    ├── activations.zig
    └── embedding.zig
```

## Key Dependencies and Integration Flow

**Layer 1 - Bindings**: Low-level FFI to CUDA libraries (CUDA, cuBLAS, cuRAND) with error handling

**Layer 2 - Core**: Type-safe abstractions managing GPU resources (devices, memory, streams, kernels)

**Layer 3 - Integrations**: Optimized third-party library integrations building on core primitives

**Layer 4 - Operations**: High-level tensor operations and neural network layers built on the foundation below

The architecture follows a clean dependency hierarchy where each layer depends only on the layers beneath it, enabling modular compilation and testing of individual components."