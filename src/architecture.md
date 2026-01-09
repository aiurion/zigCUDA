# ZigCUDA Project Architecture

## Directory Structure Overview

```
zigcuda/
├── src/
│   ├── bindings/          # Low-level CUDA API bindings
│   ├── core/             # Core runtime components
│   ├── ops/             # Tensor operations and computations
│   ├── integrations/      # External library integrations
│   ├── model/            # Model loading and parsing
│   ├── inference/       # Inference engine components
│   ├── server/            # HTTP server and API
│   ├── utils/              # Utility functions and helpers
│   ├── examples/           # Usage examples and demos
│   ├── tests/             # Test suites
│   └── init.md           # Project overview
├── build/
│   ├── zig-out/           # Build output directory
│   └── cache/            # Build cache
├── docs/                 # Documentation
├── scripts/              # Build and deployment scripts
├── tools/                 # Development tools
└── README.md             # Project README
```

## Detailed Module Architecture

### 1. Bindings Module (`src/bindings/`)

**Purpose**: Low-level CUDA API bindings with type safety and error handling

**Files**:
- `cuda.zig` - Core CUDA Driver API declarations
- `cublas.zig` - cuBLAS API bindings  
- `curand.zig` - cuRAND API bindings
- `types.zig` - CUDA type definitions and constants
- `errors.zig` - Error code mappings and Zig error types
- `ffi.zig` - Foreign function interface declarations

**Design Principles**:
- Minimal wrapper overhead
- Type-safe parameter passing
- Zig error handling instead of CUDA error codes
- Compile-time constant validation where possible

**Dependencies**: None (lowest level)

---

### 2. Core Module (`src/core/`)

**Purpose**: Fundamental runtime components and abstractions

**Files**:
- `device.zig` - GPU device enumeration and properties
- `context.zig` - CUDA context management and lifecycle
- `stream.zig` - Asynchronous stream operations
- `memory.zig` - Memory pool allocation and management
- `module.zig` - PTX/CUBIN compilation and loading
- `kernel.zig` - Type-safe kernel launch interface
- `event.zig` - Synchronization events
- `config.zig` - Runtime configuration

**Key Abstractions**:
```zig
pub const Device = struct {
    handle: CUdevice,
    properties: DeviceProperties,
};

pub const Context = struct {
    handle: CUcontext,
    device: Device,
    memory_pool: ?*MemoryPool,
};

pub const Stream = struct {
    handle: CUstream,
    context: *Context,
    // Async operations
};

pub const MemoryPool = struct {
    allocator: Allocator,
    device: Device,
    // Pool-based allocation
};
```

**Dependencies**: `bindings/`

---

### 3. Operations Module (`src/ops/`)

**Purpose**: High-level tensor operations and mathematical computations

**Files**:
- `tensor.zig` - Core tensor type and operations
- `gemm.zig` - General matrix multiplication
- `attention.zig` - Attention mechanisms (Flash, Multi-head)
- `norm.zig` - Normalization layers (LayerNorm, RMSNorm)
- `activations.zig` - Activation functions
- `reduce.zig` - Reduction operations (sum, mean, max)
- `conv.zig` - Convolution operations
- `embedding.zig` - Embedding table operations

**Tensor Type System**:
```zig
pub const Tensor = struct {
    data: DeviceMemory,
    shape: []const usize,
    dtype: DataType,
    device: Device,
    
    // Operations return new tensors
    pub fn matmul(self: Tensor, other: Tensor) !Tensor
    pub fn attention(self: Tensor, query: Tensor, key: Tensor, value: Tensor) !Tensor
    pub fn norm(self: Tensor, weight: Tensor, bias: Tensor) !Tensor
};
```

**Dependencies**: `core/`, `bindings/`

---

### 4. Integrations Module (`src/integrations/`)

**Purpose**: Wrappers for external high-performance libraries

**Files**:
- `cublas.zig` - cuBLAS integration for optimized BLAS operations
- `cublaslt.zig` - cuBLASLt for custom operations
- `marlin.zig` - Marlin INT4 kernels for quantized operations
- `flash_attn.zig` - Flash Attention integration
- `tensorrt.zig` - TensorRT integration (future)
- `cudnn.zig` - cuDNN integration (future)

**Integration Pattern**:
```zig
pub const MarlinGemm = struct {
    handle: CublasLtMatDesc,
    
    pub fn init(device: Device) !MarlinGemm
    pub fn gemm_int4(self: *MarlinGemm, a: Tensor, b: Tensor) !Tensor
    pub fn deinit(self: *MarlinGemm) void
};
```

**Dependencies**: `ops/`, `core/`, `bindings/`

---

### 5. Model Module (`src/model/`)

**Purpose**: Model loading, parsing, and format support

**Files**:
- `safetensors.zig` - Safetensors format parser
- `gptq.zig` - GPTQ quantized model loader
- `awq.zig` - AWQ quantized model loader
- `llama.zig` - L Llama architecture implementation
- `config.zig` - Model configuration parsing
- `weights.zig` - Weight loading and dequantization
- `tokenizer.zig` - Tokenizer implementations

**Model Loading Flow**:
```zig
pub const Model = struct {
    config: ModelConfig,
    weights: WeightLoader,
    architecture: Architecture,
    
    pub fn load(path: []const u8) !Model
    pub fn infer(self: *Model, input: Tensor) !Tensor
};
```

**Dependencies**: `ops/`, `core/`, `integrations/`

---

### 6. Inference Module (`src/inference/`)

**Purpose**: High-level inference orchestration and optimization

**Files**:
- `kv_cache.zig` - Key-value cache for transformer models
- `scheduler.zig` - Continuous batching and request scheduling
- `engine.zig` - Main inference engine
- `prefill.zig` - Prefill phase operations
- `decode.zig` - Decoding phase operations
- `batching.zig` - Dynamic batching logic

**Engine Architecture**:
```zig
pub const InferenceEngine = struct {
    model: *Model,
    kv_cache: KVCache,
    scheduler: BatchScheduler,
    device: Device,
    
    pub fn add_request(self: *InferenceEngine, input: []u32) !RequestId
    pub fn step(self: *InferenceEngine) !void
    pub fn get_response(self: *InferenceEngine, id: RequestId) ?Response
};
```

**Dependencies**: `model/`, `ops/`, `core/`

---

### 7. Server Module (`src/server/`)

**Purpose**: HTTP server and API endpoints

**Files**:
- `http.zig` - HTTP server implementation
- `api.zig` - OpenAI-compatible API routes
- `cli.zig` - Command-line interface
- `websocket.zig` - WebSocket support for streaming
- `middleware.zig` - Request/response middleware
- `config.zig` - Server configuration

**Server Architecture**:
```zig
pub const Server = struct {
    engine: *InferenceEngine,
    http_server: HttpServer,
    api_routes: ApiRouter,
    
    pub fn start(self: *Server, addr: []const u8, port: u16) !void
    pub fn stop(self: *Server) void
};
```

**Dependencies**: `inference/`, `core/`

---

### 8. Utils Module (`src/utils/`)

**Purpose**: Shared utilities and helper functions

**Files**:
- `alloc.zig` - Custom allocators and memory utilities
- `math.zig` - Mathematical utilities
- `io.zig` - Input/output utilities
- `log.zig` - Logging infrastructure
- `perf.zig` - Performance monitoring
- `debug.zig` - Debugging utilities

---

### 9. Examples Module (`src/examples/`)

**Purpose**: Usage examples and demonstrations

**Files**:
- `basic_tensor.zig` - Basic tensor operations
- `matrix_mult.zig` - Matrix multiplication examples
- `model_loading.zig` - Model loading examples
- `custom_kernel.zig` - Custom CUDA kernel launching
- `inference_server.zig` - Running inference server
- `benchmark.zig` - Performance benchmarking

---

### 10. Tests Module (`src/tests/`)

**Purpose**: Test suites and validation

**Files**:
- `bindings_test.zig` - CUDA bindings tests
- `core_test.zig` - Core runtime tests
- `ops_test.zig` - Tensor operations tests
- `model_test.zig` - Model loading tests
- `integration_test.zig` - End-to-end integration tests
- `perf_test.zig` - Performance regression tests

---

## Module Dependencies

```
bindings (no deps)
    ↓
core
    ↓
ops
    ↓
integrations
    ↓
model
    ↓
inference
    ↓
server

utils (independent)
examples (depends on all)
tests (depends on all)
```

## Design Patterns

### 1. Resource Management
- RAII pattern with Zig's `defer` and cleanup functions
- Pool-based allocation to reduce fragmentation
- Automatic memory cleanup on scope exit

### 2. Error Handling
- Zig error unions instead of exception handling
- Specific error types for different failure modes
- Comprehensive error propagation

### 3. Type Safety
- Compile-time shape checking where possible
- Type-safe kernel parameter passing
- Strong typing for device/host memory

### 4. Performance
- Zero-copy operations where possible
- Asynchronous operation support
- Pool-based memory allocation
- Custom allocators for specific use cases

### 5. Modularity
- Clear separation of concerns
- Minimal dependencies between modules
- Feature flags for optional components

## Build System Integration

### Zig Build System
The project uses Zig's native build system with:
- Incremental compilation
- Cross-compilation support
- Test integration
- Binary size optimization

### Compiler Flags
```zig
// Release builds
optimize: .ReleaseFast
// Debug builds  
optimize: .Debug
// Size optimization
optimize: .ReleaseSmall
```

### Target Platforms
- x86_64-linux-gnu
- x86_64-windows-msvc
- x86_64-macos
- aarch64-linux-gnu (future)

## Memory Architecture

### Device Memory Management
```
Application Request
    ↓
Memory Pool Allocator
    ↓
CUDA Malloc/Mfree
    ↓
GPU Device Memory
```

### Host Memory Management
```
Zig Allocator
    ↓
Custom Allocators (Arena, Pool)
    ↓
System Allocator
    ↓
Host RAM
```

## Performance Considerations

### Critical Path Optimization
1. **Model Loading**: Lazy loading, memory mapping
2. **Inference**: Continuous batching, KV cache reuse
3. **Memory**: Pool allocation, minimal copies
4. **Kernels**: Type-safe launch, optimal grid/block sizes

### Memory Layout
- **Row-major** for general tensors
- **Block-packed** for INT4 quantized weights
- **Cache-friendly** strides for attention operations

This architecture provides a clean separation of concerns while maintaining high performance and type safety throughout the system.