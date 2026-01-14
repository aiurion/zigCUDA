# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [0.0.1] - 2026-01-13
### ✨ Initial Public Release 

### Added
- **Library Module Architecture**: Refactored project into a consumer-ready Zig library (`zigcuda`) with a clean high-level `Context` API.
- **Zero-Dependency Driver Loader**: Implemented `dlopen/dlsym` architecture to load `libcuda.so` dynamically, removing build-time dependency on the CUDA Toolkit.
- **Type-Safe Kernel Launcher**: Introduced `comptime` checks for grid and block dimensions to prevent runtime CUDA crashes.
- **Comprehensive Memory Management**: Added support for device allocation, pinned host memory, and async Host-to-Device transfers.
- **cuBLAS Integration**: Wrapped core BLAS operations (`sgemm`, `dgemm`, `sdot`, `saxpy`) via dynamic loading.
- **CLI Diagnostic Tool**: Converted `src/main.zig` into a system diagnostic tool that queries and displays connected GPU capabilities.
- **Full Test Suite**: Added 86+ integration tests covering 100% of the core runtime and cuBLAS bindings.

### Technical Details
- **CUDA Bindings**: Complete Driver API with dynamic loading, no external dependencies
- **Runtime Core**: Device enumeration, context management, memory pools
- **cuBLAS Operations**: sgemm, dgemm, sdot, saxpy, sscal, and related BLAS functions  
- **Platform Support**: Linux/WSL2 tested on NVIDIA Blackwell GPU (Compute Capability 12.0+)
- **Compiler Compatibility**: Zig 0.15.2 or later

### Testing Coverage
- Core bindings: 46/46 tests ✅
- Kernel integration: 23/23 tests ✅  
- Simple kernel tests: 3/3 tests ✅
- Core runtime: 13/13 tests ✅
- cuBLAS integration: 12/12 tests ✅
- cuBLAS functions: 2/2 tests ✅


### Changed
- Fixed 64-bit ABI alignment in `bindings/cuda.zig` to correctly query devices with >4GB VRAM (specifically targeting Blackwell 96GB/144GB cards).
- Implemented dual-context workaround in `integrations/cublas.zig` to resolve WSL2 memory operation failures.
- Updated `LICENSE` to reflect Aiurion Inc. corporate ownership.

### Known Limitations
- Tested exclusively on Linux (Ubuntu/Debian) and Windows WSL2. Native MSVC Windows support is pending.
- Current cuBLAS bindings support dense matrix operations; (no batched, no strided ops)
