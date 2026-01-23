#!/bin/bash
# examples/build_kernels.sh
# Compile CUDA kernels from .cu source to .cubin binaries

set -e

echo "Building CUDA kernel binaries..."

# Compile vector_add.cu for Blackwell hardware (sm_120 Blackwell architecture)
nvcc --cubin -arch=compute_120 -code=sm_120 examples/kernels/vector_add.cu -o examples/kernels/vector_add.cubin

if [ $? -eq 0 ]; then
    echo "✓ Successfully compiled vector_add.cu → vector_add.cubin"
else
    echo "✗ Failed to compile kernel"
    exit 1
fi

echo ""
echo "=== Build Summary ==="
echo "Input:  kernels/vector_add.cu (CUDA C++ source)"
echo "Output: kernels/vector_add.cubin (Compiled binary)" 
echo "Ready for runtime loading with cuModuleLoad()"