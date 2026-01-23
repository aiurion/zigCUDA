// examples/kernels/vector_add.cu
// Simple vector addition kernel
//
// Compile to PTX with:
// nvcc -ptx -arch=sm_52 vector_add.cu -o vector_add.ptx

extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    unsigned int n)
{
    // Calculate global thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform vector addition if within bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
