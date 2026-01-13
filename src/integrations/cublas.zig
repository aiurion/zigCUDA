// cuBLAS integration for optimized BLAS operations

const std = @import("std");
const cublas_bindings = @import("cublas_bindings");
const cuda = @import("cuda");

pub const CUBLAS_POINTER_MODE_HOST = 0;
pub const CUBLAS_POINTER_MODE_DEVICE = 1;

pub const CublasHandle = ?*anyopaque;

/// cuBLAS error types mapped from cublasStatus_t
pub const CUBLASError = error{
    NotInitialized,
    AllocationFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    InsufficientResources,
    UnknownCUBLASError,
    CudaError,
};

fn checkCublasError(status: cublas_bindings.cublasStatus_t) !void {
    switch (status) {
        .success => return,
        .not_initialized => return error.NotInitialized,
        .allocation_failed => return error.AllocationFailed,
        .invalid_value => return error.InvalidValue,
        .arch_mismatch => return error.ArchMismatch,
        .mapping_error => return error.MappingError,
        .execution_failed => return error.ExecutionFailed,
        .internal_error => return error.InternalError,
        .insufficient_resources => return error.InsufficientResources,
    }
}

fn checkCudaError(result: i32) !void {
    if (result != 0) return error.CudaError;
}

// Actual cuBLAS implementation
pub const Cublas = struct {
    handle: CublasHandle,
    device: cuda.CUdevice,
    primary_context: ?*cuda.CUcontext,  // For cuBLAS operations
    memory_context: ?*cuda.CUcontext,   // For memory operations (WSL workaround)

    pub fn init() !Cublas {
        // Load CUDA library
        cuda.load() catch {};

        // Initialize CUDA runtime
        if (cuda.cuInit) |f| {
            const init_res = f(0);
            if (init_res != 0 and init_res != 3) return error.CudaError;
        } else {
            return error.CudaError;
        }

        // Get device
        const count = try cuda.getDeviceCount();
        if (count == 0) return error.CudaError;

        const dev = try cuda.getDevice(0);
        var ctx: ?*cuda.CUcontext = null;

        // First, try to set flags (ignore error if already active)
        if (cuda.cuDevicePrimaryCtxSetFlags) |f| {
            const flags_result = f(dev, 0);
            // Ignore error 708 (already active)
            _ = flags_result;
        }

        // Retain primary context - cuBLAS requires this
        if (cuda.cuDevicePrimaryCtxRetain) |f| {
            const ctx_result = f(&ctx, dev);
            if (ctx_result != 0 or ctx == null) {
                std.debug.print("ERROR: Failed to retain primary context\n", .{});
                return error.CudaError;
            }
        } else {
            return error.CudaError;
        }

        // Set as current
        if (cuda.cuCtxSetCurrent) |f| {
            const set_result = f(ctx.?);
            if (set_result != 0) {
                std.debug.print("ERROR: Failed to set context current\n", .{});
                return error.CudaError;
            }
        }

        // Load cuBLAS library
        cublas_bindings.load() catch {
            return error.CudaError;
        };

        // Create cuBLAS handle with primary context
        var handle: cublas_bindings.cublasHandle_t = null;

        const status = cublas_bindings.cublasCreate(&handle);
        std.debug.print("DEBUG: cublasCreate result: {}\n", .{status});

        if (status != .success or handle == null) {
            return error.CudaError;
        }

        // Set cuBLAS pointer mode
        const pm_status = cublas_bindings.cublasSetPointerMode(handle.?, cublas_bindings.CUBLAS_POINTER_MODE_HOST);
        std.debug.print("DEBUG: cublasSetPointerMode result: {}\n", .{pm_status});

        if (pm_status != .success) {
            _ = cublas_bindings.cublasDestroy(handle.?);
            return error.CudaError;
        }

        // HYBRID CONTEXT SOLUTION:
        // Primary context works for cuBLAS but not for memory operations in WSL
        // Create a regular context for memory operations
        var memory_ctx: ?*cuda.CUcontext = null;
        if (cuda.cuCtxCreate) |f| {
            const mem_ctx_result = f(&memory_ctx, 0, dev);
            std.debug.print("DEBUG: Created memory context: result={}, ctx={?}\n", .{mem_ctx_result, memory_ctx});
            if (mem_ctx_result != 0 or memory_ctx == null) {
                std.debug.print("ERROR: Failed to create memory context\n", .{});
                _ = cublas_bindings.cublasDestroy(handle.?);
                return error.CudaError;
            }
        } else {
            _ = cublas_bindings.cublasDestroy(handle.?);
            return error.CudaError;
        }

        const cublas_instance = Cublas{
            .handle = handle,
            .device = dev,
            .primary_context = ctx,
            .memory_context = memory_ctx,
        };

        std.debug.print("DEBUG: cuBLAS instance created with hybrid contexts\n", .{});

        return cublas_instance;
    }

    pub fn deinit(self: *Cublas) !void {
        // Destroy cuBLAS handle first
        if (self.handle) |h| {
            const status = cublas_bindings.cublasDestroy(h);
            try checkCublasError(status);
        }

        // Destroy the memory context
        if (self.memory_context) |mem_ctx| {
            if (cuda.cuCtxDestroy) |f| {
                const destroy_result = f(mem_ctx);
                try checkCudaError(destroy_result);
            }
        }

        // NOTE: We do NOT release the primary context - it persists for program lifetime
        // This is correct behavior for primary contexts which are shared/reference-counted
    }

    // Switch to memory context for CUDA operations (WSL hybrid context workaround)
    fn setMemoryContext(self: *Cublas) !void {
        if (self.memory_context) |ctx| {
            if (cuda.cuCtxSetCurrent) |f| {
                const result = f(ctx);
                if (result != 0) {
                    std.debug.print("ERROR: Failed to set memory context current: {}\n", .{result});
                    return error.CudaError;
                }
            }
        }
    }

    // Switch back to primary context for cuBLAS operations
    fn setPrimaryContext(self: *Cublas) !void {
        if (self.primary_context) |ctx| {
            if (cuda.cuCtxSetCurrent) |f| {
                const result = f(ctx);
                if (result != 0) {
                    std.debug.print("ERROR: Failed to set primary context current: {}\n", .{result});
                    return error.CudaError;
                }
            }
        }
    }

    pub fn sgemm(
        self: *Cublas,
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: []const f32, lda: usize,
        b: []const f32, ldb: usize,
        beta: f32,
        c: []f32, ldc: usize
    ) !void {
        std.debug.print("DEBUG sgemm: Starting with m={}, n={}, k={}\n", .{m, n, k});

        // Switch to memory context for CUDA memory operations
        try self.setMemoryContext();

        // Convert row-major to column-major by swapping matrices
        // Row-major: C(m×n) = A(m×k) × B(k×n)
        // Column-major: C^T(n×m) = B^T(n×k) × A^T(k×m)
        const m_i = @as(c_int, @intCast(n)); // Swap m and n for column-major
        const n_i = @as(c_int, @intCast(m));
        const k_i = @as(c_int, @intCast(k));
        const lda_i = @as(c_int, @intCast(ldb)); // Swap lda and ldb
        const ldb_i = @as(c_int, @intCast(lda));
        const ldc_i = @as(c_int, @intCast(ldc));

        // Allocate device memory
        var d_a: cuda.CUdeviceptr = 0;
        var d_b: cuda.CUdeviceptr = 0;
        var d_c: cuda.CUdeviceptr = 0;

        const size_a = a.len * @sizeOf(f32);
        const size_b = b.len * @sizeOf(f32);
        const size_c = c.len * @sizeOf(f32);

        std.debug.print("DEBUG sgemm: Allocating memory - A:{} bytes, B:{} bytes, C:{} bytes\n", .{size_a, size_b, size_c});

        // Verify current context before allocation
        var current_ctx: ?*cuda.CUcontext = null;
        if (cuda.cuCtxGetCurrent) |f| {
            const get_result = f(&current_ctx);
            std.debug.print("DEBUG sgemm: cuCtxGetCurrent result={}, ctx={?}\n", .{get_result, current_ctx});
        }

        // Allocate and copy A
        if (cuda.cuMemAlloc) |f| {
            const alloc_result = f(&d_a, size_a);
            std.debug.print("DEBUG sgemm: cuMemAlloc A result: {}, ptr: {}\n", .{alloc_result, d_a});
            try checkCudaError(alloc_result);
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_a);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            const copy_result = f(d_a, a.ptr, size_a);
            std.debug.print("DEBUG sgemm: cuMemcpyHtoD A result: {}\n", .{copy_result});
            try checkCudaError(copy_result);
        } else return error.CudaError;

        // Allocate and copy B
        if (cuda.cuMemAlloc) |f| {
            const alloc_result = f(&d_b, size_b);
            std.debug.print("DEBUG sgemm: cuMemAlloc B result: {}, ptr: {}\n", .{alloc_result, d_b});
            try checkCudaError(alloc_result);
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_b);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            const copy_result = f(d_b, b.ptr, size_b);
            std.debug.print("DEBUG sgemm: cuMemcpyHtoD B result: {}\n", .{copy_result});
            try checkCudaError(copy_result);
        } else return error.CudaError;

        // Allocate and copy C
        if (cuda.cuMemAlloc) |f| {
            const alloc_result = f(&d_c, size_c);
            std.debug.print("DEBUG sgemm: cuMemAlloc C result: {}, ptr: {}\n", .{alloc_result, d_c});
            try checkCudaError(alloc_result);
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_c);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            const copy_result = f(d_c, c.ptr, size_c);
            std.debug.print("DEBUG sgemm: cuMemcpyHtoD C result: {}\n", .{copy_result});
            try checkCudaError(copy_result);
        } else return error.CudaError;

        defer {
            if (cuda.cuMemFree) |f| {
                _ = f(d_c);
                _ = f(d_b);
                _ = f(d_a);
            }
        }

        // Switch to primary context for cuBLAS operation
        try self.setPrimaryContext();

        // Call cuBLAS sgemm with swapped matrices for row-major to column-major conversion
        const trans_a: c_int = 0; // CUBLAS_OP_N
        const trans_b: c_int = 0; // CUBLAS_OP_N

        std.debug.print("DEBUG sgemm: m={}, n={}, k={}, lda={}, ldb={}, ldc={}\n", .{m_i, n_i, k_i, lda_i, ldb_i, ldc_i});
        std.debug.print("DEBUG sgemm: d_a={}, d_b={}, d_c={}\n", .{d_a, d_b, d_c});

        const status = cublas_bindings.cublasSgemm(
            self.handle,
            trans_a, trans_b,
            m_i, n_i, k_i,
            &alpha,
            @as([*]const f32, @ptrFromInt(d_b)), lda_i, // Swap B and A
            @as([*]const f32, @ptrFromInt(d_a)), ldb_i,
            &beta,
            @as([*]f32, @ptrFromInt(d_c)), ldc_i
        );
        std.debug.print("DEBUG sgemm returned status: {}\n", .{@intFromEnum(status)});
        try checkCublasError(status);

        // Switch back to memory context for copy operation
        try self.setMemoryContext();

        // Copy result back to host
        if (cuda.cuMemcpyDtoH) |f| {
            try checkCudaError(f(c.ptr, d_c, size_c));
        } else return error.CudaError;
    }
    
    pub fn dgemm(
        _: *Cublas,
        m: usize, n: usize, k: usize,
        alpha: f64,
        a: []const f64, lda: usize,
        b: []const f64, ldb: usize, 
        beta: f64,
        c: []f64, ldc: usize
    ) !void {
        _ = m; _ = n; _ = k; _ = alpha; _ = a; _ = lda;
        _ = b; _ = ldb; _ = beta; _ = c; _ = ldc;
        // Stub implementation - no-op  
    }
    
    pub fn sgemv(
        self: *Cublas,
        trans_a: bool, m: usize, n: usize,
        alpha: f32,
        a: []const f32, lda: usize,
        x: []const f32, incx: c_int,
        beta: f32,
        y: []f32, incy: c_int
    ) !void {
        // Switch to memory context for allocations
        try self.setMemoryContext();

        // For row-major to column-major conversion:
        // Row-major no-transpose becomes column-major transpose
        const n_i = @as(c_int, @intCast(m)); // Swap m and n
        const m_i = @as(c_int, @intCast(n));
        const lda_i = @as(c_int, @intCast(lda));
        const trans_i: c_int = if (trans_a) 0 else 1; // Flip transpose flag: 0=CUBLAS_OP_N, 1=CUBLAS_OP_T

        var d_a: cuda.CUdeviceptr = 0;
        var d_x: cuda.CUdeviceptr = 0;
        var d_y: cuda.CUdeviceptr = 0;

        const size_a = a.len * @sizeOf(f32);
        const size_x = x.len * @sizeOf(f32);
        const size_y = y.len * @sizeOf(f32);

        // Allocate and copy A
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_a, size_a));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_a);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_a, a.ptr, size_a));
        } else return error.CudaError;

        // Allocate and copy X
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_x, size_x));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_x);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_x, x.ptr, size_x));
        } else return error.CudaError;

        // Allocate and copy Y
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_y, size_y));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_y);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_y, y.ptr, size_y));
        } else return error.CudaError;

        defer {
            if (cuda.cuMemFree) |f| {
                _ = f(d_y);
                _ = f(d_x);
                _ = f(d_a);
            }
        }

        // Switch to primary context for cuBLAS operation
        try self.setPrimaryContext();

        // Call cuBLAS sgemv
        const status = cublas_bindings.cublasSgemv(
            self.handle,
            trans_i,
            m_i, n_i,
            &alpha,
            @as([*]const f32, @ptrFromInt(d_a)), lda_i,
            @as([*]const f32, @ptrFromInt(d_x)), incx,
            &beta,
            @as([*]f32, @ptrFromInt(d_y)), incy
        );
        try checkCublasError(status);

        // Switch back to memory context for copy
        try self.setMemoryContext();

        // Copy result back to host
        if (cuda.cuMemcpyDtoH) |f| {
            try checkCudaError(f(y.ptr, d_y, size_y));
        } else return error.CudaError;
    }
    
    pub fn sdot(
        self: *Cublas,
        n: usize,
        x: []const f32, incx: c_int,
        y: []const f32, incy: c_int
    ) !f32 {
        try self.setMemoryContext();

        const n_i = @as(c_int, @intCast(n));

        // Allocate device memory
        var d_x: cuda.CUdeviceptr = 0;
        var d_y: cuda.CUdeviceptr = 0;
        var result: f32 = 0.0;

        const size_x = x.len * @sizeOf(f32);
        const size_y = y.len * @sizeOf(f32);

        // Allocate and copy X
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_x, size_x));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_x);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_x, x.ptr, size_x));
        } else return error.CudaError;

        // Allocate and copy Y
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_y, size_y));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_y);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_y, y.ptr, size_y));
        } else return error.CudaError;

        defer {
            if (cuda.cuMemFree) |f| {
                _ = f(d_y);
                _ = f(d_x);
            }
        }

        // Switch to primary context for cuBLAS operation
        try self.setPrimaryContext();

        // Call cuBLAS sdot
        const status = cublas_bindings.cublasSdot(
            self.handle,
            n_i,
            @as([*]const f32, @ptrFromInt(d_x)), incx,
            @as([*]const f32, @ptrFromInt(d_y)), incy,
            &result
        );
        try checkCublasError(status);

        return result;
    }
    
    // Vector operations - Functions 59-62
    
    pub fn saxpy(
        self: *Cublas,
        n: usize,
        alpha: f32,
        x: []const f32, incx: c_int,
        y: []f32, incy: c_int
    ) !void {
        try self.setMemoryContext();

        const n_i = @as(c_int, @intCast(n));

        var d_x: cuda.CUdeviceptr = 0;
        var d_y: cuda.CUdeviceptr = 0;

        const size_x = x.len * @sizeOf(f32);
        const size_y = y.len * @sizeOf(f32);

        // Allocate and copy X
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_x, size_x));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_x);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_x, x.ptr, size_x));
        } else return error.CudaError;

        // Allocate and copy Y
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_y, size_y));
        } else return error.CudaError;
        errdefer {
            if (cuda.cuMemFree) |f| _ = f(d_y);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_y, y.ptr, size_y));
        } else return error.CudaError;

        defer {
            if (cuda.cuMemFree) |f| {
                _ = f(d_y);
                _ = f(d_x);
            }
        }

        // Switch to primary context for cuBLAS operation
        try self.setPrimaryContext();

        // Call cuBLAS saxpy
        const status = cublas_bindings.cublasSaxpy(
            self.handle,
            n_i,
            &alpha,
            @as([*]const f32, @ptrFromInt(d_x)), incx,
            @as([*]f32, @ptrFromInt(d_y)), incy
        );
        try checkCublasError(status);

        // Switch back to memory context for copy
        try self.setMemoryContext();

        // Copy result back to host
        if (cuda.cuMemcpyDtoH) |f| {
            try checkCudaError(f(y.ptr, d_y, size_y));
        } else return error.CudaError;
    }
    
    pub fn daxpy(
        _: *Cublas,
        n: usize,
        alpha: f64, 
        x: []const f64, incx: c_int,
        y: []f64, incy: c_int
    ) !void {
        _ = n; _ = alpha; _ = x; _ = incx; _ = y; _ = incy;
        // Stub implementation - no-op
    }
    
    pub fn sscal(
        self: *Cublas,
        n: usize,
        alpha: f32,
        x: []f32, incx: c_int
    ) !void {
        try self.setMemoryContext();

        const n_i = @as(c_int, @intCast(n));

        var d_x: cuda.CUdeviceptr = 0;
        const size_x = x.len * @sizeOf(f32);

        // Allocate and copy X
        if (cuda.cuMemAlloc) |f| {
            try checkCudaError(f(&d_x, size_x));
        } else return error.CudaError;
        defer {
            if (cuda.cuMemFree) |f| _ = f(d_x);
        }

        if (cuda.cuMemcpyHtoD) |f| {
            try checkCudaError(f(d_x, x.ptr, size_x));
        } else return error.CudaError;

        // Switch to primary context for cuBLAS operation
        try self.setPrimaryContext();

        // Call cuBLAS sscal
        const status = cublas_bindings.cublasSscal(
            self.handle,
            n_i,
            &alpha,
            @as([*]f32, @ptrFromInt(d_x)), incx
        );
        try checkCublasError(status);

        // Switch back to memory context for copy
        try self.setMemoryContext();

        // Copy result back to host
        if (cuda.cuMemcpyDtoH) |f| {
            try checkCudaError(f(x.ptr, d_x, size_x));
        } else return error.CudaError;
    }
    
    pub fn dscal(
        _: *Cublas,
        n: usize,
        alpha: f64,
        x: []f64, incx: c_int  
    ) !void {
        _ = n; _ = alpha; _ = x; _ = incx;
        // Stub implementation - no-op
    }
    
    pub fn sgemmBatched(
        _: *Cublas,
        trans_a: bool, trans_b: bool,
        m: usize, n: usize, k: usize,
        alpha: f32,
        a_array: []const []const f32, lda: usize,
        b_array: []const []const f32, ldb: usize,
        beta: f32,
        c_array: [][]f32, ldc: usize,
        batch_count: usize
    ) !void {
        _ = trans_a; _ = trans_b; _ = m; _ = n; _ = k;
        _ = alpha; _ = a_array; _ = lda;
        _ = b_array; _ = ldb; 
        _ = beta; _ = c_array; _ = ldc; _ = batch_count;
        // Stub implementation - no-op
    }
    
    pub fn dgemmBatched(
        _: *Cublas,
        trans_a: bool, trans_b: bool,
        m: usize, n: usize, k: usize,
        alpha: f64,
        a_array: []const []const f64, lda: usize,
        b_array: []const []const f64, ldb: usize,
        beta: f64,
        c_array: [][]f64, ldc: usize,
        batch_count: usize
    ) !void {
        _ = trans_a; _ = trans_b; _ = m; _ = n; _ = k;
        _ = alpha; _ = a_array; _ = lda;
        _ = b_array; _ = ldb; 
        _ = beta; _ = c_array; _ = ldc; _ = batch_count;
        // Stub implementation - no-op
    }
    
    pub fn setPointerMode(
        _: *Cublas,
        pointer_mode: c_int
    ) !void {
        _ = pointer_mode;
        // Stub implementation - no-op  
    }
};

// TODO: Add more cuBLAS operations when bindings are properly integrated