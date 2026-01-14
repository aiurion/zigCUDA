const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    // Create the root module from main.zig with a known target
    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
    });

    // Keep the default target for 'zig build' and 'run'
    const exe_default = b.addExecutable(.{
        .name = "zigCUDA",
        .root_module = main_module,
    });

    // Link against system libc (required for libcuda.so interaction)
    exe_default.linkLibC();

    b.installArtifact(exe_default);

    // Customize run command to use system dynamic linker
    // This fixes the glibc mismatch issue in WSL+Nix environment for accessing native libcuda.so
    const run_cmd = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_cmd.addArtifactArg(exe_default);

    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Test configuration using Zig's built-in test infrastructure
    // Create bindings test module (low-level CUDA API testing)
    const bindings_test_module = b.createModule(.{
        .root_source_file = b.path("test/comprehensive_bindings_test.zig"),
        .target = target,
    });

    // Create core runtime test module (high-level abstraction integration testing)
    const runtime_test_module = b.createModule(.{
        .root_source_file = b.path("test/core_runtime_test.zig"),
        .target = target,
    });

    // Create cuda module (errors.zig will be imported by cuda.zig via relative import)
    const cuda_module = b.createModule(.{
        .root_source_file = b.path("src/bindings/cuda.zig"),
        .target = target,
    });

    // Add cuda to both test modules
    bindings_test_module.addImport("cuda", cuda_module);
    runtime_test_module.addImport("cuda", cuda_module);

    // Create kernel integration test module
    const kernel_integration_test_module = b.createModule(.{
        .root_source_file = b.path("test/kernel_integration_test.zig"),
        .target = target,
    });

    // Create core module for integration testing
    const core_module = b.createModule(.{
        .root_source_file = b.path("src/core/module.zig"),
        .target = target,
    });

    // Add imports to appropriate test modules
    runtime_test_module.addImport("core", core_module);

    // Create kernel abstraction module for integration tests
    const kernel_abstraction_module = b.createModule(.{
        .root_source_file = b.path("src/core/kernel.zig"),
        .target = target,
    });
    kernel_abstraction_module.addImport("cuda", cuda_module);
    kernel_abstraction_module.addImport("core", core_module);
    kernel_integration_test_module.addImport("kernel", kernel_abstraction_module);

    // Add cuda to all test modules
    bindings_test_module.addImport("cuda", cuda_module);
    runtime_test_module.addImport("cuda", cuda_module);
    kernel_integration_test_module.addImport("cuda", cuda_module);
    core_module.addImport("cuda", cuda_module);

    // Create bindings test executable (low-level API tests)
    const binding_tests = b.addTest(.{
        .root_module = bindings_test_module,
    });

    // Create runtime test executable (high-level integration tests)
    const runtime_tests = b.addTest(.{
        .root_module = runtime_test_module,
    });

    // Create kernel integration test executable
    const kernel_integration_tests = b.addTest(.{
        .root_module = kernel_integration_test_module,
    });

    // Link all test executables against system libc
    binding_tests.linkLibC();
    runtime_tests.linkLibC();
    kernel_integration_tests.linkLibC();

    // Use system dynamic linker to avoid glibc mismatch issues for bindings tests
    const run_bindings_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_bindings_tests.addArtifactArg(binding_tests);

    // Use system dynamic linker for runtime tests too
    const run_runtime_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_runtime_tests.addArtifactArg(runtime_tests);

    // Use system dynamic linker for kernel integration tests
    const run_kernel_integration_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_kernel_integration_tests.addArtifactArg(kernel_integration_tests);

    // Create test steps
    const bindings_test_step = b.step("test-bindings", "Run comprehensive CUDA API binding tests");
    bindings_test_step.dependOn(&run_bindings_tests.step);

    const runtime_test_step = b.step("test-runtime", "Run core abstraction and integration tests");
    runtime_test_step.dependOn(&run_runtime_tests.step);

    // Create simple integration test module
    const simple_integration_test_module = b.createModule(.{
        .root_source_file = b.path("test/simple_kernel_test.zig"),
        .target = target,
    });

    // Add cuda import to simple integration tests
    simple_integration_test_module.addImport("cuda", cuda_module);

    // Create simple integration test executable
    const simple_integration_tests = b.addTest(.{
        .root_module = simple_integration_test_module,
    });

    // Link simple test against system libc
    simple_integration_tests.linkLibC();

    // Use system dynamic linker for simple integration tests
    const run_simple_integration_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_simple_integration_tests.addArtifactArg(simple_integration_tests);

    const kernel_integration_test_step = b.step("test-kernel-integration", "Run high-level kernel abstraction tests");
    kernel_integration_test_step.dependOn(&run_kernel_integration_tests.step);

    // Create cuBLAS bindings module
    const cublas_bindings_module = b.createModule(.{
        .root_source_file = b.path("src/bindings/cublas.zig"),
        .target = target,
    });

    // Create cuBLAS integration module
    const cublas_integration_module = b.createModule(.{
        .root_source_file = b.path("src/integrations/cublas.zig"),
        .target = target,
    });

    // Add imports to cuBLAS integration module
    cublas_integration_module.addImport("cuda", cuda_module);
    cublas_integration_module.addImport("cublas_bindings", cublas_bindings_module);

    // Create cuBLAS integration test module
    const cublas_integration_test_module = b.createModule(.{
        .root_source_file = b.path("test/cublas_integration_test.zig"),
        .target = target,
    });

    // Add imports to cuBLAS integration tests
    cublas_integration_test_module.addImport("cuda", cuda_module);
    cublas_integration_test_module.addImport("integrations", cublas_integration_module);

    // Create cuBLAS integration test executable
    const cublas_integration_tests = b.addTest(.{
        .root_module = cublas_integration_test_module,
    });

    // Link cuBLAS test against system libc
    cublas_integration_tests.linkLibC();

    // Use system dynamic linker for cuBLAS integration tests
    const run_cublas_integration_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_cublas_integration_tests.addArtifactArg(cublas_integration_tests);

    // Add simple integration test step
    const simple_integration_test_step = b.step("test-simple", "Run basic CUDA functionality tests");
    simple_integration_test_step.dependOn(&run_simple_integration_tests.step);

    // Create simple cuBLAS test module
    const cublas_simple_test_module = b.createModule(.{
        .root_source_file = b.path("test/cublas_integration_test_simple.zig"),
        .target = target,
    });

    // Add imports to simple cuBLAS tests
    cublas_simple_test_module.addImport("cuda", cuda_module);
    cublas_simple_test_module.addImport("integrations", cublas_integration_module);

    // Create simple cuBLAS test executable
    const cublas_simple_tests = b.addTest(.{
        .root_module = cublas_simple_test_module,
    });

    // Link simple cuBLAS test against system libc
    cublas_simple_tests.linkLibC();

    // Use system dynamic linker for simple cuBLAS tests
    const run_cublas_simple_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_cublas_simple_tests.addArtifactArg(cublas_simple_tests);

    // Add cuBLAS integration test step
    const cublas_integration_test_step = b.step("test-cublas-integration", "Run cuBLAS BLAS operations integration tests");
    cublas_integration_test_step.dependOn(&run_cublas_integration_tests.step);

    // Add simple cuBLAS integration test step
    const cublas_simple_test_step = b.step("test-cublas-simple", "Run simplified cuBLAS stub tests");
    cublas_simple_test_step.dependOn(&run_cublas_simple_tests.step);

    // Create Functions 59-65 test module
    const functions_59_65_test_module = b.createModule(.{
        .root_source_file = b.path("test/cublas_functions_59_65_test.zig"),
        .target = target,
    });

    // Create Functions 59-65 test executable
    const functions_59_65_tests = b.addTest(.{
        .root_module = functions_59_65_test_module,
    });

    // Link Functions 59-65 tests against system libc
    functions_59_65_tests.linkLibC();

    // Use system dynamic linker for Functions 59-65 tests
    const run_functions_59_65_tests = b.addSystemCommand(&.{
        "/lib64/ld-linux-x86-64.so.2",
    });
    run_functions_59_65_tests.addArtifactArg(functions_59_65_tests);

    // Add Functions 59-65 test step
    const functions_59_65_test_step = b.step("test-cublas-functions-59-65", "Run cuBLAS Functions 59-65 implementation tests");
    functions_59_65_test_step.dependOn(&run_functions_59_65_tests.step);

    // Combined test step that runs all tests
    const all_tests_step = b.step("test", "Run all tests (bindings + runtime + kernel integration + simple + cuBLAS integration)");
    all_tests_step.dependOn(bindings_test_step);
    all_tests_step.dependOn(runtime_test_step);
    all_tests_step.dependOn(kernel_integration_test_step);
    all_tests_step.dependOn(simple_integration_test_step);
    all_tests_step.dependOn(cublas_integration_test_step);
    all_tests_step.dependOn(functions_59_65_test_step);
}
