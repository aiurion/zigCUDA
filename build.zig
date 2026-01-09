const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    // Create the root module from main.zig with a known target
    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
    });

    // // Build for Linux only (to test)
    // const exe_linux = b.addExecutable(.{
    //     .name = "zcode-linux",
    //     .root_module = main_module,
    // });
    // b.installArtifact(exe_linux);

    // Keep the default target for 'zig build' and 'run'
    const exe_default = b.addExecutable(.{
        .name = "zigCUDA",
        .root_module = main_module,
    });

    // Link against system libc (required for libcuda.so interaction)
    exe_default.linkLibC();

    // Link against CUDA Driver API library
    // exe_default.addLibraryPath(b.path("deps/stubs"));
    // exe_default.linkSystemLibrary("cuda");
    // We will use dynamic loading instead!

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
}
