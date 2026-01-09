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
        .name = "zcode",
        .root_module = main_module,
    });
    b.installArtifact(exe_default);

    const run_cmd = b.addRunArtifact(exe_default);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
