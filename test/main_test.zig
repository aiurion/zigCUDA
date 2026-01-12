// test/main_test.zig
// Proper ZigCUDA Testing Framework

const std = @import("std");
const testing = std.testing;
const log = std.log;

const bindings = @import("../src/bindings/cuda.zig");
const device = @import("../src/core/device.zig");
const memory = @import("../src/core/memory.zig");

/// Test result structure
pub const TestResult = struct {
    name: []const u8,
    passed: bool,
    error_message: ?[]const u8,
    duration_ns: u64,

    pub fn format(
        self: TestResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const status = if (self.passed) "PASS" else "FAIL";
        try writer.print("{s}: {s}", .{ status, self.name });

        if (!self.passed and self.error_message != null) {
            try writer.print(" - {s}", .{self.error_message.?});
        }
    }
};

/// Test runner with proper reporting
pub const TestRunner = struct {
    results: std.ArrayList(TestResult),

    pub fn init(allocator: std.mem.Allocator) TestRunner {
        return .{
            .results = std.ArrayList(TestResult).init(allocator),
        };
    }

    /// Run a single test with proper error handling
    pub fn run(
        self: *TestRunner,
        name: []const u8,
        test_fn: *const fn () anyerror!void,
    ) !void {
        const start_time = std.time.nanoTimestamp();

        var result = TestResult{
            .name = name,
            .passed = false,
            .error_message = null,
            .duration_ns = 0,
        };

        // Execute test and capture errors
        if (test_fn()) |_| {
            result.passed = true;
        } else |err| {
            result.error_message = @errorName(err);
        }

        result.duration_ns = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));

        try self.results.append(result);
    }

    /// Print formatted test report
    pub fn printReport(self: TestRunner) !void {
        const stdout = std.io.getStdOut().writer();

        var total_tests: u32 = 0;
        var passed_tests: u32 = 0;
        var failed_tests: u32 = 0;

        try stdout.writeAll("\n=== ZIGCUDA TEST REPORT ===\n\n");

        for (self.results.items) |result| {
            total_tests += 1;
            if (result.passed) {
                passed_tests += 1;
                try stdout.print("✓ PASS: {s} ({d}ns)\n", .{ result.name, result.duration_ns });
            } else {
                failed_tests += 1;
                try stdout.print("✗ FAIL: {s}", .{result.name});
                if (result.error_message != null) {
                    try stdout.print(" - {s}", .{result.error_message.?});
                }
                try stdout.print("\n");
            }
        }

        // Summary
        try stdout.writeAll("\n=== SUMMARY ===\n");
        try stdout.print("Total: {} | Passed: {} | Failed: {} | Success Rate: {:.1}%\n", .{ total_tests, passed_tests, failed_tests, if (total_tests > 0) @as(f64, @intCast(passed_tests)) / @as(f64, @intCast(total_tests)) * 100.0 else 0 });

        if (failed_tests == 0) {
            try stdout.writeAll("\n🎉 ALL TESTS PASSED! 🎉\n");
        } else {
            try stdout.print("\n�️ {} tests need attention.\n", .{failed_tests});
        }
    }

    pub fn deinit(self: *TestRunner) void {
        self.results.deinit();
    }
};

/// Test functions for core functionality
pub const CUDATests = struct {
    /// Initialize CUDA runtime test
    pub fn initCuda() !void {
        // This should replace your scattered initialization attempts
        _ = try bindings.init(0);
    }

    /// Device enumeration test
    pub fn deviceEnumeration() !void {
        const count = try device.Device.count();

        if (count == 0) {
            log.warn("No CUDA devices found - skipping device tests", .{});
            return;
        }

        // Test getting first device
        _ = try device.Device.init(0);
    }

    /// Memory allocation test
    pub fn memoryAllocation() !void {
        const ptr = try bindings.cuMemAlloc(@intFromPtr(&@as(c_ulonglong, undefined)), @sizeOf(u32) * 100);

        // Test cleanup
        _ = bindings.cuMemFree(ptr);
    }
};

/// Main test execution
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    defer {
        _ = gpa.deinit();
    }

    var runner = TestRunner.init(allocator);
    defer runner.deinit();

    log.info("Starting ZigCUDA comprehensive test suite...", .{});

    // Run core tests
    try runner.run("CUDA Initialization", CUDATests.initCuda);
    try runner.run("Device Enumeration", CUDATests.deviceEnumeration);
    try runner.run("Memory Allocation", CUDATests.memoryAllocation);

    // Print organized results
    try runner.printReport();
}
