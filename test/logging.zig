// test/logging.zig
// Proper logging for ZigCUDA development

const std = @import("std");

/// Log entry with timestamp and context
pub const LogEntry = struct {
    level: LogLevel,
    component: []const u8,
    message: []const u8,
    timestamp_ns: i64,

    pub fn format(
        self: LogEntry,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const time_ms = @divTrunc(self.timestamp_ns, 1_000_000);
        
        try writer.print("[{d}ms][{}] {}: {}", .{
            time_ms,
            self.level.toString(),
            self.component,
            self.message
        });
    }
};

pub const LogLevel = enum {
    Debug,
    Info,
    Warn,
    Error,
    
    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .Debug => "DEBUG",
            .Info  => "INFO", 
            .Warn  => "WARN",
            .Error => "ERROR"
        };
    }
};

/// Logger with proper organization
pub const Logger = struct {
    entries: std.ArrayList(LogEntry),
    
    pub fn init(allocator: std.mem.Allocator) !Logger {
        return .{
            .entries = std.ArrayList(LogEntry).init(allocator)
        };
    }

    pub fn log(self: *Logger, level: LogLevel, component: []const u8, message: []const u8) !void {
        const entry = LogEntry{
            .level = level,
            .component = component,
            .message = message,
            .timestamp_ns = std.time.nanoTimestamp(),
        };
        
        try self.entries.append(entry);
    }

    pub fn debug(self: *Logger, comptime component: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(
            self.entries.allocator,
            fmt,
            args
        );
        
        defer self.entries.allocator.free(message);
        try self.log(.Debug, component, message);
    }

    pub fn info(self: *Logger, comptime component: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(
            self.entries.allocator,
            fmt,
            args
        );
        
        defer self.entries.allocator.free(message);
        try self.log(.Info, component, message);
    }

    pub fn warn(self: *Logger, comptime component: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(
            self.entries.allocator,
            fmt,
            args
        );
        
        defer self.entries.allocator.free(message);
        try self.log(.Warn, component, message);
    }

    pub fn error(self: *Logger, comptime component: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(
            self.entries.allocator,
            fmt,
            args
        );
        
        defer self.entries.allocator.free(message);
        try self.log(.Error, component, message);
    }

    pub fn saveToFile(self: Logger, filename: []const u8) !void {
        const file = std.fs.cwd().createFile(filename, .{}) catch |err| {
            // If we can't create the log file, at least print to stderr
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("WARNING: Could not create log file: ");
            try stderr.writeAll(@errorName(err));
            try stderr.writeByte('\n');
            return;
        };

        defer file.close();

        for (self.entries.items) |entry| {
            var buffer: [1024]u8 = undefined;
            const formatted = try std.fmt.bufPrint(&buffer, "{}\n", .{entry});
            
            // Handle potential truncation
            if (formatted.len > buffer.len) {
                _ = try file.write(buffer[0..]);
            } else {
                _ = try file.write(formatted);
            }
        }

        const stdout = std.io.getStdOut().writer();
        try stdout.print("Log saved to: {s}\n", .{filename});
    }

    pub fn deinit(self: *Logger) void {
        self.entries.deinit();
    }
};