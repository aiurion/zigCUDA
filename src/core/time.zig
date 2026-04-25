const std = @import("std");

pub fn microTimestamp() i64 {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .awake).toMicroseconds();
}

pub fn milliTimestamp() i64 {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .awake).toMilliseconds();
}
