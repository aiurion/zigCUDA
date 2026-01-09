// src/core/config.zig
// Runtime configuration
// TODO: Implement configuration management

pub const Config = struct {
    device_index: u32 = 0,
    memory_pool_size: usize = 1024 * 1024 * 1024, // 1GB default
    stream_count: u32 = 4,
    enable_debug: bool = false,
    log_level: enum { error, warn, info, debug } = .info,
    
    pub fn init() Config {
        return Config{};
    }
    
    pub fn save(self: *const Config, path: [:0]const u8) !void {
        // TODO: Implement config save
        _ = path;
    }
    
    pub fn load(path: [:0]const u8) !Config {
        // TODO: Implement config load
        return Config.init();
    }
    
    pub fn validate(self: *const Config) !void {
        // TODO: Implement config validation
    }
};

// Global configuration instance
var global_config: ?Config = null;

pub fn getConfig() *Config {
    return &global_config orelse &Config.init();
}

pub fn setConfig(config: Config) void {
    global_config = config;
}

// TODO: Add more configuration functionality