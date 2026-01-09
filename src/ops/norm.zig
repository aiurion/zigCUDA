// src/ops/norm.zig
// Normalization layers (LayerNorm, RMSNorm)
// TODO: Implement normalization layers

const tensor = @import("tensor.zig");

pub const NormConfig = struct {
    eps: f32 = 1e-5,
    bias: bool = true,
    affine: bool = true,
};

pub const LayerNorm = struct {
    config: NormConfig,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    
    pub fn init(
        dim: usize,
        config: NormConfig,
    ) !LayerNorm {
        // TODO: Implement layer norm init
        _ = dim;
        _ = config;
        return LayerNorm{
            .config = config,
            .weight = undefined,
            .bias = undefined,
        };
    }
    
    pub fn forward(self: *LayerNorm, input: tensor.Tensor) !tensor.Tensor {
        // TODO: Implement layer norm forward
        _ = input;
        return undefined;
    }
};

pub const RMSNorm = struct {
    config: NormConfig,
    weight: tensor.Tensor,
    
    pub fn init(
        dim: usize,
        config: NormConfig,
    ) !RMSNorm {
        // TODO: Implement RMS norm init
        _ = dim;
        _ = config;
        return RMSNorm{
            .config = config,
            .weight = undefined,
        };
    }
    
    pub fn forward(self: *RMSNorm, input: tensor.Tensor) !tensor.Tensor {
        // TODO: Implement RMS norm forward
        _ = input;
        return undefined;
    }
};

pub const BatchNorm = struct {
    pub fn forward(
        input: tensor.Tensor,
        running_mean: tensor.Tensor,
        running_var: tensor.Tensor,
        weight: tensor.Tensor,
        bias: tensor.Tensor,
        training: bool,
        momentum: f32,
        eps: f32,
    ) tensor.Tensor {
        // TODO: Implement batch norm
        _ = input;
        _ = running_mean;
        _ = running_var;
        _ = weight;
        _ = bias;
        _ = training;
        _ = momentum;
        _ = eps;
        return undefined;
    }
};

pub const GroupNorm = struct {
    pub fn forward(
        num_groups: usize,
        input: tensor.Tensor,
        weight: tensor.Tensor,
        bias: tensor.Tensor,
        eps: f32,
    ) tensor.Tensor {
        // TODO: Implement group norm
        _ = num_groups;
        _ = input;
        _ = weight;
        _ = bias;
        _ = eps;
        return undefined;
    }
};

// TODO: Add more normalization layers