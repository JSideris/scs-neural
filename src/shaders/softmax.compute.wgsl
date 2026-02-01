struct SoftmaxParams {
    batch_size: u32,
    output_size: u32,
}

// @group(0) @binding(0) var<storage, read> params: SoftmaxParams;
// @group(0) @binding(1) var<storage, read> inputs: array<f32>;
// @group(0) @binding(2) var<storage, read_write> outputs: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * params.output_size;
    
    // Find max for numerical stability
    var max_val: f32 = -1e38;
    for (var i: u32 = 0u; i < params.output_size; i++) {
        max_val = max(max_val, inputs[base_idx + i]);
    }

    // Sum of exponentials
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.output_size; i++) {
        sum += exp(inputs[base_idx + i] - max_val);
    }

    // Normalize
    for (var i: u32 = 0u; i < params.output_size; i++) {
        outputs[base_idx + i] = exp(inputs[base_idx + i] - max_val) / sum;
    }
}
