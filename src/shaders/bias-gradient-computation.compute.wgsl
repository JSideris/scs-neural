struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,     // 0 = overwrite, 1 = accumulate (for mini-batch accumulation)
}

// Buffer bindings - added at runtime.
// Group 0: Input data for gradient computation
// @group(0) @binding(0) var<storage, read> errors: array<f32>;           // [batch_size, output_size]
// @group(0) @binding(1) var<storage, read> input_activations: array<f32>; // [batch_size, input_size]

// Group 1: Output gradients
// @group(1) @binding(0) var<uniform> params: GradientParams;
// @group(1) @binding(1) var<storage, read_write> bias_gradients: array<f32>;   // [output_size]

// Workgroup shared memory for reduction operations
var<workgroup> shared_bias_grad: array<f32, 256>;

// Alternative: Optimized bias computation using shared memory reduction
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                                   @builtin(local_invocation_id) local_id: vec3<u32>,
                                   @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let local_idx = local_id.x;
    let workgroup_idx = workgroup_id.x;
    
    // Each workgroup handles one output neuron
    let output_neuron = workgroup_idx;
    
    // Hoist uniform values to locals for better uniformity
    let output_size = params.output_size;
    let batch_size = params.batch_size;
    let accumulate = params.accumulate;
    
    let is_valid = (output_neuron < output_size);
    
    // Each thread in the workgroup handles some batch samples
    var local_sum: f32 = 0.0;
    
    // Only compute if valid
    if (is_valid) {
        // Stride through batch samples
        var batch_idx = local_idx;
        while (batch_idx < batch_size) {
            let error_idx = batch_idx * output_size + output_neuron;
            local_sum += errors[error_idx];
            batch_idx += 256u;
        }
    }
    
    // Store in shared memory (invalid workgroups store 0.0)
    shared_bias_grad[local_idx] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_bias_grad[local_idx] += shared_bias_grad[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Division by batch_size already included in error values
    // shared_bias_grad[0] = shared_bias_grad[0] / f32(params.batch_size);
    
    // First thread writes the result (only if valid)
    if (local_idx == 0u && is_valid) {
        let total_sum = shared_bias_grad[0];
        if (accumulate == 0u) {
            bias_gradients[output_neuron] = total_sum;
        } else {
            bias_gradients[output_neuron] += total_sum;
        }
    }
}