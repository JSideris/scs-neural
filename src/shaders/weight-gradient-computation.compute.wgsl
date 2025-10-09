struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,     // 0 = overwrite, 1 = accumulate (for mini-batch accumulation)
}

// // Buffer bindings
// // Group 0: Input data for gradient computation
// @group(0) @binding(0) var<storage, read> errors: array<f32>;           // [batch_size, output_size]
// @group(0) @binding(1) var<storage, read> input_activations: array<f32>; // [batch_size, input_size]

// // Group 1:Non-swappy data
// @group(1) @binding(0) var<uniform> params: GradientParams;
// @group(1) @binding(1) var<storage, read_write> weight_gradients: array<f32>; // [output_size, input_size]

// Workgroup shared memory for reduction operations
var<workgroup> shared_weight_grad: array<f32, 256>;

// Main compute kernel - compute gradients for one weight matrix element or bias
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let total_weights = params.output_size * params.input_size;
    let weight_idx = global_id.x;
    let local_idx = local_id.x;
    
    // === WEIGHT GRADIENTS COMPUTATION ===
    if (weight_idx < total_weights) {
        // Decode which weight this thread is responsible for
        let output_neuron = weight_idx / params.input_size;
        let input_neuron = weight_idx % params.input_size;
        
        // Compute gradient for this weight: sum over batch of (error * activation)
        var weight_gradient: f32 = 0.0;
        
        for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx = batch_idx + 1u) {
            let error_idx = batch_idx * params.output_size + output_neuron;
            let activation_idx = batch_idx * params.input_size + input_neuron;
            
            let error = errors[error_idx];
            let activation = input_activations[activation_idx];
            
            weight_gradient = weight_gradient + error * activation;
        }

        // Division by batch_size already included in error values
        // weight_gradient = weight_gradient / f32(params.batch_size);
        
        // Write or accumulate the gradient
        if (params.accumulate == 0u) {
            weight_gradients[weight_idx] = weight_gradient;
        } else {
            weight_gradients[weight_idx] = weight_gradients[weight_idx] + weight_gradient;
        }
    }
}