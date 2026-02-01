struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,
}

// Main compute kernel - compute gradients for one weight matrix element or bias
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let total_weights = grad_params.output_size * grad_params.input_size;
    let weight_idx = global_id.x;
    let local_idx = local_id.x;
    
    // === WEIGHT GRADIENTS COMPUTATION ===
    if (weight_idx < total_weights) {
        // Decode which weight this thread is responsible for
        let output_neuron = weight_idx / grad_params.input_size;
        let input_neuron = weight_idx % grad_params.input_size;
        
        // Compute gradient for this weight: sum over batch of (next_layer_error * activation)
        var weight_gradient: f32 = 0.0;
        
        for (var batch_idx: u32 = 0u; batch_idx < grad_params.batch_size; batch_idx = batch_idx + 1u) {
            let delta_idx = batch_idx * grad_params.output_size + output_neuron;
            let activation_idx = batch_idx * grad_params.input_size + input_neuron;
            
            let next_layer_delta = next_layer_deltas[delta_idx];
            let activation = input_activations[activation_idx];
            
            weight_gradient = weight_gradient + next_layer_delta * activation;
        }

        // Division by batch_size already included in error values
        
        // Write or accumulate the gradient
        if (grad_params.accumulate == 0u) {
            weight_gradients[weight_idx] = weight_gradient;
        } else {
            weight_gradients[weight_idx] = weight_gradients[weight_idx] + weight_gradient;
        }
    }
}
