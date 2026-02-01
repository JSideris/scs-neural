struct LayerParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    activation_type: u32,
}

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    // Clamp input to prevent overflow
    let clamped_x = clamp(x, -88.0, 88.0);
    return 1.0 / (1.0 + exp(-clamped_x));
}


fn apply_activation(x: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return relu(x); }
        case 1u: { return sigmoid(x); }
        case 2u: { return x; }
        case 3u: { return tanh(x); } // built-in.
        default: { return x; } // Linear (no activation)
    }
}

// Main compute kernel
// Each thread processes one output neuron for one batch sample
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Calculate which batch sample and output neuron this thread handles
    let total_outputs = params.batch_size * params.output_size;
    let thread_id = global_id.x;
    
    // Early exit if thread is out of bounds
    if (thread_id >= total_outputs) {
        return;
    }
    
    // Decode which batch sample and output neuron
    let batch_idx = thread_id / params.output_size;
    let output_idx = thread_id % params.output_size;
    
    // Calculate indices for memory access
    let input_base = batch_idx * params.input_size;
    let weight_base = output_idx * params.input_size;
    let output_linear_idx = batch_idx * params.output_size + output_idx;
    
    // Perform matrix multiplication: sum(weights[i] * inputs[i])
    var accumulator: f32 = 0.0;
    
    for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
        let weight = weights[weight_base + i];
        let input = inputs[input_base + i];
        accumulator = accumulator + weight * input;
    }
    
    // Add bias term
    let z_value = accumulator + biases[output_idx];
    
    // Store z-value for backpropagation
    z_values[output_linear_idx] = z_value;
    
    // Apply activation function
    let activated_value = apply_activation(z_value, params.activation_type);
    
    // Write result to output buffer
    activations[output_linear_idx] = activated_value;
}
