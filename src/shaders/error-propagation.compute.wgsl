struct BackpropParams {
    batch_size: u32,
    current_layer_size: u32,
    next_layer_size: u32,
    activation_type: u32,    // 0 = ReLU, 1 = Sigmoid, 2 = Linear
    is_output_layer: u32,    // 1 if this is output layer, 0 for hidden layers
}

// // Buffer bindings
// // Group 0: Ping-pong error buffers for error gradients.
// @group(0) @binding(0) var<storage, read> next_layer_errors: array<f32>;        // [batch_size, next_layer_size]
// @group(0) @binding(1) var<storage, read_write> current_layer_errors: array<f32>; // [batch_size, current_layer_size]

// // Group 1: Layer-specific data
// @group(1) @binding(0) var<storage, read> weights: array<f32>;                  // [next_layer_size, current_layer_size]
// @group(1) @binding(1) var<storage, read> z_values: array<f32>;                 // [batch_size, current_layer_size]

// // Group 2: Non-swappy data
// @group(2) @binding(0) var<uniform> params: BackpropParams;
// @group(2) @binding(1) var<storage, read> predictions: array<f32>;              // [batch_size, output_size]
// @group(2) @binding(2) var<storage, read> targets: array<f32>;                  // [batch_size, output_size]

// Activation derivative functions
fn relu_derivative(z: f32) -> f32 {
    return select(0.0, 1.0, z > 0.0);
}

fn sigmoid_derivative(z: f32) -> f32 {
    // Use the identity: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    let clamped_z = clamp(z, -88.0, 88.0);
    let sigmoid_z = 1.0 / (1.0 + exp(-clamped_z));
    return sigmoid_z * (1.0 - sigmoid_z);
}

fn linear_derivative(z: f32) -> f32 {
    return 1.0;
}

fn tanh_derivative(z: f32) -> f32 {
    let clamped_z = clamp(z, -88.0, 88.0);
    let tanh_z = tanh(clamped_z);
    return 1.0 - tanh_z * tanh_z;
}

fn get_activation_derivative(z: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return relu_derivative(z); }
        case 1u: { return sigmoid_derivative(z); }
        case 2u: { return linear_derivative(z); }
        case 3u: { return tanh_derivative(z); }
        // case 4u: { return softmax_derivative(z); } // A bit more tricky. Need to think about it.
        default: { return linear_derivative(z); }
    }
}

// Compute loss derivative for output layer (MSE case)
fn mse_loss_derivative(prediction: f32, target_value: f32, batch_size: u32) -> f32 {
    return 2.0 * (prediction - target_value) / f32(batch_size);
}

// Main compute kernel
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_neurons = params.batch_size * params.current_layer_size;
    let thread_id = global_id.x;
    
    // Early exit if thread is out of bounds
    if (thread_id >= total_neurons) {
        return;
    }
    
    // Decode which batch sample and neuron this thread handles
    let batch_idx = thread_id / params.current_layer_size;
    let neuron_idx = thread_id % params.current_layer_size;
    let current_idx = batch_idx * params.current_layer_size + neuron_idx;
    
    var error: f32 = 0.0;
    
    if (params.is_output_layer == 1u) {
        // === OUTPUT LAYER ERROR COMPUTATION ===
        // Error = loss_derivative * activation_derivative
        
        let prediction = predictions[current_idx];
        let target_value = targets[current_idx];
        let z = z_values[current_idx];
        
        // Compute loss derivative (assuming MSE for now)
        let loss_grad = mse_loss_derivative(prediction, target_value, params.batch_size);
        
        // Compute activation derivative
        let activation_grad = get_activation_derivative(z, params.activation_type);
        
        // Combine: dL/dz = dL/da * da/dz
        error = loss_grad * activation_grad;
        
    } else {
        // === HIDDEN LAYER ERROR PROPAGATION ===
        // Error = (weights^T × next_errors) * activation_derivative
        
        // Step 1: Compute weights^T × next_errors
        // This is matrix-vector multiplication where we sum over the next layer
        var weighted_error_sum: f32 = 0.0;
        
        for (var next_neuron: u32 = 0u; next_neuron < params.next_layer_size; next_neuron = next_neuron + 1u) {
            // Weight matrix is stored as [next_layer_size, current_layer_size]
            // So weights[next_neuron][current_neuron] = weights[next_neuron * current_layer_size + neuron_idx]
            let weight_idx = next_neuron * params.current_layer_size + neuron_idx;
            let weight = weights[weight_idx];
            
            // Get the error from the next layer for this batch sample and next layer neuron
            let next_error_idx = batch_idx * params.next_layer_size + next_neuron;
            let next_error = next_layer_errors[next_error_idx];
            
            weighted_error_sum = weighted_error_sum + weight * next_error;
        }
        
        // Step 2: Multiply by activation derivative
        let z = z_values[current_idx];
        let activation_grad = get_activation_derivative(z, params.activation_type);
        
        error = weighted_error_sum * activation_grad;
    }
    
    // Write computed error to output buffer
    current_layer_errors[current_idx] = error;
}
