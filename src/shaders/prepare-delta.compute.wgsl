struct PrepareDeltaParams {
    batch_size: u32,
    size: u32,
    activation_type: u32,
    is_output_layer: u32,
}

// @group(0) @binding(0) var<storage, read> params: PrepareDeltaParams;
// @group(0) @binding(1) var<storage, read> incoming_error: array<f32>; // Loss' or weighted_sum from above
// @group(0) @binding(2) var<storage, read> z_values: array<f32>;
// @group(0) @binding(3) var<storage, read> predictions: array<f32>; // Only used for output layer
// @group(0) @binding(4) var<storage, read> targets: array<f32>;     // Only used for output layer
// @group(0) @binding(5) var<storage, read_write> delta: array<f32>;

fn relu_derivative(z: f32) -> f32 {
    return select(0.0, 1.0, z > 0.0);
}

fn sigmoid_derivative(z: f32) -> f32 {
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
        default: { return linear_derivative(z); }
    }
}

fn mse_loss_derivative(prediction: f32, target_value: f32, batch_size: u32) -> f32 {
    return 2.0 * (prediction - target_value) / f32(batch_size);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.batch_size * params.size;
    if (global_id.x >= total_elements) {
        return;
    }

    let idx = global_id.x;
    var error_term: f32 = 0.0;

    if (params.is_output_layer == 1u) {
        let prediction = predictions[idx];
        let target_val = targets[idx];
        
        if (params.activation_type == 4u) { // SOFTMAX + Cross-Entropy shortcut
            error_term = (prediction - target_val) / f32(params.batch_size);
        } else {
            let loss_grad = mse_loss_derivative(prediction, target_val, params.batch_size);
            let activation_grad = get_activation_derivative(z_values[idx], params.activation_type);
            error_term = loss_grad * activation_grad;
        }
    } else {
        let activation_grad = get_activation_derivative(z_values[idx], params.activation_type);
        error_term = incoming_error[idx] * activation_grad;
    }

    delta[idx] = error_term;
}
