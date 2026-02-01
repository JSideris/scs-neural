struct Conv2DParams {
    batch_size: u32,
    input_height: u32,
    input_width: u32,
    input_channels: u32,
    output_height: u32,
    output_width: u32,
    output_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    activation_type: u32,
}

struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_weights = params.kernel_size * params.kernel_size * params.input_channels * params.output_channels;
    let weight_idx = global_id.x;

    if (weight_idx < total_weights) {
        // Decode weight index: [kh, kw, in_c, out_c]
        let out_c = weight_idx % params.output_channels;
        let in_c = (weight_idx / params.output_channels) % params.input_channels;
        let kw = (weight_idx / (params.output_channels * params.input_channels)) % params.kernel_size;
        let kh = weight_idx / (params.output_channels * params.input_channels * params.kernel_size);

        var grad: f32 = 0.0;

        for (var b: u32 = 0u; b < params.batch_size; b++) {
            for (var out_y: u32 = 0u; out_y < params.output_height; out_y++) {
                let in_y = i32(out_y * params.stride + kh) - i32(params.padding);
                if (in_y < 0 || in_y >= i32(params.input_height)) {
                    continue;
                }

                for (var out_x: u32 = 0u; out_x < params.output_width; out_x++) {
                    let in_x = i32(out_x * params.stride + kw) - i32(params.padding);
                    if (in_x < 0 || in_x >= i32(params.input_width)) {
                        continue;
                    }

                    let error_idx = ((b * params.output_height + out_y) * params.output_width + out_x) * params.output_channels + out_c;
                    let activation_idx = ((b * params.input_height + u32(in_y)) * params.input_width + u32(in_x)) * params.input_channels + in_c;
                    
                    grad += next_layer_deltas[error_idx] * input_activations[activation_idx];
                }
            }
        }

        if (grad_params.accumulate == 0u) {
            weight_gradients[weight_idx] = grad;
        } else {
            weight_gradients[weight_idx] += grad;
        }
    }

    // Bias gradients: Only one thread per output channel needs to do this.
    // We can use the first few threads.
    if (global_id.x < params.output_channels) {
        let out_c = global_id.x;
        var b_grad: f32 = 0.0;

        for (var b: u32 = 0u; b < params.batch_size; b++) {
            for (var out_y: u32 = 0u; out_y < params.output_height; out_y++) {
                for (var out_x: u32 = 0u; out_x < params.output_width; out_x++) {
                    let error_idx = ((b * params.output_height + out_y) * params.output_width + out_x) * params.output_channels + out_c;
                    b_grad += next_layer_deltas[error_idx];
                }
            }
        }

        if (grad_params.accumulate == 0u) {
            bias_gradients[out_c] = b_grad;
        } else {
            bias_gradients[out_c] += b_grad;
        }
    }
}
