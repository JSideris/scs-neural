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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let in_x = global_id.x;
    let in_y = global_id.y;
    let in_z = global_id.z; // batch_idx * input_channels + in_channel

    if (in_x >= params.input_width || in_y >= params.input_height) {
        return;
    }

    let batch_idx = in_z / params.input_channels;
    let in_c = in_z % params.input_channels;

    if (batch_idx >= params.batch_size) {
        return;
    }

    var weighted_error_sum: f32 = 0.0;

    for (var kh: u32 = 0u; kh < params.kernel_size; kh++) {
        let temp_y = i32(in_y + params.padding) - i32(kh);
        if (temp_y < 0 || temp_y % i32(params.stride) != 0) {
            continue;
        }
        let out_y = u32(temp_y / i32(params.stride));
        if (out_y >= params.output_height) {
            continue;
        }

        for (var kw: u32 = 0u; kw < params.kernel_size; kw++) {
            let temp_x = i32(in_x + params.padding) - i32(kw);
            if (temp_x < 0 || temp_x % i32(params.stride) != 0) {
                continue;
            }
            let out_x = u32(temp_x / i32(params.stride));
            if (out_x >= params.output_width) {
                continue;
            }

            let out_base = ((batch_idx * params.output_height + out_y) * params.output_width + out_x) * params.output_channels;
            let weight_base = ((kh * params.kernel_size + kw) * params.input_channels + in_c) * params.output_channels;

            for (var out_c: u32 = 0u; out_c < params.output_channels; out_c++) {
                let next_delta = next_layer_deltas[out_base + out_c];
                let weight = weights[weight_base + out_c];
                weighted_error_sum += next_delta * weight;
            }
        }
    }

    let current_idx = ((batch_idx * params.input_height + in_y) * params.input_width + in_x) * params.input_channels + in_c;
    current_layer_weighted_sums[current_idx] = weighted_error_sum;
}
