struct GeneticConv2DParams {
    population_size: u32,
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

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    let clamped_x = clamp(x, -88.0, 88.0);
    return 1.0 / (1.0 + exp(-clamped_x));
}

fn apply_activation(x: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return relu(x); }
        case 1u: { return sigmoid(x); }
        case 2u: { return x; }
        case 3u: { return tanh(x); }
        default: { return x; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_z = global_id.z; // (genome_idx * batch_size + batch_sample_idx) * output_channels + out_c

    if (out_x >= params.output_width || out_y >= params.output_height) {
        return;
    }

    let samples_per_genome = params.batch_size * params.output_channels;
    let genome_idx = out_z / samples_per_genome;
    let rem = out_z % samples_per_genome;
    let batch_sample_idx = rem / params.output_channels;
    let out_c = rem % params.output_channels;

    if (genome_idx >= params.population_size) {
        return;
    }

    var acc: f32 = 0.0;
    
    let start_x = i32(out_x * params.stride) - i32(params.padding);
    let start_y = i32(out_y * params.stride) - i32(params.padding);

    let weights_per_genome = params.kernel_size * params.kernel_size * params.input_channels * params.output_channels;
    let weight_genome_base = genome_idx * weights_per_genome;

    for (var kh: u32 = 0u; kh < params.kernel_size; kh++) {
        let in_y = i32(start_y + i32(kh));
        if (in_y < 0 || in_y >= i32(params.input_height)) {
            continue;
        }

        for (var kw: u32 = 0u; kw < params.kernel_size; kw++) {
            let in_x = i32(start_x + i32(kw));
            if (in_x < 0 || in_x >= i32(params.input_width)) {
                continue;
            }

            let input_base = (( (genome_idx * params.batch_size + batch_sample_idx) * params.input_height + u32(in_y)) * params.input_width + u32(in_x)) * params.input_channels;
            let weight_base = weight_genome_base + ((kh * params.kernel_size + kw) * params.input_channels) * params.output_channels + out_c;

            for (var in_c: u32 = 0u; in_c < params.input_channels; in_c++) {
                acc += inputs[input_base + in_c] * weights[weight_base + in_c * params.output_channels];
            }
        }
    }

    let bias_idx = genome_idx * params.output_channels + out_c;
    let z_value = acc + biases[bias_idx];
    let linear_idx = (( (genome_idx * params.batch_size + batch_sample_idx) * params.output_height + out_y) * params.output_width + out_x) * params.output_channels + out_c;
    
    activations[linear_idx] = apply_activation(z_value, params.activation_type);
}
