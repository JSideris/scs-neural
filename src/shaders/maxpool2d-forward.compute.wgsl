struct PoolParams {
    batch_size: u32,
    input_height: u32,
    input_width: u32,
    channels: u32,
    output_height: u32,
    output_width: u32,
    pool_size: u32,
    stride: u32,
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_z = global_id.z; // batch_idx * channels + channel

    if (out_x >= params.output_width || out_y >= params.output_height) {
        return;
    }

    let batch_idx = out_z / params.channels;
    let c = out_z % params.channels;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let start_x = out_x * params.stride;
    let start_y = out_y * params.stride;

    var max_val: f32 = -1e38; // Very small number
    var max_idx: u32 = 0u;

    for (var ph: u32 = 0u; ph < params.pool_size; ph++) {
        let in_y = start_y + ph;
        if (in_y >= params.input_height) {
            continue;
        }

        for (var pw: u32 = 0u; pw < params.pool_size; pw++) {
            let in_x = start_x + pw;
            if (in_x >= params.input_width) {
                continue;
            }

            let linear_in_idx = ((batch_idx * params.input_height + in_y) * params.input_width + in_x) * params.channels + c;
            let val = inputs[linear_in_idx];
            
            if (val > max_val) {
                max_val = val;
                max_idx = linear_in_idx;
            }
        }
    }

    let linear_out_idx = ((batch_idx * params.output_height + out_y) * params.output_width + out_x) * params.channels + c;
    activations[linear_out_idx] = max_val;
    max_indices[linear_out_idx] = max_idx;
}
