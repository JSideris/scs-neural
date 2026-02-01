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
    let in_x = global_id.x;
    let in_y = global_id.y;
    let in_z = global_id.z; // batch_idx * channels + channel

    if (in_x >= params.input_width || in_y >= params.input_height) {
        return;
    }

    let batch_idx = in_z / params.channels;
    let c = in_z % params.channels;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let current_in_idx = ((batch_idx * params.input_height + in_y) * params.input_width + in_x) * params.channels + c;
    var error: f32 = 0.0;

    // Search all output pools that this input could belong to
    let start_out_y = u32(max(0, i32(in_y) - i32(params.pool_size) + 1)) / params.stride;
    let end_out_y = in_y / params.stride;
    
    let start_out_x = u32(max(0, i32(in_x) - i32(params.pool_size) + 1)) / params.stride;
    let end_out_x = in_x / params.stride;

    for (var out_y = start_out_y; out_y <= end_out_y && out_y < params.output_height; out_y++) {
        for (var out_x = start_out_x; out_x <= end_out_x && out_x < params.output_width; out_x++) {
            let out_idx = ((batch_idx * params.output_height + out_y) * params.output_width + out_x) * params.channels + c;
            if (max_indices[out_idx] == current_in_idx) {
                error += next_layer_deltas[out_idx];
            }
        }
    }

    current_layer_weighted_sums[current_in_idx] = error;
}
