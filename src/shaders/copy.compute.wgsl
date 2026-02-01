struct CopyParams {
    size: u32,
}

// @group(0) @binding(0) var<storage, read> params: CopyParams;
// @group(0) @binding(1) var<storage, read> input_data: array<f32>;
// @group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < params.size) {
        output_data[global_id.x] = input_data[global_id.x];
    }
}
