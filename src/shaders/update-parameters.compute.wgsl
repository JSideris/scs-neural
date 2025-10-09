// // optimizer.compute.wgsl - SGD weight update
// @group(0) @binding(0) var<uniform> learning_rate: f32;

// @group(1) @binding(0) var<storage, read> gradients: array<f32>;

// @group(2) @binding(0) var<storage, read_write> parameters: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&parameters)) {
        parameters[id.x] = parameters[id.x] - learning_rate * gradients[id.x];
    }
}