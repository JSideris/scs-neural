@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&parameters)) {
        parameters[id.x] = parameters[id.x] - learning_rate * gradients[id.x];
    }
}
