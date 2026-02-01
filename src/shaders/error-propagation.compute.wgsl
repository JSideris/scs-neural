struct BackpropParams {
    batch_size: u32,
    current_layer_size: u32,
    next_layer_size: u32,
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_neurons = params.batch_size * params.current_layer_size;
    let thread_id = global_id.x;
    
    if (thread_id >= total_neurons) {
        return;
    }
    
    let batch_idx = thread_id / params.current_layer_size;
    let neuron_idx = thread_id % params.current_layer_size;
    let current_idx = batch_idx * params.current_layer_size + neuron_idx;
    
    var weighted_error_sum: f32 = 0.0;
    
    for (var next_neuron: u32 = 0u; next_neuron < params.next_layer_size; next_neuron = next_neuron + 1u) {
        let weight_idx = next_neuron * params.current_layer_size + neuron_idx;
        let weight = weights[weight_idx];
        
        let next_delta_idx = batch_idx * params.next_layer_size + next_neuron;
        let next_delta = next_layer_deltas[next_delta_idx];
        
        weighted_error_sum = weighted_error_sum + weight * next_delta;
    }
    
    current_layer_weighted_sums[current_idx] = weighted_error_sum;
}
