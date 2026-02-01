// Genetic Dense Layer Forward Pass + Activation Function
// Computes: output = activation(weights × input + bias) across a population

// Parameters for layer configuration with population dimension
struct GeneticLayerParams {
    population_size: u32,
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    activation_type: u32,  // 0 = ReLU, 1 = Sigmoid, 2 = Linear, 3 = Tanh
}

// Buffers are added at runtime.

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    // Clamp input to prevent overflow
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

// Main compute kernel
// One thread per (genome, batch_sample, output_neuron)
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let total_outputs = params.population_size * params.batch_size * params.output_size;
    let thread_id = global_id.x;

    if (thread_id >= total_outputs) {
        return;
    }

    // Decode indices
    let outputs_per_genome = params.batch_size * params.output_size;
    let genome_idx = thread_id / outputs_per_genome;
    let rem = thread_id % outputs_per_genome;
    let batch_idx = rem / params.output_size;
    let output_idx = rem % params.output_size;

    // Calculate bases for memory access
    let weights_per_genome = params.output_size * params.input_size;
    let weight_base = genome_idx * weights_per_genome + output_idx * params.input_size;
    let input_base = (genome_idx * params.batch_size + batch_idx) * params.input_size;
    let output_linear_idx = (genome_idx * params.batch_size + batch_idx) * params.output_size + output_idx;

    // Matrix multiply accumulate
    var accumulator: f32 = 0.0;
    for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
        let w = weights[weight_base + i];
        let a = inputs[input_base + i];
        accumulator = accumulator + w * a;
    }

    // Add bias (per-genome)
    let bias_idx = genome_idx * params.output_size + output_idx;
    let z_value = accumulator + biases[bias_idx];

    // Store activation
    let activated_value = apply_activation(z_value, params.activation_type);
    activations[output_linear_idx] = activated_value;
}


