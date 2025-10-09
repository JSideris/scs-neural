struct GeneticLossParams {
    population_size: u32,
    batch_size: u32,
    output_size: u32,
    loss_type: u32,    // 0 = MSE, 1 = Cross-entropy (future)
    reduction: u32,    // 0 = mean, 1 = sum (unused in-shader; we atomic add totals)
    loss_multiplier: u32,
}

// Buffer bindings are added at runtime.

fn mse_loss(prediction: f32, targetValue: f32) -> f32 {
    let diff = prediction - targetValue;
    return diff * diff;
}

fn cross_entropy_loss(prediction: f32, targetValue: f32) -> f32 {
    let clamped_pred = clamp(prediction, 1e-7, 1.0 - 1e-7);
    return -targetValue * log(clamped_pred);
}

fn compute_element_loss(prediction: f32, targetValue: f32, loss_type: u32) -> f32 {
    switch (loss_type) {
        case 0u: { return mse_loss(prediction, targetValue); }
        case 1u: { return cross_entropy_loss(prediction, targetValue); }
        default: { return mse_loss(prediction, targetValue); }
    }
}

// Each thread handles one (genome, batch_sample)
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let total_samples = params.population_size * params.batch_size;
    let t = global_id.x;
    if (t >= total_samples) {
        return;
    }

    let samples_per_genome = params.batch_size;
    let genome_idx = t / samples_per_genome;
    let batch_idx = t % samples_per_genome;

    // Compute loss for this (genome, sample)
    let base = (genome_idx * params.batch_size + batch_idx) * params.output_size;
    var sample_loss: f32 = 0.0;
    for (var o: u32 = 0u; o < params.output_size; o = o + 1u) {
        let idx = base + o;
        let pred = predictions[idx];
        let targetValue = targets[idx];
        sample_loss = sample_loss + compute_element_loss(pred, targetValue, params.loss_type);
    }

    // For MSE, divide by output_size to get mean per sample
    if (params.loss_type == 0u) {
        sample_loss = sample_loss / f32(params.output_size);
    }

    // Atomically accumulate into this genome's total
    atomicAdd(&total_loss[genome_idx], u32(sample_loss * f32(params.loss_multiplier)));
}


