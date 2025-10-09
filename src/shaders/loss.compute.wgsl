struct LossParams {
    batch_size: u32,
    output_size: u32,
    loss_type: u32,    // 0 = MSE, 1 = Cross-entropy (future)
    reduction: u32,    // 0 = mean, 1 = sum, 2 = none (per-sample)
    loss_multiplier: u32,
}

// // Buffer bindings - added at runtime.
// @group(0) @binding(0) var<uniform> params: LossParams;
// @group(0) @binding(1) var<storage, read> predictions: array<f32>;        // [batch_size, output_size]
// @group(0) @binding(2) var<storage, read> targets: array<f32>;            // [batch_size, output_size]
// @group(0) @binding(3) var<storage, read_write> total_loss: array<atomic<u32>>;     // [1] - reduced total loss
// @group(0) @binding(4) var<storage, read_write> sample_losses: array<f32>;  // [batch_size] - per-sample losses

// Loss functions
fn mse_loss(prediction: f32, targetValue: f32) -> f32 {
    let diff = prediction - targetValue;
    return diff * diff;
}

fn cross_entropy_loss(prediction: f32, targetValue: f32) -> f32 {
    // Clamp prediction to prevent log(0)
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

// Workgroup shared memory for reduction
var<workgroup> shared_data: array<f32, 64>;

// Main compute kernel - each thread processes one batch sample
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = global_id.x;
    let local_idx = local_id.x;
    
    var sample_loss: f32 = 0.0;
    
    // Compute loss for this batch sample (if within bounds)
    if (batch_idx < params.batch_size) {
        let sample_base = batch_idx * params.output_size;
        
        // Sum loss across all outputs for this sample
        for (var output_idx: u32 = 0u; output_idx < params.output_size; output_idx = output_idx + 1u) {
            let idx = sample_base + output_idx;
            let pred = predictions[idx];
            let targetValue = targets[idx];
            sample_loss = sample_loss + compute_element_loss(pred, targetValue, params.loss_type);
        }
        
        // For MSE, divide by output_size to get mean over outputs
        if (params.loss_type == 0u) {
            sample_loss = sample_loss / f32(params.output_size);
        }
        
        // Store per-sample loss
        // We don't actually need this.
        // sample_losses[batch_idx] = sample_loss;
    }
    
    // === Reduction to compute total loss ===
    
    // Load sample loss into shared memory
    shared_data[local_idx] = select(0.0, sample_loss, batch_idx < params.batch_size);
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // First thread in workgroup writes the workgroup sum
    if (local_idx == 0u) {
        let workgroup_sum = shared_data[0];
        
        // Atomic add to accumulate across workgroups
        atomicAdd(&total_loss[0], u32(workgroup_sum * f32(params.loss_multiplier)));
    }
    
    // Note: The calling code should zero out total_loss[0] before dispatch
    // and apply final reduction (mean vs sum) after all workgroups complete
}