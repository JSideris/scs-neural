(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))e(a);new MutationObserver(a=>{for(const r of a)if(r.type==="childList")for(const n of r.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&e(n)}).observe(document,{childList:!0,subtree:!0});function t(a){const r={};return a.integrity&&(r.integrity=a.integrity),a.referrerPolicy&&(r.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?r.credentials="include":a.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function e(a){if(a.ep)return;a.ep=!0;const r=t(a);fetch(a.href,r)}})();const U=new Set(["var","let","const","if","else","for","while","return","true","false","void","fn","struct","uniform","storage","workgroup","texture","sampler","array","atomic","bool","f32","i32","u32"]);function k(u){return/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(u)?!U.has(u):!1}var G=!1;class p{static get isInitialized(){return G}static async initialize(){if(!p.isInitialized){if(G=!0,p.presentationFormat=navigator.gpu.getPreferredCanvasFormat(),p.adapter=await navigator.gpu.requestAdapter(),!p.adapter)throw new Error("Failed to initialize WebGPU adapter. Ensure your system and browser support WebGPU.");if(p.device=await p.adapter.requestDevice(),!p.device)throw new Error("Failed to acquire a WebGPU device.");return p.device}}constructor(i){var t;this.executionCount=0,this.time=0,this.lastTime=0,this._bindGroupsByLayout=[],this.props=i,this.props.useExecutionCountBuffer!==!1&&(this.props.useExecutionCountBuffer=!0),this.props.useTimeBuffer!==!1&&(this.props.useTimeBuffer=!0),this.props.useExecutionCountBuffer&&!this.props.executionCountBufferName&&(this.props.executionCountBufferName="execution_count"),this.props.useTimeBuffer&&!this.props.timeBufferName&&(this.props.timeBufferName="time"),this.props.bindingLayouts||(this.props.bindingLayouts=[]);{if(!p.isInitialized)throw new Error("Call Shader.initialize() before instantiating a shader pipeline.");if(!((t=this.props.bindingLayouts)===null||t===void 0)&&t.length)for(let e=0;e<i.bindingLayouts.length;e++){let a=i.bindingLayouts[e],r=Object.keys(a)[0];for(let n of Object.keys(a))for(let s=0;s<a[n].length;s++){let o=a[n][s];if(!o?.binding)throw new Error(`Binding ${o?.name} in group ${n} has no binding.`);if(o.type!=a[r][s].type)throw new Error(`Binding type mismatch in group ${n}: expected '${a[r][s].type}', got '${o.type}'`);if(o.name!=a[r][s].name)throw new Error(`Binding name mismatch in group ${n}: expected '${a[r][s].name}', got '${o.name}'`);if(o.binding.baseType!=a[r][s].binding.baseType)throw new Error(`Binding baseType mismatch in group ${n}: expected '${a[r][s].binding.baseType}', got '${o.binding.baseType}'`);if(o.binding.dataType!=a[r][s].binding.dataType)throw new Error(`Binding dataType mismatch in group ${n}: expected '${a[r][s].binding.dataType}', got '${o.binding.dataType}'`)}}if(this.props.useExecutionCountBuffer&&!k(this.props.executionCountBufferName))throw new Error("Invalid executionCountBufferName. Must be a valid WGSL variable name.");if(this.props.useTimeBuffer&&!k(this.props.timeBufferName))throw new Error("Invalid timeBufferName. Must be a valid WGSL variable name.")}}_setupShader(i){let t="";t=this._initialize()+(typeof this.props.code=="string"?this.props.code:this.props.code.join(`
`));let e=[];for(let a=0;a<this.props.bindingLayouts.length;a++){this._bindGroupsByLayout.push({});let r=this.props.bindingLayouts[a][Object.keys(this.props.bindingLayouts[a])[0]],n=p.device.createBindGroupLayout({entries:r.map((s,o)=>({binding:o,visibility:i,buffer:s.type=="write-only-texture"?void 0:{type:s.type},texture:s.type=="write-only-texture"?{sampleType:"float",viewDimension:"2d"}:void 0}))});e.push(n)}this._configurePipeline(t,e);for(let a=0;a<this.props.bindingLayouts.length;a++){let r=this.props.bindingLayouts[a],n=Object.keys(r);for(let s=0;s<n.length;s++){let o=n[s],d=r[o],h=p.device.createBindGroup({layout:e[a],entries:d.map((g,y)=>({binding:y,resource:{buffer:g.binding.buffer,label:o}}))});this._bindGroupsByLayout[a][o]=h}}}_initialize(){var i,t;let e="",a=typeof this.props.code=="string"?this.props.code:this.props.code.join(`
`),r={};{let n=a.split(`
`);for(let s of n){let o=s.trim();if(o.startsWith("//#!binding")){let d=o.split(" ");if(d.length>=3){let h=d[1],g=d[2];r[h]=g}}}}if(this.props.bindingLayouts)for(let n=0;n<this.props.bindingLayouts.length;n++){let s=this.props.bindingLayouts[n][Object.keys(this.props.bindingLayouts[n])[0]];if(s)for(let o=0;o<s.length;o++){let d=s[o].name,h=r[d]||d,g=(i=s[o].binding)===null||i===void 0?void 0:i.dataType;if(g==="struct"&&(g=(t=s[o].binding)===null||t===void 0?void 0:t.structName),!g){console.warn(`No data type found for binding ${d}.`);continue}let y=s[o].type,_=`@group(${n}) @binding(${o}) ${y=="var"?"var":`var<${y=="read-only-storage"?"storage, read":y=="storage"||y=="write-only-texture"?"storage, read_write":"uniform"}>`} ${h}: ${g};\r
`;e+=_}}return e+=`\r
`,e}dispose(){}}function N(u){var i,t;if(u.dataType==="struct"){if(!u.fields)throw new Error("Fields must be provided for struct types");return F(u.fields,u.size||1)}let e=4;if(u.dataType.startsWith("vec2")?e=8:u.dataType.startsWith("vec3")?e=12:u.dataType.startsWith("vec4")&&(e=16),u.dataType.startsWith("mat4x4")&&(e=64),u.dataType.startsWith("array")){const a=(i=u.dataType.match(/array<(.+)>/))===null||i===void 0?void 0:i[1];let r=4;if(a&&(a.startsWith("vec2")?r=8:a.startsWith("vec3")?r=12:a.startsWith("vec4")?r=16:a.startsWith("mat4x4")?r=64:(a==="u32"||a==="i32"||a==="f32")&&(r=4)),u.size)e=r*u.size;else throw new Error("Size must be provided for array types")}return u.dataType.startsWith("texture_2d")&&(e=4*((t=u.size)!==null&&t!==void 0?t:1)),e}function F(u,i=1){let t=0;for(const r of u){const n=A(r.dataType),s=P(r.dataType);t=Math.ceil(t/n)*n,r.offset!==void 0&&(t=Math.max(t,r.offset)),t+=s}const e=Math.max(...u.map(r=>A(r.dataType)));return Math.ceil(t/e)*e*i}function A(u){return u==="f32"||u==="u32"||u==="i32"?4:u.startsWith("vec2")?8:u.startsWith("vec3")||u.startsWith("vec4")||u.startsWith("mat4x4")?16:4}function P(u){return u==="f32"||u==="u32"||u==="i32"?4:u.startsWith("vec2")?8:u.startsWith("vec3")?12:u.startsWith("vec4")?16:u.startsWith("mat4x4")?64:4}class D{constructor(i,t){if(this.sizeElements=1,!p.isInitialized)throw new Error("Call `Shader.initialize()` before creating buffers.");this.props=t;let e=i;if(this.dataType=t.dataType,t.canCopySrc&&(e|=GPUBufferUsage.COPY_SRC),t.canCopyDst&&(e|=GPUBufferUsage.COPY_DST),t.canQueryResolve&&(e|=GPUBufferUsage.QUERY_RESOLVE),t.dataType==="struct"?(this.structFields=t.fields,this.baseType=this.determineStructBaseType(t.fields),this.sizeElements=t.size||1,this.structName=t.structName):this.sizeElements=t.size||1,this.sizeBytes=N({dataType:t.dataType,size:this.sizeElements,fields:this.structFields}),this.buffer=p.device.createBuffer({size:this.sizeBytes,usage:e,mappedAtCreation:!!t.initialValue}),t.dataType!=="struct"&&(t.dataType.indexOf("f32")>-1?this.baseType="float":t.dataType.indexOf("u32")>-1?this.baseType="uint":t.dataType.indexOf("i32")>-1&&(this.baseType="int")),t.initialValue){if(t.dataType==="struct"){const a=this.serializeStructFromArray(t.initialValue,this.structFields);new Uint8Array(this.buffer.getMappedRange()).set(new Uint8Array(a))}else this.baseType=="float"?new Float32Array(this.buffer.getMappedRange()).set(t.initialValue):this.baseType=="uint"?new Uint32Array(this.buffer.getMappedRange()).set(t.initialValue):this.baseType=="int"&&new Int32Array(this.buffer.getMappedRange()).set(t.initialValue);this.buffer.unmap()}}serializeStructFromArray(i,t){const e=new ArrayBuffer(this.sizeBytes),a=new DataView(e);let r=0,n=0;for(const s of t){const o=A(s.dataType);n=Math.ceil(n/o)*o,s.offset!==void 0&&(n=Math.max(n,s.offset));const d=this.getFieldElementCount(s.dataType),h=Array.from(i).slice(r,r+d);this.writeFieldToBuffer(a,n,s.dataType,h),r+=d,n+=P(s.dataType)}return e}getFieldElementCount(i){return i==="f32"||i==="u32"||i==="i32"?1:i.startsWith("vec2")?2:i.startsWith("vec3")?3:i.startsWith("vec4")?4:i.startsWith("mat4x4")?16:1}writeFieldToBuffer(i,t,e,a){a.forEach((r,n)=>{e.includes("f32")?i.setFloat32(t+n*4,r,!0):e.includes("u32")?i.setUint32(t+n*4,r,!0):e.includes("i32")&&i.setInt32(t+n*4,r,!0)})}determineStructBaseType(i){const t=new Set(i.map(e=>e.dataType.includes("f32")?"float":e.dataType.includes("u32")?"uint":e.dataType.includes("i32")?"int":"unknown"));return t.size===1?Array.from(t)[0]:"mixed"}write(i,t=0){if(!this.props.canCopyDst)throw new Error("Buffer is not writable. Set `canCopyDst` to `true` in the buffer props.");const e=t*this.sizeBytes/this.sizeElements;p.device.queue.writeBuffer(this.buffer,e,i)}async read(i=0,t=this.sizeElements){if(!this.props.canCopySrc)throw new Error("Buffer is not readable. Set `canCopySrc` to `true` in the buffer props.");const e=t*this.sizeBytes/this.sizeElements,a=i*this.sizeBytes/this.sizeElements,r=p.device.createBuffer({size:e,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),n=p.device.createCommandEncoder();n.copyBufferToBuffer(this.buffer,i,r,a,e);const s=n.finish();p.device.queue.submit([s]),await r.mapAsync(GPUMapMode.READ);const o=r.getMappedRange();let d;return this.dataType==="struct"?d=this.deserializeStructToArray(o,this.structFields):this.baseType==="float"?d=new Float32Array(new Float32Array(o)):this.baseType==="uint"?d=new Uint32Array(new Uint32Array(o)):this.baseType==="int"&&(d=new Int32Array(new Int32Array(o))),r.unmap(),r.destroy(),d}deserializeStructToArray(i,t){const e=new DataView(i),a=[];let r=0;for(const n of t){const s=A(n.dataType);r=Math.ceil(r/s)*s,n.offset!==void 0&&(r=Math.max(r,n.offset));const o=this.readFieldFromBuffer(e,r,n.dataType);a.push(...Array.isArray(o)?o:[o]),r+=P(n.dataType)}return this.baseType==="float"?new Float32Array(a):this.baseType==="uint"?new Uint32Array(a):this.baseType==="int"?new Int32Array(a):new Float32Array(a)}readFieldFromBuffer(i,t,e){if(e==="f32")return i.getFloat32(t,!0);if(e==="u32")return i.getUint32(t,!0);if(e==="i32")return i.getInt32(t,!0);if(e.startsWith("vec")){const a=parseInt(e[3]),r=[];for(let n=0;n<a;n++)e.includes("f32")?r.push(i.getFloat32(t+n*4,!0)):e.includes("u32")?r.push(i.getUint32(t+n*4,!0)):e.includes("i32")&&r.push(i.getInt32(t+n*4,!0));return r}return 0}dispose(){this.buffer&&(this.buffer.destroy(),this.buffer=null)}}class f extends D{constructor(i){super(GPUBufferUsage.STORAGE,i)}}class E extends D{constructor(i){super(GPUBufferUsage.UNIFORM,i)}}class v extends p{constructor(i){if(super(i),this.props=i,this.props.useTimeBuffer&&(this.timeBuffer=new E({dataType:"f32",canCopyDst:!0})),this.props.useExecutionCountBuffer&&(this.executionCountBuffer=new E({dataType:"u32",canCopyDst:!0})),this.timeBuffer||this.executionCountBuffer){let t=[];this.timeBuffer&&t.push({type:"uniform",name:this.props.timeBufferName,binding:this.timeBuffer}),this.executionCountBuffer&&t.push({type:"uniform",name:this.props.executionCountBufferName,binding:this.executionCountBuffer}),this.props.bindingLayouts.push({default:t})}super._setupShader(GPUShaderStage.COMPUTE)}dispatch(i){var t,e,a,r;if(this.props.useExecutionCountBuffer&&this.executionCountBuffer.write(new Uint32Array([this.executionCount++])),this.props.useTimeBuffer){let o=performance?performance.now()/1e3:Date.now()/1e3;this.lastTime||(this.lastTime=o),this.time+=o-this.lastTime,this.lastTime=o,this.timeBuffer.write(new Float32Array([this.time]))}let n=p.device.createCommandEncoder(),s=n.beginComputePass();s.setPipeline(this.pipeline);for(let o=0;o<this._bindGroupsByLayout.length;o++){let d=this._bindGroupsByLayout[o],h=d[!((t=i?.bindGroups)===null||t===void 0)&&t[o]?(e=i?.bindGroups)===null||e===void 0?void 0:e[o]:Object.keys(d)[0]];if(!h){console.warn(`Bind group ${!((a=i?.bindGroups)===null||a===void 0)&&a[o]?(r=i?.bindGroups)===null||r===void 0?void 0:r[o]:Object.keys(d)[0]} not found for layout ${o}.`);continue}s.setBindGroup(o,h)}s.dispatchWorkgroups(this.props.workgroupCount[0],this.props.workgroupCount[1],this.props.workgroupCount[2]),s.end(),p.device.queue.submit([n.finish()])}_configurePipeline(i,t){let e=p.device.createShaderModule({code:i});this.pipeline=p.device.createComputePipeline({layout:p.device.createPipelineLayout({bindGroupLayouts:t}),compute:{module:e,entryPoint:"main"}})}}const M=`struct LossParams {
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
}`,V=`struct BackpropParams {
    batch_size: u32,
    current_layer_size: u32,
    next_layer_size: u32,
    activation_type: u32,    // 0 = ReLU, 1 = Sigmoid, 2 = Linear
    is_output_layer: u32,    // 1 if this is output layer, 0 for hidden layers
}

// // Buffer bindings
// // Group 0: Ping-pong error buffers for error gradients.
// @group(0) @binding(0) var<storage, read> next_layer_errors: array<f32>;        // [batch_size, next_layer_size]
// @group(0) @binding(1) var<storage, read_write> current_layer_errors: array<f32>; // [batch_size, current_layer_size]

// // Group 1: Layer-specific data
// @group(1) @binding(0) var<storage, read> weights: array<f32>;                  // [next_layer_size, current_layer_size]
// @group(1) @binding(1) var<storage, read> z_values: array<f32>;                 // [batch_size, current_layer_size]

// // Group 2: Non-swappy data
// @group(2) @binding(0) var<uniform> params: BackpropParams;
// @group(2) @binding(1) var<storage, read> predictions: array<f32>;              // [batch_size, output_size]
// @group(2) @binding(2) var<storage, read> targets: array<f32>;                  // [batch_size, output_size]

// Activation derivative functions
fn relu_derivative(z: f32) -> f32 {
    return select(0.0, 1.0, z > 0.0);
}

fn sigmoid_derivative(z: f32) -> f32 {
    // Use the identity: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    let clamped_z = clamp(z, -88.0, 88.0);
    let sigmoid_z = 1.0 / (1.0 + exp(-clamped_z));
    return sigmoid_z * (1.0 - sigmoid_z);
}

fn linear_derivative(z: f32) -> f32 {
    return 1.0;
}

fn tanh_derivative(z: f32) -> f32 {
    let clamped_z = clamp(z, -88.0, 88.0);
    let tanh_z = tanh(clamped_z);
    return 1.0 - tanh_z * tanh_z;
}

fn get_activation_derivative(z: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return relu_derivative(z); }
        case 1u: { return sigmoid_derivative(z); }
        case 2u: { return linear_derivative(z); }
        case 3u: { return tanh_derivative(z); }
        // case 4u: { return softmax_derivative(z); } // A bit more tricky. Need to think about it.
        default: { return linear_derivative(z); }
    }
}

// Compute loss derivative for output layer (MSE case)
fn mse_loss_derivative(prediction: f32, target_value: f32, batch_size: u32) -> f32 {
    return 2.0 * (prediction - target_value) / f32(batch_size);
}

// Main compute kernel
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_neurons = params.batch_size * params.current_layer_size;
    let thread_id = global_id.x;
    
    // Early exit if thread is out of bounds
    if (thread_id >= total_neurons) {
        return;
    }
    
    // Decode which batch sample and neuron this thread handles
    let batch_idx = thread_id / params.current_layer_size;
    let neuron_idx = thread_id % params.current_layer_size;
    let current_idx = batch_idx * params.current_layer_size + neuron_idx;
    
    var error: f32 = 0.0;
    
    if (params.is_output_layer == 1u) {
        // === OUTPUT LAYER ERROR COMPUTATION ===
        // Error = loss_derivative * activation_derivative
        
        let prediction = predictions[current_idx];
        let target_value = targets[current_idx];
        let z = z_values[current_idx];
        
        // Compute loss derivative (assuming MSE for now)
        let loss_grad = mse_loss_derivative(prediction, target_value, params.batch_size);
        
        // Compute activation derivative
        let activation_grad = get_activation_derivative(z, params.activation_type);
        
        // Combine: dL/dz = dL/da * da/dz
        error = loss_grad * activation_grad;
        
    } else {
        // === HIDDEN LAYER ERROR PROPAGATION ===
        // Error = (weights^T × next_errors) * activation_derivative
        
        // Step 1: Compute weights^T × next_errors
        // This is matrix-vector multiplication where we sum over the next layer
        var weighted_error_sum: f32 = 0.0;
        
        for (var next_neuron: u32 = 0u; next_neuron < params.next_layer_size; next_neuron = next_neuron + 1u) {
            // Weight matrix is stored as [next_layer_size, current_layer_size]
            // So weights[next_neuron][current_neuron] = weights[next_neuron * current_layer_size + neuron_idx]
            let weight_idx = next_neuron * params.current_layer_size + neuron_idx;
            let weight = weights[weight_idx];
            
            // Get the error from the next layer for this batch sample and next layer neuron
            let next_error_idx = batch_idx * params.next_layer_size + next_neuron;
            let next_error = next_layer_errors[next_error_idx];
            
            weighted_error_sum = weighted_error_sum + weight * next_error;
        }
        
        // Step 2: Multiply by activation derivative
        let z = z_values[current_idx];
        let activation_grad = get_activation_derivative(z, params.activation_type);
        
        error = weighted_error_sum * activation_grad;
    }
    
    // Write computed error to output buffer
    current_layer_errors[current_idx] = error;
}
`,W=`struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,     // 0 = overwrite, 1 = accumulate (for mini-batch accumulation)
}

// // Buffer bindings
// // Group 0: Input data for gradient computation
// @group(0) @binding(0) var<storage, read> errors: array<f32>;           // [batch_size, output_size]
// @group(0) @binding(1) var<storage, read> input_activations: array<f32>; // [batch_size, input_size]

// // Group 1:Non-swappy data
// @group(1) @binding(0) var<uniform> params: GradientParams;
// @group(1) @binding(1) var<storage, read_write> weight_gradients: array<f32>; // [output_size, input_size]

// Workgroup shared memory for reduction operations
var<workgroup> shared_weight_grad: array<f32, 256>;

// Main compute kernel - compute gradients for one weight matrix element or bias
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let total_weights = params.output_size * params.input_size;
    let weight_idx = global_id.x;
    let local_idx = local_id.x;
    
    // === WEIGHT GRADIENTS COMPUTATION ===
    if (weight_idx < total_weights) {
        // Decode which weight this thread is responsible for
        let output_neuron = weight_idx / params.input_size;
        let input_neuron = weight_idx % params.input_size;
        
        // Compute gradient for this weight: sum over batch of (error * activation)
        var weight_gradient: f32 = 0.0;
        
        for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx = batch_idx + 1u) {
            let error_idx = batch_idx * params.output_size + output_neuron;
            let activation_idx = batch_idx * params.input_size + input_neuron;
            
            let error = errors[error_idx];
            let activation = input_activations[activation_idx];
            
            weight_gradient = weight_gradient + error * activation;
        }

        // Division by batch_size already included in error values
        // weight_gradient = weight_gradient / f32(params.batch_size);
        
        // Write or accumulate the gradient
        if (params.accumulate == 0u) {
            weight_gradients[weight_idx] = weight_gradient;
        } else {
            weight_gradients[weight_idx] = weight_gradients[weight_idx] + weight_gradient;
        }
    }
}`,I=`struct GradientParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    accumulate: u32,     // 0 = overwrite, 1 = accumulate (for mini-batch accumulation)
}

// Buffer bindings - added at runtime.
// Group 0: Input data for gradient computation
// @group(0) @binding(0) var<storage, read> errors: array<f32>;           // [batch_size, output_size]
// @group(0) @binding(1) var<storage, read> input_activations: array<f32>; // [batch_size, input_size]

// Group 1: Output gradients
// @group(1) @binding(0) var<uniform> params: GradientParams;
// @group(1) @binding(1) var<storage, read_write> bias_gradients: array<f32>;   // [output_size]

// Workgroup shared memory for reduction operations
var<workgroup> shared_bias_grad: array<f32, 256>;

// Alternative: Optimized bias computation using shared memory reduction
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                                   @builtin(local_invocation_id) local_id: vec3<u32>,
                                   @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let local_idx = local_id.x;
    let workgroup_idx = workgroup_id.x;
    
    // Each workgroup handles one output neuron
    let output_neuron = workgroup_idx;
    
    // Hoist uniform values to locals for better uniformity
    let output_size = params.output_size;
    let batch_size = params.batch_size;
    let accumulate = params.accumulate;
    
    let is_valid = (output_neuron < output_size);
    
    // Each thread in the workgroup handles some batch samples
    var local_sum: f32 = 0.0;
    
    // Only compute if valid
    if (is_valid) {
        // Stride through batch samples
        var batch_idx = local_idx;
        while (batch_idx < batch_size) {
            let error_idx = batch_idx * output_size + output_neuron;
            local_sum += errors[error_idx];
            batch_idx += 256u;
        }
    }
    
    // Store in shared memory (invalid workgroups store 0.0)
    shared_bias_grad[local_idx] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_bias_grad[local_idx] += shared_bias_grad[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Division by batch_size already included in error values
    // shared_bias_grad[0] = shared_bias_grad[0] / f32(params.batch_size);
    
    // First thread writes the result (only if valid)
    if (local_idx == 0u && is_valid) {
        let total_sum = shared_bias_grad[0];
        if (accumulate == 0u) {
            bias_gradients[output_neuron] = total_sum;
        } else {
            bias_gradients[output_neuron] += total_sum;
        }
    }
}`,O=`// // optimizer.compute.wgsl - SGD weight update
// @group(0) @binding(0) var<uniform> learning_rate: f32;

// @group(1) @binding(0) var<storage, read> gradients: array<f32>;

// @group(2) @binding(0) var<storage, read_write> parameters: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&parameters)) {
        parameters[id.x] = parameters[id.x] - learning_rate * gradients[id.x];
    }
}`,$=`// Genetic Dense Layer Forward Pass + Activation Function
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

    // Store z and activation
    z_values[output_linear_idx] = z_value;
    let activated_value = apply_activation(z_value, params.activation_type);
    activations[output_linear_idx] = activated_value;
}


`,R=`struct GeneticLossParams {
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


`;class z{static initXavier(i,t){const e=Math.sqrt(1/i);return z.randomArray(i*t,-e,e)}static initHe(i,t){const e=Math.sqrt(2/i);return z.randomArray(i*t,-e,e)}static initUniform(i,t,e,a){return z.randomArray(i*t,e,a)}static initZero(i,t){return new Float32Array(i*t)}static randomArray(i,t,e){const a=new Float32Array(i),r=e-t;for(let n=0;n<i;n++)a[n]=Math.random()*r+t;return a}}var L=(u=>(u[u.RELU=0]="RELU",u[u.SIGMOID=1]="SIGMOID",u[u.LINEAR=2]="LINEAR",u[u.TANH=3]="TANH",u[u.SOFTMAX=4]="SOFTMAX",u))(L||{});const T=1e4;class q{forwardPassShader;forwardPassParamsBuffer;lossParamsBuffer;backwardPassParamsBuffer;gradientParamsBuffer;learningRateBuffer;layerBuffers=[];trainingDataBuffers=[];testActivationsBufferA;testActivationsBufferB;testZValuesBuffer;targetsBuffer;totalBatchLossBuffer;errorGradientsABuffer;errorGradientsBBuffer;isInitialized=!1;layerSizes;trainingBatchSize;testingBatchSize;lossShader;backwardErrorShader;weightGradientComputationShader;biasGradientComputationShader;updateParametersShader;hiddenActivationType;outputActivationType;get outputSize(){return this.layerSizes?.length?this.layerSizes[this.layerSizes.length-1]:0}get inputSize(){return this.layerSizes?.length?this.layerSizes[0]:0}constructor(i){if(!i.layerSizes||i.layerSizes.length<2)throw new Error("Layer sizes must be an array of at least 2 numbers.");if(i.layerSizes.some(t=>t<1))throw new Error("Layer sizes must be greater than 0.");this.layerSizes=i.layerSizes,this.testingBatchSize=i.testingBatchSize??1,this.trainingBatchSize=i.trainingBatchSize??1,this.hiddenActivationType=i.hiddenActivationType??L.RELU,this.outputActivationType=i.outputActivationType??L.RELU}async initialize(i="xavier"){if(this.isInitialized){console.warn("NeuralNetwork already initialized.");return}await p.initialize();let t=Math.max(...this.layerSizes);{this.testActivationsBufferA=new f({dataType:"array<f32>",size:this.testingBatchSize*t,canCopyDst:!0,canCopySrc:this.layerSizes.length%2==1}),this.testActivationsBufferB=new f({dataType:"array<f32>",size:this.testingBatchSize*t,canCopyDst:!1,canCopySrc:this.layerSizes.length%2==0}),this.testZValuesBuffer=new f({dataType:"array<f32>",size:this.testingBatchSize*t});for(let e=0;e<this.layerSizes.length;e++){let a,r;if(e>0){let n=this.layerSizes[e-1],s=this.layerSizes[e];switch(i){case"xavier":a=z.initXavier(n,s);break;case"he":a=z.initHe(n,s);break;case"uniform":a=z.initUniform(n,s,-.5,.5);break;case"zero":a=z.initZero(n,s);break;default:throw new Error(`Unknown initialization method: ${i}`)}r=i==="zero"?z.initZero(1,s):z.initUniform(1,s,-.1,.1)}this.layerBuffers.push({weights:e>0?new f({dataType:"array<f32>",size:this.layerSizes[e-1]*this.layerSizes[e],initialValue:a,canCopyDst:!0,canCopySrc:!0}):null,biases:e>0?new f({dataType:"array<f32>",size:this.layerSizes[e],initialValue:r,canCopyDst:!0,canCopySrc:!0}):null,errors:e>0?new f({dataType:"array<f32>",size:this.trainingBatchSize*this.layerSizes[e]}):null,weightGradients:e>0?new f({dataType:"array<f32>",size:this.layerSizes[e-1]*this.layerSizes[e]}):null,biasGradients:e>0?new f({dataType:"array<f32>",size:this.layerSizes[e]}):null}),this.trainingDataBuffers.push({trainingActivations:new f({dataType:"array<f32>",size:this.trainingBatchSize*this.layerSizes[e],canCopyDst:e==0,canCopySrc:e==this.layerSizes.length-1}),trainingZValues:e>0?new f({dataType:"array<f32>",size:this.trainingBatchSize*this.layerSizes[e]}):null})}}this.errorGradientsABuffer=new f({dataType:"array<f32>",size:this.trainingBatchSize*t}),this.errorGradientsBBuffer=new f({dataType:"array<f32>",size:this.trainingBatchSize*t}),this.forwardPassParamsBuffer=new f({dataType:"struct",structName:"LayerParams",fields:[{name:"batch_size",dataType:"u32"},{name:"input_size",dataType:"u32"},{name:"output_size",dataType:"u32"},{name:"activation_type",dataType:"u32"}],canCopyDst:!0}),this.lossParamsBuffer=new f({dataType:"struct",structName:"LossParams",fields:[{name:"batch_size",dataType:"u32"},{name:"output_size",dataType:"u32"},{name:"loss_type",dataType:"u32"},{name:"reduction",dataType:"u32"},{name:"loss_multiplier",dataType:"u32"}],canCopyDst:!0,canCopySrc:!0,initialValue:[this.trainingBatchSize,this.outputSize,0,0]}),this.backwardPassParamsBuffer=new f({dataType:"struct",structName:"BackpropParams",fields:[{name:"batch_size",dataType:"u32"},{name:"current_layer_size",dataType:"u32"},{name:"next_layer_size",dataType:"u32"},{name:"activation_type",dataType:"u32"},{name:"is_output_layer",dataType:"u32"}],canCopyDst:!0}),this.gradientParamsBuffer=new f({dataType:"struct",structName:"GradientParams",fields:[{name:"batch_size",dataType:"u32"},{name:"input_size",dataType:"u32"},{name:"output_size",dataType:"u32"},{name:"accumulate",dataType:"u32"}],canCopyDst:!0}),this.learningRateBuffer=new E({dataType:"f32",canCopyDst:!0,initialValue:new Float32Array([.01])}),this.targetsBuffer=new f({dataType:"array<f32>",size:this.outputSize*this.trainingBatchSize,canCopyDst:!0,canCopySrc:!0,initialValue:new Float32Array(this.outputSize*this.trainingBatchSize).fill(0)}),this.totalBatchLossBuffer=new f({dataType:"array<atomic<u32>>",size:1,canCopyDst:!0,canCopySrc:!0,initialValue:new Uint32Array(1).fill(0)}),this.forwardPassShader=new v({useExecutionCountBuffer:!1,useTimeBuffer:!1,code:$,workgroupCount:[Math.ceil(t/64),1],bindingLayouts:[{default:[{binding:this.forwardPassParamsBuffer,name:"params",type:"storage"}]},this.layerBuffers.reduce((e,{weights:a,biases:r},n)=>(n>0&&(e[`layer_${n}`]=[{binding:a,name:"weights",type:"storage"},{binding:r,name:"biases",type:"storage"}]),e),{}),this.trainingDataBuffers.reduce((e,{trainingActivations:a},r)=>(r>0&&(e[`training_layer_${r}`]=[{binding:this.trainingDataBuffers[r-1].trainingActivations,name:"inputs",type:"storage"},{binding:a,name:"activations",type:"storage"}]),e),{test_layer_0:[{binding:this.testActivationsBufferA,name:"inputs",type:"storage"},{binding:this.testActivationsBufferB,name:"activations",type:"storage"}],test_layer_1:[{binding:this.testActivationsBufferB,name:"inputs",type:"storage"},{binding:this.testActivationsBufferA,name:"activations",type:"storage"}]}),this.trainingDataBuffers.reduce((e,{trainingZValues:a},r)=>(r>0&&(e[`training_layer_${r}`]=[{binding:a,name:"z_values",type:"storage"}]),e),{test_layer:[{binding:this.testZValuesBuffer,name:"z_values",type:"storage"}]})]}),this.lossShader=new v({code:M,workgroupCount:[Math.ceil(t/64),1],bindingLayouts:[{default:[{binding:this.lossParamsBuffer,name:"params",type:"storage"},{binding:this.trainingDataBuffers[this.trainingDataBuffers.length-1].trainingActivations,name:"predictions",type:"storage"},{binding:this.targetsBuffer,name:"targets",type:"storage"},{binding:this.totalBatchLossBuffer,name:"total_loss",type:"storage"}]}]}),this.backwardErrorShader=new v({code:V,workgroupCount:[Math.ceil(t/64),1],bindingLayouts:[{group0:[{binding:this.errorGradientsABuffer,name:"next_layer_errors",type:"storage"},{binding:this.errorGradientsBBuffer,name:"current_layer_errors",type:"storage"}],group1:[{binding:this.errorGradientsBBuffer,name:"next_layer_errors",type:"storage"},{binding:this.errorGradientsABuffer,name:"current_layer_errors",type:"storage"}]},this.layerBuffers.reduce((e,{weights:a},r)=>(r>0&&(e[`layer${r}`]=[{binding:a,name:"weights",type:"storage"},{binding:this.trainingDataBuffers[r].trainingZValues,name:"z_values",type:"storage"}]),e),{}),{default:[{binding:this.backwardPassParamsBuffer,name:"params",type:"storage"},{binding:this.trainingDataBuffers[this.trainingDataBuffers.length-1].trainingActivations,name:"predictions",type:"storage"},{binding:this.targetsBuffer,name:"targets",type:"storage"}]}]}),this.weightGradientComputationShader=new v({code:W,workgroupCount:[Math.ceil(t/64),1],bindingLayouts:[this.layerBuffers.reduce((e,{},a)=>(a>0&&(e[`layer${a}`]=[{binding:this.errorGradientsABuffer,name:"errors",type:"storage"},{binding:this.trainingDataBuffers[a-1].trainingActivations,name:"input_activations",type:"storage"}],e[`layer_alt${a}`]=[{binding:this.errorGradientsBBuffer,name:"errors",type:"storage"},{binding:this.trainingDataBuffers[a-1].trainingActivations,name:"input_activations",type:"storage"}]),e),{}),this.layerBuffers.reduce((e,{weightGradients:a},r)=>(r>0&&(e[`layer${r}`]=[{binding:this.gradientParamsBuffer,name:"params",type:"storage"},{binding:a,name:"weight_gradients",type:"storage"}]),e),{})]}),this.biasGradientComputationShader=new v({code:I,workgroupCount:[t,1],bindingLayouts:[this.layerBuffers.reduce((e,{},a)=>(a>0&&(e[`layer${a}`]=[{binding:this.errorGradientsABuffer,name:"errors",type:"storage"},{binding:this.trainingDataBuffers[a-1].trainingActivations,name:"input_activations",type:"storage"}],e[`layer_alt${a}`]=[{binding:this.errorGradientsBBuffer,name:"errors",type:"storage"},{binding:this.trainingDataBuffers[a-1].trainingActivations,name:"input_activations",type:"storage"}]),e),{}),this.layerBuffers.reduce((e,{biasGradients:a},r)=>(r>0&&(e[`layer${r}`]=[{binding:this.gradientParamsBuffer,name:"params",type:"storage"},{binding:a,name:"bias_gradients",type:"storage"}]),e),{})]}),this.updateParametersShader=new v({code:O,workgroupCount:[Math.ceil(t/64),1],bindingLayouts:[{default:[{binding:this.learningRateBuffer,name:"learning_rate",type:"uniform"}]},this.layerBuffers.reduce((e,{weightGradients:a,biasGradients:r},n)=>(n>0&&(e[`weights_layer${n}`]=[{binding:a,name:"gradients",type:"storage"}],e[`biases_layer${n}`]=[{binding:r,name:"gradients",type:"storage"}]),e),{}),this.layerBuffers.reduce((e,{weights:a,biases:r},n)=>(n>0&&(e[`weights_layer${n}`]=[{binding:a,name:"parameters",type:"storage"}],e[`biases_layer${n}`]=[{binding:r,name:"parameters",type:"storage"}]),e),{})]}),this.isInitialized=!0}async forwardPass(i,t=!1){if(!this.isInitialized)throw new Error("NeuralNetwork not initialized.");let e=t?this.trainingBatchSize:this.testingBatchSize,a=t?this.trainingDataBuffers[0].trainingActivations:this.testActivationsBufferA;if(i.length!==e*this.inputSize)throw new Error(`Expected ${e*this.inputSize} elements, got ${i.length}`);a.write(i);const r=this.layerSizes.length;for(let s=1;s<r;s++)this.forwardPassParamsBuffer.write(new Uint32Array([e,this.layerSizes[s-1],this.layerSizes[s],s===r-1?this.outputActivationType:this.hiddenActivationType])),this.forwardPassShader.dispatch({bindGroups:{1:`layer_${s}`,2:t?`training_layer_${s}`:`test_layer_${(s+1)%2}`,3:t?`training_layer_${s}`:"test_layer"}});let n=null;return t||(n=await[this.testActivationsBufferB,this.testActivationsBufferA][this.layerSizes.length%2].read()),n}async backwardPass(i){if(!this.isInitialized)throw new Error("NeuralNetwork not initialized.");const t=this.layerSizes.length;await this.learningRateBuffer.write(new Float32Array([i]));for(let e=t-1;e>=1;e--){const a=e%2===0?"group0":"group1",r=e%2===0?"layer_alt":"layer";this.backwardPassParamsBuffer.write(new Uint32Array([this.trainingBatchSize,this.layerSizes[e],e<t-1?this.layerSizes[e+1]:0,e===t-1?this.outputActivationType:this.hiddenActivationType,e===t-1?1:0])),this.backwardErrorShader.dispatch({bindGroups:{0:a,1:`layer${e}`}}),this.gradientParamsBuffer.write(new Uint32Array([this.trainingBatchSize,this.layerSizes[e-1],this.layerSizes[e],0])),this.weightGradientComputationShader.dispatch({bindGroups:{0:`${r}${e}`,1:`layer${e}`}}),this.biasGradientComputationShader.dispatch({bindGroups:{0:`${r}${e}`,1:`layer${e}`}}),this.updateParametersShader.dispatch({bindGroups:{1:`weights_layer${e}`,2:`weights_layer${e}`}}),this.updateParametersShader.dispatch({bindGroups:{1:`biases_layer${e}`,2:`biases_layer${e}`}})}}async lossPass(){return await this.lossParamsBuffer.write(new Uint32Array([this.trainingBatchSize,this.outputSize,0,0,T])),this.lossShader.dispatch(),(await this.totalBatchLossBuffer.read())[0]}async train(i){{if(!this.isInitialized)throw new Error("NeuralNetwork not initialized.");if(i?.inputActivations.length!==i?.targetActivations.length)throw new Error("Inputs and targets must have the same number of samples.");for(let a=0;a<i.targetActivations.length;a++)if(i.targetActivations[a].length!==this.outputSize)throw new Error("Target size does not match output layer size.");if(i?.inputActivations[0].length!==this.inputSize)throw new Error("Input size does not match input layer size.")}const t=i.learningRate??.01,e=i.epochs??1;for(let a=0;a<e;a++){let r=i?.inputActivations.length,n=0;{const s=Array.from({length:r},(h,g)=>g);s.sort(()=>Math.random()-.5);const o=s.map(h=>i.inputActivations[h]),d=s.map(h=>i.targetActivations[h]);i.inputActivations=o,i.targetActivations=d}for(let s=0;s<r;s+=this.trainingBatchSize){if(Math.min(s+this.trainingBatchSize,r)-s<this.trainingBatchSize)continue;const h=new Float32Array(this.trainingBatchSize*this.inputSize),g=new Float32Array(this.trainingBatchSize*this.outputSize);for(let _=0;_<this.trainingBatchSize;_++){const B=s+_;h.set(i.inputActivations[B],_*this.inputSize),g.set(i.targetActivations[B],_*this.outputSize)}await this.targetsBuffer.write(g),await this.forwardPass(h,!0),this.totalBatchLossBuffer.write(new Uint32Array([0]));let y=i.progressCallback?await this.lossPass()/T:0;n+=y/(r/this.trainingBatchSize),await this.backwardPass(t)}i.progressCallback?.(a,n)}}async evaluatePopulation(i){if(!this.isInitialized)throw new Error("NeuralNetwork not initialized.");const t=i.populationSize,e=i.batchSize,a=this.layerSizes.length;if(t<1||e<1)throw new Error("populationSize and batchSize must be >= 1");if(!i.weights||!i.biases||i.weights.length!==a||i.biases.length!==a)throw new Error("weights/biases must be provided per layer index (same length as layerSizes). Use empty slot at index 0.");if(!i.inputs||i.inputs.length!==t)throw new Error("inputs must be provided per genome.");for(let l=0;l<t;l++)if(i.inputs[l].length!==e*this.inputSize)throw new Error(`inputs[${l}] length must equal batchSize*inputSize`);if(i.returnLoss){if(!i.targets||i.targets.length!==t)throw new Error("targets must be provided per genome when returnLoss is true.");for(let l=0;l<t;l++)if(i.targets[l].length!==e*this.outputSize)throw new Error(`targets[${l}] length must equal batchSize*outputSize`)}const r=new Array(a),n=new Array(a),s=new Array(a),o=new Array(a),d=new Float32Array(t*e*this.inputSize);for(let l=0;l<t;l++)d.set(i.inputs[l],l*e*this.inputSize);o[0]=new f({dataType:"array<f32>",size:t*e*this.inputSize,canCopyDst:!0,canCopySrc:!1,initialValue:d});let h=Math.max(...this.layerSizes);for(let l=1;l<a;l++){const w=this.layerSizes[l-1],c=this.layerSizes[l],x=new Float32Array(t*c*w),S=new Float32Array(t*c);for(let b=0;b<t;b++){const m=i.weights[l][b],C=i.biases[l][b];if(!m||m.length!==w*c)throw new Error(`weights[layer=${l}][${b}] size mismatch`);if(!C||C.length!==c)throw new Error(`biases[layer=${l}][${b}] size mismatch`);x.set(m,b*c*w),S.set(C,b*c)}r[l]=new f({dataType:"array<f32>",size:t*c*w,canCopyDst:!0,canCopySrc:!1,initialValue:x}),n[l]=new f({dataType:"array<f32>",size:t*c,canCopyDst:!0,canCopySrc:!1,initialValue:S}),s[l]=new f({dataType:"array<f32>",size:t*e*c,canCopyDst:!1,canCopySrc:!1}),o[l]=new f({dataType:"array<f32>",size:t*e*c,canCopyDst:!1,canCopySrc:!0})}const g=new f({dataType:"struct",structName:"GeneticLayerParams",fields:[{name:"population_size",dataType:"u32"},{name:"batch_size",dataType:"u32"},{name:"input_size",dataType:"u32"},{name:"output_size",dataType:"u32"},{name:"activation_type",dataType:"u32"}],canCopyDst:!0}),y=new v({useExecutionCountBuffer:!1,useTimeBuffer:!1,code:$,workgroupCount:[Math.ceil(t*e*h/64),1],bindingLayouts:[{default:[{binding:g,name:"params",type:"storage"}]},r.reduce((l,w,c)=>(c>0&&(l[`layer_${c}`]=[{binding:r[c],name:"weights",type:"storage"},{binding:n[c],name:"biases",type:"storage"}]),l),{}),o.reduce((l,w,c)=>(c>0&&(l[`layer_${c}`]=[{binding:o[c-1],name:"inputs",type:"storage"},{binding:o[c],name:"activations",type:"storage"}]),l),{}),s.reduce((l,w,c)=>(c>0&&(l[`layer_${c}`]=[{binding:s[c],name:"z_values",type:"storage"}]),l),{})]});for(let l=1;l<a;l++)g.write(new Uint32Array([t,e,this.layerSizes[l-1],this.layerSizes[l],l===a-1?this.outputActivationType:this.hiddenActivationType])),y.dispatch({bindGroups:{1:`layer_${l}`,2:`layer_${l}`,3:`layer_${l}`}});let _=null;if(i.returnLoss){const l=new Float32Array(t*e*this.outputSize);for(let m=0;m<t;m++)l.set(i.targets[m],m*e*this.outputSize);const w=new f({dataType:"array<f32>",size:t*e*this.outputSize,canCopyDst:!0,canCopySrc:!1,initialValue:l}),c=new f({dataType:"array<atomic<u32>>",size:t,canCopyDst:!0,canCopySrc:!0,initialValue:new Uint32Array(t).fill(0)}),x=new f({dataType:"struct",structName:"GeneticLossParams",fields:[{name:"population_size",dataType:"u32"},{name:"batch_size",dataType:"u32"},{name:"output_size",dataType:"u32"},{name:"loss_type",dataType:"u32"},{name:"reduction",dataType:"u32"},{name:"loss_multiplier",dataType:"u32"}],canCopyDst:!0,canCopySrc:!0});await x.write(new Uint32Array([t,e,this.outputSize,0,0,T]));const S=new v({code:R,workgroupCount:[Math.ceil(t*e/64),1],bindingLayouts:[{default:[{binding:x,name:"params",type:"storage"},{binding:o[a-1],name:"predictions",type:"storage"},{binding:w,name:"targets",type:"storage"},{binding:c,name:"total_loss",type:"storage"}]}]});await c.write(new Uint32Array(t).fill(0)),S.dispatch();const b=await c.read();_=new Float32Array(t);for(let m=0;m<t;m++)_[m]=b[m]/T/e}let B=null;return i.returnActivations&&(B=await o[a-1].read()),{losses:_,activations:B}}}export{L as A,q as N};
