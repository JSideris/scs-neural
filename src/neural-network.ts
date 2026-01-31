import { ComputeShader, Shader, StorageBuffer, UniformBuffer } from "simple-compute-shaders";

import lossWgsl from "./shaders/loss.compute.wgsl";
// import forwardPassWgsl from "./shaders/forward-pass.compute.wgsl";
import errorPropagationWgsl from "./shaders/error-propagation.compute.wgsl";
import weightGradientComputationWgsl from "./shaders/weight-gradient-computation.compute.wgsl";
import biasGradientComputationWgsl from "./shaders/bias-gradient-computation.compute.wgsl";
import updateParametersWgsl from "./shaders/update-parameters.compute.wgsl";
import forwardPassGeneticWgsl from "./shaders/forward-pass-genetic.compute.wgsl";
import lossGeneticWgsl from "./shaders/loss-genetic.compute.wgsl";
import InitializationMethods from "./initialization-methods";

// Allows loss to be stored in a u32 rather than a f32, which is required by WGSL for atomic operations.
const LOSS_MULTIPLIER = 10000;

export enum ActivationType{
	RELU = 0,
	SIGMOID = 1,
	LINEAR = 2,
	TANH = 3,
	SOFTMAX = 4,
}

export default class NeuralNetwork{
	private forwardPassShader: ComputeShader;

	private forwardPassParamsBuffer: StorageBuffer;
	private lossParamsBuffer: StorageBuffer;
	private backwardPassParamsBuffer: StorageBuffer;
	private gradientParamsBuffer: StorageBuffer;
	private learningRateBuffer: UniformBuffer;

	layerBuffers: {
		// Network:
		weights: StorageBuffer, 
		biases: StorageBuffer,

		// Training:
		errors: StorageBuffer,
		weightGradients: StorageBuffer,
		biasGradients: StorageBuffer,

	}[] = [];

	private trainingDataBuffers: {
		trainingActivations: StorageBuffer,
		trainingZValues: StorageBuffer,
	}[] = [];

	private testActivationsBufferA: StorageBuffer;
	private testActivationsBufferB: StorageBuffer;
	private testZValuesBuffer: StorageBuffer;
	
	private targetsBuffer: StorageBuffer;
	private totalBatchLossBuffer: StorageBuffer;

	private errorGradientsABuffer: StorageBuffer;
	private errorGradientsBBuffer: StorageBuffer;

	private isInitialized: boolean = false;
	layerSizes: number[];
	private trainingBatchSize: number;
	private testingBatchSize: number;

	private lossShader: ComputeShader;
	private backwardErrorShader: ComputeShader;
	private weightGradientComputationShader: ComputeShader;
	private biasGradientComputationShader: ComputeShader;
	private updateParametersShader: ComputeShader;

	private hiddenActivationType: ActivationType;
	private outputActivationType: ActivationType;

	get outputSize(){
		return this.layerSizes?.length ? this.layerSizes[this.layerSizes.length - 1] : 0;
	}

	get inputSize(){
		return this.layerSizes?.length ? this.layerSizes[0] : 0;
	}
	
	constructor(props: {
		layerSizes: number[], 
		trainingBatchSize?: number,
		testingBatchSize?: number,
		hiddenActivationType?: ActivationType,
		outputActivationType?: ActivationType,
	}){

		if(!props.layerSizes || props.layerSizes.length < 2){
			throw new Error("Layer sizes must be an array of at least 2 numbers.");
		}
		if(props.layerSizes.some(size => size < 1)){
			throw new Error("Layer sizes must be greater than 0.");
		}

		this.layerSizes = props.layerSizes;
		this.testingBatchSize = props.testingBatchSize;
		this.trainingBatchSize = props.trainingBatchSize;
		this.hiddenActivationType = props.hiddenActivationType ?? ActivationType.RELU;
		this.outputActivationType = props.outputActivationType ?? ActivationType.RELU;
	}

	async initialize(initializationMethod: "uniform" | "xavier" | "he" | "zero" = "xavier"){
		if (this.isInitialized) {
			console.warn("NeuralNetwork already initialized.");
			return;
		}

		await Shader.initialize();

		let maxLayerSize = Math.max(...this.layerSizes);

		{ // Create all the temp buffers.

			this.testActivationsBufferA = new StorageBuffer({
				dataType: "array<f32>",
				size: this.testingBatchSize * maxLayerSize,
				canCopyDst: true,
				canCopySrc: this.layerSizes.length % 2 == 1 ? true : false,
			});
			this.testActivationsBufferB = new StorageBuffer({
				dataType: "array<f32>",
				size: this.testingBatchSize * maxLayerSize,
				canCopyDst: false,
				canCopySrc: this.layerSizes.length % 2 == 0 ? true : false,
			});
			this.testZValuesBuffer = new StorageBuffer({
				dataType: "array<f32>",
				size: this.testingBatchSize * maxLayerSize,
			});

			for(let layer = 0; layer < this.layerSizes.length; layer++){

				let weightData: Float32Array;
				let biasData: Float32Array;

				if(layer > 0){
					let inputSize = this.layerSizes[layer - 1];
					let outputSize = this.layerSizes[layer];
					
					switch (initializationMethod) {
						case 'xavier':
							weightData = InitializationMethods.initXavier(inputSize, outputSize);
							break;
							
						case 'he':
							weightData = InitializationMethods.initHe(inputSize, outputSize);
							break;
							
						case 'uniform':
							weightData = InitializationMethods.initUniform(inputSize, outputSize, -0.5, 0.5);
							break;
							
						case 'zero':
							weightData = InitializationMethods.initZero(inputSize, outputSize);
							break;
							
						default:
							throw new Error(`Unknown initialization method: ${initializationMethod}`);
					}

					biasData = initializationMethod === 'zero' ? 
						InitializationMethods.initZero(1, outputSize) : 
						InitializationMethods.initUniform(1, outputSize, -0.1, 0.1);
				}
				

				this.layerBuffers.push({
					weights: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.layerSizes[layer - 1] * this.layerSizes[layer],
						initialValue: weightData,
						canCopyDst: true,
						canCopySrc: true,
					}) : null,
					biases: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.layerSizes[layer],
						initialValue: biasData,
						canCopyDst: true,
						canCopySrc: true,
					}) : null,
					errors: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.trainingBatchSize * this.layerSizes[layer],
					}) : null,
					weightGradients: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.layerSizes[layer - 1] * this.layerSizes[layer],
					}) : null,
					biasGradients: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.layerSizes[layer],
					}) : null,
				});

				this.trainingDataBuffers.push({
					trainingActivations: new StorageBuffer({
						dataType: "array<f32>",
						size: this.trainingBatchSize * this.layerSizes[layer],
						canCopyDst: layer == 0 ? true : false,
						canCopySrc: layer == this.layerSizes.length - 1 ? true : false,
					}),
					trainingZValues: layer > 0 ? new StorageBuffer({
						dataType: "array<f32>",
						size: this.trainingBatchSize * this.layerSizes[layer],
					}) : null,
				});
			}
		}

		// Create ping-pong buffers for error propagation
		this.errorGradientsABuffer = new StorageBuffer({
			dataType: "array<f32>",
			size: this.trainingBatchSize * maxLayerSize,
		});

		this.errorGradientsBBuffer = new StorageBuffer({
			dataType: "array<f32>",
			size: this.trainingBatchSize * maxLayerSize,
		});

		this.forwardPassParamsBuffer = new StorageBuffer({
			dataType: "struct",
			structName: "LayerParams",
			fields: [
				{name: "batch_size",dataType: "u32"},
				{name: "input_size",dataType: "u32"},
				{name: "output_size",dataType: "u32"},
				{name: "activation_type",dataType: "u32"}
			],
			canCopyDst: true,
		});

		this.lossParamsBuffer = new StorageBuffer({
			dataType: "struct",
			structName: "LossParams",
			fields: [
				{
					name: "batch_size",
					dataType: "u32"
				},
				{
					name: "output_size",
					dataType: "u32"
				},
				{
					name: "loss_type",
					dataType: "u32"
				},
				{
					name: "reduction",
					dataType: "u32"
				},
				{
					name: "loss_multiplier",
					dataType: "u32"
				}
			],
			canCopyDst: true,
			canCopySrc: true,
			initialValue: [
				this.trainingBatchSize,
				this.outputSize,
				0,
				0
			]
		});

		this.backwardPassParamsBuffer = new StorageBuffer({
			dataType: "struct",
			structName: "BackpropParams",
			fields: [
				{name: "batch_size",dataType: "u32"},
				{name: "current_layer_size",dataType: "u32"},
				{name: "next_layer_size",dataType: "u32"},
				{name: "activation_type",dataType: "u32"},
				{name: "is_output_layer",dataType: "u32"}
			],
			canCopyDst: true,
		});

		this.gradientParamsBuffer = new StorageBuffer({
			dataType: "struct",
			structName: "GradientParams",
			fields: [
				{name: "batch_size",dataType: "u32"},
				{name: "input_size",dataType: "u32"},
				{name: "output_size",dataType: "u32"},
				{name: "accumulate",dataType: "u32"}
			],
			canCopyDst: true,
		});

		this.learningRateBuffer = new UniformBuffer({
			dataType: "f32",
			canCopyDst: true,
			initialValue: new Float32Array([0.01]), // Default learning rate
		});

		this.targetsBuffer = new StorageBuffer({
			dataType: "array<f32>",
			size: this.outputSize * this.trainingBatchSize,
			canCopyDst: true,
			canCopySrc: true,
			initialValue: new Float32Array(this.outputSize * this.trainingBatchSize).fill(0)
		});
		
		this.totalBatchLossBuffer = new StorageBuffer({
			dataType: "array<atomic<u32>>",
			size: 1,
			canCopyDst: true,
			canCopySrc: true,
			initialValue: new Uint32Array(1).fill(0)
		});

		// Layouts:
		// 0: Params
		// 1: Weights and biases
		// 2: Inputs and activations
		// 3: Z values
		this.forwardPassShader = new ComputeShader({
			useExecutionCountBuffer: false,
			useTimeBuffer: false,
			code: forwardPassGeneticWgsl,
			workgroupCount: [Math.ceil(maxLayerSize / 64), 1],
			bindingLayouts: [
				{
					default: [
						{
							binding: this.forwardPassParamsBuffer,
							name: "params",
							type: "storage"
						},
					]
				},
				this.layerBuffers.reduce((obj, {
					weights, 
					biases,
				}, layer) => {
					if(layer > 0){
						obj[`layer_${layer}`] = [
							{
								binding: weights,
								name: "weights",
								type: "storage"
							}, {
								binding: biases,
								name: "biases",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
				this.trainingDataBuffers.reduce(
					(obj, {trainingActivations}, layer) => {
						if(layer > 0){
							obj[`training_layer_${layer}`] = [
								{
									binding: this.trainingDataBuffers[layer - 1].trainingActivations,
									name: "inputs",
									type: "storage"
								},
								{
									binding: trainingActivations,
									name: "activations",
									type: "storage"
								},
							];
						}
						return obj;
					}, 
					// Aggregator contains test ping pong buffers:
					{
						test_layer_0: [
							{
								binding: this.testActivationsBufferA,
								name: "inputs",
								type: "storage"
							},
							{
								binding: this.testActivationsBufferB,
								name: "activations",
								type: "storage"
							},
						],
						test_layer_1: [
							{
								binding: this.testActivationsBufferB,
								name: "inputs",
								type: "storage"
							},
							{
								binding: this.testActivationsBufferA,
								name: "activations",
								type: "storage"
							},
						],
					}
				),
				this.trainingDataBuffers.reduce((obj, {
					trainingZValues,
				}, layer) => {
					if(layer > 0){
						obj[`training_layer_${layer}`] = [
							{
								binding: trainingZValues,
								name: "z_values",
								type: "storage"
							},
						];
					}
					return obj;
				}, {
					test_layer: [
						{
							binding: this.testZValuesBuffer,
							name: "z_values",
							type: "storage"
						}
					]
				}),
			]
		});

		this.lossShader = new ComputeShader({
			code: lossWgsl,
			workgroupCount: [Math.ceil(maxLayerSize / 64), 1],
			bindingLayouts: [
				{
					default: [
						{
							binding: this.lossParamsBuffer,
							name: "params",
							type: "storage"
						},
						{
							binding: this.trainingDataBuffers[this.trainingDataBuffers.length - 1].trainingActivations,
							name: "predictions",
							type: "storage"
						},
						{
							binding: this.targetsBuffer,
							name: "targets",
							type: "storage"
						},
						{
							binding: this.totalBatchLossBuffer,
							name: "total_loss",
							type: "storage"
						},
					]
				},
			]
		});

		this.backwardErrorShader = new ComputeShader({
			code: errorPropagationWgsl,
			workgroupCount: [Math.ceil(maxLayerSize / 64), 1],
			bindingLayouts: [
				{
					group0: [
						{
							binding: this.errorGradientsABuffer,
							name: "next_layer_errors",
							type: "storage"
						},
						{
							binding: this.errorGradientsBBuffer,
							name: "current_layer_errors",
							type: "storage"
						},
					],
					group1: [
						{
							binding: this.errorGradientsBBuffer,
							name: "next_layer_errors",
							type: "storage"
						},
						{
							binding: this.errorGradientsABuffer,
							name: "current_layer_errors",
							type: "storage"
						},
					]
				},
				this.layerBuffers.reduce((obj, {
					weights,
				}, layer) => {
					if(layer > 0){
						obj[`layer${layer}`] = [
							{
								binding: weights,
								name: "weights",
								type: "storage"
							}, 
							{
								binding: this.trainingDataBuffers[layer].trainingZValues,
								name: "z_values",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
				{
					default: [
						{
							binding: this.backwardPassParamsBuffer,
							name: "params",
							type: "storage"
						},
						{
							binding: this.trainingDataBuffers[this.trainingDataBuffers.length - 1].trainingActivations,
							name: "predictions",
							type: "storage"
						},
						{
							binding: this.targetsBuffer,
							name: "targets",
							type: "storage"
						},
					]
				}
			]
		});

		this.weightGradientComputationShader = new ComputeShader({
			code: weightGradientComputationWgsl,
			workgroupCount: [Math.ceil(maxLayerSize / 64), 1],
			bindingLayouts: [
				this.layerBuffers.reduce((obj, {
				}, layer) => {
					if(layer > 0){
						obj[`layer${layer}`] = [
							{
								binding: this.errorGradientsABuffer,
								name: "errors",
								type: "storage"
							},
							{
								binding: this.trainingDataBuffers[layer - 1].trainingActivations,
								name: "input_activations",
								type: "storage"
							},
						];
						obj[`layer_alt${layer}`] = [
							{
								binding: this.errorGradientsBBuffer,
								name: "errors",
								type: "storage"
							},
							{
								binding: this.trainingDataBuffers[layer - 1].trainingActivations,
								name: "input_activations",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
				this.layerBuffers.reduce((obj, {
					weightGradients,
				}, layer) => {
					if(layer > 0){
						obj[`layer${layer}`] = [
							{
								binding: this.gradientParamsBuffer,
								name: "params",
								type: "storage"
							},
							{
								binding: weightGradients,
								name: "weight_gradients",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
			]
		});

		this.biasGradientComputationShader = new ComputeShader({
			code: biasGradientComputationWgsl,
			workgroupCount: [maxLayerSize, 1],
			bindingLayouts: [
				this.layerBuffers.reduce((obj, {}, layer) => {
					if(layer > 0){
						obj[`layer${layer}`] = [
							{
								binding: this.errorGradientsABuffer,
								name: "errors",
								type: "storage"
							},
							{
								binding: this.trainingDataBuffers[layer - 1].trainingActivations,
								name: "input_activations",
								type: "storage"
							},
						];
						obj[`layer_alt${layer}`] = [
							{
								binding: this.errorGradientsBBuffer,
								name: "errors",
								type: "storage"
							},
							{
								binding: this.trainingDataBuffers[layer - 1].trainingActivations,
								name: "input_activations",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
				this.layerBuffers.reduce((obj, {
					biasGradients,
				}, layer) => {
					if(layer > 0){
						obj[`layer${layer}`] = [
							{
								binding: this.gradientParamsBuffer,
								name: "params",
								type: "storage"
							},
							{
								binding: biasGradients,
								name: "bias_gradients",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
			]
		});

		this.updateParametersShader = new ComputeShader({
			code: updateParametersWgsl,
			workgroupCount: [Math.ceil(maxLayerSize / 64), 1],
			bindingLayouts: [
				{
					default: [
						{
							binding: this.learningRateBuffer,
							name: "learning_rate",
							type: "uniform"
						},
					]
				},
				this.layerBuffers.reduce((obj, {
					weightGradients,
					biasGradients,
				}, layer) => {
					if(layer > 0){
						obj[`weights_layer${layer}`] = [
							{
								binding: weightGradients,
								name: "gradients",
								type: "storage"
							},
						];
						obj[`biases_layer${layer}`] = [
							{
								binding: biasGradients,
								name: "gradients",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
				this.layerBuffers.reduce((obj, {
					weights,
					biases,
				}, layer) => {
					if(layer > 0){
						obj[`weights_layer${layer}`] = [
							{
								binding: weights,
								name: "parameters",
								type: "storage"
							},
						];
						obj[`biases_layer${layer}`] = [
							{
								binding: biases,
								name: "parameters",
								type: "storage"
							},
						];
					}
					return obj;
				}, {}),
			]
		});

		this.isInitialized = true;
	}

	async forwardPass(inputActivations: Float32Array, trainMode: boolean = false){
		if (!this.isInitialized) {
			throw new Error("NeuralNetwork not initialized.");
		}

		let batchSize = trainMode ? this.trainingBatchSize : this.testingBatchSize;
		let inputBuffer = trainMode ? this.trainingDataBuffers[0].trainingActivations : this.testActivationsBufferA;

		if (inputActivations.length !== batchSize * this.inputSize) {
			throw new Error(`Expected ${batchSize * this.inputSize} elements, got ${inputActivations.length}`);
		}

		inputBuffer.write(inputActivations);
		const numLayers = this.layerSizes.length;

		for(let layer = 1; layer < numLayers; layer++){

			this.forwardPassParamsBuffer.write(new Uint32Array([
				batchSize,
				this.layerSizes[layer - 1],
				this.layerSizes[layer],
				layer === numLayers - 1 ? this.outputActivationType : this.hiddenActivationType
			]));

			this.forwardPassShader.dispatch({
				bindGroups: {
					1: `layer_${layer}`,
					2: trainMode ? `training_layer_${layer}` : `test_layer_${(layer + 1) % 2}`,
					3: trainMode ? `training_layer_${layer}` : `test_layer`,
				},
			});
		}

		let finalActivations: Float32Array = null;

		if(!trainMode){
			let lastLayerBuffer = [this.testActivationsBufferB, this.testActivationsBufferA][this.layerSizes.length % 2];
			finalActivations = await lastLayerBuffer.read() as Float32Array;
		}

		return finalActivations;
	}

	private async backwardPass(learningRate: number){
		if (!this.isInitialized) {
			throw new Error("NeuralNetwork not initialized.");
		}

		const numLayers = this.layerSizes.length;

		// Update learning rate buffer
		await this.learningRateBuffer.write(new Float32Array([learningRate]));

		for(let layer = numLayers - 1; layer >= 1; layer--){
			// Determine ping-pong buffer group
			const errorBufferGroup = layer % 2 === 0 ? 'group0' : 'group1';
			const errorBufferName = layer % 2 === 0 ? 'layer_alt' : 'layer';

			// Update backward pass params
			this.backwardPassParamsBuffer.write(new Uint32Array([
				this.trainingBatchSize,
				this.layerSizes[layer],
				layer < numLayers - 1 ? this.layerSizes[layer + 1] : 0,
				layer === numLayers - 1 ? this.outputActivationType : this.hiddenActivationType, // activation_type
				layer === numLayers - 1 ? 1 : 0, // is_output_layer
			]));

			// Propagate errors backward
			this.backwardErrorShader.dispatch({
				bindGroups: {
					0: errorBufferGroup,
					1: `layer${layer}`,
				},
			});

			// Update gradient computation params
			this.gradientParamsBuffer.write(new Uint32Array([
				this.trainingBatchSize,
				this.layerSizes[layer - 1], // input_size
				this.layerSizes[layer],     // output_size
				0                            // accumulate (0 = overwrite)
			]));

			// Compute weight gradients
			this.weightGradientComputationShader.dispatch({
				bindGroups: {
					0: `${errorBufferName}${layer}`,
					1: `layer${layer}`,
				},
			});

			// Compute bias gradients
			this.biasGradientComputationShader.dispatch({
				bindGroups: {
					0: `${errorBufferName}${layer}`,
					1: `layer${layer}`,
				},
			});

			// Update weights
			this.updateParametersShader.dispatch({
				bindGroups: {
					1: `weights_layer${layer}`,
					2: `weights_layer${layer}`,
				},
			});

			// Update biases
			this.updateParametersShader.dispatch({
				bindGroups: {
					1: `biases_layer${layer}`,
					2: `biases_layer${layer}`,
				},
			});
		}
	}

	private async lossPass(){

		await this.lossParamsBuffer.write(new Uint32Array([
			this.trainingBatchSize,
			this.outputSize,
			0,
			0,
			LOSS_MULTIPLIER
		]));

		this.lossShader.dispatch();

		const lossData = await this.totalBatchLossBuffer.read();
		return lossData[0];
	}

	async train(props: {
		inputActivations: Float32Array[], 
		targetActivations: Float32Array[], 
		learningRate?: number, 
		momentum?: number, 
		weightDecay?: number,
		epochs?: number,
		progressCallback?: (epoch: number, loss: number) => void
	}){
		{ // Validation
			if (!this.isInitialized) {
				throw new Error("NeuralNetwork not initialized.");
			}
			if (props?.inputActivations.length !== props?.targetActivations.length) {
				throw new Error("Inputs and targets must have the same number of samples.");
			}
			for(let i = 0; i < props.targetActivations.length; i++){
				if (props.targetActivations[i].length !== this.outputSize) {
					throw new Error("Target size does not match output layer size.");
				}
			}
			if (props?.inputActivations[0].length !== this.inputSize) {
				throw new Error("Input size does not match input layer size.");
			}
		}

		const learningRate = props.learningRate ?? 0.01;
		const epochs = props.epochs ?? 1;
		
		for(let epoch = 0; epoch < epochs; epoch++){
			
			let numSamples = props?.inputActivations.length;
			let epochLoss = 0;

			{ // Shuffle inputActivations and targetActivations, but keep them in sync.
				const shuffledIndices = Array.from({length: numSamples}, (_, i) => i);
				shuffledIndices.sort(() => Math.random() - 0.5);
				const shuffledInputActivations = shuffledIndices.map(i => props.inputActivations[i]);
				const shuffledTargetActivations = shuffledIndices.map(i => props.targetActivations[i]);
				props.inputActivations = shuffledInputActivations;
				props.targetActivations = shuffledTargetActivations;
			}

			// Process in batches
			for (let batchStart = 0; batchStart < numSamples; batchStart += this.trainingBatchSize) {
				const batchEnd = Math.min(batchStart + this.trainingBatchSize, numSamples);
				const actualBatchSize = batchEnd - batchStart;
				
				// Skip incomplete batches (or handle them separately)
				if (actualBatchSize < this.trainingBatchSize) continue;
				
				// Concatenate batch samples into single array
				const batchInputs = new Float32Array(this.trainingBatchSize * this.inputSize);
				const batchTargets = new Float32Array(this.trainingBatchSize * this.outputSize);
				
				for (let i = 0; i < this.trainingBatchSize; i++) {
					const sampleIdx = batchStart + i;
					batchInputs.set(props.inputActivations[sampleIdx], i * this.inputSize);
					batchTargets.set(props.targetActivations[sampleIdx], i * this.outputSize);
				}

				await this.targetsBuffer.write(batchTargets);
				
				// Forward pass on entire batch
				await this.forwardPass(batchInputs, true);
				
				// Compute loss for batch
				this.totalBatchLossBuffer.write(new Uint32Array([0]));
				let batchLoss = props.progressCallback ? await this.lossPass() / LOSS_MULTIPLIER : 0;
				epochLoss += batchLoss / (numSamples / this.trainingBatchSize);
				
				// Backward pass updates weights once for entire batch
				await this.backwardPass(learningRate);
			}
			
			// console.log(`Epoch ${epoch + 1} loss: ${epochLoss}`);
			props.progressCallback?.(epoch, epochLoss);
		}
	}

	// === Genetic Evaluation (Forward + Loss per genome) ===
	// Evaluates a population of genomes in parallel without affecting training state
	async evaluatePopulation(props: {
		populationSize: number,
		batchSize: number,
		weights: Float32Array[][], // [layerIndex>0][genomeIndex] length input_size*output_size
		biases: Float32Array[][],  // [layerIndex>0][genomeIndex] length output_size
		inputs: Float32Array[],    // [genomeIndex] length batchSize*inputSize
		targets?: Float32Array[],   // [genomeIndex] length batchSize*outputSize (required for loss)
		returnActivations?: boolean,
		returnLoss?: boolean,
	}){
		if (!this.isInitialized) {
			throw new Error("NeuralNetwork not initialized.");
		}

		const P = props.populationSize;
		const B = props.batchSize;
		const numLayers = this.layerSizes.length;

		if (P < 1 || B < 1) {
			throw new Error("populationSize and batchSize must be >= 1");
		}
		if (!props.weights || !props.biases || props.weights.length !== numLayers || props.biases.length !== numLayers) {
			throw new Error("weights/biases must be provided per layer index (same length as layerSizes). Use empty slot at index 0.");
		}
		if (!props.inputs || props.inputs.length !== P) {
			throw new Error("inputs must be provided per genome.");
		}
		for (let g = 0; g < P; g++) {
			if (props.inputs[g].length !== B * this.inputSize) {
				throw new Error(`inputs[${g}] length must equal batchSize*inputSize`);
			}
		}
		if (props.returnLoss) {
			if (!props.targets || props.targets.length !== P) {
				throw new Error("targets must be provided per genome when returnLoss is true.");
			}
			for (let g = 0; g < P; g++) {
				if (props.targets[g].length !== B * this.outputSize) {
					throw new Error(`targets[${g}] length must equal batchSize*outputSize`);
				}
			}
		}

		// Pack per-layer weights/biases across genomes
		const genWeights: StorageBuffer[] = new Array(numLayers);
		const genBiases: StorageBuffer[] = new Array(numLayers);
		const genZValues: StorageBuffer[] = new Array(numLayers);
		const genActivations: StorageBuffer[] = new Array(numLayers);

		// Inputs buffer is activations of layer 0
		const packedInputs = new Float32Array(P * B * this.inputSize);
		for (let g = 0; g < P; g++) {
			packedInputs.set(props.inputs[g], g * B * this.inputSize);
		}
		genActivations[0] = new StorageBuffer({
			dataType: "array<f32>",
			size: P * B * this.inputSize,
			canCopyDst: true,
			canCopySrc: false,
			initialValue: packedInputs,
		});

		let maxLayerSize = Math.max(...this.layerSizes);

		for (let layer = 1; layer < numLayers; layer++) {
			const inputSize = this.layerSizes[layer - 1];
			const outputSize = this.layerSizes[layer];

			// Pack weights: [genome, out, in]
			const packedWeights = new Float32Array(P * outputSize * inputSize);
			const packedBiases = new Float32Array(P * outputSize);
			for (let g = 0; g < P; g++) {
				const w = props.weights[layer][g];
				const b = props.biases[layer][g];
				if (!w || w.length !== inputSize * outputSize) {
					throw new Error(`weights[layer=${layer}][${g}] size mismatch`);
				}
				if (!b || b.length !== outputSize) {
					throw new Error(`biases[layer=${layer}][${g}] size mismatch`);
				}
				packedWeights.set(w, g * outputSize * inputSize);
				packedBiases.set(b, g * outputSize);
			}

			genWeights[layer] = new StorageBuffer({
				dataType: "array<f32>",
				size: P * outputSize * inputSize,
				canCopyDst: true,
				canCopySrc: false,
				initialValue: packedWeights,
			});
			genBiases[layer] = new StorageBuffer({
				dataType: "array<f32>",
				size: P * outputSize,
				canCopyDst: true,
				canCopySrc: false,
				initialValue: packedBiases,
			});

			genZValues[layer] = new StorageBuffer({
				dataType: "array<f32>",
				size: P * B * outputSize,
				canCopyDst: false,
				canCopySrc: false,
			});
			genActivations[layer] = new StorageBuffer({
				dataType: "array<f32>",
				size: P * B * outputSize,
				canCopyDst: false,
				canCopySrc: true,
			});
		}

		// Forward genetic shader and params buffer
		const forwardGeneticParamsBuffer = new StorageBuffer({
			dataType: "struct",
			structName: "GeneticLayerParams",
			fields: [
				{name: "population_size", dataType: "u32"},
				{name: "batch_size", dataType: "u32"},
				{name: "input_size", dataType: "u32"},
				{name: "output_size", dataType: "u32"},
				{name: "activation_type", dataType: "u32"},
			],
			canCopyDst: true,
		});

		const forwardGeneticShader = new ComputeShader({
			useExecutionCountBuffer: false,
			useTimeBuffer: false,
			code: forwardPassGeneticWgsl,
			workgroupCount: [Math.ceil((P * B * maxLayerSize) / 64), 1],
			bindingLayouts: [
				{ default: [ { binding: forwardGeneticParamsBuffer, name: "params", type: "storage" } ] },
				// Group 1: weights/biases per layer
				genWeights.reduce((obj, wb, layer) => {
					if (layer > 0) {
						obj[`layer_${layer}`] = [
							{ binding: genWeights[layer], name: "weights", type: "storage" },
							{ binding: genBiases[layer], name: "biases", type: "storage" },
						];
					}
					return obj;
				}, {} as Record<string, any>),
				// Group 2: inputs/activations per layer
				genActivations.reduce((obj, _, layer) => {
					if (layer > 0) {
						obj[`layer_${layer}`] = [
							{ binding: genActivations[layer - 1], name: "inputs", type: "storage" },
							{ binding: genActivations[layer], name: "activations", type: "storage" },
						];
					}
					return obj;
				}, {} as Record<string, any>),
				// Group 3: z_values per layer
				genZValues.reduce((obj, _, layer) => {
					if (layer > 0) {
						obj[`layer_${layer}`] = [
							{ binding: genZValues[layer], name: "z_values", type: "storage" },
						];
					}
					return obj;
				}, {} as Record<string, any>),
			]
		});

		// Run forward pass across layers
		for (let layer = 1; layer < numLayers; layer++) {
			forwardGeneticParamsBuffer.write(new Uint32Array([
				P,
				B,
				this.layerSizes[layer - 1],
				this.layerSizes[layer],
				layer === numLayers - 1 ? this.outputActivationType : this.hiddenActivationType,
			]));
			forwardGeneticShader.dispatch({
				bindGroups: {
					1: `layer_${layer}`,
					2: `layer_${layer}`,
					3: `layer_${layer}`,
				},
			});
		}

		// Optionally compute loss per genome
		let losses: Float32Array = null;
		if (props.returnLoss) {
			// Pack targets
			const packedTargets = new Float32Array(P * B * this.outputSize);
			for (let g = 0; g < P; g++) {
				packedTargets.set(props.targets[g], g * B * this.outputSize);
			}
			const targetsBuffer = new StorageBuffer({
				dataType: "array<f32>",
				size: P * B * this.outputSize,
				canCopyDst: true,
				canCopySrc: false,
				initialValue: packedTargets,
			});
			const totalLossBuffer = new StorageBuffer({
				dataType: "array<atomic<u32>>",
				size: P,
				canCopyDst: true,
				canCopySrc: true,
				initialValue: new Uint32Array(P).fill(0),
			});
			const geneticLossParamsBuffer = new StorageBuffer({
				dataType: "struct",
				structName: "GeneticLossParams",
				fields: [
					{name: "population_size", dataType: "u32"},
					{name: "batch_size", dataType: "u32"},
					{name: "output_size", dataType: "u32"},
					{name: "loss_type", dataType: "u32"},
					{name: "reduction", dataType: "u32"},
					{name: "loss_multiplier", dataType: "u32"},
				],
				canCopyDst: true,
				canCopySrc: true,
			});

			await geneticLossParamsBuffer.write(new Uint32Array([
				P,
				B,
				this.outputSize,
				0, // MSE
				0, // mean (host will divide by batch)
				LOSS_MULTIPLIER,
			]));

			const lossGeneticShader = new ComputeShader({
				code: lossGeneticWgsl,
				workgroupCount: [Math.ceil((P * B) / 64), 1],
				bindingLayouts: [
					{ default: [
						{ binding: geneticLossParamsBuffer, name: "params", type: "storage" },
						{ binding: genActivations[numLayers - 1], name: "predictions", type: "storage" },
						{ binding: targetsBuffer, name: "targets", type: "storage" },
						{ binding: totalLossBuffer, name: "total_loss", type: "storage" },
					]} 
				]
			});

			// zero totals and dispatch
			await totalLossBuffer.write(new Uint32Array(P).fill(0));
			lossGeneticShader.dispatch();
			const totals = await totalLossBuffer.read() as Uint32Array;
			losses = new Float32Array(P);
			for (let g = 0; g < P; g++) {
				losses[g] = totals[g] / LOSS_MULTIPLIER / B;
			}
		}

		let activations: Float32Array = null;
		if (props.returnActivations) {
			activations = await genActivations[numLayers - 1].read() as Float32Array;
		}

		return { losses, activations };
	}
}