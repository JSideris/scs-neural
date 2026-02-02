import { ComputeShader, Shader, StorageBuffer, UniformBuffer } from "simple-compute-shaders";

import lossWgsl from "./shaders/loss.compute.wgsl?raw";
import forwardPassWgsl from "./shaders/forward-pass.compute.wgsl?raw";
import errorPropagationWgsl from "./shaders/error-propagation.compute.wgsl?raw";
import weightGradientComputationWgsl from "./shaders/weight-gradient-computation.compute.wgsl?raw";
import biasGradientComputationWgsl from "./shaders/bias-gradient-computation.compute.wgsl?raw";
import updateParametersWgsl from "./shaders/update-parameters.compute.wgsl?raw";
import forwardPassGeneticWgsl from "./shaders/forward-pass-genetic.compute.wgsl?raw";
import lossGeneticWgsl from "./shaders/loss-genetic.compute.wgsl?raw";
import conv2dForwardWgsl from "./shaders/conv2d-forward.compute.wgsl?raw";
import conv2dBackwardDataWgsl from "./shaders/conv2d-backward-data.compute.wgsl?raw";
import conv2dBackwardWeightsWgsl from "./shaders/conv2d-backward-weights.compute.wgsl?raw";
import maxpool2dForwardWgsl from "./shaders/maxpool2d-forward.compute.wgsl?raw";
import maxpool2dBackwardWgsl from "./shaders/maxpool2d-backward.compute.wgsl?raw";
import softmaxWgsl from "./shaders/softmax.compute.wgsl?raw";
import copyWgsl from "./shaders/copy.compute.wgsl?raw";
import prepareDeltaWgsl from "./shaders/prepare-delta.compute.wgsl?raw";

import InitializationMethods from "./initialization-methods";
import {
  ActivationType,
  NeuralNetworkOptions,
  TrainOptions,
  EvaluatePopulationOptions,
  LayerConfig,
  LayerType,
} from "./types";

// Allows loss to be stored in a u32 rather than a f32, which is required by WGSL for atomic operations.
const LOSS_MULTIPLIER = 10000;

interface LayerMetadata {
  type: LayerType;
  shape: number[];
  size: number;
  config: LayerConfig;
}

export default class NeuralNetwork {
  private forwardPassShader!: ComputeShader;
  private conv2dForwardShader!: ComputeShader;
  private maxpool2dForwardShader!: ComputeShader;
  private forwardGeneticShader!: ComputeShader;

  private lossShader!: ComputeShader;
  private backwardErrorShader!: ComputeShader;
  private conv2dBackwardDataShader!: ComputeShader;
  private maxpool2dBackwardShader!: ComputeShader;

  private weightGradientComputationShader!: ComputeShader;
  private biasGradientComputationShader!: ComputeShader;
  private conv2dBackwardWeightsShader!: ComputeShader;
  private updateParametersShader!: ComputeShader;
  private softmaxShader!: ComputeShader;
  private copyShader!: ComputeShader;
  private prepareDeltaShader!: ComputeShader;

  private forwardPassParamsBuffer!: StorageBuffer;
  private forwardGeneticParamsBuffer!: StorageBuffer;
  private lossParamsBuffer!: StorageBuffer;
  private backwardPassParamsBuffer!: StorageBuffer;
  private prepareDeltaParamsBuffer!: StorageBuffer;
  private gradientParamsBuffer!: StorageBuffer;
  private learningRateBuffer!: UniformBuffer;

  private conv2dParamsBuffer!: StorageBuffer;
  private maxpool2dParamsBuffer!: StorageBuffer;
  private softmaxParamsBuffer!: StorageBuffer;
  private copyParamsBuffer!: StorageBuffer;

  private geneticWeightsBuffers: StorageBuffer[] = [];
  private geneticBiasesBuffers: StorageBuffer[] = [];
  private geneticActivationsBuffers: StorageBuffer[] = [];
  private currentGeneticPopulationSize: number = 0;
  private currentGeneticBatchSize: number = 0;

  layerBuffers: {
    // Network:
    weights: StorageBuffer | null;
    biases: StorageBuffer | null;

    // Training:
    errors: StorageBuffer;
    weightGradients: StorageBuffer | null;
    biasGradients: StorageBuffer | null;
    maxIndices: StorageBuffer | null; // For MaxPool
  }[] = [];

  private trainingDataBuffers: {
    trainingActivations: StorageBuffer;
    trainingZValues: StorageBuffer | null;
  }[] = [];

  private testActivationsBufferA!: StorageBuffer;
  private testActivationsBufferB!: StorageBuffer;
  private testZValuesBuffer!: StorageBuffer;

  private targetsBuffer!: StorageBuffer;
  private totalBatchLossBuffer!: StorageBuffer;

  private errorGradientsABuffer!: StorageBuffer;
  private errorGradientsBBuffer!: StorageBuffer;

  private isInitialized: boolean = false;
  layers: LayerMetadata[] = [];
  private trainingBatchSize: number;
  private testingBatchSize: number;

  private hiddenActivationType: ActivationType;
  private outputActivationType: ActivationType;

  get outputSize() {
    return this.layers.length ? this.layers[this.layers.length - 1].size : 0;
  }

  get inputSize() {
    return this.layers.length ? this.layers[0].size : 0;
  }

  constructor(props: NeuralNetworkOptions) {
    this.testingBatchSize = props.testingBatchSize ?? 1;
    this.trainingBatchSize = props.trainingBatchSize ?? 1;
    this.hiddenActivationType = props.hiddenActivationType ?? ActivationType.RELU;
    this.outputActivationType = props.outputActivationType ?? ActivationType.RELU;

    if (props.layers) {
      this.parseLayers(props.layers);
    } else if (props.layerSizes) {
      // Backward compatibility
      const layers: LayerConfig[] = props.layerSizes.map((size, i) => {
        if (i === 0) return { type: LayerType.INPUT, shape: [size] };
        return { type: LayerType.DENSE, size };
      });
      this.parseLayers(layers);
    } else {
      throw new Error("Either layers or layerSizes must be provided.");
    }
  }

  private parseLayers(configs: LayerConfig[]) {
    if (configs.length < 2) {
      throw new Error("Network must have at least an input and one more layer.");
    }
    if (configs[0].type !== LayerType.INPUT) {
      throw new Error("First layer must be an input layer.");
    }

    let currentShape = configs[0].shape;
    this.layers.push({
      type: LayerType.INPUT,
      shape: currentShape,
      size: currentShape.reduce((a, b) => a * b, 1),
      config: configs[0],
    });

    for (let i = 1; i < configs.length; i++) {
      const config = configs[i];
      let nextShape: number[] = [];

      switch (config.type) {
        case LayerType.DENSE:
          nextShape = [config.size];
          break;
        case LayerType.CONV2D: {
          const [h, w, c] = currentShape.length === 3 ? currentShape : [1, currentShape[0], 1];
          const stride = config.stride ?? 1;
          const padding = config.padding ?? 0;
          const outH = Math.floor((h + 2 * padding - config.kernelSize) / stride) + 1;
          const outW = Math.floor((w + 2 * padding - config.kernelSize) / stride) + 1;

          if (outH < 1 || outW < 1) {
            throw new Error(
              `Conv2D layer output dimensions [${outH}, ${outW}] are invalid for layer ${i}. ` +
              `Ensure input dimensions [${h}, ${w}] are large enough for kernelSize ${config.kernelSize} with padding ${padding} and stride ${stride}.`
            );
          }

          nextShape = [outH, outW, config.filters];
          break;
        }
        case LayerType.MAXPOOL2D: {
          const [h, w, c] = currentShape;
          const stride = config.stride ?? config.poolSize;
          const outH = Math.floor((h - config.poolSize) / stride) + 1;
          const outW = Math.floor((w - config.poolSize) / stride) + 1;

          if (outH < 1 || outW < 1) {
            throw new Error(
              `MaxPool2D layer output dimensions [${outH}, ${outW}] are invalid for layer ${i}. ` +
              `Ensure input dimensions [${h}, ${w}] are large enough for poolSize ${config.poolSize} and stride ${stride}.`
            );
          }

          nextShape = [outH, outW, c];
          break;
        }
        case LayerType.FLATTEN:
          nextShape = [currentShape.reduce((a, b) => a * b, 1)];
          break;
        default:
          throw new Error(`Unknown layer type: ${(config as any).type}`);
      }

      this.layers.push({
        type: config.type,
        shape: nextShape,
        size: nextShape.reduce((a, b) => a * b, 1),
        config,
      });
      currentShape = nextShape;
    }
  }

  async initialize(initializationMethod: "uniform" | "xavier" | "he" | "zero" = "xavier") {
    if (this.isInitialized) {
      console.warn("NeuralNetwork already initialized.");
      return;
    }

    await Shader.initialize();

    const maxLayerSize = Math.max(...this.layers.map((l) => l.size));

    // Create ping-pong buffers for testing
    this.testActivationsBufferA = new StorageBuffer({
      dataType: "array<f32>",
      size: this.testingBatchSize * maxLayerSize,
      canCopyDst: true,
      canCopySrc: true,
    });
    this.testActivationsBufferB = new StorageBuffer({
      dataType: "array<f32>",
      size: this.testingBatchSize * maxLayerSize,
      canCopyDst: true,
      canCopySrc: true,
    });
    this.testZValuesBuffer = new StorageBuffer({
      dataType: "array<f32>",
      size: this.testingBatchSize * maxLayerSize,
    });

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      let weightBuffer: StorageBuffer | null = null;
      let biasBuffer: StorageBuffer | null = null;
      let weightGradBuffer: StorageBuffer | null = null;
      let biasGradBuffer: StorageBuffer | null = null;
      let maxIndicesBuffer: StorageBuffer | null = null;

      if (i > 0) {
        const prevLayer = this.layers[i - 1];
        if (layer.type === LayerType.DENSE) {
          const fanIn = prevLayer.size;
          const fanOut = layer.size;
          const totalWeights = fanIn * fanOut;
          const weightData = this.getInitData(initializationMethod, fanIn, fanOut, totalWeights);
          const biasData = this.getInitData(initializationMethod === "zero" ? "zero" : "uniform", 1, fanOut, fanOut);

          weightBuffer = new StorageBuffer({
            dataType: "array<f32>",
            size: totalWeights,
            initialValue: weightData,
            canCopyDst: true,
            canCopySrc: true,
          });
          biasBuffer = new StorageBuffer({
            dataType: "array<f32>",
            size: fanOut,
            initialValue: biasData,
            canCopyDst: true,
            canCopySrc: true,
          });
          weightGradBuffer = new StorageBuffer({ dataType: "array<f32>", size: totalWeights });
          biasGradBuffer = new StorageBuffer({ dataType: "array<f32>", size: fanOut });
        } else if (layer.type === LayerType.CONV2D) {
          const config = layer.config as any;
          const inChannels = prevLayer.shape[2];
          const fanIn = config.kernelSize * config.kernelSize * inChannels;
          const fanOut = config.filters;
          const totalWeights = fanIn * fanOut;
          const weightData = this.getInitData(initializationMethod, fanIn, fanOut, totalWeights);
          const biasData = this.getInitData(initializationMethod === "zero" ? "zero" : "uniform", 1, fanOut, fanOut);

          weightBuffer = new StorageBuffer({
            dataType: "array<f32>",
            size: totalWeights,
            initialValue: weightData,
            canCopyDst: true,
            canCopySrc: true,
          });
          biasBuffer = new StorageBuffer({
            dataType: "array<f32>",
            size: fanOut,
            initialValue: biasData,
            canCopyDst: true,
            canCopySrc: true,
          });
          weightGradBuffer = new StorageBuffer({ dataType: "array<f32>", size: totalWeights });
          biasGradBuffer = new StorageBuffer({ dataType: "array<f32>", size: fanOut });
        } else if (layer.type === LayerType.MAXPOOL2D) {
          maxIndicesBuffer = new StorageBuffer({
            dataType: "array<u32>",
            size: this.trainingBatchSize * layer.size,
          });
        }
      }

      this.layerBuffers.push({
        weights: weightBuffer,
        biases: biasBuffer,
        errors: new StorageBuffer({
          dataType: "array<f32>",
          size: this.trainingBatchSize * layer.size,
        }),
        weightGradients: weightGradBuffer,
        biasGradients: biasGradBuffer,
        maxIndices: maxIndicesBuffer,
      });

      this.trainingDataBuffers.push({
        trainingActivations: new StorageBuffer({
          dataType: "array<f32>",
          size: this.trainingBatchSize * layer.size,
          canCopyDst: i === 0,
          canCopySrc: i === this.layers.length - 1,
        }),
        trainingZValues: i > 0 ? new StorageBuffer({
          dataType: "array<f32>",
          size: this.trainingBatchSize * layer.size,
        }) : null,
      });
    }

    // Ping-pong for backprop errors
    this.errorGradientsABuffer = new StorageBuffer({
      dataType: "array<f32>",
      size: this.trainingBatchSize * maxLayerSize,
    });
    this.errorGradientsBBuffer = new StorageBuffer({
      dataType: "array<f32>",
      size: this.trainingBatchSize * maxLayerSize,
    });

    this.initParamsBuffers();
    this.initShaders(maxLayerSize);

    this.isInitialized = true;
  }

  private getInitData(method: string, fanIn: number, fanOut: number, totalSize: number) {
    switch (method) {
      case "xavier": return InitializationMethods.initXavier(fanIn, fanOut, totalSize);
      case "he": return InitializationMethods.initHe(fanIn, fanOut, totalSize);
      case "uniform": return InitializationMethods.initUniform(totalSize, -0.1, 0.1);
      case "zero": return InitializationMethods.initZero(totalSize);
      default: throw new Error(`Unknown initialization method: ${method}`);
    }
  }

  private initParamsBuffers() {
    this.forwardPassParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "LayerParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "input_size", dataType: "u32" },
        { name: "output_size", dataType: "u32" },
        { name: "activation_type", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.forwardGeneticParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "GeneticLayerParams",
      fields: [
        { name: "population_size", dataType: "u32" },
        { name: "batch_size", dataType: "u32" },
        { name: "input_size", dataType: "u32" },
        { name: "output_size", dataType: "u32" },
        { name: "activation_type", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.conv2dParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "Conv2DParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "input_height", dataType: "u32" },
        { name: "input_width", dataType: "u32" },
        { name: "input_channels", dataType: "u32" },
        { name: "output_height", dataType: "u32" },
        { name: "output_width", dataType: "u32" },
        { name: "output_channels", dataType: "u32" },
        { name: "kernel_size", dataType: "u32" },
        { name: "stride", dataType: "u32" },
        { name: "padding", dataType: "u32" },
        { name: "activation_type", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.maxpool2dParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "PoolParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "input_height", dataType: "u32" },
        { name: "input_width", dataType: "u32" },
        { name: "channels", dataType: "u32" },
        { name: "output_height", dataType: "u32" },
        { name: "output_width", dataType: "u32" },
        { name: "pool_size", dataType: "u32" },
        { name: "stride", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.softmaxParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "SoftmaxParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "output_size", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.copyParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "CopyParams",
      fields: [{ name: "size", dataType: "u32" }],
      canCopyDst: true,
    });

    this.lossParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "LossParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "output_size", dataType: "u32" },
        { name: "loss_type", dataType: "u32" },
        { name: "reduction", dataType: "u32" },
        { name: "loss_multiplier", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.backwardPassParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "BackpropParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "current_layer_size", dataType: "u32" },
        { name: "next_layer_size", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.prepareDeltaParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "PrepareDeltaParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "size", dataType: "u32" },
        { name: "activation_type", dataType: "u32" },
        { name: "is_output_layer", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.gradientParamsBuffer = new StorageBuffer({
      dataType: "struct",
      structName: "GradientParams",
      fields: [
        { name: "batch_size", dataType: "u32" },
        { name: "input_size", dataType: "u32" },
        { name: "output_size", dataType: "u32" },
        { name: "accumulate", dataType: "u32" },
      ],
      canCopyDst: true,
    });

    this.learningRateBuffer = new UniformBuffer({
      dataType: "f32",
      canCopyDst: true,
      initialValue: new Float32Array([0.01]),
    });

    this.targetsBuffer = new StorageBuffer({
      dataType: "array<f32>",
      size: this.outputSize * this.trainingBatchSize,
      canCopyDst: true,
    });

    this.totalBatchLossBuffer = new StorageBuffer({
      dataType: "array<atomic<u32>>",
      size: 1,
      canCopySrc: true,
      canCopyDst: true,
    });
  }

  private initShaders(maxLayerSize: number) {
    const hasConv2d = this.layers.some((l) => l.type === LayerType.CONV2D);
    const hasMaxPool2d = this.layers.some((l) => l.type === LayerType.MAXPOOL2D);

    const commonShaderProps = {
      useExecutionCountBuffer: false,
      useTimeBuffer: false,
    };

    this.forwardPassShader = new ComputeShader({
      ...commonShaderProps,
      code: forwardPassWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        { default: [{ binding: this.forwardPassParamsBuffer, name: "params", type: "storage" }] },
        this.getLayerWeightsLayout(),
        this.getLayerDataLayout(),
        this.getLayerZLayout(),
      ],
    });

    if (hasConv2d) {
      this.conv2dForwardShader = new ComputeShader({
        ...commonShaderProps,
        code: conv2dForwardWgsl,
        workgroupCount: [1, 1, 1],
        bindingLayouts: [
          { default: [{ binding: this.conv2dParamsBuffer, name: "params", type: "storage" }] },
          this.getLayerWeightsLayout(),
          this.getLayerDataLayout(),
          this.getLayerZLayout(),
        ],
      });
    }

    if (hasMaxPool2d) {
      this.maxpool2dForwardShader = new ComputeShader({
        ...commonShaderProps,
        code: maxpool2dForwardWgsl,
        workgroupCount: [1, 1, 1],
        bindingLayouts: [
          { default: [{ binding: this.maxpool2dParamsBuffer, name: "params", type: "storage" }] },
          this.getMaxPoolDataLayout(),
        ],
      });
    }

    this.lossShader = new ComputeShader({
      ...commonShaderProps,
      code: lossWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        {
          default: [
            { binding: this.lossParamsBuffer, name: "params", type: "storage" },
            {
              binding: this.trainingDataBuffers[this.layers.length - 1].trainingActivations,
              name: "predictions",
              type: "storage",
            },
            { binding: this.targetsBuffer, name: "targets", type: "storage" },
            { binding: this.totalBatchLossBuffer, name: "total_loss", type: "storage" },
          ],
        },
      ],
    });

    this.backwardErrorShader = new ComputeShader({
      ...commonShaderProps,
      code: errorPropagationWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        this.getPingPongLayout(),
        this.getBackpropErrorDataLayout(),
        {
          default: [{ binding: this.backwardPassParamsBuffer, name: "params", type: "storage" }],
        },
      ],
    });

    if (hasConv2d) {
      this.conv2dBackwardDataShader = new ComputeShader({
        ...commonShaderProps,
        code: conv2dBackwardDataWgsl,
        workgroupCount: [1, 1, 1],
        bindingLayouts: [
          { default: [{ binding: this.conv2dParamsBuffer, name: "params", type: "storage" }] },
          this.getPingPongLayout(),
          this.getLayerWeightsLayout(),
        ],
      });
    }

    if (hasMaxPool2d) {
      this.maxpool2dBackwardShader = new ComputeShader({
        ...commonShaderProps,
        code: maxpool2dBackwardWgsl,
        workgroupCount: [1, 1, 1],
        bindingLayouts: [
          { default: [{ binding: this.maxpool2dParamsBuffer, name: "params", type: "storage" }] },
          this.getMaxPoolBackwardDataLayout(),
        ],
      });
    }

    this.weightGradientComputationShader = new ComputeShader({
      ...commonShaderProps,
      code: weightGradientComputationWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [this.getGradientInputLayout(), this.getGradientWeightLayout()],
    });

    this.biasGradientComputationShader = new ComputeShader({
      ...commonShaderProps,
      code: biasGradientComputationWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [this.getGradientInputLayout(), this.getGradientBiasLayout()],
    });

    if (hasConv2d) {
      this.conv2dBackwardWeightsShader = new ComputeShader({
        ...commonShaderProps,
        code: conv2dBackwardWeightsWgsl,
        workgroupCount: [1, 1, 1],
        bindingLayouts: [
          { default: [{ binding: this.conv2dParamsBuffer, name: "params", type: "storage" }] },
          this.getGradientInputLayout(),
          this.getGradientWeightsAndBiasesLayout(),
        ],
      });
    }

    this.updateParametersShader = new ComputeShader({
      ...commonShaderProps,
      code: updateParametersWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        { default: [{ binding: this.learningRateBuffer, name: "learning_rate", type: "uniform" }] },
        this.getUpdateGradientLayout(),
        this.getUpdateParamLayout(),
      ],
    });

    this.prepareDeltaShader = new ComputeShader({
      ...commonShaderProps,
      code: prepareDeltaWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        { default: [{ binding: this.prepareDeltaParamsBuffer, name: "params", type: "storage" }] },
        this.getPingPongDeltaLayout(),
        this.getPrepareDeltaDataLayout(),
      ],
    });

    this.softmaxShader = new ComputeShader({
      ...commonShaderProps,
      code: softmaxWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [this.getSoftmaxCombinedLayout()],
    });

    this.copyShader = new ComputeShader({
      ...commonShaderProps,
      code: copyWgsl,
      workgroupCount: [1, 1, 1],
      bindingLayouts: [
        { default: [{ binding: this.copyParamsBuffer, name: "params", type: "storage" }] },
        this.getPingPongCopyLayout(),
      ],
    });
  }

  // Layout Helpers
  private getSoftmaxCombinedLayout() {
    const layout: any = {
      default: [
        { binding: this.softmaxParamsBuffer, name: "params", type: "storage" },
        { binding: this.testActivationsBufferA, name: "inputs", type: "storage" },
        { binding: this.testActivationsBufferB, name: "outputs", type: "storage" },
      ],
    };
    this.trainingDataBuffers.forEach((_, i) => {
      if (i > 0) {
        layout[`training_layer_${i}`] = [
          { binding: this.softmaxParamsBuffer, name: "params", type: "storage" },
          { binding: this.trainingDataBuffers[i].trainingZValues!, name: "inputs", type: "storage" },
          { binding: this.trainingDataBuffers[i].trainingActivations, name: "outputs", type: "storage" },
        ];
        layout[`test_layer_${i % 2}`] = [
          { binding: this.softmaxParamsBuffer, name: "params", type: "storage" },
          { binding: this.testZValuesBuffer, name: "inputs", type: "storage" },
          { binding: i % 2 === 1 ? this.testActivationsBufferB : this.testActivationsBufferA, name: "outputs", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getPingPongCopyLayout() {
    const layout: any = {
      group0: [
        { binding: this.errorGradientsABuffer, name: "input_data", type: "storage" },
        { binding: this.errorGradientsBBuffer, name: "output_data", type: "storage" },
      ],
      group1: [
        { binding: this.errorGradientsBBuffer, name: "input_data", type: "storage" },
        { binding: this.errorGradientsABuffer, name: "output_data", type: "storage" },
      ],
    };

    // Add activation copy layouts
    this.trainingDataBuffers.forEach((_, i) => {
      if (i > 0) {
        layout[`training_layer_${i}`] = [
          { binding: this.trainingDataBuffers[i - 1].trainingActivations, name: "input_data", type: "storage" },
          { binding: this.trainingDataBuffers[i].trainingActivations, name: "output_data", type: "storage" },
        ];
        layout[`test_layer_0`] = [
          { binding: this.testActivationsBufferA, name: "input_data", type: "storage" },
          { binding: this.testActivationsBufferB, name: "output_data", type: "storage" },
        ];
        layout[`test_layer_1`] = [
          { binding: this.testActivationsBufferB, name: "input_data", type: "storage" },
          { binding: this.testActivationsBufferA, name: "output_data", type: "storage" },
        ];
      }
    });

    return layout;
  }

  private getPrepareDeltaDataLayout() {
    const layout: any = {};
    this.trainingDataBuffers.forEach((_, i) => {
      if (i > 0) {
        layout[`training_layer_${i}`] = [
          { binding: this.trainingDataBuffers[i].trainingZValues!, name: "z_values", type: "storage" },
          {
            binding: this.trainingDataBuffers[this.layers.length - 1].trainingActivations,
            name: "predictions",
            type: "storage",
          },
          { binding: this.targetsBuffer, name: "targets", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getBackpropErrorDataLayout() {
    return this.layerBuffers.reduce((obj: any, _, i) => {
      if (i > 0) {
        const nextLayerBuffer = i < this.layers.length - 1 ? this.layerBuffers[i + 1] : null;
        const weightsBinding = nextLayerBuffer?.weights ?? this.layerBuffers[i].weights;

        if (weightsBinding) {
          obj[`layer_${i}`] = [
            {
              binding: weightsBinding,
              name: "weights",
              type: "storage",
            },
          ];
        }
      }
      return obj;
    }, {});
  }

  private getLayerWeightsLayout() {
    return this.layerBuffers.reduce((obj: any, { weights, biases }, i) => {
      if (i > 0 && weights && biases) {
        obj[`layer_${i}`] = [
          { binding: weights, name: "weights", type: "storage" },
          { binding: biases, name: "biases", type: "storage" },
        ];
      }
      return obj;
    }, {});
  }

  private getLayerDataLayout() {
    const layout: any = {
      test_layer_0: [
        { binding: this.testActivationsBufferA, name: "inputs", type: "storage" },
        { binding: this.testActivationsBufferB, name: "activations", type: "storage" },
      ],
      test_layer_1: [
        { binding: this.testActivationsBufferB, name: "inputs", type: "storage" },
        { binding: this.testActivationsBufferA, name: "activations", type: "storage" },
      ],
    };
    this.trainingDataBuffers.forEach((_, i) => {
      if (i > 0) {
        layout[`training_layer_${i}`] = [
          { binding: this.trainingDataBuffers[i - 1].trainingActivations, name: "inputs", type: "storage" },
          { binding: this.trainingDataBuffers[i].trainingActivations, name: "activations", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getLayerZLayout() {
    const layout: any = { test_layer: [{ binding: this.testZValuesBuffer, name: "z_values", type: "storage" }] };
    this.trainingDataBuffers.forEach(({ trainingZValues }, i) => {
      if (i > 0 && trainingZValues) {
        layout[`training_layer_${i}`] = [{ binding: trainingZValues, name: "z_values", type: "storage" }];
      }
    });
    return layout;
  }

  private getMaxPoolDataLayout() {
    const layout: any = {
      test_layer_0: [
        { binding: this.testActivationsBufferA, name: "inputs", type: "storage" },
        { binding: this.testActivationsBufferB, name: "activations", type: "storage" },
        { binding: this.layerBuffers.find(l => l.maxIndices !== null)?.maxIndices!, name: "max_indices", type: "storage" }, // Placeholder, fixed below
      ],
      test_layer_1: [
        { binding: this.testActivationsBufferB, name: "inputs", type: "storage" },
        { binding: this.testActivationsBufferA, name: "activations", type: "storage" },
        { binding: this.layerBuffers.find(l => l.maxIndices !== null)?.maxIndices!, name: "max_indices", type: "storage" }, // Placeholder, fixed below
      ],
    };
    this.trainingDataBuffers.forEach((_, i) => {
      if (i > 0 && this.layers[i].type === LayerType.MAXPOOL2D) {
        layout[`layer_${i}`] = [
          { binding: this.trainingDataBuffers[i - 1].trainingActivations, name: "inputs", type: "storage" },
          { binding: this.trainingDataBuffers[i].trainingActivations, name: "activations", type: "storage" },
          { binding: this.layerBuffers[i].maxIndices!, name: "max_indices", type: "storage" },
        ];
        // Add test layouts for specific layers
        layout[`test_layer_${(i + 1) % 2}_layer_${i}`] = [
          { binding: (i + 1) % 2 === 1 ? this.testActivationsBufferB : this.testActivationsBufferA, name: "inputs", type: "storage" },
          { binding: (i + 1) % 2 === 1 ? this.testActivationsBufferA : this.testActivationsBufferB, name: "activations", type: "storage" },
          { binding: this.layerBuffers[i].maxIndices!, name: "max_indices", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getMaxPoolBackwardDataLayout() {
    const layout: any = {};
    this.layerBuffers.forEach((_, i) => {
      if (i > 0 && this.layers[i].type === LayerType.MAXPOOL2D) {
        layout[`layer_${i}`] = [
          { binding: this.errorGradientsABuffer, name: "next_layer_deltas", type: "storage" },
          { binding: this.errorGradientsBBuffer, name: "current_layer_weighted_sums", type: "storage" },
          { binding: this.layerBuffers[i].maxIndices!, name: "max_indices", type: "storage" },
        ];
        layout[`layer_alt_${i}`] = [
          { binding: this.errorGradientsBBuffer, name: "next_layer_deltas", type: "storage" },
          { binding: this.errorGradientsABuffer, name: "current_layer_weighted_sums", type: "storage" },
          { binding: this.layerBuffers[i].maxIndices!, name: "max_indices", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getPingPongLayout() {
    return {
      group0: [
        { binding: this.errorGradientsABuffer, name: "next_layer_deltas", type: "storage" },
        { binding: this.errorGradientsBBuffer, name: "current_layer_weighted_sums", type: "storage" },
      ],
      group1: [
        { binding: this.errorGradientsBBuffer, name: "next_layer_deltas", type: "storage" },
        { binding: this.errorGradientsABuffer, name: "current_layer_weighted_sums", type: "storage" },
      ],
    };
  }

  private getPingPongDeltaLayout() {
    return {
      group0: [
        { binding: this.errorGradientsABuffer, name: "incoming_error", type: "storage" },
        { binding: this.errorGradientsBBuffer, name: "delta", type: "storage" },
      ],
      group1: [
        { binding: this.errorGradientsBBuffer, name: "incoming_error", type: "storage" },
        { binding: this.errorGradientsABuffer, name: "delta", type: "storage" },
      ],
    };
  }

  private getGradientInputLayout() {
    const layout: any = {};
    this.layerBuffers.forEach((_, i) => {
      if (i > 0) {
        layout[`layer_${i}`] = [
          { binding: this.errorGradientsBBuffer, name: "next_layer_deltas", type: "storage" },
          { binding: this.trainingDataBuffers[i - 1].trainingActivations, name: "input_activations", type: "storage" },
        ];
        layout[`layer_alt_${i}`] = [
          { binding: this.errorGradientsABuffer, name: "next_layer_deltas", type: "storage" },
          { binding: this.trainingDataBuffers[i - 1].trainingActivations, name: "input_activations", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getGradientWeightLayout() {
    const layout: any = {};
    this.layerBuffers.forEach(({ weightGradients }, i) => {
      if (i > 0 && weightGradients) {
        layout[`layer_${i}`] = [
          { binding: this.gradientParamsBuffer, name: "grad_params", type: "storage" },
          { binding: weightGradients, name: "weight_gradients", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getGradientBiasLayout() {
    const layout: any = {};
    this.layerBuffers.forEach(({ biasGradients }, i) => {
      if (i > 0 && biasGradients) {
        layout[`layer_${i}`] = [
          { binding: this.gradientParamsBuffer, name: "grad_params", type: "storage" },
          { binding: biasGradients, name: "bias_gradients", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getGradientWeightsAndBiasesLayout() {
    const layout: any = {};
    this.layerBuffers.forEach(({ weightGradients, biasGradients }, i) => {
      if (i > 0 && weightGradients && biasGradients) {
        layout[`layer_${i}`] = [
          { binding: this.gradientParamsBuffer, name: "grad_params", type: "storage" },
          { binding: weightGradients, name: "weight_gradients", type: "storage" },
          { binding: biasGradients, name: "bias_gradients", type: "storage" },
        ];
      }
    });
    return layout;
  }

  private getUpdateGradientLayout() {
    const layout: any = {};
    this.layerBuffers.forEach(({ weightGradients, biasGradients }, i) => {
      if (i > 0) {
        if (weightGradients) layout[`weights_layer_${i}`] = [{ binding: weightGradients, name: "gradients", type: "storage" }];
        if (biasGradients) layout[`biases_layer_${i}`] = [{ binding: biasGradients, name: "gradients", type: "storage" }];
      }
    });
    return layout;
  }

  private getUpdateParamLayout() {
    const layout: any = {};
    this.layerBuffers.forEach(({ weights, biases }, i) => {
      if (i > 0) {
        if (weights) layout[`weights_layer_${i}`] = [{ binding: weights, name: "parameters", type: "storage" }];
        if (biases) layout[`biases_layer_${i}`] = [{ binding: biases, name: "parameters", type: "storage" }];
      }
    });
    return layout;
  }

  async forwardPass(inputActivations: Float32Array, trainMode: boolean = false) {
    if (!this.isInitialized) throw new Error("NeuralNetwork not initialized.");

    const batchSize = trainMode ? this.trainingBatchSize : this.testingBatchSize;
    const inputBuffer = trainMode ? this.trainingDataBuffers[0].trainingActivations : this.testActivationsBufferA;

    if (inputActivations.length !== batchSize * this.inputSize) {
      throw new Error(`Expected ${batchSize * this.inputSize} elements, got ${inputActivations.length}`);
    }

    inputBuffer.write(inputActivations);

    for (let i = 1; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const prevLayer = this.layers[i - 1];
      const activationType = i === this.layers.length - 1 ? this.outputActivationType : (layer.config as any).activation ?? this.hiddenActivationType;

      if (layer.type === LayerType.DENSE) {
        this.forwardPassParamsBuffer.write(new Uint32Array([batchSize, prevLayer.size, layer.size, activationType]));
        
        // Dynamic workgroup count for Dense
        this.forwardPassShader.props.workgroupCount = [Math.ceil((batchSize * layer.size) / 64), 1, 1];
        
        this.forwardPassShader.dispatch({
          bindGroups: {
            1: `layer_${i}`,
            2: trainMode ? `training_layer_${i}` : `test_layer_${(i + 1) % 2}`,
            3: trainMode ? `training_layer_${i}` : `test_layer`,
          },
        });
      } else if (layer.type === LayerType.CONV2D) {
        const c = layer.config as any;
        const [inH, inW, inC] = prevLayer.shape;
        const [outH, outW, outC] = layer.shape;
        this.conv2dParamsBuffer.write(new Uint32Array([
          batchSize, inH, inW, inC, outH, outW, outC, c.kernelSize, c.stride ?? 1, c.padding ?? 0, activationType
        ]));
        
        if (this.conv2dForwardShader) {
          // Dynamic workgroup count for Conv2D: [Math.ceil(outW / 16), Math.ceil(outH / 16), batchSize * outC]
          this.conv2dForwardShader.props.workgroupCount = [
            Math.ceil(outW / 16),
            Math.ceil(outH / 16),
            batchSize * outC
          ];

          this.conv2dForwardShader.dispatch({
            bindGroups: {
              1: `layer_${i}`,
              2: trainMode ? `training_layer_${i}` : `test_layer_${(i + 1) % 2}`,
              3: trainMode ? `training_layer_${i}` : `test_layer`,
            },
          });
        }
      } else if (layer.type === LayerType.MAXPOOL2D) {
        const c = layer.config as any;
        const [inH, inW, inC] = prevLayer.shape;
        const [outH, outW, outC] = layer.shape;
        this.maxpool2dParamsBuffer.write(new Uint32Array([
          batchSize, inH, inW, inC, outH, outW, c.poolSize, c.stride ?? c.poolSize
        ]));
        
        if (this.maxpool2dForwardShader) {
          // Dynamic workgroup count for MaxPool2D
          this.maxpool2dForwardShader.props.workgroupCount = [
            Math.ceil(outW / 16),
            Math.ceil(outH / 16),
            batchSize * outC
          ];

          this.maxpool2dForwardShader.dispatch({
            bindGroups: { 1: trainMode ? `layer_${i}` : `test_layer_${(i + 1) % 2}_layer_${i}` },
          });
        }
      } else if (layer.type === LayerType.FLATTEN) {
        const size = batchSize * layer.size;
        this.copyParamsBuffer.write(new Uint32Array([size]));
        this.copyShader.props.workgroupCount = [Math.ceil(size / 64), 1, 1];
        
        const inputBuffer = trainMode ? this.trainingDataBuffers[i - 1].trainingActivations : [this.testActivationsBufferA, this.testActivationsBufferB][i % 2];
        const outputBuffer = trainMode ? this.trainingDataBuffers[i].trainingActivations : [this.testActivationsBufferB, this.testActivationsBufferA][i % 2];

        // We need a layout for arbitrary buffer copies or use existing ones
        // The copyShader has getPingPongCopyLayout which uses errorGradientsABuffer/BBuffer.
        // We should add a layout for activation copies.
        this.copyShader.dispatch({
          bindGroups: {
            1: trainMode ? `training_layer_${i}` : `test_layer_${(i + 1) % 2}`
          }
        });
      }

      // Handle Softmax if it's the output layer
      if (i === this.layers.length - 1 && activationType === ActivationType.SOFTMAX) {
          this.softmaxParamsBuffer.write(new Uint32Array([batchSize, layer.size]));
          this.softmaxShader.props.workgroupCount = [Math.ceil(batchSize / 64), 1, 1];
          this.softmaxShader.dispatch({
              bindGroups: {
                  0: trainMode ? `training_layer_${i}` : `test_layer_${i % 2}`
              }
          });
      }
    }

    if (!trainMode) {
      const lastLayerBuffer = [this.testActivationsBufferB, this.testActivationsBufferA][this.layers.length % 2];
      return (await lastLayerBuffer.read()) as Float32Array;
    }
    return null as any;
  }

  private async backwardPass(learningRate: number) {
    if (!this.isInitialized) throw new Error("NeuralNetwork not initialized.");

    await this.learningRateBuffer.write(new Float32Array([learningRate]));

    for (let i = this.layers.length - 1; i >= 1; i--) {
      const layer = this.layers[i];
      const prevLayer = this.layers[i - 1];
      const activationType = i === this.layers.length - 1 ? this.outputActivationType : (layer.config as any).activation ?? this.hiddenActivationType;

      // Fixed buffer assignment for simplicity: 
      // Buffer A (errorGradientsABuffer) always holds weighted sums
      // Buffer B (errorGradientsBBuffer) always holds deltas
      const prepareDeltaGroup = "group0"; // A -> B
      const propagateErrorGroup = "group1"; // B -> A
      const deltaBufferName = "layer"; // uses Buffer B for next_layer_deltas

      // Step A: Prepare Delta
      // delta = incoming_error * activation_derivative(z)
      this.prepareDeltaParamsBuffer.write(new Uint32Array([
        this.trainingBatchSize, layer.size, activationType, i === this.layers.length - 1 ? 1 : 0
      ]));
      this.prepareDeltaShader.props.workgroupCount = [Math.ceil((this.trainingBatchSize * layer.size) / 64), 1, 1];
      this.prepareDeltaShader.dispatch({
        bindGroups: {
          1: prepareDeltaGroup,
          2: `training_layer_${i}`
        }
      });

      // Step B: Compute Gradients
      // weight_grad = delta * input_activations
      if (layer.type === LayerType.DENSE) {
        this.gradientParamsBuffer.write(new Uint32Array([this.trainingBatchSize, prevLayer.size, layer.size, 0]));
        this.weightGradientComputationShader.props.workgroupCount = [Math.ceil((layer.size * prevLayer.size) / 256), 1, 1];
        this.weightGradientComputationShader.dispatch({
          bindGroups: {
            0: `${deltaBufferName}_${i}`,
            1: `layer_${i}`
          }
        });
        this.biasGradientComputationShader.props.workgroupCount = [layer.size, 1, 1];
        this.biasGradientComputationShader.dispatch({
          bindGroups: {
            0: `${deltaBufferName}_${i}`,
            1: `layer_${i}`
          }
        });
      } else if (layer.type === LayerType.CONV2D) {
        const c = layer.config as any;
        const [inH, inW, inC] = prevLayer.shape;
        const [outH, outW, outC] = layer.shape;
        this.conv2dParamsBuffer.write(new Uint32Array([
          this.trainingBatchSize, inH, inW, inC, outH, outW, outC, c.kernelSize, c.stride ?? 1, c.padding ?? 0, activationType
        ]));
        if (this.conv2dBackwardWeightsShader) {
          const totalWeights = c.kernelSize * c.kernelSize * inC * outC;
          this.conv2dBackwardWeightsShader.props.workgroupCount = [Math.ceil(totalWeights / 64), 1, 1];
          this.conv2dBackwardWeightsShader.dispatch({
            bindGroups: {
              0: "default",
              1: `${deltaBufferName}_${i}`,
              2: `layer_${i}`
            },
          });
        }
      }

      // Step C: Propagate Error to previous layer
      // incoming_error_prev = weights^T * delta
      if (i > 1) { // No need to propagate error to the input layer
        if (layer.type === LayerType.DENSE) {
          this.backwardPassParamsBuffer.write(new Uint32Array([this.trainingBatchSize, prevLayer.size, layer.size]));
          this.backwardErrorShader.props.workgroupCount = [Math.ceil((this.trainingBatchSize * prevLayer.size) / 64), 1, 1];
          this.backwardErrorShader.dispatch({
            bindGroups: {
              0: propagateErrorGroup,
              1: `layer_${i}`
            }
          });
        } else if (layer.type === LayerType.CONV2D) {
          const [inH, inW, inC] = prevLayer.shape;
          if (this.conv2dBackwardDataShader) {
            this.conv2dBackwardDataShader.props.workgroupCount = [
              Math.ceil(inW / 16),
              Math.ceil(inH / 16),
              this.trainingBatchSize * inC
            ];
            this.conv2dBackwardDataShader.dispatch({
              bindGroups: {
                0: "default",
                1: propagateErrorGroup,
                2: `layer_${i}`
              },
            });
          }
        } else if (layer.type === LayerType.MAXPOOL2D) {
          const [inH, inW, inC] = prevLayer.shape;
          if (this.maxpool2dBackwardShader) {
            this.maxpool2dBackwardShader.props.workgroupCount = [
              Math.ceil(inW / 16),
              Math.ceil(inH / 16),
              this.trainingBatchSize * inC
            ];
            this.maxpool2dBackwardShader.dispatch({
              bindGroups: {
                0: "default",
                1: `layer_alt_${i}`
              }
            });
          }
        } else if (layer.type === LayerType.FLATTEN) {
          const size = this.trainingBatchSize * layer.size;
          this.copyParamsBuffer.write(new Uint32Array([size]));
          this.copyShader.props.workgroupCount = [Math.ceil(size / 64), 1, 1];
          this.copyShader.dispatch({
            bindGroups: {
              1: propagateErrorGroup
            }
          });
        }
      }

      // Update Parameters
      if (this.layerBuffers[i].weights) {
        const totalParams = this.layerBuffers[i].weights!.sizeElements;
        this.updateParametersShader.props.workgroupCount = [Math.ceil(totalParams / 256), 1, 1];
        this.updateParametersShader.dispatch({ bindGroups: { 1: `weights_layer_${i}`, 2: `weights_layer_${i}` } });
      }
      if (this.layerBuffers[i].biases) {
        const totalParams = this.layerBuffers[i].biases!.sizeElements;
        this.updateParametersShader.props.workgroupCount = [Math.ceil(totalParams / 256), 1, 1];
        this.updateParametersShader.dispatch({ bindGroups: { 1: `biases_layer_${i}`, 2: `biases_layer_${i}` } });
      }
    }
  }

  private async lossPass() {
    const isSoftmax = this.outputActivationType === ActivationType.SOFTMAX;
    const lossType = isSoftmax ? 1 : 0; // 1 for Cross-Entropy, 0 for MSE
    
    await this.lossParamsBuffer.write(new Uint32Array([this.trainingBatchSize, this.outputSize, lossType, 0, LOSS_MULTIPLIER]));
    
    // Dynamic dispatch for loss calculation
    this.lossShader.props.workgroupCount = [Math.ceil(this.trainingBatchSize / 64), 1, 1];
    
    this.lossShader.dispatch();
    return ((await this.totalBatchLossBuffer.read()) as Uint32Array)[0];
  }

  async train(props: TrainOptions) {
    if (!this.isInitialized) throw new Error("NeuralNetwork not initialized.");
    const learningRate = props.learningRate ?? 0.01;
    const epochs = props.epochs ?? 1;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const numSamples = props.inputActivations.length;
      let epochLoss = 0;
      const indices = Array.from({ length: numSamples }, (_, i) => i).sort(() => Math.random() - 0.5);

      for (let batchStart = 0; batchStart < numSamples; batchStart += this.trainingBatchSize) {
        if (batchStart + this.trainingBatchSize > numSamples) continue;

        const batchInputs = new Float32Array(this.trainingBatchSize * this.inputSize);
        const batchTargets = new Float32Array(this.trainingBatchSize * this.outputSize);

        for (let i = 0; i < this.trainingBatchSize; i++) {
          const idx = indices[batchStart + i];
          batchInputs.set(props.inputActivations[idx], i * this.inputSize);
          batchTargets.set(props.targetActivations[idx], i * this.outputSize);
        }

        await this.targetsBuffer.write(batchTargets);
        await this.forwardPass(batchInputs, true);
        await this.totalBatchLossBuffer.write(new Uint32Array([0]));
        const batchLoss = props.progressCallback ? (await this.lossPass()) / LOSS_MULTIPLIER : 0;
        epochLoss += batchLoss / (numSamples / this.trainingBatchSize);
        await this.backwardPass(learningRate);
      }
      props.progressCallback?.(epoch, epochLoss);
    }
  }

  // === Genetic Evaluation (Forward + Loss per genome) ===
  async evaluatePopulation(props: EvaluatePopulationOptions) {
    if (!this.isInitialized) throw new Error("NeuralNetwork not initialized.");

    const P = props.populationSize;
    const B = props.batchSize;
    const numLayers = this.layers.length;

    // Check if we need to re-initialize genetic assets (if size changed)
    const needsReinit = P !== this.currentGeneticPopulationSize || B !== this.currentGeneticBatchSize;

    if (needsReinit) {
      this.currentGeneticPopulationSize = P;
      this.currentGeneticBatchSize = B;

      // Dispose old buffers if needed (simple-compute-shaders might need explicit disposal, but let's assume GC for now)
      this.geneticWeightsBuffers = new Array(numLayers);
      this.geneticBiasesBuffers = new Array(numLayers);
      this.geneticActivationsBuffers = new Array(numLayers);

      this.geneticActivationsBuffers[0] = new StorageBuffer({
        dataType: "array<f32>",
        size: P * B * this.inputSize,
        canCopyDst: true,
      });

      for (let i = 1; i < numLayers; i++) {
        const layer = this.layers[i];
        const prevLayer = this.layers[i - 1];

        if (layer.type === LayerType.DENSE) {
          const weightSize = prevLayer.size * layer.size;
          this.geneticWeightsBuffers[i] = new StorageBuffer({
            dataType: "array<f32>",
            size: P * weightSize,
            canCopyDst: true,
          });
          this.geneticBiasesBuffers[i] = new StorageBuffer({
            dataType: "array<f32>",
            size: P * layer.size,
            canCopyDst: true,
          });
          this.geneticActivationsBuffers[i] = new StorageBuffer({
            dataType: "array<f32>",
            size: P * B * layer.size,
            canCopySrc: true,
          });
        } else {
          throw new Error(`Genetic evaluation not yet implemented for layer type: ${layer.type}`);
        }
      }

      this.forwardGeneticShader = new ComputeShader({
        useExecutionCountBuffer: false,
        useTimeBuffer: false,
        code: forwardPassGeneticWgsl,
        workgroupCount: [1024, 1], // Placeholder
        bindingLayouts: [
          { default: [{ binding: this.forwardGeneticParamsBuffer, name: "params", type: "storage" }] },
          this.geneticWeightsBuffers.reduce((obj: any, _, i) => {
            if (i > 0)
              obj[`layer_${i}`] = [
                { binding: this.geneticWeightsBuffers[i], name: "weights", type: "storage" },
                { binding: this.geneticBiasesBuffers[i], name: "biases", type: "storage" },
              ];
            return obj;
          }, {}),
          this.geneticActivationsBuffers.reduce((obj: any, _, i) => {
            if (i > 0)
              obj[`layer_${i}`] = [
                { binding: this.geneticActivationsBuffers[i - 1], name: "inputs", type: "storage" },
                { binding: this.geneticActivationsBuffers[i], name: "activations", type: "storage" },
              ];
            return obj;
          }, {}),
          { default: [] }, // No z-values
        ],
      });
    }

    // Pack and write data to buffers
    const packedInputs = new Float32Array(P * B * this.inputSize);
    for (let g = 0; g < P; g++) {
      packedInputs.set(props.inputs[g], g * B * this.inputSize);
    }
    await this.geneticActivationsBuffers[0].write(packedInputs);

    for (let i = 1; i < numLayers; i++) {
      const layer = this.layers[i];
      const prevLayer = this.layers[i - 1];

      if (layer.type === LayerType.DENSE) {
        const weightSize = prevLayer.size * layer.size;
        const packedWeights = new Float32Array(P * weightSize);
        const packedBiases = new Float32Array(P * layer.size);
        for (let g = 0; g < P; g++) {
          packedWeights.set(props.weights[i][g], g * weightSize);
          packedBiases.set(props.biases[i][g], g * layer.size);
        }
        await this.geneticWeightsBuffers[i].write(packedWeights);
        await this.geneticBiasesBuffers[i].write(packedBiases);
      }
    }

    const maxLayerSize = Math.max(...this.layers.map((l) => l.size));

    for (let i = 1; i < numLayers; i++) {
      const layer = this.layers[i];
      const prevLayer = this.layers[i - 1];
      const activationType =
        i === numLayers - 1 ? this.outputActivationType : (layer.config as any).activation ?? this.hiddenActivationType;

      await this.forwardGeneticParamsBuffer.write(new Uint32Array([P, B, prevLayer.size, layer.size, activationType]));

      // Dynamic workgroup count for Genetic Forward Pass
      const totalOutputs = P * B * layer.size;
      this.forwardGeneticShader.props.workgroupCount = [Math.ceil(totalOutputs / 64), 1, 1];

      this.forwardGeneticShader.dispatch({
        bindGroups: {
          1: `layer_${i}`,
          2: `layer_${i}`,
        },
      });
    }

    let activations: Float32Array | null = null;
    if (props.returnActivations) {
      activations = (await this.geneticActivationsBuffers[numLayers - 1].read()) as Float32Array;
    }

    return { losses: null, activations };
  }
}
