export enum ActivationType {
  RELU = 0,
  SIGMOID = 1,
  LINEAR = 2,
  TANH = 3,
  SOFTMAX = 4,
}

export enum LayerType {
  INPUT = "input",
  DENSE = "dense",
  CONV2D = "conv2d",
  MAXPOOL2D = "maxpool2d",
  FLATTEN = "flatten",
}

export interface BaseLayerConfig {
  type: LayerType;
}

export interface InputLayerConfig extends BaseLayerConfig {
  type: LayerType.INPUT;
  shape: number[]; // e.g., [width, height, channels] or [size]
}

export interface DenseLayerConfig extends BaseLayerConfig {
  type: LayerType.DENSE;
  size: number;
  activation?: ActivationType;
}

export interface Conv2DLayerConfig extends BaseLayerConfig {
  type: LayerType.CONV2D;
  kernelSize: number;
  filters: number;
  stride?: number;
  padding?: number;
  activation?: ActivationType;
}

export interface MaxPool2DLayerConfig extends BaseLayerConfig {
  type: LayerType.MAXPOOL2D;
  poolSize: number;
  stride?: number;
}

export interface FlattenLayerConfig extends BaseLayerConfig {
  type: LayerType.FLATTEN;
}

export type LayerConfig =
  | InputLayerConfig
  | DenseLayerConfig
  | Conv2DLayerConfig
  | MaxPool2DLayerConfig
  | FlattenLayerConfig;

export interface NeuralNetworkOptions {
  layers?: LayerConfig[];
  layerSizes?: number[]; // Deprecated but kept for backward compatibility
  trainingBatchSize?: number;
  testingBatchSize?: number;
  hiddenActivationType?: ActivationType; // Default for dense/conv if not specified
  outputActivationType?: ActivationType; // Default for last layer if not specified
}

export interface TrainOptions {
  inputActivations: Float32Array[];
  targetActivations: Float32Array[];
  learningRate?: number;
  momentum?: number;
  weightDecay?: number;
  epochs?: number;
  progressCallback?: (epoch: number, loss: number) => void;
}

export interface EvaluatePopulationOptions {
  populationSize: number;
  batchSize: number;
  weights: Float32Array[][]; // [layerIndex>0][genomeIndex]
  biases: Float32Array[][];  // [layerIndex>0][genomeIndex]
  inputs: Float32Array[];    // [genomeIndex]
  targets?: Float32Array[];   // [genomeIndex]
  returnActivations?: boolean;
  returnLoss?: boolean;
}
