export enum ActivationType {
  RELU = 0,
  SIGMOID = 1,
  LINEAR = 2,
  TANH = 3,
  SOFTMAX = 4,
}

export interface NeuralNetworkOptions {
  layerSizes: number[];
  trainingBatchSize?: number;
  testingBatchSize?: number;
  hiddenActivationType?: ActivationType;
  outputActivationType?: ActivationType;
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
  weights: Float32Array[][]; // [layerIndex>0][genomeIndex] length input_size*output_size
  biases: Float32Array[][];  // [layerIndex>0][genomeIndex] length output_size
  inputs: Float32Array[];    // [genomeIndex] length batchSize*inputSize
  targets?: Float32Array[];   // [genomeIndex] length batchSize*outputSize (required for loss)
  returnActivations?: boolean;
  returnLoss?: boolean;
}
