# scs-neural

A high-performance, memory-efficient neural network library powered by WebGPU.

## 🚀 Overview

`scs-neural` is built for speed and efficiency, leveraging the parallel processing power of WebGPU. It is designed with a "performance-first" philosophy, prioritizing low-level buffer management and optimized compute shaders over high-level developer experience (DX).

### Key Features
- **WebGPU Accelerated**: All computations (forward pass, backpropagation, genetic evaluation) run on the GPU.
- **Memory Efficient**: Uses manual buffer swapping (ping-ponging) instead of high-level tensor abstractions to minimize memory overhead and data transfers.
- **Flexible Training**: Supports both standard backpropagation (SGD) and highly parallelized Genetic Algorithm (GA) evaluation.
- **Direct Shader Control**: Built on top of `simple-compute-shaders` for direct WGSL control.

## 🛠 Core Philosophy: Performance over DX

This library does not use standard tensor libraries. Instead:
1. **Buffer Swapping**: We use a "ping-pong" technique where two buffers are swapped between layers to avoid unnecessary allocations during inference and training.
2. **Direct Memory Access**: Inputs and outputs are handled as flat `Float32Array` buffers, mapped directly to GPU memory.
3. **Minimized Host-Device Sync**: Training loops and genetic evaluations are designed to keep data on the GPU as much as possible, reducing the bottleneck of CPU-GPU communication.

## 📦 Installation

```bash
npm install scs-neural
```

## 📖 Usage

### 1. Initialization

Define your network architecture and initialize it with WebGPU.

```typescript
import { NeuralNetwork, ActivationType } from "scs-neural";

const nn = new NeuralNetwork({
    layerSizes: [3, 12, 12, 3], // Input, Hidden Layers, Output
    trainingBatchSize: 10,
    testingBatchSize: 1,
    hiddenActivationType: ActivationType.RELU,
    outputActivationType: ActivationType.LINEAR,
});

// Initialize with a weight initialization method
await nn.initialize("xavier"); // Options: 'xavier', 'he', 'uniform', 'zero'
```

### 2. Forward Pass (Inference)

Run a single inference pass. The library handles the internal ping-ponging of buffers.

```typescript
const input = new Float32Array([0.5, 0.2, 0.8]);
const output = await nn.forwardPass(input);
console.log("Network Output:", output);
```

### 3. Training with Backpropagation

Train the network using supervised learning. The `train` method handles batching and shuffling.

```typescript
await nn.train({
    inputActivations: [new Float32Array([0, 0, 0]), ...],
    targetActivations: [new Float32Array([1, 1, 1]), ...],
    epochs: 1000,
    learningRate: 0.01,
    progressCallback: (epoch, loss) => {
        console.log(`Epoch ${epoch} - Loss: ${loss}`);
    }
});
```

### 4. Genetic Algorithms (Parallel Evaluation)

`scs-neural` excels at evaluating large populations in parallel, which is ideal for reinforcement learning or neuroevolution. This happens in a single GPU dispatch.

```typescript
const { activations, losses } = await nn.evaluatePopulation({
    populationSize: 100,
    batchSize: 1,
    weights: populationWeights, // [layerIndex][genomeIndex]
    biases: populationBiases,   // [layerIndex][genomeIndex]
    inputs: populationInputs,   // [genomeIndex]
    returnActivations: true,
    returnLoss: false
});
```

## 🧪 Examples

This repository contains several examples showcasing the library:

- **Flappy Bird Genetic Algorithm**: Training birds to play Flappy Bird using parallel genetic evaluation.
- **Color Inversion**: Standard supervised training to invert RGB colors.

To run the examples locally:

```bash
npm install
npm run dev
```

Then open the URL provided by Vite.

## 🏗 Architecture

The core library is located in `src/`.

- **Shaders**: Located in `src/shaders/`, these WGSL files handle the core mathematical operations.
- **Buffer Management**: The library maintains dedicated `StorageBuffer` objects for weights, biases, gradients, and activations for every layer to avoid runtime allocations and garbage collection overhead.

## 📄 License

ISC
