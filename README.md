# scs-neural

A high-performance, memory-efficient neural network library powered by WebGPU.

## 🚀 Overview

`scs-neural` is built for speed and efficiency, leveraging the parallel processing power of WebGPU. It is designed with a "performance-first" philosophy, prioritizing low-level buffer management and optimized compute shaders over high-level developer experience (DX).

### Key Features
- **WebGPU Accelerated**: All computations (forward pass, backpropagation, genetic evaluation) run on the GPU.
- **CNN Support**: Includes optimized kernels for 2D Convolution and Max Pooling.
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

Define your network architecture using the `layers` configuration.

#### Simple FFNN (Feed-Forward Neural Network)
```typescript
import { NeuralNetwork, ActivationType, LayerType } from "scs-neural";

const nn = new NeuralNetwork({
    layers: [
        { type: LayerType.INPUT, shape: [3] },
        { type: LayerType.DENSE, size: 12 },
        { type: LayerType.DENSE, size: 3 },
    ],
    trainingBatchSize: 10,
    testingBatchSize: 1,
    hiddenActivationType: ActivationType.RELU,
    outputActivationType: ActivationType.LINEAR,
});

await nn.initialize("xavier");
```

#### CNN (Convolutional Neural Network)
```typescript
const nn = new NeuralNetwork({
    layers: [
        { type: LayerType.INPUT, shape: [28, 28, 1] }, // 28x28 grayscale image
        { type: LayerType.CONV2D, kernelSize: 3, filters: 16, activation: ActivationType.RELU },
        { type: LayerType.MAXPOOL2D, poolSize: 2 },
        { type: LayerType.FLATTEN },
        { type: LayerType.DENSE, size: 10 }, // Output layer for 10 classes
    ],
    trainingBatchSize: 32,
});

await nn.initialize("he");
```

### 2. Forward Pass (Inference)

Run a single inference pass. The library handles the internal ping-ponging of buffers.

```typescript
const input = new Float32Array([...]); // Flat array matching input shape
const output = await nn.forwardPass(input);
console.log("Network Output:", output);
```

### 3. Training with Backpropagation

Train the network using supervised learning. The `train` method handles batching and shuffling.

```typescript
await nn.train({
    inputActivations: [...], // Array of Float32Arrays
    targetActivations: [...],
    epochs: 10,
    learningRate: 0.001,
    progressCallback: (epoch, loss) => {
        console.log(`Epoch ${epoch} - Loss: ${loss}`);
    }
});
```

### 4. Genetic Algorithms (Parallel Evaluation)

`scs-neural` excels at evaluating large populations in parallel, which is ideal for reinforcement learning or neuroevolution.

```typescript
const { activations } = await nn.evaluatePopulation({
    populationSize: 100,
    batchSize: 1,
    weights: populationWeights,
    biases: populationBiases,
    inputs: populationInputs,
    returnActivations: true
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

- **Shaders**: Located in `src/shaders/`, these WGSL files handle the core mathematical operations including Conv2D and MaxPool.
- **Buffer Management**: The library maintains dedicated `StorageBuffer` objects for weights, biases, gradients, and activations for every layer to avoid runtime allocations and garbage collection overhead.

## 📄 License

ISC
