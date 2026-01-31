export default class InitializationMethods {
  // Xavier/Glorot: Good for sigmoid/tanh activations
  static initXavier(inputSize: number, outputSize: number): Float32Array {
    const scale = Math.sqrt(1 / inputSize);
    return InitializationMethods.randomArray(inputSize * outputSize, -scale, scale);
  }

  // He initialization: Better for ReLU networks
  static initHe(inputSize: number, outputSize: number): Float32Array {
    const scale = Math.sqrt(2 / inputSize); // Note the 2 factor
    return InitializationMethods.randomArray(inputSize * outputSize, -scale, scale);
  }

  // Uniform random in range
  static initUniform(inputSize: number, outputSize: number, min: number, max: number): Float32Array {
    return InitializationMethods.randomArray(inputSize * outputSize, min, max);
  }

  // Zero initialization (usually bad, but sometimes needed)
  static initZero(inputSize: number, outputSize: number): Float32Array {
    return new Float32Array(inputSize * outputSize); // Already zeros
  }

  // Helper for random arrays
  static randomArray(size: number, min: number, max: number): Float32Array {
    const data = new Float32Array(size);
    const range = max - min;
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() * range + min;
    }
    return data;
  }
}
