export default class InitializationMethods {
  // Xavier/Glorot: Good for sigmoid/tanh activations
  static initXavier(fanIn: number, fanOut: number, totalSize?: number): Float32Array {
    const scale = Math.sqrt(1 / fanIn);
    return InitializationMethods.randomArray(totalSize ?? fanIn * fanOut, -scale, scale);
  }

  // He initialization: Better for ReLU networks
  static initHe(fanIn: number, fanOut: number, totalSize?: number): Float32Array {
    const scale = Math.sqrt(2 / fanIn); // Note the 2 factor
    return InitializationMethods.randomArray(totalSize ?? fanIn * fanOut, -scale, scale);
  }

  // Uniform random in range
  static initUniform(size: number, min: number, max: number): Float32Array {
    return InitializationMethods.randomArray(size, min, max);
  }

  // Zero initialization (usually bad, but sometimes needed)
  static initZero(size: number): Float32Array {
    return new Float32Array(size); // Already zeros
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
