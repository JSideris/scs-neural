import { NeuralNetwork } from '../../../src';

export interface Genome {
  weights: Float32Array[];
  biases: Float32Array[];
  id: number; // Matches ant.id
  fitness?: number;
}

export const INPUT_H = 7;
export const INPUT_W = 7;
export const INPUT_C = 23;
export const OUTPUT_SIZE = 8;

export interface PretrainingScenario {
  inputActivations: Float32Array[];
  targetActivations: Float32Array[];
}

export function generatePretrainingScenarios(): PretrainingScenario {
  const inputActivations: Float32Array[] = [];
  const targetActivations: Float32Array[] = [];

  const addScenario = (params: {
    foodPos?: { dx: number; dy: number };
    rockPos?: { dx: number; dy: number };
    forcedMove?: number;
    foodCarried?: number;
    queenDir?: { dx: number; dy: number };
  }) => {
    const { foodPos, rockPos, forcedMove, foodCarried = 0, queenDir = { dx: 0, dy: 0 } } = params;
    const inputs = new Float32Array(INPUT_H * INPUT_W * INPUT_C);
    const targets = new Float32Array(OUTPUT_SIZE);

    const randomHunger = 0.5 + Math.random() * 0.5; // Mostly full/healthy for pretraining
    const randomHistory = Array.from({ length: 6 }, () => Math.floor(Math.random() * 3) - 1);

    // Fill with dirt and shared scalars
    for (let gy = 0; gy < INPUT_H; gy++) {
      for (let gx = 0; gx < INPUT_W; gx++) {
        const baseIdx = (gy * INPUT_W + gx) * INPUT_C;
        inputs[baseIdx + 8] = 1.0; // Dirt
        inputs[baseIdx + 11] = 1.0; // health
        inputs[baseIdx + 12] = randomHunger;
        inputs[baseIdx + 13] = foodCarried;
        inputs[baseIdx + 14] = queenDir.dx;
        inputs[baseIdx + 15] = queenDir.dy;

        for (let h = 0; h < 6; h++) {
          inputs[baseIdx + 16 + h] = randomHistory[h];
        }
        inputs[baseIdx + 22] = 0.2; // Colony size
      }
    }

    // Ant at center
    const centerIdx = (3 * INPUT_W + 3) * INPUT_C;
    inputs[centerIdx + 2] = 1.0; // Friendly Ant
    inputs[centerIdx + 8] = 0.0; // Remove Dirt

    // Place food
    if (foodPos) {
      const foodIdx = ((foodPos.dy + 3) * INPUT_W + (foodPos.dx + 3)) * INPUT_C;
      inputs[foodIdx + 0] = 1.0;
      inputs[foodIdx + 8] = 0.0;
    }

    // Place rock
    if (rockPos) {
      const rockIdx = ((rockPos.dy + 3) * INPUT_W + (rockPos.dx + 3)) * INPUT_C;
      inputs[rockIdx + 1] = 1.0;
      inputs[rockIdx + 8] = 0.0;
    }

    // Determine target move
    let move = 0;
    if (forcedMove !== undefined) {
      move = forcedMove;
    } else if (foodPos && foodCarried < 1.0) {
      // Seek food if not full
      if (Math.abs(foodPos.dx) > Math.abs(foodPos.dy)) {
        move = foodPos.dx > 0 ? 4 : 3;
      } else if (Math.abs(foodPos.dy) > Math.abs(foodPos.dx)) {
        move = foodPos.dy > 0 ? 2 : 1;
      }
    } else if (queenDir.dx !== 0 || queenDir.dy !== 0) {
      // Seek queen if carrying food or no food in view
      if (Math.abs(queenDir.dx) > Math.abs(queenDir.dy)) {
        move = queenDir.dx > 0 ? 4 : 3;
      } else {
        move = queenDir.dy > 0 ? 2 : 1;
      }
    }

    targets[move] = 1.0;
    inputActivations.push(inputs);
    targetActivations.push(targets);
  };

  // 1. Food Seeking Scenarios (Ant is empty, ignores random queen compass)
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (dx === 0 && dy === 0) continue;

      const randomQueenDir = {
        dx: Math.random() * 2 - 1,
        dy: Math.random() * 2 - 1,
      };

      if (Math.abs(dx) === Math.abs(dy)) {
        // Diagonal: Rock blocking one path
        addScenario({
          foodPos: { dx, dy },
          rockPos: { dx: 0, dy: Math.sign(dy) },
          forcedMove: Math.sign(dx) > 0 ? 4 : 3,
          queenDir: randomQueenDir,
        });
        addScenario({
          foodPos: { dx, dy },
          rockPos: { dx: Math.sign(dx), dy: 0 },
          forcedMove: Math.sign(dy) > 0 ? 2 : 1,
          queenDir: randomQueenDir,
        });
      } else {
        addScenario({ foodPos: { dx, dy }, queenDir: randomQueenDir });
      }
    }
  }

  // 2. Return Home Scenarios (Ant is full, follows queen compass)
  // We'll generate scenarios for 8 directions of the queen compass
  const directions = [
    { dx: 1, dy: 0 },
    { dx: -1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: 0, dy: -1 },
    { dx: 0.7, dy: 0.7 },
    { dx: -0.7, dy: 0.7 },
    { dx: 0.7, dy: -0.7 },
    { dx: -0.7, dy: -0.7 },
  ];

  for (const qDir of directions) {
    // Basic return home
    addScenario({ foodCarried: 1.0, queenDir: qDir });

    // Return home with food in view (conflict - should still go home)
    // Place food in opposite direction of queen
    addScenario({
      foodPos: { dx: -Math.sign(qDir.dx) * 2, dy: -Math.sign(qDir.dy) * 2 },
      foodCarried: 1.0,
      queenDir: qDir,
    });

    // Return home with rock blocking direct path
    if (qDir.dx !== 0 && qDir.dy !== 0) {
      // Diagonal return home with rock
      addScenario({
        foodCarried: 1.0,
        queenDir: qDir,
        rockPos: { dx: 0, dy: Math.sign(qDir.dy) },
        forcedMove: Math.sign(qDir.dx) > 0 ? 4 : 3,
      });
      addScenario({
        foodCarried: 1.0,
        queenDir: qDir,
        rockPos: { dx: Math.sign(qDir.dx), dy: 0 },
        forcedMove: Math.sign(qDir.dy) > 0 ? 2 : 1,
      });
    } else {
      // Straight return home with rock in front (should try to move around? 
      // For now, let's just teach it to not crash if possible, but the game 
      // doesn't have a simple "around" logic without more state. 
      // Let's at least teach it to follow compass when clear.)
      addScenario({
        foodCarried: 1.0,
        queenDir: qDir,
        rockPos: { dx: qDir.dx, dy: qDir.dy },
        forcedMove: qDir.dx !== 0 ? (qDir.dy > 0 ? 2 : 1) : (qDir.dx > 0 ? 4 : 3), // Side step
      });
    }
  }

  return { inputActivations, targetActivations };
}

export async function runPretraining(
  nn: NeuralNetwork,
  progressCallback?: (epoch: number, loss: number) => void
): Promise<Genome> {
  const { inputActivations, targetActivations } = generatePretrainingScenarios();

  await nn.train({
    inputActivations,
    targetActivations,
    epochs: 200,
    learningRate: 0.05,
    progressCallback,
  });

  const weights: Float32Array[] = [null as any];
  const biases: Float32Array[] = [null as any];

  for (let i = 1; i < nn.layers.length; i++) {
    const buffer = nn.layerBuffers[i];
    if (buffer.weights) {
      weights.push(new Float32Array(await buffer.weights.read()));
    } else {
      weights.push(null as any);
    }
    if (buffer.biases) {
      biases.push(new Float32Array(await buffer.biases.read()));
    } else {
      biases.push(null as any);
    }
  }

  return { weights, biases, id: -1, fitness: 1000 };
}
