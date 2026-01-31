import NeuralNetwork, { ActivationType } from "./neural-network";
import FlappyBirdGame from "./flappy-bird";

interface Genome {
	weights: Float32Array[];  // per layer
	biases: Float32Array[];   // per layer
	fitness: number;
	isAlive: boolean;
}

async function start() {
	const POPULATION_SIZE = 100;
	const INPUT_SIZE = 5;  // bird_y, bird_velocity, next_pipe_x, pipe_top_y, pipe_bottom_y
	const OUTPUT_SIZE = 1; // flap or not
	
	// Create neural network for evaluation
	const neuralNetwork = new NeuralNetwork({
		layerSizes: [INPUT_SIZE, 8, 8, OUTPUT_SIZE],
		trainingBatchSize: 1,
		testingBatchSize: 1,
		outputActivationType: ActivationType.LINEAR,
	});
	
	try {
		await neuralNetwork.initialize("xavier");
	} catch (error) {
		console.error("WebGPU Initialization Failed:", error);
		const errorDiv = document.createElement("div");
		errorDiv.style.color = "red";
		errorDiv.style.padding = "20px";
		errorDiv.style.background = "#fff";
		errorDiv.style.border = "1px solid red";
		errorDiv.style.margin = "20px";
		errorDiv.innerHTML = `
			<h2>WebGPU Initialization Failed</h2>
			<p>${error.message}</p>
			<p>Your browser or system might not support WebGPU, or hardware acceleration is disabled.</p>
			<p>Try launching Chrome with: <code>--ignore-gpu-blocklist --enable-unsafe-webgpu</code></p>
		`;
		document.body.prepend(errorDiv);
		return;
	}

	// Initialize population
	let population: Genome[] = [];
	for (let i = 0; i < POPULATION_SIZE; i++) {
		population.push(createRandomGenome(neuralNetwork.layerSizes));
	}

	let generation = 0;
	let bestFitnessEver = 0;
	let bestGenomeEver: Genome = null;

	// Create game visualization
	const game = new FlappyBirdGame(POPULATION_SIZE);
	
	// Main evolution loop
	async function runGeneration() {
		generation++;
		game.reset();
		
		// Reset fitness
		population.forEach(genome => {
			genome.fitness = 0;
			genome.isAlive = true;
		});

		let aliveCount = POPULATION_SIZE;
		let frameCount = 0;
		const MAX_FRAMES = 10000; // Prevent infinite loops

		// Game loop
		while (aliveCount > 0 && frameCount < MAX_FRAMES) {
			frameCount++;
			
			// Prepare inputs for all alive birds
			const aliveIndices: number[] = [];
			const inputs: Float32Array[] = [];
			
			for (let i = 0; i < POPULATION_SIZE; i++) {
				if (population[i].isAlive) {
					aliveIndices.push(i);
					const state = game.getBirdState(i);
					inputs.push(new Float32Array([
						state.birdY / 600,           // Normalize to 0-1
						(state.birdVelocity + 10) / 20, // Normalize velocity
						state.nextPipeX / 400,
						state.pipeTopY / 600,
						state.pipeBottomY / 600,
					]));
				}
			}

			if (aliveIndices.length === 0) break;

			// Prepare weights and biases for alive birds
			const numLayers = neuralNetwork.layerSizes.length;
			const weights: Float32Array[][] = new Array(numLayers);
			const biases: Float32Array[][] = new Array(numLayers);
			
			weights[0] = [];
			biases[0] = [];
			
			for (let layer = 1; layer < numLayers; layer++) {
				weights[layer] = aliveIndices.map(idx => population[idx].weights[layer]);
				biases[layer] = aliveIndices.map(idx => population[idx].biases[layer]);
			}

			// Evaluate all alive birds in parallel
			const { activations } = await neuralNetwork.evaluatePopulation({
				populationSize: aliveIndices.length,
				batchSize: 1,
				weights,
				biases,
				inputs,
				returnActivations: true,
			});

			// Apply outputs and update game
			for (let i = 0; i < aliveIndices.length; i++) {
				const birdIndex = aliveIndices[i];
				const output = activations[i]; // Single output value
				
				// If output > 0.5, flap
				if (output > 0.5) {
					game.flap(birdIndex);
				}
			}

			// Update game physics
			game.update();

			// Check collisions and update fitness
			for (let i = 0; i < POPULATION_SIZE; i++) {
				if (population[i].isAlive) {
					if (game.isDead(i)) {
						population[i].isAlive = false;
						population[i].fitness = game.getScore(i);
						aliveCount--;
					}
				}
			}

			// Render every frame
			game.render(generation, population, bestFitnessEver);
			
			// Small delay to make it visible
			await new Promise(resolve => setTimeout(resolve, 1000 / 60));
		}

		// Generation complete - assign fitness to any remaining alive birds
		for (let i = 0; i < POPULATION_SIZE; i++) {
			if (population[i].isAlive) {
				population[i].fitness = game.getScore(i);
			}
		}

		// Find best genome
		const generationBest = population.reduce((best, genome) => 
			genome.fitness > best.fitness ? genome : best
		);

		if (generationBest.fitness > bestFitnessEver) {
			bestFitnessEver = generationBest.fitness;
			bestGenomeEver = cloneGenome(generationBest);
		}

		console.log(`Generation ${generation}: Best=${generationBest.fitness.toFixed(1)}, AllTime=${bestFitnessEver.toFixed(1)}`);

		// Evolve population
		population = evolvePopulation(population, neuralNetwork.layerSizes);

		// Continue to next generation
		setTimeout(runGeneration, 100);
	}

	runGeneration();
}

function createRandomGenome(layerSizes: number[]): Genome {
	const weights: Float32Array[] = [null]; // No weights for input layer
	const biases: Float32Array[] = [null];  // No biases for input layer

	for (let i = 1; i < layerSizes.length; i++) {
		const inputSize = layerSizes[i - 1];
		const outputSize = layerSizes[i];
		
		// Xavier initialization
		const scale = Math.sqrt(2.0 / inputSize);
		const w = new Float32Array(inputSize * outputSize);
		const b = new Float32Array(outputSize);
		
		for (let j = 0; j < w.length; j++) {
			w[j] = (Math.random() * 2 - 1) * scale;
		}
		for (let j = 0; j < b.length; j++) {
			b[j] = (Math.random() * 2 - 1) * 0.1;
		}
		
		weights.push(w);
		biases.push(b);
	}

	return { weights, biases, fitness: 0, isAlive: true };
}

function cloneGenome(genome: Genome): Genome {
	return {
		weights: genome.weights.map(w => w ? new Float32Array(w) : null),
		biases: genome.biases.map(b => b ? new Float32Array(b) : null),
		fitness: genome.fitness,
		isAlive: genome.isAlive,
	};
}

function evolvePopulation(population: Genome[], layerSizes: number[]): Genome[] {
	// Sort by fitness
	const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
	
	// Keep top 10%
	const eliteCount = Math.floor(population.length * 0.1);
	const newPopulation: Genome[] = sorted.slice(0, eliteCount).map(cloneGenome);

	// Fill rest with offspring
	while (newPopulation.length < population.length) {
		// Tournament selection
		const parent1 = tournamentSelect(sorted, 5);
		const parent2 = tournamentSelect(sorted, 5);
		
		const child = crossover(parent1, parent2);
		mutate(child, 0.1, 0.2); // 10% mutation rate, 20% mutation strength
		
		newPopulation.push(child);
	}

	return newPopulation;
}

function tournamentSelect(population: Genome[], tournamentSize: number): Genome {
	let best = population[Math.floor(Math.random() * population.length)];
	for (let i = 1; i < tournamentSize; i++) {
		const competitor = population[Math.floor(Math.random() * population.length)];
		if (competitor.fitness > best.fitness) {
			best = competitor;
		}
	}
	return best;
}

function crossover(parent1: Genome, parent2: Genome): Genome {
	const child = cloneGenome(parent1);
	
	// Uniform crossover
	for (let layer = 1; layer < child.weights.length; layer++) {
		const w1 = parent1.weights[layer];
		const w2 = parent2.weights[layer];
		const b1 = parent1.biases[layer];
		const b2 = parent2.biases[layer];
		const wChild = child.weights[layer];
		const bChild = child.biases[layer];
		
		for (let i = 0; i < wChild.length; i++) {
			wChild[i] = Math.random() < 0.5 ? w1[i] : w2[i];
		}
		for (let i = 0; i < bChild.length; i++) {
			bChild[i] = Math.random() < 0.5 ? b1[i] : b2[i];
		}
	}
	
	return child;
}

function mutate(genome: Genome, mutationRate: number, mutationStrength: number) {
	for (let layer = 1; layer < genome.weights.length; layer++) {
		const w = genome.weights[layer];
		const b = genome.biases[layer];
		
		for (let i = 0; i < w.length; i++) {
			if (Math.random() < mutationRate) {
				w[i] += (Math.random() * 2 - 1) * mutationStrength;
			}
		}
		for (let i = 0; i < b.length; i++) {
			if (Math.random() < mutationRate) {
				b[i] += (Math.random() * 2 - 1) * mutationStrength;
			}
		}
	}
}

start();