import NeuralNetwork, { ActivationType } from "./neural-network";
import setupTestingUI from "./test-ui";
import TrainingVisualizer from "./ui";

async function start() {
	const trainingBatchSize = 10;

	const neuralNetwork = new NeuralNetwork({
		layerSizes: [3, 12, 12, 3],
		trainingBatchSize: trainingBatchSize,
		testingBatchSize: 1,
		outputActivationType: ActivationType.LINEAR,
	});
	await neuralNetwork.initialize("xavier");

	const inputActivations = [];
	const targetActivations = [];

	// Testing training:
	const totalExamples = 100;
	const numBatches = totalExamples / trainingBatchSize;

	const edgeCases = [
		{ input: [0, 0, 0], target: [1, 1, 1] },
		{ input: [1, 1, 1], target: [0, 0, 0] },
		{ input: [1, 0, 0], target: [0, 1, 1] },
		{ input: [0, 1, 0], target: [1, 0, 1] },
		{ input: [0, 0, 1], target: [1, 1, 0] },
	];

	for (let b = 0; b < numBatches; b++) {
		const batchInputs = [];
		const batchTargets = [];
		for (let i = 0; i < trainingBatchSize; i++) {

			let inputs;
			let targets;
			if(b == 0 && i < edgeCases.length){
				inputs = new Float32Array(edgeCases[i].input);
				targets = new Float32Array(edgeCases[i].target);
			}else{
				inputs = new Float32Array([Math.random(), Math.random(), Math.random()]);

				targets = new Float32Array([
					1 - inputs[0],  // Invert R
					1 - inputs[1],  // Invert G
					1 - inputs[2]   // Invert B
				]);
			}


			batchInputs.push(inputs);
			batchTargets.push(targets);
		}
		inputActivations.push(...batchInputs);
		targetActivations.push(...batchTargets);
	}

	// // Add these after generating random examples


	// edgeCases.forEach(({ input, target }) => {
	// 	inputActivations.push(new Float32Array(input));
	// 	targetActivations.push(new Float32Array(target));
	// });

	// Initialize the visualizer
	const visualizer = new TrainingVisualizer({
		title: 'Neural Network Training Progress',
		maxDataPoints: 1000  // Keep last 1000 data points
	});
	visualizer.initialize();

	await neuralNetwork.train({
		inputActivations,
		targetActivations,
		epochs: 1000,
		// batchSize: trainingBatchSize,
		learningRate: 0.01,
		momentum: 0.9,
		weightDecay: 0.01,
		progressCallback: (epoch, loss) => {
			visualizer.update(epoch, loss);
		}
	});

	// Get final stats
	const stats = visualizer.getStats();
	console.log('Training completed!', stats);

	// Close the visualizer (or keep it open to view results)
	visualizer.close();

	// ADD THIS:
console.log('=== INSPECTING LEARNED PARAMETERS ===');
const weightsData = await neuralNetwork.layerBuffers[1].weights.read();
const biasesData = await neuralNetwork.layerBuffers[1].biases.read();

console.log('Weights (should be close to -1 on diagonal, 0 elsewhere):');
for(let out = 0; out < 3; out++) {
    let row = [];
    for(let inp = 0; inp < 3; inp++) {
        row.push(weightsData[out * 3 + inp].toFixed(3));
    }
    console.log(`  Row ${out}: [${row.join(', ')}]`);
}

console.log('\nBiases (should be close to [1, 1, 1]):');
console.log(`  [${Array.from(biasesData).map(b => b.toFixed(3)).join(', ')}]`);

// Manually compute what the network will output for edge cases
console.log('\n=== MANUAL FORWARD PASS ===');
const testCases = [
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
];

testCases.forEach(input => {
    let output = [0, 0, 0];
    for(let out = 0; out < 3; out++) {
        for(let inp = 0; inp < 3; inp++) {
            output[out] += weightsData[out * 3 + inp] * input[inp];
        }
        output[out] += biasesData[out];
    }
    console.log(`Input [${input}] → Output [${output.map(v => v.toFixed(3))}]`);
});

	setupTestingUI(neuralNetwork);
}

start();