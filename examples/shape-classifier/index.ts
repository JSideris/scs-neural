import { NeuralNetwork, ActivationType, LayerType } from "../../src";
import TrainingVisualizer from "../shared/ui";
import { generateDataset, ShapeType } from "./shape-generator";

async function start() {
    const IMG_SIZE = 28;
    const TRAINING_COUNT = 200;
    const BATCH_SIZE = 20;
    const EPOCHS = 100;

    const predictionText = document.getElementById('prediction-text')!;
    predictionText.textContent = "Initializing WebGPU...";

    // 1. Initialize the Neural Network
    const nn = new NeuralNetwork({
        layers: [
            { type: LayerType.INPUT, shape: [IMG_SIZE, IMG_SIZE, 1] },
            { type: LayerType.CONV2D, kernelSize: 3, filters: 4, stride: 1, padding: 1, activation: ActivationType.RELU },
            { type: LayerType.MAXPOOL2D, poolSize: 2, stride: 2 },
            { type: LayerType.FLATTEN },
            { type: LayerType.DENSE, size: 16, activation: ActivationType.RELU },
            { type: LayerType.DENSE, size: 2, activation: ActivationType.SOFTMAX },
        ],
        trainingBatchSize: BATCH_SIZE,
        testingBatchSize: 1,
    });

    try {
        await nn.initialize("xavier");
    } catch (e) {
        console.error(e);
        predictionText.textContent = "Error: " + (e as Error).message;
        return;
    }

    // 2. Prepare Dataset
    predictionText.textContent = "Generating dataset...";
    const dataset = generateDataset(TRAINING_COUNT, IMG_SIZE);
    
    const inputActivations = dataset.map(d => d.image);
    const targetActivations = dataset.map(d => {
        const target = new Float32Array(2);
        target[d.label] = 1.0;
        return target;
    });

    // 3. Setup Visualization
    const visualizer = new TrainingVisualizer({
        title: 'CNN Shape Classifier Training',
        maxDataPoints: EPOCHS
    });
    visualizer.initialize('training-visualizer');

    // 4. Training
    predictionText.textContent = "Training in progress...";
    await nn.train({
        inputActivations,
        targetActivations,
        epochs: EPOCHS,
        learningRate: 0.05,
        progressCallback: (epoch, loss) => {
            visualizer.update(epoch, loss);
        }
    });
    predictionText.textContent = "Training complete! Draw a shape!";

    // 5. Interactive Drawing Logic
    setupDrawingUI(nn, IMG_SIZE);
}

function setupDrawingUI(nn: NeuralNetwork, imgSize: number) {
    const canvas = document.getElementById('draw-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    const clearBtn = document.getElementById('clear-btn')!;
    const predictBtn = document.getElementById('predict-btn')!;
    const circleBar = document.getElementById('circle-bar')!;
    const squareBar = document.getElementById('square-bar')!;
    const predictionText = document.getElementById('prediction-text')!;

    let isDrawing = false;

    // Canvas settings
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    canvas.addEventListener('mousedown', () => isDrawing = true);
    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        ctx.beginPath();
    });
    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    });

    clearBtn.addEventListener('click', () => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        updateResults([0, 0]);
        predictionText.textContent = '';
    });

    predictBtn.addEventListener('click', async () => {
        // Create a temporary canvas to downscale
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imgSize;
        tempCanvas.height = imgSize;
        const tempCtx = tempCanvas.getContext('2d')!;
        
        // Draw main canvas onto small canvas
        tempCtx.drawImage(canvas, 0, 0, imgSize, imgSize);
        
        // Get pixel data
        const imageData = tempCtx.getImageData(0, 0, imgSize, imgSize);
        const input = new Float32Array(imgSize * imgSize);
        
        for (let i = 0; i < imageData.data.length; i += 4) {
            // Convert to grayscale and invert (so drawing is 1.0 on 0.0 background)
            const r = imageData.data[i];
            const g = imageData.data[i + 1];
            const b = imageData.data[i + 2];
            const gray = (r + g + b) / 3;
            input[i / 4] = 1.0 - (gray / 255.0);
        }

        const output = await nn.forwardPass(input);
        updateResults(output);
    });

    function updateResults(output: Float32Array | number[]) {
        const circleProb = output[0] * 100;
        const squareProb = output[1] * 100;

        circleBar.style.width = `${circleProb}%`;
        squareBar.style.width = `${squareProb}%`;

        if (output[0] > 0.5 || output[1] > 0.5) {
            const result = output[0] > output[1] ? 'Circle' : 'Square';
            const confidence = Math.max(output[0], output[1]) * 100;
            predictionText.textContent = `Prediction: ${result} (${confidence.toFixed(1)}%)`;
        }
    }
}

start();
