import React, { useEffect, useRef, useState, useCallback } from 'react';
import { NeuralNetwork, ActivationType, LayerType } from '../../src';
import { generateDataset } from './shape-generator';
import { LossGraph } from '../components/LossGraph';
import { StatCard } from '../components/StatCard';
import { Play, Eraser, Zap, Activity } from 'lucide-react';

export const ShapeClassifier: React.FC = () => {
  const [nn, setNn] = useState<NeuralNetwork | null>(null);
  const [training, setTraining] = useState(false);
  const [epochs, setEpochs] = useState<number[]>([]);
  const [losses, setLosses] = useState<number[]>([]);
  const [stats, setStats] = useState({ epoch: 0, loss: 0 });
  const [predictions, setPredictions] = useState({ circle: 0, square: 0 });
  const [status, setStatus] = useState('Initializing...');
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nnRef = useRef<NeuralNetwork | null>(null);
  const isDrawing = useRef(false);

  const IMG_SIZE = 28;

  useEffect(() => {
    const init = async () => {
      if (!navigator.gpu) {
        setError("WebGPU is not supported in this browser.");
        return;
      }

      try {
        const network = new NeuralNetwork({
          layers: [
            { type: LayerType.INPUT, shape: [IMG_SIZE, IMG_SIZE, 1] },
            { type: LayerType.CONV2D, kernelSize: 3, filters: 4, stride: 1, padding: 1, activation: ActivationType.RELU },
            { type: LayerType.MAXPOOL2D, poolSize: 2, stride: 2 },
            { type: LayerType.FLATTEN },
            { type: LayerType.DENSE, size: 16, activation: ActivationType.RELU },
            { type: LayerType.DENSE, size: 2, activation: ActivationType.SOFTMAX },
          ],
          trainingBatchSize: 20,
          testingBatchSize: 1,
        });
        await network.initialize("xavier");
        nnRef.current = network;
        setNn(network);
        setStatus('Ready to train');
        clearCanvas();
      } catch (e) {
        console.error("Failed to initialize WebGPU:", e);
        setError("Failed to initialize WebGPU: " + (e as Error).message);
      }
    };
    init();
  }, []);

  const startTraining = async () => {
    if (!nnRef.current || training) return;
    setTraining(true);
    setError(null);
    setStatus('Generating dataset...');
    
    try {
      const TRAINING_COUNT = 1000;
      const EPOCHS = 150;
      const dataset = generateDataset(TRAINING_COUNT, IMG_SIZE);
      
      const inputActivations = dataset.map(d => d.image);
      const targetActivations = dataset.map(d => {
        const target = new Float32Array(2);
        target[d.label] = 1.0;
        return target;
      });

      setStatus('Training...');
      await nnRef.current.train({
        inputActivations,
        targetActivations,
        epochs: EPOCHS,
        learningRate: 0.05,
        progressCallback: (epoch, loss) => {
          setEpochs(prev => [...prev, epoch]);
          setLosses(prev => [...prev, loss]);
          setStats({ epoch, loss });
        }
      });

      setStatus('Training complete! Draw a shape.');
    } catch (e) {
      console.error("Training failed:", e);
      setError("Training failed: " + (e as Error).message);
    } finally {
      setTraining(false);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictions({ circle: 0, square: 0 });
  };

  const predict = async () => {
    if (!nnRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = IMG_SIZE;
    tempCanvas.height = IMG_SIZE;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.drawImage(canvas, 0, 0, IMG_SIZE, IMG_SIZE);
    
    const imageData = tempCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const input = new Float32Array(IMG_SIZE * IMG_SIZE);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      const gray = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
      input[i / 4] = 1.0 - (gray / 255.0);
    }

    const output = await nnRef.current.forwardPass(input);
    setPredictions({ circle: output[0], square: output[1] });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    isDrawing.current = true;
    draw(e);
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
    const ctx = canvasRef.current?.getContext('2d');
    ctx?.beginPath();
    predict();
  };

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#f8fafc', marginBottom: '0.5rem' }}>Shape Classifier</h2>
        <p style={{ color: '#94a3b8' }}>A Convolutional Neural Network (CNN) that can distinguish between circles and squares. Train it, then draw in the box!</p>
      </header>

      {error && (
        <div style={{ 
          padding: '1rem', 
          backgroundColor: '#451a1a', 
          border: '1px solid #7f1d1d', 
          borderRadius: '0.5rem', 
          color: '#fca5a5',
          fontWeight: 500
        }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '2rem' }}>
        {/* Left: Training */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ 
            background: '#1e293b', 
            padding: '1.5rem', 
            borderRadius: '0.75rem', 
            border: '1px solid #334155',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem'
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Model Status</h3>
            <div style={{ 
              padding: '0.75rem', 
              backgroundColor: '#0f172a', 
              borderRadius: '0.5rem', 
              color: '#38bdf8',
              fontFamily: 'monospace',
              fontSize: '0.875rem'
            }}>
              {status}
            </div>
            <button
              onClick={startTraining}
              disabled={training || !nn}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                padding: '0.75rem',
                backgroundColor: training ? '#334155' : '#0284c7',
                color: 'white',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: training ? 'not-allowed' : 'pointer',
                fontWeight: 600
              }}
            >
              {training ? <Activity className="animate-spin" size={20} /> : <Zap size={20} />}
              {training ? 'Training CNN...' : 'Start Training'}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem' }}>
            <StatCard label="Epoch" value={stats.epoch} />
            <StatCard label="Loss" value={stats.loss.toFixed(6)} />
          </div>

          <LossGraph epochs={epochs} losses={losses} title="CNN Training Loss" />
        </div>

        {/* Right: Interaction */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ 
            background: '#1e293b', 
            padding: '1.5rem', 
            borderRadius: '0.75rem', 
            border: '1px solid #334155',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem'
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Draw Here</h3>
            
            <div style={{ position: 'relative', width: '280px', height: '280px', margin: '0 auto' }}>
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onMouseDown={handleMouseDown}
                onMouseMove={draw}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  width: '280px',
                  height: '280px',
                  backgroundColor: 'white',
                  borderRadius: '0.5rem',
                  cursor: 'crosshair',
                  touchAction: 'none'
                }}
              />
              <button
                onClick={clearCanvas}
                style={{
                  position: 'absolute',
                  bottom: '0.75rem',
                  right: '0.75rem',
                  padding: '0.5rem',
                  backgroundColor: '#f1f5f9',
                  color: '#64748b',
                  borderRadius: '0.375rem',
                  border: '1px solid #e2e8f0',
                  cursor: 'pointer'
                }}
              >
                <Eraser size={18} />
              </button>
            </div>

            <span>Make sure your shape is large enough or the prediction will be inaccurate.</span>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>Circle</span>
                  <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#38bdf8' }}>{(predictions.circle * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '0.5rem', backgroundColor: '#334155', borderRadius: '1rem', overflow: 'hidden' }}>
                  <div style={{ width: `${predictions.circle * 100}%`, height: '100%', backgroundColor: '#38bdf8', transition: 'width 0.3s' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>Square</span>
                  <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#38bdf8' }}>{(predictions.square * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '0.5rem', backgroundColor: '#334155', borderRadius: '1rem', overflow: 'hidden' }}>
                  <div style={{ width: `${predictions.square * 100}%`, height: '100%', backgroundColor: '#38bdf8', transition: 'width 0.3s' }} />
                </div>
              </div>
            </div>

            {Math.max(predictions.circle, predictions.square) > 0.5 && (
              <div style={{ 
                textAlign: 'center', 
                padding: '1rem', 
                backgroundColor: 'rgba(56, 189, 248, 0.1)', 
                borderRadius: '0.5rem',
                border: '1px solid rgba(56, 189, 248, 0.2)',
                color: '#38bdf8',
                fontWeight: 600
              }}>
                It's a {predictions.circle > predictions.square ? 'Circle' : 'Square'}!
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
