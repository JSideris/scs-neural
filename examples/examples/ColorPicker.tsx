import React, { useEffect, useState, useRef } from 'react';
import { NeuralNetwork, ActivationType, LayerType } from '../../src';
import { LossGraph } from '../components/LossGraph';
import { StatCard } from '../components/StatCard';
import { Play, RotateCcw, Activity } from 'lucide-react';

export const ColorPicker: React.FC = () => {
  const [nn, setNn] = useState<NeuralNetwork | null>(null);
  const [training, setTraining] = useState(false);
  const [epochs, setEpochs] = useState<number[]>([]);
  const [losses, setLosses] = useState<number[]>([]);
  const [inputColor, setInputColor] = useState('#808080');
  const [outputColor, setOutputColor] = useState('#808080');
  const [groundTruth, setGroundTruth] = useState('#7f7f7f');
  const [stats, setStats] = useState({
    currentEpoch: 0,
    currentLoss: 0,
    avgLoss: 0
  });
  const [error, setError] = useState<string | null>(null);

  const nnRef = useRef<NeuralNetwork | null>(null);
  const initializingRef = useRef(false);

  useEffect(() => {
    if (initializingRef.current) return;
    initializingRef.current = true;

    const init = async () => {
      if (!navigator.gpu) {
        setError("WebGPU is not supported in this browser.");
        return;
      }

      try {
        const trainingBatchSize = 10;
        const neuralNetwork = new NeuralNetwork({
          layers: [
            { type: LayerType.INPUT, shape: [3] },
            { type: LayerType.DENSE, size: 12 },
            { type: LayerType.DENSE, size: 12 },
            { type: LayerType.DENSE, size: 3 },
          ],
          trainingBatchSize: trainingBatchSize,
          testingBatchSize: 1,
          outputActivationType: ActivationType.LINEAR,
        });
        await neuralNetwork.initialize("xavier");
        nnRef.current = neuralNetwork;
        setNn(neuralNetwork);
      } catch (err) {
        console.error("Failed to initialize WebGPU:", err);
        setError("Failed to initialize WebGPU: " + (err as Error).message);
      }
    };

    init();
  }, []);

  const startTraining = async () => {
    if (!nnRef.current || training) return;
    setTraining(true);
    setEpochs([]);
    setLosses([]);

    const trainingBatchSize = 10;
    const totalExamples = 100;
    const numBatches = totalExamples / trainingBatchSize;
    const inputActivations = [];
    const targetActivations = [];

    for (let b = 0; b < numBatches; b++) {
      for (let i = 0; i < trainingBatchSize; i++) {
        const r = Math.random();
        const g = Math.random();
        const b_val = Math.random();
        inputActivations.push(new Float32Array([r, g, b_val]));
        targetActivations.push(new Float32Array([1 - r, 1 - g, 1 - b_val]));
      }
    }

    await nnRef.current.train({
      inputActivations,
      targetActivations,
      epochs: 1000,
      learningRate: 0.01,
      progressCallback: (epoch, loss) => {
        if (epoch % 10 === 0) {
          setEpochs(prev => [...prev, epoch]);
          setLosses(prev => [...prev, loss]);
          setStats(prev => ({
            currentEpoch: epoch,
            currentLoss: loss,
            avgLoss: (prev.avgLoss * prev.currentEpoch + loss) / (prev.currentEpoch + 1)
          }));
        }
      }
    });

    setTraining(false);
    updatePrediction(inputColor);
  };

  const updatePrediction = async (hex: string) => {
    if (!nnRef.current) return;
    
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    
    const input = new Float32Array([r, g, b]);
    const output = await nnRef.current.forwardPass(input);
    
    const toHex = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255))).toString(16).padStart(2, '0');
    
    setOutputColor(`#${toHex(output[0])}${toHex(output[1])}${toHex(output[2])}`.toUpperCase());
    setGroundTruth(`#${toHex(1-r)}${toHex(1-g)}${toHex(1-b)}`.toUpperCase());
  };

  const handleColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newColor = e.target.value;
    setInputColor(newColor);
    updatePrediction(newColor);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#f8fafc', marginBottom: '0.5rem' }}>Color Inverter</h2>
        <p style={{ color: '#94a3b8' }}>A simple dense network trained to invert RGB colors. Select a color below to see the network's prediction.</p>
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
        {/* Training Section */}
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
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Network Training</h3>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                onClick={startTraining}
                disabled={training || !nn}
                style={{
                  flex: 1,
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
                {training ? <Activity className="animate-spin" size={20} /> : <Play size={20} />}
                {training ? 'Training...' : 'Start Training'}
              </button>
              <button
                onClick={() => {
                  setEpochs([]);
                  setLosses([]);
                  setStats({ currentEpoch: 0, currentLoss: 0, avgLoss: 0 });
                }}
                style={{
                  padding: '0.75rem',
                  backgroundColor: '#334155',
                  color: '#94a3b8',
                  borderRadius: '0.5rem',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                <RotateCcw size={20} />
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <StatCard label="Epoch" value={stats.currentEpoch} />
            <StatCard label="Loss" value={stats.currentLoss.toFixed(6)} />
          </div>

          <LossGraph epochs={epochs} losses={losses} />
        </div>

        {/* Interaction Section */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ 
            background: '#1e293b', 
            padding: '1.5rem', 
            borderRadius: '0.75rem', 
            border: '1px solid #334155',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '1.5rem'
          }}>
            <div style={{ width: '100%', textAlign: 'left' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>Test the Network</h3>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
              <input 
                type="color" 
                value={inputColor} 
                onChange={handleColorChange}
                style={{
                  width: '150px',
                  height: '150px',
                  border: '4px solid #334155',
                  borderRadius: '1rem',
                  backgroundColor: 'transparent',
                  cursor: 'pointer'
                }}
              />
              <span style={{ fontFamily: 'monospace', fontSize: '1.125rem', color: '#f8fafc' }}>{inputColor.toUpperCase()}</span>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600 }}>INPUT COLOR</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1.5rem', width: '100%' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ 
                  width: '100%', 
                  height: '100px', 
                  backgroundColor: groundTruth, 
                  borderRadius: '0.5rem',
                  border: '1px solid #334155'
                }} />
                <span style={{ fontFamily: 'monospace', color: '#94a3b8' }}>{groundTruth}</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>GROUND TRUTH</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ 
                  width: '100%', 
                  height: '100px', 
                  backgroundColor: outputColor, 
                  borderRadius: '0.5rem',
                  border: '1px solid #334155'
                }} />
                <span style={{ fontFamily: 'monospace', color: '#94a3b8' }}>{outputColor}</span>
                <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>NETWORK OUTPUT</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
