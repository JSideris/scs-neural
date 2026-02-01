import React, { useEffect, useRef, useState } from 'react';
import { NeuralNetwork, ActivationType, LayerType } from '../../../src';
import { FlappyBirdGame } from './game';
import { StatCard } from '../../components/StatCard';
import { Play, Pause, RotateCcw } from 'lucide-react';

interface Genome {
  weights: Float32Array[];
  biases: Float32Array[];
  fitness: number;
  isAlive: boolean;
}

export const FlappyBird: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [generation, setGeneration] = useState(0);
  const [bestFitnessEver, setBestFitnessEver] = useState(0);
  const [aliveCount, setAliveCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const nnRef = useRef<NeuralNetwork | null>(null);
  const gameRef = useRef<FlappyBirdGame | null>(null);
  const populationRef = useRef<Genome[]>([]);
  const requestRef = useRef<number | null>(null);
  const initializingRef = useRef(false);
  const isRunningRef = useRef(false);

  const POPULATION_SIZE = 100;
  const INPUT_SIZE = 5;
  const OUTPUT_SIZE = 1;

  useEffect(() => {
    if (initializingRef.current) return;
    initializingRef.current = true;

    const init = async () => {
      if (!navigator.gpu) {
        setError("WebGPU is not supported in this browser.");
        return;
      }

      try {
        const nn = new NeuralNetwork({
          layers: [
            { type: LayerType.INPUT, shape: [INPUT_SIZE] },
            { type: LayerType.DENSE, size: 8 },
            { type: LayerType.DENSE, size: 8 },
            { type: LayerType.DENSE, size: OUTPUT_SIZE },
          ],
          trainingBatchSize: 1,
          testingBatchSize: 1,
          outputActivationType: ActivationType.LINEAR,
        });
        await nn.initialize("xavier");
        nnRef.current = nn;

        gameRef.current = new FlappyBirdGame(POPULATION_SIZE);
        
        const initialPop: Genome[] = [];
        for (let i = 0; i < POPULATION_SIZE; i++) {
          initialPop.push(createRandomGenome(nn));
        }
        populationRef.current = initialPop;
        setAliveCount(POPULATION_SIZE);
      } catch (err) {
        console.error("Failed to initialize WebGPU:", err);
        setError("Failed to initialize WebGPU: " + (err as Error).message);
      }
    };

    init();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  const createRandomGenome = (nn: NeuralNetwork): Genome => {
    const weights: Float32Array[] = [null as any]; 
    const biases: Float32Array[] = [null as any]; 

    for (let i = 1; i < nn.layers.length; i++) {
      const layer = nn.layers[i];
      const prevLayer = nn.layers[i-1];
      const scale = Math.sqrt(2.0 / prevLayer.size);
      const w = new Float32Array(prevLayer.size * layer.size);
      const b = new Float32Array(layer.size);
      
      for (let j = 0; j < w.length; j++) w[j] = (Math.random() * 2 - 1) * scale;
      for (let j = 0; j < b.length; j++) b[j] = (Math.random() * 2 - 1) * 0.1;
      
      weights.push(w);
      biases.push(b);
    }
    return { weights, biases, fitness: 0, isAlive: true };
  };

  const evolvePopulation = () => {
    const sorted = [...populationRef.current].sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.floor(POPULATION_SIZE * 0.1);
    const newPopulation: Genome[] = sorted.slice(0, eliteCount).map(g => ({
      ...cloneGenome(g),
      isAlive: true,
      fitness: 0
    }));

    while (newPopulation.length < POPULATION_SIZE) {
      const p1 = tournamentSelect(sorted, 5);
      const p2 = tournamentSelect(sorted, 5);
      const child = crossover(p1, p2);
      mutate(child, 0.1, 0.2);
      child.isAlive = true;
      child.fitness = 0;
      newPopulation.push(child);
    }
    populationRef.current = newPopulation;
  };

  const cloneGenome = (g: Genome): Genome => ({
    weights: g.weights.map(w => w ? new Float32Array(w) : null as any),
    biases: g.biases.map(b => b ? new Float32Array(b) : null as any),
    fitness: g.fitness,
    isAlive: g.isAlive,
  });

  const tournamentSelect = (pop: Genome[], size: number) => {
    let best = pop[Math.floor(Math.random() * pop.length)];
    for (let i = 1; i < size; i++) {
      const comp = pop[Math.floor(Math.random() * pop.length)];
      if (comp.fitness > best.fitness) best = comp;
    }
    return best;
  };

  const crossover = (p1: Genome, p2: Genome): Genome => {
    const child = cloneGenome(p1);
    for (let l = 1; l < child.weights.length; l++) {
      for (let i = 0; i < child.weights[l].length; i++) {
        if (Math.random() < 0.5) child.weights[l][i] = p2.weights[l][i];
      }
      for (let i = 0; i < child.biases[l].length; i++) {
        if (Math.random() < 0.5) child.biases[l][i] = p2.biases[l][i];
      }
    }
    return child;
  };

  const mutate = (g: Genome, rate: number, strength: number) => {
    for (let l = 1; l < g.weights.length; l++) {
      for (let i = 0; i < g.weights[l].length; i++) {
        if (Math.random() < rate) g.weights[l][i] += (Math.random() * 2 - 1) * strength;
      }
      for (let i = 0; i < g.biases[l].length; i++) {
        if (Math.random() < rate) g.biases[l][i] += (Math.random() * 2 - 1) * strength;
      }
    }
  };

  const runStep = async () => {
    if (!isRunning || !gameRef.current || !nnRef.current) return;

    const game = gameRef.current;
    const nn = nnRef.current;
    const pop = populationRef.current;

    const aliveIndices: number[] = [];
    const inputs: Float32Array[] = [];

    for (let i = 0; i < POPULATION_SIZE; i++) {
      if (pop[i].isAlive) {
        aliveIndices.push(i);
        const state = game.getBirdState(i);
        inputs.push(new Float32Array([
          state.birdY / 600,
          (state.birdVelocity + 10) / 20,
          state.nextPipeX / 400,
          state.pipeTopY / 600,
          state.pipeBottomY / 600,
        ]));
      }
    }

    if (aliveIndices.length === 0) {
      // End generation
      const genBest = pop.reduce((b, g) => g.fitness > b.fitness ? g : b);
      if (genBest.fitness > bestFitnessEver) setBestFitnessEver(genBest.fitness);
      
      // Pause for a moment so user can see the crash state
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Check if we are still running after the pause
      if (!isRunningRef.current) return;

      evolvePopulation();
      game.reset();
      
      // Update local state for display
      setGeneration(prev => prev + 1);
      setAliveCount(POPULATION_SIZE);

      // Render the reset state immediately
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) game.render(ctx);
      }
      
      // Schedule next generation
      if (isRunningRef.current) {
        requestRef.current = requestAnimationFrame(runStep);
      }
      return;
    }

    const numLayers = nn.layers.length;
    const weights: Float32Array[][] = new Array(numLayers);
    const biases: Float32Array[][] = new Array(numLayers);
    for (let i = 1; i < numLayers; i++) {
      weights[i] = aliveIndices.map(idx => pop[idx].weights[i]);
      biases[i] = aliveIndices.map(idx => pop[idx].biases[i]);
    }

    const { activations } = await nn.evaluatePopulation({
      populationSize: aliveIndices.length,
      batchSize: 1,
      weights,
      biases,
      inputs,
      returnActivations: true,
    });

    for (let i = 0; i < aliveIndices.length; i++) {
      if (activations![i] > 0.5) game.flap(aliveIndices[i]);
    }

    game.update();

    let newAliveCount = 0;
    for (let i = 0; i < POPULATION_SIZE; i++) {
      if (pop[i].isAlive) {
        if (game.isDead(i)) {
          pop[i].isAlive = false;
          pop[i].fitness = game.getScore(i);
        } else {
          newAliveCount++;
        }
      }
    }
    setAliveCount(newAliveCount);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) game.render(ctx);
    }

    if (isRunning) {
      requestRef.current = requestAnimationFrame(runStep);
    }
  };

  useEffect(() => {
    isRunningRef.current = isRunning;
    if (isRunning) {
      requestRef.current = requestAnimationFrame(runStep);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isRunning]);

  const toggleRunning = () => setIsRunning(!isRunning);

  const resetGame = () => {
    setIsRunning(false);
    isRunningRef.current = false;
    setGeneration(0);
    setBestFitnessEver(0);
    setAliveCount(POPULATION_SIZE);
    if (gameRef.current) gameRef.current.reset();
    if (nnRef.current) {
      const initialPop: Genome[] = [];
      for (let i = 0; i < POPULATION_SIZE; i++) {
        initialPop.push(createRandomGenome(nnRef.current));
      }
      populationRef.current = initialPop;
    }
    if (canvasRef.current && gameRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) gameRef.current.render(ctx);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#f8fafc', marginBottom: '0.5rem' }}>Flappy Bird Evolution</h2>
        <p style={{ color: '#94a3b8' }}>A genetic algorithm evolving neural networks to play Flappy Bird. Each bird has its own brain (weights and biases).</p>
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

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 300px', gap: '2rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <canvas 
            ref={canvasRef} 
            width={800} 
            height={600} 
            style={{ 
              width: '100%', 
              height: 'auto', 
              borderRadius: '0.75rem', 
              border: '4px solid #1e293b',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
            }} 
          />
          
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={toggleRunning}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                padding: '0.75rem',
                backgroundColor: isRunning ? '#ef4444' : '#22c55e',
                color: 'white',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              {isRunning ? <Pause size={20} /> : <Play size={20} />}
              {isRunning ? 'Pause' : 'Start Evolution'}
            </button>
            <button
              onClick={resetGame}
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

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <StatCard label="Generation" value={generation} />
          <StatCard label="Alive" value={`${aliveCount} / ${POPULATION_SIZE}`} />
          <StatCard label="Best Fitness" value={bestFitnessEver.toFixed(0)} />
          
          <div style={{ 
            background: '#1e293b', 
            padding: '1.25rem', 
            borderRadius: '0.5rem', 
            border: '1px solid #334155',
            marginTop: '1rem'
          }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.75rem' }}>How it works:</h4>
            <ul style={{ fontSize: '0.8125rem', color: '#94a3b8', paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li>The population is evaluated in parallel on the GPU.</li>
              <li>Top 10% are kept as "elites" for the next generation.</li>
              <li>Offspring are created through crossover and mutation.</li>
              <li>Inputs: bird height, velocity, and distance/gap of next pipe.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
