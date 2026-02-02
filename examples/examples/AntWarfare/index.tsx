import React, { useEffect, useRef, useState } from 'react';
import { NeuralNetwork, ActivationType, LayerType } from '../../../src';
import { AntWarfareGame, Ant, Colony } from './game';
import { GameRenderer } from './graphics';
import { StatCard } from '../../components/StatCard';
import { Play, Pause, RotateCcw, Save, Upload } from 'lucide-react';

const DB_NAME = 'AntWarfareDB';
const STORE_NAME = 'genomes';

const openDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const saveToDB = async (key: string, data: any) => {
  const db = await openDB();
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(data, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

const loadFromDB = async (key: string): Promise<any> => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

interface Genome {
  weights: Float32Array[];
  biases: Float32Array[];
  id: number; // Matches ant.id
  fitness?: number;
}

export const AntWarfare: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [generation, setGeneration] = useState(0);
  const [redCount, setRedCount] = useState(0);
  const [blackCount, setBlackCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  
  const nnRef = useRef<NeuralNetwork | null>(null);
  const gameRef = useRef<AntWarfareGame | null>(null);
  const rendererRef = useRef<GameRenderer>(new GameRenderer());
  const redGenomesRef = useRef<Map<number, Genome>>(new Map());
  const blackGenomesRef = useRef<Map<number, Genome>>(new Map());
  const redHallOfFameRef = useRef<Genome[]>([]);
  const blackHallOfFameRef = useRef<Genome[]>([]);
  const bestRedGenomeRef = useRef<Genome | null>(null);
  const bestBlackGenomeRef = useRef<Genome | null>(null);
  const lastScrambleFrameRef = useRef<{ [key in Colony]: number }>({ [Colony.RED]: 0, [Colony.BLACK]: 0 });
  
  const MAX_HOF_SIZE = 10;
  
  const updateHallOfFame = (colony: Colony, genome: Genome) => {
    const hof = colony === Colony.RED ? redHallOfFameRef.current : blackHallOfFameRef.current;
    
    // Check if this genome is already in the HoF (shouldn't happen with unique IDs, but good to be safe)
    if (hof.some(g => g.id === genome.id)) return;

    if (hof.length < MAX_HOF_SIZE) {
      hof.push(genome);
    } else {
      // Replace the weakest if this one is better
      hof.sort((a, b) => (a.fitness || 0) - (b.fitness || 0));
      if ((genome.fitness || 0) > (hof[0].fitness || 0)) {
        hof[0] = genome;
      }
    }
    // Keep it sorted descending by fitness for easier selection if needed
    hof.sort((a, b) => (b.fitness || 0) - (a.fitness || 0));
  };
  
  const requestRef = useRef<number | null>(null);
  const initializingRef = useRef(false);
  const isRunningRef = useRef(false);

  const INPUT_SHAPE = [7, 7, 21]; // (7*7 grid with 21 channels)
  const OUTPUT_SIZE = 9; // Move(5), Drop(1), Attack(1), Phero0(1), Phero1(1)

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
            { type: LayerType.INPUT, shape: INPUT_SHAPE },
            { type: LayerType.CONV2D, filters: 16, kernelSize: 3, activation: ActivationType.RELU },
            { type: LayerType.FLATTEN },
            { type: LayerType.DENSE, size: 64, activation: ActivationType.RELU },
            { type: LayerType.DENSE, size: OUTPUT_SIZE, activation: ActivationType.SIGMOID },
          ],
          trainingBatchSize: 1,
          testingBatchSize: 1,
        });
        await nn.initialize("xavier");
        nnRef.current = nn;

        const game = new AntWarfareGame(60, 60);
        gameRef.current = game;
        
        game.onAntHatched = (ant) => {
            const genome = createGenomeFromBest(ant);
            if (ant.colony === Colony.RED) redGenomesRef.current.set(ant.id, genome);
            else blackGenomesRef.current.set(ant.id, genome);
        };

        // Initial ants
        game.reset(); // This triggers onAntHatched for initial eggs
        
        setRedCount(game.ants.filter(a => a.colony === Colony.RED && !a.isDead).length);
        setBlackCount(game.ants.filter(a => a.colony === Colony.BLACK && !a.isDead).length);
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

  const createRandomGenome = (id: number): Genome => {
    const nn = nnRef.current!;
    const weights: Float32Array[] = [null as any]; 
    const biases: Float32Array[] = [null as any]; 

    for (let i = 1; i < nn.layers.length; i++) {
      const layer = nn.layers[i];
      const prevLayer = nn.layers[i-1];
      
      if (layer.type === LayerType.DENSE) {
        const fanIn = prevLayer.size;
        const fanOut = layer.size;
        const scale = Math.sqrt(2.0 / fanIn);
        const w = new Float32Array(fanIn * fanOut);
        const b = new Float32Array(fanOut);
        for (let j = 0; j < w.length; j++) w[j] = (Math.random() * 2 - 1) * scale;
        for (let j = 0; j < b.length; j++) b[j] = (Math.random() * 2 - 1) * 0.1;
        weights.push(w);
        biases.push(b);
      } else if (layer.type === LayerType.CONV2D) {
        const c = layer.config as any;
        const inChannels = prevLayer.shape[2];
        const fanIn = c.kernelSize * c.kernelSize * inChannels;
        const fanOut = c.filters;
        const scale = Math.sqrt(2.0 / fanIn);
        const w = new Float32Array(fanIn * fanOut);
        const b = new Float32Array(fanOut);
        for (let j = 0; j < w.length; j++) w[j] = (Math.random() * 2 - 1) * scale;
        for (let j = 0; j < b.length; j++) b[j] = (Math.random() * 2 - 1) * 0.1;
        weights.push(w);
        biases.push(b);
      } else {
        weights.push(null as any);
        biases.push(null as any);
      }
    }
    return { weights, biases, id };
  };

  const createGenomeFromBest = (ant: Ant): Genome => {
    const colony = ant.colony;
    const hof = colony === Colony.RED ? redHallOfFameRef.current : blackHallOfFameRef.current;
    
    let parentGenome: Genome | null = null;

    if (hof.length > 0) {
      // Pick a random one from the Hall of Fame for diversity
      parentGenome = hof[Math.floor(Math.random() * hof.length)];
    } else {
      // Fallback to living ants if HoF is empty
      const game = gameRef.current!;
      const living = game.ants.filter(a => a.colony === colony && !a.isDead && a.id !== ant.id);
      
      if (living.length > 0) {
        const bestAnt = living.reduce((best, a) => {
          const scoreA = game.getFitnessScore(a);
          const scoreBest = game.getFitnessScore(best);
          if (scoreA !== scoreBest) return scoreA > scoreBest ? a : best;
          return a.health > best.health ? a : best;
        });
        const parentGenomes = colony === Colony.RED ? redGenomesRef.current : blackGenomesRef.current;
        parentGenome = parentGenomes.get(bestAnt.id) || null;
      }
    }

    // Final fallbacks
    if (!parentGenome) {
      const bestEver = colony === Colony.RED ? bestRedGenomeRef.current : bestBlackGenomeRef.current;
      if (bestEver) parentGenome = bestEver;
    }

    if (parentGenome) {
      const mutationRate = ant.bornOnRock ? 0.05 : 0.02;
      return cloneAndMutate(parentGenome, ant.id, mutationRate);
    }

    return createRandomGenome(ant.id);
  };

  const cloneAndMutate = (g: Genome, newId: number, rate: number = 0.02): Genome => {
    const weights = g.weights.map(w => w ? new Float32Array(w) : null as any);
    const biases = g.biases.map(b => b ? new Float32Array(b) : null as any);
    
    const strength = 0.1;
    for (let l = 1; l < weights.length; l++) {
      if (weights[l]) {
        for (let i = 0; i < weights[l].length; i++) {
          if (Math.random() < rate) weights[l][i] += (Math.random() * 2 - 1) * strength;
        }
      }
      if (biases[l]) {
        for (let i = 0; i < biases[l].length; i++) {
          if (Math.random() < rate) biases[l][i] += (Math.random() * 2 - 1) * strength;
        }
      }
    }
    return { weights, biases, id: newId };
  };

  const scrambleColony = (colony: Colony) => {
    if (!gameRef.current) return;
    
    // Clear Hall of Fame for this colony
    if (colony === Colony.RED) {
      redHallOfFameRef.current = [];
      bestRedGenomeRef.current = null;
    } else {
      blackHallOfFameRef.current = [];
      bestBlackGenomeRef.current = null;
    }

    // Generate new random genomes for all living ants in this colony
    const genomes = colony === Colony.RED ? redGenomesRef.current : blackGenomesRef.current;
    const livingAnts = gameRef.current.ants.filter(a => a.colony === colony && !a.isDead);
    
    livingAnts.forEach(ant => {
      genomes.set(ant.id, createRandomGenome(ant.id));
      // Reset attempted directions to give them a clean slate for the next stagnation check
      ant.directionsAttempted = { up: false, down: false, left: false, right: false };
      ant.outOfBoundsAttempts = 0;
    });

    lastScrambleFrameRef.current[colony] = game.frameCount;
    // Reset the last food frame so they have a full window to find food with their new brains
    game.lastFoodFrame[colony] = game.frameCount;
    
    showStatus(`${colony === Colony.RED ? 'Red' : 'Black'} colony scrambled due to stagnation!`);
  };

  const checkStagnation = () => {
    if (!gameRef.current) return;
    const game = gameRef.current;
    
    // Wait at least 2000 frames from game start
    if (game.frameCount < 2000) return;

    [Colony.RED, Colony.BLACK].forEach(colony => {
      const framesSinceFood = game.frameCount - game.lastFoodFrame[colony];
      const framesSinceScramble = game.frameCount - lastScrambleFrameRef.current[colony];
      
      // Criteria: 
      // 1. 1500+ frames since last food
      // 2. Ensure at least 1000 frames since last scramble
      if (framesSinceFood > 1500 && framesSinceScramble > 1000) {
        const livingAnts = game.ants.filter(a => a.colony === colony && !a.isDead);
        if (livingAnts.length > 0) {
          // 3. Average unique directions < 1.5
          let totalDirections = 0;
          livingAnts.forEach(ant => {
            const count = (ant.directionsAttempted.up ? 1 : 0) + 
                          (ant.directionsAttempted.down ? 1 : 0) + 
                          (ant.directionsAttempted.left ? 1 : 0) + 
                          (ant.directionsAttempted.right ? 1 : 0);
            totalDirections += count;
          });
          const avgDirections = totalDirections / livingAnts.length;
          
          if (avgDirections < 1.5) {
            scrambleColony(colony);
          }
        }
      }
    });
  };

  const runStep = async () => {
    if (!isRunningRef.current || !gameRef.current || !nnRef.current) return;

    const game = gameRef.current;
    const nn = nnRef.current;
    const aliveAnts = game.ants.filter(a => !a.isDead);

    if (aliveAnts.length > 0) {
        const inputs = aliveAnts.map(ant => game.getAntState(ant));
        
        const weights: Float32Array[][] = new Array(nn.layers.length);
        const biases: Float32Array[][] = new Array(nn.layers.length);
        
        for (let l = 1; l < nn.layers.length; l++) {
            weights[l] = aliveAnts.map(ant => {
                const map = ant.colony === Colony.RED ? redGenomesRef.current : blackGenomesRef.current;
                return map.get(ant.id)!.weights[l];
            });
            biases[l] = aliveAnts.map(ant => {
                const map = ant.colony === Colony.RED ? redGenomesRef.current : blackGenomesRef.current;
                return map.get(ant.id)!.biases[l];
            });
        }

        const { activations } = await nn.evaluatePopulation({
            populationSize: aliveAnts.length,
            batchSize: 1,
            weights,
            biases,
            inputs,
            returnActivations: true,
        });

        const decisions = new Map<number, { move: number, attack: boolean, dropFood: boolean, pheromones: boolean[] }>();
        
        for (let i = 0; i < aliveAnts.length; i++) {
            const ant = aliveAnts[i];
            const output = activations!.subarray(i * OUTPUT_SIZE, (i + 1) * OUTPUT_SIZE);
            
            // Output interpretation:
            // 0-4: Move (Stay, Up, Down, Left, Right)
            // 5: Drop Food
            // 6: Attack
            // 7-8: Pheromones
            
            let move = 0;
            let maxMoveVal = output[0];
            let allSame = true;
            for (let j = 1; j < 5; j++) {
                if (Math.abs(output[j] - maxMoveVal) > 0.01) {
                    allSame = false;
                }
                if (output[j] > maxMoveVal) {
                    maxMoveVal = output[j];
                    move = j;
                }
            }

            // Tie-breaker: if all movement outputs are very similar, pick a random direction
            if (allSame) {
                move = Math.floor(Math.random() * 5);
            }

            decisions.set(ant.id, {
                move,
                dropFood: output[5] > 0.5,
                attack: output[6] > 0.5,
                pheromones: [output[7] > 0.5, output[8] > 0.5]
            });
        }

        game.update(decisions);
        checkStagnation();
        
        // Track best genomes
        aliveAnts.forEach(ant => {
            const genome = ant.colony === Colony.RED ? redGenomesRef.current.get(ant.id) : blackGenomesRef.current.get(ant.id);
            if (!genome) return;

            const bestRef = ant.colony === Colony.RED ? bestRedGenomeRef : bestBlackGenomeRef;
            if (!bestRef.current) {
                bestRef.current = genome;
            } else {
                // Determine if this ant is better than our current best ref
                // For simplicity, we can't easily compare the current best ref's ant because it might be dead
                // So we'll update bestRef whenever an ant performs exceptionally well
                const currentBestFitness = 0; // We'd need to store fitness in Genome to do this properly
                // Let's just update it if the current ant has more kills than any previous record we've seen?
                // Actually, createGenomeFromBest already finds the best living ant.
                // Let's just grab the current best living ant every 100 frames to update the Ref.
            }
        });

        if (game.frameCount % 100 === 0) {
            [Colony.RED, Colony.BLACK].forEach(colony => {
                const living = game.ants.filter(a => a.colony === colony && !a.isDead);
                if (living.length > 0) {
                    const bestAnt = living.reduce((best, a) => {
                        const scoreA = game.getFitnessScore(a);
                        const scoreBest = game.getFitnessScore(best);
                        if (scoreA !== scoreBest) return scoreA > scoreBest ? a : best;
                        
                        const distA = game.getDistToQueen(a);
                        const distBest = game.getDistToQueen(best);
                        if (distA !== distBest) return distA > distBest ? a : best;
                        
                        return a.health > best.health ? a : best;
                    });
                    const genome = colony === Colony.RED ? redGenomesRef.current.get(bestAnt.id) : blackGenomesRef.current.get(bestAnt.id);
                    if (genome) {
                        if (colony === Colony.RED) bestRedGenomeRef.current = genome;
                        else bestBlackGenomeRef.current = genome;
                    }
                }
            });
        }
        
        // Clean up genomes for dead ants
        for (let i = game.ants.length - 1; i >= 0; i--) {
            const ant = game.ants[i];
            if (ant.isDead) {
                const genomes = ant.colony === Colony.RED ? redGenomesRef.current : blackGenomesRef.current;
                const genome = genomes.get(ant.id);
                if (genome) {
                    genome.fitness = game.getFitnessScore(ant);
                    updateHallOfFame(ant.colony, genome);
                    genomes.delete(ant.id);
                }
            }
        }
    } else {
        game.update(new Map());
    }

    setRedCount(game.ants.filter(a => a.colony === Colony.RED && !a.isDead).length);
    setBlackCount(game.ants.filter(a => a.colony === Colony.BLACK && !a.isDead).length);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) rendererRef.current.render(ctx, game);
    }

    if (isRunningRef.current) {
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

  const saveModel = async () => {
    try {
      if (redHallOfFameRef.current.length > 0) {
        await saveToDB('red_hof', redHallOfFameRef.current);
      }
      if (blackHallOfFameRef.current.length > 0) {
        await saveToDB('black_hof', blackHallOfFameRef.current);
      }
      showStatus('Model and Hall of Fame saved to IndexedDB');
    } catch (err) {
      console.error('Failed to save model:', err);
      setError('Failed to save model');
    }
  };

  const loadModel = async () => {
    try {
      const redHof = await loadFromDB('red_hof');
      const blackHof = await loadFromDB('black_hof');
      
      if (redHof) {
        redHallOfFameRef.current = redHof;
        bestRedGenomeRef.current = redHof[0];
      }
      if (blackHof) {
        blackHallOfFameRef.current = blackHof;
        bestBlackGenomeRef.current = blackHof[0];
      }
      
      if (redHof || blackHof) {
        showStatus('Hall of Fame loaded from IndexedDB');
      } else {
        showStatus('No saved model found');
      }
    } catch (err) {
      console.error('Failed to load model:', err);
      setError('Failed to load model');
    }
  };

  const showStatus = (msg: string) => {
    setStatusMessage(msg);
    setTimeout(() => setStatusMessage(null), 3000);
  };

  const resetGame = () => {
    setIsRunning(false);
    isRunningRef.current = false;
    if (gameRef.current) {
        redGenomesRef.current.clear();
        blackGenomesRef.current.clear();
        redHallOfFameRef.current = [];
        blackHallOfFameRef.current = [];
        bestRedGenomeRef.current = null;
        bestBlackGenomeRef.current = null;
        lastScrambleFrameRef.current = { [Colony.RED]: 0, [Colony.BLACK]: 0 };
        gameRef.current.reset();
        setRedCount(gameRef.current.ants.filter(a => a.colony === Colony.RED && !a.isDead).length);
        setBlackCount(gameRef.current.ants.filter(a => a.colony === Colony.BLACK && !a.isDead).length);
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) rendererRef.current.render(ctx, gameRef.current);
        }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#f8fafc', marginBottom: '0.5rem' }}>Ant Warfare Evolution</h2>
        <p style={{ color: '#94a3b8' }}>Two warring ant colonies evolve through continuous genetic selection. Ants must gather food, defend their queen, and compete for resources.</p>
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
          <div style={{ position: 'relative', width: '100%', paddingBottom: '100%' }}>
            <canvas 
              ref={canvasRef} 
              width={600} 
              height={600} 
              style={{ 
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%', 
                height: '100%', 
                borderRadius: '0.75rem', 
                border: '4px solid #1e293b',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                backgroundColor: '#3e2723'
              }} 
            />
          </div>
          
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

          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={saveModel}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                padding: '0.75rem',
                backgroundColor: '#334155',
                color: '#f8fafc',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              <Save size={20} />
              Save Model
            </button>
            <button
              onClick={loadModel}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                padding: '0.75rem',
                backgroundColor: '#334155',
                color: '#f8fafc',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              <Upload size={20} />
              Load Model
            </button>
          </div>

          {statusMessage && (
            <div style={{
              padding: '0.75rem',
              backgroundColor: '#1e293b',
              color: '#38bdf8',
              borderRadius: '0.5rem',
              textAlign: 'center',
              fontSize: '0.875rem',
              fontWeight: 500,
              border: '1px solid #334155'
            }}>
              {statusMessage}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <StatCard label="Red Colony" value={redCount} />
          <StatCard label="Black Colony" value={blackCount} />
          
          <div style={{ 
            background: '#1e293b', 
            padding: '1.25rem', 
            borderRadius: '0.5rem', 
            border: '1px solid #334155',
            marginTop: '1rem'
          }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.75rem' }}>Legend:</h4>
            <ul style={{ fontSize: '0.8125rem', color: '#94a3b8', paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li><span style={{ color: '#b71c1c' }}>■</span> Red Queen / Ant</li>
              <li><span style={{ color: '#212121' }}>■</span> Black Queen / Ant</li>
              <li><span style={{ color: '#4caf50' }}>■</span> Food</li>
              <li><span style={{ color: '#9e9e9e' }}>■</span> Rock</li>
              <li><span style={{ color: '#ffcdd2' }}>●</span> Red Egg</li>
              <li><span style={{ color: '#f5f5f5' }}>●</span> Black Egg</li>
            </ul>
          </div>

          <div style={{ 
            background: '#1e293b', 
            padding: '1.25rem', 
            borderRadius: '0.5rem', 
            border: '1px solid #334155',
          }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.75rem' }}>Rules:</h4>
            <ul style={{ fontSize: '0.8125rem', color: '#94a3b8', paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li>Ants evolve continuously via mutation.</li>
              <li>Hatching ants inherit from the most successful living ant.</li>
              <li>Gather food to feed the queen for more eggs.</li>
              <li>Push rocks to clear paths or block enemies.</li>
              <li>Protect your queen! Defence is higher near her.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
