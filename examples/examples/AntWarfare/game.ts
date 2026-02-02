export enum TileType {
  DIRT = 0,
  FOOD = 1,
  ROCK = 2,
  RED_ANT = 3,
  BLACK_ANT = 4,
  RED_QUEEN = 5,
  BLACK_QUEEN = 6,
  RED_EGG = 7,
  BLACK_EGG = 8,
}

export enum Colony {
  RED = 0,
  BLACK = 1,
}

export interface Ant {
  id: number;
  colony: Colony;
  x: number;
  y: number;
  health: number;
  hunger: number;
  foodCarried: number;
  isDead: boolean;
  kills: number;
  totalDamageDealt: number;
  foodCollected: number;
  foodReturnedHome: number;
  distanceTravelled: number;
  outOfBoundsAttempts: number;
  directionsAttempted: { up: boolean, down: boolean, left: boolean, right: boolean };
  lastMoveDirection: { dx: number, dy: number };
  bornOnRock: boolean;
}

export interface Egg {
  colony: Colony;
  x: number;
  y: number;
  hatchProgress: number; // 0 to 1
  bornOnRock: boolean;
}

export interface Rock {
  x: number;
  y: number;
  damage: number;
  seed: number;
}

export interface Pheromones {
  red: number[];
  black: number[];
}

export class AntWarfareGame {
  public width: number;
  public height: number;
  public grid: TileType[][];
  public pheromones: Pheromones[][];
  public rocks: Map<string, Rock>; // key: "x,y"
  public eggs: Egg[];
  public ants: Ant[];
  public frameCount: number = 0;

  private readonly MAX_ANTS_PER_COLONY = 50;
  private readonly MAX_FOOD_ON_MAP = 100;
  private readonly STARVE_RATE = 0.1;
  private readonly MOVE_HUNGER_COST = 0.5;
  private readonly STAY_HUNGER_COST = 0.1;
  private readonly ATTACK_HUNGER_COST = 1.0;
  private readonly PUSH_HUNGER_COST = 1.0;
  private readonly FOOD_HUNGER_RESTORE = 30;
  public static readonly MAX_HUNGER = 100;
  public static readonly MAX_HEALTH = 100;
  private readonly EGG_HATCH_TIME = 200; // frames
  private readonly PHEROMONE_FADE_RATE = 0.005;
  public static readonly PHEROMONE_CHANNELS = 2;
  public readonly QUEEN_POSITIONS = {
    [Colony.RED]: { x: 5, y: 5 },
    [Colony.BLACK]: { x: 54, y: 54 }, // Will adjust based on width/height
  };

  constructor(width: number = 60, height: number = 60) {
    this.width = width;
    this.height = height;
    this.QUEEN_POSITIONS[Colony.BLACK] = { x: width - 7, y: height - 7 };
    this.grid = Array(height).fill(null).map(() => Array(width).fill(TileType.DIRT));
    this.pheromones = Array(height).fill(null).map(() => 
      Array(width).fill(null).map(() => ({ 
        red: Array(AntWarfareGame.PHEROMONE_CHANNELS).fill(0), 
        black: Array(AntWarfareGame.PHEROMONE_CHANNELS).fill(0) 
      }))
    );
    this.rocks = new Map();
    this.eggs = [];
    this.ants = [];
    this.reset();
  }

  public reset() {
    this.frameCount = 0;
    this.grid = Array(this.height).fill(null).map(() => Array(this.width).fill(TileType.DIRT));
    this.pheromones = Array(this.height).fill(null).map(() => 
      Array(this.width).fill(null).map(() => ({ 
        red: Array(AntWarfareGame.PHEROMONE_CHANNELS).fill(0), 
        black: Array(AntWarfareGame.PHEROMONE_CHANNELS).fill(0) 
      }))
    );
    this.rocks.clear();
    this.eggs = [];
    this.ants = [];

    // Place Queens
    this.placeQueen(Colony.RED, this.QUEEN_POSITIONS[Colony.RED].x, this.QUEEN_POSITIONS[Colony.RED].y);
    this.placeQueen(Colony.BLACK, this.QUEEN_POSITIONS[Colony.BLACK].x, this.QUEEN_POSITIONS[Colony.BLACK].y);

    // Initial Rocks
    for (let i = 0; i < 100; i++) {
      const x = Math.floor(Math.random() * this.width);
      const y = Math.floor(Math.random() * this.height);
      if (this.grid[y][x] === TileType.DIRT) {
        this.grid[y][x] = TileType.ROCK;
        this.rocks.set(`${x},${y}`, { x, y, damage: 0, seed: Math.random() });
      }
    }

    // Initial Food
    for (let i = 0; i < 40; i++) {
      this.spawnFood(1);
    }
    for (let i = 0; i < 5; i++) {
      this.spawnFood(2);
    }
    this.spawnFood(4);

    // Initial Eggs
    this.spawnInitialEggs(Colony.RED);
    this.spawnInitialEggs(Colony.BLACK);
  }

  private placeQueen(colony: Colony, qx: number, qy: number) {
    const type = colony === Colony.RED ? TileType.RED_QUEEN : TileType.BLACK_QUEEN;
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        this.grid[qy + dy][qx + dx] = type;
      }
    }
  }

  private spawnInitialEggs(colony: Colony) {
    const qpos = this.QUEEN_POSITIONS[colony];
    const type = colony === Colony.RED ? TileType.RED_EGG : TileType.BLACK_EGG;
    let count = 0;
    for (let dy = -1; dy <= 2 && count < 12; dy++) {
      for (let dx = -1; dx <= 2 && count < 12; dx++) {
        const nx = qpos.x + dx;
        const ny = qpos.y + dy;
        if (nx >= 0 && nx < this.width && ny >= 0 && ny < this.height) {
          if (this.grid[ny][nx] === TileType.DIRT || this.grid[ny][nx] === TileType.ROCK) {
            const isRock = this.grid[ny][nx] === TileType.ROCK;
            this.grid[ny][nx] = type;
            this.eggs.push({ colony, x: nx, y: ny, hatchProgress: 0, bornOnRock: isRock });
            if (isRock) this.rocks.delete(`${nx},${ny}`);
            count++;
          }
        }
      }
    }
  }

  private spawnFood(size: number = 1) {
    const startX = Math.floor(Math.random() * (this.width - size + 1));
    const startY = Math.floor(Math.random() * (this.height - size + 1));

    // Check if entire area is clear (DIRT)
    let canPlace = true;
    for (let dy = 0; dy < size; dy++) {
      for (let dx = 0; dx < size; dx++) {
        if (this.grid[startY + dy][startX + dx] !== TileType.DIRT) {
          canPlace = false;
          break;
        }
      }
      if (!canPlace) break;
    }

    if (canPlace) {
      for (let dy = 0; dy < size; dy++) {
        for (let dx = 0; dx < size; dx++) {
          this.grid[startY + dy][startX + dx] = TileType.FOOD;
        }
      }
    } else if (size === 1) {
      // Fallback for 1x1 if the random spot was taken (original behavior)
      // but maybe just let it fail for simplicity or try again.
      // Original code didn't try again.
    }
  }

  public update(antDecisions: Map<number, { move: number, attack: boolean, dropFood: boolean, pheromones: boolean[] }>) {
    this.frameCount++;

    // Fade pheromones
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        for (let c = 0; c < AntWarfareGame.PHEROMONE_CHANNELS; c++) {
          this.pheromones[y][x].red[c] = Math.max(0, this.pheromones[y][x].red[c] - this.PHEROMONE_FADE_RATE);
          this.pheromones[y][x].black[c] = Math.max(0, this.pheromones[y][x].black[c] - this.PHEROMONE_FADE_RATE);
        }
      }
    }

    // Random food drop
    if (this.frameCount % 10 === 0) {
      let foodCount = 0;
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          if (this.grid[y][x] === TileType.FOOD) foodCount++;
        }
      }
      if (foodCount < this.MAX_FOOD_ON_MAP) {
        const rand = Math.random();
        if (rand < 0.05) this.spawnFood(4);      // Very rare 4x4
        else if (rand < 0.2) this.spawnFood(2);  // Occasional 2x2
        else this.spawnFood(1);                  // Regular 1x1
      }
    }

    // Update Eggs
    for (let i = this.eggs.length - 1; i >= 0; i--) {
      const egg = this.eggs[i];
      egg.hatchProgress += 1 / this.EGG_HATCH_TIME;
      if (egg.hatchProgress >= 1) {
        this.hatchEgg(i);
      }
    }

    // Update Ants
    // Randomize ant order to avoid bias
    const shuffledAnts = [...this.ants].sort(() => Math.random() - 0.5);
    
    for (const ant of shuffledAnts) {
      if (ant.isDead) continue;

      const decision = antDecisions.get(ant.id) || { move: 0, attack: false, dropFood: false, pheromones: [] };
      
      // Hunger decay
      ant.hunger -= this.STARVE_RATE;
      if (ant.hunger <= 0) {
        ant.hunger = 0;
        ant.health -= 1.0; // Starvation damage
      }

      if (ant.health <= 0) {
        this.killAnt(ant);
        continue;
      }

      // Pheromone
      if (decision.pheromones) {
        for (let c = 0; c < Math.min(decision.pheromones.length, AntWarfareGame.PHEROMONE_CHANNELS); c++) {
          if (decision.pheromones[c]) {
            if (ant.colony === Colony.RED) this.pheromones[ant.y][ant.x].red[c] = 1.0;
            else this.pheromones[ant.y][ant.x].black[c] = 1.0;
          }
        }
      }

      // Drop food
      if (decision.dropFood && ant.foodCarried > 0) {
        const dx = -ant.lastMoveDirection.dx;
        const dy = -ant.lastMoveDirection.dy;
        const targetX = ant.x + dx;
        const targetY = ant.y + dy;
        if (this.isValidPos(targetX, targetY) && this.grid[targetY][targetX] === TileType.DIRT) {
          this.grid[targetY][targetX] = TileType.FOOD;
          ant.foodCarried--;
        }
      }

      // Movement & Attack
      let dx = 0, dy = 0;
      if (decision.move === 1) { dy = -1; ant.directionsAttempted.up = true; } // Up
      else if (decision.move === 2) { dy = 1; ant.directionsAttempted.down = true; } // Down
      else if (decision.move === 3) { dx = -1; ant.directionsAttempted.left = true; } // Left
      else if (decision.move === 4) { dx = 1; ant.directionsAttempted.right = true; } // Right

      if (dx !== 0 || dy !== 0) {
        const targetX = ant.x + dx;
        const targetY = ant.y + dy;

        if (this.isValidPos(targetX, targetY)) {
          const targetTile = this.grid[targetY][targetX];
          
          // Attack
          if (decision.attack && (targetTile === TileType.RED_ANT || targetTile === TileType.BLACK_ANT)) {
            const otherAnt = this.ants.find(a => !a.isDead && a.x === targetX && a.y === targetY);
            if (otherAnt && otherAnt.colony !== ant.colony) {
              this.performAttack(ant, otherAnt);
              ant.hunger -= this.ATTACK_HUNGER_COST;
            }
          } 
          // Move to DIRT/FOOD/EGG
          else if (targetTile === TileType.DIRT || targetTile === TileType.FOOD || targetTile === TileType.RED_EGG || targetTile === TileType.BLACK_EGG) {
            
            // Interaction with egg
            if (targetTile === TileType.RED_EGG || targetTile === TileType.BLACK_EGG) {
              const eggIndex = this.eggs.findIndex(e => e.x === targetX && e.y === targetY);
              const egg = this.eggs[eggIndex];
              if (egg) {
                if (egg.colony === ant.colony) {
                  // Push food to hatch
                  if (ant.foodCarried > 0) {
                    this.hatchEgg(eggIndex);
                    ant.foodCarried--;
                    ant.foodReturnedHome++;
                  }
                } else {
                  // Steal enemy egg
                  if (ant.foodCarried < 5) {
                    this.grid[targetY][targetX] = TileType.DIRT;
                    this.eggs.splice(eggIndex, 1);
                    ant.foodCarried++;
                    ant.foodCollected++;
                    ant.hunger += this.FOOD_HUNGER_RESTORE / 2; // Immediate satisfaction from stealing
                  }
                }
              }
            }

            // Move
            if (this.grid[targetY][targetX] === TileType.DIRT || this.grid[targetY][targetX] === TileType.FOOD) {
                const oldX = ant.x;
                const oldY = ant.y;
                
                // Clear old pos
                this.grid[oldY][oldX] = TileType.DIRT;
                
                // Move to new pos
                ant.x = targetX;
                ant.y = targetY;
                ant.distanceTravelled++;
                ant.lastMoveDirection = { dx, dy };
                ant.hunger -= this.MOVE_HUNGER_COST;

                // Pick up/Eat food
                if (this.grid[ant.y][ant.x] === TileType.FOOD) {
                  ant.foodCollected++;
                  if (ant.hunger < AntWarfareGame.MAX_HUNGER - this.FOOD_HUNGER_RESTORE) {
                    ant.hunger = Math.min(AntWarfareGame.MAX_HUNGER, ant.hunger + this.FOOD_HUNGER_RESTORE);
                  } else if (ant.foodCarried < 5) {
                    ant.foodCarried++;
                  }
                }

                this.grid[ant.y][ant.x] = ant.colony === Colony.RED ? TileType.RED_ANT : TileType.BLACK_ANT;

                // Sharing
                this.shareFood(ant);
            }
          }
          // Push Rock
          else if (targetTile === TileType.ROCK) {
            const pushX = targetX + dx;
            const pushY = targetY + dy;
              if (this.isValidPos(pushX, pushY) && this.grid[pushY][pushX] === TileType.DIRT) {
                // Successful push
                this.grid[pushY][pushX] = TileType.ROCK;
                this.grid[targetY][targetX] = TileType.DIRT;
                const rock = this.rocks.get(`${targetX},${targetY}`);
                if (rock) {
                    this.rocks.delete(`${targetX},${targetY}`);
                    rock.x = pushX;
                    rock.y = pushY;
                    this.rocks.set(`${pushX},${pushY}`, rock);
                }
                ant.hunger -= this.PUSH_HUNGER_COST;
              } else {
                // Damage rock (either blocked by non-DIRT or out of bounds)
                const rock = this.rocks.get(`${targetX},${targetY}`);
                if (rock) {
                    rock.damage++;
                    if (rock.damage >= 3) {
                        this.grid[targetY][targetX] = TileType.DIRT;
                        this.rocks.delete(`${targetX},${targetY}`);
                    }
                }
                ant.hunger -= this.PUSH_HUNGER_COST;
            }
          }
          // Bring food to Queen
          else if ((ant.colony === Colony.RED && targetTile === TileType.RED_QUEEN) || 
                   (ant.colony === Colony.BLACK && targetTile === TileType.BLACK_QUEEN)) {
            if (ant.foodCarried > 0) {
                // Bring food logic - queen lays eggs
                // Every 3 food = 1 egg
                // For simplicity, we can track "queen food" or just use a counter
                this.feedQueen(ant.colony, ant.foodCarried);
                ant.foodReturnedHome += ant.foodCarried;
                ant.foodCarried = 0;
            }
          }
        } else {
          // Out of bounds attempt
          ant.outOfBoundsAttempts++;
          ant.hunger -= this.MOVE_HUNGER_COST;
        }
      } else {
        ant.hunger -= this.STAY_HUNGER_COST;
      }

      // Auto-eat from inventory
      if (ant.hunger < 20 && ant.foodCarried > 0) {
        ant.hunger = Math.min(AntWarfareGame.MAX_HUNGER, ant.hunger + this.FOOD_HUNGER_RESTORE);
        ant.foodCarried--;
      }
    }

    // Queen auto-egg if colony is empty
    this.checkQueenAutoEgg(Colony.RED);
    this.checkQueenAutoEgg(Colony.BLACK);
  }

  private isValidPos(x: number, y: number) {
    return x >= 0 && x < this.width && y >= 0 && y < this.height;
  }

  private performAttack(attacker: Ant, target: Ant) {
    // Damage proportional to how full the attacking ant is
    const damage = (attacker.hunger / AntWarfareGame.MAX_HUNGER) * 20 + 5;
    
    // Defense buff near queen
    const qpos = this.QUEEN_POSITIONS[target.colony];
    const distToQueen = Math.sqrt((target.x - qpos.x)**2 + (target.y - qpos.y)**2);
    const defenseBuff = Math.max(0, 1 - distToQueen / 10) * 0.2; // 20% buff at queen, fades over 10 tiles

    const finalDamage = damage * (1 - defenseBuff);
    target.health -= finalDamage;
    attacker.totalDamageDealt += finalDamage;

    if (target.health <= 0) {
      attacker.kills++;
      this.killAnt(target);
    }
  }

  private killAnt(ant: Ant) {
    ant.isDead = true;
    this.grid[ant.y][ant.x] = TileType.DIRT;
    
    // Drop food around
    let dropped = 0;
    for (let dy = -1; dy <= 1 && dropped < ant.foodCarried; dy++) {
      for (let dx = -1; dx <= 1 && dropped < ant.foodCarried; dx++) {
        const nx = ant.x + dx;
        const ny = ant.y + dy;
        if (this.isValidPos(nx, ny) && this.grid[ny][nx] === TileType.DIRT) {
          this.grid[ny][nx] = TileType.FOOD;
          dropped++;
        }
      }
    }

    // Corpse chance to turn into rock
    if (Math.random() < 0.3) {
      if (this.grid[ant.y][ant.x] === TileType.DIRT) {
        this.grid[ant.y][ant.x] = TileType.ROCK;
        this.rocks.set(`${ant.x},${ant.y}`, { x: ant.x, y: ant.y, damage: 0, seed: Math.random() });
      }
    }
  }

  private shareFood(ant: Ant) {
      const friendly = this.ants.find(a => !a.isDead && a !== ant && a.x === ant.x && a.y === ant.y && a.colony === ant.colony);
      if (friendly) {
          const totalFood = ant.foodCarried + friendly.foodCarried;
          ant.foodCarried = Math.floor(totalFood / 2);
          friendly.foodCarried = totalFood - ant.foodCarried;

          // Immediate eating if hungry
          if (ant.hunger < 50 && ant.foodCarried > 0) {
              ant.hunger = Math.min(AntWarfareGame.MAX_HUNGER, ant.hunger + this.FOOD_HUNGER_RESTORE);
              ant.foodCarried--;
          }
          if (friendly.hunger < 50 && friendly.foodCarried > 0) {
              friendly.hunger = Math.min(AntWarfareGame.MAX_HUNGER, friendly.hunger + this.FOOD_HUNGER_RESTORE);
              friendly.foodCarried--;
          }
      }
  }

  private queenFoodStorage = { [Colony.RED]: 0, [Colony.BLACK]: 0 };

  private feedQueen(colony: Colony, amount: number) {
    this.queenFoodStorage[colony] += amount;
    const livingCount = this.ants.filter(a => a.colony === colony && !a.isDead).length;
    const eggCount = this.eggs.filter(e => e.colony === colony).length;
    
    while (this.queenFoodStorage[colony] >= 3 && (livingCount + eggCount) < this.MAX_ANTS_PER_COLONY) {
        this.queenFoodStorage[colony] -= 3;
        this.spawnEgg(colony);
    }
  }

  private spawnEgg(colony: Colony) {
    const qpos = this.QUEEN_POSITIONS[colony];
    const type = colony === Colony.RED ? TileType.RED_EGG : TileType.BLACK_EGG;
    
    // Find space around queen (2x2)
    for (let dy = -1; dy <= 2; dy++) {
      for (let dx = -1; dx <= 2; dx++) {
        const nx = qpos.x + dx;
        const ny = qpos.y + dy;
        if (this.isValidPos(nx, ny)) {
            if (this.grid[ny][nx] === TileType.DIRT || this.grid[ny][nx] === TileType.ROCK) {
                const isRock = this.grid[ny][nx] === TileType.ROCK;
                this.grid[ny][nx] = type;
                this.eggs.push({ colony, x: nx, y: ny, hatchProgress: 0, bornOnRock: isRock });
                if (isRock) this.rocks.delete(`${nx},${ny}`);
                return;
            }
        }
      }
    }
  }

  private checkQueenAutoEgg(colony: Colony) {
    const livingCount = this.ants.filter(a => a.colony === colony && !a.isDead).length;
    const eggCount = this.eggs.filter(e => e.colony === colony).length;
    if (livingCount === 0 && eggCount === 0) {
      this.spawnInitialEggs(colony);
    }
  }

  private hatchEgg(index: number) {
    const egg = this.eggs[index];
    this.eggs.splice(index, 1);
    
    const directions = [
      { dx: 0, dy: -1 },
      { dx: 0, dy: 1 },
      { dx: -1, dy: 0 },
      { dx: 1, dy: 0 }
    ];
    const randomDir = directions[Math.floor(Math.random() * directions.length)];

    const ant: Ant = {
      id: Math.random(), // Unique ID for decision mapping
      colony: egg.colony,
      x: egg.x,
      y: egg.y,
      health: AntWarfareGame.MAX_HEALTH,
      hunger: AntWarfareGame.MAX_HUNGER,
      foodCarried: 0,
      isDead: false,
      kills: 0,
      totalDamageDealt: 0,
      foodCollected: 0,
      foodReturnedHome: 0,
      distanceTravelled: 0,
      outOfBoundsAttempts: 0,
      directionsAttempted: { up: false, down: false, left: false, right: false },
      lastMoveDirection: randomDir,
      bornOnRock: egg.bornOnRock
    };

    // If there was an ant already there, move them out or replace?
    // Rules say "Ants hatching from eggs will assume intelligence..."
    // We'll handle intelligence/mutation in the React component by matching IDs or using a callback
    this.ants.push(ant);
    this.grid[ant.y][ant.x] = ant.colony === Colony.RED ? TileType.RED_ANT : TileType.BLACK_ANT;
    
    if (this.onAntHatched) {
        this.onAntHatched(ant);
    }
  }

  public onAntHatched?: (ant: Ant) => void;
  
  public getFitnessScore(ant: Ant): number {
    let score = (ant.kills * 100) + (ant.foodReturnedHome * 100) + (ant.foodCollected * 20);
    
    // Only penalize out-of-bounds attempts if no food has been collected yet.
    // This allows for early exploration without immediate discouragement.
    if (ant.foodCollected === 0) {
      score -= (ant.outOfBoundsAttempts * 2);
    }
    
    // Massive penalty for never moving
    if (ant.distanceTravelled === 0) {
      score -= 1000;
    }
    
    // Directional penalty (exploration incentive)
    const unattempted = [
      ant.directionsAttempted.up,
      ant.directionsAttempted.down,
      ant.directionsAttempted.left,
      ant.directionsAttempted.right
    ].filter(attempted => !attempted).length;
    
    score -= (unattempted * 50);
    return score;
  }

  public getDistToQueen(ant: Ant): number {
    const qpos = this.QUEEN_POSITIONS[ant.colony];
    return Math.sqrt((ant.x - qpos.x) ** 2 + (ant.y - qpos.y) ** 2);
  }

  public getAntState(ant: Ant) {
    const INPUT_H = 7;
    const INPUT_W = 7;
    const INPUT_C = 15;
    const inputs = new Float32Array(INPUT_H * INPUT_W * INPUT_C);
    
    // Shared scalar values
    const health = ant.health / AntWarfareGame.MAX_HEALTH;
    const hunger = ant.hunger / AntWarfareGame.MAX_HUNGER;
    const foodCarried = ant.foodCarried / 5;
    
    const qpos = this.QUEEN_POSITIONS[ant.colony];
    const qdx = qpos.x - ant.x;
    const qdy = qpos.y - ant.y;
    const distToQueen = Math.min(1.0, Math.sqrt(qdx*qdx + qdy*qdy) / 50);

    // Fill 7x7 grid
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const nx = ant.x + dx;
        const ny = ant.y + dy;
        const gridX = dx + 3;
        const gridY = dy + 3;
        const baseIdx = (gridY * INPUT_W + gridX) * INPUT_C;
        
        if (this.isValidPos(nx, ny)) {
          const type = this.grid[ny][nx];
          
          // 9 Features (0-8)
          if (type === TileType.FOOD) inputs[baseIdx + 0] = 1;
          else if (type === TileType.ROCK) inputs[baseIdx + 1] = 1;
          else if (type === TileType.RED_ANT || type === TileType.BLACK_ANT) {
            const isFriendly = (type === TileType.RED_ANT && ant.colony === Colony.RED) || 
                               (type === TileType.BLACK_ANT && ant.colony === Colony.BLACK);
            inputs[baseIdx + (isFriendly ? 2 : 3)] = 1;
          } else if (type === TileType.RED_QUEEN || type === TileType.BLACK_QUEEN) {
            const isFriendly = (type === TileType.RED_QUEEN && ant.colony === Colony.RED) || 
                               (type === TileType.BLACK_QUEEN && ant.colony === Colony.BLACK);
            inputs[baseIdx + (isFriendly ? 4 : 5)] = 1;
          } else if (type === TileType.RED_EGG || type === TileType.BLACK_EGG) {
            const isFriendly = (type === TileType.RED_EGG && ant.colony === Colony.RED) || 
                               (type === TileType.BLACK_EGG && ant.colony === Colony.BLACK);
            inputs[baseIdx + (isFriendly ? 6 : 7)] = 1;
          } else {
            inputs[baseIdx + 8] = 1; // Dirt
          }
          
          // 2 Pheromones (9-10)
          const p = this.pheromones[ny][nx];
          const myPhero = ant.colony === Colony.RED ? p.red : p.black;
          inputs[baseIdx + 9] = myPhero[0];
          inputs[baseIdx + 10] = myPhero[1];
        } else {
          // Out of bounds - treat as rock
          inputs[baseIdx + 1] = 1;
        }

        // 4 Tiled Scalars (11-14)
        inputs[baseIdx + 11] = health;
        inputs[baseIdx + 12] = hunger;
        inputs[baseIdx + 13] = foodCarried;
        inputs[baseIdx + 14] = distToQueen;
      }
    }

    return inputs;
  }

}
