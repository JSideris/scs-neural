import { AntWarfareGame, Ant, Rock, Colony, TileType } from './game';

export class GameRenderer {
  public static readonly TILE_SIZE = 10;

  private drawRock(ctx: CanvasRenderingContext2D, rock: Rock, px: number, py: number, tileSize: number) {
    const seed = rock.seed;
    const cx = px + tileSize / 2;
    const cy = py + tileSize / 2;

    // Deterministic random function based on seed
    const pseudoRandom = (offset: number) => {
      const x = Math.sin(seed * 1000 + offset) * 10000;
      return x - Math.floor(x);
    };

    // Damage-based coloring
    const baseColor = rock.damage === 0 ? '#9e9e9e' : rock.damage === 1 ? '#757575' : '#424242';
    const highlightColor = rock.damage === 0 ? '#bdbdbd' : rock.damage === 1 ? '#9e9e9e' : '#616161';
    const shadowColor = rock.damage === 0 ? '#757575' : rock.damage === 1 ? '#616161' : '#212121';

    // Generate vertices for irregular polygon
    const numVertices = 5 + Math.floor(pseudoRandom(1) * 3); // 5-7 vertices
    const vertices: { x: number, y: number }[] = [];
    const radius = tileSize * 0.4;

    for (let i = 0; i < numVertices; i++) {
      const angle = (i / numVertices) * Math.PI * 2;
      const dist = radius * (0.8 + pseudoRandom(i + 2) * 0.4);
      vertices.push({
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(angle) * dist
      });
    }

    // Draw shadow
    ctx.fillStyle = shadowColor;
    ctx.beginPath();
    ctx.moveTo(vertices[0].x + 1, vertices[0].y + 1);
    for (let i = 1; i < vertices.length; i++) {
      ctx.lineTo(vertices[i].x + 1, vertices[i].y + 1);
    }
    ctx.closePath();
    ctx.fill();

    // Draw body
    ctx.fillStyle = baseColor;
    ctx.beginPath();
    ctx.moveTo(vertices[0].x, vertices[0].y);
    for (let i = 1; i < vertices.length; i++) {
      ctx.lineTo(vertices[i].x, vertices[i].y);
    }
    ctx.closePath();
    ctx.fill();

    // Draw highlights/craters for character
    ctx.fillStyle = highlightColor;
    for (let i = 0; i < 2; i++) {
      const hx = cx + (pseudoRandom(i + 10) - 0.5) * radius;
      const hy = cy + (pseudoRandom(i + 20) - 0.5) * radius;
      const hr = 1 + pseudoRandom(i + 30) * 1.5;
      ctx.beginPath();
      ctx.arc(hx, hy, hr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private drawFood(ctx: CanvasRenderingContext2D, x: number, y: number, px: number, py: number, tileSize: number) {
    const seed = ((x * 123 + y * 456) % 1000) / 1000;
    const cx = px + tileSize / 2;
    const cy = py + tileSize / 2;

    const pseudoRandom = (offset: number) => {
      const s = Math.sin(seed * 1000 + offset) * 10000;
      return s - Math.floor(s);
    };

    // Tasty food colors: range of greens, limes, and yellows
    const hue = 60 + pseudoRandom(1) * 60; // 60 (yellow) to 120 (green)
    const baseColor = `hsl(${hue}, 80%, 45%)`;
    const highlightColor = `hsl(${hue}, 90%, 70%)`;
    const shadowColor = `hsl(${hue}, 70%, 25%)`;

    const radius = tileSize * 0.3;

    // Draw shadow
    ctx.fillStyle = shadowColor;
    ctx.beginPath();
    ctx.arc(cx + 0.5, cy + 0.5, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw body
    ctx.fillStyle = baseColor;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw highlight glint
    ctx.fillStyle = highlightColor;
    ctx.beginPath();
    ctx.arc(cx - radius * 0.3, cy - radius * 0.3, radius * 0.3, 0, Math.PI * 2);
    ctx.fill();
  }

  private drawDirt(ctx: CanvasRenderingContext2D, x: number, y: number, px: number, py: number, tileSize: number) {
    const seed = ((x * 31 + y * 17) % 1000) / 1000;
    
    const pseudoRandom = (offset: number) => {
      const s = Math.sin(seed * 1000 + offset) * 10000;
      return s - Math.floor(s);
    };

    // Base dirt color
    ctx.fillStyle = '#3e2723';
    ctx.fillRect(px, py, tileSize, tileSize);

    // Add some grains for texture
    const numGrains = 2 + Math.floor(pseudoRandom(1) * 2); // 2-3 grains
    for (let i = 0; i < numGrains; i++) {
      const gx = px + pseudoRandom(i + 2) * (tileSize - 1);
      const gy = py + pseudoRandom(i + 5) * (tileSize - 1);
      const gw = 1 + Math.floor(pseudoRandom(i + 8) * 1.5); // 1-2px wide
      const gh = 1;
      
      // Alternate between highlight and shadow grains
      ctx.fillStyle = i % 2 === 0 ? '#4e342e' : '#2e1b17';
      ctx.fillRect(gx, gy, gw, gh);
    }
  }

  private drawAnt(ctx: CanvasRenderingContext2D, ant: Ant, px: number, py: number, tileSize: number) {
    const cx = px + tileSize / 2;
    const cy = py + tileSize / 2;
    
    // Determine rotation angle based on lastMoveDirection
    let angle = 0;
    if (ant.lastMoveDirection.dx === 1) angle = 0;
    else if (ant.lastMoveDirection.dx === -1) angle = Math.PI;
    else if (ant.lastMoveDirection.dy === 1) angle = Math.PI / 2;
    else if (ant.lastMoveDirection.dy === -1) angle = -Math.PI / 2;
    
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    
    // Scale for grown-up ants
    if (ant.isGrownUp) {
      ctx.scale(1.2, 1.2);
    }
    
    // Body color based on health/hunger
    const saturation = Math.floor((ant.hunger / AntWarfareGame.MAX_HUNGER) * 100);
    const lightness = Math.floor((ant.health / ant.maxHealth) * 40) + 10;
    const hue = ant.colony === Colony.RED ? 0 : 240;
    const bodyColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    
    ctx.fillStyle = bodyColor;
    ctx.strokeStyle = bodyColor;
    ctx.lineWidth = 1;

    // 1. Abdomen (back part)
    ctx.beginPath();
    ctx.ellipse(-2, 0, 3, 2, 0, 0, Math.PI * 2);
    ctx.fill();

    // 2. Thorax (middle part)
    ctx.beginPath();
    ctx.ellipse(1, 0, 1.5, 1.2, 0, 0, Math.PI * 2);
    ctx.fill();

    // 3. Head (front part)
    ctx.beginPath();
    ctx.arc(3.5, 0, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // 4. Mandibles
    ctx.beginPath();
    ctx.moveTo(4.5, -0.5);
    ctx.lineTo(5, -1.5);
    ctx.moveTo(4.5, 0.5);
    ctx.lineTo(5, 1.5);
    ctx.stroke();

    // 5. Legs
    // Front legs
    ctx.beginPath();
    ctx.moveTo(1, -1); ctx.lineTo(2, -2.5);
    ctx.moveTo(1, 1); ctx.lineTo(2, 2.5);
    // Middle legs
    ctx.moveTo(0.5, -1); ctx.lineTo(0, -2.5);
    ctx.moveTo(0.5, 1); ctx.lineTo(0, 2.5);
    // Back legs
    ctx.moveTo(0, -1); ctx.lineTo(-1, -2.5);
    ctx.moveTo(0, 1); ctx.lineTo(-1, 2.5);
    ctx.stroke();

    // 6. Food Carried
    if (ant.foodCarried > 0) {
        const foodHue = 60 + (ant.id % 0.5) * 120; // Some variation based on ant id
        const foodBaseColor = `hsl(${foodHue}, 80%, 45%)`;
        const foodHighlightColor = `hsl(${foodHue}, 90%, 70%)`;
        
        ctx.fillStyle = foodBaseColor;
        ctx.beginPath();
        ctx.arc(5, 0, 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Small highlight on carried food
        ctx.fillStyle = foodHighlightColor;
        ctx.beginPath();
        ctx.arc(4.5, -0.5, 0.5, 0, Math.PI * 2);
        ctx.fill();

        // If carrying multiple food items, show a stack or bigger cluster
        if (ant.foodCarried > 1) {
            ctx.fillStyle = foodBaseColor;
            ctx.beginPath();
            ctx.arc(6.5, 0, 1.2, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    ctx.restore();
  }

  public render(ctx: CanvasRenderingContext2D, game: AntWarfareGame) {
    const tileSize = GameRenderer.TILE_SIZE;
    ctx.clearRect(0, 0, game.width * tileSize, game.height * tileSize);

    for (let y = 0; y < game.height; y++) {
      for (let x = 0; x < game.width; x++) {
        const type = game.grid[y][x];
        const px = x * tileSize;
        const py = y * tileSize;

        switch (type) {
          case TileType.DIRT:
            this.drawDirt(ctx, x, y, px, py, tileSize);
            break;
          case TileType.FOOD:
            this.drawFood(ctx, x, y, px, py, tileSize);
            break;
          case TileType.ROCK:
            const rock = game.rocks.get(`${x},${y}`) || { x, y, damage: 0, seed: (x * 7 + y * 13) % 100 / 100 };
            this.drawRock(ctx, rock, px, py, tileSize);
            break;
          case TileType.RED_QUEEN:
            ctx.fillStyle = '#b71c1c';
            ctx.fillRect(px, py, tileSize, tileSize);
            break;
          case TileType.BLACK_QUEEN:
            ctx.fillStyle = '#212121';
            ctx.fillRect(px, py, tileSize, tileSize);
            break;
          case TileType.RED_EGG:
            ctx.fillStyle = '#ffcdd2';
            ctx.beginPath();
            ctx.arc(px + tileSize/2, py + tileSize/2, tileSize/3, 0, Math.PI * 2);
            ctx.fill();
            break;
          case TileType.BLACK_EGG:
            ctx.fillStyle = '#f5f5f5';
            ctx.beginPath();
            ctx.arc(px + tileSize/2, py + tileSize/2, tileSize/3, 0, Math.PI * 2);
            ctx.fill();
            break;
        }

        // Render Pheromones
        const p = game.pheromones[y][x];
        for (let c = 0; c < AntWarfareGame.PHEROMONE_CHANNELS; c++) {
          if (p.red[c] > 0.1) {
            ctx.fillStyle = c === 0 ? `rgba(255, 0, 0, ${p.red[c] * 0.2})` : `rgba(255, 165, 0, ${p.red[c] * 0.2})`;
            ctx.fillRect(px, py, tileSize, tileSize);
          }
          if (p.black[c] > 0.1) {
            ctx.fillStyle = c === 0 ? `rgba(255, 255, 255, ${p.black[c] * 0.2})` : `rgba(128, 0, 128, ${p.black[c] * 0.2})`;
            ctx.fillRect(px, py, tileSize, tileSize);
          }
        }
      }
    }

    // Render Ants
    for (const ant of game.ants) {
        if (ant.isDead) continue;
        const px = ant.x * tileSize;
        const py = ant.y * tileSize;
        this.drawAnt(ctx, ant, px, py, tileSize);
    }
  }
}
