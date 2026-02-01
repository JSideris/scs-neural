export interface Bird {
  y: number;
  velocity: number;
  isDead: boolean;
  score: number;
}

export interface Pipe {
  x: number;
  gapY: number;
  passed: boolean;
}

export class FlappyBirdGame {
  public birds: Bird[];
  public pipes: Pipe[];
  public frameCount: number;

  public readonly CANVAS_WIDTH = 800;
  public readonly CANVAS_HEIGHT = 600;
  private readonly BIRD_SIZE = 20;
  private readonly PIPE_WIDTH = 60;
  private readonly PIPE_GAP = 150;
  private readonly GRAVITY = 0.5;
  private readonly FLAP_STRENGTH = -8;
  private readonly PIPE_SPEED = 3;
  private readonly PIPE_SPACING = 250;

  constructor(private populationSize: number) {
    this.birds = [];
    this.pipes = [];
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.birds = [];
    for (let i = 0; i < this.populationSize; i++) {
      this.birds.push({
        y: this.CANVAS_HEIGHT / 2,
        velocity: 0,
        isDead: false,
        score: 0,
      });
    }

    this.pipes = [];
    for (let i = 0; i < 4; i++) {
      this.pipes.push({
        x: this.CANVAS_WIDTH + i * this.PIPE_SPACING,
        gapY: Math.random() * (this.CANVAS_HEIGHT - this.PIPE_GAP - 100) + 50,
        passed: false,
      });
    }

    this.frameCount = 0;
  }

  flap(birdIndex: number) {
    if (!this.birds[birdIndex].isDead) {
      this.birds[birdIndex].velocity = this.FLAP_STRENGTH;
    }
  }

  update() {
    this.frameCount++;

    // Update pipes
    for (const pipe of this.pipes) {
      pipe.x -= this.PIPE_SPEED;
    }

    // Add new pipe if needed
    const lastPipe = this.pipes[this.pipes.length - 1];
    if (lastPipe.x < this.CANVAS_WIDTH - this.PIPE_SPACING) {
      this.pipes.push({
        x: this.CANVAS_WIDTH,
        gapY: Math.random() * (this.CANVAS_HEIGHT - this.PIPE_GAP - 100) + 50,
        passed: false,
      });
    }

    // Remove off-screen pipes
    this.pipes = this.pipes.filter(pipe => pipe.x > -this.PIPE_WIDTH);

    // Update birds
    for (let i = 0; i < this.birds.length; i++) {
      const bird = this.birds[i];
      if (bird.isDead) continue;

      // Apply gravity
      bird.velocity += this.GRAVITY;
      bird.y += bird.velocity;

      // Check collisions
      if (bird.y < 0 || bird.y > this.CANVAS_HEIGHT - this.BIRD_SIZE) {
        bird.isDead = true;
        continue;
      }

      // Check pipe collisions
      for (const pipe of this.pipes) {
        if (
          pipe.x < this.CANVAS_WIDTH / 4 + this.BIRD_SIZE &&
          pipe.x + this.PIPE_WIDTH > this.CANVAS_WIDTH / 4
        ) {
          if (bird.y < pipe.gapY || bird.y + this.BIRD_SIZE > pipe.gapY + this.PIPE_GAP) {
            bird.isDead = true;
            break;
          }

          // Score point
          if (!pipe.passed) {
            bird.score++;
            pipe.passed = true;
          }
        }
      }
    }
  }

  getBirdState(birdIndex: number) {
    const bird = this.birds[birdIndex];
    
    // Find next pipe
    let nextPipe = this.pipes.find(p => p.x + this.PIPE_WIDTH > this.CANVAS_WIDTH / 4);
    if (!nextPipe) nextPipe = this.pipes[0];

    return {
      birdY: bird.y,
      birdVelocity: bird.velocity,
      nextPipeX: nextPipe.x - this.CANVAS_WIDTH / 4,
      pipeTopY: nextPipe.gapY,
      pipeBottomY: nextPipe.gapY + this.PIPE_GAP,
    };
  }

  isDead(birdIndex: number): boolean {
    return this.birds[birdIndex].isDead;
  }

  getScore(birdIndex: number): number {
    return this.birds[birdIndex].score * 100 + this.frameCount;
  }

  render(ctx: CanvasRenderingContext2D) {
    // Clear canvas with sky gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, this.CANVAS_HEIGHT);
    gradient.addColorStop(0, '#87CEEB');
    gradient.addColorStop(1, '#E0F6FF');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_HEIGHT);

    // Draw pipes
    ctx.fillStyle = '#2ECC40';
    ctx.strokeStyle = '#01FF70';
    ctx.lineWidth = 3;
    
    for (const pipe of this.pipes) {
      // Top pipe
      ctx.fillRect(pipe.x, 0, this.PIPE_WIDTH, pipe.gapY);
      ctx.strokeRect(pipe.x, 0, this.PIPE_WIDTH, pipe.gapY);
      
      // Bottom pipe
      ctx.fillRect(
        pipe.x,
        pipe.gapY + this.PIPE_GAP,
        this.PIPE_WIDTH,
        this.CANVAS_HEIGHT - pipe.gapY - this.PIPE_GAP
      );
      ctx.strokeRect(
        pipe.x,
        pipe.gapY + this.PIPE_GAP,
        this.PIPE_WIDTH,
        this.CANVAS_HEIGHT - pipe.gapY - this.PIPE_GAP
      );
    }

    // Draw birds
    for (let i = 0; i < this.birds.length; i++) {
      const bird = this.birds[i];
      if (!bird.isDead) {
        ctx.fillStyle = 'rgba(255, 220, 0, 0.3)';
        ctx.beginPath();
        ctx.arc(
          this.CANVAS_WIDTH / 4,
          bird.y + this.BIRD_SIZE / 2,
          this.BIRD_SIZE / 2,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }

    // Draw best bird
    const bestCurrentBirdIdx = this.birds.reduce((best, bird, idx) => 
      !bird.isDead && (this.birds[best].isDead || bird.score >= this.birds[best].score) ? idx : best
    , 0);
    
    if (!this.birds[bestCurrentBirdIdx].isDead) {
      const bird = this.birds[bestCurrentBirdIdx];
      ctx.fillStyle = '#FFD700';
      ctx.strokeStyle = '#FFA500';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(
        this.CANVAS_WIDTH / 4,
        bird.y + this.BIRD_SIZE / 2,
        this.BIRD_SIZE / 2,
        0,
        Math.PI * 2
      );
      ctx.fill();
      ctx.stroke();

      // Draw eye
      ctx.fillStyle = 'black';
      ctx.beginPath();
      ctx.arc(
        this.CANVAS_WIDTH / 4 + 5,
        bird.y + this.BIRD_SIZE / 2 - 3,
        2,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }
}
