interface Bird {
	y: number;
	velocity: number;
	isDead: boolean;
	score: number;
}

interface Pipe {
	x: number;
	gapY: number;
	passed: boolean;
}

export default class FlappyBirdGame {
	private canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private birds: Bird[];
	private pipes: Pipe[];
	private frameCount: number;
	
	private readonly CANVAS_WIDTH = 800;
	private readonly CANVAS_HEIGHT = 600;
	private readonly BIRD_SIZE = 20;
	private readonly PIPE_WIDTH = 60;
	private readonly PIPE_GAP = 150;
	private readonly GRAVITY = 0.5;
	private readonly FLAP_STRENGTH = -8;
	private readonly PIPE_SPEED = 3;
	private readonly PIPE_SPACING = 250;

	private telemetryDiv: HTMLDivElement;

	constructor(private populationSize: number) {
		this.setupUI();
		this.birds = [];
		this.pipes = [];
		this.frameCount = 0;
	}

	private setupUI() {
		// Add styles
		const style = document.createElement('style');
		style.textContent = `
			body {
				margin: 0;
				padding: 20px;
				font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
				background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
				display: flex;
				flex-direction: column;
				align-items: center;
				min-height: 100vh;
			}
			canvas {
				border: 4px solid #2d3748;
				border-radius: 8px;
				box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
				background: #87CEEB;
			}
			.telemetry {
				background: white;
				border-radius: 8px;
				padding: 20px;
				margin-top: 20px;
				box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
				width: 800px;
				box-sizing: border-box;
			}
			.telemetry h2 {
				margin: 0 0 15px 0;
				color: #2d3748;
				font-size: 24px;
			}
			.stat-grid {
				display: grid;
				grid-template-columns: repeat(3, 1fr);
				gap: 15px;
			}
			.stat {
				background: #f7fafc;
				padding: 15px;
				border-radius: 6px;
				border-left: 4px solid #667eea;
			}
			.stat-label {
				font-size: 12px;
				color: #718096;
				font-weight: 600;
				text-transform: uppercase;
				letter-spacing: 0.5px;
				margin-bottom: 5px;
			}
			.stat-value {
				font-size: 28px;
				color: #2d3748;
				font-weight: 700;
			}
		`;
		document.head.appendChild(style);

		// Create canvas
		this.canvas = document.createElement('canvas');
		this.canvas.width = this.CANVAS_WIDTH;
		this.canvas.height = this.CANVAS_HEIGHT;
		this.ctx = this.canvas.getContext('2d')!;
		document.body.appendChild(this.canvas);

		// Create telemetry panel
		this.telemetryDiv = document.createElement('div');
		this.telemetryDiv.className = 'telemetry';
		document.body.appendChild(this.telemetryDiv);
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
		// Fitness = score * 100 + frames survived
		return this.birds[birdIndex].score * 100 + this.frameCount;
	}

	render(generation: number, population: any[], bestFitnessEver: number) {
		const ctx = this.ctx;

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
		let aliveCount = 0;
		let bestScore = 0;
		for (let i = 0; i < this.birds.length; i++) {
			const bird = this.birds[i];
			if (!bird.isDead) {
				aliveCount++;
				bestScore = Math.max(bestScore, bird.score);
				
				// Draw alive birds (semi-transparent yellow)
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

		// Draw best bird in the current generation (bright yellow)
		const bestCurrentBird = this.birds.reduce((best, bird, idx) => 
			!bird.isDead && bird.score >= this.birds[best].score ? idx : best
		, 0);
		
		if (!this.birds[bestCurrentBird].isDead) {
			const bird = this.birds[bestCurrentBird];
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

		// Update telemetry
		const avgFitness = population.reduce((sum, g) => sum + g.fitness, 0) / population.length;
		const bestCurrentFitness = Math.max(...population.map(g => g.fitness));
		
		this.telemetryDiv.innerHTML = `
			<h2>🧬 Genetic Algorithm Training</h2>
			<div class="stat-grid">
				<div class="stat">
					<div class="stat-label">Generation</div>
					<div class="stat-value">${generation}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Alive</div>
					<div class="stat-value">${aliveCount}/${this.populationSize}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Score</div>
					<div class="stat-value">${bestScore}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Fitness (Gen)</div>
					<div class="stat-value">${bestCurrentFitness.toFixed(0)}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Best Fitness (Ever)</div>
					<div class="stat-value">${bestFitnessEver.toFixed(0)}</div>
				</div>
				<div class="stat">
					<div class="stat-label">Avg Fitness</div>
					<div class="stat-value">${avgFitness.toFixed(0)}</div>
				</div>
			</div>
		`;
	}
}