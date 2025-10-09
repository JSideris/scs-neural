import { Chart, ChartConfiguration, registerables } from 'chart.js';

Chart.register(...registerables);

interface TrainingVisualizerOptions {
  containerId?: string;
  title?: string;
  maxDataPoints?: number;
}

export default class TrainingVisualizer {
  private chart: Chart | null = null;
  private container: HTMLElement | null = null;
  private statsContainer: HTMLElement | null = null;
  
  private epochs: number[] = [];
  private losses: number[] = [];
  private maxDataPoints: number;
  private title: string;
  
  private startTime: number = 0;
  private updateCount: number = 0;

  constructor(options: TrainingVisualizerOptions = {}) {
    this.title = options.title || 'Neural Network Training';
    this.maxDataPoints = options.maxDataPoints || 1000;
  }

  /**
   * Initialize the UI
   */
  initialize(containerId: string = 'training-visualizer'): void {
    // Create or get container
    this.container = document.getElementById(containerId);
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = containerId;
      document.body.appendChild(this.container);
    }

	// dark bg:
	document.body.style.backgroundColor = '#1e1e1e';

    // Style the container
    this.container.style.cssText = `
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      max-width: 1200px;
      margin: 20px auto;
      padding: 20px;
      background: #1e1e1e;
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    `;

    // Create title
    const titleEl = document.createElement('h2');
    titleEl.textContent = this.title;
    titleEl.style.cssText = `
      color: #ffffff;
      margin: 0 0 20px 0;
      font-size: 24px;
      font-weight: 600;
    `;
    this.container.appendChild(titleEl);

    // Create canvas for chart
    const canvas = document.createElement('canvas');
    canvas.id = 'loss-chart';
    canvas.style.cssText = `
      max-height: 400px;
      background: #2d2d2d;
      border-radius: 4px;
    `;
    this.container.appendChild(canvas);

    // Create stats container
    this.statsContainer = document.createElement('div');
    this.statsContainer.style.cssText = `
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin-top: 20px;
      color: #ffffff;
    `;
    this.container.appendChild(this.statsContainer);

    // Initialize chart
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Training Loss',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        animation: false,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Epoch',
              color: '#ffffff'
            },
            ticks: {
              color: '#aaaaaa',
              maxTicksLimit: 10
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.1)'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Loss',
              color: '#ffffff'
            },
            ticks: {
              color: '#aaaaaa'
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.1)'
            }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: '#ffffff'
            }
          }
        }
      }
    };

    this.chart = new Chart(ctx, config);
    this.startTime = Date.now();
    this.updateStats();
  }

  /**
   * Update the visualization with new training data
   */
  update(epoch: number, loss: number): void {
    if (!this.chart) {
      throw new Error('Visualizer not initialized. Call initialize() first.');
    }

    // Add data point
    this.epochs.push(epoch + 1);
    this.losses.push(loss);

    // Keep only the last maxDataPoints
    if (this.epochs.length > this.maxDataPoints) {
      this.epochs.shift();
      this.losses.shift();
    }

    // Update the chart
    this.chart.data.labels = this.epochs;
    this.chart.data.datasets[0].data = this.losses;
    this.chart.update('none'); // Update without animation for better performance

    // Update statistics
    this.updateCount++;
    this.updateStats();
  }

  /**
   * Update the statistics display
   */
  private updateStats(): void {
    if (!this.statsContainer) return;

    const currentTime = Date.now();
    const elapsedSeconds = (currentTime - this.startTime) / 1000;
    
    const stats = this.getStats();
    const updatesPerSecond = this.updateCount / elapsedSeconds;
    
    // Format elapsed time
    const hours = Math.floor(elapsedSeconds / 3600);
    const minutes = Math.floor((elapsedSeconds % 3600) / 60);
    const seconds = Math.floor(elapsedSeconds % 60);
    const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

    this.statsContainer.innerHTML = `
      ${this.createStatCard('Current Epoch', stats.totalEpochs.toString())}
      ${this.createStatCard('Current Loss', stats.currentLoss.toFixed(6))}
      ${this.createStatCard('Average Loss', stats.avgLoss.toFixed(6))}
      ${this.createStatCard('Min Loss', stats.minLoss.toFixed(6))}
      ${this.createStatCard('Max Loss', stats.maxLoss.toFixed(6))}
      ${this.createStatCard('Elapsed Time', timeStr)}
      ${this.createStatCard('Updates/sec', updatesPerSecond.toFixed(2))}
    `;
  }

  /**
   * Create a stat card HTML
   */
  private createStatCard(label: string, value: string): string {
    return `
      <div style="
        background: #2d2d2d;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #3d3d3d;
      ">
        <div style="
          font-size: 12px;
          color: #888888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 5px;
        ">${label}</div>
        <div style="
          font-size: 20px;
          font-weight: 600;
          color: #4bc0c0;
        ">${value}</div>
      </div>
    `;
  }

  /**
   * Close and clean up the UI
   */
  close(): void {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
      this.container = null;
    }
  }

  /**
   * Get the training statistics
   */
  getStats(): {
    totalEpochs: number;
    currentLoss: number;
    avgLoss: number;
    minLoss: number;
    maxLoss: number;
  } {
    if (this.losses.length === 0) {
      return {
        totalEpochs: 0,
        currentLoss: 0,
        avgLoss: 0,
        minLoss: 0,
        maxLoss: 0
      };
    }

    return {
      totalEpochs: this.epochs[this.epochs.length - 1] || 0,
      currentLoss: this.losses[this.losses.length - 1] || 0,
      avgLoss: this.losses.reduce((a, b) => a + b, 0) / this.losses.length,
      minLoss: Math.min(...this.losses),
      maxLoss: Math.max(...this.losses)
    };
  }
}