import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface LossGraphProps {
  epochs: number[];
  losses: number[];
  title?: string;
  maxDataPoints?: number;
}

export const LossGraph: React.FC<LossGraphProps> = ({ 
  epochs, 
  losses, 
  title = 'Training Loss',
  maxDataPoints = 1000 
}) => {
  const data = useMemo(() => {
    // Slice data if it exceeds maxDataPoints
    const displayEpochs = epochs.slice(-maxDataPoints);
    const displayLosses = losses.slice(-maxDataPoints);

    return {
      labels: displayEpochs,
      datasets: [
        {
          label: 'Loss',
          data: displayLosses,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: true,
        },
      ],
    };
  }, [epochs, losses, maxDataPoints]);

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Epoch',
          color: '#94a3b8',
        },
        ticks: {
          color: '#64748b',
          maxTicksLimit: 10,
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.5)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Loss',
          color: '#94a3b8',
        },
        ticks: {
          color: '#64748b',
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.5)',
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: title,
        color: '#f8fafc',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
    },
  };

  return (
    <div style={{ 
      height: '300px', 
      width: '100%', 
      position: 'relative',
      overflow: 'hidden',
      background: '#1e293b', 
      padding: '1rem', 
      borderRadius: '0.5rem',
      border: '1px solid #334155'
    }}>
      <Line data={data} options={options} />
    </div>
  );
};
