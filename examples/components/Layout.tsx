import React from 'react';
import { Activity, Bird, Shapes, Github, Bug, Pipette } from 'lucide-react';
import { NavLink } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
}

const EXAMPLES = [
  { id: 'color-picker', name: 'Color Inverter', icon: <Pipette size={20} />, description: 'Dense network with backpropagation' },
  { id: 'flappy-bird', name: 'Flappy Bird', icon: <Bird size={20} />, description: 'Genetic algorithm evolution' },
  { id: 'shape-classifier', name: 'Shape Classifier', icon: <Shapes size={20} />, description: 'CNN for image recognition' },
  { id: 'ant-warfare', name: 'Ant Warfare', icon: <Bug size={20} />, description: 'Continuous convolutional genetic evolution' },
];

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', width: '100vw' }}>
      {/* Sidebar */}
      <div style={{
        width: '300px',
        backgroundColor: '#1e293b',
        borderRight: '1px solid #334155',
        display: 'flex',
        flexDirection: 'column',
        padding: '1.5rem'
      }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#f8fafc', margin: 0 }}>SCS Neural</h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '0.5rem' }}>WebGPU Powered Neural Networks</p>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
          {EXAMPLES.map((ex) => (
            <NavLink
              key={ex.id}
              to={`/${ex.id}`}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                textAlign: 'left',
                textDecoration: 'none',
                backgroundColor: isActive ? '#334155' : 'transparent',
                color: isActive ? '#38bdf8' : '#cbd5e1',
                transition: 'all 0.2s'
              })}
            >
              {ex.icon}
              <div>
                <div style={{ fontWeight: 600 }}>{ex.name}</div>
                <div style={{ fontSize: '0.75rem' }}>{ex.description}</div>
              </div>
            </NavLink>
          ))}
        </nav>

        <div style={{ marginTop: 'auto', paddingTop: '1.5rem', borderTop: '1px solid #334155' }}>
          <a 
            href="https://github.com/joshuasideris/scs-neural" 
            target="_blank" 
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              color: '#94a3b8',
              textDecoration: 'none',
              fontSize: '0.875rem'
            }}
          >
            <Github size={18} />
            GitHub Repository
          </a>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ flex: 1, backgroundColor: '#0f172a', padding: '2rem', overflowY: 'auto' }}>
        <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
          {children}
        </div>
      </main>
    </div>
  );
};
