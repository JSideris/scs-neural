import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, icon }) => {
  return (
    <div style={{
      background: '#1e293b',
      padding: '1.25rem',
      borderRadius: '0.5rem',
      border: '1px solid #334155',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.25rem'
    }}>
      <div style={{
        fontSize: '0.75rem',
        color: '#94a3b8',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        fontWeight: 600,
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        {icon}
        {label}
      </div>
      <div style={{
        fontSize: '1.5rem',
        fontWeight: 700,
        color: '#38bdf8'
      }}>
        {value}
      </div>
    </div>
  );
};
