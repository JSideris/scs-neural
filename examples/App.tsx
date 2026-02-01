import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { ColorPicker, FlappyBird, ShapeClassifier } from './examples';

const App: React.FC = () => {
  return (
    <Layout>
      <Routes>
        <Route path="/color-picker" element={<ColorPicker />} />
        <Route path="/flappy-bird" element={<FlappyBird />} />
        <Route path="/shape-classifier" element={<ShapeClassifier />} />
        <Route path="/" element={<Navigate to="/color-picker" replace />} />
        <Route path="*" element={<div>Select an example from the sidebar.</div>} />
      </Routes>
    </Layout>
  );
};

export default App;
