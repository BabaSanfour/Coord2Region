import React from 'react';
import { createRoot } from 'react-dom/client';
import ConfigBuilder from './ConfigBuilder';

const mountNode = document.getElementById('coord2region-root') ?? document.getElementById('root');

if (mountNode) {
  const root = createRoot(mountNode);
  root.render(
    <React.StrictMode>
      <ConfigBuilder showHeaderNav={mountNode.id !== 'coord2region-root'} />
    </React.StrictMode>
  );
} else if ((import.meta as any)?.env?.DEV) {
  console.warn('Coord2Region root element not found.');
}
