import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  root: path.resolve(__dirname, 'website'),
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'website/src')
    }
  },
  build: {
    outDir: path.resolve(__dirname, 'assets'),
    emptyOutDir: false,
    sourcemap: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'website/src/main.tsx'),
      output: {
        entryFileNames: 'js/bundle.js',
        chunkFileNames: 'js/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.css')) {
            return 'css/[name].css';
          }
          return 'assets/[name]-[hash][extname]';
        }
      }
    }
  },
  server: {
    fs: {
      allow: [
        path.resolve(__dirname, '.'),
        path.resolve(__dirname, '..'),
        path.resolve(__dirname, '../..')
      ]
    }
  }
});
