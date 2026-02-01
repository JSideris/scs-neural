import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  const isLib = mode === 'lib';

  return {
    plugins: [
      react(),
      isLib && dts({
        include: ['src'],
        insertTypesEntry: true,
      }),
    ].filter(Boolean),
    build: isLib ? {
      lib: {
        entry: resolve(__dirname, 'src/index.ts'),
        name: 'SCSNeural',
        fileName: 'scs-neural',
        formats: ['es', 'umd'],
      },
      rollupOptions: {
        external: ['simple-compute-shaders', 'chart.js', 'react', 'react-dom'],
        output: {
          globals: {
            'simple-compute-shaders': 'SimpleComputeShaders',
            'chart.js': 'Chart',
            'react': 'React',
            'react-dom': 'ReactDOM',
          },
        },
      },
      outDir: 'dist',
    } : {
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
        },
      },
      outDir: 'dist-examples',
    },
  };
});
