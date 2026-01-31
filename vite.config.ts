import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  const isLib = mode === 'lib';

  return {
    plugins: [
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
        external: ['simple-compute-shaders', 'chart.js'],
        output: {
          globals: {
            'simple-compute-shaders': 'SimpleComputeShaders',
            'chart.js': 'Chart',
          },
        },
      },
      outDir: 'dist',
    } : {
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
          flappy: resolve(__dirname, 'examples/flappy-bird/index.html'),
          color: resolve(__dirname, 'examples/color-picker/index.html'),
        },
      },
      outDir: 'dist-examples',
    },
  };
});
