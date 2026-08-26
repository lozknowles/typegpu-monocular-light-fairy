import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';

export default defineConfig({
  base: './',
  plugins: [typegpu()],
  publicDir: 'public',
  build: {
    emptyOutDir: true,
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
  },
});
