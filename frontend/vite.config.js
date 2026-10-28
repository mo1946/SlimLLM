import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

// Only the frontend and its shared public Schema are readable by the dev server.
const proxy = { '/api': {
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
  rewrite: (path) => path.replace(/^\/api(?=\/|$)/, ''),
  timeout: 130_000,
  proxyTimeout: 130_000,
} }

export default defineConfig({
  plugins: [vue()],
  cacheDir: '../.cache/vite',
  server: {
    host: '127.0.0.1', port: 5173, strictPort: true, proxy,
    fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), fileURLToPath(new URL('../data/ticket.schema.json', import.meta.url))] },
  },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true, proxy },
  build: { outDir: 'dist', emptyOutDir: false },
})
