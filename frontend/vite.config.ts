import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // GitHub Pages serves the site under /<repo>/; VITE_BASE is set by the
  // Pages workflow. Locally it stays "/".
  base: process.env.VITE_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
