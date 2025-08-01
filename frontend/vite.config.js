import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  root: './', // 프로젝트 루트 디렉토리
  base: './', // 상대 경로 기반 배포 시
  server: {
    port: 8001, // 개발 서버 포트
    host: '0.0.0.0',
    https: true,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8765',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist', // 빌드 결과물 디렉토리
  },
  plugins: [wasm(), basicSsl()],
  optimizeDeps: {
    exclude: ['lz4-wasm'],
  },
});
