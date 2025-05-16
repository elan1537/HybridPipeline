import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  root: './', // 프로젝트 루트 디렉토리
  base: './', // 상대 경로 기반 배포 시
  server: {
    port: 8000, // 개발 서버 포트
    host: true,
  },
  build: {
    outDir: 'dist', // 빌드 결과물 디렉토리
  },
  plugins: [wasm()],
  optimizeDeps: {
    exclude: ['lz4-wasm'],
  },
});
