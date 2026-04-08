import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  // 使用相对路径，确保Pywebview能正确加载静态资源
  base: './',
})
