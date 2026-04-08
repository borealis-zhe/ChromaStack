import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
// 引入TDesign组件库 - 使用标准引入方式
import TDesign from 'tdesign-vue-next'
import 'tdesign-vue-next/es/style/index.css'

const app = createApp(App)
// 注册所有TDesign组件
app.use(TDesign)
app.mount('#app')
