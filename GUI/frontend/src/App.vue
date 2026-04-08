<script setup>
import { ref } from 'vue'
// 引入组件
import Sidebar from './components/Sidebar.vue'
import ConsumablesLibrary from './components/ConsumablesLibrary.vue'
import ModelGeneration from './components/ModelGeneration.vue'
import ConsumablesCalibration from './components/ConsumablesCalibration.vue'

// 当前选中的导航项
const activeKey = ref('1')
</script>

<template>
  <div class="app-container">
    <!-- 侧边栏导航 -->
    <Sidebar v-model:activeKey="activeKey" />
    
    <!-- 主内容区域 -->
    <main class="main-content">
      <div class="content-header">
        <h2>
          {{ activeKey === '1' ? '耗材库' : 
             activeKey === '2' ? '模型生成' : '耗材校正' }}
        </h2>
      </div>
      <div class="content-body">
        <!-- 使用keep-alive包裹组件，保持组件状态 -->
        <keep-alive>
          <ConsumablesLibrary v-if="activeKey === '1'" key="1" />
          <ModelGeneration v-if="activeKey === '2'" key="2" />
          <ConsumablesCalibration v-if="activeKey === '3'" key="3" />
        </keep-alive>
      </div>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: auto;
  background-color: #fff;
}

.content-header {
  padding: 20px;
  border-bottom: 1px solid #e4e7ed;
  background-color: #fff;
}

.content-header h2 {
  margin: 0;
  font-size: 24px;
  color: #333;
}

.content-body {
  flex: 1;
  padding: 20px;
  overflow: auto;
}
</style>
