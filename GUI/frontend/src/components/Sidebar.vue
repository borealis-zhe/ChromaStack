<script setup>
import { defineProps, defineEmits, ref, watch } from 'vue'

// 定义组件属性
const props = defineProps({
  // 当前激活的菜单项
  activeKey: {
    type: String,
    default: '1'
  }
})

// 定义组件事件
const emit = defineEmits(['update:activeKey'])

// 将prop转换为本地ref
const localActiveKey = ref(props.activeKey)

// 监听prop变化，更新本地ref
watch(() => props.activeKey, (newVal) => {
  localActiveKey.value = newVal
})

// 监听本地ref变化，触发事件
watch(localActiveKey, (newVal) => {
  emit('update:activeKey', newVal)
})

// 处理菜单点击事件
const handleMenuClick = (e) => {
  localActiveKey.value = e.value
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <h1 class="app-title">ChromaStack</h1>
    </div>
    <div class="nav-container">
      <!-- 使用TDesign的Menu组件 -->
      <t-menu 
        v-model:value="localActiveKey" 
        theme="light"
        @click="handleMenuClick"
      >
        <t-menu-item value="1">耗材库</t-menu-item>
        <t-menu-item value="2">模型生成</t-menu-item>
        <t-menu-item value="3">耗材校正</t-menu-item>
      </t-menu>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 240px;
  background-color: #f5f7fa;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid #e4e7ed;
  text-align: center;
}

.app-title {
  margin: 0;
  font-size: 20px;
  color: #1890ff;
}

.nav-container {
  flex: 1;
  padding: 20px 0;
}
</style>