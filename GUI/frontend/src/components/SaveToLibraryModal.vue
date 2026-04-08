<script setup>
import { ref } from 'vue'

// 定义组件属性
const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  }
})

// 定义组件事件
const emit = defineEmits(['update:visible'])

// 手动输入的校准参数
const filamentName = ref('')
const filamentType = ref('PLA')
const manualK = ref('')
const manualS = ref('')
const saveSuccess = ref(false)

// 监听props变化，同步对话框显示状态
const dialogVisible = ref(props.visible)

// 关闭保存模态框
const closeSaveModal = () => {
  emit('update:visible', false)
  filamentName.value = ''
  manualK.value = ''
  manualS.value = ''
  filamentType.value = 'PLA'
  saveSuccess.value = false
}

// 保存到耗材库
const saveToFilamentLibrary = async () => {
  if (!filamentName.value.trim()) {
    alert('请输入耗材名称')
    return
  }
  
  let k, s
  try {
    k = JSON.parse(manualK.value)
    s = JSON.parse(manualS.value)
  } catch {
    alert('请输入有效的JSON格式参数')
    return
  }
  
  try {
    const filamentData = {
      Name: filamentName.value.trim(),
      FILAMENT_K: k,
      FILAMENT_S: s,
      Type: filamentType.value
    }
    
    const response = await fetch('http://localhost:5000/save_filament', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(filamentData)
    })
    
    if (!response.ok) {
      throw new Error('保存失败')
    }
    
    const result = await response.json()
    if (result.success) {
      saveSuccess.value = true
      setTimeout(() => {
        closeSaveModal()
      }, 2000)
    } else {
      throw new Error(result.error || '保存失败')
    }
  } catch (error) {
    alert('保存失败: ' + error.message)
  }
}
</script>

<template>
  <!-- 使用原生HTML对话框 -->
  <div v-if="visible" class="dialog-overlay">
    <div class="dialog">
      <div class="dialog-header">
        <h3>保存到耗材库</h3>
        <button class="close-btn" @click="closeSaveModal">×</button>
      </div>
      <div class="dialog-content">
        <!-- 成功提示 -->
        <div v-if="saveSuccess" class="success-message">
          <h3>保存成功！</h3>
          <p>耗材已成功添加到耗材库</p>
        </div>
        <!-- 表单 -->
        <form v-else class="simple-form" @submit.prevent="saveToFilamentLibrary">
          <div class="form-item">
            <label>耗材名称 *</label>
            <input 
              type="text" 
              v-model="filamentName" 
              placeholder="请输入耗材名称"
              maxlength="50"
            >
          </div>
          
          <div class="form-item">
            <label>耗材类型</label>
            <select v-model="filamentType">
              <option value="PLA">PLA</option>
              <option value="PETG">PETG</option>
            </select>
          </div>
          
          <div class="form-item">
            <label>FILAMENT_K *</label>
            <input 
              type="text" 
              v-model="manualK" 
              placeholder='例如: [0.001] 或 [0.0005, 0.001]'
            >
          </div>
          
          <div class="form-item">
            <label>FILAMENT_S *</label>
            <input 
              type="text" 
              v-model="manualS" 
              placeholder='例如: [60] 或 [50, 60]'
            >
          </div>
        </form>
      </div>
      <div class="dialog-footer" v-if="!saveSuccess">
        <button class="cancel-btn" @click="closeSaveModal">取消</button>
        <button class="confirm-btn" @click="saveToFilamentLibrary">保存</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.dialog {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  width: 500px;
  max-width: 90%;
  max-height: 90vh;
  overflow-y: auto;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #e4e7ed;
}

.dialog-header h3 {
  margin: 0;
  font-size: 16px;
}

.close-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #666;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: #333;
}

.dialog-content {
  padding: 16px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 16px;
  border-top: 1px solid #e4e7ed;
}

.success-message {
  text-align: center;
  padding: 30px 0;
}

.success-message h3 {
  margin: 20px 0 10px 0;
  font-size: 20px;
  color: #00b42a;
}

.success-message p {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.simple-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-item label {
  font-weight: 500;
  font-size: 14px;
  color: #333;
}

.form-item input,
.form-item select {
  padding: 8px 12px;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  font-size: 14px;
}

.form-item input:focus,
.form-item select:focus {
  outline: none;
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.cancel-btn,
.confirm-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.cancel-btn {
  background-color: #f5f7fa;
  color: #333;
  border: 1px solid #e4e7ed;
}

.cancel-btn:hover {
  background-color: #e4e7ed;
}

.confirm-btn {
  background-color: #1890ff;
  color: white;
}

.confirm-btn:hover {
  background-color: #40a9ff;
}
</style>
