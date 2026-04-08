<script setup>
import { ref } from 'vue'
import SaveToLibraryModal from './SaveToLibraryModal.vue'

// 耗材校正相关变量
const uploadFiles = ref([])
const previewImage = ref('')
const calibrationRunning = ref(false)
const progress = ref(0)
const calibrationResult = ref('')
const tmpImagePath = ref('')
const saveModalVisible = ref(false)
const fileInputRef = ref(null)

// 清除图片
const clearImage = () => {
  // 清除文件列表
  uploadFiles.value = []
  
  // 清除预览图片
  previewImage.value = ''
  
  // 清除校准结果
  calibrationResult.value = ''
  
  // 释放临时URL对象
  if (tmpImagePath.value) {
    URL.revokeObjectURL(tmpImagePath.value)
    tmpImagePath.value = ''
  }
  
  // 重置文件输入元素，确保可以再次上传相同的图片
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

// 处理文件选择（原生input）
const handleFileChangeSimple = (event) => {
  const files = event.target.files
  if (files.length > 0) {
    const file = files[0]
    const reader = new FileReader()
    
    // 生成预览图片
    reader.onload = (e) => {
      previewImage.value = e.target.result
    }
    
    reader.readAsDataURL(file)
    
    // 保存文件对象
    uploadFiles.value = [{ raw: file }]
    
    // 保存临时文件URL
    tmpImagePath.value = URL.createObjectURL(file)
  }
}

// 开始校准
const startCalibration = async () => {
  calibrationRunning.value = true
  progress.value = 0
  calibrationResult.value = ''
  
  try {
    // 获取文件对象
    if (uploadFiles.value.length === 0) {
      calibrationRunning.value = false
      return
    }
    
    const file = uploadFiles.value[0].raw
    
    // 创建FormData对象，用于文件上传
    const formData = new FormData()
    formData.append('file', file)
    
    // 1. 上传文件
    progress.value = 20
    const uploadResponse = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData
    })
    
    if (!uploadResponse.ok) {
      throw new Error('文件上传失败')
    }
    
    const uploadData = await uploadResponse.json()
    if (!uploadData.success) {
      throw new Error(uploadData.error || '文件上传失败')
    }
    
    progress.value = 50
    
    // 2. 执行校准
    const calibrateResponse = await fetch('http://localhost:5000/calibrate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        file_path: uploadData.file_path
      })
    })
    
    if (!calibrateResponse.ok) {
      throw new Error('校准执行失败')
    }
    
    const calibrateData = await calibrateResponse.json()
    if (!calibrateData.success) {
      throw new Error(calibrateData.error || '校准执行失败')
    }
    
    progress.value = 100
    
    // 3. 显示校准结果（只输出结果，不解析）
    let result = '校准完成！\n\n'
    if (calibrateData.stdout) {
      result += '=== 标准输出 ===\n'
      result += calibrateData.stdout + '\n\n'
    }
    
    if (calibrateData.stderr) {
      result += '=== 错误输出 ===\n'
      result += calibrateData.stderr + '\n\n'
    }
    
    calibrationResult.value = result
  } catch (error) {
    calibrationResult.value = '校准失败！\n\n错误信息: ' + error.message
  } finally {
    calibrationRunning.value = false
  }
}

// 手动显示保存模态框
const showSaveModal = () => {
  saveModalVisible.value = true
}
</script>

<template>
  <div class="consumables-calibration-container">

    <div class="page-header">
      <h1>耗材校正</h1>
      <p class="page-description">欢迎使用耗材校正功能，通过拍摄的校准图片生成耗材的颜色参数</p>
    </div>

    <!-- 主内容区域 -->
    <div class="content-single">
      <div class="main-panel">
        <!-- 上传图片和预览区域 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">1. 上传校准图片</h2>
            <p class="card-description">上传拍摄的校准图片，用于生成耗材的颜色参数</p>
          </div>
          <div class="card-body">
            <div class="upload-section">
              <div class="upload-controls">
                <input
                  ref="fileInputRef"
                  type="file"
                  accept="image/*"
                  @change="handleFileChangeSimple"
                  class="simple-upload"
                />
              </div>
              
              <!-- 原图预览 -->
              <div class="preview-section">
                <h3 class="preview-title">图片预览</h3>
                <div class="preview-container">
                  <img v-if="previewImage" :src="previewImage" alt="校准图片" class="preview-img" />
                  <div v-else class="preview-placeholder">
                    <div class="placeholder-icon">📷</div>
                    <p class="placeholder-text">请上传校准图片</p>
                    <p class="placeholder-hint">支持 JPG、PNG 等常见图像格式</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 校准操作区域 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">2. 执行校准</h2>
            <p class="card-description">点击开始校准按钮，系统将自动分析图片并生成颜色参数</p>
          </div>
          <div class="card-body">
            <div class="card-actions">
              <t-button 
                type="primary" 
                :disabled="!previewImage"
                @click="startCalibration"
                class="action-button"
              >
                开始校准
              </t-button>
              <t-button 
                type="success" 
                :disabled="!calibrationResult"
                @click="showSaveModal"
                class="action-button"
              >
                新增耗材
              </t-button>
            </div>
          </div>
        </div>
        
        <!-- 校准进度区域 -->
        <div v-if="calibrationRunning" class="card">
          <div class="card-body">
            <div class="calibration-progress">
              <p>校准中...请稍候</p>
              <t-progress :percent="progress" :stroke-width="2" />
            </div>
          </div>
        </div>
        
        <!-- 校准结果区域 -->
        <div v-if="calibrationResult" class="card">
          <div class="card-header">
            <h2 class="card-title">3. 校准结果</h2>
            <p class="card-description">校准完成，以下是生成的颜色参数和输出结果</p>
          </div>
          <div class="card-body">
            <div class="calibration-result">
              <div class="result-content">
                <pre>{{ calibrationResult }}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  
  <!-- 保存到耗材库模态框 -->
  <!-- 保存到耗材库模态框 -->
  <SaveToLibraryModal
    v-model:visible="saveModalVisible"
  />
  </div>
</template>

<style scoped>
/* 全局样式 - 匹配模型生成页面 */
.consumables-calibration-container {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 页面头部 - 匹配模型生成页面 */
.page-header {
  margin-bottom: 32px;
  text-align: center;
}

.page-header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #1a1a1a;
  margin-bottom: 8px;
}

.page-description {
  font-size: 16px;
  color: #666;
  margin: 0;
}

/* 单一列布局 - 匹配模型生成页面 */
.content-single {
  display: flex;
  justify-content: center;
}

/* 主面板样式 - 匹配模型生成页面 */
.main-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 100%;
  max-width: 800px;
}

/* 卡片样式 - 匹配模型生成页面 */
.card {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  transition: all 0.3s ease;
}

.card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.card-header {
  padding: 20px 24px;
  border-bottom: 1px solid #f0f0f0;
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0 0 4px 0;
}

.card-description {
  font-size: 14px;
  color: #666;
  margin: 0;
}

.card-body {
  padding: 24px;
}

/* 上传区域 - 匹配模型生成页面 */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 上传行样式 - 匹配模型生成页面 */
.upload-row {
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: flex-start;
  flex-wrap: wrap;
}

/* 上传控件样式 - 匹配模型生成页面 */
.upload-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}

/* 图片预览区域样式 - 匹配模型生成页面 */
.preview-section {
  width: 100%;
}

.preview-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin: 0 0 12px 0;
}

/* 预览容器样式 - 匹配模型生成页面 */
.preview-container {
  width: 100%;
  height: 320px;
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #fafafa;
  overflow: hidden;
  transition: all 0.3s ease;
}

/* 预览图片样式 - 匹配模型生成页面 */
.preview-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

/* 预览占位符样式 - 匹配模型生成页面 */
.preview-placeholder {
  text-align: center;
  color: #999;
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.placeholder-text {
  font-size: 16px;
  font-weight: 500;
  margin: 0 0 8px 0;
  color: #666;
}

.placeholder-hint {
  font-size: 13px;
  margin: 0;
  color: #999;
}

/* 校准进度区域 - 匹配模型生成页面 */
.calibration-progress {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 20px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.calibration-progress p {
  margin: 0;
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

/* 校准结果区域 - 匹配模型生成页面 */
.calibration-result {
  margin-top: 20px;
}

.result-content {
  background: #fafafa;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
  overflow: auto;
  max-height: 400px;
}

.calibration-result pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  font-size: 14px;
  color: #333;
  margin: 0;
  line-height: 1.5;
  cursor: text;
  user-select: text;
  -webkit-user-select: text;
  -moz-user-select: text;
  -ms-user-select: text;
}

/* 原生上传按钮样式 - 匹配模型生成页面 */
.simple-upload {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  font-size: 14px;
  color: #333;
  background-color: #fff;
  transition: all 0.2s ease;
  cursor: pointer;
}

.simple-upload:hover {
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

/* 清除按钮样式 - 匹配模型生成页面 */
.clear-button {
  white-space: nowrap;
}

/* 卡片操作区 - 匹配模型生成页面 */
.card-actions {
  margin-bottom: 0;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.action-button {
  width: 100%;
  padding: 10px 0;
  font-size: 16px;
  font-weight: 500;
}

/* 响应式设计 - 匹配模型生成页面 */
@media (max-width: 992px) {
  .main-panel {
    max-width: 100%;
  }
}

@media (max-width: 768px) {
  .consumables-calibration-container {
    padding: 16px;
  }
  
  .page-header h1 {
    font-size: 24px;
  }
  
  .card-header,
  .card-body {
    padding: 16px;
  }
  
  .preview-container {
    height: 200px;
  }
  
  .upload-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .simple-upload {
    width: 100%;
  }
}
</style>