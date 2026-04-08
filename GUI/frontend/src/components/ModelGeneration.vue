<script setup>
import { ref, onMounted, computed } from 'vue'

// 固定配置参数 - 根据 ChromaStackStudio.py 配置
const fixedConfig = ref({
  model_width: 80,  // 对应 TARGET_WIDTH_MM
  model_depth: 0.8,  // 对应 BASE_HEIGHT
  layer_height: 0.08,  // 对应 LAYER_HEIGHT
  pixel_size: 0.2,  // 对应 PIXEL_SIZE
  alpha_threshold: 128,  // 对应 ALPHA_THRESHOLD
  min_pixel_size: 5,  // 对应 generate_regions_felzenszwalb 中的 min_pixel_size
  scale: 10,  // 对应 generate_regions_felzenszwalb 中的 scale
  sigma: 0.5,  // 对应 generate_regions_felzenszwalb 中的 sigma
  fixed_base_slot: 'CooBeen-白'  // 对应 SELECTED_FILAMENT_NAMES[0]
})

// 临时配置
const tempConfig = ref({
  model_width: 80,  // 对应 TARGET_WIDTH_MM
  model_height: 80,  // 默认为与宽度相同
  model_depth: 0.8,  // 对应 BASE_HEIGHT
  layer_height: 0.08,  // 对应 LAYER_HEIGHT
  color_count: 4,  // 对应颜色提取的数量
  is_double_sided: true  // 是否生成双面模型
})

// 状态变量
const configLoading = ref(false)
const configSaved = ref(false)
const uploadFiles = ref([])
const previewImage = ref('')
const originalImage = ref('')
const fileInputRef = ref(null)
const colorizeRunning = ref(false)
const colorizeResult = ref('')
const extractedColors = ref([])
const filamentCombinations = ref([])
const filaments = ref([])
const selectedFilaments = ref([])
const generateRunning = ref(false)
const previewRunning = ref(false)
const previewResult = ref('')
const finalStackMatrix = ref([])  // 保存预览时生成的矩阵
const imageAspectRatio = ref(1)  // 保存上传图片的宽高比

// 加载配置
const loadConfig = async () => {
  configLoading.value = true
  try {
    const response = await fetch('http://localhost:5000/config/model')
    const data = await response.json()
    if (data.success && data.config) {
      fixedConfig.value = { ...fixedConfig.value, ...data.config }
      tempConfig.value = {
        color_count: fixedConfig.value.color_count,
        model_width: fixedConfig.value.model_width,
        model_height: fixedConfig.value.model_height,
        model_depth: fixedConfig.value.model_depth,
        layer_height: fixedConfig.value.layer_height,
        is_double_sided: fixedConfig.value.is_double_sided !== false
      }
    }
  } catch (error) {
    console.error('加载配置失败:', error)
  } finally {
    configLoading.value = false
  }
}

// 保存固定配置
const saveConfig = async () => {
  try {
    const configToSave = { ...fixedConfig.value, ...tempConfig.value }
    const response = await fetch('http://localhost:5000/config/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(configToSave)
    })
    const data = await response.json()
    if (data.success) {
      configSaved.value = true
      setTimeout(() => { configSaved.value = false }, 2000)
    } else {
      alert('保存配置失败: ' + (data.error || '未知错误'))
    }
  } catch (error) {
    alert('保存配置失败: ' + error.message)
  }
}

// 加载耗材列表
const loadFilaments = async () => {
  try {
    const response = await fetch('http://localhost:5000/filaments')
    const data = await response.json()
    if (data.success) {
      filaments.value = data.filaments
    }
  } catch (error) {
    console.error('加载耗材列表失败:', error)
  }
}

// 处理文件选择
const handleFileChange = (event) => {
  const files = event.target.files
  if (files.length > 0) {
    const file = files[0]
    const reader = new FileReader()
    
    reader.onload = (e) => {
      previewImage.value = e.target.result
      originalImage.value = e.target.result
      
      // 读取图片尺寸并调整模型高度
      const img = new Image()
      img.onload = () => {
        // 保存图片的宽高比
        imageAspectRatio.value = img.height / img.width
        // 根据当前模型宽度和图片比例计算模型高度
        tempConfig.value.model_height = Math.round(tempConfig.value.model_width * imageAspectRatio.value)
      }
      img.src = e.target.result
    }
    
    reader.readAsDataURL(file)
    uploadFiles.value = [{ raw: file }]
    
    // 清除之前的结果
    colorizeResult.value = ''
    extractedColors.value = []
    previewResult.value = ''
  }
}

// 监听模型宽度变化，自动调整高度保持比例
const handleWidthChange = () => {
  if (imageAspectRatio.value) {
    // 使用保存的图片宽高比计算模型高度
    tempConfig.value.model_height = Math.round(tempConfig.value.model_width * imageAspectRatio.value)
  }
}

// 监听模型高度变化，自动调整宽度保持比例
const handleHeightChange = () => {
  if (imageAspectRatio.value) {
    // 使用保存的图片宽高比计算模型宽度
    tempConfig.value.model_width = Math.round(tempConfig.value.model_height / imageAspectRatio.value)
  }
}

// 清除图片
const clearImage = () => {
  uploadFiles.value = []
  previewImage.value = ''
  originalImage.value = ''
  colorizeResult.value = ''
  extractedColors.value = []
  previewResult.value = ''
  imageAspectRatio.value = 1  // 重置宽高比
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

// 自动配色
const startColorize = async () => {
  if (uploadFiles.value.length === 0) {
    alert('请先上传图片')
    return
  }
  
  colorizeRunning.value = true
  colorizeResult.value = ''
  extractedColors.value = []
  filamentCombinations.value = []
  
  try {
    const file = uploadFiles.value[0].raw
    const formData = new FormData()
    formData.append('file', file)
    formData.append('color_count', tempConfig.value.color_count)
    
    // 直接调用配色接口，使用已上传的文件
    const colorizeResponse = await fetch('http://localhost:5000/colorize', {
      method: 'POST',
      body: formData
    })
    
    if (!colorizeResponse.ok) {
      throw new Error('配色失败')
    }
    
    const colorizeData = await colorizeResponse.json()
    console.log('配色接口返回数据:', colorizeData)
    
    // 确保返回的数据结构正确
    if (colorizeData && colorizeData.success) {
      // 简单处理，直接显示返回的结果
      colorizeResult.value = '配色成功！'
      
      // 尝试获取推荐的耗材组合
      if (colorizeData.top_combinations && Array.isArray(colorizeData.top_combinations)) {
        filamentCombinations.value = colorizeData.top_combinations
      } else {
        filamentCombinations.value = []
      }
    } else {
      throw new Error(colorizeData.error || '配色失败')
    }
  } catch (error) {
    console.error('配色失败:', error)
    colorizeResult.value = '配色失败: ' + error.message
    // 确保即使出错，filamentCombinations 也是一个数组
    filamentCombinations.value = []
  } finally {
    colorizeRunning.value = false
  }
}

// 生成预览图
const generatePreview = async () => {
  if (uploadFiles.value.length === 0) {
    alert('请先在前面的步骤上传图片')
    return
  }
  
  if (selectedFilaments.value.length < 2) {
    alert('请至少选择2个耗材')
    return
  }
  
  previewRunning.value = true
  previewResult.value = ''
  
  try {
    const file = uploadFiles.value[0].raw
    const formData = new FormData()
    formData.append('file', file)
    formData.append('filaments', JSON.stringify(selectedFilaments.value))
    formData.append('min_pixel_size', fixedConfig.value.min_pixel_size)
    formData.append('scale', fixedConfig.value.scale)
    formData.append('sigma', fixedConfig.value.sigma)
    formData.append('layer_height', tempConfig.value.layer_height)
    formData.append('model_width', tempConfig.value.model_width)
    formData.append('model_height', tempConfig.value.model_height)
    formData.append('pixel_size', fixedConfig.value.pixel_size)
    formData.append('alpha_threshold', fixedConfig.value.alpha_threshold)
    
    // 调用预览接口
    const previewResponse = await fetch('http://localhost:5000/preview', {
      method: 'POST',
      body: formData
    })
    
    if (!previewResponse.ok) {
      throw new Error('预览生成失败')
    }
    
    const previewData = await previewResponse.json()
    console.log('预览接口返回数据:', previewData)
    if (previewData.success) {
      // 直接使用后端返回的预览图路径
      previewResult.value = 'http://localhost:5000' + previewData.preview_path
      console.log('预览图路径:', previewResult.value)
    } else {
      throw new Error(previewData.error || '预览生成失败')
    }
  } catch (error) {
    console.error('预览生成失败:', error)
    alert('预览生成失败: ' + error.message)
  } finally {
    previewRunning.value = false
  }
}

// 生成模型
const generateModel = async () => {
  if (uploadFiles.value.length === 0) {
    alert('请先在前面的步骤上传图片')
    return
  }
  
  if (selectedFilaments.value.length < 2) {
    alert('请至少选择2个耗材')
    return
  }
  
  generateRunning.value = true
  
  try {
    const file = uploadFiles.value[0].raw
    const formData = new FormData()
    formData.append('file', file)
    formData.append('filaments', JSON.stringify(selectedFilaments.value))
    formData.append('min_pixel_size', fixedConfig.value.min_pixel_size)
    formData.append('scale', fixedConfig.value.scale)
    formData.append('sigma', fixedConfig.value.sigma)
    formData.append('layer_height', tempConfig.value.layer_height)
    formData.append('model_width', tempConfig.value.model_width)
    formData.append('model_height', tempConfig.value.model_height)
    formData.append('model_depth', tempConfig.value.model_depth)
    formData.append('pixel_size', fixedConfig.value.pixel_size)
    formData.append('alpha_threshold', fixedConfig.value.alpha_threshold)
    formData.append('is_double_sided', tempConfig.value.is_double_sided)
    
    // 直接调用生成接口，使用已上传的文件
    const generateResponse = await fetch('http://localhost:5000/generate', {
      method: 'POST',
      body: formData
    })
    
    if (!generateResponse.ok) {
      throw new Error('模型生成失败')
    }
    
    const generateData = await generateResponse.json()
    if (generateData.success) {
      // 模型生成成功，可以显示成功消息
      alert('模型生成成功！请查看 Output 目录下的 3MF 文件。')
    } else {
      throw new Error(generateData.error || '模型生成失败')
    }
  } catch (error) {
    alert('模型生成失败: ' + error.message)
  } finally {
    generateRunning.value = false
  }
}

// 组件挂载时加载配置和耗材
onMounted(() => {
  loadConfig()
  loadFilaments()
})

// 计算属性：是否可以选择生成模型
const canGenerate = computed(() => {
  return uploadFiles.value.length > 0 && selectedFilaments.value.length >= 2
})
</script>

<template>
  <div class="model-generation-container">
    <div class="page-header">
      <h1>模型生成</h1>
      <p class="page-description">欢迎使用模型生成功能，上传图片并配置参数生成3D模型</p>
    </div>

    <!-- 主内容区域 -->
    <div class="content-single">
      <!-- 配置和操作区域 -->
      <div class="main-panel">
        <!-- 算法固定配置 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">算法固定配置</h2>
            <p class="card-description">配置算法的基本参数，影响模型生成的质量</p>
          </div>
          <div class="card-body">
            <div class="config-grid">
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">最小区域像素数</label>
                  <t-tooltip content="区域分割算法中的最小区域像素数，强制合并小于此阈值的区域到相邻区域，用于消除过小的碎片化区域。建议值：5-20">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="fixedConfig.min_pixel_size" 
                  :step="1"
                  theme="column"
                >
                <template #suffix><span>px</span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">分割 scale</label>
                  <t-tooltip content="区域分割的尺度参数，控制分割的粗细程度。值越大，分割越粗，区域数量越少；值越小，分割越细，区域数量越多。建议值：10-30">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="fixedConfig.scale" 
                  :step="1"
                  theme="column"
                >
                <template #suffix><span></span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">分割 sigma</label>
                  <t-tooltip content="高斯平滑的标准差。在构建图之前，先对图像进行高斯平滑滤波，用于抑制噪声，避免过度分割。控制平滑程度：值越大，图像越平滑，细节丢失越多。典型值：0.5 - 2.0（通常设为0.8或1.0）">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="fixedConfig.sigma" 
                  :step="0.1" 
                  :precision="1"
                  theme="column"
                >
                <template #suffix><span></span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">像素尺寸</label>
                  <t-tooltip content="每个像素对应的实际物理尺寸，直接影响生成模型的大小和精度。值越大，模型越大但细节越少；值越小，模型越小但细节越丰富。建议值：0.15-0.3">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="fixedConfig.pixel_size" 
                  :step="0.01" 
                  :precision="2"
                  theme="column"
                >
                <template #suffix><span>mm</span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">透明度阈值</label>
                  <t-tooltip content="判断像素是否透明的阈值，范围0-255。值越大，透明区域越少，更多像素会被视为不透明；值越小，透明区域越多。建议值：128">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="fixedConfig.alpha_threshold" 
                  :step="1" 
                  :min="0" 
                  :max="255"
                  theme="column"
                >
                <template #suffix><span></span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">底座耗材</label>
                  <t-tooltip content="选择模型底座使用的耗材，建议使用白色或黑色等中性色耗材，以确保模型主体颜色的准确性">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input v-model="fixedConfig.fixed_base_slot" placeholder="请输入底座耗材名称" />
              </div>
            </div>
            <div class="card-actions">
              <t-button type="primary" @click="saveConfig" :loading="configLoading">
                {{ configSaved ? '已保存' : '保存配置' }}
              </t-button>
            </div>
          </div>
        </div>

        <!-- 图像上传 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">图像上传</h2>
            <p class="card-description">上传一张图片用于生成模型</p>
          </div>
          <div class="card-body">
            <div class="upload-section">
              <div class="upload-controls">
                <input
                  ref="fileInputRef"
                  type="file"
                  accept="image/*"
                  @change="handleFileChange"
                  class="file-input"
                />
                <t-button @click="clearImage" class="clear-button">清除图片</t-button>
              </div>
              
              <!-- 原图预览 -->
              <div class="preview-section">
                <h3 class="preview-title">原图预览</h3>
                <div class="preview-container">
                  <img v-if="previewImage" :src="previewImage" alt="原图" class="preview-image" />
                  <div v-else class="preview-placeholder">
                    <div class="placeholder-icon">📷</div>
                    <p class="placeholder-text">请上传图片</p>
                    <p class="placeholder-hint">支持 JPG、PNG 等常见图像格式</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 自动配色 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">自动配色</h2>
            <p class="card-description">从上传的图片中提取颜色特征</p>
          </div>
          <div class="card-body">
            <!-- 颜色数量选择 -->
            <div class="color-count-section">
              <div class="label-with-tooltip">
                <label class="config-label">颜色数量</label>
                <t-tooltip content="从图片中提取的颜色数量，影响配色的丰富度。值越大，提取的颜色越多，细节越丰富，但可能增加打印复杂度；值越小，提取的颜色越少，整体效果越简洁。建议值：3-8">
                  <span class="info-icon">?</span>
                </t-tooltip>
              </div>
              <t-input-number 
                v-model="tempConfig.color_count" 
                :step="1" 
                :min="2" 
                :max="10"
              >
              <template #suffix><span>个</span></template>
              </t-input-number>
            </div>
            
            <div class="action-section">
              <t-button 
                type="primary" 
                :disabled="!previewImage"
                :loading="colorizeRunning"
                @click="startColorize"
                class="action-button"
              >
                自动配色
              </t-button>
            </div>
            
            <!-- 配色结果 -->
            <div v-if="colorizeResult" class="result-section">
              <h3 class="result-title">配色结果</h3>
              <div class="result-content">
                <p class="result-text">{{ colorizeResult }}</p>
                <div v-if="filamentCombinations.length > 0" class="filament-combinations">
                  <h4 class="result-title">推荐耗材组合</h4>
                  <div v-for="(combo, index) in filamentCombinations" :key="index" class="combo-item">
                    <div class="combo-header">
                      <span class="combo-rank">第 {{ index + 1 }} 名</span>
                      <t-tag theme="primary" variant="outline" class="combo-score">
                        误差: {{ combo.score.toFixed(2) }}
                      </t-tag>
                    </div>
                    <div class="combo-filaments">
                      <t-tag 
                        v-for="(filament, fIndex) in combo.filaments" 
                        :key="fIndex" 
                        theme="default"
                        class="filament-tag"
                      >
                        {{ filament }}
                      </t-tag>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 耗材选择 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">耗材选择</h2>
            <p class="card-description">请选择要使用的耗材（至少选择2个）</p>
          </div>
          <div class="card-body">
            <t-select
              v-model="selectedFilaments"
              multiple
              placeholder="请选择耗材"
              style="width: 100%; margin-bottom: 12px;"
            >
              <t-option 
                v-for="filament in filaments" 
                :key="filament.Name" 
                :value="filament.Name" 
                :label="filament.Name"
              />
            </t-select>
            <div class="selection-info">
              <span class="info-label">已选择:</span>
              <span class="info-value">{{ selectedFilaments.length }} 个耗材</span>
              <span class="info-status" :class="{ 'status-ok': selectedFilaments.length >= 2 }">
                {{ selectedFilaments.length >= 2 ? '✓ 满足要求' : '需要至少2个耗材' }}
              </span>
            </div>
          </div>
        </div>

        <!-- 生成参数配置 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">生成参数配置</h2>
            <p class="card-description">配置模型的尺寸和其他生成参数，会根据上传图片自动调整比例</p>
          </div>
          <div class="card-body">
            <div class="config-grid">
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">模型宽度</label>
                  <t-tooltip content="生成模型的目标宽度，单位为毫米。会根据模型高度自动按图片比例调整。">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="tempConfig.model_width" 
                  :step="1"
                  theme="column"
                  @change="handleWidthChange"
                >
                <template #suffix><span>mm</span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">模型高度</label>
                  <t-tooltip content="生成模型的目标高度，单位为毫米。会根据模型宽度自动按图片比例调整。">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="tempConfig.model_height" 
                  :step="1"
                  theme="column"
                  @change="handleHeightChange"
                >
                <template #suffix><span>mm</span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">模型深度</label>
                  <t-tooltip content="模型的厚度，单位为毫米。影响耗材使用量和模型的立体感。值越大，耗材使用越多，模型越厚实；值越小，耗材使用越少，模型越轻薄。建议值：0.8-2.0">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="tempConfig.model_depth" 
                  :step="0.1" 
                  :precision="1"
                  theme="column"
                >
                <template #suffix><span>mm</span></template>
                </t-input-number>
              </div>
              <div class="config-item">
                <div class="label-with-tooltip">
                  <label class="config-label">层高</label>
                  <t-tooltip content="3D打印的层高度，单位为毫米。直接影响打印精度和速度。值越小，精度越高，表面越光滑，但打印时间越长；值越大，精度越低，表面可能有层纹，但打印速度越快。建议值：0.08-0.15">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-input-number 
                  v-model="tempConfig.layer_height" 
                  :step="0.05" 
                  :precision="2"
                  theme="column"
                >
                <template #suffix><span>mm</span></template>
                </t-input-number>
              </div>
              <div class="config-item" style="grid-column: 1 / -1;">
                <div class="label-with-tooltip">
                  <label class="config-label">模型类型</label>
                  <t-tooltip content="选择生成单面模型还是双面模型。双面模型会在背面生成与正面相同的图案，适合需要双面展示的场景。">
                    <span class="info-icon">?</span>
                  </t-tooltip>
                </div>
                <t-radio-group v-model="tempConfig.is_double_sided" direction="horizontal">
                  <t-radio :value="true">双面模型</t-radio>
                  <t-radio :value="false">单面模型</t-radio>
                </t-radio-group>
              </div>
            </div>
          </div>
        </div>

        <!-- 生成模型 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">生成模型</h2>
            <p class="card-description">生成3D模型并预览结果</p>
          </div>
          <div class="card-body">
            <div class="action-section" style="display: flex; gap: 16px; margin-bottom: 24px;">
              <t-button 
                type="primary" 
                :disabled="!canGenerate"
                :loading="previewRunning"
                @click="generatePreview"
                class="action-button"
              >
                生成预览
              </t-button>
              <t-button 
                type="success" 
                :disabled="!canGenerate || !previewResult"
                :loading="generateRunning"
                @click="generateModel"
                class="action-button"
              >
                生成模型
              </t-button>
            </div>
            
            <!-- 预览结果 -->
            <div v-if="previewResult" class="comparison-section">
              <h3 class="result-title">生成结果对比</h3>
              <div class="comparison-container">
                <div class="comparison-item">
                  <h4 class="comparison-title">原图</h4>
                  <div class="tdesign-demo-image-viewer__base">
                    <t-image-viewer :images="[originalImage]" :z-index="10000"></t-image-viewer>
                  </div>
                </div>
                <div class="comparison-item">
                  <h4 class="comparison-title">预览图</h4>
                  <div class="tdesign-demo-image-viewer__base">
                    <t-image-viewer :images="[previewResult]" :z-index="10000"></t-image-viewer>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 全局样式 */
.model-generation-container {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 页面头部 */
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

/* 单一列布局 */
.content-single {
  display: flex;
  justify-content: center;
}

/* 主面板样式 */
.main-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 100%;
  max-width: 800px;
}

/* 卡片样式 */
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

/* 配置网格 */
.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 16px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-label {
  font-size: 13px;
  font-weight: 500;
  color: #333;
}

/* 带tooltip的标签样式 */
.label-with-tooltip {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

/* 圆形问号图标样式 */
.info-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: #f0f0f0;
  color: #666;
  font-size: 12px;
  font-weight: 600;
  cursor: help;
  transition: all 0.3s ease;
}

.info-icon:hover {
  background-color: #e0e0e0;
  color: #333;
}

.config-input {
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  font-size: 14px;
  transition: all 0.2s ease;
  width: 100%;
}

.config-input:focus {
  outline: none;
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

/* 确保t-input-number组件填满宽度 */
.config-item :deep(.t-input-number) {
  width: 100%;
}

/* 确保t-input组件填满宽度 */
.config-item :deep(.t-input) {
  width: 100%;
}

/* 响应式设计 - 当窗口缩小时调整配置网格列数 */
@media (max-width: 768px) {
  .config-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .config-grid {
    grid-template-columns: 1fr;
  }
}

/* 卡片操作区 */
.card-actions {
  margin-top: 24px;
  text-align: right;
}

/* 选择信息 */
.selection-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 6px;
}

.info-label {
  font-size: 13px;
  font-weight: 500;
  color: #333;
}

.info-value {
  font-size: 14px;
  color: #666;
}

.info-status {
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 12px;
  background: #ecf5ff;
  color: #409eff;
  align-self: flex-start;
  margin-top: 4px;
}

.info-status.status-ok {
  background: #f0f9eb;
  color: #67c23a;
}

/* 上传区域 */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.upload-controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

.file-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  font-size: 14px;
}

.clear-button {
  white-space: nowrap;
}

/* 预览区域 */
.preview-section {
  margin-top: 8px;
}

.preview-title {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin: 0 0 12px 0;
}

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

.preview-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

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

/* 操作区域 */
.color-count-section {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.color-count-section .label-with-tooltip {
  margin-bottom: 0;
}

.action-section {
  margin-bottom: 20px;
}

.action-button {
  width: 100%;
  padding: 10px 0;
  font-size: 16px;
  font-weight: 500;
}

/* 结果区域 */
.result-section {
  margin-top: 20px;
}

.result-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin: 0 0 16px 0;
}

/* 耗材组合面板 */
.filament-combinations {
  margin-top: 16px;
}

.combo-item {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.combo-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.combo-rank {
  margin-right: 8px;
}

.combo-score {
  font-size: 14px;
}

.combo-filaments {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.filament-tag {
  margin-right: 8px;
  margin-bottom: 8px;
}

/* 结果文本 */
.result-text {
  font-size: 16px;
  font-weight: 500;
  color: #409eff;
  margin-bottom: 16px;
}

/* 结果区域 */
.result-section {
  margin-top: 24px;
  padding: 20px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.result-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0 0 16px 0;
}

/* 对比区域 */
.comparison-section {
  margin-top: 20px;
}

.comparison-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.comparison-item {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.comparison-title {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin: 0;
  text-align: center;
}

.comparison-image-container {
  border: 1px solid #f0f0f0;
  border-radius: 6px;
  padding: 16px;
  background: #fafafa;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
}

.comparison-image {
  max-width: 100%;
  max-height: 200px;
  object-fit: contain;
}

.tdesign-demo-image-viewer__base {
  width: 100%;
  height: 200px;
  margin: 10px 0;
  border: 4px solid var(--td-bg-color-secondarycontainer);
  border-radius: var(--td-radius-medium);
}

/* 响应式设计 */
@media (max-width: 992px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
  
  .config-grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
  
  .tdesign-demo-image-viewer__base {
    height: 150px;
  }
}

@media (max-width: 768px) {
  .model-generation-container {
    padding: 16px;
  }
  
  .page-header h1 {
    font-size: 24px;
  }
  
  .card-header {
    padding: 16px 20px;
  }
  
  .tdesign-demo-image-viewer__base {
    height: 120px;
  }
  
  .card-body {
    padding: 20px;
  }
  
  .comparison-container {
    grid-template-columns: 1fr;
  }
  
  .config-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 480px) {
  .config-grid {
    grid-template-columns: 1fr;
  }
  
  .preview-container {
    height: 200px;
  }
  
  .upload-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .file-input {
    width: 100%;
  }
}
</style>