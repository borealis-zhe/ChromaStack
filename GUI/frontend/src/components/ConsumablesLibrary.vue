<script setup lang="jsx">
import { ref, reactive, onMounted } from 'vue'
import { MessagePlugin } from 'tdesign-vue-next'

// 耗材列表数据
const filaments = ref([])
// 加载状态
const loading = ref(false)
// 对话框状态
const isAddDialogVisible = ref(false)
const isEditDialogVisible = ref(false)
const isDeleteDialogVisible = ref(false)
// 当前操作的耗材
const currentFilament = ref(null)
// 表单数据
const formData = reactive({
  Name: '',
  Type: 'PLA',
  FILAMENT_K: '[0, 0, 0]',
  FILAMENT_S: '[0, 0, 0]'
})

// 表格分页配置
const pagination = {
  total: 0,
  pageSize: 10,
  current: 1,
  showJumper: true,
  showSizeChanger: true,
  pageSizeOptions: ['10', '20', '50', '100']
}

// 表格列配置
const columns = [
  { title: '名称', colKey: 'Name', width: 150, align: 'left' },
  { title: '类型', colKey: 'Type', width: 100, align: 'left' },
  { 
    title: 'FILAMENT_K', 
    colKey: 'FILAMENT_K', 
    width: 200,
    align: 'left'
  },
  { 
    title: 'FILAMENT_S', 
    colKey: 'FILAMENT_S', 
    width: 200,
    align: 'left'
  },
  { 
    title: '操作', 
    colKey: 'actions', 
    width: 180,
    align: 'left',
    fixed: 'right',
  }
]

// 获取耗材列表
const fetchFilaments = async () => {
  loading.value = true
  try {
    const response = await fetch('http://localhost:5000/filaments')
    const data = await response.json()
    let filamentList = []
    
    if (data.filaments) {
      filamentList = data.filaments
    } else if (data.error) {
      alert(`获取耗材失败: ${data.error || '未知错误'}`)
    } else {
      filamentList = []
    }
    
    // 处理数据，确保FILAMENT_K和FILAMENT_S被转换为字符串
    filaments.value = filamentList.map(filament => {
      return {
        ...filament,
        FILAMENT_K: JSON.stringify(filament.FILAMENT_K),
        FILAMENT_S: JSON.stringify(filament.FILAMENT_S)
      }
    })
    
    // 更新分页总数
    pagination.total = filaments.value.length
  } catch (error) {
    alert(`获取耗材列表失败: ${error.message}`)
  } finally {
    loading.value = false
  }
}

// 打开新增对话框
const openAddDialog = () => {
  // 重置表单
  formData.Name = ''
  formData.Type = 'PLA'
  formData.FILAMENT_K = '[0, 0, 0]'
  formData.FILAMENT_S = '[0, 0, 0]'
  isAddDialogVisible.value = true
}

// 打开修改对话框
const openEditDialog = (filament) => {
  // 填充表单数据
  currentFilament.value = filament
  formData.Name = filament.Name
  formData.Type = filament.Type
  // 直接使用字符串表示，不使用JSON.stringify，避免添加引号
  formData.FILAMENT_K = `${filament.FILAMENT_K}`
  formData.FILAMENT_S = `${filament.FILAMENT_S}`
  isEditDialogVisible.value = true
}

// 打开删除对话框
const openDeleteDialog = (filament) => {
  currentFilament.value = filament
  isDeleteDialogVisible.value = true
}

// 关闭所有对话框
const closeAllDialogs = () => {
  isAddDialogVisible.value = false
  isEditDialogVisible.value = false
  isDeleteDialogVisible.value = false
  currentFilament.value = null
}

// 检查名称唯一性
const checkNameUnique = async (name) => {
  try {
    const response = await fetch(`http://localhost:5000/filaments/check_name/${encodeURIComponent(name)}`)
    const data = await response.json()
    return data.success && data.is_unique
  } catch (error) {
    console.error('检查名称唯一性失败:', error)
    return false
  }
}

// 新增耗材
const addFilament = async () => {
  // 简单验证
  if (!formData.Name.trim()) {
    alert('请输入耗材名称')
    return
  }
  
  // 检查名称唯一性
  const isUnique = await checkNameUnique(formData.Name)
  if (!isUnique) {
    alert('耗材名称已存在')
    return
  }
  
  try {
    // 验证并转换FILAMENT_K和FILAMENT_S为数组
    let parsedK, parsedS
    try {
      parsedK = JSON.parse(formData.FILAMENT_K)
      parsedS = JSON.parse(formData.FILAMENT_S)
      
      if (!Array.isArray(parsedK) || !Array.isArray(parsedS)) {
        throw new Error('FILAMENT_K和FILAMENT_S必须是数组格式')
      }
    } catch (parseError) {
      alert(`数据格式错误: ${parseError.message}`)
      return
    }
    
    // 准备提交数据
    const submitData = {
      ...formData,
      FILAMENT_K: parsedK,
      FILAMENT_S: parsedS
    }
    
    const response = await fetch('http://localhost:5000/filaments', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(submitData)
    })
    
    const data = await response.json()
    if (data.success) {
      // 关闭对话框
      closeAllDialogs()
      // 刷新列表
      fetchFilaments()
    } else {
      throw new Error(data.error || '新增失败')
    }
  } catch (error) {
    console.error('新增耗材失败:', error)
    alert(`新增失败: ${error.message}`)
  }
}

// 修改耗材
const updateFilament = async () => {
  // 简单验证
  if (!formData.Name.trim()) {
    alert('请输入耗材名称')
    return
  }
  
  // 检查名称唯一性（如果名称改变）
  if (formData.Name !== currentFilament.value.Name) {
    const isUnique = await checkNameUnique(formData.Name)
    if (!isUnique) {
      alert('耗材名称已存在')
      return
    }
  }
  
  try {
    // 验证并转换FILAMENT_K和FILAMENT_S为数组
    let parsedK, parsedS
    try {
      parsedK = JSON.parse(formData.FILAMENT_K)
      parsedS = JSON.parse(formData.FILAMENT_S)
      
      if (!Array.isArray(parsedK) || !Array.isArray(parsedS)) {
        throw new Error('FILAMENT_K和FILAMENT_S必须是数组格式')
      }
    } catch (parseError) {
      alert(`数据格式错误: ${parseError.message}`)
      return
    }
    
    // 准备提交数据
    const submitData = {
      ...formData,
      FILAMENT_K: parsedK,
      FILAMENT_S: parsedS
    }
    
    const response = await fetch(`http://localhost:5000/filaments/${encodeURIComponent(currentFilament.value.Name)}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(submitData)
    })
    
    const data = await response.json()
    if (data.success) {
      // 关闭对话框
      closeAllDialogs()
      // 刷新列表
      fetchFilaments()
    } else {
      throw new Error(data.error || '修改失败')
    }
  } catch (error) {
    console.error('修改耗材失败:', error)
    alert(`修改失败: ${error.message}`)
  }
}

// 删除耗材
const deleteFilament = async () => {
  try {
    const response = await fetch(`http://localhost:5000/filaments/${encodeURIComponent(currentFilament.value.Name)}`, {
      method: 'DELETE'
    })
    
    const data = await response.json()
    if (data.success) {
      // 关闭对话框
      closeAllDialogs()
      // 刷新列表
      fetchFilaments()
    } else {
      throw new Error(data.error || '删除失败')
    }
  } catch (error) {
    console.error('删除耗材失败:', error)
    alert(`删除失败: ${error.message}`)
  }
}

// 组件挂载时获取耗材列表
onMounted(() => {
  fetchFilaments()
})
</script>

<template>
  <div class="consumables-library-container">
    <div class="page-header">
      <h1>耗材库</h1>
      <p class="page-description">管理3D打印耗材，添加、修改和删除耗材信息</p>
    </div>

    <!-- 主内容区域 -->
    <div class="content-single">
      <div class="main-panel">
        <!-- 耗材库操作区域 -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">耗材管理</h2>
            <p class="card-description">添加、修改和删除3D打印耗材</p>
          </div>
          <div class="card-body">
            <div class="card-actions">
              <t-button type="primary" @click="openAddDialog">添加耗材</t-button>
              <t-button @click="fetchFilaments" :loading="loading">刷新列表</t-button>
            </div>
            
            <!-- 耗材列表表格 -->
            <div>
              <t-table
                :data="filaments"
                :columns="columns"
                :pagination="pagination"
                :loading="loading"
                size="small"
                row-key="Name"
              >
                <template #actions="{ row }">
                  <div class="table-operations" style="display: flex; gap: 8px;">
                    <t-link theme="primary" hover="color" @click="openEditDialog(row)">
                      修改
                    </t-link>
                    <span style="color: #d9d9d9;">|</span>
                    <t-link theme="danger" hover="color" @click="openDeleteDialog(row)">
                      删除
                    </t-link>
                  </div>
                </template>
              </t-table>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 新增/修改对话框 -->
    <!-- 新增/修改耗材对话框 -->
    <t-dialog
      :visible="isAddDialogVisible || isEditDialogVisible"
      :title="isAddDialogVisible ? '新增耗材' : '修改耗材'"
      @close="closeAllDialogs"
      width="650px"
      :header-style="{ fontSize: '18px', fontWeight: '600' }"
      :body-style="{ padding: '24px' }"
    >
      <t-form 
        class="t-form-layout" 
        :label-width="120"
        style="max-width: 100%;"
      >
        <t-form-item label="耗材名称" required>
          <t-input 
            v-model="formData.Name" 
            placeholder="请输入耗材名称"
            maxlength="50"
            :style="{ width: '100%', maxWidth: '400px' }"
          />
        </t-form-item>
        <t-form-item label="耗材类型" required>
          <t-select 
            v-model="formData.Type" 
            placeholder="请选择耗材类型"
            :style="{ width: '100%', maxWidth: '400px' }"
          >
            <t-option value="PLA" label="PLA" />
            <t-option value="PETG" label="PETG" />
          </t-select>
        </t-form-item>
        <t-form-item label="FILAMENT_K" required>
          <t-input 
            v-model="formData.FILAMENT_K" 
            placeholder="请输入FILAMENT_K数组，例如: [0.1, 0.2, 0.3]"
            :style="{ width: '100%', maxWidth: '400px' }"
          />
        </t-form-item>
        <t-form-item label="FILAMENT_S" required>
          <t-input 
            v-model="formData.FILAMENT_S" 
            placeholder="请输入FILAMENT_S数组，例如: [0.4, 0.5, 0.6]"
            :style="{ width: '100%', maxWidth: '400px' }"
          />
        </t-form-item>
      </t-form>
      <template #footer>
        <div style="display: flex; gap: 16px; justify-content: flex-end; padding: '16px 24px 24px';">
          <t-button @click="closeAllDialogs" :style="{ padding: '6px 20px', fontSize: '14px' }">取消</t-button>
          <t-button 
            type="primary" 
            @click="isAddDialogVisible ? addFilament() : updateFilament()"
            :style="{ padding: '6px 20px', fontSize: '14px' }"
          >
            确认
          </t-button>
        </div>
      </template>
    </t-dialog>
    
    <!-- 删除确认对话框 -->
    <t-dialog
      :visible="isDeleteDialogVisible"
      title="删除确认"
      @close="closeAllDialogs"
      width="350px"
      :header-style="{ fontSize: '18px', fontWeight: '600' }"
      :body-style="{ padding: '24px' }"
    >
      <div style="padding: 20px 0; font-size: '14px'; line-height: '1.5';">
        <p style="margin: '0 0 12px 0';">确定要删除耗材 <strong>{{ currentFilament?.Name }}</strong> 吗？</p>
        <p style="color: #ff4d4f; margin: '0'; font-weight: '500';">此操作不可恢复，请谨慎执行。</p>
      </div>
      <template #footer>
        <div style="display: flex; gap: 16px; justify-content: flex-end;">
          <t-button @click="closeAllDialogs" :style="{ padding: '6px 20px', fontSize: '14px' }">取消</t-button>
          <t-button 
            type="danger" 
            @click="deleteFilament()"
            :style="{ padding: '6px 20px', fontSize: '14px' }"
          >
            确认删除
          </t-button>
        </div>
      </template>
    </t-dialog>
  </div>
</template>

<style scoped>
/* 全局样式 - 匹配模型生成页面 */
.consumables-library-container {
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

/* 表格容器 - 匹配模型生成页面 */
.table-container {
  margin-top: 20px;
  overflow-x: auto;
  border-radius: 8px;
  background: #fafafa;
  padding: 20px;
  border: 1px solid #f0f0f0;
}

/* 表格样式 - 匹配模型生成页面 */
.simple-table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.simple-table th,
.simple-table td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

.simple-table th {
  background-color: #fafafa;
  font-weight: 600;
  color: #333;
  font-size: 14px;
  white-space: nowrap;
}

.simple-table tr:hover {
  background-color: #fafafa;
}

/* 操作按钮样式 - 匹配模型生成页面 */
.action-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.action-buttons :deep(.t-button) {
  margin-bottom: 4px;
  padding: 4px 12px;
  font-size: 13px;
}

/* 卡片操作区 - 匹配模型生成页面 */
.card-actions {
  margin-bottom: 20px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

/* 对话框样式 - 匹配模型生成页面 */
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
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
  width: 700px;
  max-width: 90%;
  max-height: 90vh;
  overflow-y: auto;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid #f0f0f0;
}

.dialog-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1a1a1a;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #666;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.3s;
}

.close-btn:hover {
  background-color: #f0f0f0;
  color: #333;
}

.dialog-content {
  padding: 24px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 20px 24px;
  border-top: 1px solid #f0f0f0;
}

/* 表单样式 - 匹配模型生成页面 */
.simple-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
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
  display: flex;
  align-items: center;
  gap: 6px;
}

.form-item input,
.form-item select {
  padding: 10px 16px;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s ease;
  width: 100%;
}

.form-item input:focus,
.form-item select:focus {
  outline: none;
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

.array-inputs {
  display: flex;
  gap: 12px;
}

.array-inputs input {
  flex: 1;
}

/* 按钮样式 - 匹配模型生成页面 */
.cancel-btn,
.confirm-btn,
.delete-confirm-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
  min-width: 80px;
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
  background-color: #409eff;
  color: white;
}

.confirm-btn:hover {
  background-color: #66b1ff;
}

.delete-confirm-btn {
  background-color: #f56c6c;
  color: white;
}

.delete-confirm-btn:hover {
  background-color: #f78989;
}

/* 响应式设计 - 匹配模型生成页面 */
@media (max-width: 992px) {
  .main-panel {
    max-width: 100%;
  }
}

@media (max-width: 768px) {
  .consumables-library-container {
    padding: 16px;
  }
  
  .page-header h1 {
    font-size: 24px;
  }
  
  .card-header,
  .card-body {
    padding: 16px;
  }
  
  .dialog {
    width: 95%;
  }
  
  .dialog-header,
  .dialog-content,
  .dialog-footer {
    padding: 16px;
  }
}
</style>