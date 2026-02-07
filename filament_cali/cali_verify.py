import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os

# ================= 核心配置区域 =================

# 1. 填写您测得的 K 和 S 值 (R, G, B)
#    注意：请确保顺序对应您的耗材槽位
FILAMENT_DATA = {
    'W': {'K': [0.0000, 0.0000, 0.0000], 'S': [2.4833, 3.5147, 5.1939]},   # 假设值，或填入实测值
    'C': {'K': [4.8384, 1.2126, 0.0000], 'S': [1.0414, 2.2986, 4.3974]}, # 您的实测数据
    'M': {'K': [0.1439, 7.2885, 1.4274], 'S': [2.7596, 2.0036, 3.9949]},   # 示例数据(Red/Magenta)
    'Y': {'K': [0.0000, 0.0000, 13.9653], 'S': [4.6008, 6.0432, 1.7544]}    # 示例数据(Yellow)
}

# 2. 打印参数
LAYER_HEIGHT = 0.08  # 层高
TOTAL_LAYERS = 5     # 总层数 (例如 0.4mm = 5层)
BASE_REFLECTANCE = [0.0, 0.0, 0.0] # 黑色底座反射率 (0)

# ================= 测试矩阵设计 =================
TEST_MATRIX = [
    # Row 1: 纯色
    "WWWWW", "CCCCC", "MMMMM", "YYYYY",
    # Row 2: 两色强混合 (生成 RGB 间色)
    "CCMMM", "CCYYY", "MMYYY", "WCMYW",
    # Row 3: 复杂混合
    "CMMMY", "CCMMW", "YYCCW", "WMMCC",
    # Row 4: 亮度/灰度测试
    "WCMYCm", "WWCCM", "WWMCC", "WWYCC" # Cm 代表 C
]
STACKS = [
    # --- Row 1: 单色深度 (检测 K/S 准确性) ---
    ['C']*5, ['M']*5, ['Y']*5, ['W']*5,
    # --- Row 2: 二次色 (检测混色规律) ---
    ['C','C','M','M','M'], # Blue/Violet
    ['C','C','Y','Y','Y'], # Green
    ['M','M','Y','Y','Y'], # Red/Orange
    ['C','M','Y','W','W'], # Grey/Dirty
    # --- Row 3: 比例测试 (Cyan vs Magenta) ---
    ['C','C','C','C','M'], # 主要是C，带点M
    ['C','C','M','M','W'], # C+M+白冲淡
    ['C','M','M','M','M'], # 主要是M，带点C
    ['C','M','Y','C','M'], # 乱序混合
    # --- Row 4: 高亮测试 (加白) ---
    ['W','W','W','C','C'], # 浅蓝
    ['W','W','W','M','M'], # 浅红
    ['W','W','W','Y','Y'], # 浅黄
    ['W','W','W','C','M'], # 浅紫
]

# ================= K-M 多层物理引擎 =================

def get_layer_optical_properties(K, S, h):
    """
    计算单层材料的 R (反射率) 和 T (透射率)
    基于 Kubelka-Munk 理论的一般解
    """
    K = np.array(K)
    S = np.array(S)
    
    # 避免除零
    S = np.maximum(S, 1e-6)
    K = np.maximum(K, 1e-9)

    a = 1 + (K / S)
    b = np.sqrt(a**2 - 1)
    bSh = b * S * h
    
    sinh_bSh = np.sinh(bSh)
    cosh_bSh = np.cosh(bSh)
    
    # 单层反射率 R (在黑底上的反射率)
    # R = sinh(bSh) / (a sinh(bSh) + b cosh(bSh))
    R = sinh_bSh / (a * sinh_bSh + b * cosh_bSh)
    
    # 单层透射率 T
    # T = b / (a sinh(bSh) + b cosh(bSh))
    T = b / (a * sinh_bSh + b * cosh_bSh)
    
    return R, T

def calculate_composite_stack(stack_codes):
    """
    递归计算多层堆叠的最终颜色
    stack_codes: 从底到顶的列表，如 ['C', 'C', 'M']
    """
    # 初始背景：底座
    current_R = np.array(BASE_REFLECTANCE)
    
    for code in stack_codes:
        if code not in FILAMENT_DATA:
            continue
            
        params = FILAMENT_DATA[code]
        # 计算这一层的光学属性
        R_layer, T_layer = get_layer_optical_properties(params['K'], params['S'], LAYER_HEIGHT)
        
        # K-M 多层叠加公式 (Layer Composition)
        # R_new = R_layer + (T_layer^2 * R_bg) / (1 - R_layer * R_bg)
        denom = 1.0 - R_layer * current_R
        # 避免分母为0
        denom = np.maximum(denom, 1e-6)
        
        current_R = R_layer + (T_layer**2 * current_R) / denom
        
    return np.clip(current_R, 0, 1)

# ================= 1. 生成屏幕预览 =================

def generate_validation_preview():
    print("🎨 正在计算 4x4 混色矩阵...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 4x4 Grid
    grid_img = np.zeros((4, 4, 3))
    
    for i, stack in enumerate(STACKS):
        row = i // 4
        col = i % 4
        
        # 计算颜色
        rgb_linear = calculate_composite_stack(stack)
        # Gamma 校正用于显示
        rgb_srgb = rgb_linear ** (1/2.2)
        
        grid_img[row, col] = rgb_srgb
        
        # 在格子上标注堆叠代码
        label = "".join([c[0] for c in stack]) # 简写
        # 字体颜色根据背景亮度自动调整
        lum = 0.2126*rgb_srgb[0] + 0.7152*rgb_srgb[1] + 0.0722*rgb_srgb[2]
        text_color = 'black' if lum > 0.5 else 'white'
        
        ax.text(col, row, label, ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

    ax.imshow(grid_img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Mixing Validation Prediction\n(Layer Height: {LAYER_HEIGHT}mm, {TOTAL_LAYERS} Layers)")
    
    plt.tight_layout()
    plt.savefig("mix_prediction.png", dpi=300)
    print("✅ 预测图已保存: mix_prediction.png")

# ================= 2. 生成打印模型 (STL) =================

def generate_validation_stls():
    print("🔨 正在生成多色 STL...")
    
    BLOCK_SIZE = 20.0
    GAP = 0.0

    meshes = {
        'C': [], 'M': [], 'Y': [], 'W': []
    }
    # 底座 
    base_mesh = []
    
    # 遍历 4x4 矩阵
    for i, stack in enumerate(STACKS):
        row = i // 4
        col = i % 4
        
        # 物理位置
        x_base = col * (BLOCK_SIZE + GAP)
        y_base = (3 - row) * (BLOCK_SIZE + GAP) # 让Row0在上面
        
        # 1. 生成黑色底座 (0.6mm 厚)
        base_block = trimesh.creation.box(extents=[BLOCK_SIZE, BLOCK_SIZE, 0.6])
        base_block.apply_translation([x_base + BLOCK_SIZE/2, y_base + BLOCK_SIZE/2, 0.3])
        base_mesh.append(base_block)
        
        # 2. 生成 5 层堆叠
        for layer_idx, code in enumerate(stack):
            if code not in meshes: continue
            
            # 生成一层薄片
            layer_z = 0.6 + layer_idx * LAYER_HEIGHT + (LAYER_HEIGHT/2)
            voxel = trimesh.creation.box(extents=[BLOCK_SIZE, BLOCK_SIZE, LAYER_HEIGHT])
            voxel.apply_translation([x_base + BLOCK_SIZE/2, y_base + BLOCK_SIZE/2, layer_z])
            
            meshes[code].append(voxel)

    # 导出文件
    output_dir = "validation_stls"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 导出底座
    if base_mesh:
        combined_base = trimesh.util.concatenate(base_mesh)
        combined_base.export(os.path.join(output_dir, "00_Base_Black.stl"))
        print(f"  -> {output_dir}/00_Base_Black.stl")

    # 导出各色层
    for color_code, mesh_list in meshes.items():
        if mesh_list:
            combined = trimesh.util.concatenate(mesh_list)
            filename = os.path.join(output_dir, f"01_Color_{color_code}.stl")
            combined.export(filename)
            print(f"  -> {filename}")

if __name__ == "__main__":
    generate_validation_preview()
    generate_validation_stls()
    print("\n💡 下一步:")
    print("1. 在切片软件中加载所有 STL。")
    print("2. 黑色底座用黑色耗材，Color_C/M/Y/W 分别指派对应的耗材。")
    print("3. 打印后，对比实物与 'mix_prediction.png'。")
    print("   重点观察第二行(混色)和第三行(复杂混合)是否一致。")