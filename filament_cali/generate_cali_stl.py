import numpy as np
import trimesh
import os

# ================= 配置参数 =================
# 打印层高 (必须与你实际打印参数一致)
LAYER_HEIGHT = 0.08 

# 底座参数
BASE_THICKNESS = 0.6       # 底座厚度 (确保不透光)
BASE_WIDTH = 20.0          # 每个底座(黑/白)的宽度
CHIP_LENGTH_PER_STEP = 10.0 # 每个阶梯的长度

# 测试阶梯参数
NUM_STEPS = 5              # 测试 1~5 层
START_LAYERS = 1           # 从第几层开始测

# 输出目录
OUTPUT_DIR = "calibration_stls"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 几何生成函数 =================

def create_block(x, y, z, w, l, h):
    """创建立方体: 原点在(x,y,z)，尺寸(w,l,h)"""
    box = trimesh.creation.box(extents=[w, l, h])
    # box 默认中心在 0,0,0，我们需要移动到指定位置
    # 目标中心点:
    cx = x + w/2
    cy = y + l/2
    cz = z + h/2
    box.apply_translation([cx, cy, cz])
    return box

def generate_calibration_kit():
    print(f"正在生成黑白对比校准卡 (层高: {LAYER_HEIGHT}mm)...")
    
    total_length = NUM_STEPS * CHIP_LENGTH_PER_STEP
    
    # 1. 生成黑色底座 (左半边)
    # 位置: x=0, y=0, z=0
    mesh_base_black = create_block(
        x=0, y=0, z=0,
        w=BASE_WIDTH, l=total_length, h=BASE_THICKNESS
    )
    
    # 2. 生成白色底座 (右半边)
    # 位置: x=BASE_WIDTH, y=0, z=0
    mesh_base_white = create_block(
        x=BASE_WIDTH, y=0, z=0,
        w=BASE_WIDTH, l=total_length, h=BASE_THICKNESS
    )
    
    # 3. 生成待测颜色的阶梯 (覆盖在上方)
    # 它横跨黑色和白色底座，宽度 = 2 * BASE_WIDTH
    steps = []
    
    for i in range(NUM_STEPS):
        layers = START_LAYERS + i
        step_thickness = layers * LAYER_HEIGHT
        
        # y轴位置递增
        y_pos = i * CHIP_LENGTH_PER_STEP
        
        # Z轴起始位置：必须紧贴底座上方
        z_pos = BASE_THICKNESS
        
        step_mesh = create_block(
            x=0, y=y_pos, z=z_pos,
            w=BASE_WIDTH * 2,     # 跨越两边
            l=CHIP_LENGTH_PER_STEP,
            h=step_thickness
        )
        steps.append(step_mesh)
        print(f"  - 阶梯 {i+1}: {layers} 层 ({step_thickness:.2f}mm)")
        
    # 合并所有台阶为一个对象
    mesh_steps = trimesh.util.concatenate(steps)
    
    # ================= 导出 =================
    
    # 导出黑色底座
    path_black = os.path.join(OUTPUT_DIR, "01_Base_Black.stl")
    mesh_base_black.export(path_black)
    print(f"✅ 生成: {path_black}")
    
    # 导出白色底座
    path_white = os.path.join(OUTPUT_DIR, "02_Base_White.stl")
    mesh_base_white.export(path_white)
    print(f"✅ 生成: {path_white}")
    
    # 导出测试阶梯
    path_steps = os.path.join(OUTPUT_DIR, "03_Target_Color_Steps.stl")
    mesh_steps.export(path_steps)
    print(f"✅ 生成: {path_steps}")
    
    print("\n💡 使用说明:")
    print("1. 在切片软件中同时加载这 3 个 STL。")
    print("2. 确认提示'作为单一对象的多部分加载?' -> 选择 [是/Yes]。")
    print("3. 设置耗材映射:")
    print("   - 01_Base_Black -> 指定为黑色耗材")
    print("   - 02_Base_White -> 指定为白色耗材")
    print("   - 03_Target_Color -> 指定为你想要测试的耗材(如 Cyan)")
    print("4. 打印后，你将得到一个包含 5 级厚度的样片，")
    print("   每级厚度都有'黑底'和'白底'两种表现。")

if __name__ == "__main__":
    generate_calibration_kit()