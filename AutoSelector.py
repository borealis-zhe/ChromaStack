import json
import itertools
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab, deltaE_cie76
import sys

from ChromaStackStudio import VirtualPhysics, rgb_to_lab, load_inventory

# ================= 配置 =================
INVENTORY_FILE = "my_filament.json"
INPUT_IMAGE = "path/to/your/image.png" # 你的图片路径
SLOT_COUNT = 6           # 打印机槽位数
FIXED_BASE_SLOT = "Jade White" # 强制固定的底座颜色 (通常是白色)
SAMPLE_COLORS = 200      # 从图片提取多少个特征色进行评估 (越大越准但越慢)

# ================= 辅助函数 =================

def extract_image_features(image_path, n_colors=100):
    """
    从图片中提取主要颜色和权重
    返回: (centers_lab, weights)
    """
    print(f"📷 正在分析图片颜色: {image_path}")
    img = Image.open(image_path).convert('RGBA')
    
    # 缩小图片以加速处理 (比如缩放到 100x100)
    img.thumbnail((100, 100))
    arr = np.array(img)
    
    # 去除完全透明的像素
    mask = arr[:, :, 3] > 128
    valid_pixels = arr[mask][:, :3] # 只取 RGB
    
    if len(valid_pixels) == 0:
        print("❌ 图片似乎全是透明的？")
        return None, None

    # 使用 K-Means 聚类提取代表色
    print(f"   > 正在聚类提取 {n_colors} 个特征色...")
    kmeans = MiniBatchKMeans(n_clusters=n_colors, n_init=3, batch_size=1024, random_state=42)
    kmeans.fit(valid_pixels)
    
    centers_rgb = kmeans.cluster_centers_
    
    # 计算每个聚类中心的权重（像素数量）
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=n_colors)
    weights = counts / counts.sum()
    
    # 转为 Lab 空间以便计算人眼色差
    centers_lab = rgb_to_lab(centers_rgb)
    
    return centers_lab, weights

def evaluate_combination(engine, filament_combo, target_lab, target_weights):
    """
    评估一组耗材的表现
    """
    # 1. 生成这组耗材能混出的所有颜色 (LUT)
    # 注意：这里 filament_combo 是具体的参数对象列表
    lut_rgb, _ = engine.generate_lut_km(filament_combo)
    
    # 2. 转为 Lab
    lut_lab = rgb_to_lab(lut_rgb)
    
    # 3. 计算误差
    # 对于图片中的每一个特征色，在 LUT 中找到最接近的颜色，记录误差
    # 使用广播计算欧氏距离矩阵 (N_targets, M_lut) - 注意内存，如果 LUT 很大建议用 KDTree
    # 但这里 LUT 只有 4^5=1024 个，Target 只有 100 个，直接矩阵计算很快
    
    # 简单的距离计算 (CIELAB Delta E 76)
    # diff shape: (N_targets, M_lut)
    diff = np.linalg.norm(target_lab[:, None] - lut_lab[None, :], axis=2)
    
    # 对每个目标色，找到最小误差
    min_errors = np.min(diff, axis=1)
    
    # 加权平均误差
    score = np.sum(min_errors * target_weights)
    return score

# ================= 主逻辑 =================

def auto_select_filaments():
    print("=== 🎨 自动耗材推荐系统 (Auto Filament Selector) ===")
    
    # 1. 加载库存
    inventory = load_inventory(INVENTORY_FILE)
    if not inventory: return
    
    # 找到强制固定的底座材料
    base_filament = next((f for f in inventory if f['Name'] == FIXED_BASE_SLOT), None)
    if not base_filament:
        print(f"❌ 错误: 库存中没找到固定的底座材料 '{FIXED_BASE_SLOT}'")
        return
        
    # 候选材料 (排除底座，或者也可以包含，看你是否允许底座材料出现在其他层)
    # 这里假设底座材料也可以混在中间层
    candidates = [f for f in inventory if f['Name'] != FIXED_BASE_SLOT]
    
    # 2. 分析图片
    target_lab, weights = extract_image_features(INPUT_IMAGE, n_colors=SAMPLE_COLORS)
    if target_lab is None: return

    # 3. 遍历组合
    # 我们需要选 (SLOT_COUNT - 1) 个额外的材料
    slots_to_fill = SLOT_COUNT - 1
    combinations = list(itertools.combinations(candidates, slots_to_fill))
    print(f"🔄 共有 {len(combinations)} 种耗材组合待评估...")
    
    best_score = float('inf')
    best_combo_names = []
    
    engine = VirtualPhysics()
    
    # 这里的 VirtualPhysics 需要稍微静音，不然 loop 里 print 太多
    import contextlib
    import os
    
    print("\n   [开始暴力搜索最优解]...")
    
    for i, combo in enumerate(combinations):
        # 构建完整的 4 色列表: [底座, A, B, C]
        current_selection = [base_filament] + list(combo)
        combo_names = [f['Name'] for f in current_selection]
        
        # 临时静音 generate_lut_km 的输出
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
             score = evaluate_combination(engine, current_selection, target_lab, weights)
        
        # 打印进度
        print(f"   Combo {i+1}/{len(combinations)}: {combo_names[1:]} -> 误差分: {score:.2f}")
        
        if score < best_score:
            best_score = score
            best_combo_names = combo_names

    # 4. 输出最终结果
    print("\n" + "="*40)
    print(f"🏆 最佳推荐耗材组合 (总加权色差: {best_score:.2f})")
    print("="*40)
    for idx, name in enumerate(best_combo_names):
        print(f"  Slot {idx+1}: {name}")
    print("="*40)
    print("建议：将这些名称填入主程序的 SELECTED_FILAMENT_NAMES 列表中。")

if __name__ == "__main__":
    auto_select_filaments()