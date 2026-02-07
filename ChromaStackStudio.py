import numpy as np
import trimesh
import itertools
import json
import os
import sys
import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from PIL import Image
import scipy.ndimage as ndimage
from skimage.segmentation import felzenszwalb
from shapely.geometry import Polygon
import cv2

# ================= 配置区域 =================
# 打印物理参数
LAYER_HEIGHT = 0.08      # 颜色层层高 (mm)
TOTAL_LAYERS = 5         # 混色层数
PIXEL_SIZE = 0.2         # 像素尺寸/水平分辨率 (mm)
BASE_HEIGHT = 0.8        # 白色底座厚度 (mm)
ALPHA_THRESHOLD = 128    # PNG透明度阈值 (0-255)，低于此值视为透明不打印

# K-M 理论边界条件
BACKING_REFLECTANCE = np.array([0.94, 0.94, 0.94]) # 底座(白色PLA)的反射率

# 文件路径 (请修改为你的带透明背景的PNG)
INPUT_IMAGE = "testpic/wp004.webp"
INVENTORY_FILE = "my_filament.json"
TARGET_WIDTH_MM = 80

# 选料配置 (Slot 1 必须是底座材料，如白色)
SELECTED_FILAMENT_NAMES = [
    "Jade White",       # Slot 1 (Base Layer)
    # "White",
    "Black",             # Slot 2
    "Sunflower Yellow",  # Slot 3
    # "Red",               # Slot 4
    "Cyan",              # Slot 5
    "Magenta",            # Slot 6
    "Creativity Blue",     # Slot 7
    "Special Gray",         # Slot 8
    "Brown"
]

# ================= 1. 数据加载模块 =================

def load_inventory(json_path):
    print(f"正在读取耗材库: {json_path} ...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "Filaments" in data:
            inventory = data["Filaments"]
        elif isinstance(data, list):
            inventory = data
        else:
            inventory = []
        print(f"  > 成功加载 {len(inventory)} 种耗材数据")
        return inventory
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return []

def get_selected_filaments(inventory, target_names):
    selected = []
    name_map = {item['Name']: item for item in inventory}
    print("正在匹配并校验选定的耗材...")
    for name in target_names:
        if name in name_map:
            item = name_map[name]
            if "FILAMENT_K" not in item or "FILAMENT_S" not in item:
                print(f"  [!] 警告: '{name}' 缺少 K/S 参数，将使用默认值！")
                item["FILAMENT_K"] = [0.1, 0.1, 0.1]
                item["FILAMENT_S"] = [5.0, 5.0, 5.0]
            selected.append(item)
        else:
            print(f"  [❌] 错误: 库存中找不到名为 '{name}' 的耗材！")
            sys.exit(1)
    return selected

# ================= 2. 物理核心引擎 & 颜色转换工具 =================

class VirtualPhysics:
    @staticmethod
    def linear_to_srgb_bytes(linear):
        linear = np.clip(linear, 0, 1)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)
        return (srgb * 255).astype(np.uint8)

    @staticmethod
    def km_reflectance_vectorized(K, S, h, Rg):
        S = np.maximum(S, 1e-6)
        a = 1 + (K / S)
        b = np.sqrt(np.maximum(a**2 - 1, 1e-9))
        bSh = b * S * h
        sinh_bSh = np.sinh(bSh)
        cosh_bSh = np.cosh(bSh)
        numerator = sinh_bSh * (1 - Rg * a) + Rg * b * cosh_bSh
        denominator = sinh_bSh * (a - Rg) + b * cosh_bSh
        denominator = np.maximum(denominator, 1e-6)
        R = numerator / denominator
        return np.clip(R, 0, 1)

    def generate_lut_km(self, filaments_list):
        num_filaments = len(filaments_list) # <--- 获取动态数量
        print(f"🤖 [K-M 引擎] 检测到 {num_filaments} 种耗材，正在计算光路混合...")
        
        Ks = np.array([f['FILAMENT_K'] for f in filaments_list])
        Ss = np.array([f['FILAMENT_S'] for f in filaments_list])
        
        # ⚠️ 动态生成组合索引：range(num_filaments) 而不是 range(4)
        indices = np.array(list(itertools.product(range(num_filaments), repeat=TOTAL_LAYERS)))
        num_combos = len(indices)
        
        print(f"  > 组合总数: {num_filaments}^{TOTAL_LAYERS} = {num_combos}")
        
        current_R = np.tile(BACKING_REFLECTANCE, (num_combos, 1))
        for layer_idx in range(TOTAL_LAYERS):
            filament_ids = indices[:, layer_idx] 
            layer_K = Ks[filament_ids]
            layer_S = Ss[filament_ids]
            current_R = self.km_reflectance_vectorized(layer_K, layer_S, LAYER_HEIGHT, current_R)
        
        lut_colors_srgb = self.linear_to_srgb_bytes(current_R)
        return lut_colors_srgb, indices

def rgb_to_lab(rgb):
    """
    将 sRGB (0-255) 转换为 CIELAB 颜色空间 (D65)。
    输入: numpy array (N, 3) 范围 0-255
    输出: numpy array (N, 3) Lab 值
    """
    # 1. 归一化到 0-1
    rgb = rgb.astype(float) / 255.0

    # 2. sRGB -> Linear RGB (反 Gamma 校正)
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # 3. Linear RGB -> XYZ (D65)
    # 转换矩阵
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = np.dot(rgb, M.T)

    # 4. XYZ -> Lab
    # D65 参考白点
    XYZ_ref = np.array([0.95047, 1.00000, 1.08883])
    XYZ = XYZ / XYZ_ref

    mask = XYZ > 0.008856
    f_XYZ = np.zeros_like(XYZ)
    f_XYZ[mask] = XYZ[mask] ** (1.0/3.0)
    f_XYZ[~mask] = 7.787 * XYZ[~mask] + 16.0/116.0

    Lab = np.zeros_like(XYZ)
    Lab[:, 0] = 116.0 * f_XYZ[:, 1] - 16.0       # L
    Lab[:, 1] = 500.0 * (f_XYZ[:, 0] - f_XYZ[:, 1]) # a
    Lab[:, 2] = 200.0 * (f_XYZ[:, 1] - f_XYZ[:, 2]) # b

    return Lab

def visualize_gamut(lut_colors):
    print("\n📊 正在生成色域预览图...")
    colors_norm = lut_colors / 255.0
    
    # 1. 3D 散点图 (保持不变)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    # 为了防止点太多导致卡顿，如果点超过 5000 个，随机采样显示
    if len(colors_norm) > 5000:
        indices = np.random.choice(len(colors_norm), 5000, replace=False)
        show_colors = colors_norm[indices]
    else:
        show_colors = colors_norm
        
    ax1.scatter(show_colors[:,0], show_colors[:,1], show_colors[:,2], c=show_colors, s=20)
    ax1.set_title(f'RGB Space Distribution ({len(lut_colors)} colors)')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)

    # 2. 2D 色板图 (核心修复)
    ax2 = fig.add_subplot(122)
    
    # 按色相排序
    def get_hsv(rgb): return colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    sorted_indices = sorted(range(len(colors_norm)), key=lambda k: get_hsv(colors_norm[k]))
    sorted_colors = colors_norm[sorted_indices]

    # --- 动态计算网格大小 ---
    num_colors = len(sorted_colors)
    side_len = int(np.ceil(np.sqrt(num_colors))) # 计算最小的正方形边长
    target_size = side_len * side_len
    
    # 如果颜色数量填不满正方形，用白色填充剩余部分 zeros是黑色
    if target_size > num_colors:
        padding = np.ones((target_size - num_colors, 3)) # 黑色填充
        sorted_colors_padded = np.vstack([sorted_colors, padding])
    else:
        sorted_colors_padded = sorted_colors

    # Reshape 为动态计算出的边长
    grid_img = sorted_colors_padded.reshape(side_len, side_len, 3)
    # ---------------------

    ax2.imshow(grid_img)
    ax2.set_title(f'Available Palette\nSorted by Hue (Grid: {side_len}x{side_len})')
    ax2.axis('off')
    
    plt.tight_layout()
    try:
        plt.show(block=False)
    except:
        pass # 防止无 GUI 环境报错
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig("debug_output/gamut_check.png")
    print("📈 色域图已保存为 debug_output/gamut_check.png")

# ================= 3. 几何生成引擎 =================

def _generate_voxels_from_centers(centers, dx, dy, dz):
    if len(centers) == 0:
        return trimesh.Trimesh()
    num_voxels = len(centers)
    base_verts = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
    ])
    base_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5], 
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], 
        [2, 6, 7], [2, 7, 3], [4, 0, 3], [4, 3, 7]
    ])
    all_verts = centers[:, np.newaxis, :] + base_verts[np.newaxis, :, :]
    all_verts = all_verts.reshape(-1, 3)
    vertex_offsets = np.arange(num_voxels) * 8
    all_faces = base_faces[np.newaxis, :, :] + vertex_offsets[:, np.newaxis, np.newaxis]
    all_faces = all_faces.reshape(-1, 3)
    return trimesh.Trimesh(vertices=all_verts, faces=all_faces)

def create_voxel_mesh_masked(indices_matrix, slot_id, width_pixels, height_pixels, solid_mask_2d, z_offset=0.0, is_base_layer=False):
    """
    [修复版] 
    1. 解决了 trimesh.load_path 不接受列表的报错。
    2. 增加了孔洞处理 (RETR_CCOMP)，防止 'O' 型图案中间被填实。
    """
    meshes_to_combine = []

    # 辅助函数：将 OpenCV 轮廓坐标转换为物理坐标
    def convert_contour_to_points(cnt):
        # cnt shape: (N, 1, 2) -> (N, 2)
        pts = cnt.reshape(-1, 2)
        physical_pts = np.zeros_like(pts, dtype=float)
        # X轴转换 (注意：Main函数里可能已经做过镜像，这里只负责缩放)
        physical_pts[:, 0] = pts[:, 0] * PIXEL_SIZE
        # Y轴转换 (OpenCV原点在左上，3D打印在左下，需要翻转Y)
        physical_pts[:, 1] = (height_pixels - 1 - pts[:, 1]) * PIXEL_SIZE
        return physical_pts

    # 待处理的任务列表：(Layer_Index, Mask)
    tasks = []
    
    if is_base_layer and slot_id == 0:
        # 场景 A: 白色底座 (单层厚度 = BASE_HEIGHT)
        layer_mask_u8 = (solid_mask_2d.astype(np.uint8)) * 255
        tasks.append({
            "mask": layer_mask_u8, 
            "height": BASE_HEIGHT, 
            "z_start": z_offset
        })
        
    elif not is_base_layer:
        # 场景 B: 彩色层 (逐层切片, 单层厚度 = LAYER_HEIGHT)
        for layer_idx in range(TOTAL_LAYERS):
            current_layer_slots = indices_matrix[:, :, layer_idx]
            layer_mask = (current_layer_slots == slot_id) & solid_mask_2d
            if np.any(layer_mask):
                tasks.append({
                    "mask": layer_mask.astype(np.uint8) * 255,
                    "height": LAYER_HEIGHT,
                    "z_start": z_offset + layer_idx * LAYER_HEIGHT
                })

    # --- 核心处理循环 ---
    for task in tasks:
        mask_u8 = task["mask"]
        extrude_h = task["height"]
        z_pos = task["z_start"]

        # 1. 查找轮廓 (使用 RETR_CCOMP 以支持孔洞层级)
        # contours: 轮廓点列表
        # hierarchy: [Next, Previous, First_Child, Parent]
        contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and hierarchy is not None:
            hierarchy = hierarchy[0] # 降维 (1, N, 4) -> (N, 4)
            
            for i, cnt in enumerate(contours):
                # hierarchy[i][3] 是 Parent Index。如果为 -1，说明它是最外层轮廓 (Shell)
                if hierarchy[i][3] == -1:
                    # 1. 构建外壳 (Shell)
                    shell_pts = convert_contour_to_points(cnt)
                    if len(shell_pts) < 3: continue # 忽略噪点
                    
                    # 2. 寻找属于它的孔洞 (Holes)
                    holes_pts_list = []
                    child_idx = hierarchy[i][2] # First Child
                    while child_idx != -1:
                        hole_cnt = contours[child_idx]
                        if len(hole_cnt) >= 3:
                            holes_pts_list.append(convert_contour_to_points(hole_cnt))
                        child_idx = hierarchy[child_idx][0] # Next Sibling (同级孔洞)

                    # 3. 创建 Shapely 多边形
                    try:
                        raw_poly = Polygon(shell=shell_pts, holes=holes_pts_list)
                        
                        # 4. 清理无效几何 (修复自交)
                        # buffer(0) 可能会把一个 Polygon 变成 MultiPolygon
                        cleaned_geom = raw_poly.buffer(0)

                        if cleaned_geom.is_empty:
                            continue

                        # 统一标准化为列表处理
                        # 如果是 MultiPolygon，这就包含了多个子多边形
                        # 如果是 Polygon，就把它放进列表里
                        if cleaned_geom.geom_type == 'MultiPolygon':
                            polys_to_process = list(cleaned_geom.geoms)
                        elif cleaned_geom.geom_type == 'Polygon':
                            polys_to_process = [cleaned_geom]
                        else:
                            continue

                        # 5. 遍历列表进行拉伸
                        for p in polys_to_process:
                            if p.area > 1e-6: # 忽略极小碎屑
                                mesh = trimesh.creation.extrude_polygon(p, height=extrude_h)
                                
                                # 6. 移动到正确高度
                                z_min = mesh.bounds[0][2]
                                mesh.apply_translation([0, 0, z_pos - z_min])
                                
                                meshes_to_combine.append(mesh)
                            
                    except Exception as e:
                        print(f"    [!] 几何构建警告: {e}")
                        continue

    if not meshes_to_combine: 
        return None
        
    # 合并当前 Slot 的所有 Mesh
    print(f"    - 合并 {len(meshes_to_combine)} 个几何体片段...")
    combined_mesh = trimesh.util.concatenate(meshes_to_combine)
    return combined_mesh

def generate_preview_image_rgba(lut_colors, mapped_indices, width, height, alpha_channel, filename="preview.png"):
    print(f"正在生成带透明通道的预览: {filename} ...")
    rgb_data = lut_colors[mapped_indices] 
    rgba_data = np.dstack((rgb_data, alpha_channel)).astype(np.uint8)
    preview_img = Image.fromarray(rgba_data, 'RGBA')
    os.makedirs("Output", exist_ok=True)
    preview_img.save(os.path.join("Output", filename))
    return preview_img

def watershed_superpixels_from_lab(lab, mask=None, seed_step=12, grad_sigma=1.0):
    """
    生成超像素区域 ID map。
    优化点：增加了梯度计算的鲁棒性，确保边缘贴合。
    """
    lab = np.asarray(lab, dtype=np.float32)
    H, W, _ = lab.shape

    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    # 1. 计算 Lab 空间的梯度幅值 (Gradient Magnitude)
    # 使用 Gaussian 预平滑减少噪点对梯度的影响
    lab_s = ndimage.gaussian_filter(lab, sigma=(grad_sigma, grad_sigma, 0), mode="reflect")
    
    # 计算每个通道的梯度平方和
    gy, gx, _ = np.gradient(lab_s, axis=(0, 1, 2))
    grad_sq = gx**2 + gy**2
    # 欧氏距离梯度幅值
    grad = np.sqrt(np.sum(grad_sq, axis=2))

    # 2. 增强 Mask 边缘
    # 将 Mask 外的区域梯度设为最大，防止分水岭越界
    max_grad = grad.max() if grad.size > 0 else 1.0
    grad[~mask] = max_grad * 1.5 

    # 3. 归一化梯度并转为 uint8 (watershed_ift 需要)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    g8 = (grad * 255).astype(np.uint8)

    # 4. 生成种子点 (Markers)
    # 优化：仅在 Mask 内部生成种子
    markers = np.zeros((H, W), dtype=np.int32)
    
    # 生成网格坐标
    grid_y, grid_x = np.mgrid[seed_step//2:H:seed_step, seed_step//2:W:seed_step]
    
    # 筛选出位于 Mask 内的种子
    valid_seeds = mask[grid_y, grid_x]
    seeds_y = grid_y[valid_seeds]
    seeds_x = grid_x[valid_seeds]
    
    # 赋予唯一 ID (从 1 开始)
    num_seeds = len(seeds_y)
    marker_ids = np.arange(1, num_seeds + 1, dtype=np.int32)
    markers[seeds_y, seeds_x] = marker_ids

    print(f"  [分水岭] 生成种子点: {num_seeds} 个 (Step={seed_step})")

    # 5. 执行分水岭
    # structure 定义了连通性，默认是 3x3 十字交叉
    structure = ndimage.generate_binary_structure(2, 1) 
    regions = ndimage.watershed_ift(g8, markers, structure=structure)
    
    return regions

def generate_regions_felzenszwalb(img_arr_rgb, min_pixel_size=10, scale=100, sigma=0.5, mask=None):
    """
    使用 Felzenszwalb 算法生成区域 ID。
    img_arr_rgb: (H, W, 3) 0-255 uint8 或 float
    """
    print(f"  [Felzenszwalb] 正在分割: Scale={scale}, MinSize={min_pixel_size}px ...")
    
    # 1. 归一化 (skimage 需要 0-1 float)
    img_float = img_arr_rgb.astype(float) / 255.0
    
    # 2. 核心计算
    # scale: 观察尺度 (越大块越少)
    # sigma: 预平滑 (越小越锐利)
    # min_size: 最小像素数 (直接消灭孤岛!)
    segments = felzenszwalb(img_float, scale=scale, sigma=sigma, min_size=min_pixel_size)
    
    # 3. 处理 Mask (如果有透明背景)
    # Felzenszwalb 会对透明区域也计算分割，我们需要把透明区域重置为 0 (背景)
    if mask is not None:
        # 将 mask 之外的区域 ID 设为 0（通常 0 不参与 KDTree 匹配）
        # 注意：segments 的 ID 是从 0 开始的，为了避免冲突，
        # 我们通常把有效 ID + 1，把背景设为 0
        segments = segments + 1
        segments[~mask] = 0
        
    return segments

def region_based_rematching(img_lab, regions, tree, lut_indices_map, mask=None):
    """
    核心逻辑：区域平均 -> 唯一匹配
    (修复了 shape mismatch 错误)
    """
    print("  [重匹配] 正在计算区域平均颜色并查询 KDTree...")
    H, W = regions.shape
    
    # 1. 获取所有有效的 Region ID
    if mask is not None:
        active_regions = np.unique(regions[mask])
    else:
        active_regions = np.unique(regions)
    
    # 去除背景 (ID 0)
    active_regions = active_regions[active_regions > 0]
    
    # 2. 计算每个区域的平均颜色 (修复点)
    # 我们需要分别对 3 个通道 (L, a, b) 进行统计，否则会报错
    mean_colors = np.zeros((len(active_regions), 3))
    for i in range(3):
        # 对第 i 个通道计算均值
        mean_colors[:, i] = ndimage.mean(input=img_lab[..., i], labels=regions, index=active_regions)
    
    # 3. 对平均颜色进行 KDTree 查询
    dists, stack_indices = tree.query(mean_colors)
    
    # 4. 构建映射表
    max_region_id = regions.max()
    
    # 映射表 A: Region ID -> 物理层叠 (用于 STL)
    matched_stacks = lut_indices_map[stack_indices]
    id_to_stack_map = np.zeros((max_region_id + 1, TOTAL_LAYERS), dtype=int)
    id_to_stack_map[active_regions] = matched_stacks
    
    # 映射表 B: Region ID -> LUT 索引 (用于预览图)
    id_to_lut_idx_map = np.zeros(max_region_id + 1, dtype=int)
    id_to_lut_idx_map[active_regions] = stack_indices

    # 5. 广播回像素空间
    final_stack_matrix = id_to_stack_map[regions]       # (H, W, 5)
    final_lut_idx_matrix = id_to_lut_idx_map[regions]   # (H, W) 用于预览
    
    return final_stack_matrix, final_lut_idx_matrix


# ================= 4. 主程序流程 =================

def main():
    print("=== FWOC8 K-M Engine Image-to-STL (Lab Color Space) ===")
    
    # 1. 加载 & 2. 选料
    inventory = load_inventory(INVENTORY_FILE)
    if not inventory: return
    selected_filaments = get_selected_filaments(inventory, SELECTED_FILAMENT_NAMES)
    num_slots = len(selected_filaments)
    
    print(f"最终使用的耗材方案 (共 {num_slots} 色):")
    for i, f in enumerate(selected_filaments):
        print(f"  Slot {i+1}: {f['Name']}")
    
    # 3. K-M 物理计算
    engine = VirtualPhysics()
    lut_colors, lut_indices_map = engine.generate_lut_km(selected_filaments)
    
    visualize_gamut(lut_colors)
    print("\n" + "="*50)
    print("👀 请检查色域图 (gamut_check.png)。")
    print("="*50 + "\n")

    # 4. 读取图片
    print(f"读取图片: {INPUT_IMAGE}")
    if not os.path.exists(INPUT_IMAGE):
        print(f"错误: 找不到图片 {INPUT_IMAGE}")
        return
        
    img = Image.open(INPUT_IMAGE).convert('RGBA')
    w_pixels = int(TARGET_WIDTH_MM / PIXEL_SIZE)
    aspect = img.height / img.width
    h_pixels = int(w_pixels * aspect)
    print(f"目标分辨率: {w_pixels} x {h_pixels} px")
    
    img_resized = img.resize((w_pixels, h_pixels), Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized)
    
    alpha_channel_2d = img_arr[..., 3]
    solid_mask_2d = alpha_channel_2d > ALPHA_THRESHOLD

    # 5. KDTree 颜色匹配
    print("正在匹配像素颜色 (CIELAB 空间)...")
    lut_lab = rgb_to_lab(lut_colors)
    tree = KDTree(lut_lab)
    img_lab_2d = rgb_to_lab(img_arr[..., :3].reshape(-1, 3)).reshape(h_pixels, w_pixels, 3)
    
    # print("  > 正在执行分水岭分割 (Watershed)...")
    # regions = watershed_superpixels_from_lab(
    #     img_lab_2d, 
    #     mask=solid_mask_2d, 
    #     seed_step=5,
    #     grad_sigma=1.0
    # )

    regions = generate_regions_felzenszwalb(
        img_arr[..., :3],  # 传入 RGB
        min_pixel_size=5, # 约等于 1.6mm² 的最小打印面积
        scale=10,          # 针对复杂插画，50-100 比较合适
        sigma=0.5,
        mask=solid_mask_2d
    )
    
    final_stack_matrix, mapped_indices = region_based_rematching(
        img_lab_2d,
        regions,
        tree,
        lut_indices_map,
        mask=solid_mask_2d
    )

    generate_preview_image_rgba(
        lut_colors, 
        mapped_indices, 
        w_pixels, 
        h_pixels, 
        alpha_channel_2d, 
        "preview_simulation_km.png"
    )

    # ================= 6. 生成 3MF 双面模型 =================
    print(f"\n📦 开始打包生成 3MF 文件 (共 {num_slots} 色)...")
    
    h_color_stack = TOTAL_LAYERS * LAYER_HEIGHT 
    z_back_start = 0.0
    z_base_start = h_color_stack
    z_front_start = h_color_stack + BASE_HEIGHT
    
    # 这里选择底面和原图一致，顶面水平翻转，PEI纹理板打出来更好看
    # 1. 翻转 Mask (形状镜像) - axis=1 是水平方向
    mask_common = np.flip(solid_mask_2d, axis=1)
    
    # 2. 翻转 颜色矩阵 (像素位置镜像)
    # final_stack_matrix 形状是 (H, W, Layers)
    matrix_mirrored_base = np.flip(final_stack_matrix, axis=1)

    # 3. 分配矩阵
    # 正面 (Top): 使用镜像后的矩阵
    matrix_front = matrix_mirrored_base
    
    # 背面 (Bottom): 既要水平镜像(为了位置)，又要Z轴倒序(为了层叠顺序)
    # [..., ::-1] 是将最后一维 (Layers) 倒序
    matrix_back = matrix_mirrored_base.copy()[..., ::-1] 

    # --- 初始化场景 ---
    scene = trimesh.Scene()
    
    for i in range(num_slots):
        fil_name = selected_filaments[i]['Name'].replace(" ", "_")
        meshes_list = [] 
        
        # 1. 背面 (Bottom Layer - 贴床面)
        mesh_back = create_voxel_mesh_masked(
            matrix_back, i, w_pixels, h_pixels, mask_common, 
            z_offset=z_back_start, is_base_layer=False
        )
        if mesh_back: meshes_list.append(mesh_back)

        # 2. 中间 (仅限 Slot 1 - 白色底座)
        if i == 0: 
            mesh_mid = create_voxel_mesh_masked(
                matrix_front, i, w_pixels, h_pixels, mask_common,
                z_offset=z_base_start, is_base_layer=True
            )
            if mesh_mid: meshes_list.append(mesh_mid)

        # 3. 正面 (Top Layer)
        mesh_front = create_voxel_mesh_masked(
            matrix_front, i, w_pixels, h_pixels, mask_common,
            z_offset=z_front_start, is_base_layer=False
        )
        if mesh_front: meshes_list.append(mesh_front)

        # --- 合并 & 挂载到组 ---
        if meshes_list:
            final_mesh = trimesh.util.concatenate(meshes_list)
            
            # 视觉颜色
            hex_color = selected_filaments[i].get('Color', '#808080') 
            try:
                c_rgb = [int(hex_color[j:j+2], 16) for j in (1, 3, 5)]
                c_rgba = c_rgb + [255]
                final_mesh.visual.face_colors = c_rgba
            except:
                pass

            # 给零件命名
            final_mesh.metadata['name'] = fil_name

            # 添加到场景
            scene.add_geometry(final_mesh, node_name=fil_name, geom_name=fil_name)
            print(f"  > 已添加零件: {fil_name}")
            
        else:
            # print(f"  > 跳过空零件: Slot {i+1} ({fil_name})")
            pass

    # --- 导出 ---
    if len(scene.geometry) > 0:
        output_filename = "ChromaStack_Project_update.3mf"
        print(f"💾 正在保存 3MF 文件: {output_filename} ...")
        scene.export(os.path.join("Output", output_filename))
        print("✅ 保存成功！请将 .3mf 文件拖入 Bambu Studio / Orca Slicer。")
    else:
        print("⚠️ 场景为空，未生成文件。")

    print("\n=== 完成! ===")

if __name__ == "__main__":
    main()