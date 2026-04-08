#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型生成路由模块
"""

import yaml
import json
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify
import cv2
import trimesh
from shapely.geometry import Polygon
# 设置 matplotlib 非交互式后端，避免 Tkinter 线程错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colorsys

# 创建蓝图
model_bp = Blueprint('model', __name__)

# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent.parent / 'config' / 'model_generation.yaml'
INVENTORY_FILE = Path(__file__).parent.parent.parent.parent / 'my_filament.json'

# TOTAL_LAYERS 常量
TOTAL_LAYERS = 5

def visualize_gamut(lut_colors):
    """
    生成色彩域预览图并保存，不显示弹窗
    
    参数:
    lut_colors: 生成的颜色查找表，形状为 (N, 3)，值范围 0-255
    """
    print("\n📊 正在生成色域预览图...")
    colors_norm = lut_colors / 255.0
    
    # 创建图形
    fig = plt.figure("Gamut Analysis", figsize=(14, 6))
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

    # 2. 2D 色板图
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
        padding = np.ones((target_size - num_colors, 3)) 
        sorted_colors_padded = np.vstack([sorted_colors, padding])
    else:
        sorted_colors_padded = sorted_colors

    # Reshape 为动态计算出的边长
    grid_img = sorted_colors_padded.reshape(side_len, side_len, 3)

    ax2.imshow(grid_img)
    ax2.set_title(f'Available Palette\nSorted by Hue (Grid: {side_len}x{side_len})')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # 创建输出目录
    debug_output_dir = Path(__file__).parent.parent.parent.parent / 'debug_output'
    debug_output_dir.mkdir(exist_ok=True)
    
    # 保存图片
    output_path = debug_output_dir / 'gamut_check.png'
    plt.savefig(output_path)
    plt.close()  # 关闭图形，释放内存
    print(f"📈 色域图已保存为 {output_path}")

def create_voxel_mesh_masked(indices_matrix, slot_id, width_pixels, height_pixels, solid_mask_2d, z_offset=0.0, is_base_layer=False, layer_height=0.08, base_height=0.8, pixel_size=0.2):
    """
    [修复版] 为单个耗材创建带 Mask 的网格
    1. 解决了 trimesh.load_path 不接受列表的报错。
    2. 增加了孔洞处理 (RETR_CCOMP)，防止 'O' 型图案中间被填实。
    
    参数:
    indices_matrix: (H, W, Layers) 每个像素的层叠索引
    slot_id: 耗材在插槽中的索引
    width_pixels, height_pixels: 图片分辨率
    solid_mask_2d: (H, W) 布尔掩码，True 表示需要打印
    z_offset: 起始高度
    is_base_layer: 是否为底座层（单层）
    layer_height: 颜色层层高 (mm)
    base_height: 白色底座厚度 (mm)
    pixel_size: 像素尺寸/水平分辨率 (mm)
    """
    meshes_to_combine = []

    # 辅助函数：将 OpenCV 轮廓坐标转换为物理坐标
    def convert_contour_to_points(cnt):
        # cnt shape: (N, 1, 2) -> (N, 2)
        pts = cnt.reshape(-1, 2)
        physical_pts = np.zeros_like(pts, dtype=float)
        # X轴转换 (注意：Main函数里可能已经做过镜像，这里只负责缩放)
        physical_pts[:, 0] = pts[:, 0] * pixel_size
        # Y轴转换 (OpenCV原点在左上，3D打印在左下，需要翻转Y)
        physical_pts[:, 1] = (height_pixels - 1 - pts[:, 1]) * pixel_size
        return physical_pts

    # 待处理的任务列表：(Layer_Index, Mask)
    tasks = []
    
    if is_base_layer and slot_id == 0:
        # 场景 A: 白色底座 (单层厚度 = base_height)
        layer_mask_u8 = (solid_mask_2d.astype(np.uint8)) * 255
        tasks.append({
            "mask": layer_mask_u8, 
            "height": base_height, 
            "z_start": z_offset
        })
        
    elif not is_base_layer:
        # 场景 B: 彩色层 (逐层切片, 单层厚度 = layer_height)
        for layer_idx in range(TOTAL_LAYERS):
            current_layer_slots = indices_matrix[:, :, layer_idx]
            layer_mask = (current_layer_slots == slot_id) & solid_mask_2d
            if np.any(layer_mask):
                tasks.append({
                    "mask": layer_mask.astype(np.uint8) * 255,
                    "height": layer_height,
                    "z_start": z_offset + layer_idx * layer_height
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


def load_config():
    """加载配置文件"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f'读取配置文件失败: {e}')
            return {}
    return {}


def save_config(config_data):
    """保存配置文件"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        print(f'保存配置文件失败: {e}')
        return False


def load_filaments():
    """加载耗材库"""
    filaments = []
    if INVENTORY_FILE.exists():
        try:
            with open(INVENTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filaments = data.get('Filaments', [])
        except Exception as e:
            print(f'读取耗材库失败: {e}')
    return filaments


@model_bp.route('/config/model', methods=['GET'])
def get_model_config():
    """获取模型生成配置"""
    try:
        config = load_config()
        return jsonify({'success': True, 'config': config}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/config/model', methods=['POST'])
def update_model_config():
    """更新模型生成配置"""
    try:
        config_data = request.get_json()
        
        if not config_data:
            return jsonify({'error': '无效的配置数据'}), 400
        
        existing_config = load_config()
        existing_config.update(config_data)
        
        if save_config(existing_config):
            return jsonify({'success': True, 'message': '配置保存成功'}), 200
        else:
            return jsonify({'error': '保存配置失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/filaments', methods=['GET'])
def get_filaments():
    """获取耗材列表"""
    try:
        filaments = load_filaments()
        return jsonify({'success': True, 'filaments': filaments}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/colorize', methods=['POST'])
def colorize_image():
    """自动配色"""
    try:
        # 从FormData获取文件和参数
        if 'file' not in request.files:
            return jsonify({'error': '缺少文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名不能为空'}), 400
        
        # 获取颜色数量参数
        color_count = int(request.form.get('color_count', 5))
        
        # 保存文件到临时目录
        temp_dir = Path(__file__).parent.parent.parent.parent / 'tmp'
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        file.save(file_path)
        
        # 导入并执行自动配色
        from AutoSelector import extract_image_features
        
        # 执行颜色提取
        centers_lab, weights = extract_image_features(str(file_path), n_colors=500)
        
        if centers_lab is None:
            return jsonify({'error': '图片处理失败'}), 500
        
        # 从ChromStackStudio和AutoSelector导入必要的函数
        from ChromaStackStudio import VirtualPhysics, load_inventory
        from AutoSelector import evaluate_combination
        import itertools
        
        # 加载耗材库
        inventory = load_inventory(str(INVENTORY_FILE))
        if not inventory:
            return jsonify({'error': '耗材库为空'}), 500
        
        # 创建虚拟物理引擎实例
        engine = VirtualPhysics()
        
        # 评估所有可能的耗材组合
        combo_scores = []
        
        # 生成所有可能的耗材组合（选择color_count个耗材）
        combinations = list(itertools.combinations(inventory, color_count))
        print(f"共有 {len(combinations)} 种耗材组合待评估...")
        
        for combo in combinations:
            try:
                score = evaluate_combination(engine, list(combo), centers_lab, weights)
                combo_scores.append((score, [f['Name'] for f in combo]))
            except Exception as e:
                print(f"评估耗材组合时出错: {e}")
                continue
        
        # 如果没有找到合适的组合，返回错误
        if not combo_scores:
            return jsonify({'error': '无法找到合适的耗材组合'}), 500
        
        # 按分数排序，选择前三的组合
        combo_scores.sort(key=lambda x: x[0])
        top_combinations = combo_scores[:3]
        
        # 提取最佳组合
        best_score, best_combo = top_combinations[0]
        
        # 准备返回数据
        top_combos_data = []
        for score, combo in top_combinations:
            top_combos_data.append({
                'score': score,
                'filaments': combo
            })
        
        return jsonify({
            'success': True,
            'best_combination': {
                'score': best_score,
                'filaments': best_combo
            },
            'top_combinations': top_combos_data
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@model_bp.route('/preview', methods=['POST'])
def generate_preview():
    """生成预览图"""
    try:
        # 从FormData获取文件和参数
        if 'file' not in request.files:
            return jsonify({'error': '缺少文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名不能为空'}), 400

        # 获取耗材参数
        filaments_str = request.form.get('filaments', '[]')
        selected_filaments = json.loads(filaments_str)
        
        # 保存文件到临时目录
        temp_dir = Path(__file__).parent.parent.parent.parent / 'tmp'
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        file.save(file_path)
        
        # 加载配置
        config = load_config()
        
        # 获取算法参数
        min_pixel_size = int(request.form.get('min_pixel_size', config.get('min_pixel_size', 5)))
        scale = int(request.form.get('scale', config.get('scale', 10)))
        sigma = float(request.form.get('sigma', config.get('sigma', 0.5)))
        
        # 获取模型参数
        model_width = float(request.form.get('model_width', config.get('model_width', 80)))
        pixel_size = float(request.form.get('pixel_size', config.get('pixel_size', 0.2)))
        alpha_threshold = int(request.form.get('alpha_threshold', config.get('alpha_threshold', 128)))
        layer_height = float(request.form.get('layer_height', config.get('layer_height', 0.08)))
        
        # 导入必要的模块
        from ChromaStackStudio import VirtualPhysics, rgb_to_lab, load_inventory, generate_regions_felzenszwalb, region_based_rematching
        from scipy.spatial import KDTree
        
        # 加载耗材库
        inventory = load_inventory(str(INVENTORY_FILE))
        
        # 根据名称找到选中的耗材
        selected = []
        for name in selected_filaments:
            filament = next((f for f in inventory if f['Name'] == name), None)
            if filament:
                selected.append(filament)
        
        if len(selected) < 2:
            return jsonify({'error': '请至少选择2个耗材'}), 400
        
        # 生成LUT
        engine = VirtualPhysics()
        lut_rgb, lut_indices_map = engine.generate_lut_km(selected, total_layers=TOTAL_LAYERS, layer_height=layer_height)
        
        # 生成色彩域预览图
        visualize_gamut(lut_rgb)
        
        # 生成唯一的预览图文件名
        import uuid
        preview_filename = f'preview_result_{uuid.uuid4().hex}.png'
        output_path = temp_dir / preview_filename
        
        # 加载原始图片
        from PIL import Image
        
        img = Image.open(file_path).convert('RGBA')
        width, height = img.size
        
        # 调整大小 - 与ChromaStackStudio.py保持一致
        target_width = int(model_width / pixel_size)  # 根据模型宽度和像素尺寸计算目标宽度
        # 保持原始图片比例计算目标高度
        aspect = height / width
        target_height = int(target_width * aspect)
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # 转换为数组
        img_arr = np.array(img)
        
        # 计算透明度掩码
        alpha_channel_2d = img_arr[..., 3]
        solid_mask_2d = alpha_channel_2d > alpha_threshold  # 使用从前端传入的参数
        
        # KDTree 颜色匹配
        lut_lab = rgb_to_lab(lut_rgb)
        tree = KDTree(lut_lab)
        img_lab_2d = rgb_to_lab(img_arr[..., :3].reshape(-1, 3)).reshape(target_height, target_width, 3)
        
        # 区域分割
        regions = generate_regions_felzenszwalb(
            img_arr[..., :3],  # 传入 RGB
            min_pixel_size=min_pixel_size,  # 使用从前端传入的参数
            scale=scale,           # 使用从前端传入的参数
            sigma=sigma,          # 使用从前端传入的参数
            mask=solid_mask_2d
        )
        
        # 区域基于的重匹配
        final_stack_matrix, final_lut_idx_matrix = region_based_rematching(
            img_lab_2d,
            regions,
            tree,
            lut_indices_map,
            mask=solid_mask_2d
        )
        
        # 生成预览图
        preview_img = Image.fromarray(lut_rgb[final_lut_idx_matrix])
        preview_img.save(output_path)
        
        # 返回相对路径，前端可以直接访问
        return jsonify({
            'success': True,
            'preview_path': f'/tmp/{preview_filename}',
            'lut_colors': lut_rgb.tolist() if hasattr(lut_rgb, 'tolist') else lut_rgb,
            'target_width': target_width,
            'target_height': target_height
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@model_bp.route('/generate', methods=['POST'])
def generate_model():
    """生成模型"""
    try:
        # 从FormData获取文件和参数
        if 'file' not in request.files:
            return jsonify({'error': '缺少文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名不能为空'}), 400
        
        # 将API所有参数输出到控制台
        print(f"调试：收到的FormData参数 = {request.form}")
        
        # 获取耗材参数
        filaments_str = request.form.get('filaments', '[]')
        selected_filaments = json.loads(filaments_str)
        
        if not selected_filaments:
            return jsonify({'error': '请选择至少一个耗材'}), 400
        
        # 保存文件到临时目录
        temp_dir = Path(__file__).parent.parent.parent.parent / 'tmp'
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        file.save(file_path)
        
        # 加载配置
        config = load_config()
        
        # 获取算法参数
        min_pixel_size = int(request.form.get('min_pixel_size', config.get('min_pixel_size', 5)))
        scale = int(request.form.get('scale', config.get('scale', 10)))
        sigma = float(request.form.get('sigma', config.get('sigma', 0.5)))
        
        # 获取模型参数
        layer_height = float(request.form.get('layer_height', config.get('layer_height', 0.08)))
        model_width = float(request.form.get('model_width', config.get('model_width', 80)))
        model_height = float(request.form.get('model_height', config.get('model_height', 80)))
        model_depth = float(request.form.get('model_depth', config.get('model_depth', 0.8)))
        pixel_size = float(request.form.get('pixel_size', config.get('pixel_size', 0.2)))
        alpha_threshold = int(request.form.get('alpha_threshold', config.get('alpha_threshold', 128)))
        # 获取是否生成双面模型的参数
        is_double_sided = request.form.get('is_double_sided', str(config.get('is_double_sided', True))).lower() == 'true'
        
        # 导入必要的模块
        from ChromaStackStudio import VirtualPhysics, load_inventory
        
        # 加载耗材库
        inventory = load_inventory(str(INVENTORY_FILE))
        
        # 根据名称找到选中的耗材
        selected = []
        for name in selected_filaments:
            filament = next((f for f in inventory if f['Name'] == name), None)
            if filament:
                selected.append(filament)
        
        if len(selected) < 2:
            return jsonify({'error': '请至少选择2个耗材'}), 400
        
        # 生成LUT
        engine = VirtualPhysics()
        lut_rgb, lut_indices_map = engine.generate_lut_km(selected, TOTAL_LAYERS, layer_height)
        
        # 加载原始图片
        from PIL import Image
        
        img = Image.open(file_path).convert('RGBA')
        width, height = img.size
        
        # 调整大小 - 与ChromaStackStudio.py保持一致
        target_width = int(model_width / pixel_size)  # 根据模型宽度和像素尺寸计算目标宽度
        # 保持原始图片比例计算目标高度
        aspect = height / width
        target_height = int(target_width * aspect)
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # 转换为数组
        img_arr = np.array(img)
        
        # 计算透明度掩码
        alpha_channel_2d = img_arr[..., 3]
        solid_mask_2d = alpha_channel_2d > alpha_threshold  # 使用从前端传入的参数
        
        from ChromaStackStudio import rgb_to_lab, generate_regions_felzenszwalb, region_based_rematching
        from scipy.spatial import KDTree
        
        # KDTree 颜色匹配
        lut_lab = rgb_to_lab(lut_rgb)
        tree = KDTree(lut_lab)
        img_lab_2d = rgb_to_lab(img_arr[..., :3].reshape(-1, 3)).reshape(target_height, target_width, 3)
        
        # 区域分割
        regions = generate_regions_felzenszwalb(
            img_arr[..., :3],  # 传入 RGB
            min_pixel_size=min_pixel_size,  # 使用从前端传入的参数
            scale=scale,           # 使用从前端传入的参数
            sigma=sigma,          # 使用从前端传入的参数
            mask=solid_mask_2d
        )
        
        # 区域基于的重匹配
        final_stack_matrix, final_lut_idx_matrix = region_based_rematching(
            img_lab_2d,
            regions,
            tree,
            lut_indices_map,
            mask=solid_mask_2d
        )
        
        # 生成 3D 模型
        import trimesh
        from shapely.geometry import Polygon
        import cv2
        
        # 创建输出目录
        output_dir = Path(__file__).parent.parent.parent.parent / 'Output'
        output_dir.mkdir(exist_ok=True)
        
        # 计算尺寸
        h_color_stack = TOTAL_LAYERS * layer_height
        z_back_start = 0.0
        z_base_start = h_color_stack
        z_front_start = h_color_stack + model_depth
        
        # 翻转 Mask (形状镜像) - axis=1 是水平方向
        mask_common = np.flip(solid_mask_2d, axis=1)
        
        # 翻转 颜色矩阵 (像素位置镜像)
        matrix_mirrored_base = np.flip(final_stack_matrix, axis=1)
        
        # 分配矩阵
        # 正面 (Top): 使用镜像后的矩阵
        matrix_front = matrix_mirrored_base
        
        # 背面 (Bottom): 既要水平镜像(为了位置)，又要Z轴倒序(为了层叠顺序)
        matrix_back = matrix_mirrored_base.copy()[..., ::-1]
        
        # 初始化场景
        scene = trimesh.Scene()
        
        # 为每个耗材生成网格
        num_slots = len(selected)
        
        for i in range(num_slots):
            fil_name = selected[i]['Name'].replace(" ", "_")
            meshes_list = []
            
            # 1. 背面 (Bottom Layer - 贴床面)
            mesh_back = create_voxel_mesh_masked(
                matrix_back, i, target_width, target_height, mask_common, 
                z_offset=z_back_start, is_base_layer=False,
                layer_height=layer_height, base_height=model_depth, pixel_size=pixel_size
            )
            if mesh_back:
                meshes_list.append(mesh_back)

            # 2. 中间 (仅限 Slot 1 - 白色底座)
            if i == 0:
                mesh_mid = create_voxel_mesh_masked(
                    matrix_front, i, target_width, target_height, mask_common,
                    z_offset=z_base_start, is_base_layer=True,
                    layer_height=layer_height, base_height=model_depth, pixel_size=pixel_size
                )
                if mesh_mid:
                    meshes_list.append(mesh_mid)

            # 3. 正面 (Top Layer) - 仅在双面模式下生成
            if is_double_sided:
                mesh_front = create_voxel_mesh_masked(
                    matrix_front, i, target_width, target_height, mask_common,
                    z_offset=z_front_start, is_base_layer=False,
                    layer_height=layer_height, base_height=model_depth, pixel_size=pixel_size
                )
                if mesh_front:
                    meshes_list.append(mesh_front)

            # 合并 & 挂载到组
            if meshes_list:
                final_mesh = trimesh.util.concatenate(meshes_list)
                
                # 视觉颜色
                hex_color = selected[i].get('Color', '#808080')
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
                pass
        
        # 导出 3MF 文件
        import uuid
        model_filename = f"ChromaStack_Project_{uuid.uuid4().hex}.3mf"
        model_path = output_dir / model_filename
        
        if len(scene.geometry) > 0:
            print(f"💾 正在保存 3MF 文件: {model_filename} ...")
            scene.export(str(model_path))
            print("✅ 保存成功！")
        else:
            print("⚠️ 场景为空，未生成文件。")
            model_filename = None
        
        # 返回相对路径，前端可以直接访问
        return jsonify({
            'success': True,
            'model_path': f'/Output/{model_filename}' if model_filename else None
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
