#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件路由模块
"""

import yaml
from pathlib import Path
from flask import Blueprint, request, jsonify

# 创建蓝图
config_bp = Blueprint('config', __name__)

# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent.parent / 'config' / 'filament_calibration.yaml'


def load_config():
    """
    加载配置文件
    
    Returns:
        dict: 配置文件内容
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f'读取配置文件失败: {e}')
            return {}
    return {}


def save_config(config_data):
    """
    保存配置文件
    
    Args:
        config_data (dict): 配置文件内容
        
    Returns:
        bool: 保存是否成功
    """
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        print(f'保存配置文件失败: {e}')
        return False


@config_bp.route('/config/calibration', methods=['GET'])
def get_calibration_config():
    """
    获取校准配置
    
    Returns:
        json: 校准配置
    """
    try:
        config = load_config()
        return jsonify({'success': True, 'config': config}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@config_bp.route('/config/calibration', methods=['POST'])
def update_calibration_config():
    """
    更新校准配置
    
    Returns:
        json: 更新结果
    """
    try:
        config_data = request.get_json()
        
        if not config_data:
            return jsonify({'error': '无效的配置数据'}), 400
        
        # 合并现有配置和新配置
        existing_config = load_config()
        existing_config.update(config_data)
        
        # 保存配置
        if save_config(existing_config):
            return jsonify({'success': True, 'message': '配置保存成功'}), 200
        else:
            return jsonify({'error': '保存配置失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
