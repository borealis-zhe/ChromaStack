#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校准路由模块
"""

from pathlib import Path
from flask import Blueprint, request, jsonify
from ..utils.script_utils import run_calibration_script

# 创建蓝图
calibration_bp = Blueprint('calibration', __name__)

# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent.parent / 'config' / 'filament_calibration.yaml'


def load_config():
    """加载配置文件"""
    import yaml
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f'读取配置文件失败: {e}')
            return {}
    return {}


@calibration_bp.route('/calibrate', methods=['POST'])
def calibrate():
    """
    执行校准脚本
    
    Returns:
        json: 校准结果
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': '缺少文件路径'}), 400
        
        file_path = data['file_path']
        config = data.get('config', None)  # 获取配置参数
        
        # 检查文件是否存在
        if not Path(file_path).exists():
            return jsonify({'error': '文件不存在'}), 400
        
        # 如果没有传入配置，则从配置文件读取
        if not config:
            config = load_config()
        
        # 执行校准脚本，传入配置参数
        result = run_calibration_script(file_path, config)
        
        # 返回结果
        return jsonify({
            'success': True,
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'returncode': result['returncode']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
