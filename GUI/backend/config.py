#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaStack GUI后端配置管理
"""

from pathlib import Path

# 获取当前脚本所在目录
current_dir = Path(__file__).parent

# 配置类
class Config:
    """应用配置类"""
    # 上传配置
    UPLOAD_FOLDER = current_dir / 'tmp'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # 前端静态文件目录
    FRONTEND_DIST_DIR = current_dir.parent / 'frontend' / 'dist'
    
    # 校准脚本路径 - 使用相对路径
    CALIBRATION_SCRIPT = current_dir.parent.parent / 'filament_cali' / 'KS_calibration.py'
    
    # 耗材库文件路径 - 使用相对路径
    FILAMENT_FILE = current_dir.parent.parent / 'my_filament.json'

# 确保上传目录存在
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
