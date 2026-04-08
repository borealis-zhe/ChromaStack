#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态文件服务路由模块
"""

from flask import Blueprint, send_from_directory
from ..config import Config
from pathlib import Path

# 创建蓝图
static_bp = Blueprint('static', __name__)

# 定义tmp目录路径
tmp_dir = Path(__file__).parent.parent.parent.parent / 'tmp'


@static_bp.route('/<path:path>')
def serve_static(path):
    """
    提供静态文件服务
    
    Args:
        path (str): 文件路径
        
    Returns:
        file: 静态文件
    """
    return send_from_directory(Config.FRONTEND_DIST_DIR, path)


@static_bp.route('/tmp/<path:path>')
def serve_tmp(path):
    """
    提供tmp目录下的文件服务
    
    Args:
        path (str): 文件路径
        
    Returns:
        file: tmp目录下的文件
    """
    return send_from_directory(tmp_dir, path)


@static_bp.route('/')
def index():
    """
    根路由，返回index.html
    
    Returns:
        file: index.html
    """
    return send_from_directory(Config.FRONTEND_DIST_DIR, 'index.html')
