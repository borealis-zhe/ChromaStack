#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件上传路由模块
"""

from flask import Blueprint, request, jsonify
from ..utils.file_utils import allowed_file, save_uploaded_file

# 创建蓝图
upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    处理文件上传
    
    Returns:
        json: 上传结果
    """
    try:
        # 检查是否有文件部分
        if 'file' not in request.files:
            return jsonify({'error': '没有文件部分'}), 400
        
        file = request.files['file']
        
        # 如果用户没有选择文件，浏览器会提交一个空文件
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        if file and allowed_file(file.filename):
            # 保存文件
            file_path = save_uploaded_file(file)
            
            # 返回文件路径
            return jsonify({'success': True, 'file_path': file_path}), 200
        else:
            return jsonify({'error': '文件类型不允许'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
