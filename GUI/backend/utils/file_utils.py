#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件处理工具函数
"""

from werkzeug.utils import secure_filename
from ..config import Config


def allowed_file(filename):
    """
    检查文件扩展名是否允许
    
    Args:
        filename (str): 文件名
        
    Returns:
        bool: 是否允许
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    """
    保存上传的文件
    
    Args:
        file: Flask上传的文件对象
        
    Returns:
        str: 保存后的文件路径
    """
    filename = secure_filename(file.filename)
    file_path = Config.UPLOAD_FOLDER / filename
    file.save(str(file_path))
    return str(file_path)
