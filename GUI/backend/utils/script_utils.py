#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本执行工具函数
"""

import sys
from io import StringIO
from pathlib import Path
from ..config import Config

# 从filament_cali模块直接导入校准脚本函数
from filament_cali import KS_calibration


def run_calibration_script(image_path, config=None):
    """
    执行校准脚本
    
    Args:
        image_path (str): 图像文件路径
        config (dict): 配置参数，如果不传则从配置文件读取
        
    Returns:
        dict: 校准结果，包含stdout、stderr和returncode
    """
    # 重定向标准输出和标准错误
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 创建字符串缓冲区来捕获输出
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    try:
        # 重定向输出
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # 保存原始配置
        original_image_path = KS_calibration.IMAGE_PATH
        original_layer_height = KS_calibration.LAYER_HEIGHT
        original_num_steps = KS_calibration.NUM_STEPS
        original_backing_reflectance_white = KS_calibration.BACKING_REFLECTANCE_WHITE
        original_backing_reflectance_black = KS_calibration.BACKING_REFLECTANCE_BLACK
        original_a4_width = KS_calibration.A4_WIDTH
        original_a4_height = KS_calibration.A4_HEIGHT
        original_chip_w = KS_calibration.CHIP_W
        original_chip_h = KS_calibration.CHIP_H
        
        try:
            # 如果传入了配置参数，更新KS_calibration的全局参数
            if config:
                KS_calibration.LAYER_HEIGHT = config.get('layer_height', 0.08)
                KS_calibration.NUM_STEPS = config.get('num_steps', 5)
                KS_calibration.BACKING_REFLECTANCE_WHITE = config.get('backing_reflectance_white', 0.94)
                KS_calibration.BACKING_REFLECTANCE_BLACK = config.get('backing_reflectance_black', 0.00)
                KS_calibration.A4_WIDTH = config.get('a4_width', 1414)
                KS_calibration.A4_HEIGHT = config.get('a4_height', 1000)
                if 'chip_w' in config:
                    KS_calibration.CHIP_W = config.get('chip_w', 400)
                if 'chip_h' in config:
                    KS_calibration.CHIP_H = config.get('chip_h', 500)
            
            # 设置校准脚本的IMAGE_PATH为传入的图像路径
            KS_calibration.IMAGE_PATH = image_path
            
            # 直接调用图像处理和计算函数
            print("=== 3D打印耗材 K-M 参数校准全流程 ===")
            
            # 1. 处理图片提取数据
            df = KS_calibration.process_image_to_data(image_path)
            
            if df is None:
                print("❌ 图片处理失败或已取消，程序终止。")
                returncode = 1
            else:
                # 2. 计算 K-M 参数
                KS_calibration.calculate_and_plot_km(df)
                returncode = 0
                
            # 获取输出
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
        finally:
            # 恢复原始配置
            KS_calibration.IMAGE_PATH = original_image_path
            KS_calibration.LAYER_HEIGHT = original_layer_height
            KS_calibration.NUM_STEPS = original_num_steps
            KS_calibration.BACKING_REFLECTANCE_WHITE = original_backing_reflectance_white
            KS_calibration.BACKING_REFLECTANCE_BLACK = original_backing_reflectance_black
            KS_calibration.A4_WIDTH = original_a4_width
            KS_calibration.A4_HEIGHT = original_a4_height
            KS_calibration.CHIP_W = original_chip_w
            KS_calibration.CHIP_H = original_chip_h
    except Exception as e:
        # 获取异常信息
        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue() + f"\nException: {str(e)}"
        returncode = 1
    finally:
        # 恢复标准输出和标准错误
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return {
        'stdout': stdout,
        'stderr': stderr,
        'returncode': returncode
    }
