#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耗材管理路由模块
"""

import json
from pathlib import Path
from flask import Blueprint, request, jsonify
from ..config import Config

# 创建蓝图
filament_bp = Blueprint('filament', __name__)


def load_filaments():
    """
    加载耗材库数据
    
    Returns:
        dict: 耗材库数据，格式为{'Filaments': [filament1, filament2, ...]}
    """
    filaments = {'Filaments': []}
    if Config.FILAMENT_FILE.exists():
        try:
            with Config.FILAMENT_FILE.open('r', encoding='utf-8') as f:
                filaments = json.load(f)
        except Exception as e:
            print(f'读取耗材库失败: {e}')
    return filaments


def save_filaments(filaments):
    """
    保存耗材库数据
    
    Args:
        filaments (dict): 耗材库数据
        
    Returns:
        bool: 保存是否成功
    """
    try:
        with Config.FILAMENT_FILE.open('w', encoding='utf-8') as f:
            json.dump(filaments, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f'保存耗材库失败: {e}')
        return False


@filament_bp.route('/filaments', methods=['GET'])
def get_filaments():
    """
    获取所有耗材
    
    Returns:
        json: 耗材列表
    """
    try:
        # 加载耗材库
        filaments = load_filaments()
        return jsonify({'success': True, 'filaments': filaments['Filaments']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@filament_bp.route('/filaments/check_name/<name>', methods=['GET'])
def check_filament_name(name):
    """
    检查耗材名称是否唯一
    
    Args:
        name (str): 耗材名称
        
    Returns:
        json: 检查结果
    """
    try:
        # 加载耗材库
        filaments = load_filaments()
        
        # 检查名称是否已存在
        for filament in filaments['Filaments']:
            if filament['Name'] == name:
                return jsonify({'success': True, 'is_unique': False}), 200
        
        return jsonify({'success': True, 'is_unique': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@filament_bp.route('/filaments', methods=['POST'])
def add_filament():
    """
    新增耗材
    
    Returns:
        json: 保存结果
    """
    try:
        # 获取请求数据
        filament_data = request.get_json()
        
        # 验证必填字段
        required_fields = ['Name', 'FILAMENT_K', 'FILAMENT_S', 'Type']
        for field in required_fields:
            if field not in filament_data:
                return jsonify({'error': f'缺少必填字段: {field}'}), 400
        
        # 验证Type值
        if filament_data['Type'] not in ['PLA', 'PETG']:
            return jsonify({'error': 'Type必须为PLA或PETG'}), 400
        
        # 加载耗材库
        filaments = load_filaments()
        
        # 检查名称是否已存在
        for filament in filaments['Filaments']:
            if filament['Name'] == filament_data['Name']:
                return jsonify({'error': '耗材名称已存在'}), 400
        
        # 添加新耗材
        filaments['Filaments'].append(filament_data)
        
        # 保存到文件
        if save_filaments(filaments):
            return jsonify({'success': True, 'message': '耗材保存成功'}), 200
        else:
            return jsonify({'error': '保存耗材库失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@filament_bp.route('/filaments/<name>', methods=['PUT'])
def update_filament(name):
    """
    修改耗材
    
    Args:
        name (str): 原耗材名称
        
    Returns:
        json: 修改结果
    """
    try:
        # 获取请求数据
        filament_data = request.get_json()
        
        # 验证必填字段
        required_fields = ['FILAMENT_K', 'FILAMENT_S', 'Type']
        for field in required_fields:
            if field not in filament_data:
                return jsonify({'error': f'缺少必填字段: {field}'}), 400
        
        # 验证Type值
        if filament_data['Type'] not in ['PLA', 'PETG']:
            return jsonify({'error': 'Type必须为PLA或PETG'}), 400
        
        # 加载耗材库
        filaments = load_filaments()
        
        # 查找并修改耗材
        updated = False
        for i, filament in enumerate(filaments['Filaments']):
            if filament['Name'] == name:
                # 如果新名称与原名称不同，检查是否已存在
                if 'Name' in filament_data and filament_data['Name'] != name:
                    for f in filaments['Filaments']:
                        if f['Name'] == filament_data['Name']:
                            return jsonify({'error': '新耗材名称已存在'}), 400
                
                # 更新耗材数据
                filaments['Filaments'][i].update(filament_data)
                updated = True
                break
        
        if not updated:
            return jsonify({'error': '耗材不存在'}), 404
        
        # 保存到文件
        if save_filaments(filaments):
            return jsonify({'success': True, 'message': '耗材修改成功'}), 200
        else:
            return jsonify({'error': '保存耗材库失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@filament_bp.route('/filaments/<name>', methods=['DELETE'])
def delete_filament(name):
    """
    删除耗材
    
    Args:
        name (str): 耗材名称
        
    Returns:
        json: 删除结果
    """
    try:
        # 加载耗材库
        filaments = load_filaments()
        
        # 查找并删除耗材
        original_length = len(filaments['Filaments'])
        filaments['Filaments'] = [f for f in filaments['Filaments'] if f['Name'] != name]
        
        if len(filaments['Filaments']) == original_length:
            return jsonify({'error': '耗材不存在'}), 404
        
        # 保存到文件
        if save_filaments(filaments):
            return jsonify({'success': True, 'message': '耗材删除成功'}), 200
        else:
            return jsonify({'error': '保存耗材库失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@filament_bp.route('/save_filament', methods=['POST'])
def save_filament():
    """
    保存耗材到库（兼容旧版本）
    
    Returns:
        json: 保存结果
    """
    return add_filament()
