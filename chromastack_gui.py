#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaStack GUI入口文件
基于Pywebview的GUI应用，用于加载Vue3前端界面
"""

import webview
import sys
import threading
import time
from pathlib import Path

# 获取当前脚本所在目录（项目根目录）
current_dir = Path(__file__).parent

# 导入Flask应用
from GUI.backend.app import app


def start_backend():
    """
    启动后端服务（多线程方式）
    """
    print("正在启动后端服务...")
    # 使用Flask的run方法启动服务，threaded=True允许处理并发请求
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def main():
    """
    主函数，启动Pywebview应用
    """
    # 检查dist目录和HTML文件是否存在
    frontend_dir = current_dir / "GUI" / "frontend"
    dist_dir = frontend_dir / "dist"
    html_file = dist_dir / "index.html"
    
    # 检查dist目录和HTML文件是否存在
    if not dist_dir.exists():
        print(f"错误：dist目录 {dist_dir} 不存在！")
        print("请先进入GUI/frontend目录运行 npm run build 编译Vue应用")
        sys.exit(1)
    
    if not html_file.exists():
        print(f"错误：HTML文件 {html_file} 不存在！")
        print("请先进入GUI/frontend目录运行 npm run build 编译Vue应用")
        sys.exit(1)
    
    # 使用多线程启动后端服务
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True  # 设置为守护线程，主进程退出时自动结束
    backend_thread.start()
    
    # 等待后端服务启动
    time.sleep(2)
    
    # 配置窗口选项
    window_options = {
        "title": "ChromaStack",
        "width": 1200,
        "height": 800,
        "resizable": True,
        "fullscreen": False,
        "min_size": (800, 600),
        "background_color": "#ffffff"
    }
    
    # 使用后端Flask服务的URL
    url = "http://localhost:5000/"
    
    print("正在启动ChromaStack GUI...")
    print(f"加载URL：{url}")
    
    # 创建并启动窗口
    webview.create_window(**window_options, url=url)
    webview.start()


if __name__ == "__main__":
    main()