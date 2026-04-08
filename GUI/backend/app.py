#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaStack GUI后端主应用入口
"""

import sys
from pathlib import Path

from flask import Flask
from .routes.upload import upload_bp
from .routes.calibration import calibration_bp
from .routes.filament import filament_bp
from .routes.static import static_bp
from .routes.config import config_bp
from .routes.model import model_bp

# 创建Flask应用
app = Flask(__name__)

# 配置文件上传大小限制（16MB）
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 注册蓝图
app.register_blueprint(upload_bp)
app.register_blueprint(calibration_bp)
app.register_blueprint(filament_bp)
app.register_blueprint(static_bp)
app.register_blueprint(config_bp)
app.register_blueprint(model_bp)

if __name__ == '__main__':
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
