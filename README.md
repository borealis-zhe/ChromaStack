# ChromaStack Studio: 所见即光影

<div align="center">

**一款适用于FDM 3D打印机的多色层叠模型生成器**  
**A Multi-Color Layered Model Generator for FDM Printers**  
*Just print what you see*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)  
[英文文档 / English Documentation](README_en.md)

</div>

---

**ChromaStack** 是一款专为 FDM 3D 打印机设计的**多色层叠模型生成器**。它通过算法将二维图像转换为基于层厚的多色 3D 模型，利用不同颜色的透光叠加，打印出照片级的光影效果。

### ✨ 核心特性 (Key Features)

* **🤖 适配多色系统**：配合多色供料系统（如 AMS）进行层内换料即可实现。
* **📏 告别繁琐测试**：**无需测定 TD 值**，也不用打印庞大的调色盘矩阵。仅需拍摄一张校准薄片，即可通过算法自动计算透光参数。
* **🎨 耗材零门槛**：**手里有什么就用什么！** 无论是 2 色、8 色还是更多，更换新耗材仅需一次简单的单色校准。

---

## 🛠️ 耗材选择指南

> **核心原则：黑色无所谓，白色必须稳定。**

* **关于黑色 (Black)**：市面上大多数厂家的黑色耗材遮光能力极强（1-2 层即可完全遮盖），因此品牌和批次影响不大。
* **关于白色 (White)**：**这是唯一有硬性要求的耗材**。请确保校准时使用的白色耗材与最终打印时完全一致。
* **其他颜色 (CMY/RGB/任意颜色)**：主要关注色域覆盖，可任意选择品牌任意搭配。

---

## 🚀 使用流程
### Step 0: 安装与配置环境

首先确保电脑已安装 Python 3.8+。打开终端依次运行：

```bash
# 1. 克隆仓库
git clone https://github.com/borealis-zhe/ChromaStack.git
cd ChromaStack

# 2. 安装依赖
pip install -r requirements.txt
```

### Step 1: 单耗材校正 (Calibration)
1.  **打印校准件**：运行 `filament_cali/generate_cali_stl.py` 生成校准薄片并打印。
2.  **拍照采样**：
    * 将打印好的薄片放在 A4 纸上。
    * 确保**黑色底在左边**，厚度从下到上依次增加。
    * 在光线充足且均匀的环境下拍照（**尽量避免阴影**），如下图所示：
    ![校准图片](res/img/cyan_008.png)
3.  **计算参数**：
    * 运行校正脚本：`filament_cali/KS_calibration.py`。
    * 脚本将自动计算出该耗材的 **K/S 值**。
4.  **更新数据库**：
    * 将计算出的值复制到 `my_filament.json` 中。
    * **必填项**：`"Name"`, `"FILAMENT_K"`, `"FILAMENT_S"`。
    * *注：主程序将根据 `"Name"` 字段匹配耗材。*

### Step 2: 自动选色 (可选/Optional)
*如果你已经确定了耗材组合，可跳过此步。*

不确定哪几种颜色凑在一起效果最好？运行 `AutoSelector.py`，算法会根据你的图片和库存耗材库，计算并推荐最佳的颜色层叠组合。

### Step 3: 生成模型
打开主程序 `ChromaStackStudio.py`，在头部配置区修改以下参数：

```python
# --- 基础设置 ---
img_path = "path/to/image.png"      # 图片路径 (建议 PNG，透明背景部分将不会被打印)
filament_db = "my_filament.json"    # 耗材库路径
model_width = 100                   # 模型宽度 (mm)

# --- 打印设置 ---
layer_height = 0.08                 # 颜色层层高 (推荐 0.08 或 0.1)
base_height = 0.4                   # 底座厚度
mixing_layers = 5                   # 混色层数 (推荐: 0.08mm*5层 或 0.1mm*4层)

# --- 耗材指定 ---
# 复制 AutoSelector 的结果或手动指定
selected_filaments = [...]
```

### Step 4: 切片与打印 (Slicing)
将生成的 `.3mf` 导入切片软件（如 Orca Slicer / Bambu Studio），注意调整以下参数：

* **层高 (Layer Height)**：必须与脚本中设置的一致（0.08mm 或 0.1mm）。
* **首层层高 (First Layer)**：按需修改（0.08mm 或 0.1mm）。
* **墙生成器 (Wall Generator)**：**强烈推荐使用 Arachne**，以获得更好的细节填充。
* **冷却 (Cooling)**：对于大面积模型，建议适当调高风扇或增加辅助冷却，避免边缘翘曲。

## 📜 开源协议 (License)

本项目采用 **GPL-3.0** 开源协议。
* ✅ 你可以自由修改源码、分发副本。
* ✅ 任何基于本项目的修改版本必须同样开源（传染性）。
* ⚠️ 本项目生成的模型文件需**标注出处**。

**商业豁免**：个人创作者、街边摊贩、小型私营企业可自由使用本软件生成模型并销售实体打印品。

## 🙌 致谢 (Acknowledgments)

特别感谢以下项目提供的灵感：
* [HueForge](https://shop.thehueforge.com/) - 3D 打印浮雕画的先行者
* Also try [Lumina_Layers](https://github.com/MOVIBALE/Lumina-Layers) ! 同样优秀的开源项目
* MakerWorld中国站用户[*掐丝特工008*](https://makerworld.com.cn/zh/@user_3656819985) 所发布的模型
* @LubanDaddy所发布的[多色3D平面模型生成器](https://makingis.fun/multicolor/)

本项目代码为个人独立开发，热烈欢迎提交 PR 参与改进！(PRs are welcome!)  
**如果你觉得这个项目有用，请点亮右上角的 Star ⭐️ 支持一下！**

