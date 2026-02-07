# ChromaStack Studio: Just Print What You See

<div align="center">

**A Multi-Color Layered Model Generator for FDM Printers**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)  
[中文文档 / Chinese Documentation](README.md)

</div>

---

**ChromaStack** is a **multi-color layered model generator** designed specifically for FDM 3D printers. It uses algorithms to convert 2D images into layer-based multi-color 3D models. By utilizing the varying transmissivity of different colored filaments, it allows you to print photo-realistic light and shadow effects without expensive hardware.

### ✨ Key Features

* **🤖 Compatible with Multi-Color Systems**: Works seamlessly with multi-material systems (like AMS) using intra-layer filament changes.
* **📏 No Tedious Testing**: **Say goodbye to measuring TD values** or printing massive filament step-test matrices. Simply take a photo of a small calibration chip, and the algorithm automatically calculates the transmission parameters.
* **🎨 Zero Filament Threshold**: **Use whatever you have on hand!** Whether it's 2 colors, 8 colors, or more, adding a new filament requires only a simple one-time single-color calibration.

---

## 🛠️ Filament Selection Guide

> **Core Principle: Black is less critical, but White must be consistent.**

* **About Black**: Most black filaments on the market have extremely high opacity (fully blocking light in 1-2 layers), so the brand or batch rarely matters.
* **About White**: **This is the only filament with strict requirements.** Please ensure the white filament used during calibration is identical to the one used for the final print.
* **Other Colors (CMY/RGB/Any)**: Focus mainly on color gamut coverage; you are free to choose any brand or combination.

---

## 🚀 Usage Workflow

### Step 0: Installation & Environment Setup

Ensure your computer has Python 3.8+ installed. Open your terminal and run the following commands:

```bash
# 1. Clone the repository
git clone [https://github.com/borealis-zhe/ChromaStack.git](https://github.com/borealis-zhe/ChromaStack.git)
cd ChromaStack

# 2. Install dependencies
pip install -r requirements.txt
```
### Step 1: Single Filament Calibration
1.  **Print Calibration Chip**: Run `filament_cali/generate_cali_stl.py` to generate the calibration model and print it.
2.  **Photo Sampling**:
    * Place the printed chip on a sheet of A4 paper.
    * Ensure the **black base is on the left**, with thickness increasing from bottom to top.
    * Take a photo in a well-lit environment with uniform lighting (**avoid shadows as much as possible**), as shown below:
    ![Calibration Image](res/img/cyan_008.png)
3.  **Calculate Parameters**:
    * Run the calibration script: `filament_cali/KS_calibration.py`.
    * The script will automatically calculate the **K/S values** for that filament.
4.  **Update Database**:
    * Copy the calculated values into `my_filament.json`.
    * **Required Fields**: `"Name"`, `"FILAMENT_K"`, `"FILAMENT_S"`.
    * *Note: The main program matches filaments based on the `"Name"` field.*

### Step 2: Auto Color Selector (Optional)
*Skip this step if you have already decided on your filament combination.*

Unsure which colors work best together? Run `AutoSelector.py`. The algorithm will analyze your image and your filament inventory to recommend the optimal layered color combination.

### Step 3: Generate Model
Open the main program `ChromaStackStudio.py` and modify the parameters in the configuration section at the top:

```python
# --- Basic Settings ---
img_path = "path/to/image.png"      # Image path (PNG recommended; transparent backgrounds will not be printed)
filament_db = "my_filament.json"    # Filament database path
model_width = 100                   # Model width (mm)

# --- Print Settings ---
layer_height = 0.08                 # Layer height (0.08mm or 0.1mm recommended)
base_height = 0.4                   # Base thickness
mixing_layers = 5                   # Color mixing layers (Recommended: 0.08mm * 5 layers or 0.1mm * 4 layers)

# --- Filament Selection ---
# Copy the result from AutoSelector or manually specify
selected_filaments = [...]
```

### Step 4: Slicing & Printing

Import the generated `.3mf` file into your slicer (e.g., Orca Slicer / Bambu Studio) and adjust the following parameters:

* **Layer Height**: Must match the setting in the script (0.08mm or 0.1mm).
* **First Layer Height**: Adjust as needed (0.08mm or 0.1mm).
* **Wall Generator**: **Strongly recommended to use Arachne**, for better detail filling.
* **Cooling**: For large-area models, it is recommended to increase fan speed or add auxiliary cooling to prevent warping at the edges.

---

## 📜 License

This project is licensed under the **GPL-3.0** License.

* ✅ You are free to modify the source code and distribute copies.
* ✅ Any modified version based on this project must also be open-sourced (Viral nature).
* ⚠️ Models generated using this software must **cite the source**.

**Commercial Exemption**: Individual creators, street vendors, and small private businesses are free to use this software to generate models and sell the physical printed products.

---

## 🙌 Acknowledgments

Special thanks to the following projects for their inspiration:

* [HueForge](https://shop.thehueforge.com/) - The pioneer of filament painting.
* [Lumina_Layers](https://github.com/MOVIBALE/Lumina-Layers) - Also an excellent open-source project (Give it a try!).
* [掐丝特工008](https://makerworld.com.cn/zh/@user_3656819985) on MakerWorld CN for the models.
* [Multi-color 3D flat model generator](https://makingis.fun/multicolor/) by @LubanDaddy.

This project is independently developed. **PRs are welcome!**  
**If you find this project useful, please give it a Star ⭐️ to support it!**

