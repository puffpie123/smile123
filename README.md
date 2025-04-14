# 车牌识别系统

本项目是一个基于 Python 实现的 **车牌识别系统**，利用深度学习模型对图像或视频中的车辆车牌进行检测和识别。系统采用 YOLOv8 模型进行车牌检测，并使用自定义模型进行字符识别，同时通过 Tkinter 提供图形用户界面（GUI）以便用户交互。

---

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [代码说明](#代码说明)
  - [主要组件](#主要组件)
  - [关键函数](#关键函数)
  - [GUI 实现](#gui-实现)
- [模型详情](#模型详情)
- [输出结果](#输出结果)
- [局限性](#局限性)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 功能特性

- **车牌检测**：使用 YOLOv8 检测图像或视频中的单层和双层车牌。
- **字符识别**：识别车牌的字符和颜色，支持单层和双层车牌。
- **GUI 界面**：基于 Tkinter 的直观界面，支持选择输入文件（图像或视频）、配置参数和显示结果。
- **实时视频处理**：逐帧处理视频并在 GUI 中显示结果。
- **暂停/继续功能**：支持视频处理过程中的暂停和继续。
- **结果可视化**：在图像或视频上绘制车牌框和识别结果，并保存到指定文件夹。

---

## 环境要求

- **Python**：3.8 或更高版本
- **依赖库**：
  - `torch`（PyTorch 用于深度学习模型）
  - `opencv-python`（OpenCV 用于图像/视频处理）
  - `numpy`（数值运算）
  - `Pillow`（PIL 用于 GUI 中的图像处理）
  - `ultralytics`（YOLOv8 模型支持）
- **硬件**：
  - GPU（可选，支持 CUDA 加速推理）
  - CPU（足以应对大多数场景）
- **预训练模型**：
  - YOLOv8 检测模型（`yolov8s.pt`）
  - 车牌识别模型（`plate_rec_color.pth`）

---

## 安装步骤

1. **克隆仓库**：
   ```bash
   git clone <仓库地址>
   cd license-plate-recognition
   ```

2. **安装依赖**：
   创建虚拟环境并安装所需包：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 下：venv\Scripts\activate
   pip install torch opencv-python numpy pillow ultralytics
   ```

3. **下载预训练模型**：
   - 将 `yolov8s.pt` 和 `plate_rec_color.pth` 放入 `weights/` 目录。
   - YOLOv8 模型可从 [Ultralytics YOLOv8 仓库](https://github.com/ultralytics/ultralytics) 下载。
   - 识别模型（`plate_rec_color.pth`）为自定义训练模型（本仓库未提供）。

4. **目录设置**：
   确保以下结构：
   ```
   license-plate-recognition/
   ├── weights/
   │   ├── yolov8s.pt
   │   ├── plate_rec_color.pth
   ├── plate_recognition/
   │   ├── plate_rec.py
   │   ├── double_plate_split_merge.py
   ├── fonts/
   │   ├── cv_puttext.py
   ├── Model2.py
   ├── README.md
   ```

5. **运行程序**：
   ```bash
   python Model2.py
   ```

---

## 项目结构

```
license-plate-recognition/
├── weights/                    # 预训练模型权重
├── plate_recognition/          # 
