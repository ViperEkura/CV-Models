## CV-Models

这是用于学习记录CV模型而创建的项目，致力于实现经典计算机视觉模型的PyTorch复现，包含目标检测（DETR）、骨干网络（ResNet）等模块化实现，支持灵活配置和快速实验验证。

### 功能特点
- **模型模块**：
  - DETR目标检测模型
  - ResNet系列骨干网络(ResNet18~ResNet152)
- **数据集支持**：
  - COCO/VOC数据集自动下载与预处理
  - 支持自定义数据集加载
- **损失函数**：
  - DETR专用SetCriterion损失
  - Focal Loss
  - MOTR专用多任务损失
- **优化器**：
  - Muon优化器（基于牛顿-舒尔茨迭代的正交化优化）
  - 支持混合优化策略（Muon+AdamW）
- **工具库**：
  - 目标检测可视化工具
  - 盒子操作工具（IoU/GIoU计算）
  - 分类/检测训练模板

### 目录结构

```bash
.
│   __init__.py
│
├───dataset
│   │   dataset.py
│   │   download.py
│   │   __init__.py
│
├───loss
│   │   detr_loss.py
│   │   focal_loss.py
│   │   motr_loss.py
│   │   __init__.py
│
├───model
│   │   detr.py
│   │   matcher.py
│   │   motr.py
│   │   resnet.py
│   │   transfomer.py
│   │   __init__.py
│
├───optim
│       muon.py
│
└───utils
        box_ops.py
        classification.py
        detection.py
        plot_detection.py
        __init__.py

```
