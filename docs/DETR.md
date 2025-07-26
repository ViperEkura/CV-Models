## DETR

DETR(DEtection TRansformer) 是由 Facebook AI(现 Meta AI)在 2020 年提出的目标检测模型。它采用 Transformer 架构替代了传统的卷积方法(如 Faster R-CNN 或 YOLO)，为物体检测领域带来了全新的思路

### 1. DETR 的核心特点
1. 基于 Transformer 的端到端检测：  
   - 传统检测器依赖锚框(anchor boxes)或区域提议(region proposals)
   - DETR 将检测视为集合预测问题
   - 通过 Transformer 编码器-解码器直接并行预测一组固定数量的目标

2. 二分图匹配损失(Bipartite Matching Loss)：  
   - 使用匈牙利算法将预测框与真实框唯一匹配
   - 避免重复预测
   - 损失函数同时优化类别预测和边界框回归

### 2. 模型结构
1. CNN 主干网络:  
   - 输入图像转换为低分辨率高维特征图
   - 使用 ResNet50 时：输入特征图大小 H×W×3 → 输出特征图大小 (H/32)×(W/32)×2048

2. 特征变换与位置编码：  
   - 1×1 卷积降维：2048 → 256
   - 展平为 d×N 序列（N=(H/32)×(W/32)）
   - 添加可学习的位置编码参数

3. Transformer 模块:  
   - 自注意力机制构建编码器/解码器
   - 支持多头注意力（默认 8 头）

### 3. 训练方法
1. 损失函数：  
   - 分类损失：交叉熵损失 + 类别权重平衡
   - 边界框损失：L1 损失 + GIoU 损失
   - 总损失 = 分类损失 + $\lambda_{bbox}$ × 边界框损失 + $\lambda_{giou}$ × GIoU 损失

2. 优化策略：  
   - AdamW 优化器（学习率 1e-4）
   - 余弦退火学习率调度（T_max=50）
   - 非骨干网络参数使用 1x 学习率，骨干网络 0.1x

3. 数据增强：  
   - 随机翻转（水平方向）
   - 图像归一化：[0.485,0.456,0.406] 均值，[0.229,0.224,0.225] 标准差

### 4. 使用说明
1. 数据准备：  
   ```bash
   # COCO 数据集目录结构
   data/
   └── coco/
       ├── annotations/
       │   ├── instances_train2017.json
       │   └── instances_val2017.json
       └── images/
           ├── train2017/
           └── val2017/
   ```
