## DETR

DETR(DEtection TRansformer) 是由 Facebook AI(现 Meta AI)在 2020 年提出的目标检测模型。它采用 Transformer 架构 替代了传统的卷积方法(如 Faster R-CNN 或 YOLO)，为物体检测领域带来了全新的思路

### 1. DETR 的核心特点
1. 基于 Transformer 的端到端检测： 传统检测器依赖锚框(anchor boxes)或区域提议(region proposals)，而 DETR 将检测视为 集合预测问题。通过 Transformer 编码器-解码器 直接并行预测一组固定数量的目标。

2. 二分图匹配损失(Bipartite Matching Loss)： 使用匈牙利算法将预测框与真实框唯一匹配，避免重复预测。损失函数同时优化类别预测和边界框回归


### 2. 模型结构
1. CNN 主干网络: 将输入图像转换为低分辨率高维特征图，如果使用 ResNet50，则输入特征图大小为H x W x 3，输出特征图大小为 (H/32) x (W/32) x 2048

2. 特征变换与位置编码： 

- 1×1 卷积降维：将主干网络输出的高维特征通道数减少(如 2048 → 256)，得到 d×H/32×W/32(d=256)
- 展平为序列：将空间特征图展平为 d×N 的序列（N=(H/32)×(W/32)，作为 Transformer 的输入
- 位置编码：添加可学习的位置编码参数，为特征提供空间位置信息。

### 3. Transformer 模块
使用自注意力模块构建 Transformer 模块，包含编码器(Encoder)和解码器(Decoder)
