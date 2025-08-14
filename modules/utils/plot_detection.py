import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from torch import Tensor
from torch.nn import Module
from typing import Callable, Literal, Union, Tuple, List  # 添加 List 类型
from modules.utils.postprocess import postprocess_detection
import torch.nn.functional as F  # 保留原有导入

def plot_detection(
    model: Union[Module, Callable[..., Tuple[Tensor, Tensor]]], 
    image: Tensor, 
    threshold=0.5, 
    device: Literal["cpu", "cuda"] = "cuda",
    class_names: List[str] = None  # 新增参数
):
    model.to(device)
    pred_class, pred_bbox = model(image.to(device))

    scores = F.softmax(pred_class, dim=-1)  # [B, Q, C+1]
    max_scores, preds = scores.max(dim=-1)  # [B, Q]
    keep_mask = max_scores > threshold       # [B, Q]
    print(max_scores, preds)
    print(pred_bbox)
    # 筛选预测框和对应的类别索引
    pred_bbox = pred_bbox[keep_mask]         # [K, 4]
    preds = preds[keep_mask]                 # [K]

    image_np = image.cpu().permute(1, 2, 0).numpy()
    H, W = image_np.shape[:2]  # 获取实际高度和宽度
    plt.imshow(image_np)
    
    ax = plt.gca()
    
    # 绘制筛选后的检测框（将归一化坐标转换为像素坐标）
    for i, box in enumerate(pred_bbox):
        # 解绑归一化坐标：将 xywh 转换为像素单位
        x_center_norm, y_center_norm, w_norm, h_norm = box.tolist()
        x_center = x_center_norm * W
        y_center = y_center_norm * H
        w = w_norm * W
        h = h_norm * H

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # 添加类别标签
        if class_names is not None:
            try:
                label = class_names[preds[i]]
            except IndexError:
                label = f"Class {preds[i]}"
        else:
            label = f"Class {preds[i]}"
        # 在框左上角添加标签
        ax.text(x1, y1, label, 
                color='white', 
                backgroundcolor='red', 
                fontsize=8, 
                verticalalignment='top')
    
    plt.axis('off')
    plt.show()