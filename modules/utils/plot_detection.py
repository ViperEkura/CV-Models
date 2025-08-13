import torch
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Module
from typing import Callable, Literal, Union
from PIL import Image
import torchvision.transforms as T
import numpy as np

def plot_detection(
    model: Union[Module, Callable[..., Tensor]], 
    image: Tensor, 
    confidence_threshold=0.5, 
    device: Literal["cpu", "cuda"] = "cpu"
):
    """
    使用DETR模型进行目标检测并绘制结果
    
    Args:
        model: DETR模型
        image: 输入图像(PIL Image或Tensor)
        confidence_threshold: 置信度阈值
        device: 计算设备
        
    Returns:
        绘制了检测框的图像
    """
    # 如果输入是PIL图像，转换为tensor
    if isinstance(image, Image.Image):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)
    else:
        img_tensor = image.unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        pred_class, pred_bbox = model(img_tensor)
        print(pred_class)
    
    # 处理预测结果
    pred_class = pred_class.softmax(-1)[0, :, :-1]  # 移除背景类
    pred_bbox = pred_bbox[0]
    
    # 获取置信度和类别
    pred_scores, pred_labels = pred_class.max(-1)
    
    # 筛选高置信度的预测
    keep = pred_scores > confidence_threshold
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    pred_bbox = pred_bbox[keep]
    
    # 如果输入是PIL图像，转换为可绘制的numpy数组
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        # 反归一化
        img_array = image.cpu().numpy()
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = (img_array * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
    
    # 创建绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img_array)
    
    # 绘制边界框
    for score, label, (cx, cy, w, h) in zip(pred_scores, pred_labels, pred_bbox):
        # 将相对坐标转换为绝对坐标
        cx, cy, w, h = cx.item(), cy.item(), w.item(), h.item()
        xmin = (cx - w/2) * img_array.shape[1]
        ymin = (cy - h/2) * img_array.shape[0]
        xmax = (cx + w/2) * img_array.shape[1]
        ymax = (cy + h/2) * img_array.shape[0]
        
        # 绘制矩形框
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # 添加标签和置信度
        ax.text(xmin, ymin, f'{label.item()}: {score.item():.2f}',
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12, color='white')
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_array