import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Tuple, Dict, Union
from modules.utils.box_ops import box_iou
from modules.model.detr import PostProcess


def eval_detection(
    model: Union[nn.Module, Callable[..., Tuple[Tensor, Tensor]]], 
    dataloader: torch.utils.data.DataLoader, 
    device: str = "cuda",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    num_classes: int = 80
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    all_annotations = [[] for _ in range(num_classes)]
    all_detections = [[] for _ in range(num_classes)]
    
    for images, labels, boxes in dataloader:
        scores, pred_labels, pred_boxes = PostProcess.process(model, images, score_threshold, device=device)
        
        for i in range(len(images)):
            img_annotations = {'boxes': boxes[i], 'labels': labels[i]}
            img_detections = {'scores': scores[i],'labels': pred_labels[i], 'boxes': pred_boxes[i]}
            

def compute_ap():
    pass
