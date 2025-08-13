from typing import Literal
from torch import Tensor
import torch


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Args:
        boxes (Tensor): shape [num_queries, 4]
    Returns:
        boxes (Tensor): shape [num_queries, 4]
    """
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)

def _box_intersection(
    boxes1: Tensor, 
    boxes2: Tensor,
    style: Literal["xywh", "xyxy"] = "xyxy"
) -> Tensor:
    """
    Args:
        boxes1 (Tensor): shape [..., num_queries, 4]
        boxes2 (Tensor): shape [..., num_gt_boxes, 4],
        style: Literal["xywh", "xyxy"]
    Returns:
        Tensor: shape [..., num_queries, num_gt_boxes]
    """
    
    if style == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)
    
    boxes1 = boxes1.unsqueeze(-2)  # [..., num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(-3)  # [..., 1, num_gt_boxes, 4]

    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    w = x2 - x1
    h = y2 - y1
    
    return (w * h).clamp(min=0)

def _box_union(
    boxes1: Tensor, 
    boxes2: Tensor,
    style: Literal["xywh", "xyxy"] = "xyxy"
) -> Tensor:
    """
    Args:
        boxes1 (Tensor): shape [..., num_queries, 4]
        boxes2 (Tensor): shape [..., num_gt_boxes, 4]
        style: Literal["xywh", "xyxy"]
    Returns:
        Tensor: shape [..., num_queries, num_gt_boxes]
    """
    if style == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)
    
    inter = _box_intersection(boxes1, boxes2)
    boxes1 = boxes1.unsqueeze(-2)  # [..., num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(-3)  # [..., 1, num_gt_boxes, 4]

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    return (area1 + area2 - inter).clamp(min=0)

def _box_enclose_area(
    boxes1: Tensor, 
    boxes2: Tensor,
    style: Literal["xywh", "xyxy"] = "xyxy"
) -> Tensor:
    """
    Args:
        boxes1 (Tensor): shape [..., num_queries, 4]
        boxes2 (Tensor): shape [..., num_gt_boxes, 4]
        style: Literal["xywh", "xyxy"]
    Returns:
        Tensor: shape [..., num_queries, num_gt_boxes]
    """
    if style == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)
    
    boxes1 = boxes1.unsqueeze(-2)  # [..., num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(-3)  # [..., 1, num_gt_boxes, 4]

    x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.max(boxes1[..., 3], boxes2[..., 3])

    w = x2 - x1
    h = y2 - y1

    return (w * h).clamp(min=0)

def _box_giou(
    boxes1: Tensor, 
    boxes2: Tensor, 
    epsilon: float = 1e-6,
    style: Literal["xywh", "xyxy"] = "xyxy"
) -> Tensor:
    """
    Args:
        boxes1 (Tensor): shape [..., num_queries, 4]
        boxes2 (Tensor): shape [..., num_gt_boxes, 4]
        epsilon: float
        style: Literal["xywh", "xyxy"]
    Returns:
        Tensor: shape [..., num_queries, num_gt_boxes]
    """

    if style == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)

    inter = _box_intersection(boxes1, boxes2)
    union = _box_union(boxes1, boxes2)
    enclose = _box_enclose_area(boxes1, boxes2)
    
    iou = inter / torch.clamp(union, min=epsilon)
    giou = (iou - (enclose - union) / torch.clamp(enclose, min=epsilon)).clamp(min=-1, max=1)

    return giou