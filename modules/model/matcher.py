import lap
import torch
import numpy as np

from torch import Tensor
from typing import List, Tuple, Dict, Literal


def jonker_volgenant(cost_matrix: np.ndarray) -> Tuple[int, List[int], List[int]]:
    """
    Args:
        cost_matrix (np.ndarray): shape [num_queries, num_gt_boxes]
    Returns:
        Tuple[int, np.ndarray, np.ndarray]:
            total_cost (int): cost of the assignment
            row_indices (np.ndarray): row indices of the optimal assignment
            col_indices (np.ndarray): column indices of the optimal assignment
    """
    assert cost_matrix.ndim == 2
    return lap.lapjv(cost_matrix, extend_cost=True)

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

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)
    
    return w * h

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
    

    return area1 + area2 - inter

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

    return w * h

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
    
    iou = inter / (union + epsilon)
    giou = iou - (enclose - union) / (enclose + epsilon)

    return giou

class HungarianMatcher:
    def __init__(
        self, 
        cost_class: float = 1.0, 
        cost_bbox: float = 1.0, 
        cost_giou: float = 1.0
    ):
        self.cost_class = cost_class 
        self.cost_bbox = cost_bbox 
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def match(
        self, 
        pred_class: Tensor, 
        pred_bbox: Tensor, 
        gt_class: Tensor, 
        gt_bbox: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            pred_class (Tensor): shape [batch_size, num_queries, num_classes + 1]
            pred_bbox (Tensor): shape [batch_size, num_queries, 4]
            gt_class (Tensor): shape [batch_size, num_gt_boxes]
            gt_bbox (Tensor): shape [batch_size, num_gt_boxes, 4]
        Returns:
            Tuple[Tensor, Tensor]: 
                row_inds (Tensor): shape [batch_size, num_queries],  
                col_inds (Tensor): shape [batch_size, num_gt_boxes]
        """
        
        batch_size = pred_class.size(0)
        device = pred_class.device
        row_inds, col_inds = [], []
        
        for i in range(batch_size):
            gt_class_i = gt_class[i]      # [num_gt_boxes]
            gt_bbox_i = gt_bbox[i]        # [num_gt_boxes, 4]

            cost_class = self._compute_class_cost(pred_class[i], gt_class_i)
            cost_bbox = self._compute_bbox_cost(pred_bbox[i], gt_bbox_i)
            cost_giou = self._compute_giou_cost(pred_bbox[i], gt_bbox_i)
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.reshape(pred_class.shape[1], -1).cpu().numpy().astype(np.float64)  # [num_queries, num_gt_boxes]

            _, row_ind, col_ind = jonker_volgenant(C)
            row_inds.append(torch.from_numpy(row_ind).long())
            col_inds.append(torch.from_numpy(col_ind).long())
        
        row_inds_cat = torch.stack(row_inds).to(device=device) # [batch_size, num_queries]
        col_inds_cat = torch.stack(col_inds).to(device=device) # [batch_size, num_gt_boxes]
        
        return row_inds_cat, col_inds_cat
    
    def _compute_class_cost(self, pred_class: Tensor, gt_class: Tensor) -> Tensor:
        pred_prob = pred_class.softmax(-1)                  # [num_queries, num_classes+1]
        cost_class = -pred_prob[:, gt_class]                # [num_queries, num_gt_boxes]
        return cost_class
    
    def _compute_bbox_cost(self, pred_bbox, gt_bbox) -> Tensor:
        cost_bbox = torch.cdist(pred_bbox, gt_bbox, p=1)    # [num_queries, num_gt_boxes]
        return cost_bbox
    
    def _compute_giou_cost(self, pred_bbox: Tensor, gt_bbox: Tensor):
        giou = _box_giou(pred_bbox, gt_bbox)                # [num_queries, num_gt_boxes]
        cost_giou = 1 - giou
        return cost_giou

    def get_cost_weight_dict(self) -> Dict[str, float]:
        return {"class": self.cost_class, "bbox": self.cost_bbox,"giou": self.cost_giou}