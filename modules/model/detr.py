import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple, Dict
from torch import Tensor
from modules.model.resnet import ResNet
import lap

def jonker_volgenant(cost_matrix: np.ndarray) -> Tuple[int, List[int], List[int]]:
    """
    Jonker-Volgenant算法
    """
    assert cost_matrix.ndim == 2
    return lap.lapjv(cost_matrix, extend_cost=True)

def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    将边界框的坐标从(x, y, w, h)转换为(x1, y1, x2, y2)
    """
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)

def _box_intersection(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两组边界框的交集区域面积
    """
    boxes1 = boxes1.unsqueeze(1)  # [num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, num_gt_boxes, 4]

    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)
    # [num_queries, num_gt_boxes]
    return w * h

def _box_union(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两组边界框的并集区域面积
    """
    boxes1 = boxes1.unsqueeze(1)  # [num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, num_gt_boxes, 4]

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter = _box_intersection(boxes1.squeeze(1), boxes2.squeeze(0))

    # [num_queries, num_gt_boxes]
    return area1 + area2 - inter

def _box_enclose_area(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算包含两组边界框的最小闭包区域面积
    """
    boxes1 = boxes1.unsqueeze(1)  # [num_queries, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, num_gt_boxes, 4]

    x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.max(boxes1[..., 3], boxes2[..., 3])

    w = x2 - x1
    h = y2 - y1
    # [num_queries, num_gt_boxes]
    return w * h

def _box_giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两个box之间的GIOU
    """
    boxes1 = _xywh_to_xyxy(boxes1)
    boxes2 = _xywh_to_xyxy(boxes2)

    inter = _box_intersection(boxes1, boxes2)
    union = _box_union(boxes1, boxes2)
    iou = inter / union

    enclose = _box_enclose_area(boxes1, boxes2)
    giou = iou - (enclose - union) / enclose
    return giou


class DETR(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        in_channel: int = 3, 
        hidden_dim: int = 256, 
        nheads: int = 8,
        num_encoder_layers: int = 6, 
        num_decoder_layers: int = 6,
        num_queries: int = 100,
    ):
        super().__init__()
        self.backbone = ResNet("resnet50", in_channel)
        self.conv = nn.Conv2d(2048, hidden_dim, 1) 
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)          # +1 for background
        self.linear_bbox = nn.Linear(hidden_dim, 4)                         # bbox(x,y,w,h)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))      # 50x50
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.num_queries = num_queries

    def forward(self, inputs: Tensor):
        x: Tensor = self.backbone(inputs)       # [batch_size, 2048, H, W]
        h: Tensor = self.conv(x)                # [batch_size, hidden_dim, H, W]

        H, W = h.shape[-2:]
        B = h.shape[0]
        
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)   # [1, H*W, hidden_dim]
        
        h = h.flatten(2).transpose(1, 2)        # [batch_size, H*W, hidden_dim]
        h = self.transformer(
            src=pos + h,                                                # [batch_size, H*W, hidden_dim]
            tgt=self.query_pos.unsqueeze(0).repeat(B, 1, 1)             # [batch_size, num_queries, hidden_dim]
        )                                                               # output: [batch_size, num_queries, hidden_dim]
        
        pred_class = self.linear_class(h)                               # [batch_size, num_queries, num_classes+1]
        pred_bbox = torch.sigmoid(self.linear_bbox(h))                  # [batch_size, num_queries, 4]
    
        return pred_class, pred_bbox


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
        batch_size = pred_class.size(0)
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
        
        row_inds_cat = torch.stack(row_inds) # [batch_size, num_queries]
        col_inds_cat = torch.stack(col_inds) # [batch_size, num_gt_boxes]
        
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


