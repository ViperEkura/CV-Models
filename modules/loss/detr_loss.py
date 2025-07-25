import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from torch import Tensor
from modules.model.detr import HungarianMatcher, _box_giou

class SetCriterion(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[0] = eos_coef

    def forward(
        self, 
        pred_class: Tensor,
        pred_bbox: Tensor,
        gt_class: Tensor,
        gt_bbox: Tensor
    ) -> Dict[str, Tensor]:
        indices = self.matcher(pred_class, pred_bbox, gt_class, gt_bbox)
        
        idx = self._get_src_permutation_idx(indices)
        target_class_o = torch.cat([t[J] for t, (_, J) in zip(gt_class, indices)])
        target_class = torch.full(pred_class.shape[:2], 0, dtype=torch.int64, device=pred_class.device)
        target_class[idx] = target_class_o
        loss_class = F.cross_entropy(pred_class.transpose(1, 2), target_class, weight=self.empty_weight)

        src_boxes = pred_bbox[idx]
        target_boxes = torch.cat([t[i] for t, (i, _) in zip(gt_bbox, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / pred_bbox.shape[0]
        
        loss_giou = torch.diag(1 - _box_giou(src_boxes, target_boxes)).sum() / pred_bbox.shape[0]

        losses = {
            'loss_class': loss_class,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        total_loss = sum(self.weight_dict[k] * losses[k] for k in losses.keys())
        
        return total_loss, losses
    
    def _get_src_permutation_idx(self, indices):
        """
        将匹配索引转换为permuted预测张量索引
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx