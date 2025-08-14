import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from modules.model.matcher import HungarianMatcher
from modules.utils.box_ops import box_giou, xywh_to_xyxy


class SetCriterion(Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        class_weight: Tensor = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = matcher.get_cost_weight_dict()
        self.eos_coef = eos_coef
        
        if class_weight is None:
            self.empty_weight = torch.ones(self.num_classes + 1)
        else:
            self.empty_weight = class_weight
            
        self.empty_weight[0] = eos_coef
        
    def forward(
        self, 
        pred_class: Tensor,
        pred_bbox: Tensor,
        gt_class: Tensor,
        gt_bbox: Tensor
    ) -> Tensor:
        """
        Args:
            pred_class (Tensor): shape [B, Q, C + 1]
            pred_bbox (Tensor): shape [B, Q, 4]
            gt_class (Tensor): shape [B, G]
            gt_bbox (Tensor): shape [B, G, 4]
        Returns:
            Tensor: total loss
        """
        
        B = pred_class.size(0)
        self.empty_weight = self.empty_weight.to(pred_class.device)
        row_inds, col_inds = self.matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)   # inds: [B, min(G, Q)]
        batch_idx = torch.arange(B, device=pred_class.device).view(-1, 1).expand_as(row_inds)

        # 1. class loss
        pred_class_permuted = pred_class[batch_idx, row_inds]
        gt_class_permuted = gt_class[batch_idx, col_inds]
        
        class_loss = F.cross_entropy(
            pred_class_permuted.flatten(0, 1), 
            gt_class_permuted.flatten(), 
            self.empty_weight
        )
        
        # 2. box loss
        valid_mask = gt_class_permuted.greater(0)
        valid_num = valid_mask.sum().clamp(min=1)
        pred_bbox_permuted = pred_bbox[batch_idx, row_inds][valid_mask]
        gt_bbox_permuted = gt_bbox[batch_idx, col_inds][valid_mask]
        bbox_loss = F.l1_loss(pred_bbox_permuted, gt_bbox_permuted, reduction="none").sum() / valid_num
        
        # 3 giou loss
        box_matrix = box_giou(
            xywh_to_xyxy(pred_bbox_permuted), 
            xywh_to_xyxy(gt_bbox_permuted)
        )
        giou = torch.diagonal(box_matrix, dim1=-2, dim2=-1)
        giou_loss = torch.sum(1 - giou) / valid_num
        
        total_loss = (
            self.weight_dict["class"] * class_loss +
            self.weight_dict["bbox"] * bbox_loss +
            self.weight_dict["giou"] * giou_loss
        )

        return total_loss