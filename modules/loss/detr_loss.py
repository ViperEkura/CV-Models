import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from modules.model.detr import HungarianMatcher, _box_giou


class SetCriterion(Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = matcher.get_cost_weight_dict()
        self.eos_coef = eos_coef
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[0] = eos_coef

    def forward(
        self, 
        pred_class: Tensor,
        pred_bbox: Tensor,
        gt_class: Tensor,
        gt_bbox: Tensor
    ) -> Tensor:
        # [batch_size, num_queries], [batch_size, num_gt_boxes]
        row_ind, col_ind = self.matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
        background_mask = row_ind.eq(-1)
        target_class = row_ind.detach().clone()
        target_class[background_mask] = self.num_classes
        
        self.empty_weight = self.empty_weight.to(device=pred_class.device)
        target_class = target_class.to(device=pred_class.device)
        
        loss_class = F.cross_entropy(
            input=pred_class.transpose(1, 2), 
            target=target_class,
            weight=self.empty_weight
        )
        
        pred_bbox_selected = pred_bbox[~background_mask, :]
        gt_bbox_selected = gt_bbox.view(-1, 4)
        valid_mask = background_mask.logical_not().to(device=gt_bbox.device)
        
        loss_bbox = F.l1_loss(pred_bbox_selected.flatten(), gt_bbox_selected.flatten(), reduction='none').sum()  
        loss_giou = 1 - _box_giou(pred_bbox_selected, gt_bbox_selected)
        
        loss_class = (loss_class * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        loss_bbox = (loss_bbox * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        loss_giou = loss_giou.sum() / valid_mask.sum().clamp(min=1)
        
        total_loss = (
            self.weight_dict["class"] * loss_class +
            self.weight_dict["bbox"] * loss_bbox +
            self.weight_dict["giou"] * loss_giou
        )
        
        return total_loss