import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from modules.model.matcher import HungarianMatcher, _box_giou


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
        """
        Args:
            pred_class (Tensor): shape [B, Q, C + 1]
            pred_bbox (Tensor): shape [B, Q, 4]
            gt_class (Tensor): shape [B, G]
            gt_bbox (Tensor): shape [B, G, 4]
        Returns:
            Tensor: total loss
        """

        B, Q = pred_class.size(0), pred_class.size(1)

        row_inds, col_inds = self.matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
        self.empty_weight = self.empty_weight.to(pred_class.device)
        # rol_inds: [B, Q], col_inds: [B, G]
        
        # 1. class loss compute
        gt_class_expanded = torch.full(
            (B, Q),
            fill_value=0, 
            device=gt_class.device,
            dtype=gt_class.dtype
        )
        gt_class_expanded = gt_class_expanded.scatter(dim=1, index=col_inds, src=gt_class)
        
        class_loss = F.cross_entropy(
            pred_class.flatten(0, 1),           # [B * Q, C + 1]
            gt_class_expanded.flatten(),        # [B * Q]
            self.empty_weight
        )
        
        # 2. compute for bbox
        gt_bbox_expanded = torch.full(
            (B, Q, 4),
            fill_value=0, 
            device=gt_bbox.device,
            dtype=gt_bbox.dtype
        )
        expand_col_inds = col_inds.unsqueeze(-1).expand(-1, -1, 4)
        gt_bbox_expanded = gt_bbox_expanded.scatter(dim=1, index=expand_col_inds, src=gt_bbox)
        
        # [B, Q, 1]
        match_mask = row_inds.eq(-1).unsqueeze(-1)
        background_mask = gt_class_expanded.eq(0).unsqueeze(-1)
        invalid_mask = background_mask | match_mask
        
        # [B, Q, 4]
        pred_bbox_matched = torch.masked_fill(input=pred_bbox, mask=invalid_mask, value=0)
        gt_bbox_matched = torch.masked_fill(input=gt_bbox_expanded, mask=invalid_mask, value=0)
        num_objs = invalid_mask.logical_not().sum().clamp(min=1)
        bbox_loss = F.l1_loss(pred_bbox_matched.flatten(), gt_bbox_matched.flatten(), reduction="none").sum() / num_objs
        
        # 3. compute for giou
        valid_mask = invalid_mask.logical_not().expand(-1, -1, 4)               # [B, Q, 4]
        pred_bbox_selected = pred_bbox[valid_mask].view(-1, 4)                  # [N, 4]
        gt_bbox_selected = gt_bbox_expanded[valid_mask].view(-1, 4)             # [N, 4]
        giou = torch.diagonal(_box_giou(pred_bbox_selected , gt_bbox_selected)) # [N]
        giou_loss = (1 - giou).mean()
            
        # print(f"class_loss: {class_loss:.4f}, bbox_loss: {bbox_loss:.4f}, giou_loss: {giou_loss:.4f}")

        total_loss = (
            self.weight_dict["class"] * class_loss +
            self.weight_dict["bbox"] * bbox_loss +
            self.weight_dict["giou"] * giou_loss
        )

        return total_loss