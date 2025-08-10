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
            pred_class (Tensor): shape [batch_size, num_queries, num_classes + 1]
            pred_bbox (Tensor): shape [batch_size, num_queries, 4]
            gt_class (Tensor): shape [batch_size, num_gt_boxes]
            gt_bbox (Tensor): shape [batch_size, num_gt_boxes, 4]
        Returns:
            Tensor: total loss
        """

        batch_size, num_queries = pred_class.size(0), pred_class.size(1)

        row_inds, col_inds = self.matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
        self.empty_weight = self.empty_weight.to(pred_class.device)
        # rol_inds: [batch_size, num_queries], col_inds: [batch_size, num_gt_boxes]
        
        # 1. class loss compute
        gt_class_expanded = torch.full(
            (batch_size, num_queries),
            fill_value=0, 
            device=gt_class.device,
            dtype=gt_class.dtype
        )
        gt_class_expanded = gt_class_expanded.scatter(dim=1, index=col_inds, src=gt_class)
        
        class_loss = F.cross_entropy(
            pred_class.flatten(0, 1),           # [batch_size * num_queries, num_classes + 1]
            gt_class_expanded.flatten(),        # [batch_size * num_queries]
            self.empty_weight
        )
        
        # 2. compute for bbox
        gt_bbox_expanded = torch.full(
            (batch_size, num_queries, 4),
            fill_value=0, 
            device=gt_bbox.device,
            dtype=gt_bbox.dtype
        )
        expand_col_inds = col_inds.unsqueeze(-1).expand(-1, -1, 4)
        gt_bbox_expanded = gt_bbox_expanded.scatter(dim=1, index=expand_col_inds, src=gt_bbox)
        
        match_mask = row_inds.eq(-1).unsqueeze(-1)
        obj_mask = gt_class_expanded.eq(0).unsqueeze(-1)
        
        # [batch_size, num_queries, 4]
        pred_bbox_matched = torch.masked_fill(input=pred_bbox, mask=match_mask | obj_mask, value=0)
        gt_bbox_matched = torch.masked_fill(input=gt_bbox_expanded, mask=match_mask, value=0)
        bbox_loss = F.l1_loss(pred_bbox_matched.flatten(), gt_bbox_matched.flatten(), reduction="none").sum() / obj_mask.sum().clamp(min=1)
        
        # 3. compute for giou
        box_giou = _box_giou(pred_bbox_matched, gt_bbox_matched)
        giou_loss = 1 - torch.diagonal(box_giou, dim1=-2, dim2=-1).mean()

        total_loss = (
            self.weight_dict["class"] * class_loss +
            self.weight_dict["bbox"] * bbox_loss +
            self.weight_dict["giou"] * giou_loss
        )

        return total_loss