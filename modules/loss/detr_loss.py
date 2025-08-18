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
            gt_class (List[Tensor]): length = B
            gt_bbox (List[Tensor]): length = B
        Returns:
            Tensor: total loss
        """
        
        B = pred_class.size(0)
        self.empty_weight = self.empty_weight.to(pred_class.device)
        pred_class_list = [cls.squeeze(0) for cls in pred_class.split(1, dim=0)]
        pred_bbox_list = [box.squeeze(0) for box in pred_bbox.split(1, dim=0)]
        row_inds, col_inds = self.matcher.match(pred_class_list, pred_bbox_list, gt_class, gt_bbox)
        
        cls_losses, bbox_losses, giou_losses = 0, 0, 0 
        
        for b in range(B):
            pred_cls, pred_box = pred_class_list[b], pred_bbox_list[b]
            gt_cls, gt_box = gt_class[b], gt_bbox[b]
            row_ind, col_ind = row_inds[b], col_inds[b]
            
            target_cls = torch.zeros(
                *pred_cls.shape[:-1], 
                dtype=gt_cls.dtype, 
                device=gt_cls.device
            ) # [Q]
            target_cls[row_ind] = gt_cls[col_ind]
            
            cls_loss = F.cross_entropy(
                pred_cls, 
                target_cls, 
                weight=self.empty_weight, 
                reduction='mean'
            )
            
            pred_box_permuted = pred_box[row_ind] # [Q, 4]
            gt_box_permuted = gt_box[col_ind]     # [G, 4]
            
            box_loss = F.l1_loss(
                pred_box_permuted, 
                gt_box_permuted, 
                reduction='sum'
            ) / len(row_ind)
            
            pred_box_permuted = xywh_to_xyxy(pred_box_permuted)
            gt_box_permuted = xywh_to_xyxy(gt_box_permuted)
            giou = box_giou(pred_box_permuted, gt_box_permuted).diag()
            giou_loss = torch.mean(1 - giou)
            
            cls_losses += cls_loss
            bbox_losses += box_loss
            giou_losses += giou_loss
        
        total_loss = (
            self.weight_dict["class"] * cls_losses +
            self.weight_dict["bbox"] * bbox_losses +
            self.weight_dict["giou"] * giou_losses
        ) / B
        
        return total_loss
    