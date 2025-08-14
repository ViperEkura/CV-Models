import lap
import torch
import numpy as np

from torch import Tensor
from typing import List, Tuple, Dict
from modules.utils.box_ops import box_giou, xywh_to_xyxy


def jonker_volgenant(cost_matrix: Tensor) -> Tuple[List[int], List[int]]:
    """
    Args:
        cost_matrix (Tensor): shape [num_queries, num_gt_boxes]
    Returns:
        Tuple[List[int], List[int]]:
            row_indices (List[int]): row indices of the optimal assignment
            col_indices (List[int]): column indices of the optimal assignment
    """
    assert cost_matrix.ndim == 2
    cost_matrix = cost_matrix.cpu().detach().numpy().astype(np.float64) 
    _, row_inds, _ = lap.lapjv(cost_matrix, extend_cost=True)
    row_indices, col_indices = [], []
    
    for i in range(len(row_inds)):
        if row_inds[i] != -1:
            row_indices.append(i)
            col_indices.append(row_inds[i])
    
    return row_indices, col_indices 



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
            C = C.reshape(pred_class.shape[1], -1)   # [num_queries, num_gt_boxes]

            row_ind, col_ind = jonker_volgenant(C)
            row_inds.append(torch.tensor(row_ind).long())
            col_inds.append(torch.tensor(col_ind).long())
        
        row_inds_cat = torch.stack(row_inds).to(device=device) # [batch_size, num_queries]
        col_inds_cat = torch.stack(col_inds).to(device=device) # [batch_size, num_gt_boxes]
        
        return row_inds_cat, col_inds_cat
    
    def _compute_class_cost(self, pred_class: Tensor, gt_class: Tensor) -> Tensor:
        # fix cross_entropy -> softmax + nll_loss -> - log_softmax
        pred_prob = pred_class.log_softmax(-1)              # [num_queries, num_classes+1]
        cost_class = -pred_prob[:, gt_class]                # [num_queries, num_gt_boxes]
        return cost_class
    
    def _compute_bbox_cost(self, pred_bbox, gt_bbox) -> Tensor:
        cost_bbox = torch.cdist(pred_bbox, gt_bbox, p=1) # [num_queries, num_gt_boxes]
        return cost_bbox
    
    def _compute_giou_cost(self, pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
        giou = box_giou(xywh_to_xyxy(pred_bbox), xywh_to_xyxy(gt_bbox)) # [num_queries, num_gt_boxes]
        cost_giou = 1 - giou
        return cost_giou

    def get_cost_weight_dict(self) -> Dict[str, float]:
        return {"class": self.cost_class, "bbox": self.cost_bbox,"giou": self.cost_giou}