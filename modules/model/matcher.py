import lap
import torch
import numpy as np

from torch import Tensor
from typing import List, Tuple, Dict
from modules.utils.box_ops import box_giou, xywh_to_xyxy


def jonker_volgenant(cost_matrix: Tensor) -> Tuple[List[int], List[int]]:
    """
    Solve linear assignment problem using Jonker-Volgenant algorithm
    
    Args:
        cost_matrix (Tensor): Cost matrix with shape [Q, G] where Q is number of predictions and G is number of ground truths
        
    Returns:
        indicates(Tuple[List[int], List[int]]):
            - row_indices (List[int]): Row indices of the optimal assignment (predictions)
            - col_indices (List[int]): Column indices of the optimal assignment (ground truths)
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
        cost_giou: float = 1.0,
        background_class_index: int = 0
    ):
        """
        Initialize Hungarian matcher for bipartite matching between predictions and ground truths
        
        Args:
            cost_class (float): Weight for classification cost
            cost_bbox (float): Weight for bounding box cost (L1 distance)
            cost_giou (float): Weight for GIoU cost
            background_class_index (int): Index of background class
        """
        self.cost_class = cost_class 
        self.cost_bbox = cost_bbox 
        self.cost_giou = cost_giou
        self.background_class_index = background_class_index
        
    @torch.no_grad()
    def match(
        self, 
        pred_class: List[Tensor], 
        pred_bbox: List[Tensor], 
        gt_class: List[Tensor], 
        gt_bbox: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Performs bipartite matching between predictions and ground truths for a batch.

        Args:
            pred_class (List[Tensor]): Predicted class logits for each sample in batch.
                Shape: [num_queries, num_classes].
            pred_bbox (List[Tensor]): Predicted bounding boxes in [cx, cy, w, h] format.
                Shape: [num_queries, 4].
            gt_class (List[Tensor]): Ground truth class indices (0-based).
                Shape: [num_gt_boxes].
            gt_bbox (List[Tensor]): Ground truth boxes in [cx, cy, w, h] format.
                Shape: [num_gt_boxes, 4].

        Returns:
            indicates (Tuple[List[Tensor], List[Tensor]]):
                - row_inds: Indices of matched predictions (from queries) for each sample.
                    Length = batch size.
                - col_inds: Indices of matched ground truths for each sample.
                    Length = batch size.
        """
        
        B = len(pred_class)
        row_inds, col_inds = [], []
        
        pred_bbox = [xywh_to_xyxy(bbox) for bbox in pred_bbox]
        gt_bbox = [xywh_to_xyxy(bbox) for bbox in gt_bbox] 
        
        for i in range(B):
            device = pred_class[i].device
            cost_class = self._compute_class_cost(pred_class[i], gt_class[i])
            cost_bbox = self._compute_bbox_cost(pred_bbox[i], gt_bbox[i])
            cost_giou = self._compute_giou_cost(pred_bbox[i], gt_bbox[i])
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            row_ind, col_ind = jonker_volgenant(C)
            row_inds.append(torch.tensor(row_ind).long().to(device=device))
            col_inds.append(torch.tensor(col_ind).long().to(device=device))
        
        return row_inds, col_inds
    
    def _compute_class_cost(self, pred_class: Tensor, gt_class: Tensor) -> Tensor:
        # fix cross_entropy -> softmax + nll_loss -> - log_softmax
        pred_prob = pred_class.softmax(-1)                  # [Q, C+1]
        cost_class = -pred_prob[:, gt_class]              # [Q, G]
        return cost_class
    
    def _compute_bbox_cost(self, pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
        cost_bbox = torch.cdist(pred_bbox, gt_bbox, p=1) # [Q, G]
        return cost_bbox
    
    def _compute_giou_cost(self, pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
        giou = box_giou(pred_bbox, gt_bbox) # [Q, G]
        cost_giou = 1 - giou
        return cost_giou

    def get_cost_weight_dict(self) -> Dict[str, float]:
        return {"class": self.cost_class, "bbox": self.cost_bbox,"giou": self.cost_giou}