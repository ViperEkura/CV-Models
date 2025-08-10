import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List
from modules.model.matcher import HungarianMatcher

class MOTRLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ['labels', 'boxes', 'cardinality']
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef

    def forward(self, outputs, targets):
        """
        计算MOTR损失
        
        Args:
            outputs: 模型输出字典
            targets: 真实标签列表
        """
        # 提取预测结果
        pred_class = outputs['pred_logits']
        pred_bbox = outputs['pred_boxes']
        
        # 匹配预测和真实标签
        indices = self.matcher.match(
            pred_class, 
            pred_bbox, 
            torch.stack([t['labels'] for t in targets]), 
            torch.stack([t['boxes'] for t in targets])
        )
        
        # 计算各类损失
        losses = {}
        if 'labels' in self.losses:
            losses.update(self.loss_labels(pred_class, targets, indices))
        if 'boxes' in self.losses:
            losses.update(self.loss_boxes(pred_bbox, targets, indices))
        if 'cardinality' in self.losses:
            losses.update(self.loss_cardinality(pred_class, targets, indices))
            
        return losses
    
    def loss_labels(self, outputs, targets, indices, log=True):
        """计算分类损失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        self.empty_weight = self.empty_weight.to(src_logits.device)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """计算边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / max(target_boxes.numel(), 1)

        loss_giou = 1 - torch.diag(
            # 注意：这里需要实现giou计算函数
            self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(src_boxes),
                self._box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / max(target_boxes.numel(), 1)
        return losses

    def loss_cardinality(self, outputs, targets, indices):
        """计算基数损失"""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _box_cxcywh_to_xyxy(self, boxes):
        """
        将框从 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式
        """
        x_c, y_c, w, h = boxes.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def _generalized_box_iou(self, boxes1, boxes2):
        """
        计算广义IoU
        """
        # TODO: 实现GIoU计算
        pass