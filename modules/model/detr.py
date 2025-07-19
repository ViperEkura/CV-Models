import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cross_entropy, l1_loss
from .resnet import ResNet
from scipy.optimize import linear_sum_assignment


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
        ], dim=-1).flatten(0, 1).unsqueeze(1)   # [H*W, 1, hidden_dim]
        
        h = h.flatten(2).permute(2, 0, 1)       # [H*W, batch_size, hidden_dim]
        h = self.transformer(
            src=pos + h,                                                # [H*W, batch_size, hidden_dim]
            tgt=self.query_pos.unsqueeze(1).repeat(1, B, 1)             # [num_queries, batch_size, hidden_dim]
        )                                                               # output: [num_queries, batch_size, hidden_dim]
        
        pred_class = self.linear_class(h)                               # [num_queries, batch_size, num_classes+1]
        pred_bbox = torch.sigmoid(self.linear_bbox(h))                  # [num_queries, batch_size, 4]
        pred_class = torch.transpose(pred_class, 0, 1)                  # [batch_size, num_queries, num_classes+1]
        pred_bbox = torch.transpose(pred_bbox, 0, 1)                    # [batch_size, num_queries, 4]
    
        return pred_class, pred_bbox


def box_cxcywh_to_xyxy(x: Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1: Tensor, boxes2: Tensor):
    # Compute intersection
    inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter = inter.clamp(min=0)
    
    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    # Compute IoU
    iou = inter / union
    
    # Compute enclosing box
    enclose_x1y1 = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_x2y2 = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_area = (enclose_x2y2[..., 0] - enclose_x1y1[..., 0]) * (enclose_x2y2[..., 1] - enclose_x1y1[..., 1])
    
    # Compute GIoU
    giou = iou - (enclose_area - union) / enclose_area
    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1.0, bbox_weight=1.0, giou_weight=1.0):
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        
    @torch.no_grad()
    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, targets: Tensor):
        """
        pred_logits: [batch_size, num_queries, num_classes]
        pred_boxes: [batch_size, num_queries, 4]
        targets: list of dicts with keys 'labels' and 'boxes'
        """
        bs, num_queries = pred_logits.shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        pred_logits = pred_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
        
        indices = []
        
        for i in range(bs):
            target = targets[i]
            num_targets = len(target["labels"])
            
            # Compute classification cost
            cost_class = -pred_logits[i*num_queries:(i+1)*num_queries, target["labels"]]
            
            # Compute L1 cost between boxes
            cost_bbox = torch.cdist(pred_boxes[i*num_queries:(i+1)*num_queries], target["boxes"], p=1)
            
            # Compute giou cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[i*num_queries:(i+1)*num_queries]),
                box_cxcywh_to_xyxy(target["boxes"])
            )
            
            # Final cost matrix
            C = self.class_weight * cost_class + self.bbox_weight * cost_bbox + self.giou_weight * cost_giou
            C = C.reshape(num_queries, num_targets).cpu()
            
            # Hungarian algorithm
            indices.append(linear_sum_assignment(C))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
