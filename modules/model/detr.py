import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from modules.model.resnet import ResNet
from modules.model.transfomer import Transformer


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.row_embed = nn.Parameter(torch.empty(50, hidden_dim // 2))   # 50x50
        self.col_embed = nn.Parameter(torch.empty(50, hidden_dim // 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def forward(self, x: Tensor):
        B = x.shape[0]
        H, W = x.shape[-2:]
        x_emb = self.col_embed[:H]
        y_emb = self.row_embed[:W]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1) # [batch_size, hidden_dim, H, W]
        return pos + x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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
        self.backbone = ResNet("resnet34", in_channel)
        self.conv = nn.Conv2d(512, hidden_dim, 1) 
        self.transformer = Transformer(
            n_dim=hidden_dim,
            n_heads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)           # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)                 # bbox(x,y,w,h)
        self.postion_embed = PositionEmbeddingLearned(hidden_dim)
        self.query_pos = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.uniform_(self.query_pos)


    def forward(
        self, 
        inputs: Tensor, 
    ) -> Tuple[Tensor, Tensor]:
        
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        assert inputs.ndim == 4
        
        x = self.backbone(inputs)       # [batch_size, 2048, H, W]
        h = self.conv(x)                # [batch_size, hidden_dim, H, W]
        
        embeded = torch.flatten(self.postion_embed(h),2).transpose(1, 2)        # [batch_size, H*W, hidden_dim]
        query_pos = self.query_pos.unsqueeze(0).repeat(embeded.size(0), 1, 1)
        h = self.transformer(
            src=embeded,                                                # [batch_size, H*W, hidden_dim]
            tgt=query_pos                                               # [batch_size, num_queries, hidden_dim]
        )                                                               # output: [batch_size, num_queries, hidden_dim]
        
        pred_class = self.class_embed(h)                               # [batch_size, num_queries, num_classes+1]
        pred_bbox = torch.sigmoid(self.bbox_embed(h))                  # [batch_size, num_queries, 4]

        return pred_class, pred_bbox


class PostProcess(nn.Module):
    
    @staticmethod
    def process(pred_class: Tensor, pred_bbox: Tensor, threshold: float = 0.5) -> Tensor:
        scores, labels = pred_class.max(dim=-1)
        keep = scores > threshold
        valid_detections = (labels > 0) & keep
        
        selected_scores = scores[valid_detections]
        selected_labels = labels[valid_detections]
        selected_bbox = pred_bbox[valid_detections]

        return selected_scores, selected_labels, selected_bbox