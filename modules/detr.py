import torch
import torch.nn as nn
from torch import Tensor
from .resnet import ResNet

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
        x = self.backbone(inputs)       # [batch_size, 2048]
        h = self.conv(x)                # [batch_size, hidden_dim, H, W]

        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)   # [H*W, 1, hidden_dim]
        
        h = h.flatten(2).permute(2, 0, 1)       # [H*W, batch_size, hidden_dim]
        h = self.transformer(
            src=pos + h,  # 添加位置编码
            tgt=self.query_pos.unsqueeze(1).repeat(1, h.shape[1], 1)    # [num_queries, batch_size, hidden_dim]
        )                                                               # output: [num_queries, batch_size, hidden_dim]
        
        pred_class = self.linear_class(h)  # [num_queries, batch_size, num_classes+1]
        pred_bbox = torch.sigmoid(self.linear_bbox(h))
        return pred_class, pred_bbox