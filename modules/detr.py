import torch
import torch.nn as nn
from torch import Tensor
from .resnet import ResNet

class DETR(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        in_channel: int, 
        hidden_dim: int, 
        nheads: int,
        num_encoder_layers: int, 
        num_decoder_layers: int
):
        super().__init__()
        self.backbone = ResNet("resnet50", in_channel, 2048)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs: Tensor):
        x = self.backbone(inputs)
        h: Tensor = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), torch.sigmoid(self.linear_bbox(h))