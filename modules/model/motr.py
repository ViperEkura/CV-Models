import torch
import torch.nn as nn

from typing import Optional
from torch import Tensor
from modules.model.detr import DETR


class MOTR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channel: int = 3,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        num_track_queries: int = 300,
        aux_loss: bool = True,
    ):
        super().__init__()
        # 基础DETR模型作为backbone
        self.detr = DETR(
            num_classes=num_classes,
            in_channel=in_channel,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries + num_track_queries,  # 总的queries数量
        )
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_track_queries = num_track_queries
        self.aux_loss = aux_loss
        
        # Track query相关参数
        self.track_query_embed = nn.Embedding(num_track_queries, hidden_dim)
        self.track_query_pos_embed = nn.Embedding(num_track_queries, hidden_dim)
        
        # Track相关预测头
        self.track_class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.track_bbox_embed = nn.Linear(hidden_dim, 4)  # bbox (x, y, w, h)
        
        # 初始化track heads的参数
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        self.track_class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        nn.init.constant_(self.track_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.track_bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.track_class_embed.weight.data)
        nn.init.constant_(self.track_class_embed.bias.data, 0)
        
    def forward(self, inputs: Tensor, track_query_mask: Optional[Tensor] = None):
        """
        Args:
            inputs (Tensor): 输入图像 [batch_size, channels, height, width]
            track_query_mask (Optional[Tensor]): track query的mask [batch_size, num_track_queries]
        """
        # 使用DETR的backbone提取特征
        x = self.detr.backbone(inputs)  # [batch_size, 2048, H, W]
        h = self.detr.conv(x)           # [batch_size, hidden_dim, H, W]
        
        H, W = h.shape[-2:]
        B = h.shape[0]
        
        # 位置编码
        pos = torch.cat([
            self.detr.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.detr.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)   # [1, H*W, hidden_dim]
        
        h = h.flatten(2).transpose(1, 2)        # [batch_size, H*W, hidden_dim]
        
        # 构造object queries和track queries
        object_queries = self.detr.query_pos.unsqueeze(0).repeat(B, 1, 1)  # [batch_size, num_queries, hidden_dim]
        track_queries = self.track_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [batch_size, num_track_queries, hidden_dim]
        
        # 合并queries
        all_queries = torch.cat([object_queries, track_queries], dim=1)  # [batch_size, num_queries+num_track_queries, hidden_dim]
        
        # 如果提供了track_query_mask，则应用到track queries
        if track_query_mask is not None:
            # 扩展mask以适应hidden dimension
            track_query_mask = track_query_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
            # 应用mask到track queries
            track_queries = track_queries * track_query_mask
        
        # Transformer编码器-解码器
        transformer_output = self.detr.transformer(
            src=pos + h,                                                # [batch_size, H*W, hidden_dim]
            tgt=all_queries                                             # [batch_size, num_queries+num_track_queries, hidden_dim]
        )                                                               # output: [batch_size, num_queries+num_track_queries, hidden_dim]
        
        # 分离object和track的输出
        object_output = transformer_output[:, :self.num_queries]        # [batch_size, num_queries, hidden_dim]
        track_output = transformer_output[:, self.num_queries:]         # [batch_size, num_track_queries, hidden_dim]
        
        # DETR原始预测 (detection)
        pred_class_detr = self.detr.linear_class(object_output)         # [batch_size, num_queries, num_classes+1]
        pred_bbox_detr = torch.sigmoid(self.detr.linear_bbox(object_output))  # [batch_size, num_queries, 4]
        
        # Track预测
        pred_class_track = self.track_class_embed(track_output)         # [batch_size, num_track_queries, num_classes+1]
        pred_bbox_track = torch.sigmoid(self.track_bbox_embed(track_output))  # [batch_size, num_track_queries, 4]
        
        # 合并预测结果
        pred_class = torch.cat([pred_class_detr, pred_class_track], dim=1)  # [batch_size, num_queries+num_track_queries, num_classes+1]
        pred_bbox = torch.cat([pred_bbox_detr, pred_bbox_track], dim=1)     # [batch_size, num_queries+num_track_queries, 4]
        
        output = {
            'pred_logits': pred_class,
            'pred_boxes': pred_bbox,
            'aux_outputs': None,
        }
        
        if self.aux_loss:
            # TODO: 实现辅助loss计算
            pass
            
        return output