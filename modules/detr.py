import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
    def forward(self, src):
        return self.transformer_encoder(src)

class DETR(nn.Module):
    def __init__(self, num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6):
        super(DETR, self).__init__()
        # Feature extractor (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, hidden_dim))
        )
        
        # Transformer components
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, nheads, num_encoder_layers)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.backbone(x).flatten(2).permute(2, 0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        memory = self.transformer(x)
        
        # Output predictions
        outputs_class = self.class_embed(memory)
        outputs_coord = self.bbox_embed(memory).sigmoid()
        
        return outputs_class, outputs_coord