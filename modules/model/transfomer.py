import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor



class Attention(nn.Module):
    def __init__(self, n_dim, n_heads, head_dim, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_dim)
    
    def forward(self, q, k ,v):
        q = self._split_heads(self.q_proj(q))
        k = self._split_heads(self.k_proj(k))
        v = self._split_heads(self.v_proj(v))
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().flatten(2)
        o = self.out_proj(attn_out)
        return o

    def _split_heads(self, x: Tensor):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_dim, bias=False):
        super().__init__()
        self.in_proj = nn.Linear(n_dim, 4 * n_dim, bias=bias)
        self.gate_proj = nn.Linear(n_dim, 4 * n_dim, bias=bias)
        self.out_proj = nn.Linear(4 * n_dim, n_dim, bias=bias)
    
    def forward(self, x):
        x = self.out_proj(self.in_proj(x) * F.silu(self.gate_proj(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_dim, n_heads, head_dim, bias=False):
        super().__init__()
        self.attn = Attention(n_dim, n_heads, head_dim, bias=bias)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ffn = FeedForward(n_dim, bias=bias)
        self.norm2 = nn.LayerNorm(n_dim)
    
    def forward(self, x):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_dim, n_heads, head_dim, bias=False):
        super().__init__()
        self.self_attn = Attention(n_dim, n_heads, head_dim, bias=bias)
        self.norm1 = nn.LayerNorm(n_dim)
        self.cross_attn = Attention(n_dim, n_heads, head_dim, bias=bias)
        self.norm2 = nn.LayerNorm(n_dim)
        
    def forward(self, tgt, memory):
        # self attntion
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        
        # cross attention
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        
        return tgt
    

class Transformer(nn.Module):
    def __init__(self, n_dim, n_heads, num_encoder_layers, num_decoder_layers, head_dim=None, bias=False):
        super().__init__()
        if head_dim is None:
            head_dim = n_dim // n_heads
            
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(n_dim, n_heads, head_dim, bias=bias)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(n_dim, n_heads, head_dim, bias=bias)
            for _ in range(num_decoder_layers)
        ])
        
    def forward(self, src, tgt):
        memory = src
        for encoder_layer in self.encoder:
            memory = encoder_layer(memory)
        
        output = tgt
        for decoder_layer in self.decoder:
            output = decoder_layer(output, memory)
        
        return output
