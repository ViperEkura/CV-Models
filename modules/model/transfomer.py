import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor



class Attention(nn.Module):
    def __init__(self, n_dim, n_heads, head_dim, bias=False):
        self.n_heads = n_heads
        self.q_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(n_dim, n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_dim)
    
    def forward(self, q, k ,v):
        q = self._split_heads(self.q_proj(q))
        k = self._split_heads(self.k_proj(k))
        v = self._split_heads(self.v_proj(v))
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view_as(q)
        o = self.out_proj(attn_out)
        return o

    def _split_heads(self, x: Tensor):
        batch_size, seq_len, n_dim = x.shape
        head_dim = n_dim // self.n_heads
        x = x.reshape(batch_size, seq_len, self.n_heads, head_dim)
        x = x.transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_dim, bias=False):
        super().__init__()
        self.in_proj = nn.Linear(n_dim, 4 * n_dim, bias=bias)
        self.gate_proj = nn.Linear(4 * n_dim, n_dim, bias=bias)
        self.out_proj = nn.Linear(n_dim, n_dim, bias=bias)

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
    def __init__(self, n_dim, n_heads, head_dim, n_layers, bias=False):
        pass
    
    def forward(self, x):
        pass

