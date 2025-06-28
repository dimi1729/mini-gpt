import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # For q, k, and v tensors just concatenated
        self.qkv_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        # for output
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, L, C = x.size()  # batch size, length, n_embed

        qkv = self.qkv_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)

        q = q.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, L, hs)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, L, hs)

        attn = (q @ k.transpose(-2, -1)) * (1.0/np.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,:L,:L] == 0, float('-inf'))
        attn = F.softmax(attn, dim=2)

        y = attn @ v # (B, nh, L, L) x (B, nh, L, hs) = (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, C)

        out = self.c_proj(y)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
