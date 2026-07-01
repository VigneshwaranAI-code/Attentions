"""
MQA.py — Multi-Query Attention with the KV cache built INSIDE the block.

num_kv_heads == 1: all query heads share one kv head. The cache stores that single
head (so it's ~H x smaller than MHA); the broadcast to H heads happens AFTER caching,
only for the attention math.
"""

import torch
import torch.nn as nn

from KV_system import causal_sdpa


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=bias)  # ONE kv head
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.cache_k = None  # (b, 1, s, d)  <- compact, single head
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, 1, self.head_dim).transpose(1, 2)  # (b, 1, s, d)
        v = self.v_proj(x).view(b, s, 1, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v            # cache stays single-head

        # broadcast the one kv head across all query heads, only for the math
        k = k.expand(b, self.num_heads, k.shape[2], self.head_dim)
        v = v.expand(b, self.num_heads, v.shape[2], self.head_dim)

        out = causal_sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MultiQueryAttention(256, 8).eval()
    x = torch.randn(2, 16, 256)

    with torch.no_grad():
        full = m(x)
        m.reset_cache()
        steps = [m(x[:, t:t + 1, :], use_cache=True) for t in range(x.shape[1])]
        step = torch.cat(steps, dim=1)

    print("MQA  cached == full:", torch.allclose(full, step, atol=1e-5),
          "| kv heads cached:", m.cache_k.shape[1])
