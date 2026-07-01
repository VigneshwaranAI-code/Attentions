"""
MHA.py — Multi-Head Attention with the KV cache built INSIDE the block.

num_kv_heads == num_heads. The cache is instance state (self.cache_k / self.cache_v).

  use_cache=False  -> training / one-shot forward. Cache is untouched, grads flow.
  use_cache=True   -> generation. Each call appends to the stored cache.
  reset_cache()    -> clear before starting a new sequence.
"""

import torch
import torch.nn as nn

from KV_system import causal_sdpa


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # --- the cache lives here, inside the block ---
        self.cache_k = None  # (b, H, s, d)
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        b, s, _ = x.shape
        shp = (b, s, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shp).transpose(1, 2)  # (b, H, s, d)
        k = self.k_proj(x).view(shp).transpose(1, 2)
        v = self.v_proj(x).view(shp).transpose(1, 2)

        if use_cache:
            if self.cache_k is not None:                 # append new tokens
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v            # store full history

        out = causal_sdpa(q, k, v)                       # k/v already have H heads
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MultiHeadAttention(256, 8).eval()
    x = torch.randn(2, 16, 256)

    with torch.no_grad():
        full = m(x)                       # use_cache=False -> training path
        m.reset_cache()
        steps = [m(x[:, t:t + 1, :], use_cache=True) for t in range(x.shape[1])]
        step = torch.cat(steps, dim=1)

    print("MHA  cached == full:", torch.allclose(full, step, atol=1e-5),
          "| kv heads cached:", m.cache_k.shape[1])
