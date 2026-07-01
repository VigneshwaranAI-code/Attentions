"""
GQA.py — Grouped-Query Attention with the KV cache built INSIDE the block.

1 <= num_kv_heads <= num_heads (the general case). The cache stores num_kv_heads
worth of K/V; repeat_kv expands to num_heads AFTER caching, only for the math.

    num_kv_heads == num_heads -> behaves like MHA
    num_kv_heads == 1         -> behaves like MQA
"""

import torch
import torch.nn as nn

from KV_system import causal_sdpa, repeat_kv


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

        self.cache_k = None  # (b, Hkv, s, d)  <- compact, num_kv_heads
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v            # cache stays at num_kv_heads

        k = repeat_kv(k, self.n_rep)                     # expand to H only for the math
        v = repeat_kv(v, self.n_rep)

        out = causal_sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = GroupedQueryAttention(256, 8, num_kv_heads=2).eval()
    x = torch.randn(2, 16, 256)

    with torch.no_grad():
        full = m(x)
        m.reset_cache()
        steps = [m(x[:, t:t + 1, :], use_cache=True) for t in range(x.shape[1])]
        step = torch.cat(steps, dim=1)

    print("GQA  cached == full:", torch.allclose(full, step, atol=1e-5),
          "| kv heads cached:", m.cache_k.shape[1])
