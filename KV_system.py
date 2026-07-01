"""
KV_system.py — shared attention MATH used by MHA / MQA / GQA.

In this "attached cache" version the cache lives INSIDE each attention block
(self.cache_k / self.cache_v), so this file only holds the two pieces of pure
math the variants share:

  - repeat_kv  : broadcast kv heads up to query heads (MQA / GQA)
  - causal_sdpa: scaled dot-product attention with a cache-aware causal mask

Shapes:  b = batch, H = query heads, Hkv = kv heads, s = seq len, d = head dim
"""

import math
import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, Hkv, s, d) -> (b, Hkv * n_rep, s, d). Repeats each kv head n_rep times."""
    b, hkv, s, d = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(b, hkv, n_rep, s, d)
        .reshape(b, hkv * n_rep, s, d)
    )


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention with a KV-cache-aware causal mask.

    q: (b, H, q_len, d)   k, v: (b, H, kv_len, d)   (k/v already repeated to H heads)

    offset = kv_len - q_len makes the mask correct in both regimes:
      - training / prefill : q_len == kv_len, offset 0   -> full triangular mask
      - cached decode      : q_len == 1,      offset > 0 -> new query sees whole cache
    """
    d = q.shape[-1]
    q_len, kv_len = q.shape[2], k.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    offset = kv_len - q_len
    q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + offset
    k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
