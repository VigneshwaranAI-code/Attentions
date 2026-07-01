# Attention from scratch: MHA, MQA, GQA with a KV cache

Minimal, readable PyTorch implementations of the three attention variants used in
modern transformers, each with a key/value cache for autoregressive generation. The
code is written to be understood, not just run: every non-obvious line is commented,
and each file has a self-test that proves cached generation matches a full forward pass.

## The one idea behind all three

MHA, MQA, and GQA differ in exactly one thing: **how many key/value heads exist.**
Grouped-Query Attention is the general case, and the other two are its extremes.

| Variant | `num_kv_heads` | KV cache size | Trade-off |
|---|---|---|---|
| MHA (Multi-Head) | `= num_heads` | largest | best quality, most memory |
| GQA (Grouped-Query) | between (e.g. 2) | medium | the practical middle ground |
| MQA (Multi-Query) | `1` | smallest | least memory, some quality cost |

Because of this, `GroupedQueryAttention(num_kv_heads=num_heads)` behaves like MHA and
`GroupedQueryAttention(num_kv_heads=1)` behaves like MQA. In production you could ship
only the GQA class; the separate MHA and MQA files exist here for clarity.

## What the KV cache is for

During generation a model produces one token at a time. Without a cache, every step
recomputes the keys and values for the entire history. The KV cache stores them so each
new step only computes K/V for the single new token and reuses the rest. Fewer KV heads
(MQA/GQA) means a smaller cache, which is the whole reason those variants exist.

Two cache designs are included, because they teach different lessons:

- **Separate cache** (top-level files): the cache is an external `KVCache` object passed
  into `forward`. The attention module stays stateless with respect to the request. This
  is what real serving stacks do — one set of weights can serve many independent caches,
  enabling concurrent requests and beam search.
- **Attached cache** (`attached_cache/`): the cache lives inside the block as instance
  state, toggled by a `use_cache` flag, cleared by `reset_cache()`. This is the reference
  style (as in the Llama source) and the easiest to read, at the cost of being
  single-stream and stateful.

Same math in both; only *where the cache state lives* differs.

## Causal masking

Both designs share one causal mask that works in two regimes via a position offset:

- **Training / prefill**: the full sequence goes in at once (`q_len == kv_len`), producing
  the classic lower-triangular mask — each position attends only to itself and its past.
- **Cached decode**: one new token goes in (`q_len == 1`) at absolute position `past_len`;
  the offset lifts its row index to its true position so it correctly attends to the whole
  cache with nothing masked.

The masked entries are set to `-inf` before softmax, so they receive exactly zero weight.

## Training vs inference

The cache is an inference-only optimization. **Do not use it during training.**

- **Training**: no cache. Feed the whole sequence in one parallel forward (teacher forcing),
  let the causal mask enforce order, and backpropagate. In the separate-cache files this is
  simply calling `forward(x)` with no cache; in the attached-cache files it is
  `forward(x, use_cache=False)` (the default). Gradients flow; the cache is untouched.
- **Inference**: prefill the prompt, then decode token by token with the cache on.

## Files

Top-level (separate `KVCache` object — recommended base for real use):

- `KV_system.py` — the `KVCache` class plus shared `repeat_kv` and `causal_sdpa`
- `MHA.py` — `MultiHeadAttention`
- `MQA.py` — `MultiQueryAttention`
- `GQA.py` — `GroupedQueryAttention` (the general case)

`attached_cache/` (cache inside the block — easiest to learn from):

- `KV_system.py` — shared `repeat_kv` and `causal_sdpa` only (no cache here)
- `MHA.py`, `MQA.py`, `GQA.py` — each owns its cache via `use_cache` / `reset_cache()`

`attention_from_scratch.py` — a single self-contained file with all three variants and a
side-by-side demo, if you prefer everything in one place.

## Requirements

```
python >= 3.9
torch
```

## Running

Each attention file is runnable on its own and prints a correctness check:

```
cd <this directory>
python GQA.py
# GQA  cached == full: True | kv heads cached: 2
```

Or run the all-in-one demo:

```
python attention_from_scratch.py
```

## Usage

Separate-cache style (one `KVCache` per layer, threaded through generation):

```python
from GQA import GroupedQueryAttention
from KV_system import KVCache

attn = GroupedQueryAttention(d_model=256, num_heads=8, num_kv_heads=2)

# training / one-shot: no cache
out = attn(x)

# generation: one cache per layer, grows one token per call
cache = KVCache()
out = attn(new_token, cache=cache)
```

Attached-cache style (`use_cache` flag, remember to reset between sequences):

```python
from attached_cache.GQA import GroupedQueryAttention

attn = GroupedQueryAttention(d_model=256, num_heads=8, num_kv_heads=2)

out = attn(x)                       # training path, use_cache=False by default
attn.reset_cache()
out = attn(new_token, use_cache=True)  # generation path
```

## What is intentionally left out

To keep the focus on the attention/cache mechanics, these files omit pieces a full model
needs. Add them when building a complete transformer block:

- **RoPE** (rotary position embeddings) — applied to Q and K *before* the cache write.
  Without positional encoding, attention is order-blind.
- **The grow-by-concat cache** here reallocates each decode step. Production uses a
  pre-allocated fixed buffer or paged blocks to avoid that.
- **`F.scaled_dot_product_attention`** — the fused, faster kernel you'd swap in for the
  hand-written masked softmax (pass `is_causal=True`, or `enable_gqa=True` for grouped K/V).
- Norms (RMSNorm), the SwiGLU MLP, residual connections, and dropout wiring — the rest of
  the transformer block.

## Notes on correctness

Every file asserts that generating token-by-token with the cache produces the same output
(within floating-point tolerance) as running the whole sequence at once. If you modify the
code, keep that check passing — it catches the two classic bugs: a mask that assumes
`q_len == kv_len`, and expanding K/V to full head count *before* the cache instead of after
(which silently throws away the memory savings).
