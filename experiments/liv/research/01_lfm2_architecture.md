# LFM2 / LFM2.5 Architecture — Citation-Grounded Technical Specification

Research target: enough precision to reimplement the LIV ("gated short convolution") block and
the full block-interleaving pattern exactly, with exact parameter counts and cache behavior.

**Evidence labels used throughout:**
- `[PAPER]` — stated in the LFM2 Technical Report, arXiv:2511.23404v1
- `[CONFIG]` — present in a released HuggingFace `config.json`
- `[CODE]` — present in HuggingFace `transformers` source
- `[CKPT]` — read directly out of a released `.safetensors` header
- `[INFER]` — my derivation/arithmetic, not stated by any source

Primary sources:
- Paper abstract: https://arxiv.org/abs/2511.23404
- Paper full text (HTML): https://arxiv.org/html/2511.23404v1
- `modular_lfm2.py`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py
- `modeling_lfm2.py`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py
- `configuration_lfm2.py`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/configuration_lfm2.py
- `cache_utils.py`: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

---

## 0. TERMINOLOGY WARNING — "LIV" is not LFM2 vocabulary

**The string "LIV" appears ZERO times in the LFM2 technical report.** I grepped the full
extracted text of https://arxiv.org/html/2511.23404v1 — 0 hits for `LIV`, 0 hits for
`linear input-varying`. The paper's own name for the block is **"gated short convolution
block"** (20 hits for "short conv"). `[PAPER]`

"LIV" (linear input-varying) is terminology from the earlier **STAR** paper
(arXiv:2411.17800, Thomas et al., ICLR 2025), which the LFM2 report cites twice — and cites
*critically*, as a superseded "earlier academic prototype" whose proxy objectives "do not
transfer reliably" (see §8.1). Liquid AI's **launch blog** does use the term — it calls the
conv layers "double-gated short-range **LIV** convolutions" — so the framing has a Liquid AI
source, just not a peer-reviewed one.

In STAR's formalism the LFM2 conv block is specifically the **scaled-Toeplitz / gated
convolution** LIV class (`T_ij = C_i K_{i−j} B_j`), which is a precise and correct
identification (§8.1b). So "mostly-LIV hybrid" is defensible — but note that under STAR's
taxonomy **attention is also an LIV operator** (the dense class, `σ(C_i B_j)`). Strictly, an
LFM2 model is *entirely* LIV; what varies is *which* LIV class each layer uses. A paper title
of "mostly-LIV" is therefore ambiguous unless defined.

Recommended framing, stated once explicitly: *"We use 'LIV block' for the LFM2 gated short
convolution block — a linear input-varying operator of the gated-convolution (scaled-Toeplitz)
class in the taxonomy of STAR (arXiv:2411.17800) — and reserve 'attention block' for the dense
LIV class."*

---

## 1. The gated short convolution ("LIV") block — exact spec

### 1.1 Paper equation `[PAPER]`

Verbatim from §2.2 "Gated short convolution block" of https://arxiv.org/html/2511.23404v1
(transcribed from the HTML math markup):

> Given an input hidden sequence **h** ∈ ℝ^{L×d} (batch dimension omitted for clarity),
> each gated short convolution block applies input-dependent multiplicative gating around
> a depthwise short convolution:
>
> (**B**, **C**, **h̃**) = Linear(**h**),  **y** = **B** ⊙ **h̃**,  **z** = Conv_k(**y**),  **o** = Linear_out(**C** ⊙ **z**).
>
> Here Linear: ℝ^d → ℝ^{3d} is a linear map applied position-wise across the sequence
> length, L, and whose output channels are split along the feature dimension into
> (**B**, **C**, **h̃**) with **B**, **C**, **h̃** ∈ ℝ^{L×d}. The intermediate tensors
> **y**, **z**, **o** also lie in ℝ^{L×d}. Conv_k: ℝ^{L×d} → ℝ^{L×d} is a depthwise 1D
> convolution along the sequence with kernel size k, and ⊙ denotes element-wise
> multiplication. Linear_out: ℝ^d → ℝ^d is the linear output projection.

Notable properties, all stated or directly implied:
- **Width is preserved everywhere.** There is no inner expansion factor. `in_proj` is
  d → 3d, and each of B, C, h̃ is exactly width d. This is unlike Mamba (which expands
  by 2×) — the LFM2 conv block operates entirely at model width. `[PAPER]` + `[CONFIG]`
  (`conv_dim == hidden_size` in every released config)
- **Double gating**: one gate (B) *before* the conv, one gate (C) *after*. Both are
  input-dependent (produced by the same `in_proj`), which is exactly what makes the
  operator "input-varying" rather than a static convolution.
- **No nonlinearity is mentioned in the conv path.** The only nonlinearities are the two
  elementwise products. Confirmed in code (§1.2): `activation=None` is the effective
  path. This is a real difference vs Mamba/Hyena, which apply SiLU to the conv output.

### 1.2 Released code — exact forward pass `[CODE]`

From `Lfm2ShortConv.forward`. I checked **both** the hand-written modular source and the
auto-generated runtime file, and the `Lfm2ShortConv` class is **byte-identical** between
them, so the analysis below applies to the code that actually executes:
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py

```python
seq_len = hidden_states.shape[1]
hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
BCx = self.in_proj(hidden_states).transpose(-1, -2)     # (bsz, 3d, L)
B, C, x = BCx.chunk(3, dim=-2)                          # each (bsz, d, L)
hidden_states = B * x                                   # pre-conv gate
# ... conv (branch on cached-decode vs prefill) ...
y = C * hidden_states                                   # post-conv gate
y = y.transpose(-1, -2).contiguous()
y = self.out_proj(y)
return y
```

**Order of operations, definitively: `B ⊙ x` → depthwise causal conv → `C ⊙ (·)` →
`out_proj`.** This matches the paper equation exactly. There is **no discrepancy** between
paper and code on the op order.

**Critical detail for reimplementation — the chunk order is `(B, C, x)`, not `(B, x, C)`.**
`in_proj` output channels `[0:d]` → B (pre-gate), `[d:2d]` → C (post-gate), `[2d:3d]` → x
(the value carried through the conv). Getting this wrong silently produces a valid-looking
but different model and makes released weights unloadable-in-effect. `[CODE]`

**No activation function anywhere in the conv path.** Both `causal_conv1d_fn` and
`causal_conv1d_update` take an `activation: str | None = None` argument and apply
`ACT2FN[activation]` only `if activation is not None`. `Lfm2ShortConv.forward` calls both
**without** passing `activation`, so it defaults to `None`. `[CODE]` The block's entire
nonlinearity budget is the two elementwise gates.

**Numerically verified.** I instantiated `Lfm2ShortConv` (d=8, k=3, L=6) under
`transformers==5.14.1` / `torch==2.7.1` and compared its output against a from-scratch
reimplementation of the spec above:

| Reimplementation variant | max abs diff vs `Lfm2ShortConv` |
|---|---|
| `(B, C, x)` chunk order, no activation (**the documented spec**) | **0.0** (exact) |
| `(B, x, C)` chunk order instead | 0.0826 |
| `(B, C, x)` but with SiLU after the conv (Mamba-style) | 0.0410 |

So the spec in §1.1–1.2 is exactly right, and both plausible variations produce materially
different outputs — confirming these two details are load-bearing, not cosmetic. Causality
also verified: perturbing the last input token by +10.0 changed earlier-position outputs by
**0.0**. `[CODE]`+`[INFER]` (my own numeric test)

**No separate short-conv normalization.** There is no norm inside `Lfm2ShortConv` — no
input norm, no post-conv norm, no group norm (contrast Mamba2, which has an inner
RMSNorm/GroupNorm before `out_proj`). The only norm applied to the conv block is the
block-level `operator_norm` in the enclosing `Lfm2DecoderLayer`. `[CODE]`

**Bias**: `conv_bias: false` in every released config, and `Lfm2ShortConv` uses the single
`config.conv_bias` flag for the conv, `in_proj`, **and** `out_proj`. So in all released
checkpoints, all three are bias-free. `[CONFIG]` + `[CODE]`

### 1.3 Conv layer construction `[CODE]`

```python
self.conv = nn.Conv1d(
    in_channels=config.hidden_size,
    out_channels=config.hidden_size,
    kernel_size=self.conv_kernel_size,     # = config.conv_L_cache
    groups=config.hidden_size,             # fully depthwise
    bias=self.bias,
    padding=self.conv_kernel_size - 1,     # causal left-padding
)
self.in_proj  = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)
```

- `groups == in_channels == out_channels` → **fully depthwise**, one k-tap filter per channel.
- Kernel size is driven by **`conv_L_cache`**, which is **3** in every single released
  config (350M, 700M, 1.2B, 2.6B, 8B-A1B, 24B-A2B, LFM2.5-1.2B-Base/Instruct). `[CONFIG]`
  The paper's Table 1 lists "Conv k = 3" for all five models it covers. `[PAPER]`
- Causality is enforced by `padding = k-1` on the left plus a right-truncation to `seq_len`
  inside `causal_conv1d_fn` (`[:, :, :seq_len]`). `[CODE]`

**Naming confusion worth flagging**: the config key is `conv_L_cache`, which reads like a
cache length, but in code it is assigned directly to `self.conv_kernel_size` and used as
`nn.Conv1d(kernel_size=...)`. Kernel size and conv-state length are *the same number* here
(a k-tap causal conv needs k-1 tokens of left context; the implementation stores k). `[CODE]`

### 1.4 Weight names and shapes, read from the actual checkpoint `[CKPT]`

Read by range-requesting the safetensors header of
`https://huggingface.co/LiquidAI/LFM2-1.2B/resolve/main/model.safetensors` (d=2048):

```
model.layers.0.conv.conv.weight       [2048, 1, 3]     BF16
model.layers.0.conv.in_proj.weight    [6144, 2048]     BF16
model.layers.0.conv.out_proj.weight   [2048, 2048]     BF16
model.layers.0.feed_forward.w1.weight [8192, 2048]     BF16
model.layers.0.feed_forward.w2.weight [2048, 8192]     BF16
model.layers.0.feed_forward.w3.weight [8192, 2048]     BF16
model.layers.0.ffn_norm.weight        [2048]           BF16
model.layers.0.operator_norm.weight   [2048]           BF16
```

Note the module path is `layers.N.conv.conv.weight` (a `Lfm2ShortConv` named `conv`
containing an `nn.Conv1d` named `conv`) — doubled, easy to typo. Conv weight is
`[d, 1, k]` = `[2048, 1, 3]`, i.e. `groups=d`. No bias tensors present, confirming
`conv_bias: false`.

### 1.5 Conv state / cache implementation `[CODE]`

The conv block routes through `transformers`' `LinearAttentionLayer` cache machinery
(`cache_utils.py`). Shape, from `LinearAttentionLayer.lazy_initialization`:

```python
self.conv_states[state_idx] = torch.zeros(
    (*conv_states.shape[:-1], conv_kernel_size), ...)
```

Since the tensor entering the cache is `B * x` of shape `(bsz, d, L)`, the stored conv
state has shape **`(batch, d, k)`** = `(batch, hidden_size, conv_L_cache)`. `[CODE]`+`[INFER]`
It is allocated **statically** (fixed shape, `torch._dynamo.mark_static_address`) — it does
**not** grow with sequence length. This is the whole point of the block.

Two code paths in `Lfm2ShortConv.forward`:
1. **Single-token cached decode** (`use_precomputed_states and seq_len == 1 and not record_past`):
   calls `causal_conv1d_update`, which concatenates `[conv_state, hidden_states]`, does
   `conv_state.copy_(hidden_states_new[:, :, -state_len:])` (in-place roll), and runs a
   `padding=0` `F.conv1d`, taking `out[:, :, -seq_len:]`. O(1) state, O(k·d) work per token.
2. **Prefill / multi-token**: calls `past_key_values.update_conv_state(...)` to get the
   concatenation of stored left-context + new tokens, runs `causal_conv1d_fn` with
   `padding=k-1`, then re-slices `hidden_states[:, :, -seq_len:]` to drop the extra
   prefix positions.

Both `causal_conv1d_update` and `causal_conv1d_fn` are decorated
`@maybe_replace_from_package("causal_conv1d", ...)`, so if the `causal_conv1d` CUDA package
is installed, the fused kernels are substituted automatically. The pure-PyTorch fallbacks
shown above are functionally equivalent. `[CODE]`

During prefill, `create_recurrent_attention_mask` is used for conv layers vs
`create_causal_mask` for attention layers — the model builds a dict
`{"full_attention": ..., "conv": ...}` and indexes it per layer by `config.layer_types[i]`. `[CODE]`

---

## 2. Attention block — exact spec

### 2.1 Paper `[PAPER]`

§2.2: "a majority of inexpensive, gated short convolution blocks interleaved with a
minority of grouped-query attention (**GQA**) blocks, plus SwiGLU position-wise
multi-layer perceptrons (MLPs). **All layers use pre-norm RMSNorm** (Zhang and Sennrich,
2019) and **attention blocks use RoPE** (Su et al., 2024) **with QK-Norm** (Dehghani et
al., 2023a)."

### 2.2 Code `[CODE]`

`Lfm2Attention` subclasses `LlamaAttention` with these overrides:

```python
self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
self.out_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
del self.o_proj
del self.attention_dropout
```

Forward order (this is load-bearing — QK-norm is applied **before** RoPE):

```python
query_states = self.q_layernorm(self.q_proj(h).view(*hidden_shape)).transpose(1, 2)
key_states   = self.k_layernorm(self.k_proj(h).view(*hidden_shape)).transpose(1, 2)
value_states = self.v_proj(h).view(*hidden_shape).transpose(1, 2)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

- **QK-Norm is per-head RMSNorm over `head_dim`** (learnable, shape `[64]`), applied to Q
  and K, **then** RoPE. `[CODE]`+`[CKPT]` (`q_layernorm.weight [64]`, `k_layernorm.weight [64]`)
- Output projection is named **`out_proj`**, NOT `o_proj` (`del self.o_proj`). This breaks
  naive Llama-weight-porting scripts. `[CODE]`
- No attention dropout at all (`del self.attention_dropout`; `dropout=0.0` hardcoded). `[CODE]`
- No attention bias anywhere. `[CODE]`
- `scaling = self.head_dim ** -0.5`, and `head_dim = getattr(config, "head_dim",
  config.hidden_size // config.num_attention_heads)` — so `head_dim` is **derived** (=64 for
  all released models) unless explicitly set. Both verified in the *generated*
  `modeling_lfm2.py` (`Lfm2Attention.__init__`), not just inherited from Llama. `[CODE]`
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py
- `Lfm2RotaryEmbedding` subclasses **`Gemma2RotaryEmbedding`** (not Llama's). `[CODE]`
- `rope_theta = 1000000.0` in every released config; `Lfm2Config.default_theta = 1000000.0`. `[CONFIG]`+`[CODE]`
- **No sliding window.** `full_attention` is the only attention layer type present in any
  released config. Every attention layer is full/global. `[CONFIG]`

### 2.3 Head geometry `[CONFIG]`

`head_dim = 64` for all models. Every released dense and MoE config has
`num_key_value_heads: 8`, so **8 KV groups, head size 64 → KV width = 512** regardless of
model size. `num_attention_heads` scales with d such that `num_attention_heads * 64 == d`
exactly (16×64=1024, 24×64=1536, 32×64=2048). So `q_proj` and `out_proj` are square `d×d`,
and `k_proj`/`v_proj` are `d × 512`. Paper Table 1 confirms: `H/KV/H_size` = 16/8/64,
24/8/64, 32/8/64, 32/8/64, 32/8/64. `[PAPER]`+`[CONFIG]`

**Consequence: at d=2048 the GQA ratio is 4:1 and KV width is d/4.** This is what makes the
2.5d² figure work out — see §4.

---

## 3. Decoder layer, MLP, norms, embeddings

### 3.1 Decoder layer structure `[CODE]`

```python
class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer: self.self_attn = Lfm2Attention(config, layer_idx)
        else:                       self.conv      = Lfm2ShortConv(config, layer_idx)
        self.feed_forward  = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm      = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(...):
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(hidden_states=self.operator_norm(hidden_states), ...)
        else:
            hidden_states = self.conv(hidden_states=self.operator_norm(hidden_states), ...)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states
```

**Every layer — conv or attention — has its own MLP.** The structure is uniform:
`x + Mixer(operator_norm(x))`, then `x + MLP(ffn_norm(x))`. Standard pre-norm, two
residual branches per layer. The mixer slot is the *only* thing that varies. `[CODE]`

This is a clean property for the experiment: swapping a layer between conv and attention
changes exactly one submodule and leaves the MLP, both norms, and the residual topology
untouched.

Naming: the pre-mixer norm is **`operator_norm`** (not `input_layernorm`) and the pre-MLP
norm is **`ffn_norm`** (not `post_attention_layernorm`). The MLP is **`feed_forward`**
(not `mlp`). `[CODE]`+`[CKPT]`

### 3.2 MLP `[CODE]`

```python
class Lfm2MLP(nn.Module):
    def __init__(self, config):
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of)
        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

SwiGLU, three bias-free matrices, `w1` = gate, `w3` = up, `w2` = down.

**MAJOR TRAP — `intermediate_size` in the config is NOT the actual FFN width when
`block_auto_adjust_ff_dim: true`.** For LFM2-1.2B, config says `intermediate_size / block_ff_dim
= 12288`, but the effective width is:
`int(2*12288/3) = 8192` → `int(1.0*8192) = 8192` → round up to multiple of 256 = **8192**.
Verified against the checkpoint: `feed_forward.w1.weight` is `[8192, 2048]`. `[CKPT]`
Paper Table 1 lists "FF dim = 8192" for LFM2-1.2B — i.e. **the paper reports the effective
width, the config reports the pre-adjustment value.** `[PAPER]` vs `[CONFIG]`

Effective FF widths (computed with the code above, verified against checkpoints):
- LFM2-350M: config 6656 → **4608** (paper Table 1: 4608 ✓)
- LFM2-700M: config 10240 → **6912** (paper: 6912 ✓)
- LFM2-1.2B: config 12288 → **8192** (paper: 8192 ✓)
- LFM2-2.6B: `block_auto_adjust_ff_dim: false` → **10752** as-is (paper: 10752 ✓)
- LFM2.5-1.2B-Base/Instruct: config 12288, adjust=true → **8192**

Expansion ratio (effective FF / d): 4.5× (350M), 4.5× (700M), 4.0× (1.2B), 5.25× (2.6B).
`[INFER]` The paper says only "SwiGLU with a size-dependent expansion ratio chosen by the
search". `[PAPER]` These ratios are high vs a typical 2.67× Llama SwiGLU — consistent with
the mixers being cheap and the search pushing params into FFNs.

Also note `Lfm2Config.__post_init__` does `self.intermediate_size = kwargs.pop("block_ff_dim",
self.intermediate_size)` — so legacy configs carrying `block_ff_dim` alias onto
`intermediate_size`. `[CODE]`

### 3.3 Norms and embeddings `[CODE]`+`[CKPT]`

- `Lfm2RMSNorm` is a bare subclass of `LlamaRMSNorm` — standard RMSNorm, learnable gain,
  no bias, `eps = norm_eps = 1e-5` in all configs. `[CODE]`+`[CONFIG]`
- `Lfm2Model` **deletes `self.norm`** and adds **`self.embedding_norm`**, applied as the
  final norm after the last decoder layer (`hidden_states = self.embedding_norm(hidden_states)`).
  So despite the name, `embedding_norm` is the **final** norm, not an input norm. There is
  **no** norm applied to the embeddings on the way in. `[CODE]` Confirmed in checkpoint:
  a single `model.embedding_norm.weight [2048]` and no `model.norm.weight`. `[CKPT]`
- **Tied embeddings.** `Lfm2Config.tie_word_embeddings: bool = True` (default). Configs use
  the legacy key `tie_embedding: true`, remapped in `__post_init__` via
  `self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)`.
  Verified empirically: the LFM2-1.2B, LFM2.5-1.2B-Base, and LFM2-2.6B safetensors headers
  contain **no `lm_head.weight`** at all. `[CODE]`+`[CONFIG]`+`[CKPT]`
- `vocab_size = 65536` in every config; paper: "byte-level BPE tokenizer with a
  65,536-token vocabulary". `[CONFIG]`+`[PAPER]`
- Special tokens: `bos=1`, `eos=7`, `pad=0`. `[CONFIG]`

### 3.4 Training scale and context schedule `[PAPER]`

§3.2 "Training Stages", verbatim:

> The released dense LFM2 model checkpoints are **pre-trained for 10T tokens at a context
> length of 4,096 tokens**. We then perform a **mid-training phase on an additional 1T
> higher-quality tokens**, including sources with naturally long context, using a
> **32,768-token context window** and an accelerated learning-rate decay schedule. The
> released LFM2-8B-A1B checkpoint follows the same two-stage recipe but is trained for 12T
> tokens in the initial phase before the 1T-token long-context mid-training stage.

Also relevant to reproducibility: **the released checkpoints are distillation products, not
plain next-token-prediction models.** §3.3: "During pre-training, we leverage **LFM1-7B as a
teacher model** in a knowledge distillation (KD) framework… using only the **Top-K=32**
teacher logits per token", with a "decoupled" KL decomposed into a binary mass-matching term
plus a within-Top-K conditional KL. **LFM1-7B is not publicly released**, so this pre-training
recipe is not reproducible outside Liquid AI. `[PAPER]`+`[INFER]`

Two consequences for experiment design `[INFER]`:
1. The architecture was selected under a **4,096-token** pre-training context (32K only in a
   1T-token mid-training tail). So the 10/6 ratio is optimal *for 4K training*, which sits
   right at the FLOP crossover computed in §7.5. This makes an independent ratio sweep at
   4K a fair reproduction of their regime.
2. Any quality comparison against released LFM2 checkpoints is confounded by 11T tokens of
   distillation from a private 7B teacher. A from-scratch ablation cannot be compared to
   published LFM2 benchmark scores; it can only be compared internally against baselines
   you train yourself under the same budget.

### 3.5 Context length — paper and config DISAGREE

- Every released config: `max_position_embeddings: 128000`. `[CONFIG]`
- Paper §2.2 "Released sizes": "We release dense checkpoints at 350M, 700M, 1.2B, and 2.6B
  parameters, all with a **32,768 token context window**." `[PAPER]`
- Paper abstract also says "32K context window". `[PAPER]`

So `max_position_embeddings=128000` is **not** a validated 128K capability — the trained/
supported window is 32,768 per the paper. Treat 128000 as a positional-embedding headroom
value, not a claim. Long-context evaluation beyond 32K is unsupported by the paper.

---

## 4. Interleaving pattern — exact layer order

### 4.1 Two config encodings `[CONFIG]`+`[CODE]`

Older configs (350M, 700M, 1.2B) use **`full_attn_idxs`**; newer ones (2.6B, 8B-A1B,
24B-A2B, LFM2.5) use an explicit **`layer_types`** list. `Lfm2Config.__post_init__`
bridges them:

```python
if self.layer_types is None:
    self.full_attn_idxs = self.full_attn_idxs if self.full_attn_idxs is not None else list(range(self.num_hidden_layers))
    self.layer_types = ["full_attention" if i in self.full_attn_idxs else "conv" for i in range(self.num_hidden_layers)]
```

Note the default when *neither* is given is **all-attention** (`range(num_hidden_layers)`),
which is a footgun if you build a config from scratch and forget to set the pattern. `[CODE]`

### 4.2 The 16-layer pattern (350M / 700M / 1.2B / LFM2.5-1.2B) `[CONFIG]`

`full_attn_idxs: [2, 5, 8, 10, 12, 14]` — identical in LFM2-350M, LFM2-700M, LFM2-1.2B.
LFM2.5-1.2B-Base and LFM2.5-1.2B-Instruct spell out `layer_types`, which decodes to the
**same** indices {2, 5, 8, 10, 12, 14}.

Layer-by-layer (0-indexed), all four models:

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| type | C | C | **A** | C | C | **A** | C | C | **A** | C | **A** | C | **A** | C | **A** | C |

**10 conv + 6 attention.** Paper Table 1: "Attn. Blocks = 6" for 350M/700M/1.2B. `[PAPER]`

Structural observations `[INFER]`:
- Gaps between attention layers are **3, 3, 3, 2, 2, 2** — the pattern *densifies* toward
  the top of the network. Attention is sparser early, denser late.
- Layers 0 and 1 are both conv (no attention until layer 2), and the **last layer (15) is
  conv**. So the network neither begins nor ends with attention.
- This is NOT a uniform "every Nth layer" stride. Any experiment that models LFM2's
  interleaving as periodic (e.g. `attn if i % 3 == 2`) will reproduce {2,5,8,11,14} —
  which differs from the real pattern at layers 10, 11, 12. Use the explicit index list.

### 4.3 Full family patterns `[CONFIG]`

| Model | Layers | attn indices | conv | attn | conv:attn |
|---|---|---|---|---|---|
| LFM2-350M | 16 | 2,5,8,10,12,14 | 10 | 6 | 1.67 |
| LFM2-700M | 16 | 2,5,8,10,12,14 | 10 | 6 | 1.67 |
| LFM2-1.2B | 16 | 2,5,8,10,12,14 | 10 | 6 | 1.67 |
| LFM2.5-1.2B-Base | 16 | 2,5,8,10,12,14 | 10 | 6 | 1.67 |
| LFM2.5-1.2B-Instruct | 16 | 2,5,8,10,12,14 | 10 | 6 | 1.67 |
| LFM2-2.6B | 30 | 2,5,9,13,17,21,24,27 | 22 | 8 | 2.75 |
| LFM2-8B-A1B (MoE) | 24 | 2,6,10,14,18,21 | 18 | 6 | 3.00 |
| LFM2-24B-A2B (MoE) | 40 | 2,6,10,14,18,22,26,30,34,38 | 30 | 10 | 3.00 |

Gap sequences (differences between consecutive attention indices), computed from the
configs:
- LFM2-350M/700M/1.2B/LFM2.5-1.2B: **3, 3, 2, 2, 2** — densifies toward the top.
- LFM2-2.6B: **3, 4, 4, 4, 4, 3, 3** — dense at both ends, stride-4 through the middle;
  last layers 28, 29 are conv.
- LFM2-8B-A1B: **4, 4, 4, 4, 3** — stride 4 then one tightening at the top.
- LFM2-24B-A2B: **4, 4, 4, 4, 4, 4, 4, 4, 4** — attention at exactly `i ≡ 2 (mod 4)` for
  i = 2…38. **This one is perfectly periodic** (verified programmatically).

**Every released model ends on a conv layer and starts with two conv layers** (attention
never appears before index 2). `[CONFIG]`

**So the "10-conv/6-attn" ratio is specific to the 16-layer 350M–1.2B class. Larger models
use a LOWER attention fraction**: 6/16 = **37.5%** → 8/30 = **26.7%** (2.6B) →
6/24 = **25.0%** (8B-A1B) → 10/40 = **25.0%** (24B-A2B). The fraction appears to converge
toward ~25% (stride-4) as depth grows. `[INFER]` from configs. The paper does not explain
this scaling, and does not cover 24B-A2B at all.

### 4.4 MoE variants `[CONFIG]`+`[PAPER]`

`LFM2-8B-A1B`: `architectures: ["Lfm2MoeForCausalLM"]`, `model_type: "lfm2_moe"`,
`num_experts: 32`, `num_experts_per_tok: 4`, `moe_intermediate_size: 1792`,
`num_dense_layers: 2`, `use_expert_bias: true`, `norm_topk_prob: true`,
`routed_scaling_factor: 1.0`.
Paper §2.3: "8.3B total, 1.5B active… We keep the fast LFM2 hybrid backbone (Section 2.2)
and replace dense MLPs with sparse MoE MLPs in most layers. For stability, the first two
layers remain dense while all subsequent layers include an MoE block. Experts themselves
are SwiGLU MLPs. Each MoE layer has 32 experts and selects the Top-k=4 experts per token
with a normalized sigmoid router and adaptive routing biases for load balancing."

`LFM2-24B-A2B` (**exists on the Hub but is NOT in the paper's Table 1**): 40 layers,
`num_experts: 64`, `moe_intermediate_size: 1536`, `intermediate_size: 11776`,
`transformers_version: "5.0.0rc1"`, and uses the newer nested
`rope_parameters: {rope_theta: 1000000.0, rope_type: "default"}` form. `[CONFIG]`
https://huggingface.co/LiquidAI/LFM2-24B-A2B/blob/main/config.json

**The MoE variants only sparsify the MLP. The mixer stack (conv/attention interleave) is
unchanged.** `[PAPER]`+`[CONFIG]`

---

## 5. LFM2.5 — architectural delta vs LFM2

**Finding: for LFM2.5-1.2B, there is NO architectural delta. It is bit-for-bit the same
architecture as LFM2-1.2B.** `[CONFIG]`+`[CKPT]` Liquid AI's own announcement confirms this:
*"It **builds on the LFM2 device-optimized architecture**"* —
https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai (Jan 5, 2026)

Evidence:
- Same modeling class: `architectures: ["Lfm2ForCausalLM"]`, `model_type: "lfm2"`. There is
  no `Lfm2_5` / `Lfm25` model directory in `transformers`. `[CONFIG]`
- Same everything: `hidden_size 2048`, `num_hidden_layers 16`, `intermediate_size 12288`
  (→ effective 8192), `num_attention_heads 32`, `num_key_value_heads 8`, `conv_L_cache 3`,
  `rope_theta 1000000.0`, `max_position_embeddings 128000`, `norm_eps 1e-05`,
  `vocab_size 65536`, `tie_embedding true`. `[CONFIG]`
- Same interleaving: `layer_types` decodes to attention at {2,5,8,10,12,14}. `[CONFIG]`
- **Identical parameter count**: HF safetensors metadata reports **1,170,340,608** for both
  `LiquidAI/LFM2-1.2B` and `LiquidAI/LFM2.5-1.2B-Base`, with **148 tensors each** and the
  same tensor names/shapes. `[CKPT]`

Config diffs are cosmetic/version-related only: LFM2.5 uses `dtype` instead of
`torch_dtype`, drops `conv_dim_out`, adds explicit `intermediate_size` alongside
`block_ff_dim`, and reports `transformers_version: "4.57.2"` vs `"4.54.0.dev0"`. `[CONFIG]`

**Conclusion: LFM2.5 is a data/training-recipe release on an unchanged LFM2 backbone**
(at least at 1.2B). `[INFER]`, now corroborated by the announcement blog.

**There is no separate LFM2.5 technical report.** The LFM2.5 blog's own "technical report"
link points back to arXiv:2511.23404 (the LFM2 report), and an arXiv title search for
"LFM2.5" returns 0 entries. LFM2.5's architecture is documented only by blog + model card +
config.

What actually changed in LFM2.5, per the announcement blog:
- **Pretraining tokens: 10T → 28T.** *"We've extended pretraining from 10T to 28T tokens and
  significantly scaled up our post-training pipeline with reinforcement learning…"*
- **Post-training adds large-scale multi-stage RL.** LFM2 was SFT → length-normalized DPO →
  model merging; LFM2.5-Instruct is *"trained with supervised fine-tuning, preference
  alignment, and large-scale multi-stage reinforcement learning."*
- **New variants**: Base, Instruct, **Thinking** (new reasoning variant), JP, VL-1.6B,
  Audio-1.5B; later extended to 230M/350M and 8B-A1B.
- **Audio detokenizer replaced** (custom LFM-based, ~8× faster than LFM2's Mimi on mobile CPU
  at INT4 QAT).
- **NOT changed**: architecture, context window (still **32,768**), params (1.17B),
  vocab (65,536).

**Context length is commonly misreported for LFM2.5 too** — `max_position_embeddings: 128000`
in the config, but every official source says the trained/supported window is **32,768**.
See §3.5.

Sources:
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/blob/main/config.json
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/config.json

LFM2.5 checkpoints found on the Hub (via https://huggingface.co/api/models?author=LiquidAI):
`LFM2.5-1.2B-Base`, `LFM2.5-1.2B-Base-GGUF`, `LFM2.5-1.2B-Base-ONNX`,
`LFM2.5-1.2B-Instruct`, `LFM2.5-1.2B-Instruct-GGUF`, and MLX quantizations
(4/5/6/8-bit, bf16). **Only the 1.2B size exists in LFM2.5 as of this research.**

---

## 6. Parameter counts — exact formulas, verified against real checkpoints

### 6.1 Formulas `[INFER]` (derived from `[CODE]` module definitions)

Let d = `hidden_size`, k = `conv_L_cache`, F = **effective** FFN width (§3.2),
H = `num_attention_heads`, G = `num_key_value_heads`, h = `head_dim` (=64), V = `vocab_size`,
L = `num_hidden_layers`, with all biases off and embeddings tied.

**Conv (LIV) mixer, per layer:**
```
P_conv = 3d²   (in_proj)  +  d·k  (depthwise conv)  +  d²  (out_proj)
       = 4d² + d·k
```

**GQA mixer, per layer:**
```
P_attn = d·(H·h)  (q_proj)  +  2·d·(G·h)  (k_proj, v_proj)  +  (H·h)·d  (out_proj)  +  2h  (q/k layernorm)
```
When `H·h == d` (true for all released dense LFM2 models):
```
P_attn = 2d² + 2d·(G·h) + 2h
```
and with G·h = d/4 (i.e. G=8, h=64, d=2048): `P_attn = 2d² + 0.5d² + 2h = 2.5d² + 2h`.

**MLP, per layer:** `P_mlp = 3·d·F`
**Norms, per layer:** `2d` (`operator_norm` + `ffn_norm`)
**Embeddings:** `V·d` (once; tied, so no separate lm_head)
**Final norm:** `d` (`embedding_norm`)

**Total:**
```
P_total = V·d + d + L·(3dF + 2d) + n_conv·(4d² + dk) + n_attn·(2d² + 2d·G·h + 2h)
```

### 6.2 Verification — 5 checkpoints, exact match

I computed `P_total` from each `config.json` (applying the `block_auto_adjust_ff_dim`
transform) and compared against the HF-reported safetensors parameter totals from
`https://huggingface.co/api/models/LiquidAI/<repo>?expand[]=safetensors`:

| Model | conv mixer | attn mixer | MLP | **Predicted** | **HF actual** | Match |
|---|---|---|---|---|---|---|
| LFM2-350M | 4,197,376 | 3,145,856 | 14,155,776 | **354,483,968** | 354,483,968 | ✅ |
| LFM2-700M | 9,441,792 | 6,291,584 | 31,850,496 | **742,489,344** | 742,489,344 | ✅ |
| LFM2-1.2B | 16,783,360 | 10,485,888 | 50,331,648 | **1,170,340,608** | 1,170,340,608 | ✅ |
| LFM2-2.6B | 16,783,360 | 10,485,888 | 66,060,288 | **2,569,272,320** | 2,569,272,320 | ✅ |
| LFM2.5-1.2B-Base | 16,783,360 | 10,485,888 | 50,331,648 | **1,170,340,608** | 1,170,340,608 | ✅ |

**Exact to the parameter on all five.** Independently cross-checked by summing every tensor
in the LFM2-1.2B safetensors header: **1,170,340,608** (148 tensors), and LFM2-2.6B:
**2,569,272,320** (266 tensors). `[CKPT]`

Tensor-count sanity check `[INFER]`: LFM2-1.2B = 2 (embed + embedding_norm) + 16×5
(3 MLP + 2 norms) + 10×3 (conv: conv, in_proj, out_proj) + 6×6 (attn: q,k,v,out + 2
layernorms) = 2 + 80 + 30 + 36 = **148** ✓. LFM2-2.6B = 2 + 30×5 + 22×3 + 8×6 = 2 + 150 +
66 + 48 = **266** ✓. This confirms the module inventory is complete — there is no hidden
parameter I've missed (in particular, no norm inside the conv block).

**Sixth verification — the MoE variant.** Extending the formula with
`P_moe_mlp = E·3·d·F_moe + d·E` (experts + router gate) for the `L − num_dense_layers`
sparse layers, LFM2-8B-A1B (d=2048, L=24, n_conv=18, n_attn=6, F=7168, E=32, F_moe=1792,
num_dense_layers=2) predicts **8,339,929,856** against HF's reported BF16 total of
**8,339,929,856** — exact. And the leftover `{'F32': 704}` reported separately by the HF API
is precisely `22 sparse layers × 32 experts = 704`, i.e. the `use_expert_bias: true`
per-expert routing biases held in fp32. `[CKPT]`+`[INFER]`

Worked arithmetic for **LFM2-1.2B** (d=2048, k=3, F=8192, H=32, G=8, h=64, V=65536, L=16,
n_conv=10, n_attn=6):
```
embeddings      65536 × 2048                        =   134,217,728
final norm                                     2048 =         2,048
per-layer MLP   3 × 2048 × 8192  = 50,331,648  × 16 =   805,306,368
per-layer norms 2 × 2048         =      4,096  × 16 =        65,536
conv mixers     4(2048²) + 2048·3 = 16,783,360 × 10 =   167,833,600
attn mixers     2(2048²)+2·2048·512+128 = 10,485,888 × 6 =  62,915,328
                                              TOTAL  = 1,170,340,608  ✅
```

### 6.3 The brainlift's claims — CHECKED, both are CORRECT

The brainlift claims a stock LIV mixer at d=2048, k=3 is ~16.783M params and a comparable
GQA mixer is ~2.5d² = 10.486M.

**LIV mixer: 4d² + dk = 4(2048²) + 2048·3 = 16,777,216 + 6,144 = 16,783,360 = 16.783M. ✅
The brainlift's figure is exactly right.**

**GQA mixer: the 2.5d² figure IS correct for LFM2's actual GQA configuration.** The task
brief flagged this as suspect because it depends on `num_key_value_heads` — and rightly so
in general, but it happens to check out here. Working:
- `num_attention_heads = 32`, `head_dim = 64` → Q width = 2048 = **d** → `q_proj` = d² = 1.0d²
- `num_key_value_heads = 8`, `head_dim = 64` → KV width = 512 = **d/4**
  → `k_proj` = d·(d/4) = 0.25d², `v_proj` = 0.25d²
- `out_proj` = d² = 1.0d²
- Sum = **2.5d² = 10,485,760**, plus 2×64 = 128 for the q/k RMSNorms → **10,485,888**.

Verified against the checkpoint: `q_proj [2048,2048]` + `k_proj [512,2048]` +
`v_proj [512,2048]` + `out_proj [2048,2048]` + `q_layernorm [64]` + `k_layernorm [64]`
= 4,194,304 + 1,048,576 + 1,048,576 + 4,194,304 + 64 + 64 = **10,485,888**. `[CKPT]`

So: **no correction needed. Both brainlift figures are right, and the 2.5d² coefficient is
correct *specifically because* LFM2 uses 8 KV heads of dim 64 at d=2048 (4:1 GQA, KV width
= d/4).** The coefficient is `2 + 2·(G·h/d)`; it would be 3.0d² at 2:1 GQA and 4.0d² at MHA.
State this dependency explicitly if the write-up generalizes across d, since at d=1024
(LFM2-350M) G·h = 512 = d/2, giving `2 + 1 = 3.0d²` → 3,145,728 + 128 = **3,145,856**.
Confirmed against the 350M table row above. `[INFER]`+`[CKPT]`

### 6.4 The headline consequence for experiment design

**At d=2048, the LIV mixer is 1.60× LARGER than the GQA mixer** (16.783M vs 10.486M).
Ratio = `(4d² + dk)/(2.5d² + 2h)` ≈ 1.6. `[INFER]`

This inverts the naive intuition. The conv block is cheap in **FLOPs at long context**
(O(L·d·k) vs O(L²·d)) and cheap in **cache** (O(d·k) fixed vs O(L·G·h)), but it is
**expensive in parameters**, because `in_proj` is 3d wide (3d² alone exceeds the 2.5d² of
the entire GQA mixer) while GQA gets a 4× discount on K and V.

Implications for a "mostly-LIV" ablation:
- Replacing an attention layer with a conv layer at fixed d **increases** parameter count
  by 6.30M per swap at d=2048. A conv-vs-attention ratio sweep at fixed d is therefore
  **not parameter-matched**, and the confound points the *wrong way* (more-conv = more
  params). LFM2-1.2B's 16 layers at 10/6 carry 230.7M mixer params; an all-attention
  16-layer variant would carry 167.8M — a 62.9M (5.4% of total model) difference.
- To parameter-match, you must compensate — e.g. adjust F (each 256 of FFN width at d=2048
  is 1.57M params/layer) or trim `in_proj`. Note LFM2's own scaling *does* shift params
  into FFNs (expansion 4.0–5.25×), so F is the natural free variable.
- At d=1024 the ratio is `(4·1024² + 3072)/(3.0·1024² + 128)` = 4,197,376/3,145,856 =
  **1.33×** — the penalty shrinks as GQA becomes relatively more expensive at small d.
  So the conv/attention param tradeoff is **d-dependent**; do not extrapolate a single
  ratio across model sizes.

---

## 7. Cache / state accounting

### 7.1 Formulas `[INFER]` (from `[CODE]` shapes; b = bytes per element, 2 for fp16/bf16)

**Conv state, per conv layer, per sequence — INDEPENDENT of context length:**
```
bytes_conv = batch · d · k · b          # shape (batch, d, k), statically allocated
```
**KV cache, per attention layer, per token:**
```
bytes_kv = batch · 2 · G · h · b        # 2 for K and V
```
**Whole-model totals at context T, batch 1, bf16:**
```
total_conv = n_conv · d · k · 2                     (fixed)
total_kv   = n_attn · 2 · G · h · 2 · T             (linear in T)
```

### 7.2 Per-model constants (bf16, batch=1)

| Model | conv state (total, fixed) | KV per token (all attn layers) |
|---|---|---|
| LFM2-350M | 61,440 B = **60.0 KiB** | 12,288 B = **12.00 KiB/tok** |
| LFM2-700M | 92,160 B = **90.0 KiB** | 12,288 B = **12.00 KiB/tok** |
| LFM2-1.2B | 122,880 B = **120.0 KiB** | 12,288 B = **12.00 KiB/tok** |
| LFM2.5-1.2B | 122,880 B = **120.0 KiB** | 12,288 B = **12.00 KiB/tok** |
| LFM2-2.6B | 270,336 B = **264.0 KiB** | 16,384 B = **16.00 KiB/tok** |
| LFM2-8B-A1B | 221,184 B = **216.0 KiB** | 12,288 B = **12.00 KiB/tok** |
| LFM2-24B-A2B | 368,640 B = **360.0 KiB** | 20,480 B = **20.00 KiB/tok** |

Per-layer: conv state = `d·k·2` = 12,288 B (12 KiB) at d=2048; KV = `2·8·64·2` = 2,048 B
(2 KiB) per token per attention layer, **identical across all models** since G·h = 512
universally.

**Break-even: one attention layer's KV cache equals one conv layer's entire state after
just 6 tokens** (12,288 / 2,048 = 6) at d=2048. `[INFER]`

### 7.3 Worked table — total cache at context T (bf16, batch=1)

**LFM2-1.2B** (10 conv, 6 attn, d=2048): conv = 120.0 KiB fixed, KV = 12.00 KiB/token

| T | KV cache | conv state | conv share of total | KV/conv ratio |
|---|---|---|---|---|
| 4,096 | 48.00 MiB | 0.117 MiB | 0.244% | 410× |
| 16,384 | 192.00 MiB | 0.117 MiB | 0.061% | 1,638× |
| 32,768 | 384.00 MiB | 0.117 MiB | 0.031% | 3,277× |
| 131,072 | 1,536.00 MiB | 0.117 MiB | 0.008% | 13,107× |

**LFM2-350M** (10 conv, 6 attn, d=1024): conv = 60.0 KiB fixed, KV = 12.00 KiB/token

| T | KV cache | conv state | conv share | KV/conv |
|---|---|---|---|---|
| 4,096 | 48.00 MiB | 0.059 MiB | 0.122% | 819× |
| 16,384 | 192.00 MiB | 0.059 MiB | 0.031% | 3,277× |
| 32,768 | 384.00 MiB | 0.059 MiB | 0.015% | 6,554× |
| 131,072 | 1,536.00 MiB | 0.059 MiB | 0.004% | 26,214× |

**LFM2-2.6B** (22 conv, 8 attn, d=2048): conv = 264.0 KiB fixed, KV = 16.00 KiB/token

| T | KV cache | conv state | conv share | KV/conv |
|---|---|---|---|---|
| 4,096 | 64.00 MiB | 0.258 MiB | 0.401% | 248× |
| 16,384 | 256.00 MiB | 0.258 MiB | 0.101% | 993× |
| 32,768 | 512.00 MiB | 0.258 MiB | 0.050% | 1,986× |
| 131,072 | 2,048.00 MiB | 0.258 MiB | 0.013% | 7,944× |

### 7.4 Counterfactual — what the hybrid actually buys `[INFER]`

For LFM2-1.2B, if all 16 layers were GQA instead of 6:
- KV/token: 32.00 KiB → LFM2's actual 12.00 KiB = **2.67× reduction** (exactly 16/6)
- At T=4,096: 128.0 MiB → 48.0 MiB
- At T=32,768: 1,024.0 MiB → 384.0 MiB
- At T=131,072: 4,096.0 MiB → 1,536.0 MiB

**The conv state is numerically negligible — it rounds to zero against the KV cache at any
realistic context.** The memory win comes *entirely* from having 10 layers that hold no
per-token state at all, not from the conv state being "small". Design point: for a
cache-memory story, the metric that matters is `n_attn` (and G·h), not anything about the
conv block. This also means kernel size k is essentially free from a memory standpoint —
k=3 → k=7 would take LFM2-1.2B's conv state from 120 KiB to 280 KiB, still 0.07% of the
KV cache at 32K. A kernel-size ablation costs almost nothing in memory and only
`n_conv·d·(k−3)` in params (10·2048·4 = 81,920 params for k=7, i.e. 0.007% of the model).

### 7.5 FLOP crossover — the conv block is NOT cheaper at short context `[INFER]`

Because the conv mixer's params are 1.6× the attention mixer's (§6.3), its *dense matmul*
cost is also 1.6×, and that cost is paid at every sequence length. Attention only overtakes
it once the quadratic `T²` term dominates. Forward-pass FLOPs per layer, LFM2-1.2B geometry
(d=2048, k=3, H=32, G=8, h=64), counting 2 FLOPs per MAC, whole sequence:

```
conv:  T·(2·d·3d  +  2·d·k  +  2·d·d)                     # in_proj + depthwise + out_proj
attn:  T·(2·d·d + 2·2·d·G·h + 2·d·d)  +  2·2·T²·d          # q,k,v,out projections + QK^T,AV
```

| T | conv mixer (GFLOP) | attn mixer (GFLOP) | attn / conv |
|---|---|---|---|
| 1,024 | 34.4 | 30.1 | **0.87×** |
| 4,096 | 137.5 | 223.3 | 1.62× |
| 16,384 | 550.0 | 2,542.6 | 4.62× |
| 32,768 | 1,099.9 | 9,483.3 | 8.62× |
| 131,072 | 4,399.7 | 143,486.3 | 32.61× |

**Crossover is around T ≈ 1.4K.** Below that, the "cheap" conv block is *more* expensive in
FLOPs than a GQA block. This matters a lot for experiment design:
- If the experiment trains at a **short sequence length (1–2K)**, a mostly-LIV model will be
  **slower and larger** than an attention baseline at matched d — the architecture's entire
  advantage is invisible in that regime. LFM2's own reported wins are at 1K/4K prompts on
  quantized CPU where memory bandwidth, not FLOPs, is the binding constraint.
- The genuine advantages that survive at short context are (a) no per-token KV state and
  (b) memory-bandwidth/cache locality on CPU — not FLOP count.
- Recommend training/evaluating at **≥4K sequence length**, and reporting KV-cache bytes and
  decode-time memory traffic rather than FLOPs, if the goal is to reproduce LFM2's claims.

(These are forward-pass mixer-only FLOPs; MLPs — which dominate total FLOPs at 3dF = 50.3M
params/layer — are identical between the two layer types and so cancel in any ratio sweep.)

---

## 8. Architecture search — how the ratios were chosen

### 8.1 LFM2's search is NOT STAR, and the paper says so explicitly `[PAPER]`

This is the most important finding in this section. §2.1 of
https://arxiv.org/html/2511.23404v1, verbatim:

> Our earlier academic prototype (**STAR**) (Thomas et al., 2024) explored a specific design
> space of operator/layout choices with an evolutionary search heuristic optimized on proxy
> signals (i.e., perplexity for quality, cache size for efficiency). **In practice, these
> proxies do not transfer reliably to downstream task scores or device-level latency and
> memory, limiting their utility as optimization objectives.** By contrast, the LFM2
> pipeline centers the objective: downstream task scores and hardware-in-the-loop
> TTFT/latency/memory on release runtimes. In practice, we found **this has a much larger
> impact than the particulars of the search space or choice of search heuristic.**

So: **the 10-conv/6-attn ratio is NOT the output of STAR.** It is the output of a different,
later, hardware-in-the-loop procedure, and the LFM2 authors explicitly disown STAR's
proxy objectives (perplexity + cache size) as non-transferring. Any claim that LFM2's
ratios "come from STAR" is contradicted by the primary source. STAR is cited exactly twice
in the whole report.

### 8.1b What STAR actually is, and where the "10-conv/6-attn from STAR" claim comes from

**STAR paper**: *STAR: Synthesis of Tailored Architectures*, Armin W. Thomas, Rom Parnichkun,
Alexander Amini, Stefano Massaroli, Michael Poli — all Liquid AI. **ICLR 2025** (DBLP
`conf/iclr/ThomasPAMP25`, OpenReview `HsHxSN23rM`). https://arxiv.org/abs/2411.17800
**v1 only, PDF-only — `arxiv.org/html/2411.17800v1` does not exist** (returns an error stub).
Note the author overlap with the LFM2 report: **Amini and Parnichkun are on both.**

**This is where "LIV" is defined.** An LIV (linear input-varying) operator is one whose matrix
is generated from the input:

> y_i^α = Σ_{j∈[ℓ]} Σ_{β∈[d]} T_ij^{αβ}(x) · x_j^β

with the taxonomy by operator structure (STAR's own table):

| T_ij | structure | class |
|---|---|---|
| σ(C_i B_j) | dense | attention |
| C_i B_j | low-rank | linear attention |
| C_i A_{i−1}···A_{j+1} B_j | semi-separable | linear recurrence |
| **C_i K_{i−j} B_j** | **scaled Toeplitz** | **gated convolution** |
| σ(C) if i=j else 0 | diagonal | memoryless system |

The LFM2 conv block is the **scaled-Toeplitz / gated-convolution** row — this is the precise
sense in which calling it a "LIV" block is legitimate. STAR's genome is a sequence of
5-integer segments per LIV (class; featurizer-sharing group; featurization-sharing strategy;
feature-group-sharing group; feature-group-sharing strategy), over a 17-class alphabet
(4 softmax-attention variants, 2 recurrences, 2 gated convolutions, 1 gated memoryless unit,
and differential variants of all 8). Its featurizer class 7 is described as using
**"short convolutions of length 3"** — matching LFM2's shipped `conv_L_cache: 3`.

**STAR's objectives — and why they are NOT LFM2's.** STAR runs three *separate* two-objective
searches: quality only (perplexity); quality + parameter count; quality + cache size
(KV + fixed state). Multi-objective via **NSGA-2** Pareto dominance. **STAR never optimizes
measured device latency** — quality is proxied by perplexity and efficiency by static
cache/param counts. This is precisely what the LFM2 report says fails to transfer (§8.1).

**STAR does NOT report conv:attention ratios.** Its results tables (5.1, 5.2, A.4) report only
Size / Cache / perplexity / 5–6 downstream accuracies per backbone — **no layer-composition
columns.** The discovered topologies live in Appendix B Figures B.1–B.25, which are **pure
vector graphics with no extractable text**, so the actual per-layer LIV sequences (and hence
any ratio) **could not be recovered from the paper**. The closest quantitative statement is
§5.5 on which operators are *favored*, not in what proportion:

> We observe that STAR favors gated short convolutions (GConv-1), grouped query (Ainslie et
> al., 2023) attention variants (SA-3), and differential variants of input-varying recurrences
> (Rec-1-Diff), as well as SwiGLUs (Shazeer, 2020) (GMemless).

Note this favors GConv + GQA + SwiGLU (LFM2's ingredient list) **but also recurrences**, which
LFM2 dropped.

STAR's scale, for calibration: main experiments are **125M params, 24 LIVs @ width 768,
population 16, 18 generations**, proxy models trained **1.3B tokens** then re-trained 5B; a
single 1B-scale datapoint (48 LIVs @ 2048, 40B tokens) reports STAR-1B at 1.1B params /
**86MB cache** / 5.7 ppl vs Transformer++ 1.2B / 805MB / 5.9. Evolutionary HPs: NSGA-2,
population 16, 10% mutation, 2 crossover points, tournament selection, elitism 2.

A useful negative result for anyone planning a cheap proxy search — STAR's Finding 1:
> Synthesizing backbones at full width and depth yields consistent improvements, while reduced-
> width synthesis achieves similar results with fewer successful candidates. **Motif synthesis
> underperforms both approaches.**

i.e. searching a small repeating block and stacking it is the *worst* option.

**Where "10 conv + 6 attention came from STAR" is actually claimed: the launch blog only.**
https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
> We built **STAR**, our neural architecture search engine, to find an optimal neural
> architecture given quality, memory, and latency criteria for deployment. … **The final
> architecture found by STAR is LFM2**, a Liquid model with multiplicative gates and short
> convolutions… There are 16 blocks in total, of which 10 are double-gated short-range LIV
> convolutions…

The blog also gives pseudocode that independently corroborates the `(B, C, x)` order and the
absence of an activation:
```python
def lfm2_conv(x):
  B, C, x = linear(x)  # input projection
  x = B*x              # gating (gate depends on input)
  x = conv(x)          # short conv
  x = C*x              # gating
  x = linear(x)
  return x
```
And it attributes the design to **hardware, not quality**: *"the structure and reliance of
LFM2 on short convolutions instead of full recurrences or attention layers originates from the
target device class, the embedded SoC CPU, as well as the underlying kernel libraries being
optimized for these types of workloads and operations."*

**STAR itself contains ZERO occurrences of "LFM" or "Liquid Foundation"** and predates LFM2 by
~7 months; it never mentions 10:6, 16-layer backbones, or on-device deployment.

**Net: the blog says "STAR found LFM2"; the technical report says STAR's objectives don't
transfer and that the objective mattered more than the search space or heuristic. Treat the
blog's attribution as marketing, and cite the report for provenance.** The 10:6 ratio has
**no published quantitative justification in either source.**

### 8.2 The actual LFM2 objective `[PAPER]`

§2.1 "Objectives and constraints" — a three-axis Pareto optimization:

> 1. **Quality**: performance on an internal suite of **50+ evaluations** (spanning knowledge
>    recall, multi-hop reasoning, instruction following, multilingual robustness, tool use,
>    math, and long context performance) after training each candidate architecture on a
>    reference dataset.
> 2. **Latency**: time-to-first-token (TTFT) and p50/p95 decode latency (ms/token) at
>    batch=1, as well as prefill throughput (tokens/s) on representative prompts.
> 3. **Peak memory**: measured as maximum resident set size (RSS) during prefill and decode
>    at target context windows (4K and 32K).
>
> Candidate architectures that violate device-side budgets on TTFT, decode latency, or peak
> memory are discarded. The remaining candidates are ranked by **hypervolume improvement**
> on the quality–latency–memory Pareto frontier.

**The objective is device-metric-anchored, not perplexity-anchored.** This is a significant
constraint on transferability: LFM2's conclusions are conditioned on batch=1 CPU/mobile
inference under specific quantization schemes. The paper's own Limitations section (§9) says:

> The deployment recipes and architecture search considered here are tuned for batch size of
> 1, low-latency inference on a small set of CPU and mobile SoC configurations
> (Snapdragon-class phones and Ryzen-class laptops) using specific quantization schemes
> (8da4w in ExecuTorch and Q4_0 in llama.cpp). LFM2 also runs competitively on modern NPUs
> and GPUs, but **these accelerators were not central to the hardware-in-the-loop search,
> and we do not claim the resulting architectures or quantization schemes are optimal for
> large-batch server settings or any particular accelerator family.**

### 8.3 Search space `[PAPER]`

§2.1 "Search space" — decoder-only stacks from these block families:
- **Local context and subquadratic blocks**: "gated short convolution blocks with **varying
  kernel sizes**, sliding-window attention (Child et al., 2019), and a family of
  sub-quadratic sequence blocks including linear attention variants (Katharopoulos et al.,
  2020; Yang et al., 2024a, 2025b); state-space variants such as S4, Liquid-S4, S5, RTF,
  Mamba, and Mamba2; Liquid-Time Constant networks such as CfC, as well as internal variants
  of efficient sequence blocks. … **The search space includes variants that keep only the
  short convolution submodule (i.e., the gated short convolution block) as well as variants
  that retain the full hybrid operator. This allows the search to attribute performance
  gains to specific computational units within the overall operator.**"
- **Global context blocks**: "grouped-query attention (GQA) with varying group counts and
  head dimensions, augmented with QK-Norm."
- **Position-wise blocks**: "SwiGLU feed-forward blocks with expansion ratios chosen by search."
- **Layout**: "interleaving patterns of local context blocks, global context blocks,
  position-wise blocks, and overall block counts under fixed parameter budgets, including
  options for shared weights and cache reuse."
- **MoE options**: "per-layer sparse FFNs with varying width and expert granularity."

Note "**under fixed parameter budgets**" — the layout search *was* parameter-matched, which
is consistent with my §6.4 point that a naive ratio sweep at fixed d is not.

Also note **kernel size was searched** and the answer was 3 for every released model. `[PAPER]`+`[CONFIG]`

### 8.4 Profiling setup `[PAPER]`

> Every candidate is compiled to the deployment stacks with identical settings (batch=1,
> fixed context windows at 4K/32K, and matched quantization/backends) and benchmarked on
> target devices:
> - **CPU path**: ExecuTorch (8da4w) and llama.cpp (Q4_0) on Samsung Galaxy S24 Ultra
>   (Qualcomm Snapdragon 8 Gen 3 SoC) and AMD Ryzen HX 370.
> - **Accelerator path**: vLLM for single-request and online batching (used for sanity
>   checks; the primary target remains on-device CPU deployment).
>
> We record TTFT, prefill throughput (tokens/s), decode ms/token (p50/p95), and peak memory
> with identical prompts and tokenizer settings.

(Note: the search used a Galaxy **S24** Ultra / Snapdragon 8 Gen 3; the reported inference
results in §2.4 use a Galaxy **S25** / Snapdragon 8 Elite.) `[PAPER]`

### 8.5 Outcomes — this IS the ablation, but only qualitatively `[PAPER]`

§2.1 "Outcomes", verbatim:

> Across size targets, the hardware-in-the-loop search **repeatedly selects a minimal hybrid
> architecture where most blocks are inexpensive gated short convolution blocks, interleaved
> with a small minority of GQA blocks**. Under identical on-device performance budgets,
> **augmenting these stacks (as in recent hybrid variants) with linear-attention,
> state-space, or additional convolution operators does not improve aggregate quality on the
> evaluation suite and typically worsens device metrics.** Empirically, the selected hybrids:
> - match or exceed the aggregate quality of attention-heavier and mixed
>   (conv+linear/SSM/conv) baselines at the same budget;
> - reduce decode latency (p50/p95) and increase prefill throughput at batch=1 under
>   identical tokenizer, prompt, quantization, and backend settings;
> - lower peak RSS at long context (4K/32K), consistent with reduced KV-cache versus
>   attention-heavy layouts.
>
> These results suggest that, **in the on-device regime, most of the benefits attributed to
> recent hybrid SSM/linear-attention blocks can be captured by their short convolutional
> submodules plus a small number of global attention layers.** We therefore carry forward
> designs that minimize global blocks while prioritizing inexpensive, gated short
> convolution blocks elsewhere.

And §2.2, on the framing of these results as an ablation:

> This operator is closely related to the short-range components that appear inside many
> recent efficient sequence blocks (Fu et al., 2023; Poli et al., 2023; Gu and Dao, 2024;
> Dao and Gu, 2024; Yang et al., 2025b). **The search results from Section 2.1 can be viewed
> as an ablation of these hybrids in the on-device setting. Once a handful of GQA blocks are
> available to handle long-range retrieval, the inexpensive gated short convolution alone is
> sufficient** to reach the best quality–latency–memory trade-off we observe, without
> additional linear attention/SSM/long convolution branches.

### 8.6 Reported inference performance `[PAPER]`

§2.4, for context on what the architecture actually buys. Setup: "Snapdragon 8 Elite based
Samsung Galaxy S25 smartphone (a newer model than we used for the architecture search in
Section 2.1) and an AMD Ryzen AI 9 HX 370 laptop CPU… batch size equal to 1. All models are
run with llama.cpp using the **Q4_0** quantization format… prefill throughput (prompt tokens
processed per second) for 1K and 4K-token prompts and decode throughput (tokens generated
per second) when producing 100 continuation tokens from 1K and 4K-token prefixes."

Table 2 (Galaxy S25, Snapdragon 8 Elite, batch 1, llama.cpp Q4_0), selected rows —
prefill tok/s @1K / @4K, decode tok/s @1K / @4K:

| Model | prefill 1K | prefill 4K | decode 1K | decode 4K |
|---|---|---|---|---|
| LFM2-350M | 1,067 | 657 | 194.1 | 143.8 |
| LFM2-700M | 522 | 341 | 104.2 | 80.2 |
| LFM2-1.2B | 335 | 222 | 69.8 | 55.5 |
| LFM2-2.6B | 143 | 116 | 33.8 | 30.0 |
| LFM2-8B-A1B | 85 | 76 | 48.6 | 41.9 |
| Llama-3.2-1B | 229 | 130 | 54.6 | 37.8 |
| Qwen3-1.7B | 140 | 98 | 39.7 | 26.9 |
| Gemma-3-1B | 377 | 295 | 67.5 | 67.1 |
| Granite-4.0-H-1B | 186 | 159 | 46.1 | 44.1 |

Note the honest caveat in the paper's own prose: "**Gemma-3-1B is the strongest 1B-class
baseline in terms of raw throughput; however, LFM2-1.2B attains 0.75−0.9× of its prefill
throughput**, while offering slightly higher 1K decode throughput and remaining within 20%
on 4K decode." So LFM2's speed advantage is **not** universal — a sliding-window-attention
baseline (Gemma-3) beats it on prefill at the 1B scale. `[PAPER]`

**All reported speed numbers are Q4_0-quantized, batch-1, CPU.** There are no bf16 or GPU
or batched throughput numbers anywhere in the report, and no wall-clock training-throughput
numbers. An academic experiment measuring bf16 GPU training/eval throughput cannot compare
against any published LFM2 figure. `[INFER]`

---

## 9. Published LFM2 ablations — what exists and what does NOT

I grepped the full extracted text of https://arxiv.org/html/2511.23404v1. Counts:
`ablation`/`Ablation` = **3 total hits**, `kernel` = 5, `RULER` = **0**, `needle` = **0**,
`LIV` = **0**.

**What the paper provides:**
- A **qualitative** conv-vs-attention / operator-family ablation, described in prose in
  §2.1 "Outcomes" and §2.2 (quoted in §8.5 above). It states the *conclusions* of the
  search — minimal hybrid wins; adding SSM/linear-attention branches doesn't help.
- Statement that kernel sizes were varied in the search space (§2.1) and that k=3 was
  selected for all five released models (Table 1).

**What the paper does NOT provide — verified absent:**
- ❌ **No quantitative conv-vs-attention ratio ablation table.** No numbers for e.g. 8/8 vs
  10/6 vs 12/4. The 10/6 choice is asserted as a search outcome with no per-configuration
  scores shown.
- ❌ **No quantitative kernel-size ablation.** k was searched, but no k=3 vs k=5 vs k=7
  results are reported.
- ❌ **No RULER, no needle-in-a-haystack, no NIAH.** Zero hits for both terms.
- ❌ **No MQAR / associative-recall / copying benchmark results.** The paper *cites* this
  literature (Arora et al. 2024a,b on multi-query associative recall; Jelassi et al. 2024
  on copying; Parnichkun et al. 2025 on effective state size) in §8.2 as *motivation* for
  including attention layers, but reports no such measurements on LFM2 itself.
- ❌ **No per-layer or attention-placement ablation.** The specific indices {2,5,8,10,12,14}
  and their non-uniform gaps are never justified or ablated.
- ❌ The only quantitative "ablation" mentioned by that word in the paper is about
  **LFM2-VL data annealing** ("Ablations showed that the annealing data schedule removes
  the need for connector pre-training"), which is unrelated to the mixer architecture.

The paper does report "long context performance" as one category in the 50+ internal eval
suite, and lists "Long context 6.7%" as a data-mixture share (3.2% for 8B-A1B), but the
internal suite is not disclosed and no long-context benchmark numbers are broken out for the
architecture comparison. `[PAPER]` Tables 1–16 in the report are hyperparameters (1),
throughput (2, 3), SFT mixture (4), alignment HPs (5), benchmark scores (6, 7), VLM (8, 9, 16),
audio (10–12), and ColBERT retrieval (13–15) — **there is no architecture-search results table
at all.** The search that produced 10:6 has zero reported numbers.

The report's Limitations section concedes the long-context weakness in prose: *"we observe that
LFM2 performs best on workloads that align with its design targets: **short to medium context
interactions**, task-oriented applications, and edge deployments with tight latency and memory
budgets."* `[PAPER]`

**`Liquid4All/LiquidRULER` exists but contains NO results.** https://github.com/Liquid4All/LiquidRULER
is harness plumbing only (`RULER/`, Dockerfile, run scripts) — results are generated locally
into `./benchmark_root` and not committed, and its examples target **`lfm-40b`** against
`inference-1.liquid.ai`, i.e. the LFM1 era, not LFM2. **No published RULER scores exist for any
LFM2 or LFM2.5 model.**

The LFM2.5 blog likewise has no ablations and no long-context table (its tables are
GPQA/MMLU-Pro/IFEval/IFBench/Multi-IF/AIME25/BFCLv3, Japanese, VLM, audio, inference speed).
Liquid's research index (https://www.liquid.ai/research) has adjacent architecture papers
(e.g. *Preconditioned DeltaNet*, *The Key to State Reduction in Linear Attention*) but **none
is an LFM2 conv:attention ratio ablation.**

**Coverage caveat on these negatives**: the "no published X" claims above are established for
*primary Liquid AI sources* (report, blogs, docs, GitHub org, HF cards). Third-party coverage
is weaker — web search was unavailable during this research pass, and arXiv's API does not
index full text (so a query for papers citing "LFM2" returning 0 is a null tool result, not
evidence). Whether an external paper (Granite-4.0, Falcon-H1, Qwen3-Next, Nemotron-H, etc.)
benchmarks LFM2 on RULER/recall was **not established**. Also, some blog throughput charts and
report figures are images that were not read.

**This is the single biggest opportunity for the proposed experiment**: the quantitative
conv/attention-ratio and recall story that LFM2 asserts qualitatively — and evaluates only
via an undisclosed internal suite under batch-1 CPU device constraints — has no public
quantitative grounding. A parameter-matched ratio sweep with public recall benchmarks
(MQAR, NIAH, RULER) would be a genuine contribution rather than a replication.

---

## 10. Reimplementation checklist (consolidated)

Everything below is `[CODE]`/`[CONFIG]`/`[CKPT]`-grounded:

**LIV / gated short conv block:**
1. `in_proj`: `Linear(d, 3d, bias=False)`
2. Transpose to `(bsz, 3d, L)`, then `chunk(3, dim=-2)` → **`(B, C, x)` in that order**
3. `h = B * x`
4. Depthwise causal conv: `Conv1d(d, d, kernel_size=k, groups=d, bias=False, padding=k-1)`,
   truncate to `[:, :, :seq_len]`
5. **No activation** after the conv
6. `y = C * h`; transpose back; `out_proj: Linear(d, d, bias=False)`
7. **No norm inside the block**
8. Conv state shape `(batch, d, k)`, statically allocated, k = `conv_L_cache` = 3

**Attention block:** `q_proj(d → H·64)`, `k_proj/v_proj(d → G·64)`, `out_proj(H·64 → d)`,
all bias-free; per-head `RMSNorm(64)` on Q and K **before** RoPE; RoPE θ=1e6; scaling
`64^-0.5`; full (non-windowed) causal attention; G=8 for all sizes.

**Layer:** `x = x + Mixer(operator_norm(x))`; `x = x + SwiGLU_MLP(ffn_norm(x))`. Both norms
RMSNorm(d), eps=1e-5. Every layer has an MLP regardless of mixer type.

**MLP:** `w2(silu(w1(x)) * w3(x))`, bias-free, effective width per the
`block_auto_adjust_ff_dim` transform (`int(2·cfg/3)` then round up to multiple of 256).

**Model:** `embed_tokens(V=65536, d)`; 16 layers with attention at {2,5,8,10,12,14};
`embedding_norm = RMSNorm(d)` applied **after** the last layer; **tied** lm_head (no
separate weight). No input norm on embeddings.

**Weight-name gotchas:** `out_proj` not `o_proj`; `feed_forward.w1/w2/w3` not
`mlp.gate/up/down_proj`; `operator_norm`/`ffn_norm` not `input_layernorm`/
`post_attention_layernorm`; `embedding_norm` not `norm`; conv path is
`layers.N.conv.conv.weight` (doubled).

---

## 11. Full comparison table

All values from `https://huggingface.co/LiquidAI/<repo>/blob/main/config.json`. "FF (eff.)"
is the post-`block_auto_adjust_ff_dim` width; "FF (cfg)" is the raw config value.

| Field | LFM2-350M | LFM2-700M | LFM2-1.2B | LFM2-2.6B | LFM2-8B-A1B | LFM2-24B-A2B | LFM2.5-1.2B-Base | LFM2.5-1.2B-Instruct |
|---|---|---|---|---|---|---|---|---|
| `architectures` | Lfm2ForCausalLM | Lfm2ForCausalLM | Lfm2ForCausalLM | Lfm2ForCausalLM | Lfm2MoeForCausalLM | Lfm2MoeForCausalLM | Lfm2ForCausalLM | Lfm2ForCausalLM |
| `model_type` | lfm2 | lfm2 | lfm2 | lfm2 | lfm2_moe | lfm2_moe | lfm2 | lfm2 |
| `hidden_size` | 1024 | 1536 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 |
| `num_hidden_layers` | 16 | 16 | 16 | 30 | 24 | 40 | 16 | 16 |
| FF (cfg) | 6656 | 10240 | 12288 | 10752 | 7168 | 11776 | 12288 | 12288 |
| **FF (eff.)** | **4608** | **6912** | **8192** | **10752** | 7168 (dense layers) | 11776 (dense layers) | **8192** | **8192** |
| `block_auto_adjust_ff_dim` | true | true | true | false | (absent) | (absent) | true | true |
| `num_attention_heads` | 16 | 24 | 32 | 32 | 32 | 32 | 32 | 32 |
| `num_key_value_heads` | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| head_dim (derived) | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| `conv_L_cache` (= kernel) | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| `conv_bias` | false | false | false | false | false | false | false | false |
| layer pattern key | `full_attn_idxs` | `full_attn_idxs` | `full_attn_idxs` | `layer_types` | `layer_types` | `layer_types` | `layer_types` | `layer_types` |
| attention indices | 2,5,8,10,12,14 | 2,5,8,10,12,14 | 2,5,8,10,12,14 | 2,5,9,13,17,21,24,27 | 2,6,10,14,18,21 | 2,6,10,…,38 (stride 4) | 2,5,8,10,12,14 | 2,5,8,10,12,14 |
| n_conv / n_attn | 10 / 6 | 10 / 6 | 10 / 6 | 22 / 8 | 18 / 6 | 30 / 10 | 10 / 6 | 10 / 6 |
| `rope_theta` | 1e6 | 1e6 | 1e6 | 1e6 | 1e6 | 1e6 (nested) | 1e6 | 1e6 |
| `max_position_embeddings` | 128000 | 128000 | 128000 | 128000 | 128000 | 128000 | 128000 | 128000 |
| paper context window | 32768 | 32768 | 32768 | 32768 | 32768 | (not in paper) | (no paper) | (no paper) |
| `norm_eps` | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| `vocab_size` | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 |
| tie embeddings | true (default) | true (default) | true (default) | `tie_embedding: true` | true (default) | true (default) | `tie_embedding: true` | `tie_embedding: true` |
| `bos`/`eos`/`pad` | 1/7/0 | 1/7/0 | 1/7/0 | 1/7/0 | 1/7/0 | 1/7/0 | 1/7/0 | 1/7/0 |
| MoE experts | — | — | — | — | 32 (top-4) | 64 (top-4) | — | — |
| `moe_intermediate_size` | — | — | — | — | 1792 | 1536 | — | — |
| `num_dense_layers` | — | — | — | — | 2 | 2 | — | — |
| **params (HF actual)** | **354,483,968** | **742,489,344** | **1,170,340,608** | **2,569,272,320** | **8,339,929,856** | (not fetched) | **1,170,340,608** | (not fetched) |

Notes:
- `tie_word_embeddings` never appears literally; it is `True` by default in `Lfm2Config` and
  some configs set the legacy alias `tie_embedding: true`. Absence of `lm_head.weight` in
  the checkpoints confirms tying for LFM2-1.2B, LFM2-2.6B, LFM2.5-1.2B-Base. `[CKPT]`
- Older configs carry redundant/legacy keys: `block_dim` (= `hidden_size`), `num_heads`
  (= `num_attention_heads`), `conv_dim`/`conv_dim_out` (= `hidden_size`), `theta`
  (= `rope_theta`), `use_pos_enc`, `block_norm_eps`, `initializer_range`,
  `block_use_swiglu`, `block_use_xavier_init`, `conv_use_xavier_init`,
  `block_mlp_init_scale`, `block_out_init_scale`. Most are **not read** by
  `Lfm2Config`/`modeling_lfm2.py` (the `@strict` dataclass ignores them) — in particular
  the Xavier-init and init-scale flags have **no effect** in the HF implementation, so
  Liquid AI's actual training-time init is not reproducible from these configs. `[CODE]`+`[CONFIG]`
- `LFM2-8B-A1B` param count includes 704 F32 params (router/expert-bias terms):
  `{'F32': 704, 'BF16': 8339929856}` from the HF API. `[CKPT]`

---

## 12. Open items / unverified

- There is **no LFM2.5 technical report** (confirmed: the LFM2.5 blog's "technical report" link
  points to the LFM2 report; arXiv title search for LFM2.5 → 0 results). The
  architectural-identity conclusion (§5) rests on config + checkpoint comparison plus the blog's
  "builds on the LFM2 device-optimized architecture" statement.
- The internal 50+ evaluation suite is not disclosed, so the search's quality axis cannot
  be reproduced.
- `Lfm2MoeConfig`/`modeling_lfm2_moe.py` were not read in detail; the MoE claims here come
  from configs and the paper's §2.3 prose. (The MoE *parameter formula* was nonetheless
  verified exactly against LFM2-8B-A1B's real total — §6.2.)
- I did not verify LFM2-350M/700M/8B-A1B/24B-A2B checkpoint tensor names directly (only
  1.2B, 2.6B, LFM2.5-1.2B-Base). Param-count formulas were verified against 350M/700M/8B-A1B via
  the HF API totals, which matched exactly.
- **STAR's discovered per-layer topologies could not be recovered** — Figures B.1–B.25 are
  vector graphics with no text layer, so STAR's own conv:attention counts remain unknown (§8.1b).
- **Third-party evaluation coverage is weak** (web search unavailable during this pass). See the
  coverage caveat at the end of §9.
- Some Liquid AI blog throughput charts and report figures are images and were not read.

---

## 13. Licensing and practical reuse

`[CKPT]`/HF-metadata findings established directly:
- `LiquidAI/LFM2-1.2B` HF metadata: `license: other`, `license_name: **lfm1.0**`,
  `license_link: LICENSE`, tags include `license:other`. A `LICENSE` file is present in the
  repo file list. Source: `https://huggingface.co/api/models/LiquidAI/LFM2-1.2B`
- `LiquidAI/LFM2.5-1.2B-Base` HF metadata: `license: other`, with `license_name` and
  `license_link` **absent** from `cardData`; tags include `license:other`, `lfm2`, `lfm2.5`.
  A `LICENSE` file is present. Source:
  `https://huggingface.co/api/models/LiquidAI/LFM2.5-1.2B-Base`

So the family ships under a **custom Liquid AI license ("lfm1.0"), not Apache-2.0 or MIT**.
The `transformers` *modeling code* is Apache-2.0 (HuggingFace copyright header in
`modular_lfm2.py`/`modeling_lfm2.py`) — code and weights are licensed separately. `[CODE]`

### 13.1 The license is Apache-2.0 plus a revenue-gated commercial clause

The license is **"LFM Open License v1.0"**, shipped as `LICENSE` (10,644 bytes) in each repo:
- https://huggingface.co/LiquidAI/LFM2-1.2B/raw/main/LICENSE
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/raw/main/LICENSE
- Canonical hosted copy + plain-language summary + FAQ + an official redline-vs-Apache-2.0
  diff: **https://www.liquid.ai/lfm-license**

Liquid's own characterization: *"The license text is based on Apache 2.0 (with only a few
changes)."* The four deltas they enumerate: grants in §2/§3 made "subject to" the commercial
limitation; a **Commercial Use Limitation (§5)** capping free commercial use at companies
under **$10M annual revenue**; "Licensor" defined as Liquid AI, Inc.; auto-termination on
non-compliance (§11).

§5 verbatim:
> **5. Commercial Use Limitation.**
> (a) The rights granted under this License for Commercial Use are conditioned upon You or
> Your Legal Entity not exceeding the Threshold.
> (b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the
> Threshold is not licensed under this Agreement.
> (c) **The Threshold shall not apply to a Qualified Non-Profit Organization's use of the Work
> or a Derivative Work for Non-Commercial or Research Purposes.**

with `"Threshold" shall mean annual revenue of 10 million United States dollars
($10,000,000) or more`, measured **group-wide** ("Legal Entity" includes parents/
subsidiaries/50%+ common control).

Answering the specific questions asked:

| Question | Answer |
|---|---|
| (a) Commercial use / revenue threshold | Allowed **below $10M annual revenue** (group-wide). Above that, not licensed. |
| (b) Derivative works / finetunes | **Allowed**, and may be kept proprietary. FAQ: *"No. The license does not have a 'copyleft' requirement. You can keep your fine-tuned models proprietary."* The license does follow your derivative. |
| (c) Academic ablation / publication | **Explicitly allowed, no threshold** (§5(c)). FAQ: *"Our license specifically exempts qualified non-profits. They are free to use the models for non-commercial or research purposes, with no thresholds on revenue."* **No publication-approval, no benchmarking-gag, and no evaluation-results clause anywhere.** |
| (d) Attribution / naming | Standard Apache §4: pass on the License, mark modified files, retain notices, reproduce `NOTICE` if present (**none ships**). **No naming requirement** — no "Built with LFM2" obligation, unlike Llama. §7 is the plain Apache trademark clause. |
| (e) Using outputs to train other models | **NO RESTRICTION.** A grep of the full license text for `output`, `distill`, `train`, `AI model`, `machine learning` returns **zero matches**. No distillation ban — a real difference from Llama/Gemma-style licenses. |
| (f) LFM2 vs LFM2.5 | **Identical.** The two LICENSE files are byte-identical after whitespace/header normalization (verified programmatically). |

Repos are **ungated** (`gated: false`) and there is **no `USE_POLICY.md`** (404) — `LICENSE` is
the only legal document in the repo. Metadata quirk: HF `license_name` says `lfm1.0` while the
file's own first line says `LFM Open License v1.0`.

**Verdict: clean and permissive for an academic ablation.** Ungated weights, Apache-derived,
explicit research carve-out, derivatives allowed, no output-training restriction, no
benchmarking restriction, same terms across LFM2 and LFM2.5. (Not legal advice; the LFM Open
License has no authoritative OSI/FSF interpretation, and Liquid's separate website ToS was
not reviewed.)

### 13.2 Official training code — post-training only, NO pretraining recipe

**Liquid AI publishes no from-scratch pretraining code.** Their official training repo is
**https://github.com/Liquid4All/leap-finetune** — *"a minimal fine-tuning repo for LFM2…
distributed orchestration, checkpointing, and export for local GPU nodes, SLURM clusters,
Modal, and Kubernetes/KubeRay."* Its complete `job_configs/` set covers SFT, SFT+LoRA, DPO,
GRPO, VLM SFT/DPO/GRPO, and MoE SFT/DPO — **no pretraining or CPT config**. (The repo also
ships no LICENSE file.)

A survey of all 47 repos at https://github.com/orgs/Liquid4All/repos found only inference
(`liquid_llama.cpp`, `lfm-inference`, `onnx-export`, `console-vllm`, LEAP SDKs), eval
harnesses (`LiquidRULER`, `mt_bench`, `VLMEvalKit_Liquid`, `bfcl-liquid-public`, `openbench`),
examples (`cookbook`), and post-training. **No pretraining repo.** Official docs
(https://docs.liquid.ai/lfm/fine-tuning/overview) list only *"SFT, LoRA, DPO, and GRPO"* with
no pretraining section. The closest available thing is **continued** pre-training via Unsloth
Colabs (https://docs.liquid.ai/lfm/fine-tuning/unsloth) — CPT, not from-scratch.

Ecosystem support (all finetuning, none pretraining):

| Framework | LFM2 status |
|---|---|
| HF `transformers` | First-class: `models/lfm2/` (needs ≥4.55). Verified working locally at 5.14.1. |
| TRL | Supported; official SFT + DPO Colabs; `trl-internal-testing/tiny-Lfm2ForCausalLM` fixtures. |
| Unsloth | Supported incl. CPT + GRPO; mapped in `unsloth/models/mapper.py`. |
| axolotl | Supported since 2025/07; `examples/LiquidAI/` has `lfm2-350m-fft.yaml`, `lfm2-8b-a1b-lora.yaml`, `lfm2-vl-lora.yaml` — all **SFT** (`type: chat_template`), not `pretraining_dataset`. |
| torchtitan | **Not supported** (`models/` has only deepseek_v3, flux, gpt_oss, kimi_k2, llama3, qwen3*). |
| litgpt | **Not supported** (0 hits for `lfm2` in `config.py`). |

**No published from-scratch LFM2 pretrain exists.** An enumeration of 500 HF models under
`filter=lfm2` (446 non-LiquidAI) found only quantizations/format conversions
(`mlx-community`, `lmstudio-community`, `unsloth`, `onnx-community`, `mradermacher` GGUF),
finetunes with a declared `base_model`, and tiny random test fixtures. *(Caveats: GitHub code
search needed auth, so no exhaustive cross-repo grep of nanotron/Megatron/levanter/OLMo-core;
the HF enumeration was capped at 500.)*

### 13.3 Practical conclusion for this experiment

**For a from-scratch "mostly-LIV" ablation, the weight license is essentially moot, and the
missing pretraining code is the real (but modest) cost.** The architecture is fully specified
in an **Apache-2.0** `transformers` implementation, so you can build randomly-initialized
models of any layer pattern without touching Liquid AI weights or accepting their license at
all. I verified this end to end locally (`transformers` 5.14.1):

```python
cfg = Lfm2Config(**json.load(open("LFM2-1.2B-config.json")))
m   = Lfm2ForCausalLM(cfg)
sum(p.numel() for p in m.parameters())   # -> 1,170,340,608  (exact match to the release)
```

and the ratio sweep is genuinely a one-line config change (`full_attn_idxs=[...]` or
`layer_types=[...]`). `Lfm2ForCausalLM` implements `loss_function(logits, labels, vocab_size)`
and returns a loss; `Lfm2DecoderLayer` subclasses `GradientCheckpointingLayer`; the class is a
`LlamaForCausalLM` subclass, so HF `Trainer` works. The license only binds if you initialize
from, distribute, or finetune the **released checkpoints**. `[CODE]`+`[INFER]`
