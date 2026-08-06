# DeepSeek architecture swap in OLMo-core

## Why this branch exists

This is a focused experiment: swap **one** component of OLMo for its DeepSeek-architecture
equivalent. The point is not breadth. Over roughly six hours of concentrated work I want to gain
deep expertise in one small area of the model — deep enough that I become the person others on the
team can ask about it — and to understand the wider codebase better as a side effect of going all
the way down in a single place.

Everything below is a survey and a plan for choosing that one place. It lives on
[`edullm/adarsh-deepseek-swap`](https://github.com/edu-llm/OLMo-core/tree/edullm/adarsh-deepseek-swap).

## What already exists

This repo is [`edu-llm/OLMo-core`](https://github.com/edu-llm/OLMo-core), a hard fork of
`allenai/OLMo-core` that does not sync upstream. The DeepSeek-adjacent surface is already
substantial, and knowing what is present keeps me from re-deriving it:

- **DeepSeekMoE is largely already there.** `src/olmo_core/nn/moe/` implements fine-grained experts
  (a per-expert `hidden_size`), a `shared_mlp` shared expert, `MoERouterGatingFunction.sigmoid`
  gating, and `bias_gamma` — the auxiliary-loss-free load-balancing bias from DeepSeek-v3.
- **An MoE study is already running.**
  [`edullm/moe-m1-pilot`](https://github.com/edu-llm/OLMo-core/tree/edullm/moe-m1-pilot) adds the
  eduLLM MoE study's M1 arm together with its forward-path-matched control, router metrics, and
  dropped-token-vs-capacity reporting.
- **The "maple" line is large.**
  [`edullm/maple-infra`](https://github.com/edu-llm/OLMo-core/tree/edullm/maple-infra) plus the
  `agent/L1`–`agent/L7` stack all share `.edullm/maple-gates/gates.py`.
- **KDA is claimed.**
  [`agent/claude-01/dp2-kda-phase-0-prep`](https://github.com/edu-llm/OLMo-core/tree/agent/claude-01/dp2-kda-phase-0-prep)
  already owns the KDA work.
- **MLA is absent.** `src/olmo_core/nn/attention/` exports only `Attention`, `FusedAttention`,
  `NormalizedAttention`, and `GatedDeltaNet`. There is no Multi-head Latent Attention anywhere in
  the repo.

## Where a swap plugs in

Both DeepSeek pillars are registrable config points, so a new component is additive rather than a
rewrite:

- **Sequence mixer slot.** Mixers register additively via `@SequenceMixerConfig.register("...")` —
  for example `AttentionConfig` is registered as `"attention"`. A new mixer is a new registration
  next to the existing ones.
- **FFN / MoE slot.** The feed-forward block is the other swap point, where the dense `FeedForward`
  and the MoE variants live.

## Candidate areas

Three places a DeepSeek swap could land, each with the one-line reason it is or is not already
crowded:

1. **DeepSeek MLA as a new sequence mixer** — unclaimed, self-contained, and carries a crisp
   KV-cache-bytes-per-token story against the existing attention path.
2. **MoE router (`nn/moe/router.py`)** — overlaps the active
   [`edullm/moe-m1-pilot`](https://github.com/edu-llm/OLMo-core/tree/edullm/moe-m1-pilot) study, so
   it trades novelty for joining live work.
3. **MoE FFN block swap** — a param / FLOP accounting focus (dense to fine-grained plus shared
   experts), with most of the machinery already present.

## Decision deadline

I commit to one area by **~hour 2**. The evidence that settles it is reading, not writing:

- `src/olmo_core/nn/moe/router.py`
- `src/olmo_core/nn/moe/moe.py`
- `src/olmo_core/nn/attention/base.py`
- the [`edullm/moe-m1-pilot`](https://github.com/edu-llm/OLMo-core/tree/edullm/moe-m1-pilot) diff

## Initial focus: DeepSeek MLA

The selected area is **DeepSeek MLA (Multi-head Latent Attention) as a new sequence mixer.** It is
unclaimed, it is a single self-contained module, and its payoff is one legible number — KV-cache
bytes per token versus the existing attention path at matched parameter count.

The other two candidates — the **MoE router** and the **MoE FFN block swap** — are kept as
documented fallbacks in case the hour-2 reading changes the picture.

## Decision confirmed (hour-2 read)

Reading `nn/moe/router.py`, `nn/moe/moe.py`, and `nn/attention/base.py` confirmed the survey: the
MoE surface is already dense — `MoERouterConfig` with sigmoid gating and the DeepSeek-v3
auxiliary-loss-free `bias_gamma` centroid update, `MoELoadBalancingLossGranularity`, router z-loss,
capacity and dropped-token accounting — and it overlaps the live `edullm/moe-m1-pilot` study. The
attention package (`Attention`, `FusedAttention`, `NormalizedAttention`, `GatedDeltaNet`) has **no**
latent attention. So MLA is the uncrowded, self-contained choice with a single legible payoff. The
`SequenceMixer` / `SequenceMixerConfig` base is small and clean, and `GatedDeltaNet` is a working
example of a non-standard mixer that still satisfies the interface, so a new mixer is genuinely
additive.

## What I built

A new sequence mixer, `MultiheadLatentAttention` (`olmo_core.nn.attention.mla`), registered as
`"mla"` via `MLAConfig`. It follows DeepSeek-V2/V3 Multi-head Latent Attention.

**How MLA works.**

1. **Joint KV compression.** A single down-projection `W_DKV` maps each token from `d_model` to a
   small latent `c_kv` of width `kv_lora_rank`, *plus* a small shared decoupled-RoPE key of width
   `qk_rope_head_dim`. An up-projection `W_UKV` reconstructs the per-head non-RoPE keys
   (`qk_nope_head_dim`) and the values (`v_head_dim`) from `c_kv`. At inference time only `c_kv`
   and the decoupled key need caching — that is the whole point.
2. **Optional query compression.** If `q_lora_rank` is set, queries go through their own
   down/norm/up (`W_DQ` → `RMSNorm` → `W_UQ`); otherwise `W_Q` projects them directly.
3. **Decoupled RoPE.** RoPE can't be applied to the compressed latent (the up-projection would mix
   positions across the rank), so position lives on a separate sub-vector. Each query/key is
   `concat(nope, rope)` along the head dim: the `qk_nope_head_dim` part carries no position, and the
   `qk_rope_head_dim` part is rotated. The decoupled key is a *single* head, broadcast across all
   query heads, exactly as in DeepSeek. Reconstruction gives full multi-head keys/values, so
   `n_kv_heads == n_heads` — the savings are in the *cache*, not in the head count.

**Key design decisions.**

- **Reuse the existing SDPA backend.** MLA builds the shared `AttentionBackend` (defaulting to the
  `torch` backend, which works everywhere) with `n_kv_heads == n_heads`, and calls it just like
  `Attention.sdpa`. This inherits causal masking and intra-document masking (`cu_doc_lens` /
  `max_doc_len`) for free. The value head width may differ from the query/key head width, which
  SDPA handles. The softmax scale defaults to `1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`.
- **Reuse RoPE and RMSNorm.** The decoupled RoPE uses the repo's `RotaryEmbedding` built for the
  small `qk_rope_head_dim`; the latents get a bias-free `RMSNorm` (configurable, matching DeepSeek).
- **Scope.** Tensor/context parallelism raise `NotImplementedError` (as `GatedDeltaNet` does for
  unsupported parallelism), and an optimized inference path that caches `c_kv` is future work:
  `forward` reconstructs full per-head K/V each call and rejects `cache_leftpad`. The training
  forward path (shapes, gradients, masking kwargs) is complete and tested on CPU.

**How to configure it.** `MLAConfig` slots in wherever a `SequenceMixerConfig` is expected (e.g. a
`TransformerBlock`'s `sequence_mixer`), the same way `AttentionConfig` / `GatedDeltaNetConfig` do:

```python
from olmo_core.nn.attention import MLAConfig

mixer = MLAConfig(
    n_heads=16,
    kv_lora_rank=512,       # width of the cached joint KV latent
    q_lora_rank=None,       # or e.g. 1536 to also low-rank the queries
    qk_nope_head_dim=128,   # non-RoPE per-head width (from the latent)
    qk_rope_head_dim=64,    # decoupled RoPE per-head width (bypasses the latent)
    v_head_dim=128,         # value per-head width
    # norm=LayerNormConfig(name="rms", bias=False)  # default; set None to disable
    # rope=RoPEConfig()                              # default decoupled RoPE
    # bias=False, dropout=0.0, softmax_scale=None, backend=None, dtype="float32"
)
```

The `num_params(d_model)` estimate (asserted equal to the built module's parameter count in the
unit test) is, with `dk = qk_nope_head_dim + qk_rope_head_dim`, `H = n_heads`, and `N(size)` the
norm's parameter count (a bias-free `RMSNorm` contributes `size`):

```
# queries
  q_lora_rank is None:  d_model * H * dk                        (+ H*dk           if bias)
  else:                 d_model * q_lora_rank + q_lora_rank*H*dk (+ q_lora_rank + H*dk if bias)
                        + N(q_lora_rank)
# keys / values
  W_DKV:  d_model * (kv_lora_rank + qk_rope_head_dim)           (+ (kv_lora_rank+qk_rope_head_dim) if bias)
  norm:   N(kv_lora_rank)
  W_UKV:  kv_lora_rank * H * (qk_nope_head_dim + v_head_dim)    (+ H*(qk_nope_head_dim+v_head_dim) if bias)
# output
  W_out:  H * v_head_dim * d_model                              (+ d_model         if bias)
```

## The payoff: KV-cache bytes per token

This is the legible number the experiment was chosen for. Per layer, per token, the number of
scalars that must be cached to decode the next token:

- **MLA:** `kv_lora_rank + qk_rope_head_dim` (cache the latent `c_kv` and the one decoupled key).
- **Standard MHA/GQA/MQA:** `2 * n_kv_heads * head_dim` (cache K and V for every KV head).

At matched attention geometry (`n_heads = 16`, `head_dim = 128`, and the MLA defaults
`kv_lora_rank = 512`, `qk_rope_head_dim = 64`), at bf16 (2 bytes/scalar):

| Variant            | `n_kv_heads` | scalars / token / layer            | bf16 bytes / token / layer | vs MLA |
| ------------------ | ------------ | ---------------------------------- | -------------------------- | ------ |
| MHA                | 16           | `2 * 16 * 128 = 4096`              | 8192 (8.0 KiB)             | 7.1×   |
| GQA (groups of 8)  | 2            | `2 * 2 * 128 = 512`               | 1024 (1.0 KiB)             | 0.9×   |
| MQA                | 1            | `2 * 1 * 128 = 256`              | 512 (0.5 KiB)              | 0.44×  |
| **MLA**            | 16 (at compute) | `512 + 64 = 576`               | 1152 (≈1.13 KiB)           | 1.0×   |

The story in one line: **MLA caches about as little as GQA/MQA (576 scalars, ~7× smaller than the
4096 of full MHA) while still computing full 16-head attention.** GQA/MQA buy their small cache by
literally sharing K/V across query heads; MLA instead keeps every head at compute time and pays only
for a low-rank latent in the cache. (This module does not yet *implement* the latent-cache decode
path — see scope above — but the parameterization is exactly what makes that cache small, which is
what the table quantifies.)

## Tests and checks

`src/test/nn/attention/mla_test.py` (CPU-only, no `@pytest.mark.gpu`) builds small configs and
asserts: output shape equals input shape, the output is differentiable (backward reaches the
input), and `config.num_params(d_model)` equals `sum(p.numel() for p in module.parameters())` across
config variants (direct vs low-rank queries, bias on/off, norm on/off, default/complex/no RoPE, and
uneven head dims). `isort`, `black`, `ruff`, and `mypy` pass on the new files.
