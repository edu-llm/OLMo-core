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
