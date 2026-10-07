# P4 Phase 2 — Architecture Innovation Ideas for a Small Education LLM

**Status:** Draft v1 (idea bank) · **Owner:** Architecture team · **Date:** 2026-07-20
**Base model:** OLMo (fully open — weights, data, code, checkpoints — so we can change internals). Two flavors matter: **OLMo 2** (dense 1B/7B/13B/32B) and **OLMoE** (sparse MoE, **1B active / 7B total**, already runs on-device via 4-bit quant).
**Goal of this doc:** a concise, scannable menu of where LLMs hurt in **Inputs / Core / Memory**, the current best fixes, and how each could make our model **faster, better-per-parameter, or more education-specific**. Tags: `[faster]` `[better]` `[edu]` `[multilingual]`. "100x" is aspirational — the realistic win is *stacking* a few multiplicative gains (see §6).

---

## 1. Inputs — tokenization, embeddings, position

- **Go tokenizer-free (byte-level) — Byte Latent Transformer.** `[faster][multilingual][edu]`
  BPE tokenizers bake in problems: they mangle numbers, break on typos, and charge non-English languages 2–4× more tokens. [BLT](https://arxiv.org/abs/2412.09871) drops the tokenizer, learns from raw bytes, and groups them into entropy-based *patches* — matching Llama-3 quality at up to **50% less inference compute**, with **big gains on low-resource languages** and noisy input. Perfect fit for us: kids misspell, math is digit-level, and per-country needs script fairness. *Cost:* a real architecture change; heavier to train.
- **Fix number tokenization even if we keep BPE.** `[edu]` Tokenize digits individually — a cheap, well-known win for arithmetic, which is core to grade-4 math. Low effort, high payoff.
- **Multilingual-fair vocab.** `[multilingual]` If not going byte-level, train the tokenizer on balanced multilingual + local-curriculum data so target countries aren't penalized. Ties directly to §5.
- **Small-model embedding economy.** `[faster]` Tie input/output embeddings and/or factorize them; on a 1B model the embedding table is a large share of params — freeing it buys more depth.

## 2. Core Architecture — attention, MLPs, blocks

- **Mixture-of-Experts (start from OLMoE).** `[faster][better]`
  Sparse experts give **7B worth of knowledge at ~1B active compute**, and [OLMoE](https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514) trained **2× faster** than a dense equal while beating larger models on MMLU. Ideal for cramming many subjects/grades into a small, cheap-to-run model. It's already open and on-device — our strongest starting point.
- **Hybrid SSM + attention (Jamba / Samba / Hymba).** `[faster]`
  Interleave Mamba (linear-time) with a little attention (precise recall). [Jamba-1.5](https://arxiv.org/pdf/2408.12570) gets **~10× smaller KV cache** and 256K effective context; [Hymba](https://arxiv.org/abs/2403.19887) is SOTA **sub-2B**, beating Llama-3.2-3B with less cache and higher throughput. Great if long textbooks / multi-turn tutoring sessions matter. *Caveat:* for <128K contexts, plain Transformers are still competitive — treat as an efficiency bet, not a quality one.
- **Streaming attention (sliding window + attention sinks).** `[faster]` Cheap, unbounded tutoring dialogue without KV blowup; simple to bolt on.
- **Looped / recursive layers (parameter sharing).** `[better]` (speculative) Reuse blocks to add "reasoning depth" without adding params — promising for squeezing step-by-step reasoning out of a tiny model.
- **Built-in tool/verifier hooks (calculator, code exec).** `[edu]` A teacher must not get arithmetic wrong. An architectural interface that routes computation to a verifier makes math answers *checkable*, not hallucinated.

## 3. Memory — long context, memory layers, retrieval

- **Memory layers / product-key memory.** `[better][edu]`
  [Meta's Memory Layers at Scale](https://arxiv.org/abs/2412.09764) adds a trainable key-value lookup that grows *knowledge* capacity **without adding FLOPs** — beating dense models with **2× the compute** on factual tasks, and (key for us) **learning facts faster with less training data**. A small model that is factually reliable is exactly what teaching needs. Open code exists.
- **Curriculum RAG grounding.** `[edu][multilingual]`
  Retrieve from trusted textbooks/curriculum and condition on them. Proven at scale: [SLM+RAG tutors](https://arxiv.org/html/2506.05925) match big models while staying private/local, and grounding **cuts hallucination** (though doesn't eliminate it). Keeps the model small; for per-country, just swap in local curriculum + a **verifier layer** (RAG+CAG+verify).
- **Compressive / near-infinite context (Infini-attention).** `[faster]` [Infini-attention](https://arxiv.org/abs/2404.07143) holds a whole book or full tutoring history in *bounded* memory (~235× less at 1M tokens). *Caveat:* lossy — fine for "remember the session," not exact recall.
- **On-device via KV-cache compression + quantization.** `[faster][edu]` OLMoE already runs on a phone at 4-bit with minor quality loss — enables **offline, private, low-cost** deployment in schools with poor connectivity (an access story, not just a speed story).
- **Editable / sparse-memory finetuning.** `[edu]` Update curriculum facts (new standards, corrected content) *without* full retraining or catastrophic forgetting — Meta's 2025 sparse-memory finetuning line.

## 4. Education specialization (cross-cutting niche)

The bet: we won't beat frontier models everywhere, but an **in-domain edu model can win on trust, cost, and fit.**
- **Grounded + verified answers** (RAG + verifier) so we never confidently teach something false.
- **Calibration / "I don't know."** `[edu]` A good teacher abstains rather than bluffs; add uncertainty/abstention — a genuine differentiator vs. confident frontier models.
- **Pedagogical control.** Condition on grade level and scaffolding style (control tokens / steering vectors) so the same model can teach up or down a level.
- **Transparent reasoning** (worked-example-style steps) aligned to how the material is taught.

## 5. Per-country / cultural LLM (a real differentiator)

The sharpest research finding: **"language ≠ culture."** [Fluent but Foreign (2025)](https://arxiv.org/abs/2505.21548) shows even regional LLMs often *speak* the local language while encoding **Western values**, and most models drift toward a Western/Protestant-European default. So the niche isn't translation — it's **genuine cultural + values alignment** for what children are taught.
- **Modular cultural adapters / experts.** `[multilingual]` Shared base + swappable per-country module (LoRA adapter or a dedicated MoE expert) carrying that country's language, curriculum, and values. Adding a country becomes cheap. (Cf. "Cultural MoErges" attention-gated blending.)
- **ValuesRAG.** Retrieve region-specific cultural/curriculum profiles as context — reported **up to +21%** cultural-alignment gains over zero/few-shot.
- **Byte-level tokenizer** (from §1) for fair treatment of local scripts.
- **Community-authored local corpora + cultural eval as first-class** — a data/eval commitment, but it *implies* the modular architecture above so each country plugs in cleanly.

## 6. How these stack toward "100x"

No single trick is 100×; the pitch is a **multiplicative stack**:
- **Cheaper/faster:** MoE sparsity (OLMoE) × 4-bit quant × hybrid-SSM long-context × speculative decoding → large inference multiples on real workloads.
- **Better-per-param:** memory layers (cheap factual capacity) + curriculum RAG (grounding) → frontier-level *in-domain* accuracy at ~1B active.
- **The story:** *a small, on-device, culturally-adaptable tutor that is more accurate and trustworthy than a general frontier model within education — at a fraction of the cost.*

## 7. Recommended first bets (each independently testable on OLMo)

1. **Base = OLMoE** (open MoE, on-device proven) — capacity/efficiency for free.
2. **Add memory layers** — cheap factual reliability, open code, learns facts on less data.
3. **Curriculum RAG + verifier** — grounded, low-hallucination teaching.
4. **Byte-level or digit-level tokenization** — math correctness + multilingual fairness.
5. **Modular cultural adapters** — the per-country differentiator.

Numbers 1–3 are the highest-confidence quality/efficiency wins; 4–5 are the education/market differentiators. Suggest prototyping 1+2 first (both have open implementations and directly serve "small but knowledgeable").

---

### Sources
- Tokenization / byte-level: [Byte Latent Transformer (Meta, 2024)](https://arxiv.org/abs/2412.09871)
- MoE base: [OLMoE (Ai2)](https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514) · [OLMo 2 (Ai2)](https://allenai.org/blog/olmo2)
- Hybrid SSM/attention: [Jamba](https://arxiv.org/abs/2403.19887) · [Jamba-1.5](https://arxiv.org/pdf/2408.12570) · [Hymba / Samba / Zamba family overview](https://arxiv.org/pdf/2503.18970)
- Memory layers: [Memory Layers at Scale (Meta, 2024)](https://arxiv.org/abs/2412.09764) · [code](https://github.com/facebookresearch/memory)
- Long context: [Infini-attention (Google, 2024)](https://arxiv.org/abs/2404.07143)
- Education SLM+RAG: [Small Models, Big Support (2025)](https://arxiv.org/html/2506.05925) · [HalluGuard (2025)](https://arxiv.org/pdf/2510.00880)
- Cultural alignment: [Fluent but Foreign (2025)](https://arxiv.org/abs/2505.21548) · [Assessing Sovereign LLMs (2025)](https://arxiv.org/html/2510.14565)

*Idea bank for scoping only — not a commitment. Verify claims/benchmarks before building.*
