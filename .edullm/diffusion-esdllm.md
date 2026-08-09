# Masked diffusion at 370M: bidirectional GDN-2 / attention hybrid, MoE, MuonH

One arm: `olmo2_370M_moe` with a 3:1 hybrid of bidirectional GDN-2 and non-causal attention,
trained on a masked-diffusion objective with MuonH over 10B tokens of `regmix-10b`, against the
autoregressive GDN-2 baseline that read the same corpus with the same budget.

ES-dLLM is stage two and is not in this branch. See [What ES-dLLM actually is](#what-es-dllm-actually-is).

## What is being tested

Whether a diffusion language model at this scale, on this corpus, reaches a loss and a downstream
score worth the compute — and what its inference throughput is once early-skipping is added on top.
Diffusion's one attested advantage is data efficiency, not compute efficiency:
[Quokka](https://arxiv.org/abs/2510.03280) puts a compute-optimal 400M diffusion model at 39.3B
tokens against Chinchilla's 8.0B, so at a fixed 10B budget this arm is deliberately under-trained
relative to its own optimum. That is the correct choice for a fixed budget — it makes this a
data-efficiency question — and it means a loss slightly behind the baseline is not automatically a
failure.

## What would invalidate the comparison

1. **The arm differs from the baseline in three ways at once**: MoE against dense, MuonH against the
   baseline's optimizer, diffusion against autoregression. Active parameters are matched and the
   whole budget is held equal, but no single run can attribute a difference among the three.
   Isolating them needs further arms. Read a result as "this configuration against that one".

2. **`--moe-top-k` is a science parameter, not a throughput one.** `olmo2_370M_moe`'s own `top_k=4`
   gives this arm 299.6M active non-embedding parameters against the baseline's 409.2M — 73%, so a
   loss gap would be mostly the size difference. `top_k=8` gives 400.3M, a ratio of **0.978**. Total
   parameters are 1107.0M either way, so MuonH's per-expert blocking is unaffected. Measured, not
   estimated.

3. **`--rank-microbatch-size 8192` must not move.** For a MoE, `expert_capacity` derives from
   `max_local_microbatch_size`, so changing the microbatch changes how many tokens are dropped —
   it perturbs the model, not just the speed.

4. **Do not select on validation cross entropy.** Diffusion eval CE here is a Monte Carlo estimate
   of an expectation over the noise level, so it is noisier than an autoregressive run's, and
   [Super Data Learners](https://arxiv.org/abs/2511.03276) finds val CE *rising* while downstream
   improves in exactly this regime. Early stopping would cut the run before the effect appears.

5. **`--learning-rate 0.01` is untuned**, carried from the MuonH work on this repository, and is the
   weakest number in the spec. Under Hyperball it is a relative step size and is not comparable to
   the baseline's `4e-4`. Read a first result as "does this train and does the constraint hold".

## The per-document reversal is dormant in this run, and that is fine

`document_reversal_index` reflects each position through the midpoint of its own document, and it
is tested over five boundary layouts — but **it never sees a boundary here.**
`NumpyFSLDatasetConfig` leaves `generate_doc_lengths` at `False` and the entry script does not
enable it, so no `doc_lens` reach the batch, `cu_doc_lens` is `None`, and the reversal is a plain
whole-row flip of all 4096 positions.

That is symmetric with the baseline rather than a regression against it. Without intra-document
masking a causal scan already carries state across the concatenated documents in a packed row, so
the reversed layers do the same thing in the other direction. What *would* be wrong is flipping a
row whose boundaries were supplied, because that moves tokens between documents — and that is the
case the boundary-aware path and its tests exist for. Turning on `generate_doc_lengths` activates
them; nothing here depends on it.

## Why the architecture is what it is

**A causal recurrence cannot carry a diffusion objective.**
[DeltaFlow](https://arxiv.org/abs/2608.01240) measures unidirectional GDN under a diffusion
objective and reports *entropy collapse* — it trains, the loss descends, and generation is
degenerate. So every GDN-2 block here is bidirectional, in DeltaFlow's **alternating scan**
arrangement: one scan per layer, direction flipping between layers, which costs nothing over the
causal layer. Each layer is still individually one-directional; what corrects that is the
non-causal attention every fourth block, which DeltaFlow calls "exact bidirectional correction".

**Not DeltaFlow's parallel variant**, which runs both directions per layer and reports the better
perplexity (21.2 vs 24.7). It costs ~40% more memory, and on the 40 GB A100 in this account the
loss path already pins the microbatch — buying the second scan would mean lowering it, which per
(3) changes the model. The parallel variant is the follow-up on a card with more memory;
`gpu-8xh100` is documented unobtainable here.

**Noise-adaptive gates are not optional.** DeltaFlow finds the bidirectional core alone lowers
perplexity but "still produces overly concentrated generations": the decay gate has to know how
corrupted its input is. The projections are zero-initialised, so an untrained model is bit-exactly
the noise-independent recurrence.

**Plain cross entropy, not the ELBO.**
[Scaling Beyond Masked Diffusion](https://arxiv.org/abs/2602.15014) (ICML 2026) reports plain CE as
12% more compute efficient than MDLM's `1/t` weighting. Quokka disagrees at the margin — reweighted
ELBO ahead at end-of-training, unweighted ahead early — so this is unsettled, and
`MaskedDiffusionConfig.loss_weighting` is where the other arm would go.

**Absorbing corruption, linear schedule.** Quokka: absorbing beats uniform "by a wide margin", and
linear `alpha = 1 - t` is strongest and lowest-variance with cosine worst.

## The scale risk, stated in advance

[DiffuMamba](https://arxiv.org/abs/2511.15927) finds linear-attention diffusion *losing* to plain
attention at 240M — its variants "struggle to generalize effectively" there — and only winning from
0.5B up. **370M is inside that gap.** The mitigating evidence is that DeltaFlow's own models were
104–110M parameters in this exact configuration (3:1 hybrid plus noise-adaptive gates) and did not
collapse. If this arm underperforms the baseline, that is a plausible result rather than a bug, and
the 3:1 ratio (25% attention against DiffuMamba's 20%) is the hedge.

## What is missing

**AR-to-diffusion conversion, which is the largest efficiency lever available.** Every frontier
diffusion LM is converted from an autoregressive checkpoint — RND1, Dream and DiffusionGemma all
are — because it reuses tokens already paid for. `--init-from` is where it belongs and it
**refuses**: writing it needs the source checkpoint's key names, to map one dense feed-forward onto
32 expert rows and to decide what happens to the four layers that stop being GDN-2, and this
repository may not read that bucket. A state dict whose keys silently do not match loads nothing
and trains to completion looking like the run it was meant to improve on, so it refuses rather than
warns. `edullm run` puts a shell on a machine that may read it; that is the first step.

**DeltaFlow's Temporal State Consistency loss.** Needs a second partial forward (~+25–30% compute),
which at a fixed 10B budget trades ~2.3B tokens for an auxiliary refinement.

**The `logits_to_keep` memory optimisation.** Only masked positions carry labels, and
`LMHead.forward` already accepts a tensor of positions to gather. Masked counts vary per row, so it
needs padding to the row maximum with a position whose label is already `-100`; at two sequences
per microbatch the expected saving is ~33% of the logits tensor. Not implemented, because the
simple path is the one worth putting a 20-hour run on first.

## Library changes this needed

| Change | Why |
| --- | --- |
| `GatedDeltaNet2.reverse_scan` + `document_reversal_index` | Per-document reversal. A whole-row flip across a packed sequence moves tokens between documents. |
| `GatedDeltaNet2` noise-adaptive gates | Zero-initialised per-head shifts on the decay and both gate logits. |
| `GatedDeltaNet2.init_weights` accepts `fan_in` | **It previously raised.** MuonH reads its radius off `\|\|W_0\|\|_F`, so the initialiser is part of the method — GDN-2 and MuonH could not be combined at all before this. |
| `causal` through `AttentionConfig` → `Attention` → all five backends | 15 hardcoded `causal=True` sites. Defaults to `True`, so every existing config is byte-identical. |
| `noise_level` through `Transformer._prepare_inputs` | Leftover kwargs there are *dropped*, so it had to be named explicitly. |
| `DiffusionTransformerTrainModule` | Corrupts the batch, then delegates. Unshifted labels via the `"labels" in batch` hook. |

Refusals rather than silent fallbacks, in each case because the failure trains a plausible-looking
wrong model: reversed scan under context parallelism, `causal=False` on the TE backend or fused
attention, non-causal with a sliding window, noise conditioning with no noise level, pipeline
parallelism under the diffusion train module, and `--init-from`.

## Verification

Run the specs in order — `run-smoke-diffusion-moe-muonh.yaml` first, and read its header for what
to look for in the log.

Verified locally (`uv run`):

- per-document reversal is an involution and a permutation, over five boundary layouts
- corruption: labels unshifted, `-100` off the masked positions, empirical rate matches the draw,
  antithetic pairs sum to 1, unscoreable positions untouched, schedules monotone and in range
- `causal=True` is the default and absent from the serialised config; non-causal attention lets
  position 0 move when the last token changes and causal attention does not
- the hybrid installs as 12 GDN-2 (6 forward, 6 reverse) + 4 non-causal attention over 16 layers
- the full config builds with `MuonHConfig`, `fan_in`, bf16, and a MASK id inside the free padding
- active non-embedding parameters are within 2.2% of the baseline's
- `isort`, `black`, `ruff`, `mypy` clean; 114 tests pass in the touched suites

Only reachable on a GPU with `flash-linear-attention`, and marked `requires_fla` so they skip here:
that a reversed layer equals the causal layer viewed backwards, that the zero-initialised noise
projections are a no-op at initialisation, and `fan_in`'s per-layer scales.

## What ES-dLLM actually is

[ES-dLLM](https://openreview.net/forum?id=O2WvMkJbws) (ICLR 2026) is a **training-free inference
accelerator**, not a training method. It observes that K, V and hidden states change only subtly
between successive denoising iterations, so it skips tokens in shallow layers by an importance score
built from intermediate-tensor variation plus the previous iteration's confidence, refreshing the
cache periodically to stop error accumulation. Reported 226.57 TPS on LLaDA-8B and 308.51 on
Dream-7B, one H200. It has no training-time component at all — the reference implementation has no
training scripts — so it cannot be part of this run. It needs the checkpoint this run produces.

**And it will not reproduce those numbers on this backbone.** ES-dLLM skips tokens against a KV
cache, and only the four attention blocks have one. A recurrence's state must advance over every
position, so on the twelve GDN-2 layers the output projection and gate can be skipped but the scan
cannot. The offsetting win is that the backbone is already linear-time, which is where DiffuMamba's
4.3× comes from. **The two gains are not multiplicative**, and the write-up should report the
measured split rather than a product.

On the throughput expectation overall: the best published stack is DiffusionGemma at 3.5–4× its
autoregressive counterpart (at openly worse quality — MMLU-Pro 77.6 against 82.6) plus ES-dLLM's
~2–3×. A single-digit multiple at batch size 1 and long sequence is the honest projection.
