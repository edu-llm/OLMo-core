# RFC: Should OLMo-core adopt Manifold-Constrained Hyper-Connections?

**Status:** Proposal, for team decision. Nothing here is merged or scheduled.
**Scope:** Whether to bring mHC into `main` as a supported architecture option.
**Recommendation in one line:** Do not merge yet. Fund one cheap pretraining ablation first, because the published effect is currently within noise.

---

## 1. What is being proposed

Hyper-Connections replace the single residual stream with `n` parallel streams. Each
sub-layer reads one vector in from those streams, runs unchanged attention or MLP, then
writes the result back out. A matrix `H_res` mixes the streams between layers. mHC
constrains `H_res` to be doubly stochastic, so repeated mixing over depth can neither
amplify nor suppress the residual signal.

Adopting it means changing the tensor contract between transformer blocks from
`(B, T, D)` to `(B, T, n, D)`. That is the whole cost story, and it is not a small change.

## 2. What we already know works

A static prototype exists on `edullm/adarsh-hc-ablation` and has been verified on CPU.

- Six arms build and run: single-stream baseline, unconstrained HC, and three mHC
  parameterisations (Sinkhorn, Birkhoff, Kronecker), plus an identity control, all at `n=4`.
- The correctness gate passes 30 of 30 checks. Every constrained mixer is doubly
  stochastic; every arm is numerically identical to the baseline at initialisation;
  routing math stays float32 under bfloat16.
- 525 existing tests still pass. No regressions on the single-stream path.

Routing parameters on a 267,424,512-parameter model:

| Arm | Mixer | Routing params | Per sub-layer |
| --- | --- | --- | --- |
| `baseline` | none, 1 stream | 0 | 0 |
| `mhc_identity` | fixed `I` | 192 | 8 |
| `kromhc` | Kronecker | 288 | 12 |
| `hc_unconstrained` | raw | 576 | 24 |
| `mhc_sinkhorn` | Sinkhorn | 576 | 24 |
| `mhc_lite` | Birkhoff | 768 | 32 |

**Read this correctly.** It shows the mechanism is implementable and cheap in parameters.
It says nothing about whether it makes models better. No training or evaluation has run.

## 3. What the evidence actually supports

Three sources, and they do not all say the same thing.

**Finetuning (arXiv:2607.18130, July 2026).** Wrapped frozen OLMo-2 1B and 7B with mHC.
Standalone mHC lost to LoRA at matched parameter budgets. The best combined result at 7B
was 0.980 test loss against 0.981 for LoRA alone, and downstream wins split four
benchmarks to four. **A 0.001 gap on a single seed is not a result.** It is smaller than
ordinary seed-to-seed variation, and the paper reports limited seed coverage.

**Pretraining (Xie et al., arXiv:2512.24880).** This is where the real claims live: mHC
beating HC and residual baselines at 27B with limited overhead. Different lab, different
scale, no independent replication, and none on OLMo.

**The interesting contradiction.** The finetuning paper found that fixing `H_res` to the
identity, that is, not mixing streams at all, matched or beat learned mixing. That
directly opposes the pretraining paper, where learned mixing is the point. Nobody has
tested which holds during pretraining.

## 4. What full adoption costs

Phase 1, the prototype above, is the easy part and it is done. The remaining work is where
the risk concentrates. Risk ratings are from the internal implementation map.

| Area | Risk |
| --- | --- |
| Activation checkpointing and recomputation | High |
| FSDP, tensor parallelism, DTensor layouts | High |
| Pipeline parallelism | Very high |
| MoE and expert parallelism | High |
| Checkpoint format, resume, HF export | High |
| Optimised kernels and serving | Very high |

Three costs deserve naming plainly.

**Memory, not parameters.** At `n=4` the model carries four times the residual state
between every layer. Routing weights are negligible; activation memory is not. This, not
parameter count, is what will bound the usable configuration.

**It collides with live work.** mHC routes around the MoE branch, and the standard and
hybrid MoE blocks currently hand-code their residual adds. Those blocks are under active
experimentation in the MoE study on `edullm/moe-m1-pilot`. Adopting mHC means editing code
somebody else is running experiments against.

**Existing checkpoints have no routing state.** Every published OLMo checkpoint would need
a defined load path, and export and conversion would need updating.

## 5. Decision gates

Do not advance a gate until the previous one passes.

- **Gate 1, correctness.** Baseline-equivalent initialisation, constrained matrices,
  gradient flow, save and resume, eager and compiled parity. **Already passing** on the
  prototype branch.
- **Gate 2, science.** Matched baseline against HC and mHC at `n=4`: identical tokens,
  data, and schedule, **multiple seeds**, reporting validation loss and downstream tasks
  with variance. Not started.
- **Gate 3, systems.** Tokens per second, peak memory, step time, communication bytes, and
  wall-clock time to a target loss. Not started.

Gate 2 is the one that matters. Because the published effect sizes are near noise, a single
run per arm cannot distinguish a real improvement from luck. Multiple seeds with reported
variance is not optional rigour here; without it the experiment cannot answer the question.

## 6. Recommendation

**Do not merge mHC into `main` now.** The evidence does not yet justify a high-risk change
to a core tensor contract, and roughly eighty percent of the remaining cost sits in the
distributed, MoE, and pipeline work that only pays off if the science holds.

**Do fund Gate 2 at small scale.** The prototype already runs all six arms. A matched
pretraining ablation with several seeds is comparatively cheap and answers the one
genuinely open question: does the identity finding survive into pretraining? If it does, it
contradicts the original mHC paper and is a publishable result in its own right, and it
would also mean the expensive learned-mixing machinery can be skipped. If mixing does win,
we then have our own evidence rather than someone else's, and Gate 3 becomes worth funding.

Either outcome is informative, which is what makes this the right next spend.

## 7. What we are not claiming

No training or evaluation has been run in this repository. Every number in section 2 is a
build-time or correctness measurement. Dynamic routing, MoE integration, and fused kernels
are unimplemented. Tensor and context parallelism deliberately raise `NotImplementedError`
on a hyper-connected block rather than silently applying a plan written for a 3-D hidden
state.
