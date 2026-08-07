# RFC: Should OLMo-core adopt Manifold-Constrained Hyper-Connections?

**Status:** Proposal, for team decision. Nothing here is merged or scheduled.
**Scope:** Whether to bring mHC into `main` as a supported architecture option.
**Recommendation in one line:** Do not merge yet. Fund one small pretraining ablation first, because the strongest available evidence points at the cheap variant, not the expensive one.

---

## 1. What is being proposed

Hyper-Connections (HC) replace the single residual stream with `n` parallel streams. Each
sub-layer reads one vector in from those streams, runs unchanged attention or MLP, then
writes the result back out. A matrix `H_res` mixes the streams between layers. mHC
constrains `H_res` to be doubly stochastic, so repeated mixing over depth can neither
amplify nor suppress the residual signal.

Adopting it means changing the tensor contract between transformer blocks from
`(B, T, D)` to `(B, T, n, D)`. That is the cost story, and it is not a small change.

## 2. This is not a new idea here

A brainlift written in this org in July 2026 (`MLArchitecture/brainlifts/mhc-multi-stream-information-flow.md`)
already worked through HC and mHC and proposed a three-arm experiment: standard residual,
HC at `n=4`, mHC at `n=4`, compared on loss, gradient stability, memory and training time,
and math and reasoning accuracy.

**That experiment is now built.** The prototype on `edullm/adarsh-hc-ablation` implements
those three arms plus three more, and clears its correctness gate. This RFC is the decision
that brainlift deferred, with the implementation cost now measured rather than guessed.

## 3. What we already know works

Verified on CPU. No training has run.

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

This shows the mechanism is implementable and cheap in parameters. It says nothing about
whether it makes models better.

## 4. What the evidence actually supports

**The constraint itself is well motivated.** Unconstrained HC compounds its mixing matrix
over depth, and the mHC paper measures composite signal gains approaching 3000. Under the
doubly stochastic constraint the same quantity stays near 1.6. If we run multi-stream at
all, we should run it constrained. That part is not in question.

**Pretraining (Xie et al., arXiv:2512.24880).** Reports lower pretraining loss, steadier
gradient norms, and better benchmarks at 3B, 9B and 27B, with about 6.7% additional
training time at `n=4`. Different lab, no independent replication, and none on OLMo.

**Finetuning (arXiv:2607.18130, July 2026).** Wrapped frozen OLMo-2 1B and 7B. Standalone
mHC lost to LoRA at matched budgets. The best combined result at 7B was 0.980 test loss
against 0.981 for LoRA alone, with downstream wins splitting four benchmarks to four. On
the seed coverage reported, a 0.001 gap is not a result.

**The signal worth acting on: the mixing matrix may not be earning its keep.** Three
independent lines now point the same way.

1. The finetuning paper found that fixing `H_res` to the identity matched or beat learned
   mixing, while removing parameters.
2. It reports this agrees with earlier pretraining interpretability work (Alimaskina et
   al., arXiv:2606.03483), where deeper-layer mixing in mHC-lite also collapsed toward
   identity.
3. Causal work on stream dominance (Peng et al., arXiv:2603.14833) finds some streams
   carry the model's behaviour while representationally similar streams stay passive.

So the collapse-toward-identity behaviour has already been *observed* during pretraining.
What has **not** been done is the controlled ablation asking whether *fixing* it to identity
preserves or improves pretraining performance. That gap is narrow, cheap to close, and
sits exactly where our prototype already runs.

## 5. What full adoption costs

Phase 1, the prototype, is done. The remaining work is where risk concentrates. Ratings
are from the internal implementation map.

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
between every layer. Routing weights are negligible; activation memory and inter-layer
bandwidth are not. The 6.7% training-time figure above comes from an optimised
implementation, and ours is not one.

**It collides with live work.** mHC routes around the MoE branch, and the standard and
hybrid MoE blocks currently hand-code their residual adds. Those blocks are under active
experimentation on `edullm/moe-m1-pilot`. Adopting mHC means editing code somebody else is
running experiments against.

**Existing checkpoints carry no routing state.** Every published OLMo checkpoint would need
a defined load path, and export and conversion would need updating.

## 6. Decision gates

Do not advance a gate until the previous one passes.

- **Gate 1, correctness.** Baseline-equivalent initialisation, constrained matrices,
  gradient flow, save and resume, eager and compiled parity. **Already passing.**
- **Gate 2, science.** Matched arms on identical tokens, data and schedule, reporting
  validation loss and downstream tasks. Not started.
- **Gate 3, systems.** Tokens per second, peak memory, step time, communication bytes, and
  wall-clock time to a target loss. Not started.

**Gate 2 must meet the bar this team already uses elsewhere.** The multi-hop work
preregistered its gates and required effect sizes with confidence intervals over three to
five seeds, on the explicit reasoning that a small effect cannot be ruled in or out at one
seed. mHC effect sizes are smaller than that project's were. Anything less than the same
standard cannot answer the question, and a single run per arm would waste the compute.

## 7. Recommendation

**Do not merge mHC into `main` now.** The evidence does not justify a high-risk change to a
core tensor contract, and most of the remaining cost sits in distributed, MoE and pipeline
work that only pays off if the science holds.

**Do fund Gate 2 at small scale**, with the identity arm treated as a first-class
hypothesis rather than a control. The most likely outcome, on current evidence, is that
fixed identity matches learned mixing. That result is worth more than it sounds: it would
mean we get the multi-stream capacity while skipping Sinkhorn, the dynamic router, and most
of the high-risk column in section 5. The cheap variant winning is the good outcome, not the
null one.

If learned mixing does win, we will have our own pretraining evidence on OLMo rather than
someone else's, and Gate 3 becomes worth funding.

**A note on which benchmarks.** The generic suites (ARC, HellaSwag, PIQA, Winogrande, MMLU,
GSM8K, Minerva-MATH) are wired up and are the right common currency. But the July brainlift
argued the real hypothesis for an educational model is that separate streams might carry
subject knowledge, student state, reasoning, and pedagogical policy. If that is the actual
motivation, Gate 2 should also measure misconception identification, multi-turn student-state
consistency, and answer leakage when the model should only hint. Those are not in the harness
yet, and generic benchmarks will not detect them.

## 8. What we are not claiming

No training or evaluation has been run in this repository. Every number in section 3 is a
build-time or correctness measurement. Dynamic routing, MoE integration, and fused kernels
are unimplemented. Tensor and context parallelism deliberately raise `NotImplementedError`
on a hyper-connected block rather than silently applying a plan written for a 3-D hidden
state. The stream-specialisation hypothesis in section 7 remains a hypothesis; nothing
guarantees streams learn separable human-interpretable roles.
