# Hyper-connections residual-mixer ablation

## What this is for

Hyper-connections replace a transformer sub-layer's single residual with `n` parallel streams.
Per token, with streams `Z` of shape `n x d`:

```
x   = h_pre^T @ Z                     # (d,)  one branch input, read out of n streams
Z'  = H_res @ Z + outer(h_post, f(x)) # the branch output written back to every stream
```

`h_pre = sigmoid(theta_pre)` and `h_post = 2 * sigmoid(theta_post)`. The interesting object is
`H_res`. The original Hyper-Connections paper (Zhu et al. 2025) learns it as a raw `n x n`
matrix. Three later papers each constrain it to be *doubly stochastic* — nonnegative with unit
row and column sums — and each does so a different way, arguing that the constraint is what
makes the method stable at scale.

Nowhere public are those three constructions compared against each other under a matched budget.
That is the question this harness is built to ask, and only that: **at `n = 4` and a matched
token, data and seed budget, does the choice of doubly stochastic parameterisation matter, and
does any of them beat both the single-stream baseline and the unconstrained control?**

**No training has been run and no evaluation numbers exist.** Everything below describes a
specification plus the CPU checks that back it. The loss curves and task numbers are the part
that needs GPUs.

## The six arms

Every arm shares the model shape, the optimizer, the schedule, the data, the token budget and
the seeds. The residual mixer is the only thing that changes.

| arm | mixer | n | routing params / sub-layer | routing params / 190M model | what it answers |
| --- | --- | --- | --- | --- | --- |
| `baseline` | – | 1 | 0 | 0 | what an ordinary residual gets |
| `hc_unconstrained` | `unconstrained` | 4 | 24 | 576 | original HC; the instability control |
| `mhc_sinkhorn` | `sinkhorn` | 4 | 24 | 576 | exact mHC (Xie et al. 2026) |
| `mhc_lite` | `birkhoff` | 4 | 32 | 768 | mHC-lite (Yang & Gao 2026) |
| `kromhc` | `kronecker` | 4 | 12 | 288 | KromHC (Zhou et al. 2026) |
| `mhc_identity` | `identity` | 4 | 8 | 192 | streams and gates, no mixing |

Every count is `2n` gate logits plus the mixer's own parameters, and every one of them is
asserted exactly in `src/test/nn/hyper_connections_test.py`. The per-model column is
`2 sub-layers x 12 blocks` at the 190M shape, plus 0 for the `mean` stream readout.

The two controls are what make the middle four legible. `baseline` says what the extra streams
have to beat. `mhc_identity` keeps the streams and both gates but pins `H_res = I`, so a gain
that `mhc_identity` already captures is a gain from the read-in/write-out gating rather than
from mixing the streams at all — and that is the cheapest explanation for a positive result.

### The mixers

- **`identity`** — `H_res = I_n`. Zero parameters.
- **`unconstrained`** — a raw learned `n x n` matrix, `n^2` parameters. Deliberately *not*
  doubly stochastic: nothing bounds its spectral radius. A test asserts that it can go negative
  and that its row sums leave 1, because if that ever stops being true the control has quietly
  become a sixth constrained variant and the ablation stops measuring anything.
- **`sinkhorn`** — `H_res = Sinkhorn(logits)`, 20 iterations of alternating row and column
  normalisation, `n^2` parameters. Run in log space; see the caveat below.
- **`birkhoff`** — by Birkhoff–von Neumann, `H_res = sum_k a_k P_k` over all `n!` permutation
  matrices with `a = softmax(theta)`. 24 parameters at `n = 4`. Exactly doubly stochastic, no
  iteration.
- **`kronecker`** — `H_res = A_1 ⊗ ... ⊗ A_{log2 n}` with each `A_k = [[p, 1-p], [1-p, p]]` for
  `p = softmax(theta_k)_0`. `2 log2(n)` parameters. Requires `n` to be a power of two and raises
  `OLMoConfigurationError` otherwise.

### Initialisation

At initialisation every arm is numerically the unwrapped OLMo-2 backbone. `h_pre` is the uniform
average `1/n`, `h_post` is all ones, and every constrained mixer starts at the uniform doubly
stochastic matrix, so with `n` identical streams the update is `z + f(z)` in each of them. With
`init_noise_std = 0` this is bit-exact against a `reordered_norm` model for all five mixers,
which the gate check reports as `max|logits - baseline| = 0.000e+00`.

### Symmetry breaking

Identical streams leave an `S_n` permutation symmetry that gradient descent preserves exactly:
without breaking it the streams stay copies of each other for the whole run and `n > 1` buys
nothing but memory. Two mechanisms, both from the mHC paper:

- Small Gaussian noise on the gating logits at initialisation, `init_noise_std`, defaulting to
  `1e-2`. After the noise, `h_pre` is renormalised so it stays a convex combination.
  **Appendix C of the paper says `1e-2` and Table 13 of the same paper says `1e-4`, and the
  paper does not reconcile them.** The default here is the Appendix C value because that is the
  passage explaining the mechanism, but the discrepancy is real, it is documented on the config
  field, and it is a reasonable thing for this harness to settle rather than inherit.
- Bernoulli dropout on the residual-mixer logits during training, `residual_dropout_p = 0.1`.
  Masked entries go to `-inf` *before* the constraint map, so the survivors are renormalised
  rather than merely rescaled, with a guard that never masks a whole row or column. Disabled at
  eval. It applies to `sinkhorn`, `birkhoff` and `kronecker`; `identity` has no logits, and
  `unconstrained` has no constraint map to sit in front of, so neither uses it.

### Precision

All routing quantities — `h_pre`, `h_post`, `H_res` and the whole Sinkhorn iteration — are
computed in float32 even when the activations are bfloat16, then cast back. In bfloat16 the
Sinkhorn fixed point is reached to about two decimal digits and the row and column sums drift
far enough off 1 that the doubly stochastic property, which is the entire claim of mHC, no
longer holds.

## Evaluation

Both suites are OLMo-core's own task names, run through `DownstreamEvaluatorCallbackConfig`.
There is no second eval harness. Every name below was checked against
`src/olmo_core/eval/task_groups.py`; all thirteen exist verbatim.

**CLASSIC** (5 tasks) — the OLMES-style multiple-choice core:
`arc_easy_test_rc_5shot`, `arc_challenge_test_rc_5shot`, `hellaswag_rc_5shot`,
`piqa_val_rc_5shot`, `winogrande_val_rc_5shot`.

**MATH_REASONING** (8 tasks) — bits-per-byte on gold reasoning traces:
`gsm8k_gold_bpb_5shot`, and the seven `minerva_math_*_gold_bpb_0shot` subtasks (`algebra`,
`counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`,
`precalculus`).

BPB is in the second suite on purpose: multiple-choice accuracy at 190M is largely noise, and
the case for a richer residual topology is a case about depth-wise composition, which multi-step
arithmetic is the thing that asks for.

Running either suite needs the `eval` extra (`pip install 'ai2-olmo-core[eval]'`, which pins
`ai2-olmo-eval==0.9.0`) and a GPU. The configs are built by the ablation script but nothing in
this repository runs them.

## Running the CPU checks

Neither command trains anything, allocates a GPU, or reaches a network.

```bash
# The arm table: mixers, stream counts, routing parameter counts, both eval suites.
python src/scripts/ablations/hc_ablation.py

# Build all six arms on CPU, run a forward pass, check shapes and parameter counts.
# --model-size tiny (d_model=128, 2 layers) takes seconds; the default 190M takes about a minute.
python src/scripts/ablations/hc_ablation.py --dry-run
python src/scripts/ablations/hc_ablation.py --dry-run --model-size tiny

# Dump one arm's full model, train-module and trainer config.
python src/scripts/ablations/hc_ablation.py --show mhc_sinkhorn

# GATE_1_CORRECTNESS. Exits nonzero if any check fails.
python src/scripts/ablations/hc_gate1_check.py
python src/scripts/ablations/hc_gate1_check.py --check-compile   # slow; adds eager/compile parity

# The unit tests. All CPU, no GPU marks.
pytest -v src/test/nn/hyper_connections_test.py
```

## What is verified on CPU here, and what needs a GPU

### Verified on CPU

Each of these is an assertion in `src/test/nn/hyper_connections_test.py`, a check in
`hc_gate1_check.py`, or both.

- **Baseline-equivalent initialisation.** With the noise off, a hyper-connected model produces
  bit-identical logits to the `reordered_norm` backbone it wraps, for all five mixers, and a
  wrapped block reproduces `x + LN(branch(x))` in every stream.
- **Constrained matrices.** `identity`, `sinkhorn`, `birkhoff` and `kronecker` give nonnegative
  `H_res` with row and column sums within `1e-5` of 1, at `n = 2` and `n = 4`, both at
  initialisation and with the logits pushed well off it. `unconstrained` is asserted *not* to.
- **Parameter counts.** Exactly 8 / 24 / 24 / 32 / 12 per wrapped sub-layer at `n = 4`, measured
  off the built module and cross-checked against `HyperConnectionConfig.num_params()`.
- **Shapes and gradients.** Forward gives `(B, T, n, D)`; backward reaches every routing tensor
  with a finite, nonzero gradient.
- **Symmetry breaking.** Streams stay identical to `< 1e-6` after an optimizer step with the
  noise off, and diverge with it on. Both directions, because a change that made the streams
  diverge on their own would silently break the baseline equivalence above.
- **Precision.** With the module in bfloat16, `h_pre`, `h_post` and `H_res` all come back
  float32 while the output stays bfloat16.
- **Numerical robustness.** Sinkhorn stays finite and nonnegative for logits up to `1e8`;
  `kronecker` raises on a non-power-of-two `n`; the dropout's row/column guard never lets a
  `nan` through at `p = 0.9` over 30 draws.
- **Save/resume.** A state-dict round-trip reproduces the forward pass exactly and carries every
  routing parameter.
- **Eager/compile parity, at the block level.** `hc_gate1_check.py --check-compile` compiles a
  single hyper-connected block on CPU and agrees with eager to `~2e-6` for all five mixers. This
  is a block, on CPU, in eval mode — not a compiled training step on a GPU.
- **No regression.** The existing single-stream path is asserted unchanged, and the full
  `src/test/nn/` suite passes.

### Not verified — needs a GPU run

- **Everything empirical.** No arm has been trained. There is no loss curve, no downstream
  number, no seed variance and no wall-clock or throughput measurement for any of the six arms.
  Nothing here supports a claim that any mixer is better than any other.
- **GATE_2 (science) and GATE_3 (systems)** in the mHC dossier's sense. This harness only
  clears GATE_1.
- **The evaluation suites.** Configured, never executed. They need the `eval` extra and a GPU.
- **bfloat16 end to end.** The routing dtype policy is unit-tested, but no model has run a
  training step in bfloat16 on real hardware.
- **`torch.compile` on a real training step.** Block-level forward parity on CPU passes (above),
  which says nothing about a compiled backward, dynamic shapes, or a whole model. The ablation's
  train-module config leaves `compile_model=False` rather than turning it on unverified.
- **Distributed anything.** FSDP wrapping is inherited and untested against a four-dimensional
  stream. Tensor parallelism and context parallelism **raise `NotImplementedError`** on a
  hyper-connected block rather than silently applying a plan written for a three-dimensional
  hidden state. Pipeline parallelism is arranged for — the streams stay expanded across stage
  boundaries and collapse once, on the stage that owns the LM head — but has not been run.
- **MoE.** Out of scope. The MoE blocks hand-code their own residual adds and have not been
  touched.
- **Dynamic (input-dependent) routing.** Out of scope; the dossier's implementation order puts
  it after the static reference is stable, and this is the static reference.

## Known caveats

**Sinkhorn's 20-iteration budget only converges near zero.** 20 iterations is what the mHC paper
specifies. Sinkhorn's convergence rate degrades as the logits grow and the fixed point
approaches a permutation matrix. Past roughly `|logit| ~ 10` the result is still finite and
nonnegative and its column sums are still exactly 1 — the column normalisation is applied last —
but its **row sums drift**, by a factor of two or more at `|logit| ~ 100`. A row sum below 1
shrinks that stream's residual, so a run whose `H_res` logits grow that large is no longer doing
what the method says it does. The cheap way to notice is to watch the largest absolute residual
logit. `birkhoff` and `kronecker` have no such regime: both are exactly doubly stochastic for
any logits at all, which is a real argument in their favour that the ablation is positioned to
test.

**A doubly stochastic mixer preserves the sum over streams exactly.** That means any objective
that only sees `out.sum(dim=streams)` has *zero* gradient with respect to the `birkhoff` and
`kronecker` parameters. It is a property, not a bug, and it is easy to mistake for one: a
"do gradients reach the routing parameters" check written against `out.sum()` reports a failure
that is not there. Both the test suite and the gate check use a squared loss and say why.

**The gates' init noise is seeded through the model, not globally.**
`Transformer.init_weights` calls every module's `reset_parameters()` without a generator, which
would have left the routing noise dependent on ambient RNG state.
`HyperConnectionTransformer.init_weights` re-draws it from the model's seeded generator
afterwards, so an HC run is reproducible from its config. A `HyperConnection` built and used
outside a `HyperConnectionTransformer` gets the global RNG unless a generator is passed.

## Running it on GPUs

GPU runs go through the eduLLM platform's `edullm` CLI, not Beaker; the `ai2/*` clusters named
throughout this repository are unreachable from here. See `AGENTS.md`. Two things specific to
this ablation:

- The precision guard reads the *text* of the command, so name the dtype in it. A command that
  only sets the dtype in code is accepted onto a T4, which has no bfloat16 in hardware.
- Six arms at a matched budget is six runs, plus however many seeds the comparison needs. A
  single-seed difference in validation loss establishes nothing; the mHC dossier is explicit
  that a 0.001 single-run loss gap cannot support a superiority claim.

## Where the code lives

| path | what |
| --- | --- |
| `src/olmo_core/nn/hyper_connections.py` | `ResidualMixerType`, `HyperConnectionConfig`, `HyperConnection`, `StreamCollapse` |
| `src/olmo_core/nn/transformer/hc_block.py` | `HyperConnectionTransformerBlock` |
| `src/olmo_core/nn/transformer/model.py` | `HyperConnectionTransformer`, and the `expand_residual_streams` / `collapse_residual_streams` hooks on `Transformer` |
| `src/scripts/ablations/hc_ablation.py` | the six arms, the table, the CPU dry run |
| `src/scripts/ablations/hc_gate1_check.py` | GATE_1_CORRECTNESS |
| `src/test/nn/hyper_connections_test.py` | the CPU test suite |
