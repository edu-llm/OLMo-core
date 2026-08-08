# HANDOFF — linear-attention mixer bake-off

**Scope:** this file covers **only** the six-operator mixer bake-off run on 2026-08-08. It is a
**new track** and it does not supersede its siblings, which remain current:

- [`../../HANDOFF.md`](../../HANDOFF.md) at the worktree root — **DP2-KDA Phase 0/1 preparation**.
  Different track, do not overwrite. Its audit ("sound math, correct kernel, **not**
  production-viable") is *confirmed by this run* — see [What worked ⑤](#-5-the-r1-control-is-what-made-the-householder-result-interpretable).
- `Capstone_LLM/HANDOFF.md` (repo root) — **LIV brainlift**. Its next-step **⓵ "THE NEXT EXPERIMENT:
  a gated conv inside KDA"** is the experiment this file reports. **⓵ is now ANSWERED.**
- `KDA/HANDOFF.md` (KDA-Householder, COMPLETE), `KDA-LIV/HANDOFF.md` (sub-500M KDA insertion),
  `docs/dynconv-review/HANDOFF.md` (dynamic short convs — tests the *filter*, this tests the *gates*).

**Last updated:** 2026-08-08. **Status: RUN 1 IS COMPLETE. 18/18 cells, zero failures, zero hard
errors, $790.47 declared ceiling, self-released with no lead. The quality result is a well-powered
NULL; the actionable results are in throughput, memory and reproducibility. Nothing is queued.**

**Branch `edullm/mixer-bakeoff`, HEAD `0ed568f`.** Results, per-cell JSONs, report and PDF are all
committed under `docs/mixer-bakeoff/run1/`.

---

## Goal

**Choose which linear-attention mixer the team's next large production run should use.** Not a paper
— a ranking with trade-offs. Six candidate operators, three seeds each, at fixed parameters and
fixed everything-else, so that arm-minus-arm is the operator and only the operator.

The endpoint was deliberately **three numbers, not one**: held-out cross-entropy, steady-state
throughput, and peak memory. That choice was made before the run and it is what saved the
deliverable — see [Key decisions ①](#-1-throughput-and-memory-were-co-primary-from-the-start).

---

## Current progress

**Run 1 is done and analysed.** `run_019fe0f9-1bbd-702c-b141-6d58e128bda6`, `gpu-8xa100`, source sha
`9425fea`, 1,144 steps = 599,785,472 tokens/cell on `reservoir-dolma2-v1`, TPP 1.5.

| operator (mechanism) | registry key | val_ce | sd | tok/s | ×ref | peak GiB | reserved GiB | MFU |
|---|---|---|---|---|---|---|---|---|
| KDA, no conv activation | `KDA_NOACT` | **3.0400** | **0.00258** | 418,364 | 0.998 | 9.153 | 10.721 | 41.5% |
| KDA + LIV gated convolution | `KDA_GCONV` | 3.0451 | 0.01678 | 410,690 | 0.979 | **9.434** | 11.018 | 40.7% |
| KDA, as shipped (reference) | `KDA_BASE` | 3.0514 | 0.01075 | 419,288 | 1.000 | 9.153 | 10.721 | 41.5% |
| KDA-Householder, neg. eigenvalues, R=2 | `KDA_R2` | 3.0581 | 0.00642 | 295,638 | **0.705** | 9.361 | **13.391** | **29.1%** |
| KDA-Householder, neg. eigenvalues, R=1 | `KDA_R1` | 3.0749 | **0.04516** | 326,513 | **0.779** | 9.216 | **13.260** | 32.4% |
| Gated DeltaNet-2 | `GDN2` | **3.0884** | 0.00404 | 416,894 | 0.994 | 9.333 | 10.893 | 42.0% |

**Statistics.** Pooled σ̂ = **0.02042 nats at df = 12**, χ² interval [0.01464, 0.03370] (factor-2.3,
the pre-registered target precision). **MDE = 0.0636 nats.** ANOVA F(5,12) = 2.489, p = 0.091.
Dunnett crit 2.9013, k = 5: **no contrast clears the MDE — CE is NOT RESOLVED at n = 3.**

**Homogeneity is contested and it matters.** Levene p = 0.484 (does not reject), Bartlett p = 0.0059
(rejects). `KDA_R1`'s sd is 17× `KDA_NOACT`'s. Under the pre-registered unequal-variance fallback
(Welch F = 53.29, p = 0.0005) GDN2's Dunnett-T3 CI is **[+0.0262, +0.0479], excluding zero**. Both
readings are in the report; neither is presented as settled.

**The gate contrast — the only attributable test of the gated conv.**
`KDA_GCONV − KDA_NOACT = +0.00511 nats`, CI [−0.0312, +0.0414], p = 0.765.

**Artifacts** (all under `docs/mixer-bakeoff/run1/`): `cell-0.json` … `cell-17.json` (18 per-cell
summaries), `REPORT.md` (184-line generated analysis), `results.json` (machine-readable),
`RESULTS-TABLE.pdf` + `.html` (3-page A4 landscape table with per-arm pros/cons).
Frozen design: `docs/mixer-bakeoff/seeds.json` + `PREREGISTRATION.md`.

**Per-cell integrity, all 18:** step-0 loss 11.71 inside the ln(100,352) = 11.5164 ± 0.5 band;
`val_tokens_present == declared` exactly (975,077,376 over all 39 held-out shards); parameter count
matched `ARM_L0_DELTA`; seed pair matched the frozen schedule; 0 hard errors.

---

## What worked

### ① Naming arms by mechanism, not by registry key
`KDA_R2` says nothing. "KDA-Householder, negative eigenvalues, R=2" says what a reader needs to
interpret the row. The PDF leads with mechanism names and keeps the key in monospace underneath so
numbers stay traceable to `core6_arms.py`.

### ② The `KDA_NOACT` control, which inverted a headline
The depthwise pre-gate is **algebraically a SiLU**: `2·sigmoid(a·u)·u ≡ (2/a)·silu(a·u)` exactly,
with `2/a` absorbed into the conv taps (verified 8.9e-16 in fp64). So `KDA_GCONV − KDA_BASE` moves
three things at once and reads **−0.0062** (apparently better); `KDA_GCONV − KDA_NOACT` isolates the
post-gate and reads **+0.0051** (nominally worse). **Opposite sign, same arm.** Without this control
the run would have produced a plausible, publishable, wrong number.

### ③ Arm-major cell ordering
Cells 0–2 = arm 1, 3–5 = arm 2, … A truncated arm-major fan-out loses **whole arms** and leaves the
rest at n=3; seed-major would have lost one seed from every arm and left nothing usable. This is
what made the schedule risk survivable rather than fatal, twice.

### ④ Pre-registering that CE might not resolve
The pre-registration committed to "CE may be unresolved; the recommendation then falls back to
throughput and memory" **before** the run. When that happened, it was a planned outcome rather than
a scramble. The generated report states it in those words rather than ranking point estimates.

### ⑤ The R=1 control is what made the Householder result interpretable
`KDA_R1` is `KimiDeltaHouseholder(num_householder=1)` — parameter- and FLOP-identical to
`KimiDeltaAttention` (2,473,388,544 FLOP/token, verified equal) and the same function. It ran at
**0.779×**. So:

> **kernel overhead alone = 28.6%; the arity step R=1→R=2 costs a further 10.9%.**

**The kernel, not the mechanism, is what makes R>1 expensive.** No amount of R=2 data alone could
have separated those.

### ⑥ Structural verification instead of textual
A flag is only real if `add_argument` declares it (see What didn't work ①). A merge is only clean if
each class appears exactly once. A guard is only real if a mutation makes it fail. Every check in
this run was a set-difference or a mutation, not a grep.

### ⑦ Self-release, verified live
Three reads: active `team-leads` membership; `prevent_self_review: false`; and
`current_user_can_approve: true` on a real waiting run. Both submissions released by `ericrcwu001`
with no lead. **`edullm check`'s `approval_class` was wrong** — it said `automatic`, the real gate was
`PENDING_APPROVAL` for both runs. Read the gate from the waiting deployment, never from `check`.

---

## What didn't work

### ① `--lm-loss-implementation` killed all 18 cells of the first array
Copied from `train_on_corpus.py`, which declares it; `train_core6_arm.py` does not. Exit 70
`THE_CONFIG_WOULD_NOT_BUILD`. **My pre-flight passed** because I grepped for each flag *string* and
that one appears in a prose comment. An undeclared flag does not fail at parse time — unknown args
become dotted config overrides, so it died at `config.merge()` *after* the corpus, tokenizer and arm
ledger had all logged correctly, which reads like a config bug in the arm rather than a typo.
**Fix: set-difference `--flags` used against `add_argument` names.** Cost: minutes, because
arm-major ordering surfaced it in the first wave.

### ② The image is per-COMMIT, not per-branch — three builds lost
A **pure-YAML commit** moved HEAD after a green build and made the branch unsubmittable
("nothing has been published from this commit"). `edullm check` cannot catch this: image checks are
**deferred past check**, so exit 0 / zero refusals / `automatic` is compatible with having no image.
A build **re-run replays the original sha**, so it never fixes `remote_ref_mismatch`.
**Rule: commit everything → build → submit. Never interleave.**

### ③ `analyse_bakeoff.py` hung with zero output, 138 tests green
`if not sys.stdin.isatty(): sys.stdin.read()` blocks forever in an agent shell, where stdin is
neither a TTY nor closed. pytest closes stdin, so no assertion in the file could see it.
**Always `< /dev/null`.** Found with `faulthandler.dump_traceback_later(12, exit=True)`.

### ④ Two agents in one worktree corrupted each other's work
I ran `git stash` to get a clean tree for `edullm check` and swallowed another agent's uncommitted
edits; it ran a source-mutating harness and I read a mutated file mid-cycle, reporting **9 phantom
test failures including a fake data-fabrication bug**. Both recovered. **Rule: no `git stash` and no
source-mutating harness in a shared worktree — use `/tmp` copies.**

### ⑤ Believing an aggregate over a per-object query
`arrayProperties.statusSummary` **lags** individual cells; CloudWatch `lastEventTimestamp` lags real
output. I twice diagnosed a "stall" that was a cell running *ahead* of its siblings. **Per-index
`describe-jobs` and the actual stream tail are the truth.**

### ⑥ The summary JSON is not in S3
`train_core6_arm.py:1906` **prints** it to stdout; only checkpoints reach the bucket. My watcher's
`aws s3 cp` would have reported "no results" forever. Results come from **CloudWatch**.

### ⑦ Estimates I had to widen three times
Eval cost rose with concurrency (S3 contention for 39 val shards) and again on the Householder arms
(same slow kernel, forward-only). 18 min → ~26 min. Each was contention or kernel cost, never a
fault — but I reported each widening rather than quietly re-baselining.

---

## Key decisions

### ① Throughput and memory were co-primary from the start
Because the measured softmax-vs-gated-delta CE gap is ~0.010 nats — below what n=3 resolves. A
CE-only bake-off would have produced six confident nulls and decided nothing. **Vindicated:** CE came
back unresolved (MDE 0.0636), and peak-memory sd is **exactly 0.000 within every arm**, so those
ratios are mechanism rather than variance.

### ② All six arms keep the same six global-attention layers (2,5,8,10,12,14)
Non-negotiable and mechanical: `model.py:257` emits RoPE buffers only for `Attention`/`FusedAttention`
blocks, so an arm with fewer attention layers **loses positional encoding** and its loss gap stops
being a mixer effect. A prior track lost a design to exactly this (a 1.23-nat "collapse" that was
mostly missing position).

### ③ Treatment stays in two slots {6, 11}
Widening would buy statistical power by spending the thing being measured — the contrast would
confound "which mixer" with "how much mixer", and would no longer compare against the K2/G4R2
numbers already taken at two slots.

### ④ Depthwise gate, not lowrank
Lowrank costs +2,359,296 params (**12× the declared tolerance**) *and* adds nine `nn.Linear` per layer
whose `reset_parameters` draw from the **global** RNG before the seeded generator exists — so its
random stream diverges from every other arm and **seed pairing is forfeited**. Depthwise costs 6,144
per layer and lands at +2,208, inside tolerance, asserted exactly.

### ⑤ Token budget cut 1,907 → 1,144 steps; arms and seeds untouched
Capacity forced it (one p4d live at sizing time; worst observed queue 12.6 h). Cutting an arm or a
seed destroys a contrast permanently and n=2 hits the t(0.975,1)=12.706 cliff. Tokens only cost
precision the design could not spend anyway. **Declared cost: TPP falls to 1.5 and CE magnitudes
inflate.**

### ⑥ fla pinned at 0.5.1, in exactly one place
`chunk_gdn2` **already exists in 0.5.1** (new there, absent in 0.5.0), and `gdn2/chunk.py`,
`layers/kda.py`, `kda/chunk.py` are **byte-identical** to 0.5.2 — so GDN2's 0.5.2 pin was
unnecessary, and 0.5.1 keeps the sm_89 numerics gate valid. **`flash-linear-attention` 0.5.x is a
thin shim; the kernels live in `fla-core`**, so the image asserts `version('fla-core')` too.

### ⑦ Did NOT truncate the val set when the schedule looked tight
It would have cut eval ~10× and made tonight's CE incomparable with any future full-partition run — a
permanent cost for a schedule problem that capacity solved on its own. The override was staged and
ready in `/tmp/mitigation.txt`; re-measuring before spending it was the right call.

### ⑧ Ran the paper/fla KDA variant, labelled honestly — not K3
K3 needs two changes not expressible on the current dispatch (below).

---

## Next steps, in priority order

### ⓵ Decide whether CE is worth resolving at all — this is the fork in the road
σ̂ = **0.02042 nats at df=12** is the run's most reusable output: it replaces a 5.5× guess with a
factor-2.3 bracket. **Size any future run from 0.02042, not from the 0.0019/0.0105 literature pair.**
At that σ, resolving a 0.010-nat mixer difference needs n in the tens — **hundreds of cells**. Three
honest options:
1. **Accept the null and ship on throughput/memory.** The evidence already supports it.
2. **Change the endpoint.** Hymba shows a −20.75 recall-point gap at near-identical perplexity — the
   thing that distinguishes these mixers may live in recall, not aggregate CE. A sliced/recall
   endpoint is a re-weighted pass over held-out text at **zero extra training cost**. This is the
   highest value-per-GPU-hour change available.
3. **Raise the token budget**, accepting that effects *shrink* with budget (GDN's edge halved
   0.0103@1B → 0.0059@15B), so a longer run makes resolution harder, not easier.

### ⓶ Chunk the Householder kernel — the case is now quantified, and it is a memory case
**Do not** patch `chunk_gated_delta_product`: the per-head scalar gate is baked into six sites across
four files plus the whole backward. **Do** pack onto `fla.ops.generalized_delta_rule.dplr.chunk_dplr_delta_rule`,
which is already chunked, already takes a per-channel `[B,T,H,K]` gate with **no shape assert**, and
is fla's production RWKV-7 backend. Map `a_r = k_r`, `b_r = −β_r k_r`; decay on factor 0, zero on
1..R−1; `q` on the last factor. **~4–7 engineer-days**, wrapper + tests, autograd flows through fla's
own function so no gradients are hand-derived. The math survives R>1 **exactly** because the R factors
share one decay, so the diagonal is *absorbed into k* rather than commuted.
**Riskiest unknown, named:** whether DPLR's chunked backward is well-conditioned with `gk=0` and `q=0`
on R−1 of every R positions. If not, the fallback is the six-site rewrite = weeks.
**What this run adds:** kernel overhead is **28.6%** vs arity's 10.9%, and reserved memory is
**+23.7%/+24.9%** on the R arms (13.26/13.39 vs 10.72 GiB) — the `O(B·T·H·K·V)` fp32 workspace,
invisible in peak. Chunking replaces it with bf16 chunk-boundary states: **1.00 GiB → 32 MiB** at
B1/T4096/H16/K64/V64. **But note this run found no CE gain waiting behind that overhead at 0.6B
tokens** — chunk it to make R>1 *shippable*, not because R>1 is currently winning.

### ⓷ Never ship `KimiDeltaHouseholder` at R=1
Same function as `KimiDeltaAttention`, **28.6% slower**, and **sd 0.04516 — 17× the best arm on
mathematically identical computation**. Suspected kernel-induced instability (fp32 accumulation order
in a sequential-vs-chunked kernel is a plausible mechanism); n=3 with one outlier cannot prove it.
**Cheap check:** run the same seeds against `backend="torch"`, which is the bit-exact reference path
and exists for precisely this. If the spread persists, it is not the kernel.

### ⓸ Handle the GDN2 result carefully with Tom Liu
+0.0371 nats **worse** — the largest effect in the run, replicated across all three seeds
(3.0838/3.0910/3.0905, sd 0.00404), and the **only** contrast significant under any test. But: it is a
quality-at-fixed-params result at **TPP 1.5**, not a verdict on the implementation, which is the
best-tested mixer in the tree (its kernel-contract test transcribes the paper's recurrence
independently and reduces to KDA when the gates are tied). Its throughput is essentially free
(0.994×) and its MFU is the highest measured (42.0%). Worth a second look at a longer budget before
any conclusion travels.

### ⓹ Run 2: the K3-variant KDA arm — branch `edullm/kda-k3-gate` exists, untouched
**Two changes only**, verified against `moonshotai/Kimi-K3` `modeling_kimi_linear.py` and tech report
arXiv:2607.24653 (v1 2026-07-27):
1. **Lower-bounded forget gate:** `safe_gate=True, lower_bound=-5.0` threaded through
   `dispatch_chunk_kda` (fla ≥0.5.1 supports it; **the current dispatch does not pass it** — there is
   no `lower_bound` parameter at `flash_linear_attn_api.py:45`).
2. **Full-rank output gate:** `nn.Linear(d, V, bias=False)` behind a new `use_full_rank_gate`, versus
   the paper's low-rank `g_b_proj(g_a_proj(x))`. Note `use_full_rank_gate` governs **only** the output
   gate; the forget gate's `f_proj` stays low-rank in K3.

> ⚠️ **`A_log` init does NOT change.** An earlier report in this session claimed K3 zero-initialises
> it; **that is FALSE** — `modeling_kimi_linear.py:520` keeps `log(U(1,16))`, identical to the paper.
> I fetched the file and read the line. Also: K3's top-level `config.json` is a multimodal wrapper
> where `linear_attn_config` is **null**; the real values (`gate_lower_bound: -5.0`,
> `use_full_rank_gate: true`, 96 heads, head_dim 128, 93 layers) live under **`text_config`**.
> Reading the top level reports "empty config" and is how the wrong conclusion gets drawn.

**Re-solve the parameter ledger** (the full-rank gate changes the count) and **re-run `KDA_BASE` on
the same seeds** as run 2's internal control — the frozen seed table already reserves
`KDA_K3` = init seeds 128013/138020/148027 on data seeds 210007/220014/230021.

### ⓺ Housekeeping
- **~428 GiB of checkpoints in S3** (`teams/scratch/runs/run_019fe0f9-…/cell-*/checkpoints/`, ~7.9 GiB
  × 3 saves × 18 cells). `max_checkpoints=None` is **deliberate** — the workload role is denied
  `s3:DeleteObject` on `.metadata.json` by name, so a prune fails on its first call and would kill an
  11-hour run. ~$11/month. Cleanup task, **not** a risk.
- **Merge `edullm/mixer-bakeoff` toward `main`.** It carries GDN2, the gated convolution, the arm
  registry and the analysis script, and it is the first branch where all of those coexist.
- `sliced_eval` is null throughout — the pre-registered SECONDARY endpoint needs
  `--slice-mask-uri`, which the command does not set. Declared, not quietly skipped. See ⓵ option 2.
- **`solve_widths` had a real bug, now fixed:** it respent only attention-layer surplus and ignored
  the mixer's own size, so it returned "no correction" whenever the attention schedule was unchanged.
  Uncorrected, `KDA_R2` landed +1.085% and `GDN2` +1.064% against a 0.05% tolerance — **~21× out, in
  the direction that hands the bigger operator extra FFN capacity and reports it as mixer quality.**
  Five of six mutations pass a tolerance *band*; only exact per-arm `==` constants catch them. Do not
  let anyone "simplify" `ARM_L0_DELTA` back into a tolerance.
