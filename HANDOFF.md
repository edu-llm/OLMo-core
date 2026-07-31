# HANDOFF — DP2-KDA Phase 0/1 preparation

**Last updated:** 2026-07-31. **Status: runbook audited and corrected; DP2 source committed and
pushed; no test written, no GPU work started, no AWS resource created.**

**Scope:** this file covers **only** the DP2-KDA (strict-beta delta-product, R=2) experiment
program. Sibling handoffs are independent and still current — do not overwrite them:
`Capstone_LLM/HANDOFF.md` (LIV/LFM2 design), `Capstone_LLM/KDA/HANDOFF.md` (KDA-Householder,
COMPLETE 2026-07-26, no DP2 content), `Capstone_LLM/quant_research/HANDOFF.md`,
`Capstone_LLM/edullm-data/HANDOFF.md`.

**You are in a worktree.** `Capstone_LLM-worktrees/olmo-core/claude-01--dp2-kda-phase-0-prep`, on
branch `agent/claude-01/dp2-kda-phase-0-prep`, pushed to `origin`. Per `CLAUDE.md`, the canonical
checkout at `Capstone_LLM/OLMo-core` is an integration baseline on
`p4/interleaving-pretraining` — **do not switch its branch or commit to it.**

---

## Goal

Decide whether **strict-beta DP2-KDA** — two ordered delta-rule writes per real token after a single
channel-wise decay, giving a rank-2 state transition instead of rank-1 — earns its cost through
better long-memory behavior than ordinary KDA and fair controls.

The recurrence (state `S` is `[K,V]`, keys→values; `D_t = diag(exp(g_t))`):

```
S_{t,0} = D_t S_{t-1}
S_{t,j} = S_{t,j-1} + β_{t,j} k_{t,j} (v_{t,j}ᵀ − k_{t,j}ᵀ S_{t,j-1}),   j = 1,2
```

Two gates, both governed by `Capstone_LLM/docs/dp2-kda/phase-0-1-runbook.md`
(version `dp2_kda_phase_0_1_v4`):

- **Phase 0** — prove the implementation has the intended math, gradients, sequence-boundary
  behavior, and BF16 stability. One g6e.xlarge (L40S), ~$2/hr.
- **Phase 1** — fresh synthetic mechanism triage to decide whether a small-LM study is justified.
  One p5.48xlarge (8×H100), $55.04/hr.

Phase 2 is **not authorized** by that document (`docs/dp2-kda/phase-2-deferred.md`).

---

## Current Progress

### Done

1. **DP2 source committed and pushed.** Was entirely uncommitted before this session.
   - `6b75c06` — DP2 core, **exactly 7 files**, 3,999 insertions. This is the SHA P0.0 must name.
   - `55704ca` — 38 incidental worktree files that were already dirty. **Explicitly unreviewed**;
     recorded only to make the tree clean.
   - Both on `origin`. Verified via `git ls-remote`.

2. **Full audit of the runbook** (4 parallel auditors: code-state, math, statistics, ops/cost),
   then an **implement → review** pass (2 implementers, 2 adversarial reviewers). ~50 fixes applied.
   Evidence with every derivation and measured number:
   `docs/dp2-kda/audit-2026-07-31-verified-numbers.md`.

3. **The P1.4 gate was rebuilt** — it was effectively always-reject (a working DP2 passed 1.3–3.4%
   of the time). See Key Decisions.

### Not done — nothing has run

- **Zero of the 13 named Phase-0 tests exist** (0 grep hits against 61 existing test functions in
  the two gate files). P0.1, the R1-equivalence gate, is entirely unwritten.
- No GPU work, no image built, no AWS resource created.
- Phase-1 harness is largely unbuilt (see Next Steps).

---

## What Worked

- **Auditing claims against the code, not reading the document.** Nearly every real finding came
  from a shell command or 20 lines of arithmetic. Internal coherence is exactly what a wrong plan
  has.
- **Adversarial review of the fix pass.** The reviewers caught **8 defects the fixes introduced**,
  including three false claims that originated in the audit itself. Worth the cost; do it again.
- **Independently re-verifying every subagent claim before acting.** Multiple high-stakes subagent
  claims needed correction. Never relay an unverified finding.
- **Recomputing numerically rather than reasoning about algebra.** The rank-two oracle, tied-K's
  rank, the vacuous assertions, and the gate's power were all settled by throwaway scripts.
- **Two commits, DP2 separated from incidental drift.** Makes the Phase-0 source passport name one
  reviewable SHA.

---

## What Didn't Work

- **`EnterWorktree` cannot reach this worktree from an `edullm-data`-rooted session.** `OLMo-core`
  and `edullm-data` are *sibling* repos under `Capstone_LLM` (which is itself **not** a repo), so
  the tool correctly refuses. That is why this session was restarted here. Not a bug.
- **Trusting my own earlier conclusions.** Three claims I asserted were wrong and are corrected in
  the evidence file: (a) "the reflection update vanishes" — it does not, it is 5.75, a rank-1 write
  of `k(2v₂−2v₁)ᵀ`; `β_eff=0` makes `v_eff` a 0/0 form, degenerate not zero. (b) "flipping
  `GatedDeltaNet`'s default silently changes 10 production 7B runs" — false, all 10 pass
  `allow_neg_eigval=True` explicitly; the real default-reliant consumers are tests plus
  `gemma_like_ladder.py`. (c) "26 waves" — it is 27; calibration and R1-P confirmation cannot share
  a wave.
- **Fixing a vacuous check with another vacuous check.** Replaced `β₁+β₂ ≤ 1` with `β₁+β₂ ≤ 2`,
  which also can never fire (`2σ(ℓ) < 2` strictly). Only the *identity* `β₁+β₂ = b` does work.
- **Sizing a gate on power alone.** Left the equivalence test undecidable ~50% of the time.
  Decidability is the binding constraint, not power.
- **`grep -iE "swiglu|ffn"` on `probes/model.py` returning a hit.** It matched the docstring saying
  the model is *free of* an FFN. Read the match, not the exit code.

---

## Key Decisions

1. **P1.4 conditions 3 and 4 are confidence bounds, not point comparisons.**
   - Cond 3: `L₉₅(d) > 0` **and** `d̄ ≥ +5pp` (bound = the test; +5pp = separate practical floor).
   - Cond 4: a guardrail fails only if `U₉₅(d_g) < −2pp` — data must *affirmatively* show a loss.
   - **The "all differences nonnegative" sign clause was deleted.** Floor p of 1/8 at n=3, and it
     gets *harder* as n grows (0.78 → 0.66 → 0.51 for n=3,5,8), penalizing the very fix that makes
     the test valid.

2. **Seed count `n` is measured, not assumed (new §5.8.0).** Phase 1 launches in **two approvals**:
   calibration, then triage sized from it. σ_t falls out of P1.1's existing 80 jobs at **zero
   marginal cost**. Pick the smallest n with power ≥0.80 at +5pp **and** decidability ≥0.80
   (P(CI half-width < δ=3pp)). `σ_d > 5pp` → **do not launch**, write a feasibility memo.
   `n=3` appears nowhere: it yields 21% power and returns "underpowered" 78–99% of the time.

3. **T6 `overwrite_conflict` is a positive control, not a composite component.** Composite is
   **T1 + T7**. T6 is defined by "≥2 writes per key" and DP2's mechanism *is* two writes per token —
   a win there is close to definitional. Each component must also be individually nonnegative.
   Cost: σ_d rises by √(3/2) ≈ 1.22, so the formula is now `σ_t√(2(1−ρ))√((1+ρ_T)/2)`.

4. **Phase-0 tolerance is split by backend.** The 1e-11 float64 bar can only validate
   `backend="torch"` — Triton rejects float32 (`kda_householder.py:737-739`). Kernel-vs-reference is
   only comparable at bf16, where `ATOL=RTOL=2e-2` and a seeded cross-term bug slides under it for
   **~90% of seeds**. So: (i) float64 vs torch, (ii) triton-vs-torch with a per-tensor *relative*
   budget, (iii) a **mutation test** proving the bf16 check fails on a seeded bug. FP32 was struck
   from the 1e-11 clause — fp32 eps is 1.19e-7, making it unsatisfiable.

5. **tied-K is exactly rank 1 and R1-equivalent** (2nd singular value 2.26e-15;
   `β_eff = β₁+β₂−β₁β₂‖k‖²`, `v_eff` a convex combination). So §5.8's tied-K row was **merged into
   the stop-the-program row**, with "ties" = 90% CI within ±3pp.

6. **ρ is a free design variable.** `σ_d = σ_t√(2(1−ρ))·…`; ρ 0.5→0.85 is worth ~1.8× in σ_d and
   ~4× in n, bought by byte-identical data/task/bank streams across arms. **Buy ρ before buying
   seeds.**

7. **Cost.** Phase 0 $45–$134; Phase 1 $395–$530 at likely sizings. **Total ~$440–$665**, worst
   plausible ~$870. Wave overhead is a fixed 12 (smoke 1 + calibration 8 + confirmation 3) on top
   of `5n` triage waves. Prior probe runs: 145 runs, median 41.8s — seeds are cheap, **noise is the
   constraint**.

---

## Next Steps

### 1. Three P0.0 prerequisites — all verified blocking, none in the original plan

| # | Task | Why |
|---|---|---|
| a | **`git init` in `Capstone_LLM/probes/`** | Under **no version control** (verified: `git rev-parse` fails, no parent repo). It holds `naive_kda_householder.py`, the Phase-0 correctness oracle. The manifest's "probe source checksum" has no defined referent until this exists. |
| b | **Add `8.9`/`89` to `src/Dockerfile:55,59`** | Currently `TORCH_CUDA_ARCH_LIST="9.0 10.0"` and `FLASH_ATTN_CUDA_ARCHS="90;100"` — **no `sm_89`**, so the image cannot run the L40S that all of §4.1 specifies for Phase 0. |
| c | **Install `fla` in the image** | Absent locally (`import fla` → ModuleNotFoundError). 38 existing tests plus the new external anchor carry `@requires_fla` and would **skip** — the silent-green pathway §4.7 now forbids. |

### 2. ⚠️ Read this before running any test from this worktree

`probes/` lives at `Capstone_LLM/probes`, **outside** this worktree.
`src/test/nn/attention/kda_householder_test.py:34-58` loads its oracle via a
`Path(__file__).parents[5]/"probes"` fallback that **will not resolve from here**, and on failure
calls `pytest.skip()` — a skipped suite **exits 0**, so the gate reports green having verified
nothing. Set:

```bash
export KDA_PROBES_DIR=/Users/ericwu/Developer/Capstone_LLM/probes
```

This is the audit's highest-consequence operational finding, and from this worktree it fires for
real rather than latently.

### 3. Write the 13 named Phase-0 tests (§4.7) — the real Phase-0 effort

All absent. Note `test_kimi_delta_householder_r1_matches_kda_params`
(`recurrent_test.py:639-653`) is a **misleading neighbor** — it asserts only parameter count and
FLOPs, never copies weights, never compares outputs or gradients.

Traps the runbook now documents, each verified numerically:
- **§4.5's rank-two oracle is correct** — 4e-15 over 200 float64 trials; all three plausible
  corruptions fail by ~15 orders. Do not "simplify" any term.
- **A zero query NaNs the oracle.** `l2_normalize` (`functional/__init__.py:16-18`) is a bare
  `x/‖x‖`, no epsilon, applied at `recurrent.py:1299` — *after* §4.4's injection point. Inject
  after normalization or use a unit dummy query and discard its output.
- **Negative controls need asserted separation floors.** Factor-order-swap separation scales as
  `O(K^{-1/2})` and is **0.0** at orthogonal keys; the `v₂=0` control is 0.0 when `S_prev=0`.
- **Final-state comparison is blind to readout position** — `S` is invariant to which microstep you
  read; outputs differ median 11%, max 46%. Compare `o`, not just `S`.
- **P0.5 must assert zero skips**, with `test_context_parallel_gdn_ulysses` deselected by name
  (`@requires_multi_gpu`, always skips on a 1-GPU node).

### 4. Phase-1 harness (largest block; all in `Capstone_LLM/probes/`, not this worktree)

| Task | Note |
|---|---|
| Add SwiGLU FFN to `ProbeModel` | **Blocks R1-P**, the primary comparator. Confirmed absent — docstring says "free of an FFN". |
| Per-factor beta parameterization | New code, not a flag flip: beta is computed once for all R factors (`recurrent.py:1257-1259`). |
| Fix `train_probe.py:34,36,48` | The 3-line strict-beta fix. **Do not touch `recurrent.py`.** |
| `--manifest` interface | None today; `:80-96` is free-form argparse, so the canonical invocation cannot be typed. |
| Four-way seed plumbing | One `--seed` today (`:84`), eval derives from it (`:65`) — §5.3's no-collision requirement is currently *unsatisfiable*. |
| `build_dp2_eval_bank.py`, `analyze_dp2_phase1.py` | Do not exist. |
| Implement T6, T7, T8 + generators | Do not exist; T7 is in the composite. |
| **Declare the difficulty grid** | Still undefined — calibration's wave count is uncomputable without it. |
| De-duplicate the MQAR ladder | `mqar_d16` is byte-identical to `mqar_p16` (`tasks.py:340-344` vs `:325-329`). |

### 5. Open items I did not do

- **No PR opened.** Branch is pushed; GitHub offered the link.
- **`docs/dp2-kda/README.md` still says `Reflection / EDA2`** where the runbook says `Reflection`,
  and `EDA2` is undefined anywhere in the doc set.
- Your uncommitted `CLAUDE.md` edit (the worktree convention) sits on the canonical checkout,
  untouched. It is also inside `55704ca` here.
- **`claude-01` is already used as an agent-id in the `edullm-data` repo** (worktree
  `claude-01--reservoir-ingest`). Different repo, so not a collision, but pick a fresh id if you
  spawn more agents.

---

## Governing documents

| Path | What |
|---|---|
| `Capstone_LLM/docs/dp2-kda/phase-0-1-runbook.md` | **The plan.** v4, audited and corrected. |
| `Capstone_LLM/docs/dp2-kda/audit-2026-07-31-verified-numbers.md` | Every derivation and measured number behind v4, including corrections to the audit's own errors. |
| `Capstone_LLM/docs/dp2-kda/README.md` | Program index, arm names, roles. |
| `Capstone_LLM/docs/dp2-kda/aws-operations.md` | Instance shapes, cost equation, **mandatory pre-launch checklist**. |
| `Capstone_LLM/docs/dp2-kda/phase-2-deferred.md` | Deferred; entry needs a signed P1.4. |
| `~/.claude/skills/check-plan/SKILL.md` | The generalized plan-audit method distilled from this work. |
