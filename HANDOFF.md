# HANDOFF — DP2-KDA Phase 0/1 preparation

**Last updated:** 2026-08-01. **Status: runbook audited and corrected; DP2 source committed and
pushed; all three P0.0 prerequisites closed; all 13 named Phase-0 tests green on an L40S via
FarmShare. `origin/main` merged — the program is now on the eduLLM platform path, not Slurm. No
image built, no AWS resource created, no money spent.**

## ⚠️ FIRST RUN: the arms are `Reflection` and `R1-P` — NOT `DP2-strict`

**Decided 2026-08-01.** The first platform training run is the pair **`Reflection`** (treatment)
and **`R1-P`** (control). `DP2-strict` is explicitly *not* the first run, which reverses this
document's original framing, so the reasons are recorded here rather than left to inference:

1. **Strict-beta DP2 is structurally barred from the primary state-tracking task.** With keys
   L2-normalized (`recurrent.py:1299-1300`) each erase factor has `det(I − βkkᵀ) = 1−β` *exactly*
   (verified to 12 decimals). Strict β ∈ (0,1) ⟹ every factor has positive determinant ⟹ the
   composed state map has **det > 0 at every R** — measured 0 of 4000 trials negative at R=1,2,4.
   `s5_words` requires odd permutations, `det = −1`. Adding factors cannot escape this: it is
   closed under composition. Reflection β ∈ (0,2) crosses zero ~50% of the time.
2. **Reflection is the only arm with measured signal.** 0.978 vs 0.598–0.640 for every strict arm
   at L40; `R1-P|Reflection|64` = 56.5 vs **95.9** (`docs/dp2-kda/evidence/`). It also *trains* β
   to 1.92–1.98 — it uses the negative range rather than merely being permitted it.
3. **`R1-P`, not `R1`, is the honest control.** It spends the same parameter budget in FFN width,
   separating mechanism from capacity. FFN width alone buys +1.9pp; without `R1-P` the program
   records a spurious win.
4. **The LM null is what a real run can actually resolve.** The one parameter-matched LM pair was
   **+0.0053 nats** — a clean null. Whether probe effects survive at LM scale is precisely what
   free FarmShare probe work *cannot* answer.

**Caveat, stated plainly:** the reflection numbers are n=1 harness smoke at 150–1000 steps, not
calibration. They justify *choosing* reflection over strict for a first run; they do not establish
that reflection works. Note also that §3.2 of the runbook forbids *pooling* strict and reflection
results — running reflection first is not pooling, but any writeup must keep the regimes separate.

Sizing: `olmo-core-check-cpu` first (~$1.43, proves the path), then the pair on
`olmo-core-train-1gpu` ($1.01/hr). Not 4-GPU — the launcher and checkpoint guards are easier to get
wrong there, and $136 worst case vs $12 is the wrong place to debug a command string.

**Blocker for any DP2/KDA run:** the mixer is **not reachable from the command line.**
`--model-factory` resolves attributes on `TransformerConfig` and there is no KDA/Householder
factory there. Dot-notation also fails — `block.sequence_mixer.type=kimi_delta_householder` raises
`OLMoConfigurationError: class 'AttentionConfig' has no attribute 'num_householder'`, because
`merge()` applies overrides against the existing `AttentionConfig` without re-resolving the type,
even as a separate merge step. The working path is to assemble in a script:
`replace(cfg.block, sequence_mixer=KimiDeltaHouseholderConfig(...))` — verified to give a 269.6M
model at R=2 with strict beta. This is the guide's "level two starts from a copy". No library edit
is needed: `kimi_delta_householder` is already a registered mixer name.

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

- **No image built** — the sm_89 edit is unproven until one builds. Note the DP2 gate tests do
  **not** depend on it: `kda_householder.py` is Triton JIT (compiled at runtime for whatever GPU
  is present) and the gate files reference neither `flash_attn` nor `grouped_gemm`. The two edited
  lines govern exactly those two packages. So the image is needed for P0.0 provenance and Phase-1
  carry-through, not to make P0.1–P0.5 executable.
- No AWS resource, no spend.
- Phase-1 harness is largely unbuilt (see Next Steps) — **this is now the top of the queue.**

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

### 1. Three P0.0 prerequisites — ✅ ALL THREE CLOSED 2026-07-31

| # | Task | Outcome |
|---|---|---|
| a | **`git init` in `Capstone_LLM/probes/`** | ✅ **Done.** Now a standalone repo; baseline commit `93b60d7`, 34 files (32 `.py`, 1 `.sh`, 1 new `.gitignore` excluding `__pycache__`/`*.pyc`), contents as-is, no functional edits. Runbook §4.2 step 3a updated: **the probe source checksum is the VCS-commit form, `93b60d7`.** ⚠️ **No remote** — local-only, so the SHA is not independently fetchable. Push it somewhere durable before Phase 1. |
| b | **Add `8.9`/`89` to `src/Dockerfile:55,59`** | ✅ **Done** in `b5433c0`. Now `TORCH_CUDA_ARCH_LIST="8.9 9.0 10.0"` and `FLASH_ATTN_CUDA_ARCHS="89;90;100"`; sm_90/sm_100 unchanged. `TORCH_CUDA_ARCH_LIST` is **hardcoded, not a build arg** (Makefile passes 12 others but not this), so the Dockerfile was the right place. Costs a longer build and a larger image. **Unbuilt — the edit is not yet proven to compile.** |
| c | **Install `fla` in the image** | ✅ **No action needed — the premise was wrong.** See below. |

**(c) was a false alarm, and its correction is now in the runbook.** `src/Dockerfile:106` installs
`'.[all]'`, and `pyproject.toml:73` expands `all` to include the `fla` extra —
`flash-linear-attention==0.4.1` (`:69`), no platform marker. So `fla` **already reaches the image**.
Verified at that pin: both anchor symbols exist
(`fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product` and
`fla.ops.kda.naive.naive_recurrent_kda`), and the dependency closure is only `fla-core==0.4.1` →
`torch`, `einops`, both unpinned, so it cannot fight the image's pinned torch.

The original `ModuleNotFoundError` was observed **on this macOS laptop**, which has no
`ai2-olmo-core` install and no `triton` at all (Triton ships Linux-only wheels), so it cannot host
`fla` regardless — that result says nothing about the Linux CUDA image. Also worth knowing:
`FLA_MARKS` (`testing/utils.py:137-140`) pairs the skipif with `pytest.mark.gpu`, so these tests
skip on any CPU-only host for a **second, independent** reason. P0.0's job here is **verify and
record, not install**: on the built image assert `import fla` succeeds and record the version; if it
fails, the `all` extra has regressed and *that* is the finding. (Also: the count is **58**
`@requires_fla` decorators, not 38 — 78 total `requires_fla` mentions.)

### 2. ⚠️ `KDA_PROBES_DIR` — now pinned, but only for Claude Code sessions

`probes/` lives at `Capstone_LLM/probes`, **outside** this worktree.
`src/test/nn/attention/kda_householder_test.py:34-58` loads its oracle via a
`Path(__file__).parents[5]/"probes"` fallback that **will not resolve from here** — it points at
`Capstone_LLM-worktrees/olmo-core/probes`, which does not exist (verified) — and on failure calls
`pytest.skip()`. A skipped suite **exits 0**, so the gate reports green having verified nothing.

Now set in `.claude/settings.local.json` (gitignored, so the machine-specific absolute path stays
out of the shared repo — do **not** promote it to `.claude/settings.json`). Verified: the oracle
imports cleanly with it set.

**This only covers Claude Code sessions.** A plain terminal, CI, or any hook-spawned process still
needs it explicitly:

```bash
export KDA_PROBES_DIR=/Users/ericwu/Developer/Capstone_LLM/probes
```

The durable fix is to widen the test's own path resolution (it has a two-candidate list; a sibling
`../../Capstone_LLM/probes` candidate would resolve from any worktree) or, better, to make the
oracle-missing case **fail** rather than skip. Not done — it edits a file inside `6b75c06`.

### 3. ✅ DONE — the 13 named Phase-0 tests (§4.7), commit `4f747f5`

**All 13 written; 156 passed / 0 skipped / 1 sanctioned deselect on an NVIDIA L40S.** All four
§4.7 skip-discipline assertions pass. `make style-check`/`lint-check` clean; mypy clean for these
files (2 remaining errors are pre-existing in `kda_householder.py:595` and `hf/config.py:114` —
no production source was touched).

**FarmShare has the Phase-0 GPU, free.** This is the session's most reusable finding. The `oat`
partition (6 nodes × 4 GPUs) is **NVIDIA L40S, compute capability 8.9** — exactly §4.1's Phase-0
target. Allocation took seconds. Working tree:

```
/scratch/users/ericrcwu/agent-runs/dp2-kda-p0/{OLMo-core, probes, venv, results}
srun --ntasks=1 --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --exclude=wheat-01
```

The venv layers `pytest` + `pip install -e .` onto the prior `kda-phase0` env via a `.pth` file,
so the original is unmutated. Stack: python 3.12.3, torch 2.9.1+cu128, triton 3.5.1, **fla 0.5.1**.

⚠️ **This is not a P0.5 pass.** P0.0 needs the pinned image and its digest, and FarmShare runs
**fla 0.5.1 against the repo's pinned 0.4.1**. Re-verify the external anchor on the pinned image
specifically — its conventions were measured against 0.5.1.

**Two runbook claims were wrong and are now corrected in place:**

1. **fla's `naive_recurrent_gated_delta_product` ignores its own `scale` argument** (never applied
   to `q`; `scale=1.0` and `scale=K**-0.5` give byte-identical output) **and casts inputs to
   float32** internally. So "agrees to float64 ulp" is unattainable by construction. Written to the
   original spec, the anchor test fails at **0.82 relative** on correct code. It now compares at
   `scale=1.0` against a float32 floor computed in-test (measured 6.4e-8 vs floor 1.0e-7).
2. **`BF16_RTOL` (1e-5) is not a standalone bound** — it's the relative half of an `assert_close`
   pair whose absolute half is 5e-3. Module parity uses 2e-2 (realized: max 8.6e-3, median 5.4e-4).

Traps the runbook documents, all confirmed numerically while implementing:
- **§4.5's rank-two oracle is correct** — reproduced at **1.4e-16**; all three corruptions
  (`drop_rho`, `rho_on_u2`, `plain_k`) fail by ~**1e15×**. Do not "simplify" any term.
- **Negative controls need asserted floors.** Both zero-difference regimes were reproduced
  exactly: the `v₂=0` control gives **0.000e+00** when `S_prev=0`, and factor-order separation is
  0.0 at orthogonal keys. The swap keys are now built with an *exact* inner product
  (`_tied_angle_key`) so `|k₁·k₂|` is assertable rather than luck-of-the-draw.
- **Compare `o`, not just `S`** — `S` is invariant to readout microstep.
- **The zero-query NaN is module-level only.** `l2_normalize` NaNs on zeros, but the operator is
  *downstream* of it — a zero query there yields a zero output, not NaN. Verified: substituting a
  zero dummy does **not** fail the operator tests. The unit dummy is kept for reuse safety, and
  the docstring says plainly that it is not load-bearing at this level. Do not cite this as a
  passing check that the NaN is handled.

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
- ✅ `docs/dp2-kda/README.md` `Reflection / EDA2` → `Reflection`, matching the runbook.
- **`Capstone_LLM/docs/` is unversioned** — same class of problem as `probes/` was. Three
  governing documents were edited across these two sessions with no history and no way to diff.
  The runbook alone is 84 KB of audited reasoning. Worth a `git init`.
- **`probes/` has no remote.** `93b60d7` is the manifest's checksum but is local-only, so it is
  not independently fetchable. Push it somewhere durable before Phase 1.
- **The `_load_oracle` fail-open is still fail-open** (`kda_householder_test.py:34-58`). Mitigated
  operationally (`KDA_PROBES_DIR` set everywhere + zero-skip assertion), not fixed — the fix edits
  a file inside `6b75c06`.
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
