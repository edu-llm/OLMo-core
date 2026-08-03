# HANDOFF — DP2-KDA / KDA-Householder

**Last updated:** 2026-08-03.

**Status in one paragraph.** Phase 0 is done and green. Three independent audits then concluded the
program has **no publishable novelty as framed** and recommended terminating the DP2 follow-on. The
work pivoted to (a) a KDA write-up, (b) one cheap decisive experiment (**A5**), and (c) migrating
everything to AWS per a team mandate. The A5 platform submission is **built, imaged, and
pre-validated at $1.01 — not submitted.** Nothing has run on AWS; no money spent.

> **⚠️ YOU ARE ON A DIFFERENT BRANCH THAN THIS FILE'S HISTORY SUGGESTS.**
> Current branch is **`edullm/a5-solvability`** at `bd2ede9`, pushed and in sync. The old
> `agent/claude-01/dp2-kda-phase-0-prep` still exists on `origin` at `320495a` and is **3 commits
> behind**. The rename was mandatory: the image build only fires on `edullm/**` or `main`, and an
> `agent/**` branch pushes green while **publishing no image**.

---

## Goal

Originally: decide whether **strict-beta DP2-KDA** — two ordered delta-rule writes per token after
one channel-wise decay, giving a rank-2 state transition — beats ordinary KDA and fair controls.

```
S_{t,0} = D_t S_{t-1}
S_{t,j} = S_{t,j-1} + β_{t,j} k_{t,j} (v_{t,j}ᵀ − k_{t,j}ᵀ S_{t,j-1}),   j = 1..R
```

**That question is now answered in the negative, on theory grounds** (see Key Decisions #1). The
live goals are: run A5 to separate two competing explanations, and salvage the one publishable unit.

---

## Current Progress

### Done and verified

1. **All 13 named Phase-0 tests (§4.7) written and green** — `4f747f5`. **157 passed / 0 skipped /
   1 sanctioned deselect** on an NVIDIA L40S. §4.7's four skip-discipline assertions all pass.
2. **`l2_normalize` epsilon fix** — `320495a`, with CHANGELOG entry. **PR not opened**;
   `guides/olmo-core.md:13` requires one for `src/olmo_core/` changes.
3. **Two test holes closed** — `92a2bed` (oracle fail-open), `0bdf578` (GVA `repeat_interleave`).
4. **`probes/` and `docs/dp2-kda/` vendored into the repo** — `8107642`. Both were previously
   outside any remote.
5. **Review evidence preserved** — `08717e6`, 192 files (155 run records + 3 aggregates + 9 scripts
   + logs) rescued from purgeable FarmShare scratch.
6. **KDA write-up drafted** — `f251d58`, `docs/kda-householder/`, ~12.5k words, 30 TSVs, 11 scripts.
7. **`origin/main` merged** — `caa5021`. Delivered `.edullm/train_on_corpus.py`, the Dockerfile, and
   the build workflow.
8. **A5 task added** — `probes/` commit `ea42f54`. Verified on FarmShare: closure order exactly 60,
   both generators even, all 60 elements even, proper subset of S5, targets fill `[0,59]`.
9. **A5 platform adapter** — `01713e4`, `.edullm/probe_group_words.py`. End-to-end on an L40S: A5
   trains (loss 4.27→1.14), β_max 1.996, record written, **8 seconds**.
10. **132 ruff violations cleared** — `bd2ede9`. `ruff check --no-cache .` at repo root now passes.
11. **Image built and published.** Run `30772225118`, all three jobs green.
    **Digest `sha256:d79cfa6db767d17b098b534863fc7549254e587bedce90fc396237ff591166fa`.**
12. **Pre-validated offline against the real commit and real digest** — compiled clean, **$1.01**
    worst case, ROUTINE, manifest SHA `sha256:facec0dd11fb…`.

### Not done

- **A5 not submitted.** Waiting on: the ~7-minute ECR security scan after the build, then a lead's
  release. Nobody is paged — you must ask.
- No AWS run has executed. No spend.
- The `l2_normalize` PR is unopened.
- `docs/dp2-kda/` is not formally closed despite three recommendations to terminate.

---

## The A5 submission — everything needed to fire it

```
Repository        OLMo-core
Commit            bd2ede993c927f9588f334bfb3d7234a8e2a16fe
Image digest      sha256:d79cfa6db767d17b098b534863fc7549254e587bedce90fc396237ff591166fa
Workload profile  olmo-core-check      (1h, 1 attempt, NO checkpoint contract)
Compute profile   gpu-1xa10g           ($1.006/hr, provisioned: true)
Team              scratch
Experiment        a5-solvability-vs-arity
Dataset release   none                 (synthetic task — no corpus)
Submitter         ericrcwu001          (NOT "ericrcwu" — that fails submitter_not_in_roster)
Worst case        $1.01
```

```
bash -lc 'python .edullm/probe_group_words.py "$EDULLM_RUN_ID" --task a5_words --arm Reflection --bundle-id 1101 --output-dir "$EDULLM_OUTPUT_PREFIX" --steps 4000'
```

Pre-validation inputs are at `/tmp/pv/{inputs.json,images.json}` (recreate if gone; `command` is an
**argv list**, not a string, and `commit_sha` must be full 40-hex).

**Consider `--steps 300` instead of 4000.** The FarmShare smoke converged in 300 steps / 8 seconds.
Same $1.01 ceiling, more headroom under the 1-hour limit.

### What A5 is for

Every group tested so far is equally consistent with two explanations of why `s5_words` is hard:
**solvability** (S5 is non-solvable) or **arity** (R Householder factors can only reach so far). A5
is non-solvable but order 60 with both generators even. DeltaProduct (arXiv:2502.10297v7 §5) settles
it for a per-**head** gate, verbatim: *"Unexpectedly, S₄ and A₅ can extrapolate robustly using only
n_h = 2."* Whether that carries to a per-**channel** gate is unmeasured. That is the experiment.

---

## What Worked

- **Verifying every subagent claim before relaying it.** This caught: a claimed "runbook
  contradiction" that was wrong (line numbers were right); a claimed "only one bare test" that was
  15; my own recomputation of a contrast that was 5× off because I pooled across geometries.
- **AST fingerprints as a behavioral contract for style-only edits.** Captured
  `ast.dump(..., include_attributes=False)` for 23 files *before* letting three agents loose, then
  diffed after. Made "did you change behavior?" mechanically checkable on scripts with no tests.
- **Extracting paper text on FarmShare rather than trusting a citation.** `pypdf` on the
  DeltaProduct PDF produced the verbatim A₅ sentence that demoted a headline claim.
- **Splitting agents by directory with explicit "the other two exist" notes.** Zero write conflicts
  across three concurrent agents on 132 violations.
- **Running the exact command CI runs.** `ruff check --no-cache .` at the repo root. Slice-level
  checks are what let 132 errors reach the build.

---

## What Didn't Work

- **Scoping style checks to hand-edited files.** I ran `ruff check <file>` all session; CI runs
  `ruff check .`. That single mismatch cost a failed image build.
- **My `FLASH_ATTN_CUDA_ARCHS="89"` edit was worse than a no-op.** flash-attn 2.8.2 tests four
  hardcoded literals — `"80"`, `"90"`, `"100"`, `"120"` — and `"89"` matches none. Upstream's
  default is already `"80;90;100;120"`, so **my edit removed sm_89 coverage while appearing to add
  it.** Fixed to `"80"` and verified with `cuobjdump`.
- **Repeating `+0.0053 nats` as "your LM result."** It is DeltaProduct's Table 5 figure, imported as
  a power prior and misread. The actual measured result is **+0.0357 nats, 95% CI [+0.0198,
  +0.0516], p=0.0056** — significant *against* R=4. The write-up caught this at `:1064`; I read that
  and still repeated the wrong number.
- **`git merge-tree` as proof of a clean merge.** It reports *content* conflicts. The real merge hit
  three **modify/delete** conflicts it cannot see.
- **Telling agents "black/isort must pass."** Unsatisfiable: 49 files fail at baseline, 36 on
  `main`. Two agents independently measured this and told me.
- **Copying `probes/` instead of moving it.** The two copies diverged within hours — the vendored
  one lacked `a5_words`, so an image built from that commit would have had no A5 task.
- **One agent used `git stash` while two siblings had in-flight edits.** Nothing was lost, but it
  briefly stashed their work. Tell agents not to.

---

## Key Decisions

1. **Strict-beta DP2 cannot do state tracking — proven, not measured.** With L2-normalized keys
   (`recurrent.py:1299-1300`) every erase factor has `det(I − βkkᵀ) = 1−β` **exactly** (12 decimals).
   Strict β ∈ (0,1) ⟹ det > 0 at every R — **0 of 4000 trials negative** at R=1,2,4. Odd
   permutations need det = −1. Closed under composition, so more factors cannot help. Reflection
   β ∈ (0,2) crosses zero ~50%. **⚠️ This result is *dominated* by Grazzi et al. arXiv:2411.12537
   Proposition 1 item 3**, which forbids *every* non-identity permutation at all k — strictly
   stronger. Not novel.
2. **First run arms are `Reflection` + `R1-P`, not `DP2-strict`.** Follows from #1: strict is barred
   from the primary task. Reflection is the only arm with measured signal (0.978 vs 0.598–0.640 at
   L40) and *trains* β to 1.92–1.98. `R1-P` not `R1` is the control, since FFN width alone buys
   +1.9pp. **Caveat:** those are n=1 smoke runs, not calibration.
3. **Use fla's kernel, not this one.** Measured: 6.9–9.5× slower, ~10× the memory vs `chunk_kda` at
   R=1 — ~127 extra GPU-hours per 2T-token 1B run. The old "406× faster" compared against the
   author's own Python loop.
4. **`l2_normalize` fixed in the shared function (Option A), not locally.** Measured bit-identical
   for real inputs (`max|before − after| = 0.000e+00`, float32 and bf16), and the three other
   callers (`feed_forward.py:283`, `lm_head.py:568`, `layer_norm.py:386`) had the same latent bug —
   the two weight-normalization sites were silently corrupting rather than raising.
5. **The evidence archive is fixed by hand, not by `black`.** `docs/dp2-kda/evidence/` must match
   what produced the 158 records; black would rewrite 166 lines in a 143-line file.
6. **A5 goes through the platform despite being a 3-minute job.** I recommended FarmShare
   (free, no approval); the user cited a **team mandate that everything be on AWS**. Mandate wins.

---

## Next Steps

### 1. Submit A5 (blocked on two things, both external)

- Wait for the ECR **security scan** (~7 min after the build; it completed 2026-08-02 ~23:00 UTC, so
  this is almost certainly clear now). Submitting early gives `image_scan_findings_unreviewed`, which
  reads like a vulnerability problem but usually means the scan hasn't finished.
- Get **a lead to release it**. Any team lead can; `scratch` has none recorded.

Then `gh workflow run submit-run.yml --repo edu-llm/platform` with the fields above. Report the
`run_019f…` id back — it is the Batch job name, S3 folder, and W&B run name.

### 2. Decide the program's fate

Three independent audits say the same thing. The consolidated recommendation:

- **No paper about the operator.** fla's `generalized_delta_rule.dplr` computes it **exactly**
  (verified 1.11e-16 at R=1–4, `gk` documented `[B,T,H,K]` per-channel). **EDA** (arXiv:2606.26560,
  18 Qwen/Alibaba authors, verified) publishes essentially this at 2.5B/25B. **DeltaProduct
  Appendix B.4 Eq. 6** already writes per-channel `diag(w)` inside the product at arbitrary R.
- **The one publishable unit:** the real LM result (+0.0357 nats, p=0.0056) plus the §7.2
  methodological lesson — an endpoint can be **vacuous** rather than underpowered (MDE 0.010–0.018
  nats vs arm differences ≤0.003). **TMLR**, which does not require novelty. ~4 weeks.
- **Terminate `docs/dp2-kda/`.** Premise is strict β, ruled out by #1. Its own gate fires ~3% of the
  time on a genuinely good arm, at $300–2,800.

### 3. Open engineering items, roughly by value

| Item | Note |
|---|---|
| **Open the `l2_normalize` PR** | Required by `guides/olmo-core.md:13`. CHANGELOG entry already in `320495a`. |
| **TBPTT gradients wrong by 29–39%** | Nothing raises. Live bug from the mixer audit. |
| **0.12-token memory half-life** (`recurrent.py:1378` `dt_bias.zero_()`) | ~90× shorter than fla's. **Every probe run so far trained under this** — may affect how A5 is interpreted. |
| **Double-backward returns a finite wrong HVP** at module level (rel 0.9962) | The `once_differentiable` docstring is false where users call it. |
| **Unguarded OOM cliff** | `hs_GiB = 4·B·T·H·K·V/2³⁰`, R-independent; at B8/T4096/H16/K128/V128 total is 48.31 GiB and `hs` is 66% of it. The comment at `kda_householder.py:534-535` credits it all to `hs`. |
| **`probes/` two-copy divergence** | Vendored copy + `Capstone_LLM/probes` (repo `ea42f54`, **no remote**). Already diverged once. Make one canonical. |
| Three lint-pass bugs, reported not fixed | `analyze_sigma.py` t-crit fallback too small for df∈{16-19,21-29,31} (headline n=12/df=11 used the exact tabulated value, so **no reported number is affected**); `mutate_kernel.py` leaks every mutated Triton module (`sys.modules[mid.lower()]` vs registered `f"kh_{tag}"`); `depth_ladder_check.py` unguarded `st.mean(hs)` where `horizon_depth.py:16` guards it. |
| `docs/` still unversioned | `Capstone_LLM/docs/` has no git history; the runbook is 101 KB of audited reasoning edited in place by several agents. |

---

## Platform facts the guide gets wrong (verified by running it)

| Guide says | Reality |
|---|---|
| `olmo-core-check-cpu`, `olmo-core-train-1gpu` | **`olmo-core-check`**, **`olmo-core-train`**. `workload-catalog.yaml:163-174` documents the rename chain. |
| Leave `compute_profile` alone | **Required.** The compiler refuses without it. Profiles no longer name a machine. |
| `olmo-core-train` is 12h | **24h, 2 attempts.** |
| Build runs `ruff check .` over your checkout | Step is named *"Run research repository tests"* and runs `${TEST_COMMAND}`. Ruff is what failed; whether tests also run is unconfirmed. |
| Workflow comment: "black and isort pass today" | **False** — 49 files fail on this branch, 36 on `main`. Only ruff gates the build. |

Config **churns** (332 commits in 8 days). Always `git -C /tmp/edullm-platform pull` and read
`.github/workflows/submit-run.yml` live before trusting any dropdown value.

---

## Environment

**FarmShare** (free, zero marginal cost — do not ration it, and do not frame its wall-clock as a
cost). GPU partition `oat-*` is **NVIDIA L40S sm_89**, the exact Phase-0 target.

```bash
ssh -S /tmp/farmshare-ericrcwu.sock -o BatchMode=yes ericrcwu@login.farmshare.stanford.edu 'CMD'
srun --ntasks=1 --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:25:00 --exclude=wheat-01 bash -s
```

`--exclude=wheat-01` is mandatory. Omitting `--ntasks=1` duplicates output. `/tmp` is **not** shared
between nodes — stage in `/scratch/users/ericrcwu/`. The socket expires (Kerberos); re-auth with
`ssh -M -S /tmp/farmshare-ericrcwu.sock -o ControlPersist=yes …`.

Working tree `/scratch/users/ericrcwu/agent-runs/dp2-kda-p0/{OLMo-core,probes,venv}` — python
3.12.3, torch 2.9.1+cu128, triton 3.5.1, **fla 0.5.1** (repo pins 0.4.1; conventions verified
bit-identical across both). Sync with tar **including `.git`**, then delete macOS `._*` files.

**The user was explicit: run all code on FarmShare, never on the local Mac.**

---

## Governing documents

| Path | What |
|---|---|
| `docs/dp2-kda/phase-0-1-runbook.md` | The plan, v4. **Recommended for termination.** |
| `docs/dp2-kda/audit-2026-07-31-verified-numbers.md` | Derivations behind v4. |
| `docs/dp2-kda/evidence/` | 192 preserved files + `README.md` with the stratification warning (**pooling across geometries inflates the contrast to +16.4pp — I made that mistake**). |
| `docs/kda-householder/kda-householder.md` | The write-up draft. |
| `Capstone_LLM/KDA/HANDOFF.md` | Sibling project, science COMPLETE, write-up not. Has a 2026-08-01 correction block. |
| `.claude/skills/edullm-platform-runs/` | Platform skill; `references/prevalidate.md` is the offline compiler. |
| `/tmp/edullm-platform/guides/olmo-core.md` | Training guide. See the corrections table above. |
