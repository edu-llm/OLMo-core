# Phase 0–1 DP2-KDA execution runbook

## Material Passport

- **Origin skill:** academic-research-suite / experiment-agent
- **Origin mode:** plan
- **Origin date:** 2026-07-31
- **Verification status:** DESIGN REVIEWED; audited against live code and recomputed numerically on 2026-07-31, then corrected in an implement-and-review pass. Phase-0 tolerance policy is now split by backend, oracle independence requires an external anchor, and every negative control carries an asserted separation floor. The P1.4 gate was rebuilt: conditions 3 and 4 are confidence-bound tests rather than point comparisons, and the triage seed count \(n\) is **derived from measured seed variance** (§5.8.0) rather than fixed in advance — so Phase 1 launches in two approvals, calibration then triage.
- **Version label:** dp2_kda_phase_0_1_v4
- **Canonical index:** [README](README.md)
- **Cloud operations:** [AWS operations and cost guardrails](aws-operations.md)

## 1. Objective and scope

This runbook executes only two gates:

1. **Phase 0:** prove that R1 and strict DP2 have the intended mathematics, gradients, sequence-boundary behavior, and BF16 stability.
2. **Phase 1:** run fresh, synthetic, strict-beta mechanism triage to decide whether a small language-model study is justified.

The Phase-2 small-LM plan is not authorized by this document. It becomes eligible only under the P1.4 decision rule and is specified separately in [phase-2-deferred.md](phase-2-deferred.md).

### Primary hypothesis

At matched model geometry and controlled write strength, strict DP2 improves a preregistered long-memory composite relative to R1-P without harming local behavior.

### Primary nonclaims

- No Phase-0/1 result is a K3-faithful result.
- No Phase-0/1 result is a target-hybrid or target-model result.
- No Phase-0/1 result establishes target-scale GPU training efficiency.
- No reflection result is pooled with strict-beta evidence.
- No small effect is treated as established from a seed count that cannot resolve it. The triage seed count is derived from measured variance (§5.8.0), never assumed; a gate whose power at the claimed effect size has not been computed is not a gate.

## 2. Current code state and hard blockers

| Item | Current location | Consequence |
|---|---|---|
| Native R-factor KDA substrate | OLMo-core/src/olmo_core/nn/attention/recurrent.py | R=2 exists, but its current gate/output path is local Kimi Linear-style rather than target/K3-confirmed. |
| Fast implementation | OLMo-core/src/olmo_core/nn/attention/kda_householder.py | Correctness needs fresh independent-oracle coverage before new experiments. |
| Differentiable reference | OLMo-core/src/olmo_core/nn/attention/kda_householder_torch.py | Use as one oracle; do not let it become the only oracle. It is a transcription of probes/naive_kda_householder with the same einsums in the same order, so it and the probe count as **one** oracle, not two — see §4.5's required external anchor. |
| Existing probe harness | probes/train_probe.py and probes/model.py | Hard-codes reflection beta and lacks R1-P, R1-2step-tiedK, DP2-budgeted, per-run manifests, and fixed evaluation banks. |
| Existing probe model | probes/model.py | Has no FFN, so a same-\(d_{\rm model}\) FFN-width R1-P control does not yet exist. |
| Existing LM harness | KDA/lm/train_lm.py | Deferred to Phase 2; do not use as Phase-1 evidence. |
| Source-control state | OLMo-core branch `agent/claude-01/dp2-kda-phase-0-prep`, **pushed to `origin`** | **Resolved 2026-07-31.** DP2 is committed at `6b75c06` (exactly the 7 files below); the 38 incidental worktree edits are isolated in `55704ca`. Tree clean, and the work now survives this machine. Per `OLMo-core/CLAUDE.md` the work lives in a dedicated worktree at `Capstone_LLM-worktrees/olmo-core/claude-01--dp2-kda-phase-0-prep`; the canonical checkout stays on `p4/interleaving-pretraining` as an integration baseline. A fresh worktree from that branch now *includes* DP2. |

**Do not begin GPU work until P0.0 preserves the source state.**

## 3. Shared experiment contract

### 3.1 Strict DP2 recurrence

\[
S_{t,0}=D_tS_{t-1},
\]

\[
S_{t,j}=S_{t,j-1}+
\beta_{t,j}k_{t,j}
\left(v_{t,j}^{\mathsf T}-k_{t,j}^{\mathsf T}S_{t,j-1}\right),
\quad j\in\{1,2\}.
\]

The KDA decay is applied once before both factors. Output is read after factor 2.

State orientation and symbols (load-bearing, do not paraphrase):

- \(D_t=\operatorname{diag}(\exp(g_t))\), the per-key decay gate. It is diagonal, hence symmetric — §4.5's \([D_tk]\) form is correct *only* because of this.
- \(S\) is \([K,V]\): it maps keys to values. \(k\in\mathbb R^{K}\) and \(v\in\mathbb R^{V}\) are columns, so \(k v^{\mathsf T}\) is \([K,V]\) and \(k^{\mathsf T}S\) is a row in \(V\)-space.

**Precondition:** keys are L2-normalized, \(\|k_{t,j}\|=1\). The erase \(I-\beta kk^{\mathsf T}\) has eigenvalues \(\{1,\,1-\beta\|k\|^2\}\), so the stability requirement is \(\beta\|k\|^2<1\), *not* \(\beta<1\). Without normalization a strict \(\beta=0.9\) at \(\|k\|=1.5\) gives eigenvalue \(-1.025\). The module normalizes \(k\) (`recurrent.py:1300`); the operator `kda_householder.py` does **not**, and P0.3/P0.4 are operator-level — so operator-level tests must either normalize keys themselves or assert \(\beta\|k\|^2<1\) directly.

### 3.2 Required arm behavior

| Arm | Required implementation behavior |
|---|---|
| R1 | One factor and \(\beta=\sigma(\ell)\); no reflection multiplier, and the erase eigenvalue \(1-\beta\|k\|^2\) stays positive — assert \(\beta\|k\|^2<1\), which given the §3.1 normalization precondition reduces to \(\beta<1\). |
| R1-P | Same \(d_{\rm model}\), heads, state dimensions, depth, and tokenization as R1/DP2; match non-embedding parameter count by changing only FFN width. |
| R1-2step-tiedK | Two updates with \(k_2=k_1\) forced at the recurrence boundary; factor-specific \(v,\beta\) retained. Manifests carry the exact ID; "tied-K" elsewhere in this doc is informal shorthand. **This arm is R1-equivalent — see "Why tied-K is an R1 control" below.** |
| DP2-budgeted | \(b=2\sigma(\ell_b)\), \(\pi=\sigma(\ell_\pi)\), \(\beta_1=b\pi\), \(\beta_2=b(1-\pi)\). Assert \(\beta_1+\beta_2=b\) to dtype tolerance — the only non-vacuous check here. **See "Write-mass matching" below** for why the factor 2 is mandatory. |
| DP2-strict | Two independent \(\sigma\)-betas. The mathematical range is \((0,1)\), but assert the §4.6 form \(0\le\beta\le1\) on the realized tensor — \(\sigma\) saturates to exactly 1.0 in low precision. |
| Reflection | Explicitly separate \(2\sigma\) beta regime. It must never appear in the Phase-1 primary table or composite. |

#### Why tied-K is an R1 control, not a weakened DP2

With \(k_2=k_1\) the state update is **exactly rank 1** (measured 2nd singular value 2.26e-15) and collapses to a single R1 step:

\[
\beta_{\rm eff}=\beta_1+\beta_2-\beta_1\beta_2\|k\|^2,\qquad
v_{\rm eff}=\frac{\beta_1(1-\beta_2\|k\|^2)\,v_1+\beta_2 v_2}{\beta_{\rm eff}}.
\]

Verified to 2.7e-15 over 500 trials at general \(\beta,\|k\|\). Under the §3.1 normalization \(\|k\|=1\), \(\beta_{\rm eff}=1-(1-\beta_1)(1-\beta_2)\in(0,1)\) and \(v_{\rm eff}\) is a genuine convex combination — so tied-K spans the **same function class as R1**, differing only in parameterization and parameter count. Interpret it as an R1-equivalent control; §5.8 treats a DP2-vs-tied-K tie accordingly.

Under the **reflection** regime \(\beta_1=\beta_2=2\) with \(\|k\|=1\), \(\beta_{\rm eff}=0\) and the reduction **degenerates rather than vanishing**: \(v_{\rm eff}\) is a \(0/0\) form, the composed erase multiplier \((1-\beta_1\|k\|^2)(1-\beta_2\|k\|^2)=+1\) restores the \(S_{t-1}\) read-back exactly, and the surviving write is the rank-1 difference \(k\,(2v_2-2v_1)^{\mathsf T}\) — measured nonzero (magnitude \(O(\|v_2-v_1\|)\)). Reflection tied-K is therefore not an R1-with-\(\beta_{\rm eff}\) model at all. That is the concrete mechanism for why strict and reflection beta regimes must never be pooled.

#### Write-mass matching for DP2-budgeted

The factor \(2\) in \(b=2\sigma(\ell_b)\) is mandatory. With \(b=\sigma(\ell_b)\) the arm carries mean total write mass 0.50 against DP2-strict's 1.00 (measured), so "strict beats budgeted" would be confounded with strict simply writing twice as hard — the exact confound this arm exists to isolate. With \(b=2\sigma(\ell_b)\) the mean total \(\beta\) is 1.00, matching strict's mean while still forcing the two factors to share one budget. The distributions match in mean but not in shape: budgeted's SD is ~1.4x wider (0.42 against 0.29), so report the realized write-mass distribution rather than asserting equality of spread.

Do **not** assert \(\beta_1+\beta_2\le1\) *or* \(\le2\). \(b\pi+b(1-\pi)\equiv b\) identically (verified to 1.1e-16) and \(2\sigma(\ell)<2\) strictly, so neither bound can ever fire — and \(2\sigma\) rounds to exactly 2.0 in bf16 and fp32, so \(\le2\) holds with zero slack in every dtype. \(0\le\beta_1+\beta_2\le2\) is a documentation-only invariant. The **identity** \(\beta_1+\beta_2=b\) is the check that does real work: it catches a factor computing its own budget, or \(\pi\) applied to the wrong factor.

### 3.3 Artifact layout

Create this hierarchy outside tracked source files, or in an approved remote result location. Do not invent an S3 bucket; the program owner must name it before any upload.

    dp2-kda-results/
      manifests/
        p0-<run-id>.json
        p1-calibration-<run-id>.json
        p1-triage-<run-id>.json
        p1-index.json            # Phase-1 scope, not per-run: the ordered launch
                                 # list (§5.6), both pre-registration digests
                                 # (§6.1), the named verifier (§6.2), and the
                                 # eval-bank checksum set. This is the "manifest
                                 # index" probes/analyze_dp2_phase1.py consumes.
      phase0/
        <run-id>/environment.json
        <run-id>/pytest.log
        <run-id>/numerics.json
        <run-id>/benchmark.json
        <run-id>/p0-decision.md
      phase1/
        eval-banks/<bank-id>.pt
        calibration/<run-id>/result.json
        triage/<run-id>/result.json
        analysis/phase1-decision.md
        analysis/phase1-decision.json

Producers for the `phase0/<run-id>/` files, so none is an orphan: `environment.json` from P0.0 step 4, `pytest.log` and `numerics.json` from P0.5's test run, `benchmark.json` from the P0.5 R1/R2 timing-and-memory measurement (§4.7), `p0-decision.md` from the §4.8 exit package.

Every manifest must include:

| Field | Required value |
|---|---|
| run_id | Globally unique string, such as p1-triage-s5_words-dp2_strict-b2101 |
| phase | p0 or p1 |
| arm | Exact arm ID from Section 3.2, verbatim — `R1`, `R1-P`, `R1-2step-tiedK`, `DP2-budgeted`, `DP2-strict`, `Reflection`. Informal short forms such as `tied-K` are rejected. |
| source | OLMo-core commit, uncommitted patch checksum if applicable, and probe source checksum as defined in §4.2 step 3a |
| environment | Image digest (ECR repository URI + immutable digest), GPU type/UUID, driver, CUDA, PyTorch, Triton, FLA, the absolute path of the exact Python executable, and **every environment variable the test suite consumes** — explicitly including `KDA_PROBES_DIR`, plus any `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, and `TRITON_*` value in effect |
| seeds | Bundle ID plus init, data, task-instance, and evaluation seeds |
| geometry | Model width, FFN width, layers, heads, key/value dimensions, convolution size |
| numerical mode | dtype, state accumulation dtype, beta regime, decay mode, backend |
| budget | steps, batch, accumulation, sequence lengths, exact loss-token count where applicable |
| outcome | completed, failed, stopped, or invalid; failure reason if not completed |
| prereg_digest | Content hash of the frozen pre-registration artifact (§6.1) in effect for this run. A Phase-1 run whose digest does not match the revision covering its stage is invalid. |
| retry | Absent for a first attempt. Otherwise the pre-enumerated fault class and its detection timestamp, per §5.7. A retry recorded without a fault class is invalid. |
| param_ledger | Non-embedding parameter count; for R1-P also the solved FFN width and the achieved mismatch against DP2-strict, absolute and percentage, per §5.4. |
| wall_clock | Per-job elapsed seconds, with setup/compile separated from training per `aws-operations.md` "On-node operating rules". Required for the P1.0 smoke jobs, which supply \(t\) to the cost equation (§5.4). |
| task_settings | The realized difficulty parameters actually used — including the filler span for `long_gap_retrieval` (§5.3) — not merely the setting's label. |

This list is exhaustive. A step that requires a value be recorded "in the manifest" must have a field
here; if it does not, that is a defect in this section rather than licence to invent one.

## 4. Phase 0 — semantic and numerical gate

### 4.1 Infrastructure assignment

Use one user-approved g6e.xlarge in us-east-1 only after the [AWS pre-launch checklist](aws-operations.md#mandatory-pre-launch-checklist) is complete. It supplies one L40S GPU and is intentionally not a multi-seed environment.

### 4.2 P0.0 — preserve and pin the source

**Owner:** implementation owner

**Inputs:** current OLMo-core worktree and probe sources

**Outputs:** source passport and reproducible environment definition

**Stop rule:** missing source preservation blocks every later package.

Required actions:

1. Record the OLMo-core branch, both SHAs (`6b75c06` DP2 core, `55704ca` incidental drift), and the complete `git status --porcelain` — which must be **empty**. A dirty tree at P0.0 time is a stop until explained.
2. Preservation is the commit, not a patch bundle. Record `6b75c06` in the manifest `source` field and verify it with `git show 6b75c06 --name-only`, which must list exactly the 7 paths below — no more, no fewer. (Superseded: earlier revisions of this runbook called for a binary patch bundle plus checksums for untracked files, because the DP2 work was uncommitted. It is now committed; a commit SHA is the stronger artifact and the patch-bundle step is retired.)
3. **Verify by rule, not by hand-maintained list.** The 7 files below are the expected DP2 core. Any file in `6b75c06` that is not on this list, or any listed file missing from it, is a stop. Note that `55704ca` deliberately carries ~38 unrelated files (`cli.py`, `jobs.py`, ~20 training scripts) that were dirty before the DP2 work began and are **not** part of the passport — they are recorded only to make the tree clean, and are explicitly unreviewed.
   - src/olmo_core/nn/attention/recurrent.py
   - src/olmo_core/nn/attention/kda_householder.py
   - src/olmo_core/nn/attention/kda_householder_torch.py
   - src/olmo_core/nn/attention/__init__.py — adds `KimiDeltaHouseholder` and `KimiDeltaHouseholderConfig` to the imports and `__all__`
   - src/olmo_core/nn/attention/flash_linear_attn_api.py — +85 lines of backend dispatch
   - src/test/nn/attention/kda_householder_test.py
   - src/test/nn/attention/recurrent_test.py

   The two wiring files are not optional, and they fail in different ways. `flash_linear_attn_api.py` is a **hard** dependency: `recurrent.py:18-22` imports `dispatch_chunk_kda` at module level, and that symbol does not exist at the base commit `f17824e` (verified, 0 occurrences), so omitting it raises `ImportError` on `import olmo_core.nn.attention.recurrent` and takes `KimiDeltaAttention` and `GatedDeltaNet` down with it. Omitting `attention/__init__.py` is narrower: the class stays reachable by its fully-qualified module path but is absent from the package namespace and `__all__`, so every `from olmo_core.nn.attention import KimiDeltaHouseholder` dispatch path fails.

   3a. **Pin `probes/` and define the probe source checksum.** ✅ **DONE 2026-07-31.** `probes/` is now a standalone git repository — `git init` plus one baseline commit, `93b60d7`, covering 34 files (32 `.py`, 1 `.sh`, 1 new `.gitignore` excluding `__pycache__`/`*.pyc`); contents committed as-is with no functional edits, exactly as they stood when this runbook was audited against them. **The probe source checksum is therefore the VCS-commit form: `93b60d7`** (full: `93b60d73de574fc7a2b93d326864ceee8b8e757c`). Note this repo has **no remote** — it is local-only, so the SHA is not independently fetchable; pushing it somewhere durable before Phase 1 is advisable. The original alternative is retained below for reference only.

   Previously: `probes/` was under no version control at all (verified: `git rev-parse` failed there and there was no parent repo), yet `probes/naive_kda_householder.py` is the Phase-0 correctness oracle. Either place `probes/` under version control and record its commit SHA, or archive it as `tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner`, excluding `__pycache__` and `*.pyc`, and record the sha256 of that archive. The manifest's "probe source checksum" is defined as **exactly that value** — the sha256 of the `__pycache__`-excluded deterministic tar of `probes/`, or the VCS commit SHA if `probes/` is tracked. State which of the two forms was used. Leaving this unspecified produces three different values from three engineers.
4. Pin the image/environment: Python, PyTorch, CUDA, Triton, FLA, compiler flags, and GPU architecture. **Record every environment variable the test suite consumes**, explicitly including `KDA_PROBES_DIR` (§4.7 reads this field; without it the handoff is broken), plus `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, and any `TRITON_*` value. Record the absolute path of the exact Python executable (`python -c 'import sys; print(sys.executable)'`). These go into `phase0/<run-id>/environment.json` and into the manifest `environment` field.
5. Build the image once from `OLMo-core/src/Dockerfile` and record its immutable digest. The built image must support **both** `sm_89` (L40S, Phase 0) and `sm_90` (H100, Phase 1) so the same image carries through both phases without a rebuild. **This requires editing the Dockerfile first — it does not build `sm_89` today.** *As originally committed* (before `b5433c0`), `src/Dockerfile:55` set `TORCH_CUDA_ARCH_LIST="9.0 10.0"` and `:59` set `FLASH_ATTN_CUDA_ARCHS="90;100"`, with no `8.9`/`89` anywhere in the file. Editing both lines is a prerequisite action of P0.0, not an assumption, and the values actually used must be recorded in `environment.json`. **But the two lines do not take the same token — see the correction immediately below. `8.9` is right for `:55` and `89` is silently wrong for `:59`, where the correct value is `80`.** Push it and record the ECR repository URI plus the immutable digest — `<ECR_REPO_URI>@sha256:<digest>`, never a mutable tag. Do not tune kernels or dependencies after P0.0 without making a new manifest revision.

> **Correction (2026-07-31): the two edits are not equivalent, and the flash-attn half is a silent no-op — BUILT AND VERIFIED WITH `cuobjdump`, not merely read.** Commit `b5433c0` made both edits as this step directs. The two build systems consume the value in completely different ways:
>
> - **`src/Dockerfile:55` (grouped_gemm) — the edit works, confirmed by `cuobjdump`.** `grouped_gemm/setup.py:12-17` (SHA `f1429a3`) checks whether `TORCH_CUDA_ARCH_LIST` is *set at all*; if so it emits **no** `--generate-code` flag of its own and lets torch's `_get_cuda_arch_flags` parse the list. Torch understands `"8.9"` and emits `arch=compute_89,code=sm_89`. Built for real (6m46s, rc=0): `grouped_gemm_backend.cpython-312-x86_64-linux-gnu.so` contains six cubins — **`sm_89`, `sm_90`, `sm_100`, two of each** — so the L40S target is genuinely present. (No PTX, but with a native `sm_89` cubin none is needed.)
> - **`src/Dockerfile:59` (flash-attn 2.8.2) — the edit does nothing.** `FLASH_ATTN_CUDA_ARCHS` is not a generic arch list; `setup.py:179-191` tests it against **four hard-coded string literals only** — `"80"`, `"90"`, `"100"`, `"120"`. The token **`"89"` does not appear anywhere in flash-attn 2.8.2's `setup.py`** (verified: `grep -c 89` returns 0), so there is no branch for it to enable. `FLASH_ATTN_CUDA_ARCHS="89;90;100"` and the original `"90;100"` produce the **identical** gencode set `{sm_90, sm_100}`. Worse, `setup.py:306` passes those flags through `extra_compile_args["nvcc"]`, and torch's `_get_cuda_arch_flags` returns `[]` as soon as it sees a caller-supplied flag containing `arch` (`cpp_extension.py`, "If cflags is given, there may already be user-provided arch flags"), so torch does **not** backfill sm_89 either. Nor is there a PTX fallback: `code=sm_XX` alone embeds SASS with no PTX, so there is nothing for the driver to JIT onto an L40S.
>
>   **Empirical confirmation.** Both edited layers were built for real on a FarmShare compute node (rootless `podman`, x86_64, same `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` base, same pins and SHAs, torch 2.10.0+cu128; 25m19s wall for the full two-layer build, rc=0; harness `src/scripts/dp2_kda_phase0/arch_proof.Dockerfile`). `cuobjdump` on the resulting `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`, built with `FLASH_ATTN_CUDA_ARCHS="89;90;100"`, reports:
>
>   - `-lelf` (SASS): **`sm_90`, `sm_100`** — no `sm_89`.
>   - `-lptx` (PTX): **empty** — no PTX at all, so no JIT fallback onto an L40S.
>
>   That is the edited value producing byte-for-byte the arch set of the unedited one. The claim is now measured, not inferred from `setup.py`.
>
>   The correct token for flash-attn is **`"80"`**, not `"89"`: sm_80 SASS is binary-compatible upward within major version 8, which is how flash-attn's own default (`"80;90;100;120"`) covers every Ada/Ampere card. So the working value is `FLASH_ATTN_CUDA_ARCHS="80;90;100"`.
>
>   **✅ VERIFIED AND FIXED (commit `61c7cf6`).** The substitution was built on a FarmShare compute node (job `1671023`, rc=0, 27m48s, 34 CPUs / 128G — note a 16-CPU/48G attempt was OOM-killed at rc=137, so size this build generously) and inspected with `cuobjdump`. Direct comparison of the two variants, same base, pins and SHAs:
>
>   | `FLASH_ATTN_CUDA_ARCHS` | SASS arches in `flash_attn_2_cuda*.so` | Ada/L40S? |
>   |---|---|---|
>   | `"89;90;100"` (as in `b5433c0`) | `sm_90`, `sm_100` | ❌ no, and no PTX to JIT from |
>   | `"80;90;100"` (the fix) | **`sm_80`**, `sm_90`, `sm_100` | ✅ yes, via sm_80 |
>
>   Note the practical consequence of the original error: because upstream's default is already `"80;90;100;120"`, the **unedited** Dockerfile covered sm_89 via sm_80, and `b5433c0` *removed* that coverage while appearing to add it. This is the failure mode the runbook should generalize from — an edit that exits 0, reads plausibly, and silently does the opposite of its intent.
>
>   **This does not block Phase 0.** The DP2-KDA path imports no flash-attn: `kda_householder.py`, `kda_householder_torch.py` and `recurrent.py` contain zero `flash_attn` imports, and `flash_attn_api.py:9-12` wraps the import in `try/except ImportError` setting `flash_attn_2 = None`. An image whose flash-attn lacks sm_89 still runs every §4.7 gate test on an L40S. The consequence is narrower and should be recorded rather than fixed under time pressure: the image does not deliver the "one image for both phases" property for *ordinary* attention on sm_89, so anything outside the KDA path that reaches flash-attn 2 on an L40S will fail at kernel launch, not at import.
>
>   Flash-attn **3** is a separate matter and is *not* a defect: `src/Dockerfile:69-70` builds it from the upstream `hopper/` directory with `FLASH_ATTENTION_DISABLE_SM80=TRUE`. FA3 is Hopper-and-later by construction, so it was never going to carry sm_89 and no edit should try to make it.

**Where to build this image.** The macOS dev laptop cannot build it (arm64; the Dockerfile hardcodes `ARG TARGET_PLATFORM=x86_64`, a Linux-x86_64 miniforge installer at `:35`, and an x86_64 MLNX OFED tarball at `:155`). FarmShare can, and it is free, but **only with three non-obvious flags** — plain `podman build` fails there three different ways in sequence (established 2026-07-31):

1. The user has **no `/etc/subuid`/`/etc/subgid` range**, so rootless podman maps a single UID. `apt-get` cannot `seteuid` to `_apt` and dies with `setgroups (1: Operation not permitted)`. Fix: `APT::Sandbox::User "root"` — affects package fetching only, no compiler flag.
2. The default graph root is on **NFS**, which forces `force_mask="700"` and makes the layer `COMMIT` fail with `permission denied`. Fix: point `--root`/`--runroot` at node-local `/tmp` (ext4, ~1.1 TB free on compute nodes).
3. Layer extraction then fails on `lchown /etc/gshadow: invalid argument` for the same single-UID reason. Fix: `--storage-driver overlay --storage-opt overlay.ignore_chown_errors=true`.

Working invocation, under `srun` (never the login node), with `--cgroup-manager=cgroupfs --events-backend=file`. Image references built this way are **local to that node's `/tmp`** and are not a substitute for the ECR digest this step requires — FarmShare is for de-risking the build, not for producing the manifest artifact.

### 4.3 P0.1 — R1 equivalence

**Question:** Does the R-factor implementation with R=1 behave like ordinary KDA?

Required tests:

1. Copy ordinary KDA weights into R1 DP-R KDA by an explicit name/range map.
2. Compare module outputs and input/shared-parameter gradients on the same inputs.
3. Compare operator final state **and operator output** through the matching low-level KDA and DP-R operator calls. The output comparison is mandatory, not a convenience: the returned `final_state` is **invariant to the readout microstep** — it is always the post-factor-2 state, because where you read the output does not change the recurrence — while the outputs read after factor 1 versus factor 2 differ by a median 11% and up to 46% (measured). A state-only comparison therefore passes an implementation that reads out at the wrong microstep — precisely the bug §4.7's virtual-token test claims to exclude.
4. Test dense and packed inputs separately.

Acceptance — **the tolerance policy is split by backend, and this split is mandatory.** `kda_householder.py:737-739` rejects float32 on the Triton backend and the fwd kernel accumulates in `tl.float32` unconditionally, so a float64 oracle can only ever validate `backend="torch"` — the reference, not the production kernel. The kernel and the reference are comparable only at bf16. Require three distinct things:

1. **Float64 semantic tests against `backend="torch"`,** \(rtol=0\) and a calibrated absolute tolerance in the \(10^{-11}\)–\(10^{-12}\) range. This bar is **float64-only**: `float32 eps = 1.19e-7` and the best attainable absolute accuracy at these output scales is ~\(10^{-8}\), four orders above the demand, so an FP32 test written to this criterion fails unconditionally. FP32 is explicitly out of scope for the \(10^{-11}\) bar.
2. **A separate triton-vs-torch equivalence check** with a per-tensor **relative** error budget — the pattern already exists at `kda_householder_test.py:670-679`. Report max, median, and p99 relative error per compared tensor rather than one opaque `allclose`. At bf16 the repo's own constants are `ATOL = RTOL = 2e-2` (`kda_householder_test.py:30-31`), and that threshold is loose: a seeded dropped-cross-term bug produces median relative error 3.5e-3, so **~90% of random seeds slide under 2e-2**. Treat 2e-2 as a smoke bound, not as the semantic gate.
3. **A mutation test** demonstrating that the bf16 check actually *fails* on a deliberately seeded cross-term bug (drop \(\rho\) in the §4.5 form, or reorder the two factors). Without this the bf16 gate is unfalsified and, per the measurement above, will usually pass a broken kernel.

Also required:

- BF16 semantic tests use the existing calibrated relative-error policy; record max, median, and p99 error rather than one opaque allclose.
- No unexplained parameter mismatch or factor-order discrepancy.

### 4.4 P0.2 — independent R2 virtual-token oracle

**Question:** Does native R2 equal two ordinary KDA microsteps with only one real-token decay?

Construct virtual states after Q/K/V projections and after their real-rate ShortConv paths:

| Virtual position | Query | Factor data | Log-decay | Output retained |
|---|---:|---:|---:|---:|
| first | zero (injected **after** `l2_normalize`; see below) | factor 1 | real \(g_t\) | no |
| second | real \(q_t\) | factor 2 | zero | yes |

**The zero query must be injected after `l2_normalize`, or the oracle NaNs.** `olmo_core.nn.functional.l2_normalize` (`functional/__init__.py:16-18`) is a bare \(x/\|x\|\) with **no epsilon** and returns NaN on a zero vector — unlike PyTorch's `F.normalize`, which returns zeros. It is applied at `recurrent.py:1299`, i.e. *after* the injection point named above, so a zero query written in before normalization NaNs the entire doubled sequence. Either inject the zero query after `l2_normalize`, or use a unit dummy query and discard its output.

**Compare on the operator output \(o\), pre-`o_norm`.** The module's gate is \(g=-\exp(A_{\log})\cdot\mathrm{softplus}(\cdot)\), which is strictly negative and therefore cannot produce the zero log-decay the second virtual position requires. The oracle is an operator-level construction; do not attempt it through the module's normalized output.

Required cases:

- random dense tensors;
- nonzero incoming state;
- unequal packed documents;
- virtual cumulative lengths exactly double the real token cumulative lengths;
- prefix/suffix recurrence handoff — **operator-level only**, threading `initial_state` explicitly. This is not implementable at module level: `CausalConv1d.forward` (`convolution.py:53`) takes and returns no conv state, so a module-level split zero-pads the suffix's first \(\text{kernel\_size}-1=3\) positions and fails for reasons unrelated to the recurrence.
- factor order swapped as a negative control, which must fail **by at least a stated floor**. "Must fail" is half a spec. VERIFIED: the swap difference scales with \(|k_1\cdot k_2|=O(K^{-1/2})\); at \(K=256\) the 1st-percentile separation is 2.1e-6, the minimum 2.3e-7, and with exactly orthogonal keys it is **0.000e+00** — the control cannot fail. Therefore: (a) construct the two keys so non-orthogonality is guaranteed, e.g. fix \(k_2=\mathrm{normalize}(k_1+\epsilon r)\) with a recorded \(\epsilon\) giving \(|k_1\cdot k_2|\ge 0.3\); (b) state the floor as a **fraction of the output scale**, never as a multiple of the comparison tolerance: require \(\max\lvert o_{\rm swap}-o\rvert/\max\lvert o\rvert\ge 5\times10^{-2}\). Measured at \(K=256\) with \(|k_1\cdot k_2|\ge0.3\), the realized separation is min 6.6e-2, p1 1.1e-1, median 4.0e-1, so this clears with ~1.3x margin at the minimum while still sitting above the bf16 2e-2 band. A tolerance-multiplied floor does not work in either regime: \(10^3\times\)1e-11 is 1e-8, inert (cleared by 6.6e6x), while \(10^3\times\)2e-2 is 20.0, unsatisfiable (max separation 1.13). Record the realized separation and \(|k_1\cdot k_2|\) in `numerics.json`.

Never duplicate raw input tokens through Q/K/V ShortConv. A test that does so is invalid even if it appears numerically close. With kernel size 4, virtual position \(2t+1\) would convolve \([x_{t-1},x_{t-1},x_t,x_t]\) instead of \([x_{t-3},x_{t-2},x_{t-1},x_t]\). Constructing the virtual positions after the projections and after their real-rate ShortConv paths **is sufficient** for that specific problem, because the conv is depthwise. The residual leaks are the `l2_normalize` NaN and the conv-state handoff above, not the receptive field.

### 4.5 P0.3 — independent rank-two algebraic oracle

Implement a float64 test oracle using:

\[
u_j=\beta_jk_j,\qquad \rho=k_2^{\mathsf T}u_1,
\]

\[
U=[u_1-u_2\rho,\;u_2],\quad
V=[D_tk_1,\;D_tk_2],\quad
R=[v_1,\;v_2],
\]

\[
S_t=D_tS_{t-1}+U(R^{\mathsf T}-V^{\mathsf T}S_{t-1}).
\]

**This formula has been independently verified and every term is load-bearing.** Max error versus the sequential §3.1 recurrence is ~1e-15 over 200 random float64 trials, and all three plausible corruptions fail loudly, each landing roughly **fifteen orders of magnitude above** the correct residual: dropping \(\rho\), putting \(\rho\) on the \(u_2\) column, and using plain \(k\) instead of \(D_tk\) in \(V\). Gate on the **ratio** to the verified residual, not on an absolute constant — the corruption magnitudes scale with \(\|S_{t-1}\|\|v\|\) and are not reproducible numbers. Do not "simplify" any term.

\(V=[D_tk_1,\,D_tk_2]\) is right because \(V\) is contracted against the **undecayed** \(S_{t-1}\), and since \(D_t\) is diagonal hence symmetric, \((D_tk)^{\mathsf T}S_{t-1}=k^{\mathsf T}(D_tS_{t-1})\) — so the decay appears exactly once overall, matching §3.1's \(S_{t,0}=D_tS_{t-1}\).

**Required external anchor.** P0.2 and P0.3 are independent *implementations* of §3.1, not independent *specifications* — if §3.1 itself is wrong, all of Phase 0 passes. Add one case comparing against the external reference the repo already documents at `kda_householder.py:689-693`: `fla.ops.gated_delta_product.naive`. Construct \(g\) constant along \(K\) for that case.

> **Correction (measured on L40S against fla 0.5.1, 2026-07-31).** The "agrees to float64 ulp" claim above is **wrong on two counts**, both verified by direct measurement, and a test written to it fails at 0.82 relative error while the implementation is correct:
>
> 1. **fla ignores its own `scale` argument.** It is accepted but never applied to \(q\) — passing `scale=1.0` and `scale=K**-0.5` returns byte-identical output. The comparison must therefore pass `scale=1.0` on *both* sides. Using `K**-0.5` produces a 0.65 relative disagreement that looks like a correctness failure and is not one.
> 2. **fla computes in float32.** The first line of its body is `q, k, v, beta = map(lambda x: x.float(), ...)`, so float64 inputs are downcast and float64-ulp agreement is **unattainable by construction**. Measured residual is 6.4e-8 against a float32 floor of 1.0e-7 — i.e. the disagreement is entirely explained by the downcast.
>
> The implemented test (`test_r2_matches_external_gated_delta_product_naive`) therefore computes this side's own float32-vs-float64 gap in-test and requires agreement within `max(10 × floor, 1e-6)`, rather than hard-coding a constant. A genuine disagreement about the recurrence lands orders above that band. This case is required, not optional; without it Phase 0 is self-referential. Every FLA-dependent test in the suite carries `@requires_fla`, which *skips* when it is absent, so P0.0 step 4 must record the installed `fla` version and P0.5 must assert this test **ran** — not merely that it did not fail. A `@requires_fla` skip here is precisely the silent-green pathway of §4.7, not an acceptable outcome.
>
> **Re-verified against the repo-pinned fla 0.4.1 (L40S, 2026-07-31). Both conventions hold identically; the test is pin-safe and needs no version branch.** The concern was that the conventions above were measured on FarmShare's 0.5.1 while `pyproject.toml:69` pins `flash-linear-attention==0.4.1`, so the manifest-grade image would run a different reference. A separate venv was built at 0.4.1 (same torch 2.9.1+cu128 / triton 3.5.1, existing 0.5.1 venv left untouched) and both versions were measured on the same inputs and seed. The results are not merely close, they are **bit-identical across the two versions**:
>
> | Measurement | fla 0.4.1 | fla 0.5.1 |
> |---|---|---|
> | `scale=1.0` vs `scale=K**-0.5` byte-identical | true | true |
> | max abs gap between those two calls | 0.0 | 0.0 |
> | body mentions `scale` after the signature | none | none |
> | float64 input vs float32 input, relative | 1.171e-7 | 1.171e-7 |
> | anchor relative residual | 6.426e-8 | 6.426e-8 |
> | in-test float32 floor / budget | 1.048e-7 / 1.048e-6 | 1.048e-7 / 1.048e-6 |
> | anchor passes | yes | yes |
> | residual under the *wrong* scale convention | 0.823 | 0.823 |
>
> The source of `naive_recurrent_gated_delta_product` is byte-for-byte the same at both pins, including the `q, k, v, beta = map(lambda x: x.float(), ...)` downcast line, and the signature `(q, k, v, g, beta, scale, cu_seqlens, initial_state=None, output_final_state=False, num_householder=1)` is unchanged. Note the wrong-convention residual is **0.823**, not the 0.65 recorded above — 0.65 appears to have come from a different geometry or seed. Treat "orders of magnitude above the budget" as the claim, not the specific constant. Harness: `src/scripts/dp2_kda_phase0/fla_pin_probe.py`.

**No Dockerfile change is needed to get `fla` into the image** (verified 2026-07-31). `src/Dockerfile:106` installs `'.[all]'`, and `pyproject.toml:73` expands `all` to include the `fla` extra, which is `flash-linear-attention==0.4.1` (`pyproject.toml:69`) with no platform marker. Both anchor symbols exist at that pin — `fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product` and `fla.ops.kda.naive.naive_recurrent_kda` — and its dependency closure is only `fla-core==0.4.1` → `torch`, `einops`, both unpinned, so it cannot fight the image's pinned torch.

The earlier `import fla` → `ModuleNotFoundError` observation was made on the **macOS dev laptop**, which has no `ai2-olmo-core` install and no `triton` (Triton publishes Linux-only wheels), so it cannot host `fla` regardless. That result says nothing about the Linux CUDA image. Note also that `FLA_MARKS` (`testing/utils.py:137-140`) pairs the skipif with `pytest.mark.gpu`, so these tests skip on *any* CPU-only host for a second, independent reason — the L40S node satisfies both. **P0.0's job here is therefore to verify and record, not to install:** on the built image, assert `python -c "import fla; print(fla.__version__)"` succeeds and record the version. If it fails, the `all` extra has regressed and *that* is the finding.

Note explicitly that `kda_householder_torch.py` is **not** an independent oracle. Its own docstring states it is a transcription of `probes/naive_kda_householder` using the same einsum calls in the same order, so it and the probe are **one** oracle, not two.

This test is a semantic oracle only. It is not a production-kernel design or a GPU-speed claim.

### 4.6 P0.4 — gradients, identity, and precision

**P0.4 gates the operator level.** State this in the manifest, and record the gap it leaves: on the production Triton backend there is **no cotangent path for the final state** — the forward calls `ctx.mark_non_differentiable(final_state)` (`kda_householder.py:639-640`) and the backward deletes `dht` (`:654`) — while the torch backend's final state *is* differentiable. So gradcheck through a carried state validates `backend="torch"` only, and P0.2's prefix/suffix recurrence handoff is exactly the shape where those gradients are wanted. Any module-level gradient claim is a separate, additional check: at module level the differentiable set also includes `w_q`, `w_k`, `w_v`, `w_b`, `f_proj`, `A_log`, `dt_bias`, `g_proj`, `o_norm`, `w_out` and the three conv weights, and `A_log`/`dt_bias` reach the loss only through a saturating softplus — so a near-zero gradient there is expected, not a bug.

| Check | Required assertion |
|---|---|
| Gradcheck | Float64 gradients pass for \(q,k,v,g,\beta,S_0\) on `backend="torch"`; the Triton backend is exempt for \(S_{\rm final}\) only, per the note above. |
| True identity | An explicit post-gate mask making \(\beta_2=0\) leaves R1 behavior exactly unchanged. VERIFIED to be a true identity — measured difference exactly 0.0 — so assert equality with zero tolerance. |
| Negative identity control | \(v_2=0,\beta_2\ne0\) changes state (VERIFIED: 2.99 on a non-degenerate case); it must not be used as an identity shortcut. Requires a **non-degenerate incoming state and an asserted separation floor**: with \(S_{t-1}=0\) and \(\beta_1v_1=0\) the difference is exactly 0.000e+00 and the control silently "passes". Seed a nonzero \(S_{t-1}\) and nonzero \(\beta_1v_1\), assert the difference exceeds a stated floor, and record the realized value. |
| Beta range | Every strict arm logs and asserts \(0\le\beta\le1\) on the realized tensor, and logs the **pre-sigmoid logit** distribution. Do **not** assert \(0<\beta<1\): in bf16 \(\sigma(\ell)\) returns exactly 1.0 for \(\ell\ge6.235\) (fp16: 8.32; fp32: 16.64), so one unconstrained `w_b` logit crossing ~6.2 during training trips the strict-arm assertion on a perfectly correct model. Alternatively evaluate the assertion in fp32. |
| Long rollout | BF16 input with FP32 state/gate accumulation has no NaN/Inf and no superlinear error growth, **operationalized as follows**: at a fixed seed, measure \(\max\lvert o-o_{\rm fp64}\rvert\) over a length ladder \(T\in\{32,64,128,256,512,1024\}\), fit \(\log e=a+p\log T\), and require \(p\le1+\text{margin}\) with the margin recorded in the manifest. Log the per-\(T\) errors and doubling ratios. Measured baseline for reference: 6.3e-3 → 8.6e-3 with doubling ratios 1.02 / 1.25 / 0.94 / 1.09 / 1.05 (\(\sqrt T\) growth would be 1.41, linear 2.0). Without a reference, metric, exponent, threshold, and ladder this check is unfalsifiable. |
| Decay stress | Test near-zero retention, strong decay, and separate reflection beta regime. |

The factor mask is required if checkpoint expansion will ever be claimed. It does not authorize a target-model checkpoint experiment.

> **Decay stress executed on L40S, 2026-07-31.** This is the one §4.6 row with no corresponding named test in §4.7, so it was run as a standalone harness: `src/scripts/dp2_kda_phase0/p04_p05_measurements.py --skip-timing`. Geometry `B=2, T=32, H=2, K=V=16`, each regime at `R ∈ {1,2,3}`.
>
> **Choosing the reference is the whole difficulty here, and the obvious choice is vacuous.** Comparing `kda_householder_torch` against `probes/naive_kda_householder` returns **exactly 0.0** in all twelve cells. That is not a pass — it is a live confirmation of the warning at the end of §4.5 that the torch backend is a transcription of the probe using the same einsum calls in the same order, so the two are **one** oracle. A decay-stress table built on that comparison would report twelve perfect zeros while testing nothing about decay. The harness keeps that column only as a transcription check and gates on a second arm: the fused **Triton** kernel, which shares no code with the probe. It consumes bf16, so its floor is bf16 round-off.
>
> | Regime | min exp(g) | max β | Triton vs float64 reference, relative | finite |
> |---|---|---|---|---|
> | near-zero retention (g = −30) | 9.36e-14 | 0.94 | 3.93e-3 – 4.50e-3 | yes |
> | strong decay (g ≈ −3) | 3.47e-2 | 0.95 | 4.61e-3 – 5.86e-3 | yes |
> | reflection beta (β ∈ (0,2)) | 2.25e-2 | 1.93 | 4.11e-3 – 5.39e-3 | yes |
> | baseline (logsigmoid gate) | 3.56e-2 | 0.98 | 3.53e-3 – 6.30e-3 | yes |
>
> All twelve cells finite, no NaN/Inf, and — the load-bearing observation — the three stress regimes land in the **same** 3.5e-3–6.3e-3 band as the baseline. Extreme decay and the reflection regime do not degrade agreement, which is the property this row exists to establish. `exp(g) = 9.4e-14` confirms the near-zero-retention regime genuinely annihilated the state rather than merely shrinking it, and `β_max = 1.93` confirms the reflection regime actually left the contraction range.

### 4.7 P0.5 — GPU test and operator preflight

Before the GPU run, add the following named test coverage. Names are part of the deliverable so a reviewer can find the required checks without reverse-engineering a prose description.

| Test file | Required test ID | What it proves |
|---|---|---|
| kda_householder_test.py | test_r2_matches_post_shortconv_virtual_token_oracle | Native R2, factor order, one decay, and retained second output are correct. |
| kda_householder_test.py | test_r2_matches_rank_two_float64_oracle | The sequential update equals the independent algebraic form. |
| kda_householder_test.py | test_r2_virtual_oracle_gradients_and_initial_state | Every differentiable input and incoming state has matching gradients. |
| kda_householder_test.py | test_r2_virtual_oracle_varlen_doubled_offsets | Packed boundaries and doubled offsets are correct. |
| kda_householder_test.py | test_factor_two_zero_beta_is_identity | Exact dormant factor behavior exists. |
| kda_householder_test.py | test_zero_value_is_not_factor_identity | The erase-term negative control is covered. |
| kda_householder_test.py | test_r2_bf16_long_rollout_is_finite | BF16 rollout is finite and bounded. |
| kda_householder_test.py | test_r2_virtual_oracle_factor_order_swap_separates | The §4.4 factor-order-swap negative control fails, and fails by at least the asserted floor. |
| kda_householder_test.py | test_r2_triton_matches_torch_relative_error_budget | Kernel-versus-reference agreement at bf16, reported as per-tensor max/median/p99 relative error. |
| kda_householder_test.py | test_r2_bf16_check_fails_on_seeded_cross_term_bug | The bf16 gate is falsifiable — the mutation is caught, not slid under 2e-2. |
| kda_householder_test.py | test_r2_matches_external_gated_delta_product_naive | External anchor: agreement with `fla.ops.gated_delta_product.naive` at \(g\) constant along \(K\). |
| recurrent_test.py | test_kimi_delta_householder_r1_copied_weight_module_parity | R1 module output and shared gradients match ordinary KDA. |
| recurrent_test.py | test_kimi_delta_householder_strict_beta_contract | Strict/reflection beta modes cannot be conflated. |

The test identifiers may be implemented as parameterized cases, but each name must appear in pytest output or an equivalent collected-test report.

**Status: ✅ all 13 written and passing on an L40S, 2026-07-31** (commit `4f747f5`). They were all absent before that — the nine original names got zero grep hits against the 61 existing test functions, and the four added were new. `test_kimi_delta_householder_r1_matches_kda_params` (`recurrent_test.py:639-653`) remains a **misleading neighbour**: it asserts only parameter count and FLOPs, never copies weights, never compares outputs or gradients. The new `test_kimi_delta_householder_r1_copied_weight_module_parity` is the real check.

Verification run: Stanford FarmShare `oat` GPU partition, NVIDIA L40S (sm_89, driver 595.71.05), python 3.12.3 / torch 2.9.1+cu128 / triton 3.5.1 / **fla 0.5.1**. Result: **156 passed, 0 skipped, 1 sanctioned deselect**, all four §4.7 skip-discipline assertions passing.

> **This is not a P0.5 pass.** FarmShare is free and carries the correct silicon, which makes it the right place to de-risk the tests before spending on the g6e — but P0.0 requires the pinned image and its immutable digest, and FarmShare runs **fla 0.5.1 against the repo's pinned 0.4.1** (`pyproject.toml:69`). The manifest-grade run must still happen on the pinned image. Re-verify the external anchor there specifically: its conventions were measured against 0.5.1.

Two implementation notes discovered while writing these, both of which contradict text elsewhere in this document and are corrected in place: the fla anchor's `scale`/float32 conventions (§4.5), and `BF16_RTOL` being unusable as a standalone bound (below).

**Module-parity tolerance.** `BF16_RTOL` (1e-5, `attention_test.py:49`) is the *relative* half of a `torch.testing.assert_close` pair whose absolute half is `BF16_ATOL = 5e-3`. It was never a standalone max-relative-error bound and is unattainable for a bf16 module forward — measured realized error is max 8.6e-3, median 5.4e-4, p99 3.2e-3. Module parity therefore uses **2e-2**, matching the calibrated constant in `kda_householder_test.py:30-31`, with gradients at 5e-2 since the backward composes two already-diverged forwards. This is a plumbing check on the weight map and projections; the tight semantic bar lives on the float64 operator tests.

Run the following logical sequence on the g6e.xlarge environment:

    cd OLMo-core
    python -m pytest src/test/nn/attention/kda_householder_test.py \
      src/test/nn/attention/recurrent_test.py -q -p no:randomly -ra

The exact Python executable and every environment variable — explicitly including `KDA_PROBES_DIR` — must come from the P0.0 environment manifest (§4.2 step 4). Do not substitute a local CPU result for the CUDA gate.

**P0.5 must assert ZERO SKIPS in the two gate files. A skipped suite exits 0, so without this the gate reports green having verified nothing.** `kda_householder_test.py:34-58` loads its correctness oracle from *outside* the repo (`probes/naive_kda_householder.py`) and calls `pytest.skip(...)` when it cannot find it. This defect is latent locally and fires exactly where it matters: the fallback candidate `Path(__file__).parents[5]/"probes"` resolves on the dev machine, so the suite passes there, and fails only where OLMo-core is checked out without its sibling `probes/` — i.e. the Phase-0 GPU node. Required:

- pass `-ra` so skip reasons are printed, and archive the full output to `phase0/<run-id>/pytest.log`;
- deselect the one sanctioned skip by name — `--deselect src/test/nn/attention/recurrent_test.py::test_context_parallel_gdn_ulysses`. That test carries `@requires_multi_gpu` (`recurrent_test.py:244`), which skips whenever `torch.cuda.device_count() <= 1`, so it skips unconditionally on the single-GPU g6e.xlarge of §4.1. It must be deselected explicitly, never absorbed into a tolerated skip count;
- then parse the summary and **assert `collected == passed` and `skipped == 0`** on the remainder (`--junitxml` or `-q` tail parsing, not eyeballing);
- additionally **assert the `-ra` skip-reason set contains nothing matching `naive_kda_householder`, `KDA_PROBES_DIR`, or `requires_fla`** — those are the silent-green pathways this check exists for, and a count-only assertion can be satisfied by deselecting the wrong thing;
- assert `collected` is at least the count recorded in the P0.0 manifest — baseline at `6b75c06` is **25** tests in `kda_householder_test.py` and **36** in `recurrent_test.py`, 61 total — so a collection error that silently drops tests is also caught;
- a nonzero skip count outside the named deselect is a P0.5 **failure**, not a warning, and is not waivable under §4.8.

Then measure R1 and R2 at the probe geometry used in Phase 1:

- forward-only, backward-only, and forward+backward;
- BF16 inputs with FP32 recurrence accumulation;
- warm kernels excluded from timing;
- peak allocated and reserved memory;
- logical real tokens/second, never virtual positions/second.

This is an operational feasibility check only. It does not decide Phase 2 or establish an optimized DP2 kernel.

> **Measured on FarmShare L40S (sm_89), 2026-07-31.** Harness `src/scripts/dp2_kda_phase0/p04_p05_measurements.py`, `backend="triton"`, 5 warmup iterations discarded then 20 measured, `reset_peak_memory_stats()` called **after** warmup so autotuning scratch is not attributed to the measured region. Probe geometry from `probes/train_probe.py` argparse defaults: `batch=64`, `d_model=256`, `n_heads=4`, `head_dim=64`. BF16 q/k/v/β with the gate `g` kept in float32 — that float32 gate *is* the "FP32 recurrence accumulation" the bullet asks for. Throughput is `B*T / t`, real tokens only.
>
> | T | mode | R1 ms | R2 ms | R2/R1 | R1 peak alloc MiB | R2 peak alloc MiB | R1 real tok/s | R2 real tok/s |
> |---|---|---|---|---|---|---|---|---|
> | 128 | forward | 0.162 | 0.205 | 1.26× | 44.1 | 60.2 | 5.05e7 | 3.99e7 |
> | 128 | backward | 3.223 | 3.739 | 1.16× | 785.1 | 882.3 | 2.54e6 | 2.19e6 |
> | 128 | fwd+bwd | 3.386 | 3.914 | 1.16× | 777.1 | 874.3 | 2.42e6 | 2.09e6 |
> | 256 | forward | 0.269 | 0.353 | 1.31× | 88.2 | 120.5 | 6.08e7 | 4.64e7 |
> | 256 | backward | 6.909 | 7.996 | 1.16× | 1570.3 | 1764.5 | 2.37e6 | 2.05e6 |
> | 256 | fwd+bwd | 7.224 | 8.387 | 1.16× | 1554.3 | 1748.5 | 2.27e6 | 1.95e6 |
> | 512 | forward | 0.546 | 0.765 | 1.40× | 176.5 | 241.0 | 6.00e7 | 4.28e7 |
> | 512 | backward | 14.940 | 16.961 | 1.14× | 3140.5 | 3529.0 | 2.19e6 | 1.93e6 |
> | 512 | fwd+bwd | 15.537 | 17.769 | 1.14× | 3108.5 | 3497.0 | 2.11e6 | 1.84e6 |
>
> Peak *reserved* memory tracked allocated closely throughout (T=512 fwd+bwd: R1 3154 MiB reserved vs 3108.5 allocated; R2 3540 vs 3497). The table was produced twice on separate `oat` nodes; timings reproduced to within ~2% and every memory figure was identical, so the third decimal on the millisecond columns is not meaningful. Reading:
>
> - **R2 costs ~1.14–1.16× R1 wall-clock in training, not 2×**, despite consuming exactly 2× the virtual positions. The DP2 arm is operationally affordable at probe scale. Had this been quoted in virtual positions/sec the R2 column would have read ~8.6e7 at T=512 forward and *beaten* R1 — which is precisely the inflation the bullet forbids.
> - **The backward dominates by ~27×** (0.55 ms vs 14.94 ms at T=512, R1) and is where essentially all the memory goes. Forward-only peak is 176–241 MiB; adding the backward takes it to 3.1–3.5 GiB. This is the pure-operator figure at one layer, so a 3-layer probe model plus optimizer state will exceed it — the L40S's 48 GiB has room, but the margin is not unlimited at longer T.
> - Memory scales cleanly linearly in T for both arms; the R2/R1 memory ratio is a steady ~1.11–1.13× in the backward.
> - The forward-mode R2/R1 ratio drifts up with T (1.26× → 1.40×) while the backward ratio drifts *down* (1.16× → 1.14×). The forward is short enough at these sizes to be partly launch-bound, so its ratio is the less trustworthy of the two.
>
> These are single-GPU, single-layer operator numbers on a *free* cluster node, not a Phase-1 throughput budget.

### 4.8 Phase-0 exit package

The implementation owner creates `phase0/<run-id>/p0-decision.md` (§3.3) containing:

1. source/image manifest IDs;
2. complete test command and pass/fail summary;
3. all known deviations from ordinary KDA;
4. numerical error table;
5. R1/R2 memory and throughput table;
6. open issues explicitly marked blocking or non-blocking.

**Phase 0 passes only if every required package P0.0–P0.5 passes.** A failing or skipped check is not a waiver.

## 5. Phase 1 — fresh synthetic mechanism triage

### 5.1 Infrastructure assignment

Use one user-approved p5.48xlarge in us-east-1 only after Phase 0 passes and the [AWS pre-launch checklist](aws-operations.md#mandatory-pre-launch-checklist) is complete. Each H100 runs one independent process. The first eight-worker smoke wave must verify that every GPU runs the same source/image/manifest.

### 5.2 Mandatory harness work before any Phase-1 run counts

The implementation owner must make and test these changes before P1.0:

| Required change | Why it is required |
|---|---|
| **Prerequisite to every other row in this table: put `probes/` under version control.** No harness change below may land first. | Verified: `probes/` is under no version control at all — `git rev-parse` fails there, the only `.git` directories in the workspace are `OLMo-core/.git` and `edullm-data/.git`, and there is no parent repo. Every change below is therefore unrecoverable and unattributable until this is done. Same requirement as §4.2 step 3a, reached from the other direction. |
| Replace the hard-coded `allow_neg_eigval=True` **in the probe harness** with named strict and reflection beta regimes. The lines are `probes/train_probe.py:34`, `:36`, and `:48` — three lines, and **not** in `recurrent.py`. | Current probe results are reflection results, not strict DP2 evidence. **Do not "fix" `recurrent.py`.** Its KDA classes already default `allow_neg_eigval=False` (`:608` for `KimiDeltaAttention`; `:1102` and `:1480` for `KimiDeltaHouseholder`). What defaults `True` there is `GatedDeltaNet` (`recurrent.py:76`, `:447`). The 10 production 7B scripts under `src/scripts/official/OLMo-hybrid/` and `src/scripts/train/OLMo_hybrid/` all pass `allow_neg_eigval=True` **explicitly** at the call site (verified: `OLMo-hybrid-7B-pretrain.py:78` and 9 siblings), so flipping the default would not change them. The binding reason is different: the sites that *do* rely on the default are the tests in `src/test/nn/` and `src/scripts/train/ladder/gemma_like_ladder.py`, so a flip silently moves the test baseline and one ladder config while doing nothing for the probe harness. |
| Add a **per-factor** beta parameterization to the R-factor path. This is new code, not a flag flip. | Beta is computed **once for all \(R\) factors**: `recurrent.py:1257-1259` builds one `(B, T, R * n_heads)` tensor from `self.w_b(x).sigmoid()` and applies the `allow_neg_eigval` doubling to all of it. A single shared flag cannot express per-factor beta regimes, so DP2-budgeted (\(\beta_1=b\pi,\ \beta_2=b(1-\pi)\)) and tied-K require a new parameterization. The arm-dispatch row below understates this work. |
| Add arm-config dispatch for `R1`, `R1-P`, `R1-2step-tiedK`, `DP2-budgeted`, and `DP2-strict` — the canonical IDs of §3.2, which are what the manifest `arm` field accepts. | The current mixer/householder flags cannot express the required controls. |
| Add a common residual FFN to ProbeModel and expose FFN width. | Current ProbeModel has no FFN, so it cannot make same-geometry R1-P. |
| Solve R1-P FFN width from actual non-embedding parameter counts. | Do not use d-model widening as the capacity control. |
| Add recurrence-boundary `R1-2step-tiedK` and `DP2-budgeted` beta behavior. | These are central mechanism controls, not post-hoc labels. |
| Add immutable evaluation banks and a per-run JSON manifest. | Current evaluation generates fresh examples per call and cannot guarantee a shared held-out bank. |
| Add generator **version hashes**. Task self-tests are already **partially** in place — extend them, do not rewrite them. | Prevent degenerate or changed tasks entering the composite. `probes/tasks.py:363-419` already validates group closure orders (S3=6, S4=24, S5=120), target ranges against `out_vocab`, ignore-index exclusion, and a naive-versus-vectorized cross-check of `mod_arith` (`:404`). The genuinely missing half is the version hashes. |

**Already satisfied — do not "re-fix" it.** Ignored targets are already correctly excluded from **both** loss and accuracy. Accuracy: `probes/train_probe.py:70-74` computes `correct & valid` with `valid = y != -100` and divides by `valid.sum()`. Loss: `cross_entropy`'s default `ignore_index=-100` against targets carrying `MQAR_IGNORE = -100` (`tasks.py:184`). Carrying this as an open work item risks regressing correct code.

The Phase-1 runner may not accept an arbitrary free-form flag set. It must accept a manifest-defined arm and emit the exact manifest fields from Section 3.3.

### 5.2.1 Required runner contract

Create or extend the following in-repository interfaces before P1.0:

| Path | Required interface |
|---|---|
| probes/build_dp2_eval_bank.py | Takes one task specification plus evaluation seed, writes a tensor bank and JSON checksum record, and refuses to overwrite a different bank ID. |
| probes/train_probe.py | Accepts --manifest and --out. It reads all architecture, optimizer, task, seed, and beta choices from the manifest; free-form arm/seed overrides are rejected. |
| probes/analyze_dp2_phase1.py | Accepts a manifest index and result root, verifies that every expected run is present exactly once, then emits raw, paired, and decision-summary tables. |
| probes/model.py | Adds a common SwiGLU FFN with explicit ffn_dim so R1-P can match DP2 without changing d_model. |

The canonical training invocation is:

    python probes/train_probe.py \
      --manifest <absolute-path-to-one-frozen-run-manifest.json> \
      --out <absolute-path-to-result.json>

The manifest, not the shell command, is the source of truth. A run must fail before training if its evaluation-bank checksum, source revision, or expected parameter ledger does not match the manifest.

**That invocation cannot be typed today.** `probes/train_probe.py` has no `--manifest` flag at all: `:80-96` is free-form argparse (`--mixer`, `--task`, `--seed`, `--steps`, …). The block above is a specification for work not yet done, which is fine — but nobody should read it as a working command.

**Single-seed limitation.** The harness has exactly one `--seed` (`:84`), and evaluation derives from it (`:65`, `gen = torch.Generator().manual_seed(seed * 100_003 + length)`, called at `:137` with `args.seed`). §5.5's four-way seed map (init / data / task-instance / evaluation bank) has no plumbing to land in, and §5.3 requirement 4 — "no train/evaluation instance collision under the declared seed maps" — is therefore currently **unsatisfiable**, not merely unimplemented. Splitting the single seed into the four declared streams is part of this contract.

### 5.3 Task panel

The following panel is frozen by role, not by a favorable result:

| ID | Task | Role | Current status |
|---|---|---|---|
| T1 | s5_words | primary state-tracking discriminator | implemented |
| T2 | parity | solvable control | implemented |
| T3 | s3_words | hard solvable control | implemented |
| T4 | s4_words | hard solvable control | implemented |
| T5 | mod_arith | structurally different local/state guardrail | implemented |
| T6 | overwrite_conflict | last-write-wins retrieval under conflicting writes | must be implemented and unit-tested |
| T7 | long_gap_retrieval | key/value retrieval after controlled distractor gap | must be implemented and unit-tested |
| T8 | local_delay_composition | local-processing non-degradation guardrail | must be implemented and unit-tested |
| D1 | MQAR ladder | conditional diagnostic only | excluded unless fresh calibration passes |

Required generator specifications for new tasks:

| Task | Input pattern | Scored target | Difficulty variables | Invalid condition |
|---|---|---|---|---|
| overwrite_conflict | Repeated key/value writes, then queries; every queried key has at least two writes | newest value for each queried key | keys, overwrite count, gap, distractors | a query can be solved from only the latest local write |
| long_gap_retrieval | Key/value pairs, distractor span, then randomized key queries | original value for queried key | pairs, gap, vocab, distractors | filler span is below the minimum defined below |
| local_delay_composition | Delimiter-separated short operations with delayed answer token | local composition outcome only | local window, delay, vocabulary | target requires global retrieval rather than local composition |

**`long_gap_retrieval` minimum filler span: at least 16 distractor tokens between the last write and the first query,** adopting the existing `MQAR_MIN_FILLER = 16` (`probes/tasks.py:186`, "below this the task is a lookup, not a recall test"). Below the minimum the task measures local lookup rather than retrieval. The generator must **assert** it, not document it, and the realized filler span must appear in the run manifest.

Every generator must have:

1. deterministic output for a supplied torch generator;
2. a reference implementation or hand-checkable property test;
3. a minimum scored-target assertion;
4. no train/evaluation instance collision under the declared seed maps;
5. a test that deliberately violates its invalid condition and raises;
6. **demonstrated mechanism neutrality.** Requirements 1–5 are all correctness properties; none prevents a task from being architecturally privileged toward a two-write mechanism. Each new task must be shown solvable **at ceiling** by a sufficiently wide R1 or R1-P — at whatever width and step budget it takes, which is a separate demonstration from the §5.5 calibration and is deliberately *not* held to §5.5's 25–75% window. The point is existence: if no single-write model can reach ceiling at any width, the task structure intrinsically favours two writes per token and a DP2 win over it measures the task's construction rather than the mechanism's value. This is required precisely because T6 `overwrite_conflict` is defined by "at least two writes per queried key" while DP2's mechanism **is** two ordered writes per token. Record the demonstrating width, budget, and score alongside the calibration table.

**De-duplicate the MQAR ladder before calibrating D1.** `mqar_d16` (`tasks.py:340-344`) is byte-identical to `mqar_p16` (`:325-329`) — both `max_pairs=16, n_keys=128, n_values=128, n_distractors=128` — even though the comment at `:311-313` states that the d-ladder varies pair count **and** vocabulary together. Calibrating D1 over the declared grid as it stands burns one cell on a duplicate. Remove the duplicate rung, or restore the intended vocabulary step, before any D1 calibration run.

### 5.4 P1.0 — p5 image and worker smoke

**Prerequisite, completed before the smoke wave launches — not inside it.** Generate every evaluation bank the smoke wave will use, record its checksum, and replay it. This cannot be a smoke slot: §5.2.1 requires a run to fail *before training* if its evaluation-bank checksum does not match the manifest, so putting bank generation in the same wave as the training slots that consume it inverts the dependency and every training slot in the wave would legitimately fail. Bank generation is CPU work and does not need the p5 node at all.

Run exactly eight independent smoke jobs, one per GPU. A "short run" means the triage code path at reduced steps — enough to produce a per-job wall-clock and a non-degenerate score, not a single step.

| Slot | Work |
|---|---|
| 0 | R1 on T1 — short run |
| 1 | DP2-strict on T1 — short run |
| 2 | R1-P on T1 — parameter ledger plus short run |
| 3 | R1 on the first implemented new task (T6 or T7) — short run |
| 4 | DP2-strict on the same new task — short run |
| 5 | R1-P on the same new task — short run |
| 6 | DP2-budgeted on T1 — short run |
| 7 | tied-K (`R1-2step-tiedK`) on T1 — short run |

Two changes from a naive allocation, both load-bearing. **At least one new task must appear**: T6, T7, and T8 do not exist yet, and T6/T7 are two thirds of the primary composite, so a smoke wave over T1/T2 only gives the never-executed code zero coverage on the node it will run on. **R1-P gets a full short run**, not one training step: it is the primary comparator in P1.4 condition 3, and one step exercises neither its evaluation path nor its runtime. And **T2 is dropped** from the smoke wave — it is fully covered in P1.2, and the smoke wave's job is infrastructure, not breadth.

**Residual gap, accepted deliberately.** Eight slots cannot cover five arms across three new tasks. `DP2-budgeted` and `R1-2step-tiedK` are smoked on T1 only, and **T8 gets no smoke coverage at all** despite being equally new, never-executed code that P1.4 condition 4 depends on. T8 feeds a guardrail rather than the primary composite, so it carries the first failure on the node rather than in the smoke wave; if T8 is implemented before P1.0, swap it into slot 6. State this in the launch request rather than leaving it implicit — the argument above is that untested code needs coverage, and this is the part that does not get it.

Exit criteria:

- every worker reports the same source/image/manifest;
- no CUDA/Triton architecture error;
- no NaN/Inf;
- parameter counts and beta-range assertions pass;
- the R1-P mismatch is within the frozen tolerance, defined as **within 0.5% of the DP2-strict non-embedding parameter count**. Exact equality is unachievable because FFN width is integer-quantized, so an exact-match criterion would fail unconditionally. Record **both** the solved FFN width and the achieved mismatch (absolute and percentage) in the manifest;
- evaluation-bank checksums, generated in the prerequisite step above, replay exactly;
- **per-job wall-clock is recorded for all eight jobs, and reported per arm.** Feed the **maximum**, not the median, into the cost equation in [aws-operations.md](aws-operations.md#phase-1--p548xlarge) §"Phase 1 — p5.48xlarge" (\(H_1\approx\lceil N/8\rceil t+o\)). A wave's duration is its slowest slot, and this wave is *deliberately* heterogeneous — five arms whose per-step cost differs, with the DP2 arms doing roughly twice R1's recurrence work. That violates the equation's stated "\(N\) equal-duration probe jobs" precondition, so a median-based \(t\) runs ~25–30% low and the launch approval is granted against a biased number. Report the median and the per-arm spread as well, and prefer a per-arm application of the equation where the arm mix is known. That equation's only input is otherwise produced by no step in this runbook, which makes the cost model well-formed and unusable. **No full matrix may launch before \(t\) is measured and the resulting cost estimate is approved.**

Any failure returns the work to Phase 0/harness repair; do not treat a p5 failure as a scientific result.

### 5.5 P1.1 — task calibration

Purpose: select difficulty settings before treatment-arm comparison, not to discover a favorable DP2 task.

Use **R1 and R1-P** — see the headroom criterion below — with calibration bundle IDs 1101 through **1105: five bundles, not three.** Map each bundle deterministically:

| Seed field | Value |
|---|---:|
| model initialization | 100000 + bundle_id |
| training/data stream | 200000 + bundle_id |
| task-instance generator | 300000 + bundle_id |
| held-out evaluation bank | 400000 + bundle_id |

**Why five.** The seed-SD screen below is nearly inert at \(n=3\): P(observed \(s\le20\)pp) = 0.982 / 0.632 / **0.359** for true \(\sigma\) = 10 / 20 / 30pp, so a task with 30pp true seed noise passes a third of the time. \(\sigma_t\) is the single number every downstream design choice depends on — the composite's paired SD, the required effect size, the honest job count — and it cannot be estimated from three samples. Two extra bundles is **32 additional jobs** at four tasks and three grid settings — 24 calibration plus 8 R1-P confirmation, i.e. 4 extra waves, scaling with the grid. It is still the highest-value spend in this plan, because every threshold downstream is a function of \(\sigma_t\), but budget it as 4 waves rather than a rounding error. Five bundles also tightens the SD screen itself: P(observed \(s\le20\)pp) at true \(\sigma=30\)pp falls from 0.359 to 0.224.

**Occupancy.** `aws-operations.md` checklist item 5 makes fewer than six occupied slots a no-launch condition. At five bundles the calibration tail wave holds 4 jobs (60 = 7×8+4) and the confirmation tail wave holds 4 (20 = 2×8+4). Before launch, either obtain explicit program-owner acceptance of the waste, pad the tail by co-scheduling the next task's grid, or run calibration on a cheaper shape — it is R1/R1-P only and does not need H100s.

**This seed-map form is not calibration-local.** It also defines the mapping for the P1.2 triage bundles; see §5.6, which fixes the triage bundle IDs and states what must be byte-identical across arms.

**Declare the difficulty grid before running.** For each new or conditional task the grid must be enumerated — axis names, axis ranges, and the explicit list of candidate settings — in the frozen pre-registration artifact required by §6, and hashed **before** the first calibration job. Until it exists, the calibration wave count in §5.6 is uncomputable and no calibration job may launch.

Then test that grid. Select one setting only if, at the primary long length:

- **R1** mean held-out score is between **25% and 75%** — tightened from 15–85% so both tails stay clear of floor and ceiling, where score differences compress and variance is structural rather than informative;
- **no individual calibration seed scores below 10% or above 90%.** Mean-plus-SD screens alone admit degenerate sets in which some seeds never learned the task: under the former three-bundle design and 15–85% window, `{0,10,35}` gives mean 15.0 / SD 18.0 and passes, as do `{4,4,37}` and `{95,95,65}`. Tightening the window shrinks but does not close that class: at five bundles `{5,30,30,30,31}` has mean 25.2, clearing the 25% floor with one dead seed. So the per-seed floor and ceiling is a separate, binding screen;
- all five runs complete without numerical failure;
- seed standard deviation is at most 20 percentage points;
- **the R1-P five-seed mean also falls inside the 25–75% window at the primary long length.** No separate headroom threshold is imposed: the 75% ceiling already guarantees ≥25pp of headroom whenever both arms are in-window, so a "≥20pp headroom" criterion could never reject anything and is omitted rather than listed as a dead condition. R1-P is run at the **selected** setting only, not at every grid cell: if it lands outside the window the setting is rejected, logged, and the next candidate is tried. That keeps the added cost to five jobs per task. R1-P, not R1, is the P1.4 primary comparator, and it spends the DP2 parameter delta in FFN width — so it should score at least as high as R1. Calibrating on R1 alone lets ceiling compression work **against** DP2: an R1 mean of 80% can put R1-P at 90–95%, leaving under 5pp of headroom and making the +5pp gate in §5.8 condition 3 arithmetically impossible. The stated purpose above guards the opposite direction ("not to discover a favorable DP2 task"); that is not the direction that bites. The operative demand here is that **R1-P be measured** at the selected setting rather than assumed to track R1;
- there are at least 1,000 scored held-out targets **per (task, evaluation bank)**;
- the task’s local/retrieval validity assertions pass.

MQAR remains excluded unless one fixed variant meets every condition. Record every rejected setting in the pre-registration artifact's rejected-settings log. Do not replace a setting with a new one after examining DP2 results.

### 5.6 P1.2 — strict-DP2 triage matrix

After task calibration, freeze the final panel:

- T1 through T8 are required, except that D1/MQAR stays excluded unless it passed P1.1.
- The primary long-memory composite is the equally weighted mean of **T1 and T7** at their frozen longest evaluation lengths. **Each component must also have a nonnegative \(n\)-seed mean individually** — a composite is not a laundering device, and without this clause a gain confined to one component passes identically to a broad one while the other shows nothing.
- **T6 `overwrite_conflict` is a mechanism positive control, not a composite component.** DP2-strict is *required to pass it* — a two-write mechanism that cannot win a last-write-wins task is not working — but passing it is **not evidence that DP2 is better**, and a T6 result may not be reported as such. The reason is structural: §5.3 defines T6 as "every queried key has at least two writes," scored newest-value-wins, and DP2's mechanism *is* two ordered writes per token where the second reads the state the first left. A two-write architecture beating a one-write architecture on a benchmark whose defining property is two writes per key is close to definitional, and its difficulty knobs (keys, overwrite count, gap, distractors) are chosen by the same people who want DP2 to win. Report T6 alongside the guardrails, with an explicit line stating whether DP2-strict passed.
- T2, T3, T4, T5, and T8 are controls/guardrails and are reported separately rather than reweighted into the primary composite.
- Reducing the composite from three components to two raises \(\sigma_d\) by \(\sqrt{3/2}\approx1.22\) at \(\rho_T=0\), so the §5.8.0 sizing must use \(\sigma_d=\sigma_t\sqrt{2(1-\rho)}\sqrt{(1+\rho_T)/2}\) for a two-task composite. That is the honest cost of removing a tautological component, and it is paid in \(n\), not in credibility.

Run all five core arms against every required T1–T8 task with triage bundle IDs **2101 through \(2100+n\)**, where \(n\) is fixed by §5.8.0 from the seed variance P1.1 measures — it is not 3, and it is not chosen before that measurement exists. Each arm receives the same seed bundle for a given task.

**Triage seed map — asserted and manifest-checked, not a convention.** Bundles \(2101\ldots2100+n\) use the same deterministic mapping form as §5.5: initialization \(100000+\mathrm{id}\), data stream \(200000+\mathrm{id}\), task-instance generator \(300000+\mathrm{id}\), evaluation bank \(400000+\mathrm{id}\). Within one bundle the **data stream, the task-instance sequence, and the evaluation bank must be byte-identical across all five arms**; only model initialization may differ. It differs **necessarily**, because the arms have different parameter shapes — R1-P has a wider FFN, DP2 carries extra \(k_2,v_2,\beta_2\) tensors — so the same initialization seed feeds a different-shaped parameter tree and the RNG stream diverges. Nothing else may.

This is quantitative, not hygienic. Identical data and task streams are what buy the pairing correlation \(\rho\), and \(\rho\) is worth more than tripling the seed count: \(\sigma_d=\sigma_t\sqrt{2(1-\rho)}\), so moving \(\rho\) from 0.5 to 0.85 cuts \(\sigma_d\) by about 1.8x, for free. The runner must assert stream identity against the manifest and fail the run on mismatch.

**Triage evaluation banks are a prerequisite to the first triage job.** Banks \(400000+2101\ldots2100+n\) must be generated, checksummed, and size-verified against the target-count requirement **before** any triage job launches — which means \(n\) must be fixed per §5.8.0 before bank generation, not after. Today the ≥1,000-target and score-window guarantees are established on *calibration* banks (seeds \(400000+1101\ldots1105\)) that are then discarded, and nothing verifies the banks actually used in P1.2. State the requirement unambiguously: **at least 1,000 scored targets per (task, evaluation bank)** — not 1,000 pooled across the three banks, which would leave ~333 per seed and materially degrade the per-seed measurement SE.

| Quantity | Value without qualified MQAR | Value with qualified MQAR |
|---|---:|---:|
| Tasks | 8 | 9 |
| Arms | 5 | 5 |
| Paired seed bundles | \(n\), per §5.8.0 | \(n\), per §5.8.0 |
| Independent training jobs | \(40n\) | \(45n\) |
| Minimum p5 waves at 8 workers, **P1.2 triage matrix only** | \(\lceil 40n/8\rceil=5n\) | \(\lceil 45n/8\rceil\) |
| Honest minimum Phase-1 total, **all packages** | \(5n+12\), grid-dependent | \(\lceil45n/8\rceil+12\), grid-dependent |

Worked at the two most likely sizings (see §5.8.0; the 12 is smoke 1 + calibration 8 + confirmation 3):

| \(n\) | Triage jobs | Triage waves | Phase-1 total waves | Cost at 10 min/wave |
|---:|---:|---:|---:|---:|
| 5 | 200 | 25 | 37 | ~$395 |
| 8 | 320 | 40 | 52 | ~$530 |

The triage row counts the triage matrix and nothing else. Total Phase-1 node time is

\[
\underbrace{\lceil N_{\rm triage}/8\rceil}_{5n\text{ or }\lceil45n/8\rceil}
+\underbrace{1}_{\text{P1.0 smoke}}
+\underbrace{\left\lceil\frac{\text{bundles}\times\text{tasks}\times\text{settings}}{8}\right\rceil}_{\text{calibration}}
+\underbrace{\left\lceil\frac{\text{bundles}\times\text{tasks}}{8}\right\rceil}_{\text{R1-P confirmation}}.
\]

The last two terms **cannot be pooled**: §5.5 runs R1-P at the *selected* setting only, and the selection is not known until calibration is read, so confirmation cannot share a wave with calibration. At five bundles, four tasks, and three grid settings that is 60 calibration jobs in \(\lceil60/8\rceil=8\) waves plus 20 confirmation jobs in \(\lceil20/8\rceil=3\) waves — 11, not the 10 a pooled \(\lceil80/8\rceil\) would suggest. So the fixed overhead is \(1+8+3=12\) waves on top of triage: at \(n=5\) the honest minimum is \(25+12=37\) waves against a triage row of 25 — about 48% more node time — and the overhead grows linearly in the grid size while triage grows in \(n\). The calibration term is **uncomputable until the §5.5 difficulty grid is declared**, so the total and its cost estimate must be recomputed and re-approved once the grid is frozen.

Recompute whenever a bundle count changes: the five-bundle calibration in §5.5 raises these two terms from \(5+2=7\) waves to \(8+3=11\).

The deterministic launch order is **bundle ID, then task ID, then arm ID.** Divide that ordered list into contiguous groups of eight. Store the complete ordered list in the Phase-1 manifest before the first job starts.

Task-major order is prohibited, for two reasons. It fully resolves T1 — the declared primary discriminator — by wave 2 of 15, with 87.5% of the budget unspent, no stopping rule and no blinding rule, while [aws-operations.md](aws-operations.md) mandates progress reporting at least every 30 minutes; the primary answer would be visible to the operator with most of the spend still discretionary. And `D1` sorts **before** `T1` lexically, so a qualified MQAR would consume the first ~2 waves on the one task that feeds neither the primary composite nor any guardrail, and hence no decision rule.

Two consequences of bundle-major order to plan for. A wave now spans **two to three tasks**, so 2–3 immutable evaluation banks load per wave rather than one — which is why §5.6 requires every triage bank generated and checksummed before the first job. And the reordering defers the *statistically usable* answer, not the *readable* one: bundle 2101 completes at wave 5 of 15, giving a full single-seed reading of every task and arm with two thirds of the spend uncommitted. Under bundle-major, T1 resolves at wave 11 rather than wave 2, but the absence of a blinding rule remains an open item rather than a solved one.

**No decision, exclusion, difficulty change, or protocol revision may be made on partial results.** If a run must stop for cost or safety, the matrix is declared incomplete and §5.8 condition 2 fails.

### 5.7 P1.3 — phase-1 analysis

For each task and seed bundle, calculate one paired difference:

\[
d_i = m_{i,\mathrm{DP2}}-m_{i,\mathrm{control}}.
\]

**Name the control.** For the primary gate, \(m_{i,\mathrm{control}}\) is **R1-P** — the same comparator §5.8 condition 3 names. Every other contrast reported here (DP2 versus R1, DP2 versus tied-K, budgeted versus strict) must name its own control explicitly in the results table; an unlabelled \(d_i\) is not a reportable quantity.

Use the \(n\) seed-level values only for triage. Do not pool contexts, evaluation examples, checkpoints, or tasks as independent training observations — the seed bundle is the unit of inference, and \(n\) is the sample size, however many evaluation examples each run scores.

Report:

- every raw seed-level score;
- mean, standard deviation, and paired differences;
- task-specific context-length curves;
- beta, beta-sum, decay, key-norm, state-norm, gradient-norm, and clipping distributions;
- parameter count, FLOP ledger, peak memory, and real-token throughput;
- every failed, retried, or excluded run.

**Retry rule — prevention, not just disclosure.** The bullet above and `aws-operations.md`'s "do not rerun a failed seed silently" are *disclosure* rules: nothing in them caps retries, defines what makes a run legitimately failed, or stops a replacement bundle being drawn after a result has been seen. That matters quantitatively. If the effective practice becomes "best 3 of \(3+k\)", one undisclosed redraw raises the null pass rate for §5.8 condition 3 from 0.107 to 0.269 — 2.5x — and *lowers* the likelihood ratio, so the gate becomes weaker evidence while appearing unchanged. The design also **forces** retries: conditions 2 and 5 require all 120 jobs valid, and at a 1% per-job infrastructure failure rate P(≥1 failure) = 0.70. Therefore:

1. A run may be retried **only** for a pre-enumerated infrastructure fault class: CUDA or driver fault, out-of-memory, node preemption, or manifest/evaluation-bank checksum mismatch.
2. The fault must be **detected before any evaluation score is computed for that run.** Once a run has produced a score it may not be retried for any reason. This clause is the load-bearing one: it makes the "redraw a bad-looking seed" loophole mechanically unavailable rather than merely discouraged.
3. A retry reuses the **same bundle ID and the same seeds**. A substitute bundle is never drawn.
4. Every retry records its fault class and its detection timestamp in the manifest.
5. §5.8 conditions 3 and 4 are evaluated only if retries are at most 5% of jobs. Above that threshold the matrix is reported as a contaminated sample and the verdict is inconclusive.

### 5.8 P1.4 — Phase-2 eligibility gate

#### 5.8.0 Seed count is measured, not assumed — the two-stage rule

**The number of triage seed bundles \(n\) is not fixed by this document.** It is derived from the seed variance that P1.1 measures, and the triage matrix is not authorized until that derivation is done and approved. Phase 1 therefore launches in **two approvals**: calibration first, then triage sized from calibration's output.

This is not extra work or extra cost. P1.1 already runs five bundles of R1 and R1-P across the difficulty grid — 80 jobs — and \(\sigma_t\) falls out of that data at **zero marginal cost**. What changes is only that we read the number before betting on it.

**Stage 1 — measure (P1.1, already budgeted).** From the five calibration bundles at the selected setting, report for each task: the per-arm seed SD \(\sigma_t\), and the arm-to-arm correlation \(\rho\) within a bundle. Then form the composite paired SD

\[
\sigma_d=\sigma_t\sqrt{2(1-\rho)}\sqrt{\tfrac{1+\rho_T}{2}}
\qquad\text{(two-task composite: T1 and T7, per §5.6)}
\]

using the measured inter-task correlation \(\rho_T\) of the differences (use \(\rho_T=0\) if it cannot be estimated; that is the conservative direction). For a \(k\)-task composite the last factor is \(\sqrt{(1+(k-1)\rho_T)/k}\) — recompute it if composite membership ever changes. Record \(\sigma_t\), \(\rho\), \(\rho_T\), and \(\sigma_d\) in the pre-registration artifact's second revision (§6.1) **before** any triage job launches.

For orientation, \(\sigma_d\) as a function of what P1.1 might return:

| \(\sigma_t\) | \(\rho=0.5\) | \(\rho=0.7\) | \(\rho=0.85\) |
|---:|---:|---:|---:|
| 6pp | 4.24 | 3.29 | 2.32 |
| 8pp | 5.66 | 4.38 | 3.10 |
| 12pp | 8.49 | 6.57 | 4.65 |
| 20pp | 14.14 | 10.95 | 7.75 |

These are ~22% larger than a three-task composite would give (\(\sqrt{3/2}\)), which is the price of dropping T6. Note what the table implies: at \(\rho=0.7\), staying inside the \(\sigma_d\le5\)pp launchable band needs \(\sigma_t\lesssim9\)pp. The \(\sigma_t\le8\)pp target in §5.5 is therefore not advisory — it is roughly the boundary of feasibility for this design.

\(\rho\) is a **design variable, not a fact**: it is bought by making the data stream, task-instance sequence, and evaluation bank byte-identical across arms within a bundle, which §5.6 already requires and asserts. Moving \(\rho\) from 0.5 to 0.85 is worth about 1.8x in \(\sigma_d\) and roughly 4x in required \(n\), for no compute at all. Buy it before buying seeds.

**Stage 2 — size, then launch.** Choose the smallest \(n\) that satisfies **both** requirements below, and enter it in the manifest:

- **power** ≥ 0.80 to detect a true +5pp composite effect under condition 3;
- **decidability** ≥ 0.80, i.e. P(the two-sided 90% CI half-width \(<\delta=3\)pp) ≥ 0.80, so the equivalence rows can return a verdict rather than "underpowered."

Decidability, not power, is the binding constraint — sizing for power alone leaves the equivalence test undecidable about half the time. The resulting schedule:

| Measured \(\sigma_d\) | \(n\) | Triage jobs | Total waves | Cost at 10 min/wave | power | decidability |
|---:|---:|---:|---:|---:|---:|---:|
| ≤2.0pp | 4 | 160 | 32 | ~$350 | 1.00 | 0.82 |
| ≤2.5pp | 5 | 200 | 37 | ~$395 | 0.99 | 0.82 |
| ≤3.0pp | 6 | 240 | 42 | ~$440 | 0.98 | 0.81 |
| ≤3.5pp | 8 | 320 | 52 | ~$530 | 0.98 | 0.88 |
| ≤4.0pp | 10 | 400 | 62 | ~$625 | 0.98 | 0.91 |
| ≤5.0pp | 12 | 480 | 72 | ~$715 | 0.95 | 0.80 |
| **>5.0pp** | — | — | — | — | — | **do not launch** |

**The last row is a real outcome, not a formality.** Above \(\sigma_d=5\)pp the requirement passes \(n=16\) (640 jobs, ~$900) and reaches \(n=30\) (1,200 jobs, ~$1,540) by 8pp — at which point the honest report is that *this design cannot answer the question at a defensible cost*. Do not launch a matrix that is known in advance to be underpowered, and do not weaken \(\delta\) or the +5pp floor to make an affordable \(n\) appear adequate. Write the negative feasibility memo instead, and revise the protocol: longer runs per job, tasks less prone to saturation, more evaluation mass per run, or a different endpoint. Every one of those attacks \(\sigma_d\), which is the variable that actually controls whether Phase 1 can conclude anything.

Note that \(n=3\) appears nowhere in this table. It is not merely underpowered for condition 3 (21% power at \(\sigma_d\approx11\)pp); its equivalence half-width requires \(s_d<1.78\)pp, so the tie rows return "underpowered — no decision" between 78% and 99% of the time across the plausible range. Three bundles cannot carry this gate in either direction.

#### 5.8.1 Conditions

This is a resource-allocation rule, not a confirmation claim. Phase 2 becomes eligible only if every condition below holds:

1. Phase 0 passed without an unresolved semantic issue.
2. All required triage jobs are present, valid, and manifest-complete — the full \(N_{\rm triage}=8\times5\times n\) set at the \(n\) fixed by §5.8.0, or \(9\times5\times n\) when qualified MQAR is included.
3. **DP2-strict beats R1-P on the primary composite by a margin the data can actually support:** the one-sided 95% lower confidence bound on the paired difference is positive, \(L_{95}(d)=\bar d-t_{0.95,n-1}s_d/\sqrt n>0\), **and** \(\bar d\ge+5\)pp. The bound is the statistical test; the +5pp floor is the separate practical-relevance requirement, and both must hold. There is no "all differences nonnegative" clause — a sign test at \(n=3\) has a floor \(p\) of \(1/8=0.125\) and so cannot reach significance at any observed value, and the requirement gets *harder* as \(n\) grows (P(all nonnegative) falls 0.78 → 0.66 → 0.51 for \(n=3,5,8\) at a true +5pp effect), which would penalize the very sample-size increase that makes the test valid.
4. **DP2-strict is non-inferior on every guardrail:** for each of T2, T3, T4, T5, T8, the one-sided 95% *upper* bound \(U_{95}(d_g)=\bar d_g+t_{0.95,n-1}s_g/\sqrt n\ge-2\)pp. A guardrail fails only when the data **affirmatively demonstrate** a loss worse than 2pp, not merely fail to exclude one. A point comparison of the three-seed mean against \(-2\)pp is not usable here: with a truly neutral DP2 the mean-based form trips at least one of the five guardrails 90% of the time at \(\sigma_d=10\)pp and 94% at 20pp, because \(-2\)pp sits about 0.2–0.35 SE from zero. The bound form cuts that false-trip rate roughly 30-fold.
5. No arm has a NaN/Inf, unexplained OOM, or beta-range violation.
6. The **Phase-1 statistical verifier** (see §6) verifies that the result is not explained solely by more parameters, more total beta, or tied-key sequential updates.
7. The program owner signs phase1-decision.md and explicitly authorizes Phase-2 planning.

**Caveat on condition 6, for the tied-K contrast specifically.** Tied-K has strictly **fewer free parameters** than DP2-strict — \(k_2\) is not free — so any DP2-over-tied-K gap is *also* consistent with "more parameters," which is exactly the confound R1-P exists to control. R1-P does not control it for this contrast: R1-P is matched to DP2-strict, not to tied-K. State this in the memo so the rank-two claim is not adjudicated by a control that is knowingly unmatched for it. If the rank-two claim is to be made cleanly, a parameter-matched tied-K variant must be nominated in Phase 2, not inferred here.

**Every row below is an equivalence or superiority test with a named control and a numeric threshold.** "Ties," "matches," and "favorable" are otherwise decided by eye, and at \(n=3\) they are decided by noise. Throughout: \(\delta = 3\) percentage points, and the interval is the two-sided 90% CI of the **paired** difference on the primary long-memory composite.

| Observation | Numeric definition | Permitted decision |
|---|---|---|
| Strict DP2 clears all gates and budgeted DP2 is also favorable | "Favorable" = DP2-budgeted versus control **R1-P**, paired-difference mean \(\ge+5\)pp with the one-sided 95% lower bound \(>0\) — the same bar condition 3 sets for DP2-strict, same control | Phase 2 may test practical and mechanistic claims |
| Strict DP2 clears but budgeted DP2 does not | Budgeted fails the bar above while strict passes it | Phase 2 may test strict DP2 as a performance candidate; label extra write mass as an unresolved explanation |
| **DP2 ties `R1-2step-tiedK`, or R1-P matches DP2 — one row, one consequence** | "Ties"/"matches" = the two-sided 90% CI of the paired difference lies entirely within \([-\delta,+\delta]\), \(\delta=3\)pp. Containment already forces the half-width below \(\delta\) (subtract the bounds: \(2h\le2\delta\)), so no separate width clause is needed and none should be added. A CI too wide to be contained is adjudicated by the underpowered bullet in the stopping rule below, never as a tie | Stop the DP2 mechanism program; the adequate explanation is capacity, or parameterization within the R1 function class |
| Any required gate fails | — | Do not start Phase 2; write a negative or inconclusive decision memo per the stopping rule below |

The tied-K and R1-P rows are merged because **tied-K is R1-equivalent**: its state update is exactly rank 1 (measured 2nd singular value 2.26e-15) and collapses to a single R1 step with \(\beta_{\rm eff}=\beta_1+\beta_2-\beta_1\beta_2\|k\|^2\), per §3.2. So "DP2 ties tied-K" means DP2 ties a model in the same function class as R1, differing only in parameterization and parameter count — which is the program-stopping observation, not the weaker "may not claim rank-two geometry." Containment is load-bearing in the other direction: without a width requirement a three-seed null result would satisfy "ties" by noise and the program-killing row would fire on ignorance rather than on evidence. Containment supplies that requirement for free — it is not a second clause.

**Sizing warning, and the reason \(n=3\) cannot carry this rule.** At \(n=3\), \(t_{0.95,2}=2.920\), so the half-width is \(1.686\,s_d\) and containment within \(\pm3\)pp demands \(s_d<1.78\)pp. Measured against the noise this document tolerates elsewhere, that almost never happens: P(a decision is reachable at all) is 0.219 at \(\sigma_d=3.58\)pp, 0.026 at \(10.95\)pp, and 0.008 at the \(\sigma_t=20\)pp/\(\rho=0.5\) baseline §5.5 currently permits. So with \(n=3\) the verdict is "underpowered — no decision" with probability 0.78 to 0.99, and the 120-job matrix buys a non-answer. **Size the matrix from the \(\sigma_t\) that P1.1 measures; do not assume \(n=3\) suffices.** At \(\sigma_d=3.58\)pp, \(h<3\)pp needs roughly \(n=5\)–6.

**Stopping and inconclusive verdicts.** Porting the Phase-2 rule ([phase-2-deferred.md](phase-2-deferred.md) §6) into Phase 1, where saturation remains live because calibration admits scores up to the top of its window and is measured on R1/R1-P only — not on the treatment arms, which may saturate where the controls did not: **if the endpoint is saturated, floor-limited, or mixture-like, stop and issue a new protocol rather than forcing a mean-based analysis.** A result is **inconclusive** — a permitted verdict in §6 item 7 — when any of the following holds:

- the two-sided 90% CI of the primary paired difference spans both \(0\) and \(+5\)pp, so neither condition 3 nor the equivalence rows above can be decided;
- the CI half-width exceeds \(\delta = 3\)pp, so no equivalence row can be decided (underpowered, per the merged row);
- **the CI lies entirely above \(+\delta\) but the mean is below condition 3's \(+5\)pp gate** — a real but sub-threshold effect. Without this clause the band between \(+3\)pp and \(+5\)pp matches no row: a tightly-measured \(+4\)pp win (say \(d=\{3.5,4.0,4.5\}\), 90% CI \([+3.16,+4.84]\)) fails condition 3, is not contained in \([-\delta,+\delta]\), spans neither \(0\) nor \(+5\), and is not underpowered — so it would default to "stop" and a genuine positive result would kill the program. At a true \(+4\)pp effect and \(\sigma_d=3.58\)pp this lands in roughly 8% of triples, so it is a live case rather than a corner;
- any arm's primary-composite \(n\)-seed mean is above 90% or below 10% at the primary long length (saturated or floor-limited), regardless of the paired difference;
- retries exceed 5% of jobs (§5.7);
- the matrix is incomplete for any reason (§5.6).

Inconclusive is not a negative architecture result and may not be reported as one. It requires a new protocol, not a rerun of this one.

## 6. Phase-1 closeout package

The statistical reviewer must produce phase1-decision.md and phase1-decision.json containing:

1. exact source/image/manifests and task-bank checksums;
2. calibration table and all excluded task settings;
3. triage matrix completeness table;
4. raw results and paired analysis;
5. all planned claims and nonclaims;
6. the P1.4 gate verdict for each condition;
7. a clear one-line next action: “Phase 2 eligible,” “stop,” or “new protocol required.” "Inconclusive" resolves to "new protocol required" and must cite the triggering clause from §5.8;
8. **the digest of the Phase-1 pre-registration artifact**, quoted verbatim from the Phase-1 manifest (see below);
9. **the named Phase-2 candidate arm**, if the verdict is "Phase 2 eligible": exactly one arm, with its factor mode, beta initialization, decay mode, and the full non-embedding parameter ledger. [phase-2-deferred.md](phase-2-deferred.md) §2 item 3 makes this an **entry condition** for Phase 2, but no P1.4 condition produces it — P1.4 emits only a binary eligible/not, and §5.6 runs both DP2-strict and DP2-budgeted. Without this item a team can satisfy every P1.4 condition, sign the memo, and then be blocked at the Phase-2 gate with no defined procedure for choosing between the two DP2 arms. The naming rule: the arm that cleared condition 3 against R1-P; if both DP2 arms cleared it, DP2-budgeted is named, because it clears the extra-write-mass explanation that condition 6 exists to exclude.

### 6.1 Pre-registration artifact

The statistical reviewer writes a single content-hashed pre-registration artifact and stores its digest in the Phase-1 manifest. §3.3's `source` field already carries checksum semantics — reuse that form. `phase1-decision.md` must quote that digest (item 8 above), and any protocol change after a hash is recorded as a **new revision with every prior digest retained**, never an edit.

It is hashed in exactly two revisions, because two different deadlines bind. **Revision 1, before the first calibration job:** the task panel, the difficulty grid, and the seed maps — everything §5.5 must not be free to change once calibration scores are visible. **Revision 2, before the first triage job:** revision 1 plus the measured \(\sigma_t\), \(\rho\), \(\rho_T\), and \(\sigma_d\) with the \(n\) derived from them per §5.8.0, the selected settings, the rejected-settings log, the composite membership and weights, the named controls, and every numeric threshold. Both digests go in the manifest and both are quoted in the decision memo.

Together the two revisions must contain:

- the frozen task panel, including which of T6/T7/T8 were implemented and which were dropped;
- the primary-composite membership and weights;
- the declared difficulty grid per task (§5.5) and every selected setting, **with the complete rejected-settings log**;
- the §5.5 calibration and §5.6 triage seed maps, with bundle IDs;
- the named control for **every** §5.8 interpretation row and for every §5.7 contrast;
- every numeric threshold: the +5pp gate, \(\delta=3\)pp, the 25–75% calibration window, the 10%/90% per-seed bounds, the 0.5% R1-P parameter tolerance, the 5% retry cap.

Reason: "frozen by role, not by a favorable result" (§5.3) and "record every rejected setting" (§5.5) are aspirations with no verification mechanism. A rejected-settings log written after the results are seen is byte-indistinguishable from one written before. A pre-committed digest is what makes the difference checkable. The runbook already applies exactly this discipline to source code in P0.0 (§4.2 — patch bundle plus checksum before any GPU work); this extends it to the protocol, which is the thing an unfavorable result actually creates pressure to change.

### 6.2 Roles for the decision package

§5.8 condition 6 requires a verifier and §6 requires a producer, and they must not be the same person or the separation-of-duties rule is self-defeating.

| Function | Role | Constraint |
|---|---|---|
| Produces `phase1-decision.md` / `.json` and the §6.1 pre-registration artifact | statistical reviewer | as defined in the [README](README.md) role table |
| **Phase-1 statistical verifier** — verifies §5.8 condition 6 and that the decision package matches the pre-registered artifact | a second named person holding the statistical-reviewer or program-owner role | **must be distinct from the producer of the memo *and* from whoever selected the §5.5 difficulty settings** — the second clause carries over the README rule ("reviewed by someone other than the person who selected the task difficulty") and is not weakened by this row. Named in the Phase-1 manifest before the first triage job |

The "Phase-1 reviewer" wording previously used in §5.8 condition 6 named no role in the README table (program owner / implementation owner / experiment operator / statistical reviewer) and is replaced by the verifier row above.

Do not write a Phase-2 training manifest until this package is complete and approved.
