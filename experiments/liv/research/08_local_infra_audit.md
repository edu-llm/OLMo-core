# Local Infrastructure Audit — what the LIV experiment can reuse

Date: 2026-07-30. Method: direct inspection of `/Users/ericwu/Developer/Capstone_LLM`.

## Headline

**There is already a design-reviewed experiment protocol in this repo, dated today, that covers
the same architecture — and it explicitly defers all three of the brainlift's proposals.**

- `docs/liv-kda-gqa-sub500m-experiment.md` (521 lines, "DESIGN REVIEWED", version label
  `liv_kda_gqa_sub500m_v1`)
- `docs/kda-liv-architecture-redesign.md` (306 lines)

Both cite the brainlift PDF directly (`kda-liv-architecture-redesign.md:283-285` references
"Eric LIV Brainlift ... pp. 1-6: LIV equations, state and parameter claims, factorization, and the
multiscale/router caveats"). So the brainlift has already been through one design review pass.

The three brainlift proposals appear in those docs as **explicit deferrals**, not as the
experiment:

| Brainlift proposal | Status in existing protocol | Stated reason |
|---|---|---|
| §3.1 Factorized gates | Deferred (`liv-kda-gqa-sub500m-experiment.md:206`) | "Changes parameters, thin-matmul behavior, and quality simultaneously. It may be slower despite having fewer parameters." Gated on: "L0 survives and a full-gate control is available." |
| §3.2 Cross-layer KV sharing | Deferred (`:214`) | "Separated GQA layers operate on transformed residual streams, so borrowed K/V may be stale." Gated on: "Test independently after the base topology is established." |
| §3.3 Multiscale/routed conv | Deferred (`:207`) | "Soft routing evaluates every branch and does not itself save compute." Gated on: "A fixed LIV baseline survives and a systems harness exists." |

The redesign memo repeats these at `kda-liv-architecture-redesign.md:182-195` under "Explicit
Deferrals", and the ladder at `:234-248` places gate factorization at **Gate 2**, after **Gate 1**
topology survival.

**Consequence for the design:** the brainlift's three changes are Gate-2+ work in an existing
plan whose Gate 0 is not yet passed. Any new experiment design should either (a) slot into that
ladder, or (b) argue explicitly for re-sequencing. It should not be written as if greenfield.

## 1. Frozen geometry already chosen

`liv-kda-gqa-sub500m-experiment.md:84-96` freezes an LFM2.5-350M-shaped anchor:

| Field | Frozen value |
|---|---:|
| Decoder layers | 16 |
| `d_model` | 1,024 |
| Query heads / KV heads / head dim | 16 / 8 / 64 |
| LIV conv | causal depthwise, kernel 3 |
| Vocabulary | 65,536 (tied embed/LM head) |
| Effective SwiGLU branch width | 4,608 |
| Context target | 32,768 |

Note `:97`: the LFM2.5 config field `block_ff_dim=6656` is transformed by the implementation into
an **effective per-branch SwiGLU width of 4,608** — a real trap for a reimplementation.

Layer schedule (`:107-111`), GQA at `[2,5,8,10,12,14]`, final mixer is LIV:

```
layer:  0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
L0:     LIV  LIV  GQA  LIV  LIV  GQA  LIV  LIV  GQA  LIV  GQA  LIV  GQA  LIV  GQA  LIV
```

Parameter ledger already computed (`:138-153`): `L0` = 354,483,968 params; stock LIV mixer at
d=1024 = 4,197,376 params.

**Important:** this is d=1024, whereas the brainlift's parameter arithmetic (16.783M LIV vs
10.486M GQA) is at d=2048. The two documents are at different scales — the brainlift's numbers
correspond to the LFM2-1.2B geometry, the local protocol to the 350M geometry. Any new design
must state which scale it uses.

## 2. What exists in OLMo-core — and the two hard blockers

`OLMo-core/` is a vendored, **heavily modified** copy of AI2's OLMo-core (28 modified files,
uncommitted, on a merge commit `f17824e`). It has an `src/edullm/` package added.

### The extension point is clean

`src/olmo_core/nn/attention/base.py:20` defines `SequenceMixer` (abstract: `apply_tp`, `apply_cp`,
`num_flops_per_token`, `init_weights`) and `:67` defines `SequenceMixerConfig` (`num_params`,
`build`), a `Registrable`. Registration is by decorator:

```python
@SequenceMixerConfig.register("kimi_delta_attention")   # recurrent.py:904
@SequenceMixerConfig.register("gated_delta_net")        # recurrent.py:411
@SequenceMixerConfig.register("attention")              # __init__.py:188
```

`nn/transformer/block.py:115,129` takes a `sequence_mixer: SequenceMixerConfig` and comments that
`self.attention` "could contain any `SequenceMixer` implementation". **So adding a `LIVMixer` is a
register-a-new-config job, not a refactor.** This is the single most valuable reusable asset.

There is already `nn/convolution.py:12` — a `CausalConv1d(nn.Conv1d)` (102 lines). **Inspected: it
is NOT directly usable as the LIV conv, for three concrete reasons.**

```python
super().__init__(in_channels=hidden_size, out_channels=hidden_size,
                 kernel_size=kernel_size, groups=hidden_size,   # depthwise: good
                 bias=bias, padding=kernel_size - 1, ...)       # causal pad: good
...
output = dispatch_causal_conv1d(x=x, weight=weight.squeeze(1), bias=bias,
                                activation=self.activation,     # defaults to "silu"
                                backend=self.backend, cu_seqlens=cu_seqlens)
return output[0]                                                # <-- discards final state
```

1. **Baked-in activation.** `activation: Literal["silu","swish"] | None = "silu"` is applied
   *inside* the fused kernel. LFM2's LIV applies gates multiplicatively around the conv; a fused
   SiLU is not the same operator. This is exactly the trap `liv-kda-gqa-sub500m-experiment.md:231`
   warns about: "Do not reuse the current FLA causal convolution unchanged: its SiLU behavior is
   not sufficient evidence of LIV equivalence." Verified in code.
2. **No dilation parameter.** `nn.Conv1d` is constructed without `dilation`, and padding is
   hardcoded `kernel_size - 1` (correct only for dilation=1). **The brainlift's §3.3 dilated
   multi-branch proposal cannot use this class or its fused backend at all** — it needs
   `padding = dilation * (kernel_size - 1)` and a backend that accepts dilation. Whether
   `causal-conv1d` supports dilation is a separate open question (being researched).
3. **`return output[0]` drops the returned final state** — the same bug class as the KDA
   final-state discard, so decode caching would need reworking here too.

It does have a useful `apply_cp` (Ulysses channel-parallel) implementation worth copying.

### Blocker 1 — no LIV mixer exists

`grep -riE "\bLIV\b|lfm2|conv_L_cache"` over `OLMo-core/src/` returns **zero matches**. Confirms
`liv-kda-gqa-sub500m-experiment.md:488`. The LIV mixer must be written from scratch.

### Blocker 2 — hybrid cached generation is asserted impossible

`generate/generation_module/transformer/generation_module.py:108-116`:

```python
def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
    for block in self.model.blocks.values():
        assert isinstance(block.attention, Attention)     # <-- blocks any non-attention mixer
        attn = cast(Attention, block.attention)
```

and again in `free_inference_cache` (`:120-123`). The docstring even acknowledges the problem
("not all models use key-value caches ... For example, Mamba requires cache state but doesn't use
a kv-cache") but the code asserts otherwise. **Any decode-latency or cache-size measurement is
blocked until this is replaced with a typed per-layer state API.** Both design docs specify that
API (`kda-liv-architecture-redesign.md:208-220`, `liv-kda-gqa-sub500m-experiment.md:243-258`).

### Blocker 3 — KDA discards final recurrent state

Documented at `liv-kda-gqa-sub500m-experiment.md:490` and `:517`. Only matters if KDA arms are
retained; not on the critical path for a pure LIV experiment.

### Existing mixers available as baselines/controls

`nn/attention/recurrent.py`: `GatedDeltaNet` (`:33`), `KimiDeltaAttention` (`:566`),
`KimiDeltaHouseholder` (`:1041`) + Triton kernels in `kda_householder.py`. Plus `Attention` and
`FusedAttention` in `attention/__init__.py`. So the **recurrent-hybrid baseline family the
brainlift asks for (§4 "Mamba, gated-delta, Griffin, Jamba, Hymba") is partly already
implemented** — gated-delta and KDA are in-tree and GPU-validated.

Also present: `nn/moe/`, `nn/rope.py`, `nn/hf/config.py` (HF conversion), `model_ladder/base.py`,
`internal/ladder.py`, and ladder training scripts under
`src/scripts/train/ladder/2026Q1/` (`gated_attn_ladder.py`, `gnope_ladder.py`,
`init_style_ladder.py`, `gated_attn_gnope_ladder.py`) — i.e. **an existing pattern for running
architecture-variant ladders**, which is exactly the shape of this experiment.

## 3. Prior completed experiment — KDA Householder (reusable methodology)

`KDA/HANDOFF.md` (611 lines, updated 2026-07-26): **"Status: COMPLETE. 278 tests pass on GPU,
98/98 probe runs, 3 audits closed, 0 failures."** Householder-on-KDA (DeltaProduct R factors ×
KDA per-channel gate).

Headline result — R=4 vs R=1 on S5 (non-solvable) minus parity (solvable), paired, n=8:
+58.15pp interaction at length 128, SIG at all seven lengths. Plus a solvability control (S3/S4
solvable-but-hard are all ns) that kills the difficulty confound.

**This is the methodological template the LIV experiment should copy.** Specifically reusable:

- **Paired-seed design with an explicit capacity control.** `KDA/lm/train_lm.py` docstring: a third
  arm `r1_wide` (R=1 with d_model raised to match R=4's params) separates "R helps" from "more
  parameters help". The LIV factorization experiment needs the mirror-image control.
- **A hard-won power lesson.** From `KDA/lm/train_lm.py`: "The DeltaProduct paper's one strictly
  parameter-matched LM comparison is **+0.0053 nats**, which needs roughly **n=43 seeds** to
  detect. Measured on this codebase, seed-to-seed variance at ~1M params swamps a 4x change in
  task difficulty (one-way ANOVA: eta^2 = 5.9%, F(3,16) = 0.337, ns)." → **val-loss contrasts at
  small scale are near-certainly underpowered.** This directly threatens the brainlift's
  "non-inferior held-out CE" gates and argues for large-effect endpoints (extrapolation, recall)
  as primary.
- **Bimodality finding.** `KDA/run_mqar_var.sbatch:8-19`: MQAR trainability at ~1M params is
  **bimodal** — a run either finds the recall algorithm or stays at chance. So "the correct
  endpoint is SUCCESS RATE over seeds", not mean accuracy. Critical for designing the recall
  composite.
- **6-level verification chain** (`HANDOFF.md`): naive ref → fp64 gradcheck → manual backward →
  emulator → Triton → GPU acceptance, with bit-exactness at each level. Template for validating a
  new LIV kernel.

## 4. Reusable evaluation / probe harness

`probes/` (~30 files) — a working small-model probe trainer:

- `probes/train_probe.py`, `probes/tasks.py`, `probes/model.py` — the harness.
- `probes/mqar_patch.py` — **MQAR implementation with the two difficulty axes already
  decoupled**: number of pairs D (capacity, capped via `MQAR_MAX_PAIRS`) vs retention distance
  (filler). Layout `k1 v1 ... kD vD <filler> SEP q1 ... qD`, disjoint vocab ranges, keys sampled
  without replacement, `-100` ignore_index masking, patched `evaluate()`. **Directly reusable for
  the brainlift's MQAR requirement.**
- `probes/mqar_hard_patch.py` — a harder variant (also mirrored at `KDA/mqar_hard_patch.py`).
- Tasks available: parity, S3/S4/S5 word problems, mod_arith, MQAR — i.e. state-tracking and
  associative-recall probes.
- `KDA/analyze_mqar.py`, `analyze_mqar_var.py`, `analyze_mqar_anova.py` — analysis incl. ANOVA and
  per-seed/success-rate reporting.
- 14 `audit_exp*.py` scripts — determinism, drift, boundary, memory, stability, power, tolerance
  audits. A ready-made rigor checklist.

**Gap:** no lm-eval-harness integration found, and no RULER/needle/passkey implementation. Those
must be built or imported.

## 5. LM training harness + corpus pipeline

`KDA/lm/` — a small self-contained LM trainer independent of OLMo-core:
- `prepare_data.py` — **tokenizes FineWeb-Edu into flat uint16 `.npy` memmaps** (GPT-2 vocab,
  EOS between documents, held-out val taken from the END of the stream, streaming download).
- `train_lm.py` — ~50M-param trainer; `FlatWindowLoader` samples fixed-length windows at seeded
  random offsets **so all arms see identical data order at a given seed** (paired contrast).
- `run_lm_grid.sbatch`, `analyze_lm.py`.

Its primary endpoint is **length extrapolation** (train at 2048, eval at 4096/8192/16384) — chosen
precisely because val loss lacked power. Reusable design decision.

Note the EOS-only concatenation: windows can splice documents. For long-range recall work,
document-isolated packing is needed (the OLMo-core protocol calls for exactly this,
`liv-kda-gqa-sub500m-experiment.md:281`).

## 6. Corpora available (large, already tokenized)

`docs/dataset-creation/s3-dataset-audit-2026-07-28.md`: **~1.63 TB of datasets across 6 buckets in
`sbsandbox` (056956104102)**, read-only via the `sb-aws` MCP broker. In `edullm-datasets` (1.57 TB):

| Prefix | Size | Tokens | Format |
|---|---|---|---|
| `olmo-150b-dolma2/` | 633 GB | **155.6B tok** | `part-NNN-00000.npy` + `.csv.gz` sidecar, no checksums |
| `olmo100b/olmo-mix-1124-30b/` | 532 GB | — | `NNNNN__domain__src__hash10.npy` |
| `olmo30b/olmo-mix-1124-30b/` | 183 GB | — | same inner naming |
| `mythos-rdt/` | 65 GB | — | cosmo2 uint16, `shard_NNNNN.bin` + sha256 |
| `regmix/regmix-10b/` | 53 GB | 10B tok | one `.json.gz` per domain |
| `datamix1-jul22/` | 38.5 GB | 9.28B tok | CAS `objects/sha256.npy` + `views/` |
| `curriculum-p1-jul23/` | 37.5 GB | — | `part-NNNNN-of-NNNNN-sha256.u32le.bin`, "BEST ENGINEERED" |

**A pretraining corpus is not a blocker.** 155.6B Dolma2 tokens far exceeds what a sub-500M
architecture study needs (protocol budgets are 2-5B tokens/run).

Caveats: no versioning/tags/lifecycle on most buckets, no checksums on the 150B set, and the audit
flags `edullm-checkpoints` as misnamed (holds ~25 GB of datasets). Also **unknown**: the document
length distribution — needed to confirm 16K/32K-capable long documents exist. Worth checking
before promising long-context results.

`edullm-data/` is the airlock repo (CLAUDE.md, HANDOFF.md, `families/`, `infra/`, `skill/`,
`pyproject.toml`, `tests/`) implementing the dataset standard —
`docs/dataset-creation/DATASET-STANDARD.md`. Use it for publishing any new eval/probe set.

`docs/corpus-pipeline/` + `IMPLEMENTATION_AGENT_PROMPT.md` describe a separate, larger program: a
250-260B GPT-2-equivalent reservoir with nested 10/20/30B views, built in `edu-llm/dolma`. Note
its constraint: "not permission to download the production corpus, create AWS resources, submit
Batch or EMR jobs". Not needed for this experiment.

## 7. Compute

Two targets, and the docs disagree — a live decision the design must settle:

- **FarmShare (Slurm)** — what prior work actually ran on. `KDA/run_mqar_*.sbatch`:
  `-p gpu --gres=gpu:1 -c 8 --mem=48G -t 06:00:00`, work dir `/scratch/users/ericrcwu/kda/probes`,
  array-worker sharding with `SKIP`-if-exists idempotency. `HANDOFF.md` says GPU-validated on
  **L40S**. Also `scripts/farmshare_run_kda_probe.sh`, `probes/b200_worker.sh`,
  `handoffs/pedagogical-sft-farmshare-auto-pilot/`, and an `operate-farmshare` skill.
- **SB-AWS** — `liv-kda-gqa-sub500m-experiment.md:5` sets "Execution target: SB-AWS CUDA/Linux",
  and §10 requires a read-only preflight before any resource is created. No accelerator type or
  cost is prescribed. Corpora live in `sbsandbox` S3, which favors AWS for data locality.

A single L40S/A100-class GPU is plausible for 350M × 2-5B tokens per run, but the protocol asks
for 5 screening + ≥8 confirmation **paired** seeds across ≥4 arms — that is dozens of runs and is
the real budget question. `docs/pre-launch-throughput-checklist.md` (256 lines) exists; read it
before sizing.

W&B: project `eduLLM/test`, online logging works from FarmShare GPU nodes (`wandb-primary` skill).
`docs/scaling-audit/` has `pull_wandb.py`, `wandb_history.csv.gz`, `refit.py`, plus
`ARITHMETIC_CEILING.md` and `OVERFITTING_AND_CAPABILITY.md`.

## 8. Other context

- `src/`, `index.html`, `vite.config.ts`, `package.json` — a Vite/TS web app (workbook/report
  viewer; `scripts/build-workbook-data.mjs`). Unrelated to training.
- `P4 Validating Learning Science - *.docx` (5 revisions) + `Qwen3_Project_Recommendation.docx` +
  `docs/curriculum-evaluation-swarm-experiment-prd.md` — a **separate capstone track** on
  learning-science validation / curriculum evaluation. The brainlift's §2 "Learning Science as an
  Inspiration" connects to it rhetorically. Binary; not read here.
- `pipelines/week1_corpus`, `week1_curriculum`, `week1_datadecide` — DataDecide-style ablation
  pipelines. Worth inspecting for the ablation-protocol pattern.
- `papers/` — includes `co-lmlm_continuous-query_limited-memory_language_models_arxiv_2607.07707v1.pdf`
  (limited-memory LMs — likely relevant) and an `obsidian-vault`.
- **Brainlift→experiment pattern is established:** `Brainlifts/*.pdf` →
  `docs/<topic>-architecture-redesign.md` → `docs/<topic>-experiment.md` with a Material Passport
  header (origin skill, origin mode, date, verification status, version label, scope). A new
  design should match that format. Sibling brainlifts:
  `brainlift_hybrid_ssm_attention.pdf`, `brainlift_worked_examples_cot.pdf`.

## What can be reused vs must be built

### Reuse directly
| Asset | Path |
|---|---|
| Mixer registration pattern (`SequenceMixerConfig.register`) | `OLMo-core/src/olmo_core/nn/attention/base.py:67` |
| GQA baseline (`Attention`, `FusedAttention`) | `OLMo-core/src/olmo_core/nn/attention/__init__.py:331,981` |
| Recurrent-hybrid baselines (GDN, KDA, KDA-Householder) | `OLMo-core/src/olmo_core/nn/attention/recurrent.py:33,566,1041` |
| MQAR with decoupled capacity/distance axes | `probes/mqar_patch.py`, `probes/mqar_hard_patch.py` |
| Probe trainer + state-tracking tasks | `probes/train_probe.py`, `tasks.py`, `model.py` |
| Per-seed / success-rate / ANOVA analysis | `KDA/analyze_mqar_var.py`, `analyze_mqar_anova.py` |
| Paired-seed LM trainer + extrapolation endpoint | `KDA/lm/train_lm.py`, `analyze_lm.py` |
| FineWeb-Edu tokenization to uint16 memmap | `KDA/lm/prepare_data.py` |
| Tokenized pretraining corpora (155.6B tok) | `s3://edullm-datasets/olmo-150b-dolma2/` |
| Slurm array-worker pattern w/ idempotent skip | `KDA/run_mqar_var.sbatch` |
| Architecture-ladder scripts | `OLMo-core/src/scripts/train/ladder/2026Q1/` |
| Rigor/audit checklist (14 scripts) | `probes/audit_exp*.py` |
| Frozen geometry + param ledger + gates | `docs/liv-kda-gqa-sub500m-experiment.md` |
| Doc format (Material Passport) | same |

### Must be built
| Item | Why |
|---|---|
| `LIVMixer` + config, registered | zero LIV/lfm2 matches in OLMo-core |
| Weight-parity test vs HF `modeling_lfm2.py` | design docs require it; don't infer gates from the PDF |
| Typed per-layer inference state API | replaces `assert isinstance(block.attention, Attention)` at `generation_module.py:108` |
| Low-rank gate variant + rank sweep | brainlift §3.1 |
| Cross-layer KV sharing (producer/consumer) | brainlift §3.2; RoPE pre/post-rotary decision undecided |
| Multi-branch dilated conv + router | brainlift §3.3; check whether `causal-conv1d` supports dilation at all |
| Declarative 16-slot arm builder + meta-device param assertion | protocol §6.1.2 |
| Long-context eval (RULER / passkey / phonebook) | not present anywhere |
| lm-eval-harness integration | not present |
| Component-level memory/state profiler | protocol §8.1.3 endpoint |
| Document-isolated packing | `KDA/lm/prepare_data.py` only does EOS concatenation |
| Corpus document-length distribution check | unknown whether 16K/32K docs exist |

## Open decisions this audit surfaces

1. **Scale conflict.** Brainlift arithmetic is d=2048 (1.2B geometry); local protocol freezes
   d=1024 (350M geometry). Pick one and restate the parameter table.
2. **Sequencing conflict.** All three brainlift proposals are deferrals gated behind
   "L0 survives vs A16-P". Does the new design accept that gate, or argue to run factorization
   first as a standalone study?
3. **Compute target.** FarmShare (where everything has run) vs SB-AWS (what the protocol
   declares, and where the corpora are).
4. **Power.** Prior measured variance says a +0.005-nat CE contrast needs ~n=43 seeds. The
   protocol's CE non-inferiority margin is +0.010 nats. Verify that margin is reachable at the
   affordable seed count before committing to it as a gate.
5. **KDA in or out?** The existing protocol's primary contrast is K2 vs L0-P (a KDA insertion
   question). The brainlift is not about KDA at all. These are different experiments sharing a
   backbone — say which one this is.
