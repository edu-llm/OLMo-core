# 04 — Inventory of DECISIVE experiments that need NO from-scratch training run

**Author:** reassessment team member 4. **Started:** 2026-08-01. **Status:** IN PROGRESS (written
incrementally; sections appended as evidence lands).

**Labels used throughout:** `MEASURED` = I ran it or read it out of a file/result JSON in this repo.
`INFERRED` = derived by arithmetic or by reading source code, not executed. `ASSUMED` = a planning
estimate with no measurement behind it; treat as the weakest class.

---

## 0. Why this document exists

The track's own history is the argument. Three cheap probes have already overturned expensive plans:

| probe | cost | what it killed | source |
|---|---|---|---|
| `p1_launch_bench.py` + `p1_verify.py` | 2 FarmShare jobs, single L40S, minutes of GPU | P1's entire decode-latency claim (best case **8.2% slower**) | `HANDOFF.md:192-217` MEASURED |
| `spectra_v2.py` | CPU/MPS on the Mac, one checkpoint load | P1's motivating premise ("gates are low-rank") — falsified under both metrics | `probes/README.md:77-93` MEASURED |
| `structure_energy.py` | CPU/MPS, one checkpoint load | `grouped` as a quality candidate (0.130 vs 0.929, identical to a random mask) | `probes/README.md:40-59` MEASURED |

None of those required a training run. All three produced a **reportable fact regardless of outcome**.
The 8-day 8×GPU program has not started. So the question this document answers is:

> How much of the capstone's publishable content can be produced with **zero** from-scratch pretraining?

Preliminary answer, stated up front so the ranking below can be read against it: **most of it.** The
framing decision already in `HANDOFF.md:158-161` — *"Liquid ships this architecture to real devices and
published no ratio ablation, no kernel-width ablation, and no recall benchmark"* — has three clauses,
and **two of the three are inference-only measurements on released checkpoints.** The recall benchmark
needs no training. The kernel-width fact (what the trained k=3 taps actually look like) needs no
training. Only the ratio ablation genuinely needs from-scratch runs.

*(Sections 1-8 and the ranked table follow as evidence lands.)*

---

## 1. Ground truth established for this document (MEASURED, 2026-08-01)

Before costing anything, I settled the facts the estimates depend on. Everything in this section was
read off disk or off the live cluster, not assumed.

### 1.1 LFM2-350M `config.json` — read from the local HF cache

Path: `/Users/ericwu/.cache/huggingface/hub/models--LiquidAI--LFM2-350M/snapshots/b3afba27815ee83a64b76162cef4d8a4780d6ca7/config.json`

| field | value | why it matters here |
|---|---:|---|
| `max_position_embeddings` | **128000** | MEASURED. The declared max context is **128K, not 32K.** A length sweep to 32,768 is *inside* the declared window — so a failure at 8K is a real architectural finding, not out-of-distribution abuse |
| `rope_theta` | 1,000,000.0 | Long-context RoPE base. Consistent with a genuine long-context claim |
| `conv_L_cache` | 3 | k=3. Confirms the width question |
| `conv_bias` | false | The 3 taps per channel are the *entire* conv parameterization: `3 × 1024 = 3,072` weights per layer |
| `full_attn_idxs` | [2,5,8,10,12,14] | 6 attention layers; the other 10 are LIV |
| `hidden_size` / `num_hidden_layers` | 1024 / 16 | |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | GQA-8, head_dim 64 |
| `block_ff_dim` | 6656 | Note this is the **pre-transform** value; the transform `256·ceil(⌊2/3·6656⌋/256)` = 4608 is the real width (HANDOFF trap #3) |
| `vocab_size` | 65536 | |
| `torch_dtype` | bfloat16 | 709 MB single-shard `model.safetensors` MEASURED |

**This 128K number is load-bearing and I did not expect it.** It converts the length sweep from "we
pushed it past its spec and it broke" (weak, unpublishable) into "**Liquid declares 128K and we measured
where recall actually dies**" (strong, publishable, and *negative-result-safe in both directions*). If
recall survives to 32K that is a notable positive result for a 10:6 conv-heavy hybrid; if it dies at 4K
that is a 128K-claim-versus-reality gap on a shipped production model.

⚠️ **One check still owed before publishing that framing:** `max_position_embeddings` in a config is not
always the trained/claimed context. Liquid's model card and paper should be read for the *declared*
usable context (INFERRED risk). Whatever it says, the config value is 128000 and that is quotable.

LFM2-1.2B config confirms the family shape is identical apart from width: d=2048, ff 12288 pre-transform,
32 heads / 8 KV, **same `conv_L_cache=3`, same `full_attn_idxs=[2,5,8,10,12,14]`, same 128000 / 1e6 RoPE**
(MEASURED via HF raw config). So **the 10:6 ratio and k=3 are held fixed across the whole family** — Liquid
never varied them publicly. That is precisely the gap the capstone claims.

### 1.2 Checkpoint tensor names — read from the safetensors header (no execution)

Read directly out of the header bytes of the cached `model.safetensors`. Per LIV layer there are exactly
three tensors, and per attention layer six:

```
model.layers.{0,1,3,4,6,7,9,11,13,15}.conv.conv.weight        # depthwise, (1024, 1, 3)
model.layers.{...}.conv.in_proj.weight                        # (3072, 1024) -> B, C, x
model.layers.{...}.conv.out_proj.weight                       # (1024, 1024)
model.layers.{2,5,8,10,12,14}.self_attn.{q,k,v,out}_proj.weight
model.layers.{2,5,8,10,12,14}.self_attn.{q,k}_layernorm.weight  # per-head QK-norm, size 64
```

**`conv.conv.weight` is 10 layers × 1024 channels × 3 taps = 30,720 scalars for the entire model.** This
is the object the kernel-width question is about, and it is a **30 KB read**. Section 4 builds an
experiment on it that costs zero GPU-hours.

### 1.3 FarmShare capability inventory (MEASURED, live, 2026-08-01)

Queried over the existing control socket; read-only.

| fact | value | source |
|---|---|---|
| GPU partition | `gpu`, nodes `oat-[01-06]`, **4× GPU each = 24 GPUs total** | `sinfo` / `scontrol show partition gpu` |
| Partition `MaxTime` | **2-00:00:00 (48 h)** — *not* 6 h | `scontrol show partition gpu` |
| QOS `gpu` limit | `gres/gpu=4`, MaxJobs 4, MaxSubmit 32, **no MaxWall** | `sacctmgr show qos` |
| QOS `normal` limit | `gres/gpu=1` | `sacctmgr show qos` |
| Default time | 02:00:00 if `-t` unset | `scontrol` |
| `DefMemPerCPU` / `MaxMemPerCPU` | 4000 MB / **4000 MB** | `scontrol` — so `--mem=48G` **requires `-c 12`**, not `-c 8` |
| Node RAM / CPUs | 256 GB / 64 per node | `sinfo` |
| Venv | `/scratch/users/ericrcwu/kda/venv`, Python 3.12.3 | `pip list` |
| torch / transformers | **2.11.0+cu128 / 5.14.1** | `pip list` |
| Also installed | `fla-core 0.4.1`, `triton 3.6.0`, `datasets 5.0.0`, `einops`, `ai2-olmo-core 2.5.0` (editable at `/scratch/users/ericrcwu/kda/olmo`) | `pip list` |
| **`lm-eval` / lm-eval-harness** | **NOT INSTALLED** (103 packages, no match) | `pip list` MEASURED |
| **`accelerate`** | **NOT INSTALLED** | `pip list` MEASURED |
| HF cache on FarmShare | only `Qwen3-4B` and `gpt2`. **No LFM2 checkpoint is cached there** | `ls ~/.cache/huggingface/hub` MEASURED |
| Login-node egress to `huggingface.co` | **works** (HTTP 307 redirect in 0.12 s) | `curl` MEASURED |
| `/scratch` free | 69 T of 106 T | `df -h` |
| Existing `liv/` contents | `probes/` (P1 bench + verify, results + sbatch) and `mqar/` only | `ls` MEASURED |

**Three corrections to the operating assumptions I was handed:**

1. **The "6-hour wall-clock limit" is not the binding constraint I was told it is.** The `gpu` partition
   `MaxTime` is **48 hours** and the `gpu` QOS carries **no `MaxWall`**. The 6 h figure may be a
   self-imposed convention or come from the `dev` QOS (`MaxWall=08:00:00`, 1 job, CPU-only 8c/32G). MEASURED.
   **This materially expands what is affordable** — a single 20-30 h job is legal. I still cost things in
   ≤6 h chunks below (checkpoint-and-resume is good practice given "the machine has died mid-run"), but
   the plan should not be *designed around* a 6 h ceiling that does not exist. **Verify with the human
   before relying on it**, since a local policy may exist that `scontrol` does not express.
2. **"1 GPU per job" is also not binding.** QOS `gpu` allows `gres/gpu=4` and `MaxJobs=4`. So up to **4
   concurrent GPU jobs, or 4 GPUs in one job.** That is a **4× throughput multiplier** on every sweep in
   this document. (Note QOS `normal` allows only 1 GPU — submit to `-p gpu` and get the `gpu` QOS.)
3. **`--mem=48G -c 8` will be rejected.** `MaxMemPerCPU=4000` means 8 cores caps at 32 G. Use
   `-c 12 --mem=48G`, or `-c 16 --mem=64G`.

**The one real blocker for the recall work: nothing is downloaded and nothing is installed.** No LFM2
checkpoint on FarmShare, no `lm-eval`, no `accelerate`. Section 3 costs that setup explicitly rather
than hiding it.

### 1.4 The single biggest finding in this document (MEASURED, 2026-08-01)

I fetched the LFM2-350M **model card** (`https://huggingface.co/LiquidAI/LFM2-350M/raw/main/README.md`).

| declared | value |
|---|---|
| **Context length** | **32,768 tokens** — declared for *all four* sizes (350M / 700M / 1.2B / 2.6B) |
| Training budget | **10 trillion tokens**, all sizes |
| Benchmarks reported | **MMLU, GPQA, IFEval, IFBench, GSM8K, MGSM, MMMLU** — seven, all short-context |
| LFM2-350M scores | MMLU 43.43 · GPQA 27.46 · IFEval 65.12 · IFBench 16.41 · GSM8K 30.1 · MGSM 29.52 · MMMLU 37.99 |
| **Long-context / recall / needle / RULER / RAG benchmark** | **ZERO. None reported.** |

But the card *markets* the model as "particularly suited for agentic tasks, **data extraction, RAG**,
creative writing, and multi-turn conversations."

**This is the whole capstone contribution in one sentence, and it needs no training run:**

> Liquid AI ships LFM2 to production devices, declares a **32,768-token context**, markets the model
> for **RAG and data extraction**, trained it on **10T tokens** — and publishes **not one** retrieval,
> recall, or long-context benchmark. Every number on the card is a short-context aggregate. We measured
> the retrieval behaviour of the shipped weights across the declared window.

Note the **discrepancy I found between the two sources**: `config.json` says `max_position_embeddings:
128000` while the model card says the context length is **32,768**. Both MEASURED. That gap is itself a
publishable line in the paper ("the config permits 128K; the card claims 32K; we measured where it
actually holds"), and it settles the sweep design: **sweep to 32,768 as the in-spec range, and add 65,536
as one out-of-spec-but-in-config point.**

This reframes the ranking below. The recall benchmark is not merely "cheap and nice to have" — it is the
**strongest, most defensible, and cheapest** deliverable available, and it is available *today* on
weights that already exist.

---

## 2. Off-the-shelf availability of the recall benchmarks (MEASURED, primary sources)

| harness | availability | fits LFM2? | verdict |
|---|---|---|---|
| **BABILong** (`RMT-team/babilong`) | HF dataset. Configs `0k,1k,2k,4k,8k,16k,32k,64k,128k,256k,512k,1M`; splits `qa1..qa20` at **100 rows each** in the 100-sample variant. Columns `input`/`question`/`target`, single-word targets | **YES, best fit.** One `load_dataset(...,"32k")["qa1"]` call | **USE THIS FIRST.** ⚠️ Full repo is **28.2 GB across 25,000 rows** — do **not** clone it; load only the length configs and qa1-qa5 splits you need |
| **RULER** (NVIDIA, Apache-2.0) | GitHub. 4 categories / **13 tasks**: `niah` (8 variants), `variable_tracking`, `common_words_extraction`, `freq_words_extraction`, `qa` (squad/hotpotqa). Sweeps 4K→128K | **YES — `MODEL_FRAMEWORK="hf"` works with plain transformers, no vLLM/TRT-LLM endpoint required** (MEASURED from README) | **USE SECOND.** Caveats: the documented setup path is marked **deprecated**, latest pipelines live on branches `rulerv1-ns` / `rulerv2-ns`; it wants a docker image; needs Paul Graham essays + SQuAD + HotpotQA downloads (sizes not published) |
| **Passkey / needle** (Mohtashami & Jaggi style) | ~120 lines of Python, no dependency, no download | **YES** | **WRITE IT.** Fastest path to a first number; also the only one that gives clean **position×length** resolution |
| **Phonebook** | Not a packaged benchmark; it is a generator (N name→number pairs, query one). ~60 lines | **YES** | Cheap add-on; same script as passkey with a different filler |
| **MQAR** | **Already implemented in this repo** at `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/mqar/mqar_data.py`, 43 tests | Only as a *from-scratch* probe task — **the released checkpoint cannot be evaluated on MQAR**, since MQAR uses a synthetic vocabulary the pretrained model has never seen | See §7 |
| **lm-eval-harness** | **NOT installed on FarmShare** (MEASURED). `pip install lm-eval` needed | Would reproduce/extend the card's 7 benchmarks | Lowest value — reproducing the vendor's own short-context table teaches nothing new |

**Decision that falls straight out of this table:** do **not** start with lm-eval-harness. The card
already publishes those 7 numbers. Start with **passkey (write it) → BABILong (load it) → RULER (port
it)**, because that is exactly the axis where the card is empty.

---

## 3. EXPERIMENT A — the recall/length sweep on released checkpoints

**Question answered:** *Does stock LFM2 actually retrieve, and where does it break, across the window
Liquid declares?* Nobody has published this. It is the capstone's headline claim, and it requires
inference only.

### A1 — Passkey sweep (write it; ~120 lines)

**Design.** Standard passkey: filler text ("The grass is green. The sky is blue…" repeated) with one
inserted `The pass key is <5-digit number>. Remember it.`, then `What is the pass key?`. Score = exact
match of the 5 digits.

- **Lengths:** 512, 1024, 2048, 4096, 8192, 16384, 32768 (in-spec) + 65536 (out-of-spec, config allows)
  = **8 lengths**
- **Needle depths:** 0%, 10%, 25%, 50%, 75%, 90%, 100% = **7 depths**
- **Trials:** 20 per (length, depth) cell with a fresh random key → binomial SE ≈ 11 pp at p=0.5,
  ≈ 5 pp at p=0.95. **Use 50 trials for the final table** (SE ≈ 7 pp / 3 pp).
- **Checkpoints:** LFM2-350M, LFM2-700M, LFM2-1.2B (+ optionally LFM2.5 variants and a **matched-size
  pure-transformer control** — Qwen3-0.6B is *already cached on FarmShare*, MEASURED)
- **Grid:** 8 × 7 × 50 = **2,800 forward passes per checkpoint.**

**Cost (INFERRED, arithmetic shown).** Prefill FLOPs ≈ `2·N·T` + attention-score term. At 350M and
T=32,768 with 6 attention layers of d=1024: score FLOPs = `6 layers · 2 · 2 · T² · d` = `24·T²·d`
= 2.64e16 for the whole 32K prefill… no — per sequence: `6 · 4 · T² · d_model` ≈ 6·4·(3.28e4)²·1024
= 2.64e13 FLOP, plus dense `2·N·T` = 2·3.54e8·3.28e4 = 2.32e13 FLOP. **≈ 5.0e13 FLOP per 32K sequence.**
L40S bf16 dense ≈ 181 TFLOP/s peak; at a realistic 25% MFU for short single-sequence prefill ≈ 45
TFLOP/s → **≈ 1.1 s per 32K forward.** The 8-length grid is dominated by the two longest rungs; summing
`T` and `T²` terms over the 8 lengths gives an effective ≈ 1.9× the 32K cost per (depth, trial) sweep
row → **≈ 2.1 s × 350 cells ≈ 12 min per checkpoint** for 350M at batch 1, ignoring batching wins.

Rounding up hard for tokenization, model load, KV-cache allocation at 64K, and no batching:

| item | estimate | label |
|---|---:|---|
| LFM2-350M full grid (8×7×50) | **0.5 L40S-h** | INFERRED |
| LFM2-700M | 0.8 L40S-h | INFERRED |
| LFM2-1.2B | 1.4 L40S-h | INFERRED |
| Qwen3-0.6B control (already cached) | 0.7 L40S-h | INFERRED |
| **Total, 4 checkpoints** | **≈ 3.4 L40S-h** | INFERRED |
| Wall-clock with 4 concurrent jobs (QOS allows it) | **≈ 1.5 h** incl. queue | INFERRED |
| Engineering | **4-6 h** (script + sbatch + plotting) | ASSUMED |
| Download (350M 709 MB + 700M ~1.4 GB + 1.2B ~2.4 GB) | ~4.5 GB, minutes; **compute-node egress to HF CONFIRMED WORKING** (MEASURED: HTTP 307 in 0.14 s from `wheat-01` inside `srun`) | MEASURED |

**Memory check (the one thing that could break it):** at T=65,536, KV = 12 KiB/token (HANDOFF, MEASURED)
→ **768 MiB**, plus 0.7 GB weights, plus a 65,536 × 65,536 vocab logits tensor if you are careless
(**that would be 8.6 GB in bf16** — only materialize the last position's logits). Fits in 46 GB with
room to spare. Even 128K would fit at 1.5 GiB KV. **No memory blocker.**

**Outcomes and what each implies:**

| outcome | implication | still publishable? |
|---|---|---|
| Recall holds ≥90% out to 32K at all depths | LFM2's 6 global attention layers are sufficient; the conv-heavy ratio costs nothing for retrieval. **This weakens the motivation for P2/topology arms** and is a genuinely positive result for Liquid | **YES** — first published recall benchmark for LFM2, confirming the vendor claim |
| Recall degrades gradually past ~8K | The declared 32K is "supported" but not "effective". Establishes an effective-context number Liquid never published | **YES — strongest outcome.** Directly motivates the whole architecture study |
| Recall collapses at some depth band (e.g. mid-document) | Classic lost-in-the-middle, now measured on a hybrid | **YES** |
| Recall is near-zero even at 2K | Either the model is genuinely bad at retrieval, **or your harness is broken.** ⚠️ **Mandatory positive control** — this repo has already been burned twice by exactly this (MQAR jobs 1670922 and 1670963, `probes/mqar/README.md:83-95`). Run the 512-token / depth-50% cell FIRST and require it to pass before believing any sweep | **only after the control passes** |

**⚠️ Two design traps I want on record.**
1. **Base vs instruct.** `LiquidAI/LFM2-350M` is the instruct-tuned release. Passkey with a chat template
   vs raw completion can differ by tens of points. **Run both prompt formats on the 2K rung** and report
   which was used. Cheap (adds ~2% to the grid), and omitting it invites a reviewer to dismiss the table.
2. **The 5-digit key must not be tokenizer-fragile.** With a 65,536 vocab, a 5-digit number may tokenize
   inconsistently. Score on the *decoded string*, not on token IDs.

**NEGATIVE-RESULT-SAFE: YES, maximally.** Every possible outcome is a first-ever published number for a
shipped model. This is the single best property in the whole inventory.

### A2 — BABILong (load it)

**Marginal cost over A1 is small** because the harness (batching, generation, scoring) is already written.
Run `qa1` (single supporting fact) and `qa2` (two supporting facts) at configs `0k,1k,2k,4k,8k,16k,32k`
— 100 rows each × 7 lengths × 2 tasks = **1,400 generations per checkpoint.** Generation is short (one
word), so cost is prefill-dominated and comparable to A1: **≈ 0.4 L40S-h at 350M**, ~1.5 L40S-h for three
checkpoints. **Engineering ≈ 3 h** (exact-match scorer + prompt format + a normalization decision).

**Why it adds value beyond passkey:** passkey is verbatim copy; BABILong requires a one- or two-hop
inference over facts hidden in distractors. A model can pass passkey and fail qa2. That contrast — **copy
survives, reasoning-over-retrieved-facts does not** — would be the most interesting single finding
available at this cost. ⚠️ **Do not download the full 28.2 GB repo**; select configs.

**NEGATIVE-RESULT-SAFE: YES.**

### A3 — RULER (port it)

Highest credibility (it is *the* standard), highest engineering cost. 13 tasks × 6 lengths. The `hf`
backend means no serving stack, but the documented path is deprecated and the live pipelines are on
branches. **Engineering 8-12 h** (ASSUMED — dominated by fighting the repo's config scripts, not by
model code). **Compute ≈ 3-5 L40S-h per checkpoint** (13 tasks vs 1). 

**Recommendation: do A1 and A2 first, get the story, then decide whether RULER is worth 12 engineering
hours to make the table citable.** If A1/A2 show a clean effect, RULER converts "we measured it" into
"we measured it on the benchmark reviewers accept" — worth it. If A1/A2 show nothing, skip it.

**NEGATIVE-RESULT-SAFE: YES** (but expensive to reach).

---

## 4. EXPERIMENT C — the kernel-width question, answered WITHOUT training

### 4.1 I ran this. Here are the numbers. (MEASURED, 2026-08-01, FarmShare login node, CPU, ~8 seconds)

This is the cheapest decisive experiment in the entire inventory, so rather than only specify it, I
executed it — on FarmShare, CPU-only, no GPU, no job submission.

- Downloaded `LiquidAI/LFM2-350M` to `/scratch/users/ericrcwu/liv/ckpt/` (709 MB, ~40 s, compute/login
  egress confirmed).
- Script: `/scratch/users/ericrcwu/liv/tapread.py`. Reads the safetensors header, seeks to each
  `model.layers.{i}.conv.conv.weight`, decodes bf16→fp32 by bit-shift (no torch needed), reshapes to
  (1024 channels, 3 taps).
- **Tap index convention (verified against the code, load-bearing):** `nn.Conv1d(padding=k-1)` followed
  by `[..., :seq_len]` gives `out[t] = w[0]·x[t-2] + w[1]·x[t-1] + w[2]·x[t]`. So **index 0 is the
  OLDEST lag and index 2 is the CURRENT token.** Confirmed against
  `short_conv.py:249-252` (`_conv_dense`) and the released `Lfm2ShortConv`.

**Per-layer energy distribution across the 3 taps, all 10 LIV layers of released LFM2-350M:**

| layer | mean\|w\| t-2 | mean\|w\| t-1 | mean\|w\| t | **E% t-2** | **E% t-1** | **E% t** | frac ch. \|t-2\|>\|t\| |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0218 | 0.1502 | 0.0109 | 5.25 | **92.98** | 1.77 | 0.730 |
| 1 | 0.0318 | 0.1321 | 0.0228 | 6.49 | **88.74** | 4.77 | 0.675 |
| 3 | 0.0412 | 0.0809 | 0.0900 | **11.37** | 35.46 | 53.16 | 0.304 |
| 4 | 0.0349 | 0.0777 | 0.1044 | 7.17 | 30.51 | 62.32 | 0.222 |
| 6 | 0.0232 | 0.0474 | 0.1316 | 3.23 | 12.41 | 84.36 | 0.097 |
| 7 | 0.0260 | 0.0554 | 0.1292 | 4.13 | 15.79 | 80.07 | 0.095 |
| 9 | 0.0310 | 0.0610 | 0.1258 | 5.51 | 17.87 | 76.62 | 0.123 |
| 11 | 0.0157 | 0.0449 | 0.1360 | 1.82 | 11.70 | 86.48 | 0.055 |
| 13 | 0.0081 | 0.0286 | 0.1500 | 0.62 | 6.35 | 93.03 | 0.052 |
| 15 | 0.0038 | 0.0133 | **0.1797** | **0.11** | 2.27 | **97.62** | 0.004 |

**Pooled over all 10,240 channels:**

```
energy by tap (t-2, t-1, t)          =  4.26% / 29.62% / 66.12%
mean|w| by tap                        =  0.0237 / 0.0691 / 0.1081
frac channels where t-2 is largest    =  2.08%
frac channels where |t-2| > |t|       =  23.55%
frac channels |t-2| > 0.5·max|tap|    =  10.33%   <- "boundary not decayed"
frac channels |t-2| > 0.9·max|tap|    =   2.46%   <- "boundary saturated"
|t-2| / |t| ratio: median 0.220, p25 0.046, p75 0.870
```

### 4.2 What this says about k=3→k=5/9/15

**The trained kernels are NOT saturated at the boundary.** Only **4.26%** of total conv energy sits on
the oldest available tap, and only **2.46%** of channels have their largest weight there. If the
receptive field were binding, the model would push mass onto the boundary tap — the classic signature
of a truncated filter that wants to be longer. It does the opposite: **energy decays sharply toward
older lags** (66% / 30% / 4%), with a median `|t-2|/|t|` ratio of **0.22**.

**This is a direct, cheap, previously-unpublished empirical fact about LFM2, and it is a PREDICTION that
widening the kernel will do little.** It independently corroborates the published width sweeps that were
already flat past k=3 (Sieberling et al., HANDOFF `Open items`) — but with the crucial difference that
**those used ungated convs, and this measurement is on Liquid's actual double-gated block.** The
counter-argument in `HANDOFF.md:455-462` ("their negative results may not transfer because LFM2's conv
sits inside two gates") is now weakened by evidence from inside the gates themselves.

### 4.3 A second finding I did not expect: the LIV layers are NOT all doing the same thing

**Layers 0 and 1 are qualitatively different operators from layers 6-15.** Layers 0/1 put **93% and 89%**
of their energy on the **middle** tap (t-1), with the *current* token carrying only 1.77% and 4.77% —
these are essentially **shift/delay operators**, mixing the previous token into the current position. By
layer 15 the filter has collapsed to a near-passthrough of the current token (97.62%), i.e. the conv is
barely convolving at all — it is acting as a **gated pointwise transform**.

There is a clean monotone story with depth: t-2 energy peaks at **layer 3 (11.37%)** and decays to
**0.11%** at layer 15; t-1 energy falls monotonically from 93% to 2.3%.

**Three consequences that reshape the arms:**
1. **A uniform k for all 10 LIV layers is the wrong design.** If width helps anywhere, it helps in the
   *early* layers (0-4, where 5-11% of energy already reaches the boundary), not the late ones. **A
   depth-varying width arm (`k9` early / `k3` late) is a better-motivated and more novel arm than the
   uniform `W-k5/k9/k15` sweep currently declared in `liv_arms.py`** — and this measurement is the
   justification for it.
2. **Deep LIV layers may be nearly free to simplify.** Layer 15's conv contributes 97.6% passthrough; a
   `k=1` (pure gated pointwise) arm at the deep layers is now a *motivated* cheap arm, not a guess. This
   is a new arm the design doc does not contain.
3. **It supplies a mechanism for why the published width sweeps were flat**: averaged over depth, a
   width increase is mostly spent on layers that have already learned not to look back.

### 4.4 The honest limits of this argument (read before citing it)

⚠️ **This is a weight-space measurement, and this team has already been burned once by exactly that.**
`spectra_v2.py` established that *plain* spectra of the LFM2 gates gave a misleading answer and only the
**activation-aware** version settled the question (`probes/README.md:77-93`). The same objection applies
here with full force:

- **Tap magnitude ≠ contribution.** The conv input is `Bx = pre_gate ⊙ value`, whose per-channel variance
  differs by orders of magnitude. A small weight on a high-variance channel can dominate. **The correct
  metric is activation-weighted:** `E_j = Σ_c w[c,j]² · Var(x_c)` over calibration tokens. INFERRED that
  the qualitative conclusion survives (the depth trend is far too large — 11% → 0.11% — to be an
  artifact of channel scaling), but **the pooled 4.26% number must be re-derived activation-aware before
  publication.**
- **"Trained kernels don't use lag 2" does not logically entail "a k=9 model trained from scratch would
  not use lag 8."** The k=3 model was never *given* lag 8. This is a **prior, not a proof** — exactly the
  status the team correctly assigned to `structure_energy.py`'s grouped-vs-lowrank result. State it that way.
- **The autoregressive residual stream already carries history.** Each LIV layer's input has passed
  through previous layers' convs and the global attention layers, so "lag 2 at layer 15" is not "2 tokens
  ago in the input" — effective reach compounds with depth. The per-layer numbers are still meaningful;
  the *aggregate* interpretation needs this caveat.

### 4.5 Cost and follow-ups

| variant | GPU-h | wall-clock | eng-h | status |
|---|---:|---:|---:|---|
| **C1 — weight-space tap energy (this section)** | **0** (CPU) | **~8 s + 40 s download** | ~1 h | ✅ **DONE 2026-08-01** |
| C2 — same on LFM2-700M / 1.2B / 2.6B / LFM2.5 (does the depth pattern replicate across scale?) | 0 | ~5 min | 0.5 h | **Do next — near-free, and cross-scale replication makes it a real result rather than an anecdote** |
| C3 — activation-weighted version (32k calibration tokens, mirroring `spectra_v2.py`'s protocol) | ~0.2 | ~20 min | 3 h | **Required before publishing C1** |
| C4 — FIR frequency response per channel: classify each filter as low-pass (smoothing → wants width) vs high-pass/differencing (→ does not) | 0 | ~1 min | 2 h | High value, zero cost. See §4.6 |
| C5 — causal-intervention control: zero the t-2 tap at inference and measure Δ perplexity / Δ recall | ~0.3 | ~30 min | 2 h | **This is the one that converts the prior into evidence.** If zeroing 4.26% of energy costs ~0 perplexity, the width claim is settled empirically, not just spectrally |

**C5 is the important one** and it belongs to §5 (surgery) as much as here: it is the *causal* version of
this correlational measurement, and it is still inference-only.

**NEGATIVE-RESULT-SAFE: YES, and unusually so.** There is no outcome of "characterize the released conv
weights" that is not a reportable first. It already produced two facts (boundary not saturated; layers
are heterogeneous by depth) before any GPU was touched.

### 4.6 C2 + C4 EXECUTED — the pattern replicates across all four released scales (MEASURED, 2026-08-01)

I did not stop at 350M. All four released LFM2 checkpoints are now on FarmShare at
`/scratch/users/ericrcwu/liv/ckpt/` (350M 709 MB, 700M 1.4 GB, 1.2B 1.9 GB, **2.6B 3.8 GB**; downloaded
in parallel, **~4 min total**). Script: `/scratch/users/ericrcwu/liv/tapfreq.py`. **Zero GPU. Login node.
Runtime a few seconds.**

**Pooled tap-energy across the family:**

| checkpoint | d | LIV layers | E%[t-2] | E%[t-1] | E%[t] | oldest-is-argmax | \|oldest\|>0.9·max |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFM2-350M | 1024 | 10 | **4.26** | 29.62 | 66.12 | 2.08% | 2.46% |
| LFM2-700M | 1536 | 10 | **5.24** | 26.89 | 67.87 | 3.16% | 3.67% |
| LFM2-1.2B | 2048 | 10 | **5.34** | 24.05 | 70.61 | 3.53% | 4.29% |
| **LFM2-2.6B** | 2048 | **22** | **4.78** | 20.40 | 74.82 | 9.75% | 10.30% |

**Two facts that are now robust rather than anecdotal:**

1. **The oldest tap never carries more than ~5.3% of conv energy at any released scale.** The
   boundary-saturation signature that would motivate a wider kernel is **absent in every checkpoint
   Liquid has shipped.** This is the cross-scale replication that turns §4.2 from an observation into a
   finding. **The published fact is: "across LFM2-350M/700M/1.2B/2.6B, 4.3-5.3% of depthwise-conv energy
   sits on the oldest of 3 taps; k=3 is not a binding constraint in the weights."**
2. **The depth pattern replicates exactly, and 2.6B proves it is depth-relative, not layer-index
   absolute.** LFM2-2.6B has **22 LIV layers** (not 10 — a different topology, `full_attn_idxs` differs),
   and it shows the *same* trajectory stretched over more layers: layers 0-4 are delay-dominated (86%,
   89%, 64%, 58% on t-1), the middle is mixed, and layers 22-29 collapse to passthrough (89-93% on t).
   **A structural property of how LFM2 trains, not a quirk of one checkpoint.**

**C4 — frequency response, executed in the same pass.** Per channel I computed the DC gain
`H(0)=Σw` and Nyquist gain `H(π)=Σw·(−1)^lag`, and classified each of 10,240-45,056 channels:

| checkpoint | low-pass (\|H0\|>2\|Hπ\|) | high-pass (\|Hπ\|>2\|H0\|) | layer-0 "pure delay" (>90% energy in history) |
|---|---:|---:|---:|
| LFM2-350M | 23.8% | 9.7% | **95.5%** |
| LFM2-700M | 24.7% | 9.5% | **95.8%** |
| LFM2-1.2B | 24.9% | 9.3% | **94.9%** |
| LFM2-2.6B | 21.2% | 14.2% | 55.8% |

**Low-pass beats high-pass ~2.5:1 across the family** — the conv is predominantly a *smoother*, which is
the filter class that *would* benefit from more taps, and it still does not use the taps it has. That
tension is worth stating explicitly in the paper: **the filters are the right type to want width, and
they still decline it.**

**And the layer-0 result is striking: 95% of layer-0 channels put >90% of their energy on history, not
the current token.** LFM2's first LIV layer is, to a first approximation, **a learned per-channel token-
shift** — closely related to the `token_shift` primitive in RWKV and to the "shift-SSM" in H3. **This is
a mechanistic-interpretability-flavored finding about a production model that nobody has published, and
it cost zero GPU-hours.** It also generates a concrete, free arm: **initialize layer 0's conv as an
explicit shift** rather than the near-identity the current `init_weights` uses
(`short_conv.py:347-348` sets `weight[:, :, -1] = 1.0`, i.e. *current-token* identity — the **opposite**
of what the trained model converges to in its first two layers). That is a one-line change with a
measured empirical justification behind it.

**Per-layer detail is in the raw output; the script is committed at
`/scratch/users/ericrcwu/liv/tapfreq.py` and reproduces in seconds.**

⚠️ Same caveat as §4.4 applies to all of it: **weight-space, not activation-weighted.** C3 remains owed.

---

## 5. EXPERIMENT B — layer ablation / surgery on the released checkpoint

**Question answered:** *What does each of the 6 attention layers and 10 conv layers actually contribute?*
This is the closest inference-only proxy for the conv:attention-ratio question, which otherwise costs an
8-day training program.

### 5.1 The interventions, in increasing order of aggressiveness

All are done with forward hooks on the released model. No weights are updated. All are inference-only.

| # | intervention | implementation | what it isolates |
|---|---|---|---|
| B1 | **Zero-out a mixer** — replace `block.mixer(x)` output with 0, keeping the residual | 3-line forward hook | Total contribution of that layer's mixer |
| B2 | **Identity/skip a whole block** — `block(x) → x` | hook | Contribution of mixer + MLP together |
| B3 | **Attention → local window** — mask attention to the last `w ∈ {8, 32, 128, 512}` tokens | attention-mask edit | **Is each attention layer doing global retrieval or local smoothing?** The single most informative one |
| B4 | **Attention → conv-like** — restrict to `w = 3` | as B3 | "Could this attention layer have been a LIV layer?" |
| B5 | **Conv → pointwise (k=1)** — zero taps t-1, t-2 | weight edit | The causal counterpart of §4's correlational finding |
| B6 | **Conv tap zeroing** — zero only t-2 | weight edit | **C5.** Directly tests the 4.26% |
| B7 | **Widen a conv** — pad the k=3 kernel to k=5/9 with zeros | weight edit | Sanity: must be an exact no-op. **A required correctness control for the whole width arm** |
| B8 | **Cross-layer KV sharing at inference** — force attention layer *j* to read layer *i*'s K/V | hook | **A zero-training pilot for P2** |

**Metrics for each:** (a) Δ perplexity on ~1M held-out tokens, (b) Δ passkey/BABILong accuracy from
Experiment A, (c) Δ on 2-3 of the card's own benchmarks (MMLU/GSM8K) so the numbers are comparable to
Liquid's published table.

### 5.2 Cost

Each intervention is one forward pass over the eval set. With 17 single-layer ablations × (1M-token ppl
+ a 200-case passkey subset) ≈ **0.15 L40S-h per intervention at 350M**, and there are roughly
`16 zero-outs + 16 identity + 6×4 windows + 10 k=1 + 10 tap-zero + ~15 KV-share pairs ≈ 91` cells.

| item | estimate | label |
|---|---:|---|
| Full single-layer sweep, 350M | **≈ 14 L40S-h** | INFERRED |
| Trimmed to the decisive cells (B1 all 16, B3 on all 6 attn × 4 windows, B5/B6 on all 10 conv) = 50 cells | **≈ 7.5 L40S-h** | INFERRED |
| Wall-clock, 4 concurrent jobs | **≈ 2.5 h** | INFERRED |
| Engineering | **6-8 h** (hook plumbing + the eval loop, which Experiment A already provides) | ASSUMED |

**Do B6 and B3 first — 16 cells, ~2.5 L40S-h, ~1 h wall-clock.** They carry most of the information.

### 5.3 What each outcome implies

| result | implication |
|---|---|
| **B3:** attention layers survive a 128-token window with ~0 Δ | Those layers are **not** doing global retrieval. **This would be the most consequential cheap finding available for the topology arm** — it says LFM2 could ship with fewer/cheaper global attention layers, which is exactly `A-fewer3`'s hypothesis, evidenced *without training* |
| **B3:** ppl explodes at window 512 for a specific subset (e.g. layers 8, 12) | Retrieval is localized to identifiable layers → **tells you which layers `A-fewer3` must keep**, converting a guess into a design |
| **B6:** zeroing the t-2 tap costs ~0 ppl | The k=3→k=9 arm is very likely to be flat. **Strong pre-registration evidence.** Combined with §4.6 this could justify *cancelling* the width arm and reallocating that budget |
| **B6:** zeroing t-2 costs real ppl despite carrying 4.26% of energy | The energy metric understates importance → **§4's prior is wrong, and the width arm is well-motivated.** Either way you learn the answer for ~1 GPU-hour |
| **B1:** some LIV layers are near-free to delete | A depth/ratio result at zero training cost |
| **B8:** forced KV sharing between adjacent attention layers costs little | A P2 pilot. See the hard limit below |

### 5.4 The limits of the inference-only argument — state these or a reviewer will

⚠️ **This is the section that keeps the surgery honest.**

1. **Ablation measures the marginal contribution of a layer *to a network trained with that layer*, not
   what a network trained without it would do.** A from-scratch model reallocates. This is the *same*
   caveat the team already recorded for `structure_energy.py` ("a from-scratch grouped layer ≠ the
   block-diagonal part of a trained dense layer", `probes/README.md:55-59`) and the same one GaLore
   illustrates. **Ablation systematically OVERSTATES a component's necessity** (nothing compensates) and
   is therefore a **conservative** test for "this layer is removable": if ablation says removable, it is
   strong evidence; if ablation says essential, it is weak evidence.
2. **B8 (KV sharing at inference) is the weakest of the set and should be labelled as a pilot only.**
   Forcing layer *j* to read layer *i*'s KV in a model never trained to do so will look catastrophic
   almost regardless of whether trained CLA works. **A large B8 degradation is nearly uninformative;
   only a *small* one is informative.** CLA/Hymba work precisely because training adapts. Run it, but
   pre-register that only the negative direction (small damage) counts.
3. **Instruct-tuned confound.** The released checkpoints are post-trained. Ablation deltas partly reflect
   post-training, not pretraining architecture.
4. **Single-checkpoint conclusions are anecdote.** Replicate the decisive cells on 700M and 1.2B — the
   checkpoints are **already downloaded** (§4.6), so this is a 3× compute multiplier and zero extra
   engineering. The cross-scale replication in §4.6 shows how much credibility that buys.

**NEGATIVE-RESULT-SAFE: YES for B1/B3/B5/B6** (any Δ is a reportable per-layer attribution profile for a
production model). **NO for B8** — a bad result there is uninterpretable, so it cannot be sold as a
finding.

---

## 6. EXPERIMENT D — the missing decode path for `ShortConv`

### 6.1 What is actually missing (MEASURED by reading both implementations)

`HANDOFF.md:364-366` lists this as the top remaining Phase 0 item and says it "blocks all latency and
cache measurement." I read both sides to size it.

**Reference implementation** (`transformers` `main`, `models/lfm2/modeling_lfm2.py`, fetched
2026-08-01). `Lfm2ShortConv.forward` has **three** paths:
- `use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(layer_idx)`
- **decode:** pulls `conv_state = past_key_values.layers[layer_idx].conv_states[0]` and calls
  `causal_conv1d_update(...)`; the fused per-step kernel **mutates the state in place**
- **prefill:** `past_key_values.update_conv_state(..., conv_kernel_size=...)` then `causal_conv1d_fn(...)`,
  then trims leading cached positions
- Both `causal_conv1d_update` and `causal_conv1d_fn` are wrapped in
  `@maybe_replace_from_package("causal_conv1d", ...)` with **pure-PyTorch fallbacks** using `F.conv1d`
  with `groups=hidden_size`; the update fallback concatenates prior state with the new input and does
  `conv_state.copy_(...)` on the trailing window.

**Our side** (`/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer/src/olmo_core/nn/attention/short_conv.py`, 476 lines):
- `forward()` (lines 254-271) is **prefill-only**. No `past_key_values` parameter, no state buffer.
- `_conv_dense()` (249-252) does `nn.Conv1d(padding=k-1)` then `[..., :seq_len]`.
- The surrounding plumbing **already exists and already accommodates non-attention mixers**:
  `KVCacheManager` (`nn/attention/kv_cache.py`, 118 lines) is attention-specific, but
  `generation_module.py:108-129` `prepare_inference_cache`/`free_inference_cache` **already `continue`
  past non-`Attention` mixers** (the 2-line relaxation from commit `83e4dce`). So the hook points exist;
  they just skip us today.

### 6.2 Engineering estimate

| task | hours | label |
|---|---:|---|
| `ConvStateManager` (a `[B, d, k]` buffer + `reset`/`reallocate`, mirroring `KVCacheManager`) | 2 | INFERRED |
| `ShortConv.forward(..., conv_state=None)`: prefill writes the trailing `k` frames; decode does roll-and-append + a `k`-tap dot | 3 | INFERRED |
| Wire into `prepare_inference_cache` / `free_inference_cache` (change `continue` → dispatch on `ShortConv`) | 1.5 | INFERRED |
| Tests: **prefill-then-decode == full prefill**, bit-exact for k∈{3,5,9,15}; state shape `[B,d,k]`; batch>1; `k=15` grows state exactly 5× | 3 | INFERRED |
| Parity vs the HF decode path | 2 | INFERRED |
| **Total** | **≈ 11.5 h, call it 1.5-2 focused days** | INFERRED |

⚠️ **One trap already flagged in the HANDOFF and confirmed by my read.** `HANDOFF.md:358-361`: the
transformers decode path "reportedly drops one tap of history per step via `conv_state.roll(-1)`
(flagged, unconfirmed on `main`)." My fetch of `main` shows the fallback does
`conv_state.copy_(...)` on a **trailing window of a concatenation**, which is the *correct* formulation
— so the roll-based bug may already be fixed or may only exist in the fused-kernel branch.
**Consequence: do NOT use HF decode as the parity oracle.** Use **our own prefill as the oracle**
(`prefill(x)[-1] == decode(x[-1] | state(x[:-1]))`), which is a self-contained invariant and is what the
committed parity test already does for prefill. This is a real 2-4 h saving and a real correctness risk
avoided.

Also confirmed still true: `olmo_core/nn/convolution.py::CausalConv1d` **cannot** be reused —
`return output[0]` **drops the conv state**, which is precisely the thing needed here
(`HANDOFF.md:348-351`).

### 6.3 What ~12 hours unlocks — and, more importantly, what it does NOT

**Unlocks:**
- Any decode-latency or decode-bandwidth measurement of the arms.
- Conv-state memory measurement: `[B, d, k]`, so k=15 is **5× k=3** — but that is `B·1024·15·2 B` =
  **30 KiB/sequence** at 350M vs **12 KiB/token** for KV. **At any context beyond ~3 tokens the conv
  state is negligible.** INFERRED from HANDOFF-measured KV figure.
- Generation-based evals (BABILong, RULER) **on our own trained arms**.

**Does NOT unlock, and this is the load-bearing point:**
- **It is not needed for Experiment A, B, or C.** Those all run on the **released HF checkpoint through
  the transformers stack**, which already has a working decode path.
- The latency question it serves **was already answered and answered negatively** — P1's decode-latency
  claim is dead (`HANDOFF.md:192-217`), P3 is reframed as quality-only (`HANDOFF.md:104`), and the
  surviving efficiency claim belongs to the topology, not the proposals (`HANDOFF.md:483-486`).

**⇒ Recommendation: the decode path is NECESSARY-BUT-NOT-URGENT. It is a prerequisite for the training
program, not for any cheap experiment.** Given the training program has not started, **it should be
sequenced AFTER Experiments A/B/C**, not before. Doing it first spends 1.5 days of the highest-value
engineering time on a capability that only pays off in a phase that may be rescoped by A/B/C's results.

**NEGATIVE-RESULT-SAFE: N/A** — it is infrastructure, not an experiment. It produces no reportable fact
on its own. That is itself an argument for deferring it.

---

## 7. EXPERIMENT E — small-scale from-scratch that is NOT the 8-day plan

### 7.1 The throughput anchor is MEASURED, not guessed — from this repo's own sibling track

`KDA/HANDOFF.md` contains a **completed** 15-run L40S LM study whose numbers calibrate everything here:

| measured fact | value | source |
|---|---|---|
| Model | 12 layers, d=512, 8 heads × 64, SwiGLU, tied emb | `KDA/HANDOFF.md:549-551` |
| Params | **52.1M non-embed** (77.8M total) | `:550` |
| Tokens | **1.04B** FineWeb-Edu (Chinchilla 20×) | `:551,587-589` |
| Per-run wall-clock, one L40S | **`hh1` ~4.9 h**, heaviest arm `hh4` **1.34 s/step ≈ 11.9 h** | `:594-595` |
| Whole 15-run grid | **~114 GPU-h, ~29 h wall-clock** | `:541` |
| Job walltime actually used | **20 h limit, longest run ~12 h** | `:596` |
| Probe program total | **<2 GPU-h** | `:160` |

**Note `:596` independently refutes the "6-hour limit":** the sibling track ran **20-hour** GPU jobs on
this cluster. Combined with `scontrol`'s 48 h `MaxTime` (§1.3), the 6 h figure should be treated as a
convention, not a constraint. MEASURED, two independent sources.

**Implied throughput.** `6ND` = 6·5.21e7·1.04e9 = 3.25e17 FLOP in 4.9 h ⇒ **18.4 TFLOP/s ≈ 10% MFU** —
but that arm is a custom Triton gated-delta-net. A LIV/GQA hybrid is a stack of dense GEMMs plus a
depthwise conv and should reach **25-35% MFU (45-63 TFLOP/s)**. I use **45 TFLOP/s (25% MFU)** below as
the conservative planning number. INFERRED.

### 7.2 What fits, at Chinchilla-optimal `D = 20N`

`FLOPs = 6N·20N = 120N²`, so `t = 120N² / 4.5e13`:

| N (non-embed) | tokens | **one run, one L40S** | fits ≤6 h? | 8 seeds × 6 arms = 48 runs | wall-clock @4 concurrent |
|---:|---:|---:|:--:|---:|---:|
| **20M** | 0.4B | **18 min** | ✅ | **14 L40S-h** | **~4 h** |
| **35M** | 0.7B | 55 min | ✅ | 44 L40S-h | ~11 h |
| **50M** | 1.0B | **1.9 h** | ✅ | 89 L40S-h | ~22 h |
| **100M** | 2.0B | 7.4 h | ✗ (needs 1 job >6 h, legal at 48 h) | 355 L40S-h | ~4 days |
| **150M** | 3.0B | 16.7 h | ✗ | 800 L40S-h | ~8 days |

**The sweet spot is 20-50M.** At **20M, a full 6-arm × 8-seed study costs 14 L40S-h and lands in an
afternoon.** That is *seven times cheaper than the probe budget the KDA track already spent*, and it
buys the thing `HANDOFF.md:180-182` identifies as the single biggest methodological risk: **seeds.**
The KDA study's own +8.92 pp at n=3 collapsing to +2.01 pp (ns) at n=8 is the cautionary tale.

⚠️ **Do not skip the pre-flight checks that the KDA track paid for.** All four of its bugs
(`KDA/HANDOFF.md:580-597`) apply verbatim: no-weight-init (loss 400 vs `ln(V)`), sub-Chinchilla/repeated
data, OOM at the configured micro-batch, and walltime underestimation. **Assert `loss(step 0) ≈ ln(V)`
before every run.** That assertion is free and has already saved 114 GPU-h once on this cluster.

### 7.3 Which questions survive at 20-50M — being honest about scale

| arm / question | survives at 20-50M? | reasoning |
|---|:--:|---|
| **P3 widths `k3/k5/k9/k15` inside the real gate** | ✅ **YES — best fit in the whole set** | It is a *local receptive-field* question about a depthwise operator with `k·d` parameters. Nothing about "does tap 4 help" requires 350M. And §4.6 gives a **pre-registered prediction** (flat), which makes a small-scale null informative rather than merely quiet |
| **Conv:attention ratio / topology (`L0` vs `A16-P` vs `A-fewer3`)** | ⚠️ **PARTIALLY** | Ratio *optima shift with scale* — attention's relative value grows with model capacity and context. A small-scale result gives the **direction and the FLOP-vs-param crossover shape**, not the optimum. Report as a trend, never as "the right ratio is X" |
| **Recall / MQAR / needle on the arms** | ✅ **YES** | Zoology itself calibrates MQAR at 2-layer, d=64-256. Synthetic recall is where small scale is *standard practice*, not a compromise |
| **Depth-varying width (the new arm from §4.6)** | ✅ **YES** | Same argument as widths; and the depth pattern in §4.6 replicated across d=1024→2048, so it is not width-bound |
| **P1 low-rank gates (`F-r128` etc.)** | ❌ **NO — actively misleading** | The failure mode (GaLore's `W=BA` collapsing 142.53 vs 15.56 ppl) **appears at 1B and not below**. A small-scale "low-rank is fine" is exactly the false negative that motivated the whole `structure_energy.py` caveat. **A small-scale P1 null is uninformative.** Do not run it and do not cite it |
| **P2 cross-layer KV sharing** | ❌ **MOSTLY NO** | The *bytes* saved are scale-invariant (12 KiB/token, HANDOFF key decision 1) but the *quality cost* is a capacity question. At d=256 the KV capacity being shared is negligible, so "sharing is free" is nearly guaranteed and nearly meaningless |
| **Absolute downstream benchmarks (MMLU/GSM8K/COPA)** | ❌ **NO** | At chance below 1B (`HANDOFF.md:425-427`). Use sliced CE / AR-Hits perplexity and synthetic recall only |
| **Any 32K long-context claim** | ❌ **NO** | Needs a matched 32K training stage (`HANDOFF.md:449`) |

**Net:** small-scale from-scratch **cleanly settles P3 (widths) and the new depth-varying arm, gives a
directional read on topology, and cannot speak to P1 or P2.** That is a well-defined 40-50% of the
science for ~2% of the compute.

### 7.4 The strategic consequence

Cross-referencing §7.3 with the existing arm list (`liv_arms.py`: `L0, A16-P, F-r128, F-r256, G-grouped,
N-narrow, W-k5/k9/k15, A-fewer3, Q-mqa`):

- **The arms that survive small scale (`W-k*`, `L0`, `A16-P`, `A-fewer3`) are exactly the ones §4.6
  supplies a prior for, and exactly the ones that are NOT P1.**
- **The arms that need 350M (`F-r*`, `G-grouped`, `N-narrow`, `Q-mqa`) are exactly the P1/P2 arms whose
  motivating claims are already dead or narrowed** (P1's latency claim killed, P1's premise falsified,
  P2's latency ≈ 0 by construction).

**⇒ A defensible re-plan: run E at 20-50M with 8 seeds to settle widths + topology direction, run A/B/C
on released checkpoints for the headline, and let those results decide whether the 8-day 350M program is
still the right way to spend the SB-AWS budget.** That is the concrete recommendation this section
supports.

**NEGATIVE-RESULT-SAFE: YES for the width arm** (it has a pre-registered prediction from §4.6, so
confirming it is a result and contradicting it is a bigger result). **PARTIALLY for topology** (a null
at 20M is weak evidence about 350M — say so up front).

---

## 8. EXPERIMENT F — MQAR on real `L0`, and whether it is still needed

### 8.1 What recalibration would actually cost (INFERRED, arithmetic shown)

`probes/mqar/README.md:97-104` warns the operating point does **not** transfer from the 4-layer/d=128
calibration model to real `L0` (16 layers, 6 attention at `[2,5,8,10,12,14]`, d=1024). Costing it:

- Calibrated budget is fixed and enforced: **8000 steps × batch 64 = 512k examples** at `seq_len` 512
  ⇒ **2.6e8 tokens per run** (`mqar_calibrate.py` refuses to run below this — MEASURED behaviour).
- The full sweep is 4 capacity configs + 5 distance configs = **9 configs × 5 seeds = 45 runs**.
- Reference: job **1670987** ran the whole 45-run sweep at d=128 in **2 h 53 min** on one L40S (MEASURED
  via `sacct`).

| model used for recalibration | params | FLOP/run (`6ND`) | time/run @45 TFLOP/s | **45-run sweep** |
|---|---:|---:|---:|---:|
| current calibration model (4L, d=128) | ~1M | 1.6e15 | (measured: ~3.9 min) | **2.9 L40S-h** MEASURED |
| **`L0`-topology proxy: 16L, 6 attn at `[2,5,8,10,12,14]`, d=256** | **~13M** | 2.0e16 | ~20 min* | **≈ 15 L40S-h** |
| `L0`-topology proxy, d=512 | ~50M | 7.8e16 | ~1.0 h* | ≈ 45 L40S-h |
| **real `L0`, d=1024** | **354M** | **5.5e17** | **≈ 3.4 h** | **≈ 153 L40S-h** |

\* small models run well below peak; times inflated ~1.7× over the naive FLOP division. INFERRED.

**Recalibrating on the real 354M `L0` costs ~153 L40S-h ≈ 38 h wall-clock at 4 concurrent jobs.** That is
**not a cheap experiment** — it is comparable to a real training stage, and it produces *no scientific
result at all*, only a difficulty setting for a later experiment. **It does not belong in this inventory
as a cheap experiment, and the fact that it does not is worth stating loudly.**

### 8.2 Is it even needed? Three reasons to say largely no

1. **The caveat's own mechanism says use a topology proxy, not the real model.** `README.md:99-103`
   identifies the confounds as **depth (4→16), attention count and placement (2 layers at (1,3) → 6 at
   `[2,5,8,10,12,14]`), and width (128→1024)**, and explicitly states the cliff is **"not a
   receptive-field limit — the attention layers are global, so reach is not binding; what degrades is the
   difficulty of *finding* the recall circuit as distractors multiply."** **Circuit-findability is
   governed by depth and attention placement far more than by width.** So a **16-layer, 6-attention-at-
   the-real-indices, d=256 proxy captures the confounds that matter at ~1/10 the cost** (15 vs 153
   L40S-h). **This is the actionable recommendation.**
2. **The arms are changing, so the operating point is a moving target.** §7.4 argues the surviving arms
   may be widths + topology, not P1/P2. Calibrating today against an arm list that §7.4 recommends
   revising spends 153 GPU-h on a setting that may be re-derived anyway. **Calibrate last, against the
   final arm list — never first.**
3. **Experiment A largely supersedes MQAR as the headline recall endpoint.** MQAR's role was "measure
   recall, which nobody does for hybrids." Experiment A does that on **real natural-language retrieval,
   on a 10T-token production checkpoint, against a vendor's own unsupported 32K claim** — strictly more
   publishable than a synthetic task on a 20M model. **MQAR demotes from headline endpoint to
   arm-discrimination instrument.** As an instrument it needs only *enough* difficulty separation, which
   the proxy provides.

### 8.3 What I would keep from the MQAR work

- **The 1/D floor and `degenerate_floor()` — keep unconditionally.** `ln(vocab/2)` / `ln(D)` / 0 ladder,
  observed to 2 decimals. This is a genuine methodological contribution and transfers to any scale.
- **The high-load bimodality correction** (report success rate AND median accuracy vs floor at `N512_D64`).
- **The two guarded process failures** (positive control before sweep; budget constants owned by the
  script). These are the reason the harness is trustworthy.
- **43 generator tests.**

**Cost of the recommended path:** proxy recalibration ≈ **15 L40S-h, ~4 h wall-clock at 4 concurrent,
~2 eng-h** (the sweep script exists; it needs a topology swap in `mqar_model.py`).

**NEGATIVE-RESULT-SAFE: NO.** A calibration sweep produces an operating point, not a finding. If it
returns "everything is at ceiling" or "everything is at floor" you have burned the compute and must
re-run. **This is the only item in the inventory with that property, and it is the main reason it ranks
low despite being technically cheap in its proxy form.**

---

## 9. BEYOND THE ASSIGNED LIST — six more zero-training experiments

The brief asked me to go past the six categories. These are additional, all inference-only or
weights-only, ordered by value.

### G — Attention-map retrieval attribution (which of the 6 attention layers is the retriever?)

**Question:** on a passkey prompt, which attention layer/head attends from the query position to the
needle position? Score = attention mass on the needle token.

**Cost:** 0.3 L40S-h, 3 eng-h. One forward pass with `output_attentions=True` over ~100 passkey prompts
per length. INFERRED.

**Why it matters:** it is the **mechanistic complement to B3**, and the two cross-validate. B3 says
"layer 8 breaks when windowed" (causal); G says "layer 8 puts 0.7 attention mass on the needle"
(correlational). Agreeing methods make a much stronger claim than either alone. It directly answers
**"which layers must `A-fewer3` keep?"** — a design decision currently made by guessing.

⚠️ **Limit:** attention mass ≠ information flow (the classic attention-is-not-explanation objection).
Pair it with B3; do not publish G alone.

**NEGATIVE-RESULT-SAFE: YES.** "Retrieval is diffuse across all 6 layers" is as reportable as
"concentrated in 2."

### H — KV-cache-traffic measurement to validate the one surviving efficiency claim

`HANDOFF.md:483-486` states the **only** testable efficiency claim belongs to the topology: mostly-LIV
vs param-matched all-GQA saves 20 KiB/token, crossing 10% of decode traffic at **T ≈ 4,121**. That is
currently an **analytic** number. The released checkpoints let you measure real decode traffic
(`torch.cuda.max_memory_allocated` + `ncu --metrics dram__bytes_read`) on **LFM2-350M vs a
param-matched all-attention model of the same size** (Qwen3-0.6B is already cached; not param-matched,
but a real second datapoint).

**Cost:** 0.5 L40S-h, 4 eng-h. **Why it matters:** the team's own history says analytic models mislead —
`l40s_breakeven.py`'s roofline prediction was overturned by the iso-byte control, and the README's own
lesson is *"for any decode-time factorization, run an iso-byte control before trusting a roofline
estimate."* **The T≈4,121 crossover has not had that treatment.** It is the last unmeasured efficiency
claim in the design and it is measurable on released weights today.

**NEGATIVE-RESULT-SAFE: YES.**

### I — Logit-lens / per-layer readout on retrieval prompts

Project each layer's residual stream through the (untied) LM head on passkey prompts and find **the
layer at which the needle token becomes the top prediction.** ~0.2 L40S-h, 3 eng-h. Tells you the depth
at which retrieval resolves — which, combined with §4.6's depth profile, could show whether retrieval
completes *before* the passthrough-dominated late LIV layers. **NEGATIVE-RESULT-SAFE: YES.**

### J — Sliced perplexity / AR-Hits on the released checkpoint

`HANDOFF.md:143-146` makes AR-Hits sliced perplexity a **primary endpoint**, chosen because published
ratio sweeps span only 0.06 ppl while recall differs by 20+ points. **The metric has never been computed
on anything.** Computing it on the four released checkpoints (a) validates the implementation before
betting the study on it, and (b) calibrates the expected effect size, feeding directly into the `s_δ`
power analysis that `HANDOFF.md:423` says must precede any gate. **0.3 L40S-h, 4 eng-h.**

**This is a cheap de-risking of a primary endpoint, and it is currently unscheduled.** **SAFE: YES.**

### K — LFM2 vs LFM2.5 weight diff

LFM2.5 variants exist. Diffing the conv taps and gate spectra between LFM2 and LFM2.5 at matched size
shows **what Liquid changed between generations**. **0 GPU-h** (weights-only, same script as §4.6, ~10
min). If LFM2.5's taps shifted toward the boundary, that is evidence Liquid found width mattered — a
directly relevant signal for free. **SAFE: YES.**

### L — Document-length distribution audit of the training corpus

`HANDOFF.md:370` and `:488` list "Audit Dolma2 document-length distribution" as unresolved. The
`ShortConv` docstring (`short_conv.py:203-211`) already asserts a **~622-token median**, which is
load-bearing for document-isolated packing (a k-tap filter crossing a document boundary is a *different
operator*). **CPU-only, ~1 h, 0 GPU-h.** It is listed as open but is trivially closable and it validates
a number already baked into the code.

**SAFE: YES.**

---

## 10. RANKED TABLE

Ranked by (information gained) / (cost). "L40S-h" = GPU-hours on one FarmShare L40S. Wall-clock assumes
up to **4 concurrent GPU jobs** (QOS-verified, §1.3). Engineering hours are ASSUMED unless noted.

| # | experiment | GPU-h | wall-clock | eng-h | decisive for | neg-safe? |
|---:|---|---:|---:|---:|---|:--:|
| **0** | **§4.1/4.6 — conv tap-energy + frequency response, all 4 released checkpoints** | **0** ✅ | **~5 min** ✅ | **~3** ✅ | **k=3 is not boundary-saturated at any scale (4.3-5.3% oldest-tap energy) ⇒ predicts the width arm is flat. Plus: LIV layers are heterogeneous by depth; layer 0 is a learned token-shift.** | **YES** |
| **1** | **§3 A1 — passkey length×depth sweep, 350M/700M/1.2B + control** | **3.4** | **~1.5 h** | **4-6** | **THE HEADLINE. First published recall benchmark for LFM2. Vendor declares 32K + markets RAG and publishes zero recall numbers.** | **YES (max)** |
| 2 | §3 A2 — BABILong qa1/qa2, 7 lengths | +1.5 | +1 h | +3 | Copy-retrieval vs reason-over-retrieved-facts. Distinguishes "finds the needle" from "uses it" | YES |
| 3 | §5 B3+B6 (trimmed) — attention→local-window sweep + conv t-2 tap zeroing | 2.5 | ~1 h | 6-8 | **Are the 6 attention layers global retrievers or local smoothers?** Directly evidences `A-fewer3`. B6 is the *causal* test of rank 0's prior | YES |
| 4 | §9 K — LFM2 vs LFM2.5 tap diff | **0** | ~10 min | 1 | Did Liquid change the conv between generations? Free corroboration for rank 0 | YES |
| 5 | §9 L — Dolma2 doc-length audit | **0** (CPU) | ~1 h | 1 | Closes an open HANDOFF item; validates the 622-token median already asserted in code | YES |
| 6 | §4 C3 — activation-weighted tap energy (32k tokens) | 0.2 | ~20 min | 3 | **Required before publishing rank 0.** The `spectra_v2` lesson says weight-space alone is not enough | YES |
| 7 | §9 J — AR-Hits / sliced perplexity on released checkpoints | 0.3 | ~20 min | 4 | De-risks a **primary endpoint** that has never been computed; feeds the `s_δ` power analysis | YES |
| 8 | §9 G — attention-map retrieval attribution | 0.3 | ~20 min | 3 | Which layers retrieve. Cross-validates #3 | YES |
| 9 | §9 H — measured KV/decode traffic vs the analytic T≈4,121 crossover | 0.5 | ~30 min | 4 | The **last unmeasured efficiency claim** in the design; the team's own history says roofline models mislead | YES |
| 10 | §9 I — logit-lens depth-of-retrieval | 0.2 | ~20 min | 3 | At what depth retrieval resolves; pairs with rank 0's depth profile | YES |
| 11 | §7 E — 20M from-scratch, widths `k3/k5/k9/k15` + depth-varying, 8 seeds | **14** | **~4 h** | 10-14 | **Settles P3 at small scale against a pre-registered prediction from rank 0.** Cheapest real training result available | YES |
| 12 | §5 B1/B2/B5 — full single-layer ablation profile | 7.5 | ~2.5 h | (reuses #3) | Per-layer contribution map; ratio-adjacent evidence | YES |
| 13 | §3 A3 — RULER (13 tasks, `hf` backend) | 3-5/ckpt | ~4 h | 8-12 | Converts #1 into a table reviewers accept. **Do only if #1/#2 show an effect** | YES |
| 14 | §7 E — topology direction at 20-50M (`L0` vs `A16-P` vs `A-fewer3`) | 14-45 | 4-11 h | +4 | Direction and FLOP-vs-param crossover shape only — **not** the optimal ratio | PARTIAL |
| 15 | §6 D — `ShortConv` incremental decode path | 0 | — | **~11.5** | Infrastructure. **Prerequisite for the training program, NOT for #0-#14.** Defer until after them | **N/A** |
| 16 | §8 F — MQAR recalibration on a 16L/6-attn/d=256 **proxy** | 15 | ~4 h | 2 | An operating point, not a finding. Do it **last**, against the final arm list | **NO** |
| — | §8 F′ — MQAR recalibration on the real 354M `L0` | **153** | **~38 h** | 2 | Same as #16 at 10× cost. **Not a cheap experiment. Do not do this.** | **NO** |
| — | lm-eval-harness on released checkpoints | ~2 | ~1 h | 3 (+install) | Reproduces the vendor's own published table. **Lowest value in the inventory** | YES but trivial |

### Totals for the recommended cheap program (ranks 0-11)

| | |
|---|---:|
| **GPU** | **≈ 23 L40S-h** |
| **Wall-clock** | **≈ 10-12 h of GPU time**, spread over ~1 week of engineering |
| **Engineering** | **≈ 40 h ≈ 1 focused week** |
| **From-scratch pretraining required** | **NONE except rank 11 (20M, 14 L40S-h)** |

**For scale: the 8-day 8×GPU program is ~1,500 GPU-hours. The entire inventory above is ~1.5% of it and
is already producing results.**

---

## 11. The three things I would tell the team

1. **Rank 0 is done and it already changed the plan.** 4.3-5.3% oldest-tap energy across all four
   released checkpoints says the uniform width arm will very likely be flat, and the depth heterogeneity
   (layer 0 = token-shift at 95% of channels; layer 15 = 97.6% passthrough) says the *right* width arm is
   **depth-varying**, which is not in `liv_arms.py`. **Cost: zero GPU-hours.** Also note
   `short_conv.py:347-348` initializes the conv to *current-token* identity, the opposite of what layers
   0-1 converge to — a one-line, evidence-backed change.

2. **The recall benchmark is not a nice-to-have; it is the paper.** MEASURED: the LFM2-350M model card
   declares **32,768 tokens**, reports **seven** benchmarks (MMLU/GPQA/IFEval/IFBench/GSM8K/MGSM/MMMLU),
   markets the model for **RAG and data extraction**, was trained on **10T tokens** — and publishes
   **zero** retrieval numbers, while `config.json` separately says `max_position_embeddings: 128000`.
   **≈3.4 L40S-h and ~1.5 h of wall-clock produces a first-ever table on shipped production weights, and
   every possible outcome is publishable.** Nothing else in the plan has that ratio.

3. **Sequence the decode path AFTER the cheap experiments, not before.** It is ~11.5 h of the best
   engineering time, it produces no reportable fact, and it is a prerequisite only for the phase that
   ranks 0-11 might rescope. Meanwhile: the **6-hour job limit does not exist** (partition `MaxTime` is
   **48 h**; the sibling KDA track ran **20-hour** jobs on this cluster) and **4 concurrent GPU jobs are
   allowed** — so every wall-clock estimate the plan was built on is ~4× pessimistic.

**Artifacts produced by this document, all on FarmShare:**
- `/scratch/users/ericrcwu/liv/ckpt/` — LFM2-350M, 700M, 1.2B, 2.6B (7.8 GB, downloaded 2026-08-01)
- `/scratch/users/ericrcwu/liv/tapread.py` — per-layer tap energy, 350M
- `/scratch/users/ericrcwu/liv/tapfreq.py` — tap energy + FIR frequency response, all four checkpoints

*(END)*
