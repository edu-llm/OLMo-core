# Running M20 (Maple-Preview shape) on your own data

**Audience:** someone outside the eduLLM AWS account who wants to train
`TransformerConfig.maple_m20` on their own data servers.

**Status of this document:** every claim below was read out of the code on this branch. Where a
number was *measured*, it says so. Where a number is *derived and has never been observed*, it says
that too — and the most important thing on this page is in that second category.

---

## 0. Read this first: what has and has not been proven

**M20 has never been constructed. Not on a GPU, not on a CPU, not once.** No forward pass has ever
run at this size. There is no loss curve, no throughput number, no memory measurement, and no
checkpoint.

What *does* exist is strong: an assertion in the factory that fires at config-build time, closed-form
arithmetic, an adversarial term-by-term walk of the config tree, and 87 passing tests. The parameter
counts are a **falsifiable prediction**, and the factory is arranged so that a wrong prediction fails
the run before a GPU is touched. See the comment on `MAPLE_EXPECTED_PARAMS` in
`src/olmo_core/nn/transformer/config.py` — it labels the M20 row `DERIVED, NOT MEASURED` and explains
why two agreeing derivations is not proof.

So: **the first thing you should do is build the config and print the parameter ledger.** If the
counts match, you have independently confirmed something we never have. If they do not, you have
found a real bug for the cost of a CPU process.

Related unproven item, if you go near quantization: the quality cost of a 4-bit head was measured
against a **full-precision body**. The intended body is **ternary**. Nothing bounds the interaction
between those two, and we have not measured it.

---

## 1. The blocker: `.edullm/train_on_corpus.py` will not run for you

The platform entrypoint imports two private modules inside `resolve_corpus()`
(`.edullm/train_on_corpus.py:458-459`):

```python
from edullm_data.read import dataset_paths, resolve_latest
from edullm_data.s3 import Boto3S3
```

`edullm-data` is a private package installed in the platform image from a commit-pinned GitHub
tarball. **You cannot `pip install` it, and it would not help you if you could** —
`edullm_data/read.py:25` hard-codes a module-level

```python
DATA_BUCKET = "<a private S3 bucket in our account>"
```

with **no environment-variable override**. It is a default argument threaded through
`dataset_paths`, `resolve_latest` and friends, so it is overridable *in Python* but not from any
command line or env var the entrypoint exposes.

This is not a soft dependency you can work around with flags. Three independent locks:

1. **`build_config()` calls `resolve_corpus()` as its first statement** (`:842`), so there is no
   code path — not `--dry-run`, not `--prepare-heldout-only` — that reaches config construction
   without importing `edullm_data`. You get exit 70 and `ModuleNotFoundError`.
2. **There is deliberately no path flag.** The file header says so outright: *"There is no path
   literal in this file and there is deliberately no flag to supply one."* `--dataset-id`,
   `--dataset-version` and `--dataset-tokenizer` are *identifiers for a private resolver*, not paths.
3. **`main()` refuses empty identifiers before anything else** (`:1977-1994`) — exit 64 unless all of
   `EDULLM_DATASET_ID`, `EDULLM_DATASET_VERSION`, `EDULLM_DATASET_TOKENIZER` and
   `EDULLM_CHECKPOINT_DIR` are non-empty. The dotlist override cannot rescue you either:
   `config.merge(overrides)` is the *last* line of `build_config` (`:1166`), long after the failure.

There is also a fourth, quieter coupling: **`vocab_size` is not a flag.** It is
`corpus.tokenizer.padded_vocab_size()` (`:877`), i.e. it comes off the resolved private manifest.

**Use `src/scripts/train/maple_m20_local.py` on this branch instead.** That is why it exists.

### The good news

**The model itself has no private dependencies at all.** `TransformerConfig.maple_m20` is pure
`olmo_core`. The entrypoint's *top-level* imports (`:65-125`) are stdlib + `torch` + `rich` +
`olmo_core` — nothing private. The blocker is entirely in the data-resolution layer, and that layer
is duck-typed at `corpus_from_manifest` (`:375-382`), which is the intended seam if you would rather
patch the platform script than use the local one.

### What is portable, what needs a flag, what is un-runnable

| | Verdict |
|---|---|
| `TransformerConfig.maple_m20` and the whole ladder | **Portable, zero changes.** Pure `olmo_core`. |
| `--model-factory` dispatch (`getattr(TransformerConfig, name)`, `:856`) | **Portable.** No allowlist; any classmethod name works. |
| Chunked cross-entropy (`--lm-loss-implementation chunked_linear`) | **Portable.** In-tree, no dependency. |
| Trainer, FSDP2, checkpointer, W&B | **Portable.** Leave `EDULLM_WANDB_PROJECT` unset and W&B silently disables itself (`:1056`). |
| Local checkpoint dir | **Needs a flag.** `--save-folder /path` — S3 is not required. |
| Corpus resolution (`:454-517`) | **Un-runnable.** Private package + hard-coded bucket. |
| `vocab_size` | **Un-runnable as written.** No flag; comes off the private manifest. |
| `fused_linear` loss | **Asymmetric.** Needs liger-kernel, which is *absent* from the platform image but installable for you. Works for you, not for us. |
| `_download_to` (`:740-754`) | **Un-runnable.** boto3-only, no local branch. Only on the held-out eval path. |
| `max_checkpoints=None` (`:1041`) | **Portable but wrong for you.** It exists because our IAM role is denied `s3:DeleteObject` on `.metadata.json`. On local disk you probably want the library default of 3, or you will fill the disk. |
| `downstream_evaluator` | **Does not exist.** Never imported. Nothing to disable. |
| `lm_evaluator` | Auto-enables only if the private manifest yields `val_paths`. With none, training runs unevaluated after a warning (`:1150-1155`). |

No AWS environment variable is read anywhere in the entrypoint, and there is **no hard-coded bucket,
ARN, account number or region in its executable code** — credentials come from the implicit boto3
chain.

---

## 2. The model, from code

Read off `MAPLE_RUNGS["M20"]` in `src/olmo_core/nn/transformer/config.py`:

| | |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 24 |
| `num_experts` | 256 |
| `top_k` | 8 |
| expert FFN hidden (`f_e`) | 512 |
| `n_heads` | 16 |
| `n_kv_heads` | 4 (GQA 4:1) |
| `head_dim` | 128 |
| shared experts | 0 |
| embeddings | untied |
| attention | 3:1 SWA-512 : global, **NoPE on the global layers** |
| rotary | partial, factor 0.5 |
| QK-norm | on, per-head |
| SwiGLU clamp | `MAPLE_SWIGLU_LIMIT = 7.0` (`nn/feed_forward.py:31`) — an *architecture* choice, gpt-oss's shape |
| vocab | 100,352 (padded dolma2) — **not** Maple's 151,936 |

**Do not hand-copy the parameter counts.** Read them from
`TransformerConfig.MAPLE_EXPECTED_PARAMS[100352]["M20"]` and
`MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[100352]["M20"]`. Four errors in this project entered by
transcription. The local script prints them for you.

Ratio identities, all exact at M20 (which is why it is the fixed point the ladder scales *down*
from): `f_e/d = 1/4`, `k·f_e/d = 2.0`, `k/E = 1/32`, `n_heads·head_dim = d` (1.0×, **an asserted
invariant** — it is what catches a width error), `GQA = 4:1`, `L ≡ 0 mod 4`.

### External validation worth knowing

Our M20 total is **20.00B, not DeepGrove's published 20.2B, and that is correct rather than an
error.** The entire difference is the untied embedding pair at a different vocabulary:
`2·d·(151936 − 100352) = 211,288,064`.

Run the same closed form at **V=151,936** and it reproduces DeepGrove's published
**20,214,030,336 total / 1,490,657,280 active exactly**. That is an independent external check on the
geometry, and it is enforced as a test:
`src/test/nn/transformer/maple_ladder_test.py::test_m20_differs_from_maple_only_by_the_vocabulary`.

So: the shape is validated against a third party. The *parameter count at our vocab* is still only
derived. Both statements are true at once.

---

## 3. Things that will otherwise cost you a day

Ordered by how expensive the mistake is.

### 3.1 `accumulate_grads_without_comm=False` is MANDATORY at 20B

`TransformerDataParallelConfig.accumulate_grads_without_comm` defaults to **`True`**
(`src/olmo_core/train/train_module/transformer/config.py:179`). At 20B that wants an **unsharded fp32
gradient accumulator of 74.51 GiB/rank**, which fits nothing.

With it `False`: **40.30 GiB/rank**. That is **106.1% of an A100-40GB — it still does not fit** — and
about **51% of an H100-80GB**, which does.

The platform entrypoint does **not** set it (`:955-959` sets only `name`, `param_dtype`,
`reduce_dtype`), so it inherits `True`. The local script sets it `False` explicitly.

**M20 needs 80GB cards.** Do not plan an A100-40GB run.

### 3.2 `--rank-microbatch-size` defaults to 16384 and OOMs

`.edullm/train_on_corpus.py:1806`, `default=16 * 1024`. Its own help text says the default does not
fit the R3 flagship on A100-40GB with the default loss implementation. It is worse at M20. Also note
it is **not numerics-neutral for MoE** — expert capacity is derived from it — so hold it fixed across
any arms you compare.

### 3.3 Five simultaneously-live fp32 `(N,V)` logit buffers, not one

The binding memory constraint is the logits, and the naive single-buffer estimate is low by 5×. At
R3 scale the real figure is **~13.81 GiB, not 3.06** — two of the five are *backward* buffers, and
one is a z-loss `logsumexp` backward. A five-term model agreed with measurement to 0.2%.

**Z-loss is on by default** in the factory (`z_loss_weight=0.001`), and the train-module
`--z-loss-multiplier` defaults to `1e-5`.

Fix: `--lm-loss-implementation chunked_linear` (in-tree, `nn/functional/chunked_cross_entropy_loss.py`).
Costs ~4.9% extra step FLOPs at R3, so hold it fixed across throughput comparisons.
**Do not reach for `fused_linear`** unless you have installed liger-kernel — it is not in the
platform image and raises at the first micro-step.

### 3.4 MoE defaults that are silently wrong (the factory already fixes these — do not undo them)

Stock OLMo-core defaults, every one of which the Maple factory overrides explicitly:

- **`normalize_expert_weights` must be 1.0.** Left unset, measured gate mass is **0.161 vs 1.000** —
  a **6.2× error that trains happily**. This is Maple's `norm_topk_prob`. **Zero of five shipped
  recipes set it.** The factory hard-codes `1.0` (not even a kwarg) and
  `_maple_assert_ladder` rejects anything else.
- **`top_k` defaults to 1.** Left unset you have a top-1 model, not top-8.
- **`capacity_factor=None` silently becomes 1.2**, because `MoEConfig.build` calls
  `as_dict(exclude_none=True)` and the key vanishes. The factory sets **2.0** explicitly.
- **`z_loss_weight` defaults off.** Factory sets `0.001`.
- **Both aux-loss weights are divided by total `n_layers`, not MoE depth** → 1.5× low.
- **`bias_gamma=None` deliberately.** Maple has no expert bias
  (`moe_router_enable_expert_bias: false`). Do not switch it on as a load-balancing fallback — it
  changes the architecture under test.

### 3.5 Load balancing is rank-local only

`MoELoadBalancingLossGranularity` offers only `local_batch` and `instance`. There is **no
global-batch option and no DP all-reduce.** The only globally-balanced mechanism in the codebase is
`bias_gamma` — and per 3.4, Maple has no expert bias, so that fallback is unavailable. **This bites
hardest at E=256**, which is M20. Budget for it; do not be surprised by it.

Also: the `load imbalance` metric is **max/mean**, which is **not comparable across different expert
counts.** Use the CV / entropy / dead-fraction metrics, and read the per-block `block NN/` series —
the aggregates are summed across blocks (entropy logs 15.97 for a quantity that lives in [0,1]).

### 3.6 On A100 there is no real grouped kernel — so do not bother with dropless

`grouped_gemm` does **not build** in a standard container: no nvcc, no CUDA headers, and CUTLASS
arrives only as a git submodule (and the image has no git). If you *do* build it, note upstream ships
`TORCH_CUDA_ARCH_LIST="9.0 10.0"`, which compiles cleanly and then dies at the first GPU call on
A100 — you need `8.0`.

`torch._grouped_mm` is not an escape hatch: its fast path gates on **sm90/sm100**, so A100
(**sm_80**) falls through to an `offs.cpu()` + per-group `at::mm_out` loop. **A dropless throughput
arm on A100 measures that fallback loop, not the architecture.**

The code now refuses rather than warns: `nn/moe/mlp.py` raises above 64 experts without
`grouped_gemm`, because the fallback issues one host sync per expert per matmul per layer — order
10^4 syncs per step at E=256 — and would *train successfully* while reporting a throughput nobody
can explain. Dropless also rejects `capacity_factor` outright, so it is not a like-for-like kernel
swap.

### 3.7 Report bits-per-byte, never raw loss, across tokenizers

`ln(100352) = 11.516` vs `ln(151936) = 11.931` — a **0.415-nat offset before any fertility
difference.** Our loss is not comparable to Maple's published loss. If you train at a third
vocabulary, it is comparable to neither.

### 3.8 Ternary QAT buys you nothing at training time

Measured: **MFU 14.244 bf16 vs 14.183 ternary, −0.43%.** `nn/quantization.py` states
`num_flops_per_token` is **deliberately unchanged**: the forward consumes a dequantized tensor in the
compute dtype, the backward runs full precision through the straight-through estimator, and the
latent master weights are full size. No memory saving, no arithmetic saving. **The win is at
inference, which is out of scope here.** Anyone budgeting ternary as a training-time saving is wrong.

Full precision is preserved on embeddings, LM head, **router**, and all norms. The router carve-out
is load-bearing, not stylistic — routing is discrete, so quantizing the router changes *which experts
fire*. `audit_quantization` walks the built model and raises if anything in the carve-out got
quantized.

### 3.9 Throughput measurement discipline

A tokens/s number with no denominator is how a benchmark reports the wrong sign and reproduces
cleanly. Start the clock **after steady state** — `torch.compile` alone can take minutes on step 1.
Discard ~50 steps, report a fixed window, report **median** step time excluding checkpoint and eval
steps. Watch for periodic stalls as the loader touches new cold shards mid-run.

**And print the FLOPs formula you used.** `6·(active_params − embed_params)` is **9.4% LOW at R3** —
the lm_head is **43.7%** of R3's FLOPs/token (a sibling dense model's was ~0.05%) because d=1024 runs
against V=100,352. At M20's d=2048 the head's share is smaller but still not negligible. You cannot
lump the head with the embedding and drop it.

### 3.10 One operational trap you will not be able to reproduce

`--prepare-heldout-only` must be run **once, single-process, before torchrun, sharing the same
`--work-dir`.** Skipping it makes `prepare()` open a 96-worker spawn pool behind a collective, which
deadlocked two 8-GPU runs at a 900s gloo timeout. On a small box `os.cpu_count()` is low enough that
**the deadlock does not reproduce** — so you may reasonably conclude the step is unnecessary and
delete it. It is not unnecessary at 8 GPUs on a big host.

---

## 4. How to actually run it

`src/scripts/train/maple_m20_local.py` on this branch. It is **untested — see the banner in the
file.** It takes local or HTTP `.npy` token shards and never imports `edullm_data`.

Start with the free step, which needs no GPU and no data:

```bash
python src/scripts/train/maple_m20_local.py --dry-run
```

That builds the config, runs every ladder assertion, and prints the parameter ledger against
`MAPLE_EXPECTED_PARAMS`. **If those numbers match, you have proven something we have not.**

Then, on 80GB cards:

```bash
torchrun --nproc-per-node=8 src/scripts/train/maple_m20_local.py \
  --data /data/tokens/shard_0000.npy /data/tokens/shard_0001.npy \
  --save-folder /scratch/m20 \
  --rank-microbatch-size 4096 \
  --sequence-length 2048
```

Your data must be **headerless, flat `uint32` token IDs** (`.npy`), which is the dolma2 convention —
`--data-dtype` covers `uint16`/`uint32`. If your tokenizer is not dolma2, pass `--vocab-size` and
`--eos-token-id` to match it, and re-read §3.7 before comparing any loss.
