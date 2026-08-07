# Frontload-cl training design decisions

Companion to:

- `EXPERIMENT-early-behavior-primer.md` — scientific claim, arms, decision rule
- `DATASET-DESIGN-frontload-cl.md` — what gets published to `edullm-data`

This note records **implementation** choices in `.edullm/frontload_cl/`: why the train path looks the way it does on the eduLLM platform, and what was deliberately deferred.

---

## Status (2026-08-07)

| Piece | State |
| --- | --- |
| Train scripts (`.edullm/frontload_cl/`) | Ready: shared HQ warmup; primer block vs control flat after warmup; ladder 370M; `--param-dtype` |
| FlashAttention-2 + A100 defaults | Wired: image wheel + `--attn-backend flash_2`, microbatch `24×4096`, HSDP, `--smoke` |
| Local token build (`data/frontload-cl/`) | Complete (~10.1B train tokens); used only as build input |
| Publish `pretrain/frontload-cl-10b` | **`v1` on `s3://edullm-data`** (~37.7 GiB; train ~10.09B / val 31M tokens) |
| Publish `sft/frontload-cl-chat-sft` | **`v1` on `s3://edullm-data`** (conversation JSONL) |
| Platform `datasets.yaml` registration | **Done** — `frontload-cl-10b-v1` runs (`edullm data`); SFT release is `exits_69` if named as train dataset (no tokenizer) |
| `.edullm/run.yaml` (+ smoke / control specs) | Committed: `gpu-8xa100`, dtype + checkpoint dir in command text |
| Image / `edullm/**` branch | **Not pushed** for latest HEAD — push so ECR builds; wait for scan before submit |
| Shared SFT (`train_sft.py`) | Wired: tokenize conversations → masks, 1-epoch 370M SFT from `--checkpoint` |
| Colab 1×A100 smoke | `.edullm/frontload_cl/colab_smoke.ipynb` + `colab_smoke.py` (microbench / synthetic; not 8-GPU) |
| Held-out SFT-like NLL callback | Not wired |
| Downstream eval (GSM8K / ARC / IFEval) | After shared SFT, via `olmo-eval-full` — not this image |

**Next concrete steps:** push `edullm/frontload-cl` (image + scan) → `edullm check --spec .edullm/run-smoke.yaml …` → submit smoke → two full arms (`run.yaml` / `run-control.yaml`).

---

## 1. Where the code lives

**Decision:** custom scripts under `.edullm/frontload_cl/`, not flag-only use of `.edullm/train_on_corpus.py`, and not a separate GitHub repo.

**Why:**

- Platform Level 1 (`train_on_corpus.py` + flags) cannot express a multi-phase curriculum (primer block vs flat mix). Level 2 is “one Python file on your branch.”
- The experiment is OLMo-core training on a published corpus. Platform already registers `OLMo-core`; a fork repo would need its own image, workload profiles, and ECR entry for no gain.
- Keeping schedules in-repo matches the dataset design: *“Schedules (primer block vs flat): in the train script, not separate `curriculum/` datasets.”*

**Layout:**

| File | Responsibility |
| --- | --- |
| `constants.py` | Ladder 370M hparams + token budgets (single source of truth for numbers) |
| `attn.py` | Resolve `--attn-backend`; refuse early if flash-attn missing |
| `corpus.py` | Resolve published corpus; group shards by `tokens/<source>/` |
| `schedule.py` | Build primer vs control `InstanceSource` phase lists |
| `train_pretrain.py` | Platform entrypoint (`--arm primer\|control`, `--smoke`) |
| `smoke_pretrain.py` | Thin wrapper that forces `--smoke` |
| `train_sft.py` | Shared post-PT SFT: tokenize + 1-epoch train from `--checkpoint` |
| `sft_tokenize.py` | Conversation JSONL → OLMo 2 chat tokens + label masks |

---

## 2. One corpus, two arms

**Decision:** both arms read the same published dataset `pretrain/frontload-cl-10b`. Arm difference is **ordering only**, implemented in `schedule.py`.

**Why:**

- Isolates the claim (timing of SFT-like tokens) from data-identity confounds.
- Avoids publishing two near-duplicate 10B corpora or `curriculum/` orderings that would freeze schedule into sealed bytes.
- Source folders under `tokens/` (`fineweb-edu-main`, `fineweb-edu-anneal`, `finewiki`, …) are the mix units; the train script filters paths by folder name after `edullm_data.read.dataset_paths`.

**Implication:** the platform form picks one `dataset_release`. `--arm` on the command selects the curriculum. Two submissions = two runs, same corpus, different arm.

---

## 3. Curriculum via composable API, not dataloader hacks

**Decision:** use OLMo-core’s composable stack (`NumpyDocumentSource` → `ConcatAndChunkInstanceSource` → `MixingInstanceSource` → phases passed to `ComposableDataLoader` with `ShuffleStrategy.intra_source`).

**Why:**

- Official curriculum pattern in `olmo_core.data.composable`: shuffle within each phase, concatenate phases in order.
- Ratio mixes (`MixingInstanceSource`) give exact token targets for HQ / SFT-like / anneal without hand-rolled index files.
- `set_composable_seed(DATA_SEED)` keeps sampling reproducible across arms where they share structure.

**What “HQ” means here:** FineWeb-Edu main (`fineweb-edu-main`), FineWeb-Edu anneal (`fineweb-edu-anneal`), and FineWiki (`finewiki`). Not HQ (SFT-like): `cosmopedia-v2`, `finemath-4plus`, `openhermes-pt`, `natural-reasoning`.

**Shared warmup (both arms, identical split seed):**

1. ~371M HQ main only (FineWeb-Edu main + FineWiki @ 5%) — covers the LR warmup window.
   ``CosWithWarmup`` steps are separate and also identical. No SFT-like during warmup.

**Primer after warmup:**

2. 100M contiguous SFT-like block
3. Remaining HQ main + remaining 100M SFT-like, mixed
4. 1B anneal (FineWeb-Edu anneal + FineWiki @ 5%; no SFT-like)

**Control after warmup:**

2. Remaining HQ main + all 200M SFT-like, flat mix
3. Same 1B anneal as primer

**Arm differences beyond curriculum:** primer-only checkpoint milestone after the SFT block; W&B run name includes the arm; post-warmup mix seeds are `DATA_SEED + 69` (primer) and `DATA_SEED + 420` (control). Same model, batch, LR, corpus, anneal, and loss otherwise.

**FineWiki disjointness:** the published FineWiki pool is one folder (~490M). Main (440M) and anneal (50M) slices are a seeded `split` before mixing so the two HQ phases do not resample the same documents.

**Implementation details (schedule robustness):**

- Mix `num_tokens` are floored to a multiple of sequence length before `MixingInstanceSource` sees them.
- Specs use `max_repetition_factor=1.05` so a `ConcatAndChunk` trailing remainder (< seq len) does not refuse mixes that sit on published pool sizes. Without this, a complete corpus still fails at the FineWeb / FineWiki edges under factor `1.0`.
- Shared HQ warmup is split from the *actual* `hq-main` size (`WARMUP_TOKENS / hq.num_tokens`), not the nominal `HQ_PRE_ANNEAL` constant, so the data phase lines up with 472 LR warmup steps. Primer and control call the same helper with the same seed.
- Primer SFT block uses `PRIMER_BLOCK / SFT_LIKE_TOTAL` (not a bare `0.5`).
- `train_pretrain.py` refuses early if total curriculum tokens `< steps × global_batch` (incomplete FineWeb fails loudly instead of under-training).

---

## 4. Hyperparameters: ladder 370M, not production OLMo2

**Decision:** hardcode AI2 model-ladder 370M settings in `constants.py` (and default CLI flags), not OLMo2 1B/7B production defaults from `train_on_corpus.py`.

| Knob | Value |
| --- | --- |
| Model | `TransformerConfig.olmo2_370M` |
| Seq length | 4096 |
| Global batch | 192 × 4096 = 786,432 tokens |
| Peak LR | 7.8×10⁻⁴ |
| Warmup | 472 steps (~371M tokens) |
| Schedule | linear warmup → cosine over the run |
| Optim | SkipStepAdamW, wd 0.1, betas (0.9, 0.95) |
| Grad clip | 1.0 |
| Duration | 12,715 steps (= 10B ÷ global batch) |

**Why:** the experiment doc requires ladder Table 1; using `train_on_corpus` defaults (190M-ish, shorter seq, different LR) would invalidate the comparison.

**Microbatch:** default `--rank-microbatch-size` is `24 × 4096` (fills the per-rank
share of global batch 192 on 8 GPUs — no grad accumulation). Lower if OOM
(e.g. `8 × 4096` or `2 × 4096`). Global batch stays fixed at the ladder value.

**Attention / compile:** default `--attn-backend flash_2` (Dao FlashAttention-2).
The eduLLM image installs the official prebuilt wheel (SM80/A100+) — see
`.edullm/Dockerfile`. Escape hatch: `--attn-backend torch` (PyTorch SDPA).
`compile_model=True` remains on (image has `gcc` for Inductor).

**DP:** HSDP + bf16 params / fp32 reduce (matches OLMo2-1B-style scripts).

**Checkpoints:** every **1000 steps** by default (`--save-interval`), plus curriculum milestones via `CheckpointerCallback.fixed_steps`:

| Milestone | Approx step | Arms |
| --- | --- | --- |
| After LR warmup | 472 | both |
| After SFT-like primer block | ~599 | primer only |
| At anneal start | ~11,444 | both |
| End of run | 12,715 | both (`post_train`, automatic) |

A milestone within `--checkpoint-milestone-proximity` (default 100) of a periodic save is skipped. With the nominal budgets above, all three milestones are kept (none are near a multiple of 1000). Final checkpoint is always written by the checkpointer’s `post_train`. ~13 periodic + a few milestones, all retained (`max_checkpoints=None`).

---

## 5. Platform contract (eduLLM)

**Decision:** follow `.edullm/train_on_corpus.py` constraints rather than `src/examples/llm/train.py`. Guide: `edu-llm/platform` `guides/olmo-core.md` + `guides/the-platform.md`.

| Rule | How we satisfy it |
| --- | --- |
| Resolve data from env / form | `EDULLM_DATASET_*` → `edullm_data.read`; dtype/byte-order from manifest, never inferred |
| Checkpoints | `--save-folder "$EDULLM_CHECKPOINT_DIR"` on the command line; `max_checkpoints=None`; torn-step cleanup before resume. Platform sets that to `s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/<run_id>/checkpoints/` (e.g. team `pre-training`) |
| No broken evaluators | omit `lm_evaluator` / `downstream_evaluator` (they fail at trainer build in this image) |
| Explicit duration | `Duration.steps(TOTAL_STEPS)` |
| Multi-GPU | submitter puts `torch.distributed.run --nproc-per-node=N` in the command; script is one rank |
| Branch / image | push `edullm/**` so ECR builds; submit full SHA after scan (wait ~10 min after green for vuln scan) |

**Workload / compute (recommendation, not hardcoded):**

- Path check: `olmo-core-check` + `--dry-run` (CPU or 1×GPU; waive checkpoint if the profile requires one)
- GPU smoke: `olmo-core-train` on the **target** shape (e.g. 8×A100) with `--smoke` — same model / microbatch / flash_2, 20 steps, no mid-run checkpoints
- Real train: `olmo-core-train` + e.g. `gpu-8xa100` with `--nproc-per-node=8`

**24h ceiling:** ~12,715 steps may exceed routine 24h on slower shapes. Options: faster compute profile, runtime exception approval, or rely on Batch’s second attempt (same run id / checkpoint dir). A *new* submission is a new run id and does not auto-continue the previous prefix.

**Dataset registration:** `frontload-cl-10b-v1` is registered and `runs: true`
(`edullm data frontload-cl-10b-v1`). Name it with `--dataset frontload-cl-10b-v1` on
`edullm check` / `submit`. Training scripts still resolve shards via `edullm_data.read`
from the env the platform sets (`EDULLM_DATASET_*`).

---

## 6. What we did *not* put in sealed datasets

| Idea | Rejected because |
| --- | --- |
| Separate `curriculum/` orderings per arm | Freezes schedule into bytes; experiment says schedule is train-script |
| Two full 10B pretrain publishes (primer vs control) | Duplicate storage; confounds “same pools” |
| Publishing tokenizer again | `tokenizer/dolma2-bpe` already exists; corpus names it |
| Flat path mix via `NumpyFSLDatasetConfig` only | Cannot express contiguous primer block or phase anneal |

---

## 7. SFT path

**Decision:** ``train_sft.py`` resolves ``sft/frontload-cl-chat-sft`` conversations, tokenizes
on the fly (or reuses ``--tokens-dir``) with Dolma2 + the OLMo 2 ``<|user|>``/``<|assistant|>``
chat template and assistant-only label masks, then runs **one epoch** of OLMo2-370M SFT from
``--checkpoint`` (each PT arm’s final weights). Same mix and hparams for both arms.

**Why this shape:**

- Published ``v1`` is ``sft-conversations/v1`` (JSONL). OLMo-core needs ``.npy`` tokens + masks.
- Tokenization lives in ``sft_tokenize.py`` so we do not depend on open-instruct at train time.
- Prefer ``--dataset none`` plus explicit ``--dataset-id sft/frontload-cl-chat-sft``;
  naming ``frontload-cl-chat-sft-v1`` as the training dataset exits 69 (no tokenizer).
  Conversations are resolved via ``edullm_data.read`` either way.

**SFT hparams (defaults):** seq 4096, global batch 64×4096, lr ``8e-5``,
``LinearWithWarmup(warmup_fraction=0.03)``, wd 0, ``Duration.epochs(1)``, SkipStepAdamW.
Mix content is fixed by the published conversation dataset (no_robots + UltraChat + Numina 250k
+ OpenHermes 100k); do not grow the mix to chase a token target.

**Commands:** ``--dry-run`` (resolve + tokenize/report + print config), ``--tokenize-only``,
or full train with ``--checkpoint``.

---

## 8. Exit codes and operability

**Decision:** reuse the staged exit-code pattern from `train_on_corpus.py` (`Refusal` / `Stage` 64–73).

**Why:** platform CloudWatch logs are hard to read from the submitter side; a distinct exit code separates “role can’t read corpus” from “bad config” from “training crashed.” Exit 73 is bfloat16 on silicon that has none.

---

## 9. Open risks / follow-ups

1. **Image / push** — this `edullm/frontload-cl` branch must be pushed so ECR builds; wait for the vuln scan before submit. Any SHA without the Dockerfile flash-attn layer fails `--attn-backend flash_2` (use `--attn-backend torch` only as a temporary escape).
2. **Held-out NLL during PT** — experiment wants SFT-like domain NLL logged through pretrain; not wired yet (evaluators disabled on platform image). Val shards are already on the published corpus; need a custom callback over them.
3. **Downstream eval** — GSM8K / ARC / IFEval after shared SFT belong in `olmo-eval-full` submissions, not this train image.
4. **SFT tokenization cache** — first train/tokenize writes ``--tokens-dir``; multi-node
   needs that directory on shared storage (single-node 8×GPU ``/tmp`` is fine).
5. **24h ceiling** — full 10B may need the second attempt or a runtime exception on slower shapes; see §5.

Pool-edge repetition (anneal FineWiki / exact FineWeb budgets) is handled in `schedule.py` (`max_repetition_factor=1.05` + seq-aligned targets); no longer an open risk for the published corpus.

---

## 10. Submit cheat-sheet

```bash
# After image build/scan on this edullm/** branch
# (corpus registered as frontload-cl-10b-v1):

# Price / refuse without dispatching (reads .edullm/run.yaml by default)
edullm check --json --team pre-training --experiment frontload-cl \
  --dataset frontload-cl-10b-v1 --compute gpu-8xa100

# 1) GPU smoke on the target 8×A100 shape
edullm check --json --spec .edullm/run-smoke.yaml --team pre-training \
  --experiment frontload-cl --dataset frontload-cl-10b-v1 --compute gpu-8xa100
# then edullm submit --spec .edullm/run-smoke.yaml … (same flags)

# 2) Full arms (default run.yaml = primer; control via --spec)
edullm submit --team pre-training --experiment frontload-cl \
  --dataset frontload-cl-10b-v1 --compute gpu-8xa100
edullm submit --spec .edullm/run-control.yaml --team pre-training \
  --experiment frontload-cl --dataset frontload-cl-10b-v1 --compute gpu-8xa100
```

Equivalent command text (must include ``--param-dtype`` and ``$EDULLM_CHECKPOINT_DIR``):

```bash
# Path check (CPU / cheap)
bash -lc 'EDULLM_CHECKPOINT_CHECK=waived python .edullm/frontload_cl/train_pretrain.py \
  "$EDULLM_RUN_ID" --arm primer --dry-run --save-folder "$EDULLM_CHECKPOINT_DIR" \
  --param-dtype bfloat16'

# GPU smoke
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \
  .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \
  --arm primer --smoke --save-folder "$EDULLM_CHECKPOINT_DIR" \
  --param-dtype bfloat16'

# Full arms
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \
  .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \
  --arm primer --save-folder "$EDULLM_CHECKPOINT_DIR" \
  --param-dtype bfloat16'

bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \
  .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \
  --arm control --save-folder "$EDULLM_CHECKPOINT_DIR" \
  --param-dtype bfloat16'
```

CLI fields: `--experiment frontload-cl`, `--dataset frontload-cl-10b-v1`, `--compute gpu-8xa100`, `--team` one of your groups. Workload comes from the spec (`olmo-core-train`).