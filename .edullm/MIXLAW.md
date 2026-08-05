# MixLaw 370M validation

This integration runs standard OLMo-core OLMo2-370M pre-training four times.
The only experimental input that changes is the selected seven-domain weight
vector in `.edullm/mixlaw_recipe.json`.

The recipe records the exact values and provenance copied from
`edu-llm/edullm@b435cbe9c352399fc4ab54b310f36d28f6c9746f`. No eduLLM training
code is vendored here. Construction divides each source vector by its own sum
because OLMo-core requires unit-sum target ratios; the checked-in source
numbers remain unchanged.

## Fixed training contract

- model: `TransformerConfig.olmo2_370M`
- sequence length: 2,048
- global batch: 4,194,304 tokens
- rank microbatch: 32,768 tokens
- optimizer: SkipStepAdamW, LR `4e-4`, betas `(0.9, 0.95)`, weight decay
  `0.1` except `embeddings.weight` at `0`
- HSDP: bf16 parameters, fp32 reductions
- z-loss `1e-5`, max grad norm `1`, compiled model
- cosine schedule: 24 warmup steps, `alpha_f=0.1`
- seed: 12,536
- production: floor of 10B/global-batch = 2,384 steps

`NumpyFSLDatasetConfig.from_src_mix()` receives a standard
`SourceMixtureDatasetConfig`. Each domain is resolved directly from validated
`pretrain/olmo-127b/v1` with:

```python
dataset_paths(..., split="train", labels={"source": domain})
```

The entrypoint refuses anything except explicit uint32, little-endian,
headerless shards. It derives one conservative repetition bound per domain
from the maximum demand across all four production recipes and the domain's
published token count. Those bounds and all other configuration fields are
identical across arms; only `SourceMixtureConfig.target_ratio` changes.
OLMo-core supports these `s3://` inputs through its standard ranged-read path,
so no staging loop or custom data stream is needed.

`EDULLM_CHECKPOINT_DIR` is the trainer `save_folder`. The entrypoint calls
`trainer.maybe_load_checkpoint()` before `fit()`, preserving the workload's
two-attempt resume behavior. The standard checkpointer writes permanent
checkpoints at step 0, every 125 steps, and the final step.

W&B uses the required `EDULLM_WANDB_PROJECT` (normally `mixlaw`),
`WANDB_RUN_GROUP` from the platform, and `WANDB_NAME=<run-id>-<mixture>`.
At each permanent checkpoint, training synchronously runs the complete
20-task OLMES BPB suite and uploads every metric and result JSON to W&B.
Intermediate checkpoints remain in the trainer save folder; only the final
checkpoint is uploaded as a W&B model artifact.

## Launch shape

The public CLI requires one `--arm-index` from 0 through 3. It execs
`torch.distributed.run --standalone --nproc-per-node=8` and the worker refuses
any other world size.

| index | mixture |
|---:|---|
| 0 | `olmo-mix-1124` |
| 1 | `mix01` |
| 2 | `ML-pilot_caps` |
| 3 | `LGB-min1pct` |

Benchmark one arm at 100 global batches before production. Use
`.edullm/fixtures/mixlaw-benchmark-olmo-mix-1124-submission.json` for arm 0
(`olmo-mix-1124`), or the generic benchmark fixture for arm 1 (`mix01`):

```bash
python /opt/olmo-core/.edullm/mixlaw_entrypoint.py \
  --arm-index 0 --length-tokens 419430400
```

`419430400` is exactly 100 global batches. `--length-tokens` is benchmark-only
and must be a positive multiple of the global batch. Production omits it.

The platform currently permits one `p4d.24xlarge`, so submit the four
production fixtures sequentially, waiting for each run to leave the queue
before submitting the next. Do not use fan-out. Both fixtures deliberately use
the standard checkpoint contract: the benchmark uses one attempt, while
production uses two and resumes directly from `EDULLM_CHECKPOINT_DIR`.

Production bounds the run at 12 hours. Omitting the field inherits the
workload profile's 24 hour ceiling, and with two attempts that is what an
approver is shown as the worst case: $1,053.96 on a shape charged at $21.9576
an hour. The 12 hours is the bound the curriculum branch already carries for
the same run, which is the right comparison rather than a guess, because the
two are the same training contract to the step: OLMo2-370M, 2,048 sequence,
4,194,304 token global batch, 2,384 steps, permanent checkpoints at step 0 and
on the 125 grid, and the full 20-task OLMES BPB suite run synchronously at
every one of them. It is still a ceiling rather than an estimate, and the
first production arm to finish is the measurement that should replace it.

## Local validation

The fixtures compile credential-free against the untouched platform
`origin/main` checkout:

```powershell
py -3 C:\alpha_ai\platform\tools\compile_submission.py `
  --inputs .edullm\fixtures\mixlaw-benchmark-submission.json `
  --config-dir C:\alpha_ai\platform\config `
  --published-images .edullm\fixtures\mixlaw-published-image.json `
  --submitter nzhao721 `
  --repository-url https://github.com/edu-llm/OLMo-core `
  --run-id run_019fa439-203e-70c7-bf8a-9ce33bc71f20 `
  --output "$env:TEMP\mixlaw-benchmark-compiled.json" `
  --summary "$env:TEMP\mixlaw-benchmark-summary.md"
```

Repeat with `mixlaw-submission.json` for the production arm. Compilation only
writes local temporary files; it does not submit or contact AWS.
