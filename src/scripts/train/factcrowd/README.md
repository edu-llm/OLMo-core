# factcrowd — how to run it

`PRD.md` is why. This is how.

Does storing facts consume capacity that would otherwise serve reasoning? We train small models on
synthetic biographies, sweep the fact bits they demand per parameter, and measure reasoning on the
same checkpoints.

## Check it locally first

Nothing here needs a GPU, and `--dry-run` catches every config and arithmetic error:

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/train_cell.py local \
    --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml \
    --dry-run
```

It resolves the cell, generates the entity table, builds the vocabulary and the renderer, builds the
token-offset index, and prints the plan with a sample biography. Add `--json` for a machine-readable
version.

Run the tests the same way — the suite is torch-free apart from two skipped cases:

```bash
python3 -m pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd
```

## Submit a cell

One cell is one run. Push to a branch named `edullm/…` (nothing else gets an image), wait 8–11
minutes for the build and its scan, then [submit a
run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml):

| Field | Value |
|---|---|
| `repository` | `OLMo-core` |
| `commit_sha` | full SHA of a built commit |
| `workload_profile` | `olmo-core-check` for a smoke, `olmo-core-train` for a cell |
| `compute_profile` | `gpu-1xa10g` for a smoke, `gpu-8xh100` or `gpu-8xa100` for a cell |
| `team` | `memory-split` |
| `dataset_release` | **`none`** — the corpus is generated in-process |
| `wandb_project` | yours |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone
  src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID"
  --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml
  --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

`bash -lc` is required — the container runs the command directly, so without it `$EDULLM_RUN_ID`
arrives as fourteen literal characters. `--save-folder` must be on the line even though the program
defaults to it: the platform reads the command text to check that a run promising a checkpoint writes
one, and cannot see inside the process.

## The three jobs

The first run is 14 cells. Submit it as **three fan-outs, one per row**, so the rows run concurrently:

| `--row` | `fanout_size` | wall clock on 8×H100 |
|---|---|---|
| `13M` | 5 | 0.5 h |
| `28M` | 5 | 2.3 h |
| `64M` | 4 | 5.4 h |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone
  src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID"
  --row 28M --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

with `fanout_size: 5` and `fanout_index_parameter: cell`. Each cell reads
`AWS_BATCH_JOB_ARRAY_INDEX` and gets its own checkpoint prefix. Sequential the grid is 8.1 h and about
$445; three jobs bring the wall clock to 5.4 h, and a failure in one row does not strand the others.

The entropy sweep is a fourth job, `--sweep entropy --row 28M`, `fanout_size: 6`, 1.8 h and about $98.
It is the identified axis (`PRD.md` §3.1) and the cheapest thing here.

## Configs

One YAML per cell under `configs/cells/{count,entropy}/`. Two directories because a fan-out maps an
array index to a cell by position, so its size has to be what `ls` says.

A cell states **either** its demand or its entity count, never both — the other is derived, so a cell's
label and its corpus cannot disagree. Regenerate them rather than hand-editing the set:

```python
from factcrowd import cells
cells.write_cells(cells.first_run_cells(), "src/scripts/train/factcrowd/configs/cells/count")
cells.write_cells(cells.entropy_sweep_cells("28M"), "src/scripts/train/factcrowd/configs/cells/entropy")
```

Editing one field of one cell by hand is fine; a test resolves every committed config, so a cell that
does not add up fails locally rather than on a GPU.

## What is here, and what is OLMo-core's

Ours is the corpus and the arithmetic that places a cell on the demand axis:

| | |
|---|---|
| `ladder/rho.py` | demand ↔ entity count, both parameter bases, the name term |
| `ladder/sizes.py` | the four widths at depth 12, with the built parameter count asserted |
| `corpus/entities.py` | the entity table; exact bits, unique names, a prefix property across cells |
| `corpus/values.py` | attribute values as words; the bioS and entropy schemas |
| `corpus/vocab.py` | the closed word-level vocabulary |
| `corpus/render.py` | 32 templates over four length bands; exact value spans |
| `corpus/stream.py` | stream order, the token-offset index, token assembly |
| `corpus/source.py` | a 90-line `TokenSource` adapter |
| `cells.py` | one cell, and everything derivable from it |
| `train_cell.py` | assembles OLMo-core's configs and calls `fit()` |

Everything else is OLMo-core's and is **not** reimplemented: `TransformerConfig.llama_like` for the
model, `AdamWConfig` and `WSD` for optimisation, `ConcatAndChunkInstanceSource` for packing,
`ComposableDataLoaderConfig` for shuffling and batching, `ListCheckpointerCallback` for the log-spaced
snapshots, and the trainer for checkpointing and resumption.

The split between `corpus/stream.py` and `corpus/source.py` is deliberate. All the logic is on the
torch-free side and tested; the adapter that needs the full install is four methods. Earlier in this
branch a module put its only real logic behind a torch import, its tests skipped, and a call that
raised `TypeError` for every input passed both review and type-checking.

## Still to build

The measurement half. `measure/bits.py` (the Allen-Zhu estimator over the value spans the renderer
already returns), `measure/reasoning.py` and its gates, `measure/recall.py` as a post-hoc job, and the
reasoning slices themselves — Mano at L=10 and Brevo1, mixed at fixed absolute token counts through
`MixingInstanceSource`. Cells currently carry `reasoning_tokens: 0`, so a run today trains on facts
alone. `PRD.md` §8 and §12 have the order.
