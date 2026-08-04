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

Run the tests the same way. The suite is torch-free apart from the end-to-end smoke runs, which are
marked `slow` and take a few minutes each because they train a real model in a subprocess:

```bash
python3 -m pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd -m 'not slow'
python3 -m pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd -m slow
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

The first run is 17 cells — five demands per row, four on 64M, plus a reasoning-only control on each.
Submit it as **three fan-outs, one per row**, so the rows run concurrently:

| `--row` | `fanout_size` | tokens | wall clock on 8×H100 | cost |
|---|---|---|---|---|
| `13M` | 6 | 40.2B | 1.5 h | $85 |
| `28M` | 6 | 78.0B | 5.0 h | $278 |
| `64M` | 5 | 82.3B | 9.6 h | $528 |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone
  src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID"
  --row 28M --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

with `fanout_size: 6` and `fanout_index_parameter: cell`. Each cell reads
`AWS_BATCH_JOB_ARRAY_INDEX` and gets its own checkpoint prefix. Sequential the grid is 16.2 h and about
$891; three jobs bring the wall clock to 9.6 h, and a failure in one row does not strand the others.

Those hours are `PRD.md` §10's FLOP estimate at 12/16/20% MFU per row, scaled to the current token
counts — **not measured**. Read the real MFU off the first cell's first 50 steps and re-budget before
submitting the 64M row, which is three-fifths of the cost.

The index maps to a cell by *position in the sorted directory*, and `ctrl` sorts before `d0p3`, so the
control is always index 0 and every demand cell sits one later than it would without it. Check
`ls configs/cells/count | grep <row>` against the `fanout_size` you submit; a stale size runs a
different cell under the name that was approved.

The entropy sweep is a fourth job, `--sweep entropy --row 28M`, `fanout_size: 6`, 3.4 h and about $187.
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
| `corpus/tasks.py` | the reasoning slices — Mano at L=10, and the related comparison task |
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

## The reasoning slices

Set `reasoning_tokens` on a cell and it trains on facts *and* reasoning, mixed through
`MixingInstanceSource` at **fixed absolute token counts** — not fixed ratios. Every cell that carries a
slice sees the same number of that slice's tokens, so a difference in reasoning score cannot be a
difference in reasoning exposure. A bare `CellSpec` defaults to `0`, which trains on facts alone; the
grid generators default to 1.0B.

The budget is **per slice**, not a total split between them. That distinction is load-bearing: the
control carries only `<mano>`, so splitting a fixed total would have handed it twice the `<mano>`
exposure of every cell it is the reference for, and its score would have beaten theirs for a reason
that has nothing to do with facts. So a count-axis cell's reasoning total is 2.0B and the control's is
1.0B, and that asymmetry is the correct consequence rather than a violated invariant.

Two slices, each prefixed with a domain token so the endpoints stay separable:

| | | |
|---|---|---|
| `<mano>` | mod-23 mental arithmetic, 10 operands, no chain of thought | unrelated to the facts |
| `<compare>` | which of two people was born earlier | needs two facts to answer |

`<mano>` is Physics 4.1's task at the length that paper found the transition at. It is the primary
endpoint because it shares no entity with the corpus: if it degrades as fact demand rises, the cost is
capacity, not fact access. `<compare>` is the contrast — it reads two biographies, so it should
degrade earlier and harder if the mechanism is retrieval instead.

Both are generated per example from a seed rather than drawn from a fixed set, so the slice holds
nothing to memorise and cannot itself compete for the capacity being measured. Both report a
**measured** degenerate floor, not an assumed one: **4.59%** for Mano against a 4.35% uniform baseline,
and **0.035%** for compare over the 25,000-entity probe subset. Measuring it changed the task — with a
free choice of operand, `× 0` is absorbing and the floor came out at 8.34%, above the 6.80% the paper
reports for the length this design rejected, so zero is excluded as a multiplicand.

`<compare>` is skipped where there is nothing to compare: on the entropy axis, whose attributes are six
positional four-word composites with no birth year, and on the control, which has no facts at all. Both
carry `<mano>` alone. `CellSpec.reasoning_slice_names` is the single source of truth for that choice and
`BuiltCorpus` asserts its own built set against it, so the token arithmetic and the data cannot disagree
about a cell.

### The reasoning-only control

One per row — `13m_ctrl`, `28m_ctrl`, `64m_ctrl` — stating `demand_bits_per_param: 0.0`. Zero facts:
no entity table, no renderer, no fact stream. It keeps the row's schema and vocabulary so the model is
architecturally identical to the cells it anchors, and it is the demand-0 endpoint of every crowding
curve — the one cell where a low reasoning score cannot be blamed on fact load.

Zero entities is *stated*, never solved for: `rho.solve` refuses a zero target and should keep refusing
it, since its linear path divides by bits-per-entity and its name-term path is non-monotone there. A
cell with demand 0 and no reasoning tokens is refused outright, being a run with no data at all.

## Still to build

The measurement half. `measure/bits.py` (the Allen-Zhu estimator over the value spans the renderer
already returns), `measure/reasoning.py` and its gates, and `measure/recall.py` as a post-hoc job.
Training writes the checkpoints those read; nothing scores them yet. `PRD.md` §8 and §12 have the
order.
