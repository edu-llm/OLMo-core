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
version. The worst case in the grid — `64m_d2p4`, 2.79M entities — takes about two minutes and 8.6 MB of
scratch.

`--build-only` goes one step further: it builds the model, the mixture and the trainer, prints the
settings the platform enforces, and exits without training. That is where the checkpoint schedule, the
mixture targets and the parallelism config are decided, so it is the check worth running against a real
cell before spending a queue slot:

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/train_cell.py preflight \
    --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml \
    --save-folder /tmp/preflight --build-only
```

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
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

One line, deliberately. The newlines an earlier draft used sit *inside* the single quotes, so `bash -lc`
reads them as statement separators and tries to run `src/scripts/...` as its own command.

`bash -lc` is required — the container runs the command directly, so without it `$EDULLM_RUN_ID`
arrives as fourteen literal characters. `--save-folder` must be on the line even though the program
defaults to it: the platform reads the command text to check that a run promising a checkpoint writes
one, and cannot see inside the process.

## The three jobs

The first run is 17 cells — five demands per row, four on 64M, plus a reasoning-only control on each.
Submit it as **three fan-outs, one per row**, so the rows run concurrently:

| `--row` | `fanout_size` | tokens | wall clock on 8×H100 | cost |
|---|---|---|---|---|
| `13M` | 6 | 34.7B | 1.3 h | $73 |
| `28M` | 6 | 71.5B | 4.6 h | $252 |
| `64M` | 5 | 76.6B | 8.8 h | $487 |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --row 28M --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

with `fanout_size: 6` and `fanout_index_parameter: cell`. Each cell reads
`AWS_BATCH_JOB_ARRAY_INDEX` and gets its own checkpoint prefix. Sequential the grid is 14.7 h and about
$812; three jobs bring the wall clock to 8.8 h, and a failure in one row does not strand the others.

Those hours are `PRD.md` §10's FLOP estimate at 12/16/20% MFU per row, scaled to the current token
counts — **not measured**. Read the real MFU off the first cell's first 50 steps and re-budget before
submitting the 64M row, which is three-fifths of the cost.

The index maps to a cell by *position in the sorted directory*, and `ctrl` sorts before `d0p3`, so the
control is always index 0. Verify the mapping before submitting — a `fanout_size` from an older
directory runs a different cell under the name that was approved:

```
[0] 13m_ctrl  [1] 13m_d0p3  [2] 13m_d0p6  [3] 13m_d1p2  [4] 13m_d2p4  [5] 13m_d4p8
[0] 28m_ctrl  [1] 28m_d0p3  [2] 28m_d0p6  [3] 28m_d1p2  [4] 28m_d2p4  [5] 28m_d4p8
[0] 64m_ctrl  [1] 64m_d0p3  [2] 64m_d0p6  [3] 64m_d1p2  [4] 64m_d2p4
```

The entropy sweep is a fourth job, `--sweep entropy --row 28M`, `fanout_size: 6`, 36.0B tokens,
2.3 h and about $127.
It is the identified axis (`PRD.md` §3.1) and the cheapest thing here. Its indices sort alphabetically,
so `[0] b0 [1] b16 [2] b24 [3] b32 [4] b4 [5] b8` — every cell is self-labelled, so the order is
harmless, but it is not numeric.

### Pre-flight, all verified on this commit

| | |
|---|---|
| worst-case startup | 120 s and 8.6 MB of scratch for `64m_d2p4`, the 2.86M-entity cell |
| `max_checkpoints` | `None`, so no prune deletes a key the workload role may not delete |
| `ephemeral_save_interval` | `None` |
| evaluators | none attached |
| `max_duration` | explicit, in steps |
| W&B | attached, enabled only when `EDULLM_WANDB_PROJECT` is set |
| every checkpoint | carries the cell config and the schema, vocabulary, stream and task fingerprints |

That last row is what makes it safe to train before the scoring half exists. The corpus is generated
rather than stored, so a checkpoint is only scoreable if the run recorded how to rebuild it — the cell
config replays the vocabulary, entity table and reasoning items exactly, and the fingerprints let a
scorer prove it rebuilt the right ones rather than assume. Nothing about scoring needs to be decided
before these jobs run.



## Measured throughput, and the A10G route

`13m_ctrl` ran end to end on **4×A10G in 19 min 15 s** — 1.0B tokens, 3,814 steps, ten checkpoints,
`865,801 tokens/s` including setup and every save. That is the number the table below is built from, not
an estimate.

Two notes on reading the run's own metrics. OLMo-core's speed monitor reports `MFU 7.3%` against a
declared device peak of 312 TFLOP/s — that is the A100 figure; A10G's bf16 peak is 125, so the real MFU
is about **18%**, healthy for a 13M model. And that cell is 100% `<mano>`, the *slowest* generator here
(0.59M tok/s single-threaded against the bio renderer's 10.1M), yet data loading was 0.06% of step time
— so the fact cells, which use the fast renderer, are not data-bound either.

Hours scale with non-embedding parameters. Eight devices are taken as 1.9× four, which is **not**
measured — g5 has no NVLink, so treat the 8×A10G column as the optimistic end.

| cell | tokens | 4×A10G | 8×A10G |
|---|---|---|---|
| `13m_ctrl` … `13m_d4p8` | 1.0–15.9B | 0.3–5.1 h | 0.2–2.7 h |
| `28m_ctrl` … `28m_d1p2` | 1.0–9.3B | 0.7–6.7 h | 0.4–3.5 h |
| `28m_d2p4` | 17.8B | 12.9 h | 6.8 h |
| `28m_d4p8` | 35.3B | **25.4 h — over the cap** | 13.4 h |

**`28m_d4p8` must use eight devices**; four would breach the platform's hard 24 h per-cell ceiling.
Everything else goes on 4×A10G, which is the *cheaper* way to buy the same eight GPUs —
$1.418/GPU/h against $2.036 — and gives two concurrent slots instead of one.

**64M cannot run on A10G at all.** `64m_d2p4` needs 33.9 h even on eight devices, and `64m_d1p2` needs
17.1 h with `d2p4` impossible behind it. That row belongs on the P pool.

### 13M + 28M on A10G: ~18.6 h, ~$430, four submissions, all `ROUTINE`

| # | `compute_profile` | `nproc` | selector | `fanout_size` | `max_runtime_hours` | ceiling |
|---|---|---|---|---|---|---|
| 1 | `gpu-4xa10g` | 4 | `--row 13M` | 6 | 8 | $272 |
| 2 | `gpu-4xa10g` | 4 | `--row 28M` | **4** | 10 | $227 |
| 3 | `gpu-4xa10g` | 4 | `--cell …/count/28m_d2p4.yaml` | — | 18 | $102 |
| 4 | `gpu-8xa10g` | 8 | `--cell …/count/28m_d4p8.yaml` | — | 20 | $326 |

Submissions 1–3 put eleven cells into the 4×A10G queue and Batch packs them across its two slots, so the
makespan is `max(total ÷ 2, longest cell)` = 18.6 h. Submission 4 finishes in 13.4 h alongside them.

`--row 28M` with `fanout_size: 4` is deliberate: filenames sort by ascending demand, so a prefix fan-out
is exactly the four shortest cells, and the two long ones get their own submissions with runtime limits
that fit them rather than one limit stretched to cover the longest.

Every ceiling is under the $500 routine bound, so a team lead can release all four — no admin, and the G
quota is independent of the P quota that H100 and A100 draw on.

## Before you submit: what an expert review changed

An independent review returned a stop-ship verdict on the commit before this one. Four defects would
have invalidated the runs, and all four are fixed with regression tests. Two are worth knowing about
because they change what you should expect to see:

- **Eight GPUs used to train eight independent models.** No `dp_config` meant FSDP was never applied,
  gradients were never reduced, and each rank saw an eighth of the corpus — ~25 exposures instead of
  200. Now FSDP with bf16 parameters, fp32 reductions, clipping at 1.0 and compilation on GPU. Runtime
  and cost estimates in this file describe *that* configuration; the older ones described fp32 without
  compilation, so treat both as unmeasured until the 13M row reports its MFU.
- **The entropy sweep varied the model.** Its vocabulary grew with entropy — 1,920 padded tokens at b=0
  against 8,064 at b=32, i.e. 8.1% more parameters and a 4.2× wider softmax on the high-entropy arm.
  Every cell now shares one union vocabulary. If you are running job 4, use a commit at or after this
  one; earlier entropy results are not comparable across cells.

The other two were latent: the reasoning eval set was 100% leaked from training by its keying, and
`init_seed` was never set so replicates would have shared one initialisation. Neither had run yet.

`PRD.md` §16.5 has the full account, including four places I judged differently from the review and the
list of accepted-but-unbuilt work.

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
that has nothing to do with facts.

The two slices carry deliberately *different* budgets — `reasoning_tokens` 1.0B for `<mano>`,
`related_reasoning_tokens` 50M for `<compare>`. The related one is sized on **per-entity coverage**, not
on parity: it names two of the fixed 25,000 probe entities per item, so at 1.0B each entity's birth-year
rank would be supervised 4,211 times against the 200 exposures the fact slice gets. At that rate the
slice teaches the ranks outright, "needs two facts" stops being true, and a decline would say nothing
about fact access. 50M gives 211 mentions per entity, level with the facts.

Two slices, each prefixed with a domain token so the endpoints stay separable:

| | | |
|---|---|---|
| `<mano>` | mod-23 mental arithmetic, 10 operands, no chain of thought | unrelated to the facts |
| `<compare>` | the earlier of two people's birth years | needs two facts to answer |

`<mano>` is Physics 4.1's task at the length that paper found the transition at. It is the primary
endpoint because it shares no entity with the corpus: if it degrades as fact demand rises, the cost is
capacity, not fact access. `<compare>` is the contrast — it reads two biographies, so it should
degrade earlier and harder if the mechanism is retrieval instead.

Both are generated per example from a seed rather than drawn from a fixed set, so neither slice holds an
item worth memorising. Both report a **measured** degenerate floor, searched over two policy families —
the best constant answer *and* the best copy of a fixed span of the prompt: **4.64%** for Mano against a
4.35% uniform baseline, **0.70%** for compare.

Measuring the floors changed both tasks, which is the argument for measuring them.

- Mano: with a free choice of operand, `× 0` is absorbing and the floor came out at **8.34%** — above
  the 6.80% the paper reports for the length this design *rejected*. Zero is now excluded as a
  multiplicand.
- `<compare>` originally answered with the earlier person's **name**, which is a span of its own prompt,
  so "always name the first person" scored **50.2%** while the best constant name managed 0.02%. A binary
  endpoint with a 50% floor has half its range available to a policy that reads no facts, and any score
  below 50% would be under its own floor. It now answers with the earlier **birth year**, a word that
  never appears in the prompt, so no copy policy can reach it.

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

## Scoring a finished run

Training writes checkpoints; `score_run.py` reads them and produces one CSV you can analyse.

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/score_run.py \
    --prefix s3://.../runs/$RUN_ID/checkpoints \
    --out    s3://.../runs/$RUN_ID/scores.csv
```

`--prefix` takes either one cell's checkpoint directory or a fan-out's parent — it finds the `cell-N/`
segments itself. Add `--steps 0,1220,3814` to score a subset.

**It is a separate single-process job, not a callback.** Recall needs free-running generation, which
`TransformerGenerationModule` cannot provide from inside a trainer without re-parallelising the model
(`PRD.md` §8.2). And `load_model_and_optim_state` reshards a checkpoint saved on four ranks into one
unsharded model with no process group, so there is nothing to distribute. Scoring the whole first run is
minutes of forward passes on one small GPU, dominated by pulling shards from S3.

### What each checkpoint yields

| | |
|---|---|
| **reasoning** | per endpoint: three counts, accuracy, answer-token CE in bits, and the **measured** floor |
| **achieved bits** | Allen-Zhu's estimator over the value spans the renderer returns, with the per-entity distribution |
| **recall** | generation *and* recognition per attribute, each with its own chance level |

Both endpoints render a **single-token answer at a known position**, so grading is a teacher-forced argmax
— identical to what a greedy decode would produce, with no continuation to truncate and no string to
parse. That is what removes the failure mode `PRD.md` §1 catalogues four times: an eval whose score is
bounded by its parser rather than by the model. `n_unparseable` is reported anyway, because §8.6's G7
requires it under 5% and a future multi-token endpoint could break the property.

### The layering

| | |
|---|---|
| `measure/spans.py` | which loss position pays for which token — one rule, one place |
| `measure/endpoints.py` | `EndpointResult`: the shape every endpoint reports in |
| `measure/checkpoint.py` | find a checkpoint, rebuild its corpus, verify fingerprints, load weights |
| `measure/reasoning.py` | the dependent variable |
| `measure/bits.py` | achieved bits against demanded |
| `measure/recall.py` | generation and recognition |
| `measure/gates.py` | `PRD.md` §8.6's G1–G8 |
| `measure/collect.py` | one tidy row per (cell, replicate, step, endpoint) |
| `analysis/trend.py` | per-seed slopes, TOST equivalence, non-inferiority, MDE |

`spans.py` exists because an off-by-one there is invisible: `ce_loss[t]` scores `input_ids[t+1]`, so the
cost of token *p* is `ce_loss[p-1]`. Get it backwards and a bit count charges a value token's cost to the
literal before it, and an endpoint grades the token before the answer. Both produce plausible numbers. It
is checked against a manually computed cross-entropy from a real model.

Every scorer takes a `forward` callable rather than a model, so the logic is testable on a stub that
returns chosen logits — including a model that answers one position early, which is the only way to catch
the off-by-one from the scorer's side.

### Two honesty properties worth knowing before you read a number

**Achieved bits are an upper bound while intra-document masking is off.** A packed biography can attend to
its neighbour, so some of what looks stored was read from context. Every row carries
`bits_is_upper_bound`, and `check_against_capacity()` refuses a figure above Physics 3.3's ~2 bits/param
ceiling — such a figure is a broken measurement, not a striking result.

**The rebuild is verified, not assumed.** The corpus is generated rather than stored, so scoring replays
the cell recorded beside the weights and checks the schema and vocabulary fingerprints against what
training wrote. This has already caught a real case: a checkpoint from before the entropy axis moved to a
union vocabulary rebuilds to a different schema today, and scoring it would have produced entirely
reasonable-looking numbers about a corpus the model never saw.

### Analysis

`analysis/trend.py` takes the **per-seed slope** as the inferential unit, not the individual cell.
`PRD.md` §8.5 asked for one regression over all cells; that treats correlated observations as independent,
and on the planned 3×6 design the naive cell-level standard error comes out **2.83× smaller** — its 90%
interval declares equivalence where the blocked interval cannot, on the same data. `pooled_trend` is
therefore per ladder row, and both `tost` (genuine ±2pp equivalence) and `non_inferiority` (the one-sided
rule, correctly named) are provided so a null can say which one it is.

## Still to build

The gates' *evidence*, and the endpoints that need building to satisfy them. `measure/gates.py`
implements G1–G8, but several need runs that do not exist yet — a label-permuted control, a
premise-ablated probe, a reasoning-token dilution ladder — and a gate whose evidence is missing returns
**false**, never a silent pass. `PRD.md` §8.3 also still wants Brevo1 and Reasoning Core: until an
in-context endpoint lands, a `<mano>` decline is ambiguous between "reasoning crowded out" and "mod-23
tables crowded out".

Intra-document masking (`PRD.md` §7.3) and the 504-token sequence length that would stop chunking cutting
~3% of reasoning items are both specified and unbuilt.
