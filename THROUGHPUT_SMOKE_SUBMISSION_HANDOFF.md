# 24-cell mixer comparison submission

Run from:

```bash
cd /home/vs/AlphaAI/eduLLM/OLMo-core-flash-pd
```

## Run matrix

| Array indices | Arm | Data seeds |
| --- | --- | --- |
| 0–2 | `mamba-b3` | 210007, 220014, 230021 |
| 3–5 | `xlstm` | 210007, 220014, 230021 |
| 6–8 | `mamba3-siso-pd` | 210007, 220014, 230021 |
| 9–11 | `native-pd` | 210007, 220014, 230021 |
| 12–14 | `gdn` | 210007, 220014, 230021 |
| 15–17 | `kda` | 210007, 220014, 230021 |
| 18–20 | `kda-hh-r2` | 210007, 220014, 230021 |
| 21–23 | `kda-gconv` | 210007, 220014, 230021 |

Spec: `.edullm/run-comparison.yaml`

Dataset: `reservoir-dolma2-v1`

Compute: `gpu-8xa100`

W&B project: `memory-split`

Runtime bound: `--hours 3`. Batch applies the attempt timeout to each child of an array
job, so this is three hours per cell and not three across the twenty-four. A cell that
outruns it is killed on its own and its siblings keep running.

## 1. Push and verify the image

```bash
git status --short
git rev-parse HEAD
git push origin edullm/mamba-comparison
git fetch origin \
  "refs/heads/edullm/mamba-comparison:refs/remotes/origin/edullm/mamba-comparison"
git branch -r --contains "$(git rev-parse HEAD)"
```

```bash
gh run list \
  --workflow "Build eduLLM research image" \
  --branch edullm/mamba-comparison \
  --limit 5
```

Continue only after `Build eduLLM research image` succeeds for the exact output
of `git rev-parse HEAD`.

## 2. Check the 24-cell run

```bash
edullm check --json \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --compute gpu-8xa100 \
  --hours 3 \
  --attempts 1 \
  --fanout-size 24 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Require an empty `refusals` list. Read `cost` and `approval_class` out of the same
document; `--hours` moves both.

## 3. Submit the 24-cell run

```bash
edullm submit \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --compute gpu-8xa100 \
  --hours 3 \
  --attempts 1 \
  --fanout-size 24 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Save the returned run ID.

## 4. Monitor

Free GitHub-side status:

```bash
edullm status --json <run-id>
```

Authoritative AWS Batch status:

```bash
edullm status <run-id>
```

Recent program output:

```bash
edullm logs <run-id>
```

## 5. Results to collect

Each cell writes the whole record twice: as JSON on stdout, and into the summary of its own
W&B run, grouped under `mamba-comparison`. Read it off W&B — the field names are the same in
both copies.

Identify a cell by its W&B run **id**, which is `<run-id>-cell-<index>`. The display name is
not it: that comes from `EDULLM_RUN_ID`, which the fan-out does not vary, so all 24 runs show
the same name. `arm` and both seeds are in each run's summary, and the arm is also in its
config, so either one separates them without counting indices.

- training and validation loss;
- `throughput_tok_s_steady`;
- `throughput_tok_s_steady_per_device`;
- `step_time_s_p50`;
- `step_time_s_p90`;
- peak allocated and reserved memory;
- decode latency;
- recurrent-state bytes per sequence;
- arm, data seed, init seed, world size, steps, parameter count, and image SHA.

## Cancel

```bash
edullm cancel <run-id> --reason "<reason>"
```
