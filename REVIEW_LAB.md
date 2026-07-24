# Review Timing and Delayed Factual Retention

This directory contains a paired continual-training experiment that asks two questions:

1. Does reviewing old fictional facts improve their retention after continued learning?
2. Under the same review budget, does uniform or expanding-interval review work better?

The final experiment compares `no_review`, `uniform`, and `expanding` using three paired seeds. It
is a LoRA pilot on an OLMo ~370M checkpoint, not a full pretraining run.

## Experiment design

- **Model:** `allenai/DataDecide-dolma1_7-300M`
- **Pinned revision:** `4b1b42ff7c5224a077c4f5624824dd7c8bbe98d7`
- **Adaptation:** LoRA, rank 16
- **Seeds:** 17, 23, and 42
- **Old knowledge:** 20 fictional events
- **New knowledge:** 40 separate fictional events
- **Stage one:** 60 updates learning old facts
- **Stage two:** 180 updates learning new facts, with scheduled old-fact review
- **Delayed test:** 180 additional new-fact updates with no old-fact review
- **Review budget:** 12 review updates for both uniform and expanding
- **Primary metric:** final held-out old-fact answer-token loss after the buffer; lower is better

Every condition for a seed starts from the same stage-one checkpoint and uses the same old and new
facts. Uniform and expanding receive the same number of review updates and share the same first and
last review steps. Only the placement of the intervening reviews changes.

The no-review buffer is important because it prevents the experiment from ending immediately after
a review. During the buffer, all arms receive new-fact training but no old-fact review.

## Dataset

The experiment uses the public
[`jwkirchenbauer/fictionalqa`](https://huggingface.co/datasets/jwkirchenbauer/fictionalqa) dataset:

- configuration: `fict_qa`
- revision: `131cb74fdc3e601b5e896ed768ad9852ea35a8f9`
- training input: the fictional statement in the dataset
- evaluation: the separately worded question and natural answer

Duplicate question clusters are reduced to their canonical roots. Entire fictional events are
assigned to either the old-fact or new-fact split so information from one invented story cannot
leak across the boundary. The generated JSONL file is reproducible from the pinned source and is
not meant to be committed.

## Code map

- `src/olmo_core/review_lab/`: dataset construction, schedules, training, evaluation, and reports
- `src/test/review_lab/`: unit tests
- `configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml`: final experiment
- `configs/review_lab/olmo370m_fictionalqa_smoke.yaml`: short GPU smoke test

Other configurations in `configs/review_lab/` are exploratory pilots and are not the source of the
headline result below.

## Results

Values are means across three paired seeds. Lower loss is better.

| Condition | Reviews | Final old-fact loss | Final new-fact loss | Buffer old-loss increase |
|---|---:|---:|---:|---:|
| `no_review` | 0 | 3.981 | 2.993 | +0.151 |
| `uniform` | 12 | 3.809 | 3.018 | +0.161 |
| `expanding` | 12 | 3.808 | 3.015 | +0.158 |

Uniform and expanding review reduced final old-fact loss by approximately 4.3% relative to no
review. Their final old-fact losses differed by less than 0.001, so this pilot provides no evidence
that one spacing pattern is better.

All three conditions experienced a similar old-loss increase during the 180-step buffer. Review
therefore improved the final retained level, but it did not measurably slow the subsequent
forgetting rate. Review also caused a small increase in new-fact loss, showing the expected
retention–plasticity trade-off.

These findings support repeated review in this setting. They do not establish a general pretraining
efficiency gain, reasoning retention, or superiority of expanding review. Larger samples and
full-parameter training would be needed for those claims.

## Reproduce the experiment

Use Python 3.10–3.12 on a CUDA machine:

```bash
git clone https://github.com/edu-llm/OLMo-core.git
cd OLMo-core
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[review-lab,dev]'
python -m pip install --no-deps 'ai2-olmo==0.6.0'
python -m pip install datasets omegaconf boto3 google-cloud-storage
python -m pip install --force-reinstall 'transformers==4.57.6'
python -m pip install 'numpy<2' 'fsspec<=2026.4.0'
```

The Transformers pin is required by the legacy `hf_olmo` model wrapper used for the public
DataDecide checkpoint.

Run the tests and inspect the resolved design:

```bash
python -m pytest --confcutdir=src/test/review_lab src/test/review_lab -q
olmo-review \
  --config configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml \
  build-data
olmo-review \
  --config configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml \
  inspect
olmo-review \
  --config configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml \
  doctor
```

Run all three conditions:

```bash
olmo-review \
  --config configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml \
  sweep
```

Completed condition runs are skipped automatically. Use `--force` only when intentionally
replacing an existing run.

Rebuild the report without retraining:

```bash
olmo-review \
  --config configs/review_lab/olmo370m_fictionalqa_3seeds_buffer_t1000.yaml \
  summarize
```

Outputs are written under `runs/review_lab/olmo370m-fictionalqa-3seeds-buffer180/`:

- `REPORT.md`: summary table and interpretation
- `results_by_seed.csv`: paired seed-level results
- `results_aggregate.csv`: mean and standard deviation
- `retention_vs_adaptation.png`: retention–plasticity comparison
- per-condition `metrics.jsonl`: complete training and evaluation traces

The `runs/` directory contains generated outputs and is intentionally ignored by Git.
