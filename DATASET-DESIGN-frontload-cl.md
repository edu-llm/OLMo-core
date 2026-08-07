# Dataset design: frontload-cl (early behavior primer)

Experiment companion: `EXPERIMENT-early-behavior-primer.md`  
Train implementation: `.edullm/frontload_cl/DESIGN.md`  
edullm-data client: **v0.6.3** (latest at design time)  
Profiles verified: `pretrain-tokens/v1`, `sft-conversations/v1`, `tokenizer/v1`  
`publish()` has **no** `labels` param — slices live in the path only.

Shared seed for all subsamples: **42069666**  
Tokenizer dependency (already published): **`tokenizer/dolma2-bpe`** — do not republish unless you need a private fork.

Schedules (primer block vs flat): **in the train script**, not separate `curriculum/` datasets.

## Status (2026-08-07)

| Artifact | State |
| --- | --- |
| Local PT shards under `data/frontload-cl/tokens/` | Complete (~10.1B train tokens across sources + thin carved vals). FineWeb main/anneal finished. |
| Local SFT JSONL under `data/frontload-cl/conversations/` | Present (train ~558k / val ~28k rows) |
| `pretrain/frontload-cl-10b` on `edullm-data` | **Published `v1`** (`_VALIDATED.json`, catalog, README; ~37.7 GiB, 53 objects; train 10 085 769 122 / val 31 000 000 tokens) |
| `sft/frontload-cl-chat-sft` on `edullm-data` | **Published `v1`** (`_VALIDATED.json` present) |
| Platform `config/datasets.yaml` entry | **Registered** as `frontload-cl-10b-v1` (`runs: true`). SFT release is usable via `--dataset none` + `--dataset-id` (naming it as train dataset exits 69). |

---

# Dataset design: pretrain/frontload-cl-10b

purpose: 10B-token HQ+SFT-like Dolma2 pretrain mix for OLMo2-370M frontload-cl arms, to decide whether early SFT-like timing beats flat mixing on GSM8K/ARC/IFEval after shared SFT

family:   pretrain  
profile:  pretrain-tokens/v1   [verified in registry: yes]  
name:     frontload-cl-10b     [validate_dataset_id: PASS]

## Irreversible decisions

slice path: `tokens/<source>/<split>-NNNNN.u32le.bin` — **one level** under `tokens/` (matches `pretrain/regmix-10b`)

| source folder | content | train tokens | val tokens (carve **before** tokenize) |
|---|---|---|---|
| `fineweb-edu-main` | FineWeb-Edu, exclude anneal docs | 8.36B | part of 20M FineWeb-Edu val |
| `fineweb-edu-anneal` | FineWeb-Edu `int_score >= 4` | 950M | (optional thin val; or share HQ val) |
| `finewiki` | FineWiki subsample | 490M | 5M |
| `cosmopedia-v2` | SmolLM Cosmopedia-v2 `text` | 80M | part of 5M SFT-like val |
| `finemath-4plus` | FineMath-4plus `text` | 60M | part of 5M SFT-like val |
| `openhermes-pt` | OpenHermes plain-text; **disjoint** from SFT 100k draw | 30M | part of 5M SFT-like val |
| `natural-reasoning` | Natural Reasoning plain-text | 30M | part of 5M SFT-like val |

heldout source: documents held out **before** tokenizing (not a post-shuffle split of the same packed stream). HQ val = 20M FineWeb-Edu + 5M FineWiki. SFT-like val = 5M in 40/30/15/15 proportions across cosmopedia / finemath / openhermes-pt / natural-reasoning.

eval status: n/a

## Layout

```
tokens/<source>/train-NNNNN.u32le.bin
tokens/<source>/val-NNNNN.u32le.bin
```

dtype: uint32  
ext: .u32le.bin  
target shard size: ~1 GB (same as regmix-10b)  
expected shards: on the order of ~40 train shards for ~10B uint32 tokens (plus small val shards)

No `-of-N` in filenames. No `.npy`.

## Dependencies

tokenizer: `tokenizer/dolma2-bpe` — MUST already be published (it is)  
parent: n/a

## Build notes

1. Download/stream HF sources; subsample with seed 42069666.  
2. Carve val docs first; then tokenize train.  
3. OpenHermes: draw SFT 100k examples first (see SFT design), then take PT plain-text from the remainder until 30M Dolma2 tokens.  
4. Count per-source tokens at mix time if you want measured `sources[]` in the README.  
5. Stage under a local dir or `s3://edullm-landing/...` with the `tokens/` tree above, then `publish()`.

## Deferred (backfillable, don't block)

about / sources[] / license / notes / limitations[]

## publish()

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime

publish(
    "<local dir or s3://edullm-landing/.../frontload-cl-10b/>",  # must contain tokens/
    dataset_id="pretrain/frontload-cl-10b",
    purpose=(
        "10B-token HQ+SFT-like Dolma2 pretrain mix for OLMo2-370M frontload-cl arms, "
        "to decide whether early SFT-like timing beats flat mixing on GSM8K/ARC/IFEval after shared SFT"
    ),
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/dolma2-bpe",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    # Optional README fields (backfillable later if skipped):
    # about="...",
    # sources=[...],  # prefer measured-in-this-dataset counts
    # license={"id": "...", "basis": "declared"},
)
```

After publish: landing → validator → `s3://edullm-data/pretrain/frontload-cl-10b/vN/`. On failure read `_REJECTED.json` next to the upload.

### Read example

```python
from edullm_data.read import dataset_paths
from edullm_data.s3 import Boto3S3

r = dataset_paths("pretrain/frontload-cl-10b", "v1", split="train", s3=Boto3S3.default())
# Filter r.paths by source folder for mixing / primer vs flat schedules in the train script.
# r.dtype is uint32 — do not default to uint16.
```

---

# Dataset design: sft/frontload-cl-chat-sft

purpose: Shared 1-epoch chat+math SFT mix (UltraChat, Numina, OpenHermes, no_robots) for both frontload-cl 370M PT arms, to surface GSM8K/ARC/IFEval differences without confounding post-training

family:   sft  
profile:  sft-conversations/v1   [verified in registry: yes]  
name:     frontload-cl-chat-sft  [validate_dataset_id: PASS]

## Irreversible decisions

slice path: flat under group `conversations/` (row data; source can be a field on each JSONL record if you want later filtering — path nesting is optional here; peers use flat `conversations/train-*.jsonl.gz`)

| source | take | train rows | held out |
|---|---|---|---|
| `HuggingFaceH4/no_robots` | all `train` | 9,500 | all `test` (500) |
| `HuggingFaceH4/ultrachat_200k` | all `train_sft` | 207,865 | all `test_sft` |
| `AI-MO/NuminaMath-1.5` | subsample seed 42069666 | 250,000 | 5,000 disjoint (same seed) |
| `brahmairesearch/OpenHermes-2.5-Formatted` | subsample seed 42069666 (**before** PT remainder draw) | 100,000 | TODO: optional small heldout from remainder, or rely on UltraChat/no_robots/Numina vals |

heldout source: use upstream test splits where they exist (no_robots test, UltraChat test_sft) plus a Numina 5k carve **before** finalizing the 250k train draw. Do not sample val from the same shuffled train rows after the fact.

eval status: n/a

## Layout

```
conversations/train-NNNNN.jsonl.gz
conversations/val-NNNNN.jsonl.gz
```

Each row: `{"messages": [{"role": "...", "content": "..."}, ...], ...}`  
(extra fields like `source` OK if present)

target shard size: one or few shards is fine (peer SFT sets are small)  
expected shards: ~1–3 train, ~1 val

## Dependencies

tokenizer: n/a for this artifact (SFT count unit is rows; tokenizer is a training-run property)  
parent: n/a  
coordination: OpenHermes 100k train IDs must be reserved before building `openhermes-pt` in `pretrain/frontload-cl-10b`

## Deferred (backfillable, don't block)

about / sources[] / license / notes / limitations[]

## publish()

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime

publish(
    "<local dir or s3://edullm-landing/.../frontload-cl-chat-sft/>",  # must contain conversations/
    dataset_id="sft/frontload-cl-chat-sft",
    purpose=(
        "Shared 1-epoch chat+math SFT mix (UltraChat, Numina, OpenHermes, no_robots) "
        "for both frontload-cl 370M PT arms, to surface GSM8K/ARC/IFEval differences "
        "without confounding post-training"
    ),
    profile="sft-conversations/v1",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    group_meta={
        "conversations": {
            "record_schema": {"messages": [{"role": "str", "content": "str"}]},
            "partitions": [
                {"name": "train", "by": "path", "glob": "train-*.jsonl.gz"},
                {"name": "val", "by": "path", "glob": "val-*.jsonl.gz"},
            ],
            "dedup": {"method": "sha256-content"},
            "leakage": {"reported_overlap": 0},
        }
    },
)
```

Validator recomputes train/val leakage from row contents; declared `leakage` is not trusted.

### Read example

```python
from edullm_data.read import dataset_paths
from edullm_data.s3 import Boto3S3

r = dataset_paths("sft/frontload-cl-chat-sft", "v1", split="train", s3=Boto3S3.default())
```

---

# Build / publish order

1. Confirm `tokenizer/dolma2-bpe` resolves (already on `edullm-data`).  
2. Draw OpenHermes SFT 100k; freeze IDs. *(local: `openhermes_sft_ids.json` present)*  
3. Build and tokenize `pretrain/frontload-cl-10b` (including `openhermes-pt` from remainder). **Done.**  
4. `publish()` pretrain → wait for `_VALIDATED.json`. **Done (`v1`).**  
5. Build `sft/frontload-cl-chat-sft` JSONL (train + val). **Done.**  
6. `publish()` SFT → wait for `_VALIDATED.json`. **Done (`v1`).**  
7. Register `pretrain/frontload-cl-10b/v1` in `edu-llm/platform` `config/datasets.yaml`. **Done** (`frontload-cl-10b-v1`).  
8. Point training at the registered release (`.edullm/run.yaml` + `edullm check`/`submit`; see DESIGN §10). SFT still tokenizes conversations in-script — see DESIGN §7.

Raw HF downloads stay on the build machine (or ephemeral landing staging). Only `tokens/` / `conversations/` trees are published.

When bytes are ready, use the `edullm-datasets` skill/workflow to run `publish()` — write to `s3://edullm-landing` only; never write `edullm-data` directly.
