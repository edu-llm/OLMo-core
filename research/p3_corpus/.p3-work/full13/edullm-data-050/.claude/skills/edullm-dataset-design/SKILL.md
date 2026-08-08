---
name: edullm-dataset-design
description: MUST USE at the DESIGN stage of an eduLLM dataset — before the data exists. Trigger when someone says they want to build/create/collect/generate a dataset, corpus, eval set, curriculum, or probe; is planning a tokenization run, an eval harness, a data mix, or a scrape; asks "how should I format/shard/lay out/name my data"; or asks what the pipeline requires of them. Interviews them through the irreversible decisions and emits a design spec plus the exact publish() call. NOT for publishing data that already exists — that is the edullm-datasets skill.
---

# Designing an eduLLM dataset (before the bytes exist)

Your job is to stop a teammate from generating data that the pipeline will reject — or worse,
accept while missing information they can never add back. The human doc is
`docs/DESIGN-A-DATASET.md`; this is how you run it as an interview.

**Companion skill:** `edullm-datasets` covers publishing/reading data that already exists. If
their bytes are already written, hand off to that one. If they are still deciding *how to write
them*, stay here.

## How to run this

Interview, don't lecture. Ask in the order below — it is ordered by cost of getting it wrong.
Ask a few questions at a time, and **skip anything they've already told you**. End by writing
the spec (template at the bottom) to a file they can keep.

Do not invent answers for them. If they don't know something yet, mark it `TODO` in the spec and
say what it blocks.

---

## Question 1 — what decides what? (→ `purpose`, and the whole shape)

"What is this data, what will consume it, and what decision does it drive?"

Their answer picks the family and profile. If they can't answer the third part, that is worth
surfacing gently — a dataset nobody's decision depends on may not need publishing at all.

## Question 2 — family + profile

| Their data | Family | Profile | Status |
|---|---|---|---|
| packed token shards for pretraining | `pretrain` | `pretrain-tokens/v1` | ✅ |
| an ordering / index vector over a parent token pool | `curriculum` | `token-order/v1` | ✅ |
| instruction-tuning conversations | `sft` | `sft-conversations/v1` | ✅ |
| model outputs + scores from an eval run | `eval` | `eval-results/v1` | ✅ |
| a tokenizer | `tokenizer` | `tokenizer/v1` | ✅ |
| benchmark **items** (questions, not answers) | `eval`/`probe` | `eval-items/v1` | ⚠️ **unregistered** |
| a vendored upstream tree, verbatim | `vendor` | `vendored/v1` | ⚠️ **unregistered** |

**Verify, don't trust this table** — it can go stale:

```bash
python3 -c "from edullm_data.profiles.registry import available; print(sorted(available()))"
```

⚠️ **`probe` and `vendor` default to profiles that do not exist.** `families/probe.json` names
`eval-items/v1` and `families/vendor.json` names `vendored/v1`; neither is in
`registry.py:_SHIPPED`. `experimental/v1` (referenced in `CONTRIBUTING.md`) is also unregistered.
Publishing into those families today raises `ProfileError`. If that's their shape, tell them
plainly: writing the profile is part of their work — a registry entry, a schema fragment, checks
that recompute, and two fixtures (`CONTRIBUTING.md`) — and they should talk to Eric first.

If nothing fits, **writing a profile is the expected path**, not an exception.

## Question 3 — the name (validate it live, don't eyeball it)

`<family>/<name>`, kebab-case, 2–5 words. It must state **what it is + the one axis that
distinguishes it from its siblings**.

Never approve a name by eye. Run it:

```bash
python3 -c "
from edullm_data.contracts import validate_dataset_id
for n in ['pretrain/dolma2-150b', 'pretrain/final-v2']:
    try: print('OK  ', validate_dataset_id(n))
    except Exception as e: print('FAIL', n, '->', e)
"
```

Rejected: non-kebab, <2 or >5 words, dates (YYYYMMDD / 19xx / 20xx / month tokens), version
tokens (`v2`, `final`, `latest`, `fixed`), content-free words (`data`, `results`, `test`, `tmp`,
`misc`), relative words (`big`, `improved`, `better`), bare ordinals (`experiment-3`).
Allowed: 4-digit upstream release codes that aren't years (`1124`), scale suffixes (`370m`,
`150b`, `5b5`, `4hop`).

Person names and ticket IDs are **not** mechanically caught — flag them yourself.

## Question 4 — THE IRREVERSIBLE ONE: slice structure in the path

> "Six months from now, will you want to train on, or measure, only *part* of this — one source,
> one domain, one difficulty band?"

If there is any chance of yes, that structure must be **in the object key**, decided before they
write a byte:

```
tokens/arxiv/science/train-00000.u32le.bin      ← keeps source + domain
tokens/train-00000.u32le.bin                    ← flattened; information gone
```

**Why it can't wait:** nesting is preserved in `entry.path`, which is serialized by
`ManifestEntry.to_dict()` into the manifest and hashed into `manifest_sha256` — the dataset's
identity. Re-pathing later means republishing, which means **re-copying every payload byte**.
This is exactly where the 150B corpus ended up; its mapping was recoverable only because sorted
order happened to be bijective, which is luck, not a plan.

**Tell them two levels, `<source>/<domain>` — that is the whole budget.** `publish()` derives
`entry.labels` from exactly those segments, and exactly two are named. Show them, don't assert it:

```bash
python3 -c "
from edullm_data.manifest import PATH_LABEL_KEYS, labels_from_path
print('keys:', PATH_LABEL_KEYS)
for p in ['tokens/train-00000.u32le.bin',
          'tokens/arxiv/train-00000.u32le.bin',
          'tokens/arxiv/science/train-00000.u32le.bin']:
    print(' ', p, '->', labels_from_path(p))
try: labels_from_path('tokens/a/b/c/train-00000.u32le.bin')
except Exception as e: print('  3 levels ->', type(e).__name__)"
```

A third level **raises** rather than inventing a `level_3` — deliberately, because labels are
inside `manifest_sha256` and a wrong one can't be fixed without republishing.

**Do not tell them to pass labels to `publish()`.** There is no `labels` parameter and there is
not meant to be one — a hand-typed label would be a producer assertion nothing falsifies, which
the golden rule forbids. `publish()` reads them off the key; Gate A **recomputes** them from that
same key (`_check_labels_match_path`) and rejects a label contradicting the path, or a nested
entry whose labels were omitted. The path is the single source of truth; the only thing they
control is how they name files.

Absence is one-directional: a flat layout yields no labels, silently and legally. Only a *nested*
key with mismatched labels is rejected — a reader slicing on labels would drop that object from
every slice.

**Labels are a real reader selector**, so the tree you nest decides what a trainer can later ask
for: `dataset_paths(..., labels={"source": "arxiv"})` narrows a read, and `build_mixture(...)`
uses the same predicate to build a weighted, seeded subset. A flat tree yields no labels and
there is nothing to select on.

**The one caveat:** `partitions[]` still cannot express a label selector — the empty-split check
fires regardless of `by`, so selection is a read-side concern rather than a declared partition.
That costs a producer nothing; the slice lives in the key and the manifest.

## Question 5 — where does held-out data come from?

Not "is there a val split" — **"is val drawn from a different pool than train?"**

The right answer holds out *documents before tokenizing*. A val split sampled from the same
shuffled pool as train is not a val split. The validator catches duplicates by content digest
within a group, which is real but limited protection.

`sft-conversations/v1` requires ≥2 partitions with one held out, plus `dedup` and `leakage`
blocks that its checks recompute.

## Question 6 — eval runs only: per-row status

`eval-results/v1` refuses a file where `n_ok == 0`. This check exists because the audit found 12
CSVs of 66 bytes (header, no rows) and 3 identical files where every row said `error`.

A row's status **only exists while the eval is running**. Tell them to write a `status` field per
row as they go and accumulate `status_counts`. Also captured at run time, from the config object
and not retyped later: `model{id,revision}`, `task`, `decode{temperature,top_p,max_tokens}`.

## Question 7 — layout, shards, dtype

```
artifacts/public/
└── tokens/                      ← group = first path segment = picks the profile
    ├── train-00000.u32le.bin
    └── val-00000.u32le.bin
```

- One subdirectory per group; a file not under a group prefix is a hard publish error.
- **Split is matched by glob on the filename** (`train-*.u32le.bin`) — the split lives in the name.
- **No `-of-NNNNN`** — the surviving shard count is unknown at write time.
- **`.u32le.bin`, never `.npy`.** OLMo-core memmaps from byte 0 and derives token count from raw
  file size; a real `.npy` header corrupts both, silently.
- **`tokens × dtype_size == bytes`, exactly.** dtype is declared, never inferred — OLMo-core
  defaults to `uint16` while these corpora are `uint32`, which halves the count.
- Shard size: shipped corpora run ~500 MB–1 GB. Warn against degenerate tiny shards (two 20-byte
  shards once tripped both the duplicate-digest and eos-fraction checks).

## Question 8 — tokenizer ordering

If it's a pretrain/curriculum corpus: **the tokenizer must be published first.** The corpus names
it via `tokenizer="tokenizer/<name>"`, and the validator derives `vocab_size`/`eos_token_id` from
the real `tokenizer.json` and asserts every sampled id against it. That's two publishes, in order.

Never suggest a family-wide tokenizer default: a mismatched tokenizer is nearly undetectable
because vocab sizes cluster near 100k, so wrong ids stay in range and pass the decode check while
being semantically wrong.

## Question 9 — the cheap stuff (mention, don't block on)

`about`, `sources[]`, `license`, `notes`, `limitations[]` feed the generated `README.md`, which is
a control file outside the hash chain — **all backfillable in place** on a frozen dataset. Never
let a missing `about` delay a publish. Absent sections are omitted, never faked.

One thing worth doing at collection time: if they want the README's mix table to show *this*
dataset's real per-source token counts, they must count during the mix. Afterwards they can only
cite upstream totals with `scope: "upstream-full-collection"`, which prints an honesty caveat.

---

## Emit the spec

Write this to a file (e.g. `DATASET-DESIGN.md` next to their work) and hand them the `publish()`
call. Keep `TODO` markers visible rather than guessing.

```markdown
# Dataset design: <family>/<name>

purpose:  <what it is, what uses it, what it decides>
family:   <one of pretrain curriculum sft eval probe vendor tokenizer>
profile:  <registered profile>   [verified in registry: yes/no]
name:     <name>                 [validate_dataset_id: PASS]

## Irreversible decisions
slice path:       <group>/<source>/<domain>/<shard>   or  "flat, because <reason>"
heldout source:   <different pool, carved before tokenizing>
eval status:      <how per-row status is recorded during the run>   [eval only]

## Layout
<group>/[<source>/<domain>/]<split>-<NNNNN><ext>   dtype: <uint32>  ext: <.u32le.bin>
target shard size: <~1 GB>            expected shards: <n>

## Dependencies
tokenizer: <tokenizer/name>  — MUST be published before this corpus
parent:    <dataset_id>      — for curriculum/derived only

## Deferred (backfillable, don't block)
about / sources[] / license / notes / limitations[]
```

Then the call, filled in from the above:

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime

publish(
    "<local dir or s3://edullm-landing/...>",
    dataset_id="<family>/<name>",
    purpose="<...>",
    profile="<profile>",
    tokenizer="<tokenizer/name>",          # pretrain/curriculum only
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    group_meta={"<group>": { ... profile-required fields ... }},
)
```

## Close the loop

Remind them: they write to `s3://edullm-landing` (anyone may write, 14-day expiry); the validator
re-checks every byte and copies passes into `s3://edullm-data`, which they cannot write to. On
failure, read the `_REJECTED.json` dropped next to the upload — each violation names the concrete
failure. Then point them at the `edullm-datasets` skill for the publish itself.

Publishing is meant to be cheap — four hand-typed arguments, seconds of hashing. If their design
makes it expensive, that's a pipeline bug worth reporting to Eric, not something they should
work around.
