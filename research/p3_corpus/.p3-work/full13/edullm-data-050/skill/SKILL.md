---
name: edullm-datasets
description: MUST USE for anything involving eduLLM datasets on S3 — creating, publishing, naming, reading, or discovering a training/eval/curriculum/probe dataset, or auditing what exists. Covers where datasets live (s3://edullm-data), how they are named, how to publish one through the airlock validator, how to read one into training, and what the validator will reject. Trigger whenever a task mentions publishing a dataset, a token corpus, an eval set, a curriculum ordering, edullm-landing, edullm-data, dataset.json, a manifest, or "where is the dataset".
---

# eduLLM Datasets

The single way datasets are created, stored, and read in this project. The full spec is
`docs/dataset-creation/DATASET-STANDARD.md`; the diagrams are `DATASET-STANDARD-DIAGRAMS.md`.
This skill is the operational summary — follow it, don't re-derive it.

## The one thing to understand first

There are two buckets:

- **`s3://edullm-landing`** — anyone may write here. Nothing trains from here. 14-day expiry.
- **`s3://edullm-data`** — everyone reads; **only the validator role can write.** This is a
  locked bucket (an IAM Deny), not a convention. You cannot `aws s3 cp` into it, and neither
  can anyone else. Datasets get here only by passing the validator.

So publishing is: write to landing → the validator checks it → it copies the passing dataset
into `edullm-data` and writes a catalog entry. You never write `edullm-data` directly.

## Prerequisite: the `edullm-data` package MUST be installed

This skill is all you need to *know* — but it is not self-executing. Every command below
calls the **`edullm-data` Python package**. It is a hard requirement: without it there is no
`publish()` and no `dataset_paths()`, and you cannot correctly hand-roll a compliant
`dataset.json` + manifests (the validator will reject anything that isn't byte-for-byte what
the publisher produces).

**Before doing anything else, ensure it is installed, and install it if not:**

```bash
python3 -c "import edullm_data" 2>/dev/null && echo "present" || \
  uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"
# local dev:  python3 -m pip install -e /path/to/edullm-data
```

Install it **wherever you run `publish()`** — the machine that holds the data and can reach
AWS (laptop, FarmShare node, GPU box, or an AWS Batch job). The package is a *client*: it
uploads to `s3://edullm-landing`. Nothing is installed "onto AWS" — the buckets, validator,
and the event rule that auto-validates on upload are already deployed and running.

You also need **AWS credentials that can write `edullm-landing`** (in this project: the
`sb-aws` broker; elsewhere: ordinary AWS creds with `s3:PutObject` on that bucket).

The package gives you `publish()` (produce a dataset) and `dataset_paths()` (read one, with
the correct dtype). Confirm it's importable before you start, or the steps below will fail.

## Publishing a dataset — four hand-typed arguments

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

publish(
    source,                       # local dir OR "s3://edullm-landing/<staged-prefix>/"
    dataset_id="eval/mcq-arc",    # <family>/<name> — see naming rules below
    purpose="ARC/OpenBookQA/SciQ loglikelihood scores for 23 baseline models, for the baseline table",
    profile="eval-results/v1",    # or a dict {group: profile} for multi-group datasets
    s3=Boto3S3.default(),
    created_at="<ISO-8601 UTC>",
    group_meta={...},             # profile-specific fields per group (see profiles below)
)
```

Everything else is derived: version (auto-allocated), object hashes, byte/row/token counts,
formats, build provenance (from the environment), and `license`/`tokenizer`/`sources`
(inherited from `families/<family>.json`, written once per family).

From an AWS Batch job that already wrote output to landing, pass the `s3://` prefix as
`source` — no bytes leave S3.

## Every dataset gets a generated README — always fill in the data-mix fields

Every published dataset carries a `README.md` at its prefix
(`s3://edullm-data/<id>/<version>/README.md`), **generated from `dataset.json`** by the validator
at promotion time (§3: the README is a derived artifact, never hand-written — so it can't drift
from the manifest). This is not optional and not something you upload: the validator writes it for
**every** promotion.

Your job is to give it real content by passing the descriptive fields to `publish()` (all
optional, none validator-required — they only feed the README):

```python
publish(..., 
    about="A curated paragraph: what this data is and how it was produced.",
    sources=[{"name": "DCLM-Baseline", "tokens": 3_700_000_000_000, "license": "CC-BY-4.0",
              "scope": "upstream-full-collection",   # → prints an honesty caveat (see below)
              "uri": "https://huggingface.co/datasets/allenai/olmo-mix-1124"}],
    license={"id": "ODC-By-1.0", "basis": "declared"},   # override the family's honest "unknown"
    notes="Free-text caveat.",
    limitations=[{"kind": "contamination", "benchmark": "gsm8k", "overlap_rate": 0.003}])
```

- `about` is the one **curated** prose block; contents/tokenizer/splits/inventory/how-to-read are
  **derived**. A section is omitted when its data is absent (empty `sources=[]` → no mix table,
  never a fake one — do not fabricate).
- Mark a source `scope: "upstream-full-collection"` when the figures describe the upstream
  collection and not a measured breakdown of *this* dataset (a subset/derivation). The README then
  prints a caveat so the numbers read as provenance, not as this dataset's realized mix. **Never
  present upstream proportions as this dataset's measured mix.**
- **For a dataset that is ALREADY promoted** (frozen, live in `edullm-data`): you still add its
  README. The README is a control file, so a validator-role job re-writes `dataset.json` with the
  added descriptive keys and writes `README.md` **in place** — no payload byte, manifest hash, or
  inventory changes (the frozen contract on the data holds). Do this rather than re-publishing a v2
  just to add documentation. The backfill driver + guardrails (descriptive-keys-only, recompute the
  group `manifest_sha256`/inventory unchanged before writing) are the proven pattern; see the
  package HANDOFF.

## Naming — the user will manually verify name quality, so get it right

`s3://edullm-data/<family>/<name>/<version>/`

- **`<family>`** is a fixed enum: `pretrain` · `curriculum` · `sft` · `eval` · `probe` · `vendor` · `tokenizer`
- **`<name>`** is kebab-case, 2–5 words, stating **what the data is + the one axis that
  distinguishes it from its siblings**. No dates, no version tokens, no person names, no
  relative words.
- **`<version>`** is `v1`, `v2`, … auto-allocated. Never put it in the name.

**Good names** (copy this style):

| Name | Why |
|---|---|
| `pretrain/dolma2-150b` | corpus + token budget (the distinguishing axis) |
| `pretrain/olmo-mix-1124-30b` | upstream release code + budget |
| `pretrain/fineweb-edu-10b` | corpus + budget |
| `curriculum/flesch-linear-370m` | difficulty signal + schedule + target model |
| `curriculum/zlib-strict-370m` | same axes, different values — siblings read clearly |
| `sft/pedagogical-tutoring-100students` | task + defining scale |
| `eval/mcq-arc-openbookqa-sciq` | task type + exactly which benchmarks |
| `eval/judge-blinded-5x5` | protocol + design shape |
| `probe/multihop-wikidata-4hop` | task + source + depth |
| `vendor/dclm-hero-run-fasttext` | upstream name preserved verbatim |

**Reject these** (the validator rejects them mechanically, but don't propose them):
`pretrain/datamix1-jul22` (date), `pretrain/final-v2` (version tokens), `eval/results`
(says nothing), `curriculum/experiment-3` (meaningless ordinal), `sft/good-data` (relative),
`pretrain/dolma2_150B` (snake_case + caps), `eval/mcq-v2-fixed-final` (three version tokens).

Test: *if two people independently built this, would they pick the same name? Could you tell
it from its five nearest siblings without opening it?*

## Purpose — one line: what it is, what consumes it, what it decides

Good: `"150B-token Dolma2 mix for 370M ladder pretraining at 1.25xC"` ·
`"Blinded judge verdicts, 5 judges x 5 prompt variants x 3 replicates, to measure judge reliability"`

Reject: `"training data"`, `"the dataset"`, `"TODO"`, `"experiments"`, `"see README"` — all
either say nothing or will never be filled in. Shape: `<what it is> for <what consumes it> to
<what it decides>`.

## Choosing a profile

A profile is the contract for one payload **group** (a dataset can have several — e.g. token
shards + a sidecar table — each with its own profile). Pick by what the bytes are:

| Payload | Profile | Required in `group_meta` |
|---|---|---|
| a published tokenizer (tokenizer.json + friends) | `tokenizer/v1` | nothing — vocab_size/eos are DERIVED from tokenizer.json |
| packed pre-tokenized corpus (`.u32le.bin`) | `pretrain-tokens/v1` | `publish(tokenizer="tokenizer/<name>")` — names the per-dataset tokenizer; vocab derived from it, not typed |
| raw untokenized documents | `text-corpus/v1` | record schema, `text` field |
| instruction / conversation data | `sft-conversations/v1` | `record_schema` w/ `messages[]`, `partitions` incl. heldout, `dedup`, `leakage` |
| benchmark inputs | `eval-items/v1` | stable per-item id |
| model outputs / scores | `eval-results/v1` | `model{id, revision}`, `task`, `decode{...}`, `status_counts{}` |
| per-record derived metrics | `annotations/v1` | `parent_group`, row-i == parent row-i |
| index vectors (views, curricula) | `token-order/v1` | `depends_on[]`, ordering kind |
| parallel float arrays (loss weights) | `weights-sidecar/v1` | `parent_group` |
| any tabular w/ column schema | `tabular/v1` | column schema |
| image/video/audio | `media/v1` | decode facts; label index if supervised |
| telemetry / timeseries | `metrics-timeseries/v1` | timestamp field |
| verbatim third-party tree | `vendored/v1` | `upstream{}`, `vendor_root` |
| transfer archives | `distribution-artifact/v1` | `packages{}` + part checksums |
| doesn't fit / shipping today | `experimental/v1` | `exception{reason, approver, expires_at}` — max 2 live per family |

**If nothing fits, write a profile** — that's the expected path, not `experimental`. A profile
is a small PR (see `edullm-data/CONTRIBUTING.md`): a registry entry, a schema fragment, check
functions that RECOMPUTE something, and two fixtures. Only reach for `experimental/v1` when you
must ship today and don't yet know the shape.

### Tokenizers: publish once, name per dataset

The tokenizer is an owned artifact, not an HF reference — otherwise your token shards become
undecodable if the upstream repo moves. And **there is no single canonical tokenizer** — every
corpus may use its own — so each corpus **names the tokenizer it was built with**.

**Step 1 — publish the tokenizer** (once per distinct tokenizer):

```python
publish(tokenizer_dir,                 # a dir containing tokenizer.json (+ merges, config…)
        dataset_id="tokenizer/dolma2-bpe",
        purpose="Published Dolma2 tokenizer so corpora own the tokenizer they were built with",
        profile="tokenizer/v1", s3=..., created_at=...)
```

**Step 2 — name it when publishing a corpus**, via the first-class `tokenizer=` argument:

```python
publish(tokens_dir, dataset_id="pretrain/dolma2-150b", purpose="…",
        profile="pretrain-tokens/v1",
        tokenizer="tokenizer/dolma2-bpe",   # ← THE per-dataset tokenizer. use the exact one
        s3=..., created_at=...)             #    this corpus was tokenized with
```

`tokenizer=` accepts `"tokenizer/<name>"` (latest published version) or
`"tokenizer/<name>/vN"` (exact). The validator **derives** `vocab_size`/`eos_token_id` from that
tokenizer's `tokenizer.json` and asserts every token id against *that* vocabulary — the real
one this corpus used, never a guess or a shared default. **A pretrain corpus with no resolvable
tokenizer is rejected** (the vocab check can't run without one). Do NOT rely on a family-wide
default — tokenizers vary per dataset, and a wrong default passes silently (mismatched vocab
sizes are usually still in range). Only set `tokenizer_dependency_optional` in the family if a
team genuinely standardizes on one tokenizer for everything.

## Format rules that bite

- **Packed uint32 tokens are `.u32le.bin`, NEVER `.npy`.** OLMo-core memmaps from byte 0 and
  derives the token count from raw file size; a real `.npy` header corrupts both the tokens and
  the count. Declare `dtype` explicitly (uint32, not the uint16 default) — it is read, never
  inferred.
- **Shard names are `<split>-<NNNNN>.<ext>`, no `-of-N`.** The total is unknowable at write
  time; completeness is proven by the manifest, not the filename.
- **Every file's extension must match its declared format.** The validator sniffs magic bytes.

## What the validator will reject (so you don't ship it)

Gate A recomputes, it doesn't trust. It rejects: a `manifest_sha256` that doesn't match the
canonical manifest JSON; a shard listed but missing, or present but unlisted; a HEAD size
disagreeing with the manifest's `bytes`; `count × dtype_size != bytes` (truncation, or a size
that isn't a whole multiple of the item width); a dtype too narrow for the tokenizer's derived
vocab; a `.npy` that's really headerless raw; a partition whose `rows` don't sum from its entries;
a `coverage: "partition"` whose splits overlap; duplicate shard digests; an inventory count that
doesn't match reality; a shard sha256 shared with a `depends_on` parent (copy instead of
reference); an unknown profile. Per profile it also reads bytes — ~64 KB per shard at seeded
offsets: all-zeros / all-EOS / wrong-endianness token shards; an eval-results file where
every row errored (`n_ok == 0`); a degenerate curriculum ordering that isn't a permutation;
train/heldout leakage in SFT.

**Gate A does not re-hash your payload.** Only the producer hashes (`publish()` streams every
file once); the validator checks SIZE per entry, not a digest. Your declared `sha256` is used for
content addressing — catching a shard duplicated inside the group, or one copied from a
`depends_on` parent instead of referenced — and for the hash chain. So a correct `sha256` is not
a substitute for correct bytes: what keeps `edullm-data` trustworthy is that only the validator
role can write it (an IAM Deny, not a convention), plus S3 durability and CRC64NVME.

## Reading a published dataset

```python
from edullm_data.read import dataset_paths, resolve_latest
from edullm_data.s3 import Boto3S3

s3 = Boto3S3.default()
ver = resolve_latest("pretrain/dolma2-150b", s3=s3)         # e.g. "v3"
r = dataset_paths("pretrain/dolma2-150b", ver, split="train", s3=s3)
# r.paths  -> ["s3://edullm-data/pretrain/dolma2-150b/v3/tokens/train-00000.u32le.bin", ...]
# r.dtype  -> "uint32"   (feed this to the loader; do NOT let it default to uint16)
# r.rows   -> declared row/token count for the split
```

It refuses a prefix with no `_VALIDATED.json` — unvalidated data is not readable.

## Discovering what exists

List `s3://edullm-data/_catalog/` — one immutable JSON per published `<dataset_id>/<version>`.
That is the discovery surface; there is no single mutable index. To find datasets by family,
list `s3://edullm-data/<family>/`.

## Governance you can rely on

`edullm-data` is versioned, deny-delete except break-glass, AES256, one region (us-east-1),
weekly S3 Inventory. Nightly `wu-fsck` (owner: Eric Wu) re-checks that published datasets still
resolve (sources, parents, object presence) — the failures that only appear after publish.
There is no PII in scope; if you are handling personal data, this standard does not cover it —
stop and escalate.
