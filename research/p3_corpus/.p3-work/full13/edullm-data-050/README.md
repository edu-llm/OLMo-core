# edullm-data

Publisher, validator, and reader for the eduLLM dataset standard. The standard itself — naming,
manifest shape, the airlock, profiles, validation gates — lives at
[`../docs/dataset-creation/DATASET-STANDARD.md`](../docs/dataset-creation/DATASET-STANDARD.md). This
package is code that implements that document; it is not a second source of truth. If the two disagree,
the standard wins and this package has a bug.

## Start here

| You are… | Read |
| --- | --- |
| brand new to this | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) — the 2-minute mental model |
| **about to build a dataset** (no bytes yet) | [`docs/DESIGN-A-DATASET.md`](docs/DESIGN-A-DATASET.md) — the decisions that can't be undone later |
| publishing or reading data that exists | [`USAGE.md`](USAGE.md) |
| adding a profile / changing a check | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## What is in `s3://edullm-data` right now

Two datasets. **6,927 objects, 586.6 GiB** (as of 2026-07-30):

| Dataset | Contents |
| --- | --- |
| `pretrain/olmo-150b-dolma2/v1` | **157,467,202,883 dolma2 tokens** across 6,911 headerless uint32 shards — 6,851 `train` + 60 `val`. One `tokens/` group nested `<source>/<domain>/`, so every shard's provenance is in its key and mirrored into `entry.labels` (which Gate A recomputes from that key). 6 sources / 65 source-domain strata. |
| `tokenizer/dolma2-bpe/v1` | `allenai/dolma2-tokenizer` — tokenizer.json, merges.txt, vocab.json, configs. `vocab_size 100278` / `eos 100257` **derived** from tokenizer.json, never typed. The corpus above pins it by `manifest_sha256`. |

Read it the way a trainer does — **always pass `r.dtype` to the loader**, because OLMo-core
defaults to `uint16` and reading these `uint32` shards as `uint16` does not raise, it silently
doubles the token count:

```python
r = dataset_paths("pretrain/olmo-150b-dolma2", "v1")   # trainable data only
r.dtype        # "uint32"   -> pass this explicitly
r.numpy_dtype  # "<u4"      -> little-endian is explicit; np.dtype("uint32") is host order
r.train, r.val # 6,851 and 60 shards, disjoint
```

An earlier corpus, `pretrain/olmo-mix-1124-31b/v1` (31.3B tokens, 218 shards), was deleted on
2026-07-29: it had no `val` split and, being frozen, could not gain one, so under
validation-required-by-default it failed permanently. Its bytes remain recoverable two ways — a
byte-identical legacy copy at `s3://edullm-datasets/olmo30b/` (218 `.npy` shards,
125,336,003,336 bytes) and 342 noncurrent versions in the bucket itself.

Publishing a replacement corpus is the open work. See `HANDOFF.md`.

## Install

Public repo — the `git+https` install needs no auth. Pin a tag so the publisher and validator agree:

```bash
uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"
```

For local development, editable install with test extras:

```bash
python3 -m pip install -e ".[test]"
```

## Publishing a dataset

Four arguments, everything else derived or inherited from `<family>/family.json`:

```python
publish(source,
        dataset_id="eval/mcq-arc",
        purpose="ARC/OpenBookQA/SciQ loglikelihood scores for 23 baseline models, "
                "for the adaptive-inference baseline table",
        profile="eval-results/v1")
```

`source` is a local path or an `s3://edullm-landing/...` URI already written by a Batch job. `publish()`
only ever writes to landing — never to `edullm-data` — and if `source` is already in landing it just
seals the manifest without moving bytes.

### The generated README

Every promoted dataset carries a `README.md` at its prefix, **generated from `dataset.json`** by the
validator (§3: the README is a derived artifact, never hand-written). Pass optional descriptive
fields to `publish()` to fill it — `about` (one curated prose block), `sources` (the data-mix table;
mark a source `scope:"upstream-full-collection"` to print an honesty caveat when it's upstream
figures rather than this dataset's measured mix), `license`, `notes`, `limitations`. None are
validator-required; they only feed `render_readme()` (`src/edullm_data/readme.py`). Sections whose
data is absent are omitted, never faked. Already-promoted datasets get a README too — it's a control
file, backfilled in place without touching any payload byte or manifest hash.

## The one thing to know

**Producers write ONLY to `s3://edullm-landing`.** The validator role
(`<BATCH_JOB_ROLE>`, service-assumable only — no human or intern session can
assume it) is the *only* principal with `PutObject` on `s3://edullm-data`. That is an IAM bucket-policy
Deny, not a convention.

This package is a convenience, not the gate. If you refuse to install it and hand-roll a manifest, the
airlock still catches you the same way it catches everyone else — you cannot write directly to the read
bucket regardless of what tooling you use. Use the package because it's easier, not because it's
required.

## Package tree

Status legend: **done** — shipped and in `main`; **in progress** — being written now; **TODO** — not
started.

```
edullm-data/                    ← git root
├── pyproject.toml              [project.name = "edullm-data"]                    done
├── src/edullm_data/
│   ├── contracts.py            canonical_json, hashing, naming                   done
│   ├── manifest.py             build + verify manifests                          done
│   ├── s3.py                   S3 protocol + Boto3S3 + FakeS3                     done
│   ├── publish.py              publish() — landing-only writes                   done
│   ├── read.py                 dataset_paths() reader                            done
│   ├── validate.py             Gate A + promote() (writes the generated README)  done
│   ├── readme.py               render_readme() — README generated from dataset.json  done
│   ├── fsck.py                 wu-fsck, Gate B, weekly integrity re-check        done
│   └── profiles/
│       ├── base.py             Violation / GroupContext / sample_offsets         done
│       ├── registry.py         profile lookup by name                            done
│       ├── pretrain_tokens_v1.py   tokenizer pin, vocab bound, alignment         done
│       ├── eval_results_v1.py      model pin, decode params, failure accounting  done
│       ├── token_order_v1.py       permutation/index-vector checks               done
│       └── sft_conversations_v1.py messages[] schema, dedup + leakage report     done
├── infra/                      CloudFormation + policies + Dockerfile + runbooks done (deployed)
├── families/                   the six family.json files                         done
└── tests/                      541 passing                                       done
```

## Build status (standard §13)

| # | Step | Status |
|---|---|---|
| 1 | git root created, package dirs scaffolded | done |
| 2 | infrastructure: `edullm-landing` / `edullm-data`, bucket policy, lifecycle, versioning | **done — deployed live** |
| 3 | `contracts.py` + `manifest.py` | done |
| 4 | validator (Gate A), Batch-only, profile-driven | done |
| 5 | four v1 profiles + tests (`pretrain-tokens`, `eval-results`, `token-order`, `sft-conversations`) | done |
| 6 | `publish()` | done |
| 7 | `dataset_paths()` reader | done |
| 8 | six `family.json` files | done |
| 9 | event wiring (EventBridge on landing → Batch queue), deployed DISABLED | **done — deployed live** |
| 10 | S3 Inventory on `edullm-data` | **done — deployed live** |
| 11 | `wu-fsck` (Gate B), weekly, owner **Eric Wu** | done (code); re-scheduling TODO |
| 12 | generate the agent skill from the profile registry | todo |

**One packaging step remains before the validator runs in-cluster** (`infra/05-validator-jobdef.md`):
the container needs the package, which requires either a Docker host + ECR push, or a git remote to
`pip install` from — neither available from the build workstation. The wheel builds cleanly; two deploy
paths are documented. The airlock itself is proven end-to-end (live promotion test).

`wu-fsck` is named for its owner deliberately — an unowned recurring job gets muted after its first
false alarm. Ownership transfers by renaming the job, not by editing a config field.

It runs **weekly**, not nightly. Every fact it re-checks (an object deleted or truncated, a payload
overwritten at the same length, a `depends_on` parent republished or removed, an ECR image expired)
changes only when something mutates a frozen prefix or another dataset's lifecycle — rare, and no more
urgent at 24-hour granularity than at 7-day. The sweep is cheap either way; the scarce resource is the
owner's attention, and nightly spent it seven times over for the same information.
