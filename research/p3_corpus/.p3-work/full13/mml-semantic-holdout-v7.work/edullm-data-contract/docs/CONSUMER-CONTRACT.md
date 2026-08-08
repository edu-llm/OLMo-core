# The consumer contract — what a trainer must carry across the boundary

For whoever writes the training-side adapter. `docs/ONBOARDING.md` is the 2-minute tour of the
*producer* side; this is the read side, stated precisely enough to be implemented against.

Everything below is verified against the code as of this branch, with `file:line`. Where a claim is
about the **consumer** (OLMo-core) rather than this package, it is marked **[CONSUMER]** — this repo
cannot enforce those, and nothing here will stop you getting them wrong.

```
THE BOUNDARY
  edullm-data  ──dataset_paths()──►  ResolvedSplit  ──you map it──►  training config
  (the library)                      paths + dtype                  (OLMo-core)
                                     + splits, kept apart

  Three things cross: WHICH objects, WHAT WIDTH to read them at, and WHICH ONES
  you are allowed to train on. Drop any one of the three and the run still starts.
```

---

## 1. The address, and how to resolve it

Every dataset lives at exactly one shape:

```
s3://edullm-data/<family>/<name>/<version>/
```

`<family>` is one of seven (`contracts.py:126`), `<name>` is kebab-case 2–5 words, `<version>` is
`v1`, `v2`, … and is **immutable once published**. You never construct this path by hand — you
resolve it:

```python
from edullm_data.read import dataset_paths, resolve_latest
from edullm_data.s3 import Boto3S3

s3 = Boto3S3.default()                                   # s3.py:109-113
version = resolve_latest("pretrain/olmo-mix-1124-31b", s3=s3)   # -> "v1" | None
r = dataset_paths("pretrain/olmo-mix-1124-31b", version, s3=s3)
```

`Boto3S3.default(region="us-east-1")` imports boto3 lazily and builds a client from the ambient
environment (`s3.py:109-113`) — on Batch that is the task role, locally it is your profile. `S3` is a
`Protocol` (`s3.py:34`), so an adapter can be unit-tested against `FakeS3` (`s3.py:251`) with zero
AWS. Pass the accessor explicitly; it is a keyword-only argument on every read function, deliberately,
so no module-level singleton decides which account you are reading from.

`resolve_latest` reads the catalog, not the payload: it lists `_catalog/<dataset_id>/` and returns the
highest `vN` it finds, or `None` (`read.py:404-415`). It does **not** check that the data behind that
version still exists — the catalog entry is the claim, `dataset_paths` is the verification.

**Reading `edullm-data` is open; writing is an IAM Deny.** The bucket policy's Deny statements list
data-plane *write* actions only — `s3:PutObject`, `s3:PutObjectTagging`, `s3:AbortMultipartUpload`,
plus `s3:DeleteObject*` in v2 (`infra/02-bucket-policy.json`). `s3:GetObject` is not among them, and
`infra/DEPLOY.md:573` is the smoke test that pins this: any authenticated principal in the account
that has read permission via its identity policy can read published data. The bucket is *not* public
(`BlockPublicPolicy: true`, `RestrictPublicBuckets: true`, `infra/01-buckets.yaml:113`, `:115`) —
"open for reads" means "no resource-policy Deny stands in your way," not "anonymous."

So: a training job needs `s3:GetObject` + `s3:ListBucket` on `edullm-data` and nothing else. If your
adapter ever needs a write grant to this bucket, the design has gone wrong — see the airlock in
`docs/ONBOARDING.md`.

---

## 2. What `dataset_paths` gives you back

`ResolvedSplit` (`read.py:80-155`) is a plain dataclass. Every field, exactly:

| field | type | what it is |
| --- | --- | --- |
| `dataset_id` | `str` | echoed back, e.g. `"pretrain/olmo-mix-1124-31b"` |
| `version` | `str` | echoed back, e.g. `"v1"` |
| `split` | `str` | the split you asked for, or `"*"` for an unsplit read (`read.py:350`) |
| `paths` | `list[str]` | **full `s3://` URIs** for the objects this call selected (`read.py:90`, built at `read.py:264-265`) |
| `dtype` | `str \| None` | the numpy dtype **name**, e.g. `"uint32"` — see §3 (`read.py:91`) |
| `rows` | `int \| None` | the declared row/token count **of the requested split** (`read.py:92`) |
| `kwargs` | `dict` | loader hints a non-path partition implies (`read.py:93`) |
| `byte_order` | `str \| None` | `"little"` / `"big"`, or `None` for a self-typing container (`read.py:94-102`) |
| `header_bytes` | `int` | leading bytes to skip before element 0 — `0` for `.u32le.bin` (`read.py:103-107`) |
| `splits` | `dict[str, list[str]]` | **every declared split, separately keyed** (`read.py:108-114`) |
| `split_rows` | `dict[str, int \| None]` | per-split declared counts, same keys as `splits` (`read.py:115-116`) |

Plus three properties and one predicate:

```python
r.train          # list[str] — trainable URIs. [] if nothing trainable is declared. read.py:118-121
r.val            # list[str] | None — HELD-OUT URIs, or None.                       read.py:123-132
r.has_split("val")   # bool — membership in r.splits, nothing more.                 read.py:134-135
r.numpy_dtype    # str | None — "<u4": dtype AND byte order, ready for np.dtype.    read.py:137-155
```

### Five precise things about these

**`.val` is `None`, not `[]`, when there is no held-out data** (`read.py:132`, `return held or None`).
This is not stylistic. `[]` collapses two different facts into one value: *"this dataset declares no
validation data"* and *"the validation split is declared but resolved to zero objects."* A caller that
branches on the first — "skip the eval callback, this corpus has nothing to measure on" — must not
have to guess which one it is holding. `None` is the honest answer to a question that has no list. The
property never raises (`read.py:129`): asking is not an error.

The empty-list case is reachable and distinguishable: `has_split("val")` is `True` while
`r.splits["val"] == []`. That state is a Gate A violation (`empty-split`, `validate.py:679`), so it
should not survive publication — but the reader still reports it faithfully rather than papering over
it.

**`.val` means "held out", not literally the split named `val`.** Both properties are derived through
`contracts.is_trainable` (`read.py:121`, `read.py:131`), so a `test` partition also lands in `.val`.
If you need the specific word, index `r.splits["test"]`.

**`.paths` is hardened; `.splits` / `.train` / `.val` are declaration-derived.** For an unsplit read,
`paths` gets a second filter applied that recomputes each object's split from its own filename and
drops anything non-trainable regardless of what `dataset.json` declared (`read.py:339-344`).
`splits` is built earlier and straight from the declared partitions (`read.py:282-289`), so it never
gets that filter. Consequence for an adapter: **feed `r.paths` to the training dataset** — it is the
list that survives a lying manifest. Use `r.train` / `r.val` for the *shape* of the run (does an eval
callback exist? which URIs are the eval set?), and if the two disagree, treat that as a bug worth
failing on, not a preference.

**`rows` and `kwargs` are populated only on a split-specific read.** Both default empty
(`read.py:291-292`) and are assigned only inside the `split is not None` branch
(`read.py:309-310`). An unsplit read therefore always returns `rows=None` and `kwargs={}` — use
`r.split_rows` instead, which is always filled. The `kwargs` asymmetry matters: a `by: "field"` /
`"range"` / `"indices"` partition is a **row predicate, not a file subset**, so `_select` returns
*every* shard plus a `{"row_predicate": ...}` kwarg the loader is expected to apply
(`read.py:275-278`). On an unsplit read that predicate is silently dropped. If a dataset you consume
uses non-path partitions, read it with an explicit `split=` and honour `kwargs`, or you will train on
rows the split excludes.

**`byte_order` and `header_bytes` are part of the read instruction, not trivia.** See §3 — `dtype`
alone does not tell you how to decode the bytes.

### Groups

A dataset may hold several typed payload groups (`tokens/`, a sidecar group, …). `dataset_paths`
resolves exactly one:

```python
r = dataset_paths(ds_id, version, group="tokens", s3=s3)
```

With one group, `group=` is optional. With several and no `group=`, it **raises** and names them
(`read.py:242-252`) rather than guessing — the same reasoning as everywhere else here: an ambiguous
read that succeeds is worse than one that stops.

### Selecting a subset by label

Shards carry `entry.labels` — flat, string-valued, and **recomputed by Gate A from the object's
own key** (`validate._check_labels_match_path`), so a label cannot drift from the file it
describes. `labels=` narrows a read to the shards carrying every given key/value:

```python
r = dataset_paths(ds_id, version, labels={"source": "stack-edu"}, s3=s3)
r = dataset_paths(ds_id, version, labels={"source": "stack-edu", "domain": "Python"}, s3=s3)
```

The keys are **whatever the producer used** — `{source, domain}` for a pretrain corpus laid out
`tokens/<source>/<domain>/`, something else for another family. Nothing in the reader hardcodes a
key name.

Two behaviours worth knowing:

- **`rows` and `split_rows` are recomputed** from the selected entries' own counts. The
  partition's declared total describes a superset, and handing a trainer an inflated row count for
  data it did not select is the failure `partition-rows-mismatch` exists to catch on the write side.
- **Asking for labels on an unlabelled dataset raises.** A flat corpus, or any v1 manifest, carries
  no labels; returning `[]` would be indistinguishable from "your predicate missed", and you would
  train on nothing without noticing.

### Building a weighted mixture

`build_mixture` chooses a weighted, budgeted, **reproducible** subset:

```python
from edullm_data.read import build_mixture, MixtureSource

m = build_mixture(
    ds_id, version, s3=s3, seed=42, total=2_000_000_000,
    sources=[
        MixtureSource({"source": "stack-edu", "domain": "Python"}, 0.4),
        MixtureSource({"source": "finemath-3plus"},                0.4),
        MixtureSource({"source": "arxiv"}, 0.2, max_repetition_ratio=1.05),
    ],
)
m.paths            # feed these to the loader
m.numpy_dtype      # "<u4" — same rule as §3, pass it explicitly
m.actual_ratios    # what you GOT: {"source=arxiv": 0.214, …}
m.shortfall        # a component that could not reach its ratio, and by how much
```

**Whole shards, drawn in a seed-determined order.** The same `seed` always selects the same
shards, so `(dataset, version, sources, ratios, total, seed)` fully describes your training data —
six values in a run config instead of a list of 6,911 URIs. A different seed gives a different
sample of the same mixture.

This deliberately differs from OLMo-core's `SourceMixtureDatasetConfig`, which takes
`ceil(available * ratio)` from the **head of every path**: a 10% mixture there reads the first 10%
of every shard and never touches a tail, so any ordering inside a shard becomes a systematic skew.
(Its own `seed` field is declared, documented as controlling sampling, and never read.) The cost of
whole shards is that a budget lands *near* the target rather than on it — measured at ~2% over on
the 150B corpus.

Two knobs carried over from that config, both of which its real usage needed:

| knob | meaning |
| --- | --- |
| `max_repetition_ratio` | upsample by repeating shards, up to this multiple of the component's own size. A source smaller than its ratio demands cannot reach it otherwise. Default `1.0` = never reuse. |
| `max_source_fraction` | never consume more than this fraction of the component. A **hard cap**, not a target: unlike the budget, it will not overshoot by part of a shard. |

`shortfall` is the honest half. A component whose pool is too small under-delivers; without it
being reported you would ask for 5% wikipedia, get 3%, and never know.

**One dataset at a time, by construction.** Mixing two datasets could combine corpora tokenized
with *different* tokenizers whose vocab sizes are similar enough that every id still looks valid —
semantically wrong, and silent. Doing it safely needs a tokenizer-identity check across the
datasets' `depends_on` pins; that is not built, so the API does not offer the option.

---

## 3. THE dtype rule

**`r.dtype` is a string like `"uint32"`, read from the manifest, never inferred.**

It comes from `_resolve_format` (`read.py:362-394`), which collects the
`(dtype, byte_order, header_bytes)` **triple** from every fixed-width manifest entry and requires them
all to agree.

Three outcomes, and the differences matter:

- **All typed entries agree** → the triple. This is the normal case.
- **No entry carries a dtype** → `(None, None, 0)` (`read.py:382-384`). Legitimately untyped: parquet,
  jsonl, a tokenizer tree, a vendored directory. The container does its own typing.
- **Typed entries disagree** → **`MixedFormat` is raised** (`read.py:388-394`).

That last one is a deliberate change from returning `dtype=None`, and the reasoning is worth carrying
into your adapter (`read.py:57-77`): `ResolvedSplit` hands back **one** dtype because a loader memmaps
the group as one array, so *"these shards are uint16 and those are uint32"* has **no valid single
resolution**. A caller handed the softer answer must either raise itself or pick one — and picking one
is precisely the silent-corruption bug this module exists to prevent. Worse, `dtype=None` is
*indistinguishable* from the legitimate "container types itself" answer, so it flows into a loader and
gets defaulted. The recourse is structural and already in the standard: put differently-typed shards in
separate **groups** and pass `group=`.

`header_bytes` participates in the triple for the same reason: same dtype, different header sizes
cannot be read by one memmap stride either, and a disagreement there is the ".npy lie" shape — some
shards headerless, some not (`read.py:378-380`).

### Getting a numpy dtype out of it

The reader hands you *strings*, not `np.dtype`, on purpose: **`read.py` does not import numpy**, so the
package stays importable in a metadata-only environment (`read.py:84-85`, and `validate.py:782-794`
reimplements `numpy.min_scalar_type` by hand for the same reason). A CI job or a catalog browser can
`pip install edullm-data` and inspect datasets with no scientific stack.

**Use `r.numpy_dtype`, not `np.dtype(r.dtype)`** (`read.py:137-155`):

```python
import numpy as np
np_dtype = np.dtype(r.numpy_dtype)      # "<u4"  -> correct on ANY host
# NOT: np.dtype(r.dtype)                # "uint32" -> NATIVE order, correct only by luck
```

`np.dtype("uint32")` uses the **host's** byte order. A big-endian shard read on a little-endian host
(or the reverse) decodes every token to a different, **in-range-looking** id — the count is right, the
ids are plausible, the loss curve is merely bad (`read.py:96-101`). `numpy_dtype` combines dtype with
byte order into `"<u4"` / `">u4"`, which is unambiguous everywhere.

One numpy wrinkle the implementation exists to paper over: numpy accepts `"<u4"` but **rejects**
`"<uint32"` — the long names carry no order prefix (`read.py:27-38`, verified). Hence the
`_NUMPY_CHAR` map. If byte order was not declared, `numpy_dtype` falls back to the bare name, which is
honest: that dataset genuinely does not say, so native is the only available reading.

And **honour `header_bytes`.** It is `0` for every `.u32le.bin` shard, but a loader that hardcodes
offset 0 instead of reading the field will decode a header AS DATA the first time it meets a dataset
that has one (`read.py:103-107`).

Canonical dtype names only. The manifest's `DTYPE_SIZES` is an 8-entry map (`manifest.py:58-67`), and
Gate A rejects numpy aliases like `"u4"` / `"<u4"` **in a manifest** with `dtype-not-checkable`
(`validate.py:825`) because an alias nothing can size is indistinguishable from a lie. Note the
direction: aliases are rejected on the *write* side and produced on the *read* side, which is
consistent — the manifest must be unambiguous, the loader needs numpy's spelling.

### Why you must pass it explicitly, every time

**[CONSUMER]** OLMo-core's dataset classes default to `uint16`. Not one of them — all of them:
`numpy_dataset.py:131` (base), `:378`, `:495` (`NumpyFSLDataset`), `:706`, `:857`, `:1008`, `:1359`,
`:1939`. These corpora are `uint32`.

**[CONSUMER]** The config layer will *sometimes* rescue you, by luck. `NumpyDatasetConfig.get_dtype()`
(`numpy_dataset.py:2380-2394`) returns your explicit `dtype` if set; otherwise it walks
`uint8 → uint16 → uint32 → uint64` and picks the first width where
`tokenizer.vocab_size - 1 <= np.iinfo(dtype).max`. For dolma2, `vocab_size = 100278 > 65535`, so it
lands on `uint32` and logs `"Assuming dtype ... based on vocab size"`. That is the right answer for the
wrong reason: it is inferred from the *tokenizer config you happened to pass*, not from the bytes. Pass
a tokenizer config with a smaller vocab, or build a dataset class directly instead of through the
config, and the uint16 default is what you get.

### The asymmetry that makes this urgent

Both misreads are silent at the data layer — nothing in `numpy_dataset.py` range-checks token ids
against `vocab_size`; it is stored (`:141`, `:185-186`) and passed to the model, never asserted. So
whether a mistake is loud depends entirely on the *direction*, and the direction you fall into by
default is the quiet one:

**uint32 corpus read as uint16 — SILENT, and this is the default you get.** Verified exhaustively over
all 100,278 dolma2 ids: every uint32 id splits into two uint16 halves, and **every half is ≤ 65535,
hence every one is < 100278 — zero out-of-range values.** Ids below 65536 give
`[id, 0]`; ids in 65536–100277 give `[id & 0xFFFF, 1]`. Nothing raises. What you actually get is an
element count that has **doubled**, every second element a near-zero high half, and a token stream
that is structured noise. The run trains to completion and the loss curve looks plausible-but-bad —
the single most expensive failure mode in this document.

**uint16 corpus read as uint32 — LOUD.** Adjacent ids fuse into values like `4,294,901,761`, far above
any vocab. 32,767 of 32,768 fused values are out of range. The embedding lookup raises. You find out in
seconds.

So the dangerous direction is the one nobody chooses: you forget `dtype=`, OLMo-core's default is
`uint16`, and these corpora are `uint32`. **Always pass it explicitly.** The reader exists to make that
the path of least resistance (`read.py:3-7`); a gate nobody routes through is not a gate.

Note in passing: `read.py:5-6`'s own docstring says inferring dtype "silently halves the token count."
The *element* count doubles (each 4-byte token becomes two 2-byte elements). The comment's spirit is
right, its arithmetic is inverted.

---

## 4. Splits, and what you are allowed to train on

### `split=None` returns TRAINABLE data only

This changed, and the old behaviour is the bug (`read.py:312-316`). It used to return every entry, so a
caller who asked for nothing in particular got the held-out shards mixed into `paths` with no way to
tell which was which — train on your own validation set, silently, while `.val` looked populated.

Now: **silence means the safe subset, never "everything."**

```python
r = dataset_paths(ds_id, version, s3=s3)
r.paths          # trainable objects only
r.splits         # {"train": [...], "val": [...]} — BOTH, separately keyed
r.val            # the held-out URIs, for your eval callback
```

Both splits still come back, in `splits` / `train` / `val`, because a real run needs both — train for
the dataset config, val for the eval callback. They are kept **separate rather than concatenated into
`paths`** because a flat list is precisely the failure: a caller cannot tell the two apart, so held-out
shards end up in training with nothing to notice (`read.py:108-113`).

One exception, deliberate: a dataset that declares **no** trainable split at all — a tokenizer, a
vendored tree, an eval set — returns everything (`read.py:323-326`). There is nothing to protect and
the whole artifact is the payload. Those families opt out of the validation requirement in their own
family file with a stated reason (`families/eval.json:11-12`, `probe.json:33-34`,
`tokenizer.json:13-14`, `vendor.json:27-28`).

### The vocabulary is closed, and `held_out` is derived

```python
contracts.SPLITS            # frozenset({"train", "val", "test"})     contracts.py:138
contracts.TRAINABLE_SPLITS  # frozenset({"train"})                    contracts.py:142
contracts.is_trainable(s)   # s in TRAINABLE_SPLITS                   contracts.py:145-154
```

There is no `held_out` field anywhere. Held-out is **derived** as `not is_trainable(split)` — one fact,
one place. The enum is closed because before it, families disagreed (pretrain said `train`, sft said
`heldout`, probe said `test`, the 150B plan said `val`) and `sft-conversations/v1` had to *guess* which
side was held out by substring-matching the partition name — which classifies a partition named
`trainval` as HELD OUT because it contains "val", and rejects an ordinary `dev` outright
(`contracts.py:128-137`). A closed enum turns "is this trainable?" from a pattern guess into a set
lookup.

`is_trainable(None)` is **False** (`contracts.py:148-153`): an unlabelled object is *unknown*, and
treating unknown as safe-to-train is the exact failure the vocabulary exists to prevent. Callers that
legitimately read a whole unsplit artifact go through the "no trainable split declared" path above,
not through `None` meaning `train`.

### The escape hatch

```python
r = dataset_paths(ds_id, version, include_held_out=True, s3=s3)   # read.py:186, 319-322
```

This opts back into the old everything-in-`paths` behaviour, and it also skips the filename-recompute
filter (`read.py:328`). It is spelled out in full, as a named keyword, so that it shows up in a code
review of a training config. If you find it in a pretraining run, that is a finding.

### Asking for a split the dataset lacks

```python
r = dataset_paths(ds_id, version, split="val", s3=s3)
# no val partition, but "val" IS in SPLITS  ->  EMPTY result, no exception   read.py:299-304
#   r.paths == []          r.rows is None
#   r.splits still lists every split that DOES exist

dataset_paths(ds_id, version, split="dev", s3=s3)
# "dev" is outside SPLITS  ->  ReadError                                     read.py:305-308
```

The distinction is the point (`read.py:296-298`). *"Does this dataset have validation data?"* is a
question, and answering a question must not require catching an exception. *"Give me the `dev`
split"* is a mistake — `dev` is not a word in this vocabulary — and mistakes get reported.

### Why the reader distrusts the manifest here

For an unsplit read without `include_held_out`, after selecting by declared partition name the reader
runs one more pass that **recomputes the split from each object's own filename** and drops anything
parsing to a non-trainable word, whatever the declaration said (`read.py:328-344`):

```python
selected = [
    e for e in selected
    if (parsed := parse_shard_name(e.path)) is None
    or parsed[0] not in SPLITS
    or is_trainable(parsed[0])
]
```

Everything above that line reasons from declared partition *names* — a claim in `dataset.json`. Every
way that claim can be wrong ends with held-out data inside the trainable set: a val-only partition, an
empty or malformed `partitions` list, a partition with no name, a `by: field` partition that selects
every shard, a group whose val shards nobody declared. So the standard's own naming rule
(`<split>-<NNNNN>.<ext>`, `manifest.py:661-672`) is applied to the read path as a backstop. Names that
do not parse as shards are **kept**, so tokenizer files and vendored trees are unaffected.

This is why §2 says feed `r.paths`: it is the only list that has been through this.

---

## 5. The seal — verified on every read

`dataset_paths` does two things before it will hand you a path.

**It refuses an unvalidated prefix.** No `_VALIDATED.json` (or a legacy `_SUCCESS`) means the dataset
is not readable — `NotValidated` (`read.py:164-174`). With the airlock, unvalidated bytes cannot be in
`edullm-data` at all, so this is belt-and-suspenders for the case where somebody points the reader at a
landing prefix or a hand-assembled directory (`read.py:9-12`). `require_validated=False` exists
(`read.py:184`) for that debugging case; do not ship it.

**It recomputes the seal and raises `SealMismatch` if the bytes disagree** (`read.py:209-231`). Not
"observes that a marker exists" — recomputes. `SealMismatch` is deliberately a *different* exception
from `NotValidated` (`read.py:49-54`): absence is ambiguous, but a marker that is present *and
disagrees* means something changed a frozen dataset.

The specific attack this closes: a rewritten `dataset.json` whose **train and val globs have been
swapped**. The marker is present, every group manifest is byte-intact, and `split="train"` hands back
the val shards. Cost of the check: two small GETs per group, no payload bytes.

Standalone form, for an fsck sweep or a pre-flight:

```python
from edullm_data.read import verify_seal
problems = verify_seal(dataset_id, version, s3=s3)   # read.py:418-494
if problems:
    for p in problems:
        print(p)
```

It returns a list of human-readable mismatches rather than raising, so all of them are reported at
once (`read.py:437-438`). Empty list = intact. It walks the chain the way a verifier should
(`read.py:427-431`): recompute `sha256(dataset.json)` and compare to the seal's `dataset_sha256` root;
then per group recompute `sha256(manifest.json)` and compare to **both** the seal's copy and
`dataset.json`'s own copy. It raises `NotValidated` if there is no seal at all (`read.py:450`).

### One live dataset is verifiable; the other is not

**`pretrain/olmo-150b-dolma2/v1` verifies CLEAN** — it was promoted by the rooting code, so its seal
carries `dataset_sha256`, a per-group `manifest_sha256` map, and a CRC64NVME reference for each of
its 6,911 objects. `verify_seal` returns no problems (checked against the live bucket).

**`tokenizer/dolma2-bpe/v1` is still pre-root**, and the rest of this section is about that case.
It stays that way until it is republished — `promote()` refuses a sealed prefix, so that means a new
version rather than a rewrite of `v1`.

A seal written before the chain had a root carries no `dataset_sha256`. `verify_seal` reports that as a
problem string — *"seal carries no dataset_sha256 — written before the chain was rooted, so it cannot
be verified"* (`read.py:457-465`) — rather than passing silently, because an unverifiable seal is a
different state from a verified one. But it is **allowed through**: refusing would make every
already-published dataset unreadable, which is the retroactive invalidation the standard forbids.

`dataset_paths` filters exactly that one string out before deciding whether to raise
(`read.py:224-225`):

```python
problems = [p for p in verify_seal(...) if "no dataset_sha256" not in p]
```

**Consequence for an adapter: `dataset_paths` succeeding does NOT mean the seal was verified.** It
means the seal was verified *or* was unrootable. If you want to know which, call `verify_seal`
yourself and inspect the strings.

Both live datasets are in that state right now — confirmed against the live bucket:

```
pretrain/olmo-mix-1124-31b/v1/_VALIDATED.json
  {"bytes":125336003336,"dataset_id":"...","objects":218,"validated_at":"...","version":"v1"}
tokenizer/dolma2-bpe/v1/_VALIDATED.json
  {"bytes":6769971,"dataset_id":"...","objects":5,"validated_at":"...","version":"v1"}
```

No `dataset_sha256`, no `manifest_sha256` map. They were promoted before `promote()` rooted the chain
(`validate.py:1203`, `:1251-1260`). They stay unverifiable until re-promoted, and because
`promote()` now refuses an already-sealed prefix (`validate.py:1143-1151`), that means a new version —
not a rewrite of `v1`.

---

## 6. Constraints a trainer must respect

Four of these are **[CONSUMER]** facts about OLMo-core. This repo cannot enforce any of them; they are
here because getting them wrong wastes a training run.

### `work_dir` must be a local path, never a URL — **[CONSUMER]**

Two independent guards, both raising `OLMoConfigurationError`:

- `NumpyDatasetBase.work_dir` setter: `if is_url(work_dir): raise` (`numpy_dataset.py:242-245`)
- `Trainer.__post_init__`: `if is_url(self.work_dir): raise` (`trainer.py:326-331`)

`work_dir` is where the dataset caches derived index files it later `is_file()`-checks and memory-maps
(`numpy_dataset.py:2095`, `:2106`). An S3 URL cannot answer `is_file()`. Point it at node-local scratch,
shared across local ranks (`numpy_dataset.py:116`).

### `save_folder` needs ≥3 path levels below the bucket — **[CONSUMER]**

The mechanism, so you can reason about your own value rather than memorising a number:

- checkpoints are written to `join_path(save_folder, "step{N}")` (`trainer.py:967-968`, dirname from
  `checkpoint.py:85`, `:291-292`)
- `Checkpointer._prepare_dir` calls `clear_directory(dir)` on that path when `save_overwrite=True`
  (`checkpoint.py:397-399`)
- `clear_directory` refuses a remote prefix "too close to the root of a bucket":
  `if not force and prefix.count("/") < 2: raise ValueError` (`io.py:389-393`)

So the *checkpoint* directory needs at least three segments below the bucket, which means
`save_folder` itself needs at least two:

```
s3://my-bucket/run                     -> step dir "run/step1000",  count("/")==1  -> ValueError
s3://my-bucket/checkpoints/run         -> "checkpoints/run/step1000", count("/")==2 -> OK
```

It is a safety rail against a config typo wiping a bucket root, and it only fires on the
`save_overwrite` path — which is exactly the path you hit when you resume a crashed run under time
pressure. Give `save_folder` three levels and stop thinking about it.

### Shards are headerless uint32 LE — do not expect a `.npy` header

`tokens × dtype_size == file bytes`, exactly, with no header. The extension says so: `.u32le.bin`,
never `.npy`. The legacy files *were* `.npy`-named headerless raw uint32 — the ".npy lie" — and the
standard's rule is that the extension must match the real bytes (`docs/DECISIONS.md:87-96`). Gate A
recomputes the arithmetic per entry (`validate.py:415`) and checks extension against declared
format (`:417`). The reader also carries `header_bytes` explicitly (§3) so a loader never has to
assume the offset.

**[CONSUMER]** OLMo-core reads these correctly by construction: `load_array_slice` computes
`bytes_start = start_idx * item_size` and `np.frombuffer`s the result (`utils.py:297-315`) — byte 0 is
token 0. A real `.npy` header would corrupt both the leading tokens and the size-derived count. Do not
"helpfully" add one, and do not point `np.load` at these files.

### Use plain `NumpyFSLDatasetConfig`. The other five classes hard-fail on these shards — **[CONSUMER]**

This is the sharpest edge in the document. Five of OLMo-core's six numpy dataset classes need
**per-document boundaries**, and when the shards are remote they get them from a per-shard `.csv.gz`
sidecar whose name is derived like this (`utils.py:217`):

```python
metadata_filename = os.path.basename(data_path).replace(".npy", ".csv.gz")
```

On `train-00000.u32le.bin` that `.replace` is a **no-op**. The "sidecar" name resolves to the shard's
own name, `resource_path` happily returns the shard itself, and `gzip.open` is then handed 125 GB of
raw uint32 — `BadGzipFile`.

Which classes reach that code, verified by tracing `prepare()`:

| class | `prepare()` | reaches `iter_document_indices`? |
| --- | --- | --- |
| `NumpyFSLDataset` | `len(self)` only (`:576-577`) | **No — safe** |
| `NumpyFSLDatasetMixture` | `_write_document_indices` (`:749-752`) → `segment_documents_into_instances` | Yes |
| `NumpyPaddedFSLDataset` | `_write_instance_indices` (`:913-917`) → `segment_documents_into_instances` | Yes |
| `NumpyPackedFSLDataset` | `_pack_all_documents_into_instances` (`:1114-1118`) → `pack_documents_into_instances` | Yes |
| `NumpyInterleavedFSLDataset` | `_write_instance_indices` + interleaving (`:1431-1437`) | Yes |
| `NumpyVSLDataset` | `_write_document_indices` (`:2035-2041`) → `bucket_documents` | Yes |

Plain `NumpyFSLDataset` never needs boundaries: it concatenates and chunks into fixed windows, and its
optional `doc_lens` output is computed from the in-memory tensor via `get_document_lengths`
(`:618-621`), not from a sidecar.

The precise nuance — and it matters, because it means "hard-fail" is conditional. `iter_document_indices`
takes the sidecar branch only when the path is a URL (`utils.py:193-197`): given `eos_token_id` **and**
`dtype` **and** a non-URL path, it memory-maps the array and finds EOS positions itself. Nothing in
`numpy_dataset.py` passes `local_cache`, so reading straight from `s3://` — which is the entire point
here — always lands in the sidecar branch and dies. Stage the shards to local disk first and the
mmap path opens up. That is not the deployment we have.

So: **use plain `NumpyFSLDatasetConfig`** (`numpy_dataset.py:2522`) with an explicit
`dtype=` from `r.dtype`. If you genuinely need document boundaries, that is a producer-side change —
publish real `.csv.gz` sidecars as their own manifest group — not something to hack around in the
adapter. The 150B migration explicitly **excluded** the 6,915 legacy `.csv.gz` files as non-payload,
so they do not exist in `edullm-data` today — recorded in the published dataset's own
`limitations` under `kind: "no-document-boundaries"`, which is the durable statement of it.

One correction worth keeping, because an audit of this file got it backwards: the **composable**
stack (`NumpyDocumentSource` → `ConcatAndChunkInstanceSource` → `MixingInstanceSource`) is NOT
exempt. `NumpyDocumentSource.get_document_offsets` calls `iter_document_indices`
*unconditionally* (`composable/numpy_document_source.py:557-576`), so on an `s3://` path it takes
the same sidecar branch and fails the same way — verified by execution, `403` on the derived
`.csv.gz` key. It only appears to work if you never call that method. The local-staging escape
hatch applies to it too.

Note also that `NumpyFSLDatasetConfig` reaches `NumpyFSLDatasetMixture` whenever
`source_mixture_config` is set (`numpy_dataset.py:2580-2599`). "Use the plain config" means the plain
config *without* a source mixture.

---

## 7. What is NOT guaranteed

Honesty here is load-bearing: an adapter author who over-trusts the seal will skip a check they should
have written.

**Gate A does not re-hash payload bytes.** `sha256` in a manifest is an **unfalsified producer
assertion**. `s3.hash_object`'s only non-definition caller is `publish.py:280` — the producer, at
publish time. The validator's per-entry loop (`validate.py:400-435`) HEADs for **size**, checks the
count arithmetic, checks extension-vs-format and shard naming, and does *set membership* on the
**declared** digest to catch duplicates. It never reads the payload to recompute a digest. `fsck.py:32`
reads "LIST and HEAD, never a payload byte" on purpose.

So what is `sha256` actually for? Two real jobs:

1. **Content addressing** — `duplicate-shard-digest` (a byte-identical shard listed twice) and
   `shared-sha-with-parent` (a child re-materializing a parent's bytes instead of referencing them,
   the 37 GB duplication from the audit).
2. **The hash chain** — `manifest_sha256` and `dataset_sha256`, which *are* recomputed, by
   `verify_seal` (§5).

**What actually defends the bytes:** the airlock's IAM Deny (only the validator role can `PutObject` to
`edullm-data`, and in policy v2 *nobody* can `DeleteObject`), S3's own durability, and **CRC64NVME** —
`s3.head()` returns `crc64nvme` and deliberately omits `sha256`, because S3 stores no whole-object
SHA-256 for a multipart object (`s3.py:47-49`, `:144`). Gate B (`wu-fsck`) is the sweep that re-checks
presence, size, and CRC against the manifest's claims after publication; a byte replacement at
identical length is exactly what CRC catches and nothing else here does.

A full re-hash is affordable (~16.5 min / ~$0.18 for 758 GB) and is an **open decision**, not
something the pipeline does. Do not document it as existing. `docs/ONBOARDING.md:45-51` states this
same limitation in the newcomer's register; the two documents must not drift.

**Also not guaranteed:**

- **`resolve_latest` can name a version whose data is gone.** It reads `_catalog/` only
  (`read.py:404-415`). A stale catalog entry resolves fine and then `dataset_paths` fails on the
  missing `_VALIDATED.json` or `dataset.json`. Handle `ReadError`.
- **`entry.split` is currently always absent.** The v2 field exists and is validated
  (`manifest.py:215`, `:267-276`), and Gate A checks a declared value against the filename
  (`split-contradicts-filename`, `validate.py:773`) — but `publish()` never populates it
  (`publish.py:281-287` constructs `ManifestEntry` with no `split=`). Split information today comes
  from partitions and filenames. Do not build an adapter that requires `entry.split`.
- **`rows` is a declared number, recomputed against per-entry counts, not against payload bytes.**
  Gate A sums the entries a partition selects and compares (`partition-rows-mismatch`,
  `validate.py:973`) — and those per-entry counts were themselves derived from object sizes. It is
  arithmetic all the way down, which is sound for fixed-width shards and says nothing about content.
- **A `MixedFormat` group cannot be read at all.** Not a limitation so much as a design choice
  (§3), but an adapter must catch it: the fix is `group=`, on the producer's side or yours.
- **Nothing range-checks token ids at read time.** Gate A samples ~64 KB per shard at publish time and
  bounds ids against the tokenizer's derived vocab; the reader reads no payload at all. If you want a
  runtime assertion that `max(id) < vocab_size`, write it in the adapter.

---

## The five-line version

```python
import numpy as np
from edullm_data.read import dataset_paths, resolve_latest
from edullm_data.s3 import Boto3S3

s3 = Boto3S3.default()
ver = resolve_latest("pretrain/olmo-mix-1124-31b", s3=s3)
r = dataset_paths("pretrain/olmo-mix-1124-31b", ver, s3=s3)   # trainable only; seal verified

# r.paths         -> train URIs, hardened against a lying manifest
# r.numpy_dtype   -> "<u4"  — PASS THIS. The consumer defaults to uint16 and fails SILENTLY.
# r.header_bytes  -> 0      — honour it anyway; do not hardcode the offset
# r.val           -> held-out URIs, or None if this dataset has none
dtype = np.dtype(r.numpy_dtype)
```

Then, consumer side: plain `NumpyFSLDatasetConfig` (no `source_mixture_config`), explicit `dtype=`,
local `work_dir`, `save_folder` three levels deep.

Cross-references: `docs/ONBOARDING.md` (the pipeline, for a newcomer) · `docs/DECISIONS.md` (why each
rule exists) · `USAGE.md` (the producer side) · `CONTRIBUTING.md` (the golden rule, if you are adding
a check).
