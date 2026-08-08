# Designing a dataset — decide these *before* you collect

Every other doc here starts after your bytes exist. `USAGE.md` tells you how to publish a
finished dataset; `CONTRIBUTING.md` tells you what to do when no profile fits. This one is for
the moment before that: **you are about to build something, and some choices are cheap now and
expensive-to-impossible later.**

Read this once at design time. It takes about ten minutes and it is ordered by *cost of getting
it wrong*, not by the order you'll do the work.

> Working with a coding agent? Say **"help me design an eduLLM dataset"** and the
> `edullm-dataset-design` skill walks you through this as an interview and emits a filled-in
> spec plus the exact `publish()` call. This doc is the same content, for reading.

---

## The one-paragraph version

Pick a **family** (fixed list of 7) and a **profile** (what your bytes *are*). The profile
decides what metadata you must record **while you generate the data** — some of it cannot be
reconstructed afterwards at any price. Name it `<family>/<name>` where name is kebab-case, 2–5
words, no dates and no version words. Lay it out as `<group>/<split>-<NNNNN><ext>`. Publish the
**tokenizer first** if you have one. Then `publish()` and the validator re-derives everything
you claimed.

---

## Tier 0 — irreversible. Getting these wrong means re-generating the data.

These are the only ones worth losing sleep over. Everything in Tier 0 is baked into
`manifest_sha256`, or is information that simply stops existing once your job exits.

### 0.1 Put your slice structure in the object key — flattening is irreversible

If you will *ever* want to ask "train only on the arxiv portion" or "how much of this is code?",
that structure has to be **in the path**, decided before you write:

```
tokens/arxiv/science/train-00000.u32le.bin      ← keeps source + domain
tokens/train-00000.u32le.bin                    ← flattened; the information is gone
```

Nesting under the group is allowed today — `check_shard_naming` accepts it, and the segments are
preserved verbatim in `entry.path`, which is inside the manifest hash. Flattening throws the
slice away at write time, and `entry.path` is not editable afterwards: changing it changes
`manifest_sha256`, which is the dataset's identity, which means republishing and **re-copying
every payload byte**. On a 630 GB corpus that is hours of transfer to recover information you
had for free.

> Not hypothetical — this is exactly where the 150B corpus ended up. Its mapping was recoverable
> only by luck (sorted order happened to be bijective), which is not a plan.

**Two levels, `<source>/<domain>` — that is the whole budget.** `publish()` derives
`entry.labels` from exactly those segments, and exactly two of them are named
(`PATH_LABEL_KEYS = ("source", "domain")`):

```
tokens/train-00000.u32le.bin                 →  {}                                  (flat, fine)
tokens/arxiv/train-00000.u32le.bin           →  {"source": "arxiv"}
tokens/arxiv/science/train-00000.u32le.bin   →  {"source": "arxiv", "domain": "science"}
tokens/a/b/c/train-00000.u32le.bin           →  raises — 3 levels, only 2 are named
```

Three levels is a hard error, not a silent `level_3` — deliberately, because labels land in
`manifest_sha256` and a wrong one is unfixable without republishing.

**You do not pass labels to `publish()`.** There is no `labels` parameter and there is not meant
to be one: a hand-typed label would be a producer assertion nothing falsifies, which the golden
rule forbids. `publish()` reads them off the key, and Gate A **recomputes** them from that same
key (`_check_labels_match_path`), rejecting a label that contradicts the path or a nested entry
whose labels were omitted. So the path is the single source of truth, and the only thing you
control is how you name your files.

Absence is treated one-directionally: a flat layout produces no labels and that is silent and
legal. What is rejected is a *nested* key whose labels don't match — because a reader slicing on
labels would drop that object from every slice.

**Labels ARE a reader-level selector.** `dataset_paths(..., labels={"source": "arxiv"})` narrows a
read to the shards carrying those keys, and `build_mixture(...)` uses the same predicate to
assemble a weighted, seeded subset — so how you nest your tree at publish time decides what a
trainer can ask for later:

```python
r = dataset_paths(ds, ver, labels={"source": "arxiv"}, s3=s3)
m = build_mixture(ds, ver, s3=s3, seed=42, total=2_000_000_000, sources=[
    MixtureSource({"source": "arxiv"}, 0.3), MixtureSource({"source": "stack-edu"}, 0.7)])
```

**The one real caveat:** `partitions[]` still cannot express "select by label" — the empty-split
check fires regardless of the `by` field, so a label-named partition is rejected. Selection is a
*read-side* concern, not a declared partition. That distinction costs you nothing as a producer;
it just means the slice lives in the key and the manifest, not in `dataset.json`.

This is exactly why the nesting decision is expensive to get wrong: a flat tree yields no labels,
and there is then nothing for either API to select on.

### 0.2 Held-out data must come from a different place than train

The validator checks train/heldout leakage, and for `sft-conversations/v1` it will reject
overlap outright. But its reach is limited: it catches duplicates by **content digest, within a
group**. If you carve a val split by copying shards that also appear in train, that is 100%
leakage and it may or may not be caught depending on how your groups are arranged.

**Decide at collection time where heldout comes from** — hold out documents *before* tokenizing,
not shards after. A val split sampled from the same shuffled pool as train is not a val split.

### 0.3 Eval runs: record per-row status *during* the run

`eval-results/v1` requires `status_counts` and refuses a file where `n_ok == 0`. This exists
because the audit found 12 CSVs of 66 bytes (header, zero rows) and 3 byte-identical files where
every row read `Finish Reason: 'error'`.

A row's status — did the model answer, error, or get filtered — **exists only while the eval is
running**. You cannot reconstruct it from the output file afterwards. Write a `status` field on
every row as you go, and count them.

Also required per group: `model` (`id` + `revision`), `task`, and `decode`
(`temperature`, `top_p`, `max_tokens`). The decode params must be the ones that *actually*
produced the output — capture them from your config object, don't retype them later.

### 0.4 Anything you'd want in `sources[]`, measure while you mix

The README renders a data-mix table from `sources=[...]`. If you want it to state *this
dataset's* real per-source token counts, count them during the mix. Afterwards, all you can
honestly do is cite upstream's totals with `scope: "upstream-full-collection"`, which prints a
caveat saying these are not measured for this subset. That is fine and honest — but it is a
weaker artifact than the real numbers you could have counted for free.

---

## Tier 1 — expensive. Fixable only by publishing a `v2`.

Frozen means frozen: a published `vN` is never edited to change data. These are all cheap to get
right up front and require a full republish to change.

### 1.1 The name

`<family>/<name>`. The name is **mechanically enforced** (`contracts.py:validate_name`) and the
user also reviews it by hand. Rules, all of which reject at publish time:

| Rule | Rejected | Fine |
|---|---|---|
| kebab-case, `[a-z0-9]` only | `dolma2_150B` | `dolma2-150b` |
| 2–5 words | `corpus`, `a-b-c-d-e-f` | `dolma2-150b` |
| no dates (YYYYMMDD, 19xx/20xx years, month tokens) | `mix-jul22`, `run-2026` | `olmo-mix-1124-31b` ¹ |
| no version tokens | `final-v2`, `latest`, `fixed` | — version is a separate segment |
| no content-free words | `data`, `results`, `test`, `tmp`, `misc` | say what it actually is |
| no relative words | `big`, `improved`, `better`, `fast` | `dolma2-150b` (state the axis) |
| no bare ordinals | `experiment-3` | `4hop`, `370m`, `5b5` |

¹ `1124` survives because a 4-digit run that isn't `19xx`/`20xx` is read as an upstream release
code, not a year.

**The test that matters:** a name must state *what the data is* **plus** *the one axis that
distinguishes it from its siblings*. `pretrain/dolma2-150b` — corpus plus token budget.
`curriculum/flesch-linear-370m` — ordering method plus model scale.
`eval/mcq-arc-openbookqa-sciq` — task type plus exactly which benchmarks.

Two rules are **not** mechanically enforced because they can't be: no person names, no ticket
IDs. Review catches those. (`eric-test` happens to die on the `test` ban.)

### 1.2 Family and profile

Family is a fixed list of 7. Profile is what your bytes actually *are* — this is one of the four
arguments you hand-type, and the one that must not be guessed.

| Your data | Family | Profile | Status |
|---|---|---|---|
| packed token shards for pretraining | `pretrain` | `pretrain-tokens/v1` | ✅ shipped |
| an ordering / index vector over a parent token pool | `curriculum` | `token-order/v1` | ✅ shipped |
| instruction-tuning conversations | `sft` | `sft-conversations/v1` | ✅ shipped |
| model outputs + scores from an eval run | `eval` | `eval-results/v1` | ✅ shipped |
| a tokenizer (`tokenizer.json` + friends) | `tokenizer` | `tokenizer/v1` | ✅ shipped |
| benchmark **items** (the questions, not the answers) | `eval` / `probe` | `eval-items/v1` | ⚠️ **not implemented** |
| a vendored upstream tree, verbatim | `vendor` | `vendored/v1` | ⚠️ **not implemented** |

> **⚠️ Read this before you plan around `probe` or `vendor`.** Those two families default to
> profiles that **do not exist in the registry** (`registry.py:_SHIPPED` ships five). Publishing
> into them today fails with `ProfileError`. `experimental/v1` is also referenced in
> `CONTRIBUTING.md` and is likewise unregistered. If your data is one of those shapes, you are
> writing the profile as part of your work — budget for it (see §1.3) and talk to Eric first.

**If nothing fits, write a profile.** That is the expected path, not an exception. Four things: a
registry entry, a schema fragment, check functions, and two fixtures (one passing, one
deliberately broken). See `CONTRIBUTING.md`.

### 1.3 The metadata your profile forces you to produce

Look this up *now*, because it determines what your generation job must emit:

- **`pretrain-tokens/v1`** — every manifest entry needs a token `count`. Your corpus must
  `depends_on` a published tokenizer; the validator derives `vocab_size`/`eos_token_id` from it
  and asserts every sampled id against it. Tunable bounds (`min_distinct_ids`,
  `max_eos_fraction`, `max_zero_fraction`) can be *tightened* per group but not loosened — that
  requires editing the family file, where it's visible.
- **`sft-conversations/v1`** — rows are `{messages: [{role, content}]}`. Needs **≥2 partitions**
  (one held out), plus `dedup` and `leakage` blocks, both recomputed by checks.
- **`eval-results/v1`** — `model{id,revision}`, `task`, `decode{temperature,top_p,max_tokens}`,
  `status_counts`. See §0.3.
- **`token-order/v1`** — `depends_on` the parent pool (≥1, pinned by content). Default ordering
  is `permutation`, and the validator checks it really is one.
- **`tokenizer/v1`** — requires *nothing* hand-typed on purpose. `vocab_size` and `eos_token_id`
  are derived from `tokenizer.json`; asserting them by hand would reintroduce the guess.

### 1.4 Layout, shard naming, and dtype

```
artifacts/public/
└── tokens/                      ← group name; its first path segment picks the profile
    ├── arxiv/science/           ← optional: <source>/<domain>, see §0.1 — decide NOW
    │   ├── train-00000.u32le.bin
    │   └── val-00000.u32le.bin
    └── code/python/
        └── train-00000.u32le.bin
```

- **One subdirectory per group.** A file not under a group prefix is a hard publish error —
  the group is what makes the profile unambiguous. A dataset may have several groups with
  different profiles (e.g. `tokens/` + a sidecar table).
- **Between the group and the filename you may nest at most two levels** (`<source>/<domain>`);
  they become `entry.labels`, and a third level is a hard error. Flat is fine if the corpus
  genuinely has no slices. See §0.1 — this is the irreversible one.
- **Splits are matched by glob on the filename** (`train-*.u32le.bin`). The split lives in the
  *name*. Get the prefix right at write time or the partition comes up empty.
- **No `-of-NNNNN` in shard names.** The surviving shard count is unknowable at write time
  because filtering hasn't run yet; completeness is proven by path-set equality instead.
- **Token shards are `.u32le.bin`, never `.npy`.** The legacy `.npy` files were headerless raw
  uint32 — the extension lied. OLMo-core memmaps from byte 0 and derives token count from raw
  file size, so a real 128-byte `.npy` header silently corrupts both the leading tokens *and* the
  count. `tokens × dtype_size == file bytes`, exactly.
- **dtype is declared, never inferred.** OLMo-core defaults to `uint16`; these corpora are
  `uint32`. Inferring halves your token count. `dataset_paths()` returns the declared dtype —
  feed it to your loader.

### 1.5 Publish the tokenizer *first*

A pretrain or curriculum corpus names the tokenizer it was built with
(`tokenizer="tokenizer/<name>"`). That tokenizer must **already be published** — the validator
resolves it, pins it by `manifest_sha256`, and derives the vocab from its real `tokenizer.json`.

So the order is: publish the tokenizer → publish the corpus. If you're planning a new
tokenization run, that's two publishes, and the first one gates the second.

Do **not** set a family-wide tokenizer default. A mismatched tokenizer is nearly undetectable —
vocab sizes cluster around 100k, so wrong ids still land in range and pass the decode check while
being semantically garbage.

### 1.6 Shard size

Not validator-enforced, so use judgement: the shipped corpora run ~500 MB–1 GB per shard (218
shards / ~125 GB). Very many tiny shards make manifests and hashing slow; very few huge ones hurt
loader parallelism. Also note two 20-byte shards once tripped *both* the duplicate-digest and
eos-fraction checks — degenerate tiny shards are a real failure mode.

---

## Tier 2 — cheap. Backfillable in place, so don't block on them.

The generated `README.md` is a **control file**, not a manifest entry, and never enters the hash
chain. So all of its descriptive inputs can be added to an already-frozen dataset without
touching a payload byte:

`about` (the one curated prose block), `sources[]`, `license`, `notes`, `limitations[]`.

Write them if you have them — they're what makes the dataset legible to the next person — but a
missing `about` is never a reason to delay a publish. A section with no data is **omitted**, never
faked.

`purpose` is the exception: it's a hand-typed `publish()` argument, so decide it now. Good ones
state **what it is, what uses it, and what it decides** — "ARC/OpenBookQA/SciQ loglikelihood
scores for 23 baseline models, for the baseline table."

---

## Before you write a single byte — the checklist

```
[ ] Family + profile chosen, and the profile is actually REGISTERED
    (pretrain-tokens, token-order, sft-conversations, eval-results, tokenizer)
[ ] Name passes: kebab, 2-5 words, no date/version/generic/relative/ordinal
    -> states WHAT IT IS + THE AXIS vs its siblings
[ ] I know every field my profile requires, and my generation job emits all of them
[ ] Slice structure is IN THE PATH: <group>/<source>/<domain>/<shard>   <- UNBACKFILLABLE
    (becomes entry.labels; flat is fine; 3 levels is a hard error)
[ ] Heldout is carved from a different pool than train, before tokenizing
[ ] Eval only: every row carries a status; counts accumulate during the run
[ ] Tokenizer is already published (or is the first of my two publishes)
[ ] Layout is <group>/<split>-<NNNNN><ext>, split in the filename, no -of-NNNNN
[ ] Token shards are .u32le.bin, dtype declared, tokens x 4 == bytes
[ ] purpose written: what it is, what uses it, what it decides
```

## Then

You write to `s3://edullm-landing` (14-day expiry, anyone may write). The validator re-checks
every byte and, only if it passes, copies into `s3://edullm-data` — which you physically cannot
write to. If it fails, read the `_REJECTED.json` it drops next to your upload; each violation
names the concrete failure.

Publishing is deliberately cheap — four hand-typed arguments and a couple of seconds of hashing —
because if it isn't, people go back to dumping things in scratch. If it feels expensive, that's a
bug in this pipeline; tell Eric.

**Next:** [`USAGE.md`](../USAGE.md) for the actual `publish()` call ·
[`CONTRIBUTING.md`](../CONTRIBUTING.md) to write a profile ·
[`ONBOARDING.md`](ONBOARDING.md) for the 2-minute mental model
