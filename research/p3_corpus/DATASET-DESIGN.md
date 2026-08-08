# Dataset design: pretrain/formal-proof-premises-500m

purpose:  Formal-mathematics proofs paired with the named premises each one cites, as packed
          token shards, to train dense and split fact-masking arms and test whether a model
          can maintain mathematical reasoning while not learning the supplied facts.
family:   pretrain
profile:  pretrain-tokens/v1   [verified in registry: yes]
name:     formal-proof-premises-500m   [validate_dataset_id: PASS]
          Published v2 historical count: 494,862,336 train Qwen tokens plus
          11,862,016 validation tokens. Recompute from repaired bytes before
          publishing the next version.

The current publication gate is checked against exact
`edullm-data@38bf831a6c3f445e394784018441fd59288b876c`, package 0.5.0, rather than
inferred from the skill text.

## Irreversible decisions

**slice path:** `tokens/<corpus>/<split>-<NNNNN>.u32le.bin`

One level of nesting, naming the corpus. Six values: `metamath`, `prf2`, `enigma`, `mizar`,
`thproofs`, `isabelle`. Chosen because the corpora differ enough that measuring one alone is
likely — `isabelle` targets average 15 tokens against `prf2`'s 2,965, and `metamath` is the only
shard whose 8-gram novelty is below 60%. Flattening would put that distinction beyond reach
without re-copying every byte, since `entry.path` is serialized into the manifest and hashed into
`manifest_sha256`.

`check_shard_naming("tokens/metamath/train-00000.u32le.bin")` returns no violations [verified].

**heldout source:** a different pool, carved before tokenizing [verified in the corpus].

Held-out is drawn per *family of shards sharing a fact namespace*, not per shard: `mizar`+`thproofs`
both cite the MML and `prf2`+`enigma` both cite MPTP, so a fact withheld from one was originally
being trained on by its sibling — the first cross-shard sweep caught 415 such facts. 2,000 facts
are held out across four families. Every example citing one, and each held-out fact's own proof
(its statement is the goal of its own proof), is in `corpus/eval/`. `verify_corpus.py` re-checks
both directions over all 262,253 source examples.

**loss mask: derived at read time, not published.**

The split arm needs to know where each example's fact block ends. That is NOT shipped as a second
group, for two reasons. First, mechanically: `weights-sidecar/v1` is **not in the v0.2.0 registry**
[verified] — publishing into it raises `ProfileError`. Second, and the better reason: the boundary
is already in the token stream, so the loader can recompute it. A recomputed mask is falsifiable;
a published one is a producer assertion nothing checks, which the golden rule forbids.

**The searched run is `---\nGOAL`, not the separator itself** [measured over all 258,316
documents]. The corpus contains `\n---\nGOAL `, but BPE does not respect its edges: the trailing
space merges rightward into the goal's first word (` |-`, ` ![`, ` lemma`) in 98.4% of documents,
and the leading newline merges leftward into the fact block's last characters (` )\n`, `"\n`) in
88.5%. The full string encodes to `[198, 10952, 15513, 969, 220]`, a run present in 777 of 258,316
documents — 0.30%, and **0% in metamath, prf2, enigma and isabelle**. Searching for it would have
left the split arm unable to find any boundary, supervising every token, and reporting a plausible
loss curve while being a second dense arm.

The three-token core `[10952, 15513, 969]` is present in 258,316 of 258,316, never twice, and the
token after it always begins at or past `mask_end`. `tokenize_corpus.py` now probes real documents
and refuses to write shards if the run is ever missing or repeated.

A related fact worth knowing: because of the leftward merge, `mask_end` does not land on a token
boundary in 88.5% of documents — the fact block's last token also carries the separator's newline.
That is harmless as long as the straddling token counts as fact block, which it does, but any code
assuming a clean character-to-token boundary is wrong for most of the corpus.

This makes the mask a property of the bytes rather than a parallel artifact that can drift out of
alignment with them — the failure `mask_alignment_test.py` exists to catch.

## Layout

```
${P3_REPAIRED_PUBLISH_ROOT}/
└── tokens/                                  ← group, picks the profile
    ├── metamath/train-00000.u32le.bin
    ├── metamath/val-00000.u32le.bin
    ├── prf2/train-00000.u32le.bin
    ├── enigma/…  mizar/…  thproofs/…  isabelle/…
```

dtype: **uint32** — Qwen2.5's vocab is 151,936, so uint16 cannot hold it. Declared explicitly;
OLMo-core defaults to uint16 and would halve the count.
ext: **`.u32le.bin`, never `.npy`** — OLMo-core memmaps from byte 0 and derives the token count
from raw file size, so a real `.npy` header corrupts both, silently.
split: matched by glob on the filename; `-of-NNNNN` is rejected [verified].
payload inventory: one or more train and validation shards per family, determined
from the repaired token counts and the fixed shard-token target. Counts and
digests are derived from the final bytes, never copied from v2.

`artifacts/public/` is the preserved v2/resumable working tree and includes
token caches, done markers, manifests, and stale payloads. Neither it nor
`artifacts/release/` is a valid publish source. Create a fresh
`P3_REPAIRED_PUBLISH_ROOT` and copy only manifest-declared repaired
`.u32le.bin` shards into it. Final shard counts, SHA-256 digests, and byte
divisibility are recomputed from those fresh bytes.

## Dependencies

tokenizer: `tokenizer/qwen25-vendored/v1` — **already published and fixed.**
The validator derives `vocab_size`/`eos_token_id` from the real `tokenizer.json` and asserts every
sampled id against it; a pretrain corpus with no resolvable tokenizer is rejected.

**Do not retrain, modify, or republish the tokenizer.**

Eleven tokenizers were measured over every example, and the headline re-verified independently on
a 1-in-37 sample (agreement within 0.5%):

| tokenizer | vocab | bytes/token | corpus tokens | params to reinit |
|---|---:|---:|---:|---:|
| gpt2 | 50,257 | 1.657 | 734 M | 136.1 M |
| **qwen2.5-0.5b** | **151,936** | **1.933** | **627 M raw JSONL** | **none** |
| dolma2 | 100,278 | 2.045 | 595 M | 89.8 M |
| custom, identifier-aware pre-tokenizer, 32k | 32,768 | 3.208 | 379 M | 29.4 M |

Qwen's own tokenizer wins because `train.py:218` calls `load_hf_weights(model)` — the run starts
from the pretrained checkpoint. Any other tokenizer forces reinitialising the embedding and LM
head, which with `hidden_size=896` and tied embeddings is 106.8 M of ~494 M parameters, 22% of the
model. Nothing relearns that here.

The identifier-aware tokenizer is genuinely better on this corpus — keeping `[A-Za-z0-9_]+` runs
whole takes the TPTP shards from 1.46 to 4.01 bytes/token, a 39.8% saving over Qwen — but it is
unusable while the run initialises from pretrained weights. **dolma2 is strictly dominated**: it
asks you to relearn three times more embedding parameters than the custom option while delivering
57% more tokens per epoch, so it is never the right choice for this corpus.

Revisit only if the run ever starts from scratch or gains a large domain-adaptation phase; the
saving is 250 M tokens per epoch and 106.8 M parameters freed for depth.

## Training budget

Historical v2: **13 epochs over 494,862,336 packed train tokens =
6,433,210,368 tokens seen.** The full repaired run is still exactly 13 loader
epochs, but its token count, complete batches, steps, and exposures must be
recomputed from the new packed train bytes.
The 16k build retains 253,113 documents and drops 5,203 over-length documents; it packs to
30,204 fixed-width instances with **0.08% padding**.

At 13 epochs, a fact cited at least seven times clears 80 exposures:

| threshold | retained facts | share of facts | share of fact uses |
|---:|---:|---:|---:|
| ≥80 exposures | 25,853 | 18.60% | 85.61% |
| ≥100 exposures | 22,765 | 16.38% | 84.30% |
| ≥200 exposures | 11,627 | 8.36% | 77.12% |
| ≥500 exposures | 4,964 | 3.57% | 67.55% |

Only 18.6% of distinct retained facts saturate, but they account for 85.6% of premise uses.
The remaining long tail is the regime where the supplied fact block is intended to carry facts
the model does not memorize.

Both `configs/{dense,split}.yaml` set 13 epochs, 16,384-token sequences, packed
intra-document attention, a fixed 262,144-token global batch, and identical optimizer controls.
`configs_test.py::test_arm_is_the_only_top_level_difference` enforces that only `arm` differs.

## Historical v2 build gates — must be rerun

1. Qwen tokenizer vendored and measured.
2. Corpus tokenized into raw uint32 little-endian shards with no mask sidecars.
3. Full source separator check: exactly one `[10952, 15513, 969]` run per document.
4. Final manifests present for train and validation.
5. All 12 payloads have exact declared byte sizes, in-range ids, unique SHA-256 digests, and
   decoded samples that agree with source text.
6. The real derived mask was exercised over packed rows: fact/separator tokens are unsupervised
   in split, while goal/target tokens are supervised.

Remaining gate: rebuild, retokenize, validate, and publish a repaired corpus
version through `edullm-landing`, referencing the existing
`tokenizer/qwen25-vendored/v1`.

## Deferred (backfillable in place, do not block)

`about`, `sources[]`, `license`, `notes`, `limitations[]` feed the generated `README.md`, which is
a control file outside the hash chain. `license` is TODO: the corpus mixes Metamath (public
domain), Mizar/MML, MPTP/ENIGMA derivatives, and Magnushammer (Apache-2.0), so the aggregate needs
a decision rather than a guess.

Per-source token counts must be counted *during* the mix if the README's table is to show this
dataset's real breakdown; afterwards only upstream totals can be cited, with
`scope: "upstream-full-collection"` and an honesty caveat.

## The publish call

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime
import os

publish(
    os.environ["P3_REPAIRED_PUBLISH_ROOT"],
    dataset_id="pretrain/formal-proof-premises-500m",
    purpose="Formal proofs with their cited premises for dense/split fact-masking arms on "
            "Qwen2.5-0.5B, to test whether mathematical reasoning can be maintained "
            "without learning the supplied facts",
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/qwen25-vendored/v1",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    group_meta={"tokens": {
        "seq_len": 16384,
        "coverage": "partition",
        "partitions": [
            {"name": "train", "by": "path", "glob": "train-*.u32le.bin",
             "rows": RECOMPUTED_TRAIN_TOKENS},
            {"name": "val", "by": "path", "glob": "val-*.u32le.bin",
             "rows": RECOMPUTED_VAL_TOKENS},
        ],
    }},
)
```

`P3_REPAIRED_PUBLISH_ROOT` must be a newly created staging directory populated
only from the freshly audited token output. Never point it at
`artifacts/release/`: that preserved tree contains the forbidden v2 payload and
could be promoted successfully under a new version.

The tokenizer is not part of this publish source. The first-class
`tokenizer="tokenizer/qwen25-vendored/v1"` dependency resolves its already
promoted artifact.

There is no `labels` parameter and there is not meant to be one — a hand-typed label would be a
producer assertion nothing falsifies. `publish()` reads structure off the key.

## Publisher / deployed-validator compatibility

Checked live, because the skill says to verify rather than trust:

Historical v1/v2 publication issues were resolved before v2 promotion:
path-derived labels are present and the deployed pretrain-family distinct-ID
floor accepts the formal shards. The exact package-0.5.0 FakeS3
publish→validate integration gate must pass again on repaired output.

`weights-sidecar/v1` is unnecessary: the split mask is derived from the
separator already present in the token bytes.

`count` is `{unit, value}`, not an integer — worth knowing before hand-building anything.
