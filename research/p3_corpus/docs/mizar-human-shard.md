# Direct human-Mizar raw shard

## Status

`scripts/build_mizar_human_shard.py` is an isolated production builder for
Mizar 8.1.15 / MML 5.94.1493. It creates only a raw staging shard. It does not
select held-out facts or write `shards/` or `eval/`; the pooled MML planner owns
the final split.

The builder does not accept a legacy `html2` argument and does not import the
legacy Mizar or thproof builders.

## Source authority

The required source tuple is:

- all 1,500 official `.miz` articles from the pinned Mizar archive;
- the accepted `mizar-semantic-index-v1` SQLite file and its exact physical
  SHA-256;
- `manifests/mizar-8.1.15_5.94.1493.json`;
- the matching current semantic HTML and thproof trees;
- all three downloaded source archives, whose SHA-256 values are mandatory;
- the sealed vendored Qwen2.5 tokenizer.

Source roles are intentionally separate:

- official `.miz` bytes are authoritative for the human target;
- the semantic index is authoritative for `ARTICLE:N`, the canonical goal, and
  every emitted global fact statement;
- thproof proof hashes in the index are identity anchors, not target content.

The builder verifies tree hashes, archive hashes, source-manifest hash, SQLite
application/user versions and integrity, index metadata, release versions,
tokenizer bytes/version/behavior, and the exact local index SHA before reading
proofs. It repeats source and index hash verification before atomically
installing the fresh output directory.

## Production command

The output path must not already exist.

```bash
PYTHONPATH=scripts python scripts/build_mizar_human_shard.py \
  --mml-root /tmp/p3-source-audit/extract-mizar/mml \
  --html-root /tmp/p3-source-audit/extract-html-current/html \
  --thproofs-root /tmp/p3-source-audit/extract-thproofs/thproofs \
  --semantic-index /tmp/mizar-current-8.1.15-final2.sqlite \
  --semantic-index-sha256 8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835 \
  --source-manifest manifests/mizar-8.1.15_5.94.1493.json \
  --mizar-archive /tmp/p3-sources/mizar-8.1.15_5.94.1493-i386-linux.tar \
  --html-archive /tmp/p3-sources/html-abstr-8.1.15_5.94.1493.tar.gz \
  --thproofs-archive /tmp/p3-sources/thproofs-8.1.15_5.94.1493.tar.gz \
  --tokenizer-path tokenizers/qwen25-vendored \
  --out /tmp/<fresh-output> \
  --heldout 0
```

Any nonzero `--heldout` is refused.

## Parsing and identity mapping

Each article is decoded independently (strict UTF-8 with a reported lossless
Latin-1 fallback) and then released, so the 122 MB MML is not materialized as a
single string.

Literal `theorem` declarations are bounded at the next theorem before proof
parsing. The proof parser:

- distinguishes canceled, inline-justified, no-proof, malformed, and complete
  explicit declarations;
- balances nested `proof`, `now`, `hereby`, `suppose`, and `case` blocks;
- treats `per cases;` as a command, not as an extra `end;`-delimited block;
- rejects malformed `end` tokens and same-line trailing garbage;
- emits the exact trimmed source span between outer `proof` and outer `end;`,
  preserving all internal comments and layout.

Production identity mapping is a unique, order-preserving longest common
subsequence over exact `(proof SHA-256, source goal, source label)` anchors.
This is necessary because the index's set-based `literal_goal_match` diagnostic
can also describe generated semantic identities. Generated index rows and
source proofs without an exact index proof anchor are skipped explicitly.
Ambiguous maximum alignments fail the entire build.

The release has hard gates for all declaration categories, exact mappings, and
every row disposition. A parser, source, order, index, or filtering regression
therefore cannot silently produce a smaller shard.

## References and local context

`by` and `from` clauses are parsed as grammar-bounded, comma-separated citation
lists. A clause stops before subsequent calculational text such as
`by A1 .= x by A2;`.

The resolver supports:

- `ARTICLE:N` with inherited numeric shorthand;
- `ARTICLE:def N` / `ARTICLE:sch N`;
- qualified article labels such as `OTHER:Lm4`;
- arbitrary unqualified labels, including `Def5a` and digit-leading labels;
- reused current-article labels through
  `MizarIndex.resolve_local_label(..., at_identity=theorem)`.

Proof-local labels are tracked by source position and never become global
facts. A qualified label in another article must have exactly one indexed
identity. Every required global identity must have a canonical index statement.
Any unresolved or ambiguous item drops the row with a typed counter; no name or
statement is guessed.

Explicit `assume`, `suppose`, and `given` source context is retained in
`local_assumptions`. It remains in the exact target as well; the global fact
block contains global index facts only.

## Row contract

The row schema is the shared six-family `mizar-proof-v2`; there is no
direct-human schema fork. Each JSON object includes:

- deterministic SHA-256 `id`, `family`, `split: "raw"`, and `heldout: 0`;
- canonical `theorem`, `facts`, and source-order `cited`;
- `proof_local_labels` and `local_assumptions`;
- index-canonical `goal` and exact human `target`;
- reconstructed `text`, `mask`, `mask_start`, and `mask_end`;
- exact Qwen `token_length_with_eos`;
- theorem/source offsets, hashes, encoding, lines, and declaration ordinal;
- semantic identity, statement/proof hashes, HTML anchor/line, and alignment;
- source/index/archive/tokenizer/quality/schema provenance.

Fact dictionary order preserves the accepted `mizar-human-proof-v1` SHA-256
rank namespace during the schema-only migration. It remains deterministic and
independent of citation order, while preserving every accepted candidate and
prompt byte.

Rows over `text + one EOS > 16,384` are dropped without truncation. Exact text
duplicates are deduplicated through a disk-backed SHA-256 plus byte-equality
gate.

## Outputs

Production requires `--name mizar`. The fresh output tree is:

```text
raw/mizar.jsonl
manifests/mizar.json
reports/mizar.build.json
reports/mizar.fact_frequencies.json
checksums/mizar.json
```

`manifests/mizar.json` is exactly
`p3-family-source-manifest/v2`. It has only the canonical top-level fields:
`schema_version`, `family`, `row_schema_version`, `row_source_metadata`,
`source_snapshots`, `builder`, `license`, `source_verifier_acceptance`,
`test_only`, and `manifest_root_sha256`. The row metadata seals source archive
and tree roots, semantic-index root, quality-filter root, schema-generation
root, tokenizer seal, and context policy. The builder declaration exposes the
closed external command and exact recursive inventory consumed by the
six-family orchestrator.

Runtime, counters, output hashes, and replay evidence remain in the build
report. The complete accepted-row fact-frequency map remains separate.

## Verification

Before installation, every emitted row is reopened and checked against:

- the exact source file, declaration ordinal, target offsets, and hashes;
- the semantic identity, canonical goal, and proof anchor;
- a fresh reference resolution and every canonical fact statement;
- deterministic fact order, local context, ID, text, and mask reconstruction;
- an exact sealed-Qwen encode with one EOS.

A second deterministic 100-row replay is stratified over token length, fact
count, and local-context presence. Production requires exactly 100 rows.

## Verified full-source build

The final recovered integration run is at
`/tmp/p3-full13/mizar-recovered`. It measured:

- 1,500 source files and 75,158 literal theorem declarations;
- 67,863 complete explicit proofs, 5,258 inline justifications, 2,036
  no-explicit-proof declarations, one malformed explicit proof, and zero
  canceled/malformed declarations;
- 63,595 unique exact source/index proof mappings and 4,268 source proofs
  without an exact index proof anchor;
- 55,353 accepted raw rows, including 5,239 uniquely label-recovered rows;
- typed mapped-row drops: 5,645 unresolved references, 2,575 no-global-citation
  proofs, 10 overlength rows, and 12 exact duplicates;
- 47,138 distinct accepted global facts;
- 37,783 accepted rows with explicit local assumptions/context;
- exact Qwen `text + EOS` lengths: minimum 51, p50 470, p90 1,538,
  p95 2,297, p99 5,206, maximum 16,355;
- token-length sequence SHA-256
  `ea246e12e76ec67827c91bd919bc6271abf39ff05906d4812454b1255b44ea01`;
- all 55,353 rows passed source/index/reference/reconstruction checks;
- the stratified replay checked 100/100 rows;
- runtime 469.13 seconds and peak RSS 772.57 MiB.

Output bytes and hashes:

- raw JSONL: 429,265,906 bytes,
  `54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f`;
- canonical family source manifest file:
  `dfda6cfb3815f8032044450b0d8378b1da8efd2ec0e793e05add13159ea7f551`;
- canonical family source-manifest root:
  `fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762`;
- schema-generation root:
  `ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242`;
- quality-filter root (unchanged):
  `9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d`;
- complete fact-frequency report:
  `d214c7e60492a2664fa9b96e83c304ba1fe62fec1ce8ad621cbd1fd9a2b3e8e0`;
- runtime build report:
  `2ed7b6fa2d8ff69c41c915a7bd13ef69e7a80aa47f6a9992e730dccd44a12328`.

## Integration API

The builder API remains:

```python
tokenizer = load_vendored_tokenizer(path)
report = build_corpus(BuildConfig(..., heldout=0, production=True), tokenizer)
```

The six-family transaction ingests only `raw/mizar.jsonl`, validates the exact
builder inventory and `mizar-proof-v2` rows, then delegates train/eval/drop
routing to the pooled MML planner. The finalized direct-Mizar source-policy
entry seals 55,353 rows plus source, index, recovery, raw, quality, schema, and
family-manifest roots.

`verify_corpus.direct_mizar_record_errors()` replays current rows against the
official `.miz` declaration, target offsets and hashes, deterministic ID,
canonical family manifest, `MizarIndex` goal/facts, direct-human reference
resolver, local assumptions, and rendered prompt. Transaction verification
also checks pooled route accounting and typed drop ledgers. There is no
compatibility fallback to legacy `html2`.

The schema migration changed serialization and deterministic row IDs, but not
candidate semantics: all 50,114 old/new semantic projections match. Their
shared semantic projection SHA-256 is
`81fc54954d784bf3cb35b4b8b894476c2db03c35125b9365d4df13e5000cf15c`,
and their shared ordered-text SHA-256 is
`1ac8035a6cb7a00c3989a701329aa1479bb99047f58f541ab439d68350cbe5ef`.

## Licensing blocker

Building and local verification do not authorize publication. The pinned
manifest does not assert redistribution rights. Individual MML headers refer
to GPL-3.0-or-later or CC-BY-SA-3.0-or-later subject to
`COPYING.interpretation`, while no archive-specific grant for semantic HTML or
thproofs was independently verified. Legal review remains mandatory before
publishing or redistributing this aggregate output.
