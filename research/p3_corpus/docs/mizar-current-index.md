# Current Mizar semantic index

## Status and authority

`scripts/mizar_current_index.py` is an isolated adapter for Mizar 8.1.15 /
MML 5.94.1493. It does not import or modify the legacy `html2` parser.

Source roles are deliberately separate:

- current semantic HTML is authoritative for `ARTICLE:N`,
  `ARTICLE:def_N`, and `ARTICLE:sch_N` identities and for expanded,
  semantically rendered statements;
- official `.miz` files provide literal-source goal cross-checks;
- thproof filenames establish extract identity, and thproof content supplies
  source goals and proof-completion diagnostics.

The exact URLs, archive hashes, selected-tree hashes, expected counts, proof
policy, and licensing uncertainty are pinned in
`manifests/mizar-8.1.15_5.94.1493.json`. The manifest explicitly does **not**
assert redistribution rights.

## Build

```bash
PYTHONPATH=scripts python scripts/mizar_current_index.py --manifest manifests/mizar-8.1.15_5.94.1493.json --mml /path/to/mml --html /path/to/direct/html/articles --thproofs /path/to/thproofs --sqlite /tmp/mizar-current.sqlite --jsonl /tmp/mizar-current.jsonl --report /tmp/mizar-current.report.json
```

Pass `--mizar-archive`, `--html-archive`, and `--thproofs-archive` when the
downloaded archives are retained. Tree hashes are always verified; supplied
archive paths are additionally checked against their pinned SHA-256 values.

The builder writes temporary files and replaces final outputs only after all
source, parser, count, and proof-policy gates pass.

## Builder-facing API

```python
from mizar_current_index import MizarIndex, theorem_identity

with MizarIndex("/tmp/mizar-current.sqlite") as index:
    statements = index.statement_map()
    labels = index.article_local_label_maps()
    identity = index.theorem_identity("t36_partpr_1")
    source_goal = index.source_goal(identity)
    prior = index.resolve_local_label(
        "ABSRED_0", "Lem11A", at_identity="ABSRED_0:58"
    )
```

The equivalent one-shot functions are:

- `load_statement_map(path)`
- `load_article_local_label_maps(path)`
- `load_source_goal(path, identity)`
- `theorem_identity(thproof_filename)`

`article_local_label_maps()` returns
`article -> label -> tuple[identity, ...]` in declaration order. This is
intentional: 179 labels are reused in the current MML. A global
`label -> identity` dictionary is wrong for those articles. Proof parsers must
call `resolve_local_label(..., at_identity=current_theorem)`, which selects the
nearest strictly preceding declaration and never resolves a theorem to itself.

Each JSONL record includes the canonical plain statement, the original
semantic HTML fragment, its SHA-256, HTML file/anchor/line provenance, local
label, and optional thproof metadata. SQLite stores the same data in
`statements`, `local_labels`, `thproofs`, and `metadata`.

## Full-release measurements

The manifest-pinned build measured:

- 1,500 HTML articles;
- 106,317 statements: 91,114 theorems, 14,291 definition theorems, and
  912 schemes;
- zero duplicate identities, zero missing identity numbers, and all
  76,696 thproof filenames joined;
- 50,648 local-label declarations, including 179 reused-label groups /
  378 declarations;
- 329 duplicate-statement groups / 16,832 records. These are distinct,
  authoritative identities, frequently cancellation/generated forms, not
  duplicate IDs;
- 76,237 thproof source goals found literally in official `.miz` theorem
  declarations and 459 generated-or-unmatched diagnostics. The latter are
  retained because semantic HTML identity is authoritative; the label does not
  claim all 459 are generated;
- one official `.miz` file requiring the reported, lossless Latin-1 fallback
  for legacy comment bytes. Semantic HTML remains strict UTF-8.

The pinned 240-record sample has 239 thproof/literal-`.miz` goal matches and
one explicitly pinned generated cancellation (`AFF_1:1`), for 240 agreements
and zero unexplained mismatches.

Proof categories:

- complete explicit proof: 58,658;
- malformed/truncated explicit proof: 11,040;
- inline justification: 5,160;
- no explicit proof: 1,679;
- malformed declaration: 150;
- missing theorem marker: 9.

The hard completion denominator is only the 69,698 explicit-proof-bearing
extracts: `58,658 / 69,698 = 84.160234%`. The descriptive all-file rate is
`58,658 / 76,696 = 76.481172%`; no 90% all-file gate is valid for this source.

Two final WSL2 builds took 237.3 and 242.4 seconds with 477.8 and 479.6 MiB
peak RSS. Outputs were 390,225,920 bytes (SQLite) and 232,463,349 bytes
(JSONL). Both full builds produced identical hashes:

- SQLite: `8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835`
- JSONL: `1924067218e6875737260dda35e166d13aadc6c93261ead0d55c132cc3ee789a`

Canonical JSONL bytes and their logical SHA-256 are portable. SQLite insertion,
schema, page settings, and vacuum order are deterministic in a fixed Python /
SQLite runtime; its physical file SHA may differ across SQLite versions.

## Deferred integration changes

Do these only after the active parser review finishes.

### `build_mizar_shard.py`

1. Add a required semantic-index argument and verify
   `metadata.schema_version`, `source_manifest_sha256`, and release versions.
2. Replace legacy `parse_article()` statement dictionaries with
   `MizarIndex.statement_map()`.
3. Replace one-value article label dictionaries with contextual
   `resolve_local_label(article, label, at_identity=theorem)`.
4. Keep the reviewed official-`.miz` theorem/proof extraction logic separate;
   compare its identity/source goal to the index rather than reparsing HTML.
5. Treat semantic `canceled` records and nonliteral/generated diagnostics
   explicitly; do not infer current IDs by counting literal `.miz` theorems.

### `build_thproofs_shard.py`

1. Require the semantic index and current source manifest; remove all runtime
   dependence on legacy 7.13 `html2`.
2. Resolve each filename through `index.theorem_identity(path.name)` and verify
   that the indexed thproof `file_name` association is identical.
3. Emit `index.statement_map()[identity]` as the canonical expanded goal and
   retain `index.source_goal(identity)` for diagnostics.
4. Resolve local references contextually at the current theorem identity.
5. Replace the 90%-of-all-files completion gate with the manifest-pinned exact
   categories and the explicit-proof-bearing rate gate. Keep acceptance and
   citation-completeness gates separate from source proof completion.

### `verify_corpus.py`

1. Require per-row/release source metadata containing the semantic index
   schema, index SHA-256, source-manifest SHA-256, and MML release.
2. Verify theorem identities, canonical goals, and fact statements against the
   read-only index.
3. Verify thproof filename/identity association and reject mixed legacy/current
   source metadata.
4. Report generated/nonliteral and proof-category diagnostics separately from
   malformed/truncated records.

No generated index is checked into the active corpus or artifacts by this
adapter.
