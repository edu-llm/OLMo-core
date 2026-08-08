# P3 corpus source and provenance snapshot

This directory preserves the Git-suitable P3 corpus builders, release tooling,
focused tests, source-control manifests, tokenizer seal, decision records, and
publication evidence that produced:

- `tokenizer/qwen25-vendored/v1`
- `pretrain/formal-proof-premises-500m/v3`

The packed pretraining payload is already promoted in `edullm-data`; it is not
duplicated in Git. The unpublished evaluator JSONL payload also remains outside
Git. `provenance/evaluator-v3/` retains only its control JSON and README, while
the local canonical payload remains `corpus-v3/` in the original build
workspace until an evaluator dataset is validated and promoted.

## Package contract

The pretraining image/reader contract is `edullm-data` 0.5.0 at commit
`38bf831a6c3f445e394784018441fd59288b876c`, whose live registry includes
`pretrain-tokens/v1` but not `p3-evaluator-corpus/v1`.

The evaluator publisher contract is `edullm-data` 0.8.0 at commit
`f91d92d1a541ef96686b9cbcad4220d58bf71dac`, whose live registry additionally
includes `text-corpus/v1` and `p3-evaluator-corpus/v1`.

Do not hand-write `dataset.json` or manifests. Publishing must use the pinned
`edullm_data.publish()` client through `edullm-landing`; validation and promotion
own the published control files.

## Layout

- `scripts/`: corpus construction, transaction, verification, staging, and
  publisher sources.
- `tests/`: the focused corpus/release tests and their small fixtures.
- `tokenizers/qwen25-vendored/`: the exact tokenizer files used by local
  byte-level verification.
- `manifests/`: small source identity manifests.
- `provenance/`: sealed corpus, tokenization, evaluator, and source-control
  records. Historical absolute paths are retained here as evidence; payload
  bytes are intentionally absent.
- `docs/`: decision ledger, runbooks, schemas, reports, and figures.

Run the archived tests from this directory's repository root with:

```bash
PYTHONPATH=research/p3_corpus \
python -m pytest -q research/p3_corpus/tests
```

Tests requiring multi-gigabyte source or evaluator payloads must be supplied
those paths explicitly. The archive never substitutes generated fixture bytes
for missing production data.
