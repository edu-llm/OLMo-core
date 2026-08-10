# P3 corpus source and provenance snapshot

This directory preserves the Git-suitable P3 corpus builders, release tooling,
focused tests, source-control manifests, tokenizer seal, decision records, and
publication evidence that produced:

- `tokenizer/qwen25-vendored/v1`
- `pretrain/formal-proof-premises-500m/v3`

**Start here for agents:** [`AGENTS.md`](AGENTS.md) — scientific contract, data
shapes, gates, thresholds, and the rebuild pipeline.

The packed pretraining payload is already promoted in `edullm-data`; it is not
duplicated in Git. The unpublished evaluator JSONL payload also remains outside
Git. `provenance/evaluator-v3/` retains only its control JSON and README.

## Quick rebuild path

```bash
# Verify upstream sources (URLs + SHA-256 in source-lock.json)
python scripts/bootstrap_sources.py --root /tmp/p3-sources --build-mizar-index

# Resumable full pipeline. Also resolves the generation-input templates and
# rebuilds the accepted ENIGMA base. See AGENTS.md for what each stage does.
python scripts/orchestrate_rebuild.py \
  --work-root /tmp/p3-rebuild-work \
  --sources-root /tmp/p3-sources

# Compare output to canonical v3 expectations
python scripts/verify_rebuild.py \
  --tokenized-root /tmp/p3-rebuild-work/tokenized-v3 \
  --publish-root /tmp/p3-rebuild-work/publish-stage-v3
```

Skeleton CI tests (no multi-GB data):

```bash
PYTHONPATH=research/p3_corpus \
python -m pytest -q research/p3_corpus/tests/test_rebuild_skeleton.py \
                 research/p3_corpus/tests/test_archive_portability.py
```

## Key files

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Agent-oriented contract, structure, gates, thresholds |
| `source-lock.json` | Immutable upstream URLs and SHA-256 pins |
| `expected-release-v3.json` | Row/token counts and hashes for verification |
| `PINNED_DEPENDENCIES.md` | edullm-data and runtime package pins |
| `archive-inventory.json` | SHA-256 inventory of every tracked skeleton file |

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

- `scripts/`: corpus construction, bootstrap, orchestrator, verification, staging, publisher
- `tests/`: focused corpus/release/skeleton tests and small fixtures
- `templates/generation-inputs/`: policies, tokenizer seal, six family manifests, SUMMARY
- `tokenizers/qwen25-vendored/`: exact tokenizer files for local byte verification
- `manifests/`: small source identity manifests
- `provenance/`: sealed corpus, tokenization, evaluator, and source-control records
- `docs/`: decision ledger, runbooks, schemas, reports, and figures

Tests requiring multi-gigabyte source or evaluator payloads must be supplied
those paths explicitly. The archive never substitutes generated fixture bytes
for missing production data.
