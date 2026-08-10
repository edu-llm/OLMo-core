# Pinned dependencies for P3 corpus rebuild

Do not upgrade these casually. The pretrain reader, evaluator publisher, and
local verification scripts were validated against exact package bytes.

## Python packages (runtime)

| Package | Pin | Used for |
| --- | --- | --- |
| `tokenizers` | 0.22.2 | Must match vendored Qwen seal in `templates/generation-inputs/tokenizer-seal.json` |
| `pytest` | ≥7 | Local regression tests |
| `transformers` | compatible with Qwen2.5-0.5B load | Strict smoke tests only |
| `jsonschema` | ≥4 (Draft 2020-12) | `tests/test_evaluator_release.py` only (see caveat below) |

Install the OLMo-core project environment from the repository root for
tokenization (`src/scripts/train/p3_math_split/tokenize_corpus.py`).

### Two test modules do not collect in a bare environment

`pytest tests/` exits 2 rather than running, because both modules import at
top level:

| Module | Missing import | Note |
| --- | --- | --- |
| `tests/test_evaluator_release.py` | `jsonschema` | Installing it does **not** make the module pass — see below |
| `tests/test_publish_eval_v1.py` | `edullm_data` | Needs the 0.8.0 pin checked out into `.p3-work/full13/edullm-data/` so `src/edullm_data/` resolves |

The second is expected, not a missing declaration: this skeleton deliberately
does not vendor `edullm-data` source (see *What this skeleton does not vendor*),
only its family and infra control files.

**Installing `jsonschema` does not turn `test_evaluator_release.py` green.** The
module then collects 127 tests and 98 fail on
`EvaluatorReleaseError: packaged semantic source roots drift`. That is a red
suite, not a missing dependency; it is tracked as a blocker in `AGENTS.md`. Do
not convert either import into `pytest.importorskip` — that would make
`pytest tests/` exit 0 while burying the 98 failures.

Until the evaluator suite is repaired, run the pretrain-path suite explicitly:

```bash
PYTHONPATH=scripts python -m pytest -q tests/ \
  --ignore=tests/test_evaluator_release.py \
  --ignore=tests/test_publish_eval_v1.py
```

## eduLLM data contracts

| Role | Version | Commit | Profiles |
| --- | --- | --- | --- |
| Pretrain image/reader | 0.5.0 | `38bf831a6c3f445e394784018441fd59288b876c` | `pretrain-tokens/v1`, `tokenizer/v1` |
| Evaluator publisher | 0.8.0 | `f91d92d1a541ef96686b9cbcad4220d58bf71dac` | + `text-corpus/v1`, `p3-evaluator-corpus/v1` |

Set `EDULLM_SRC` to a checkout at the pinned commit when running
`scripts/read_check_v3.py` or publish scripts locally.

Publishing path:

```python
edullm_data.publish(...)  # → s3://edullm-landing only
```

Promotion and cataloging are validator-owned; never write `s3://edullm-data`
directly.

## Published dataset artifacts (external, already promoted)

| Artifact | ID | Notes |
| --- | --- | --- |
| Tokenizer | `tokenizer/qwen25-vendored/v1` | Sole external dependency of the corpus; do not republish |
| Pretrain v3 | `pretrain/formal-proof-premises-500m/v3` | 12 shards, SHA-256 `7360db01…` |

## Training stack pins (OLMo-core)

| Component | Pin |
| --- | --- |
| Base model | Qwen2.5-0.5B @ `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Loss kernel | Liger 0.7.0 fused linear CE |
| Attention | FlashAttention2 required |
| Branch | `edullm/p3-math-split` |

## Upstream source lock

All downloadable upstream bytes are listed in `source-lock.json` with URLs and
SHA-256 digests. `scripts/bootstrap_sources.py` is the supported fetch path.

## What this skeleton does not vendor

- Multi-gigabyte JSONL corpus payloads
- Packed `.u32le.bin` training shards (already in `edullm-data`)
- Full `edullm-data` source trees (pin commits above; optional local checkouts)
- Hugging Face credentials (required for Isabelle `all_data.json` fetch)
