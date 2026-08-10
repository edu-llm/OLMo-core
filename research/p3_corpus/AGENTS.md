# P3 corpus agent guide

Read this file first when working on the P3 formal-proof-premises corpus. It
explains the scientific contract, the data shapes you must preserve, the gates
that must pass, and how to regenerate the release from pinned upstream sources.

Authoritative decision history lives in `docs/reports/P3_DECISION_LEDGER.md`.
Operational status and audit evidence live in `docs/reports/P3_WORK_STATUS.md`
and `docs/reports/checklist.md`. Dataset layout and publishing rules live in
`docs/reports/DATASET-DESIGN.md`. Builder mechanics live in
`docs/p3-generation-v2.md` and `docs/reports/CORPUS_BUILD_PLAN.md`.

## What you are building

P3 trains Qwen2.5-0.5B on a six-family formal-math corpus to compare two arms on
**identical packed bytes**:

- **dense** supervises every token before the goal boundary and all goal/derivation tokens.
- **split** attends to the fact block but receives **no loss** before `---`; goal and
  derivation tokens are supervised in both arms.

The published training artifact is:

- `pretrain/formal-proof-premises-500m/v3`
- tokenizer dependency: `tokenizer/qwen25-vendored/v1` (already published; never replace)

The evaluator JSONL release is a **separate immutable artifact** linked by exact
version and manifest SHA-256. It is not duplicated in Git; only control JSON
remains under `provenance/evaluator-v3/`.

## Read order for a new agent

1. `AGENTS.md` (this file)
2. `P3_DECISION_LEDGER.md` — user-approved constants and non-negotiables
3. `expected-release-v3.json` — row counts, token counts, hashes a clean rebuild must hit
4. `source-lock.json` — upstream URLs and SHA-256 pins
5. `docs/reports/DATASET-DESIGN.md` — on-disk layout for packed tokens
6. `docs/p3-generation-v2.md` — six-family generation transaction
7. `docs/reports/CORPUS_BUILD_PLAN.md` — shard invariants and sample gate

After any context compaction, reread the decision ledger and checklist before
changing thresholds, scripts, or data paths.

## Directory layout (Git skeleton)

```
research/p3_corpus/
├── AGENTS.md                     ← you are here
├── README.md                     ← quick human overview
├── PINNED_DEPENDENCIES.md        ← edullm-data / Python package pins
├── source-lock.json              ← upstream acquisition contract
├── expected-release-v3.json      ← rebuild acceptance targets
├── archive-inventory.json        ← SHA-256 of every tracked skeleton file
├── scripts/                      ← corpus builders, bootstrap, orchestrator, verifier
├── tests/                        ← focused regression + skeleton CI tests
├── tokenizers/qwen25-vendored/   ← exact local tokenizer seal for byte verification
├── manifests/                    ← Mizar 8.1.15 source identity manifest
├── templates/generation-inputs/  ← policies.json + tokenizer-seal.json templates
├── provenance/                   ← sealed control JSON (no multi-GB payloads)
│   ├── sealed-corpus-manifest.json
│   ├── tokenized-v3/{train,val}_meta.json
│   ├── evaluator-v3/
│   └── source-controls/
└── docs/                         ← runbooks, schemas, reports, figures
```

**Not in Git:** packed `.u32le.bin` shards, JSONL train/eval payloads, upstream
archives, token caches, publish staging trees, generation work directories.

## Example row shape (JSONL)

Every training/eval row renders to text of the form:

```
I know these mathematical statements:
<name> : <statement>
...
Local assumptions:
<used $e assumption lines, Metamath only>
---
GOAL <proposition>
<derivation or tactic trace>
```

Design rules encoded across builders:

| Rule | Rationale |
| --- | --- |
| Fact block is an **oracle** | Every cited premise appears with full statement; partial blocks are dropped, not rendered |
| Exactly one mask span, 5–60% of text | The split arm manipulation is the fact block |
| Facts **not** in citation order | Prevents handing the step sequence for free |
| One name → one statement corpus-wide | Split arm needs stable keys |
| Held-out facts isolated bidirectionally | No train row cites a held-out fact; no held-out fact's own proof leaks |
| Metamath `(reuse)` kept in source replay, omitted from visible targets | Semantic verifier reuses earlier matching expressions |
| Local `$e` assumptions: only decoded/used ones appear before `---` | Push noise removed from visible targets |
| Isabelle target: `facts + state_before -> tactic + state_after` | Not tactic-only |
| `text + EOS ≤ 16,384` before held-out selection | Whole row dropped; never truncated or split |

Row schemas by family:

| Family | Schema | Notes |
| --- | --- | --- |
| metamath | `metamath-proof-v2` | 500 statement-equivalence held-out classes after 16K filter |
| mizar | `mizar-proof-v2` | Direct recovered alignment; 55,353 source-backed rows accepted |
| thproofs | `mizar-proof-v2` | Current MML 8.1.15 + semantic index only |
| prf2 | `atp-v2` | prf2 before ENIGMA in dedup order |
| enigma | `atp-v2` | +2,087 low-tier alternative proofs only (cap 8,192 text+EOS) |
| isabelle | `isabelle-transition-v2` | 500 family-local held-out facts |

## Packed token layout (published pretrain)

Profile: `pretrain-tokens/v1` via `edullm-data` 0.5.0.

```
tokens/<family>/{train,val}-NNNNN.u32le.bin
```

- dtype: **uint32** little-endian (Qwen vocab 151,936)
- sequence length: **16,384**
- separator search: token IDs **`[10952, 15513, 969]`** (`---\nGOAL` core)
- loss mask: **derived at read time**, not published (boundary already in bytes)
- six families: `metamath`, `mizar`, `thproofs`, `prf2`, `enigma`, `isabelle`

Published v3 targets (from `expected-release-v3.json`):

| Quantity | Value |
| --- | ---: |
| Sealed train JSONL rows | 181,652 |
| Sealed eval JSONL rows | 4,191 |
| Packed train tokens (reader) | 467,206,144 |
| Packed val tokens (reader) | 10,059,776 |
| Train packed instances | 28,516 |
| Val packed instances | 614 |
| Complete batches / epoch | 1,782 |
| Steps @ 13 epochs | 23,166 |
| Metamath overlength drops (pre-holdout) | 960 |
| Tokenizer-time overlength drops | 0 |

**Never reuse v2 counts.** v2 (`pretrain/formal-proof-premises-500m/v2`) is
byte-reproducible but scientifically stale and forbidden for final training.

## Held-out and pooling policy (approved)

| Pool | Policy | Seed / size |
| --- | --- | --- |
| Mizar + thproofs + prf2 + ENIGMA | One pooled semantic-class draw | exactly **1,000** classes, seed **`20260801`** |
| Metamath | Tail statement-equivalence classes by total exposure | exactly **500** classes after 16K eligibility filter |
| Isabelle | Family-local held-out facts | **500** facts |
| ATP MPTP scope | Explicit bookkeeping filter only | other stable named premises remain eligible |

Removed from contract: `facts_shuffled` evaluation condition.

Evaluation runs **all three** conditions (`facts_present`, `facts_absent`,
`facts_corrupted`) on each family's full context-eligible cohort (4,191 total
eval rows). **facts_present** is the headline; the others are mechanism
diagnostics.

## Fixed training constants (do not drift)

| Constant | Value |
| --- | ---: |
| Model | Qwen2.5-0.5B pretrained |
| Tokenizer | `tokenizer/qwen25-vendored/v1` |
| Seed | 42 |
| Epochs | 13 |
| Sequence length | 16,384 |
| Global batch | 262,144 tokens |
| Rank microbatch | 16,384 tokens |
| LR | 2e-5 |
| Warmup | 2,400 steps |
| Loss | Liger 0.7.0 fused linear CE |
| Hardware | two independent 8×GPU jobs (A100 or H100) |
| Metamath snapshot | `set.mm@82830c78861b96e906d9868c30c35dbd98be5db5` |

## Gates and thresholds you must enforce

### Corpus shard invariants (`CORPUS_BUILD_PLAN.md` §10.1)

Eighteen tests / eleven invariant groups (I1–I11): non-empty oracle blocks,
held-out isolation (both directions), unique names, non-degenerate targets,
single mask span 5–60%, no duplicates, no eval-in-train leakage, no control
chars/truncation, facts not in citation order, shared held-out manifest hash.

Mutation tests must go **red** on corrupted shards; green alone is insufficient.

### Metamath 16K eligibility

- Measure exact fixed-Qwen `text + EOS` on final rendered row
- Drop whole row when length > 16,384 **before** exposure counting and held-out selection
- Persist drop ledger with ID, theorem, length, native hash, reason, tokenizer seal
- Expected: **65,122** train, **952** builder eval, **960** overlength drops

### Mizar current-source floors

From recovered 8.1.15 triplet + semantic index:

- minimum **45,000** accepted rows
- minimum **80%** explicit-proof completion rate (denominator = explicit-proof-bearing extracts)
- never combine nn_conj20 `html2` with 8.1.15 thproofs

### ENIGMA low-tier recovery (approved tier only)

- at most **2,087** materially distinct alternative traces
- preserve all **27,079** accepted rows
- cap text+EOS at **8,192** for variants
- reject superficial/exact/dead-step variants
- route theorem variants together in pooled holdout planning

### Tokenization gate

Before writing shards:

- append EOS to every document
- exactly-one separator check using `[10952, 15513, 969]`
- refuse if separator missing or repeated
- uint32 little-endian, zero header bytes, `.u32le.bin` not `.npy`

### Publication gate

- publish only manifest-declared `.u32le.bin` payloads through `edullm_data.publish()` → `s3://edullm-landing`
- never write `s3://edullm-data` directly
- never hand-write `dataset.json` or platform manifests
- pretraining reader pin: `edullm-data@38bf831a6c3f445e394784018441fd59288b876c` (0.5.0)
- evaluator publisher pin: `edullm-data@f91d92d1a541ef96686b9cbcad4220d58bf71dac` (0.8.0)

### Production generation gate

`build_p3_generation.py` is the **only** authorized production entry point.
Generic transaction-v2 JSONL is consumable by the generic tokenizer but **not**
authorized P3 input without the deep P3 verifier passing first.

Required inputs:

- six `p3-family-source-manifest/v2` files
- `templates/generation-inputs/policies.json`
- `templates/generation-inputs/tokenizer-seal.json`
- persisted MML v7 contract root
- Metamath overlength drop ledger
- Mizar semantic index SQLite matching `source-lock.json`

Run `--dry-run` until `p3-generation-preflight/v2` reports zero blockers.

## Rebuild pipeline (compute-only path)

```bash
# 1. Fetch and verify upstream sources (~4 GB download, ~28 GB peak disk)
python scripts/bootstrap_sources.py \
  --root /tmp/p3-sources \
  --build-mizar-index

# 2. One-command resumable orchestrator. Resolves templates/generation-inputs/
#    into the work root and rebuilds the accepted ENIGMA base along the way.
python scripts/orchestrate_rebuild.py \
  --work-root /tmp/p3-rebuild-work \
  --sources-root /tmp/p3-sources

# 3. Compare rebuilt bytes to canonical expectations
python scripts/verify_rebuild.py \
  --tokenized-root /tmp/p3-rebuild-work/tokenized-v3 \
  --publish-root /tmp/p3-rebuild-work/publish-stage-v3
```

Stages recorded in `<work-root>/orchestrator-state.json`; re-run skips completed
stages unless `--force`.

Focused tests (no multi-GB payloads required):

```bash
PYTHONPATH=research/p3_corpus \
python -m pytest -q research/p3_corpus/tests/test_rebuild_skeleton.py \
                 research/p3_corpus/tests/test_archive_portability.py
```

Full corpus tests additionally need supplied source/evaluator paths; the archive
never substitutes generated fixture bytes for missing production data.

## Generation manifests

Portable templates live in `templates/generation-inputs/`:

| File | Role |
| --- | --- |
| `policies.json` | Pooled MML / family held-out policy pins |
| `tokenizer-seal.json` | Fixed Qwen2.5 tokenizer four-part seal |
| `metamath.json` … `isabelle.json` | Six `p3-family-source-manifest/v2` builder contracts |
| `SUMMARY.json` | Root SHA-256 index over the six family manifests |

Templates use placeholders (`{{P3_CORPUS_ROOT}}`, `{{P3_SOURCES_ROOT}}`,
`{{P3_WORK_ROOT}}`, `{{PYTHON}}`). Resolve them before generation:

```bash
python scripts/materialize_generation_inputs.py \
  --out /tmp/p3-rebuild-work/generation-inputs \
  --corpus-root research/p3_corpus \
  --sources-root /tmp/p3-sources \
  --work-root /tmp/p3-rebuild-work
```

`orchestrate_rebuild.py` runs this automatically as the
`materialize_generation_inputs` stage.

**ENIGMA note:** the enigma manifest references
`{{P3_WORK_ROOT}}/atp/enigma-accepted-base-v1`, a ~1.5 GiB derived artifact
from an initial prf2/ENIGMA acceptance pass. It is rebuilt from the locked
`mzr01/03/02/08` archives by the `build_accepted_bases` orchestrator stage:

```bash
python scripts/build_accepted_bases.py \
  --sources-root /tmp/p3-sources \
  --work-root /tmp/p3-rebuild-work
```

The base is `build_atp_shard.py` over the four extracted runs *without* the
low-tier flags, under the legacy CLI recorded in `ENIGMA_LOW_TIER_POLICY`
(`min-steps 4`, `jaccard 0.5`, `seed 20260801`, fenced, dedup, heldout 0). Its
exact bytes/rows/sha256 are pinned in
`ENIGMA_LOW_TIER_SOURCE_CONTRACT["accepted_base"]`, and both the stage and the
low-tier build refuse to continue on drift, so a regenerated base is either
bit-identical or the rebuild stops.

Rebuilding it needs all four runs. `mzr02` and `mzr08` are not incidental: they
supply 1,831 of the 2,087 low-tier alternatives, and the base itself spans
231,520 files across all four. All four are therefore required entries in
`source-lock.json`, since `bootstrap_sources.py` skips `optional` archives
without downloading them.

The stage is idempotent: a base already matching the pin is left alone unless
`--force` is passed.

## What not to infer or implement without user approval

- random-mask control arm
- fact-block dropout
- rare-fact upweighting or name anonymization
- sub-proof re-rooting or separate raw-math phase
- strong dense facts-present > facts-absent load-bearing gate
- proof-level seen-fact vs new-fact axis
- minimum effect/equivalence/non-inferiority margin
- new tokenizer or changed sequence length
- copying v2 token/step/exposure statistics into v3 reports
- fabricating corpus license metadata

## Open items an agent should treat as blockers

1. Clean-room rebuild has **not** yet been demonstrated end-to-end from
   `source-lock.json` alone; some recovered upstream paths still require the
   generation-input manifests to be materialized with verified absolute paths.
2. Aggregate corpus license metadata remains honestly unresolved (backfillable,
   not a technical generation gate).
3. Evaluator JSONL promotion is deferred from the pretrain-readiness critical path.
4. **The evaluator release suite is red, and the failure is masked by a missing
   import.** `tests/test_evaluator_release.py` cannot collect without
   `jsonschema`, so `pytest tests/` exits 2 before running anything. Install
   `jsonschema` and the module collects 127 tests, of which **98 fail**:

   ```
   build_evaluator_release.EvaluatorReleaseError: packaged semantic source roots drift
   scripts/build_evaluator_release.py:2516
   ```

   Verified pre-existing at commit `6db9a5a` in a clean worktree, so it is not
   fallout from the rebuild-skeleton work. Root cause is **not diagnosed**: the
   check compares packaged semantic source identity against recomputed source,
   policy, tokenizer, and artifact-inventory roots, and it is not yet known
   whether the packaged fixtures are stale or `build_evaluator_release.py`
   regressed. Anyone touching the evaluator path must diagnose this first.

   Do not silence it with `pytest.importorskip`. That turns 98 failures into a
   green run. The pretrain path is unaffected and stays green at
   887 passed / 34 skipped with both evaluator modules ignored.

## Package and platform contracts

| Role | Package | Commit / version |
| --- | --- | --- |
| Pretrain reader | edullm-data | 0.5.0 @ `38bf831a6c3f445e394784018441fd59288b876c` |
| Evaluator publisher | edullm-data | 0.8.0 @ `f91d92d1a541ef96686b9cbcad4220d58bf71dac` |
| Training entry | OLMo-core branch `edullm/p3-math-split` | see repo |
| Tokenizer | published artifact | `tokenizer/qwen25-vendored/v1` |

Platform launch until v3 appears in the submission registry:

```text
--dataset-id pretrain/formal-proof-premises-500m
--dataset-version v3
--dataset-tokenizer tokenizer/qwen25-vendored/v1
```

Form: **Data: None** (`dataset_release=none`). Do not set raw S3 paths or
`EDULLM_DATASET_*` yourself.

## Success criteria for your work

You are done when:

1. `bootstrap_sources.py` verifies every non-optional `source-lock.json` entry.
2. `build_p3_generation.py --dry-run` passes with zero blockers.
3. A full generation + `verify_corpus.py` completes cleanly.
4. Tokenization produces `train_meta.json` / `val_meta.json` matching
   `expected-release-v3.json` counts and tokenizer seal.
5. `verify_rebuild.py` exits 0 against those trees.
6. Focused pytest skeleton tests pass in CI.

The burden after that is compute time and human review before any S3 upload.
