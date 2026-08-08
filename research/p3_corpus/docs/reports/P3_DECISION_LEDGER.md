# P3 split-vs-dense decision ledger

Last reconciled: 2026-08-05 23:31 CDT

This file is the compaction-safe source of truth for experimental decisions.
Current implementation, audit, bug, and subagent status is consolidated in
`P3_WORK_STATUS.md`. After any conversation-summary/context compaction, reread
both files and `checklist.md` before editing code, data, commands, or thresholds.

Primary decision history: [P3 decision thread](2012f2f4-1524-4fe4-9fa5-58a20c2e155e)

## User-approved experimental contract

- Question: compare two models on identical inputs when both can attend to every
  required premise, but only dense receives loss on supplied premise tokens.
- Model/tokenizer: pretrained Qwen2.5-0.5B with
  `tokenizer/qwen25-vendored/v1`; vocab 151,936; EOS/pad 151,643.
- Arms: dense and split only, one seed (`42`), one independent 8xA100 or
  8xH100 node/arm; H100 remains the recommended faster final shape.
- Data stream: identical packed bytes/order, 16,384-token context, 13 epochs.
- Loss backend: Liger 0.7.0 fused-linear cross-entropy in both arms and on both
  GPU profiles. It replaces only materialized-logits CE; FlashAttention2,
  labels/masks, summed loss, and the fixed divisor remain unchanged.
- Mask:
  - dense scores all pre-`---` fact/local-assumption tokens;
  - split attends to them but receives no loss before `---`;
  - boundary derived at runtime from `---\nGOAL` token IDs
    `[10952, 15513, 969]`;
  - no sidecar.
- Goal/derivation tokens are supervised in both arms.
- Local Metamath `$e` assumptions: include only decoded/used assumptions in
  `Local assumptions:` before `---`; remove their pushes from visible targets.
- Metamath `(reuse)`: retain in source replay, omit from visible targets, let
  semantic verification reuse an earlier matching expression.
- Training controls: LR 2e-5, warmup 2,400, weight decay 0, global batch
  262,144 tokens, rank microbatch 16,384 tokens. Configs differ only in `arm`.
- No rare-fact upweighting, no name anonymization, no sub-proof re-rooting,
  no separate raw-math phase, no matched-supervised-token cooldown.
- Evaluation:
  - teacher-forced target NLL and next-token match on all families;
  - generated Metamath proof validity is a versioned tri-state metric under
    `p3-metamath-tristate-v1` when `set.mm`, `iset.mm`, and `nf.mm` match
    `metamath_sources.json`; preserve valid/invalid/unknown separately;
  - validity is reportable for `facts_present` and `facts_absent`;
    `facts_corrupted` remains explicitly unsupported because hidden canonical
    rules do not validate the corrupted statements shown to the model;
  - normalized exact output is diagnostic, not general proof validity;
  - all three conditions (`facts_present`, `facts_absent`, `facts_corrupted`)
    run on each family's full context-eligible cohort; the six family cohorts
    total 4,191 source rows, and dense/split are paired on identical IDs within
    family and condition;
  - facts-present is the headline condition; facts-absent and facts-corrupted
    are mechanism diagnostics and do not replace the headline result;
  - facts-shuffled is removed from the evaluation contract;
  - evaluate only the final checkpoint step 23166; do not run intermediate
    checkpoint evaluations;
  - exclude `text + EOS > 16,384`;
  - paired dense/split IDs and matching saved training configs.
- Isabelle rebuild target selected on 2026-08-03:
  `facts + state_before -> tactic + state_after`.
- Mizar/thproofs/prf2/ENIGMA holdout selected on 2026-08-03:
  one pooled draw of exactly 1,000 semantic classes, seed `20260801`.
- Direct-Mizar recovery selected on 2026-08-04:
  include only the independently audited fail-closed unique-label alignment
  expected to recover 5,239 source-backed rows while preserving all existing
  50,114 rows byte-for-byte. Do not include broader contextual-reference or
  inline-justification recovery without a separate decision.
- ENIGMA alternative-proof recovery selected on 2026-08-04:
  add only the conservative audited tier of 2,087 materially distinct traces
  (about 9.67M packed Qwen tokens). Preserve all 27,079 accepted rows, add at
  most one variant to currently single-variant theorems, cap text+EOS at 8,192,
  reject superficial/exact/dead-step variants, and route every theorem variant
  together during pooled heldout planning. Central/high tiers are not approved.
- MPTP fact scope remains the explicit bookkeeping filter; other stable named
  premises remain eligible global facts.
- Metamath heldout identity selected on 2026-08-03:
  sample exactly 500 normalized statement-equivalence classes from the tail
  using total visible row exposure across names, fact values, goals, target
  expressions, and local assumptions. Do not sample rare labels and then expand
  them to common statement classes. All identities/exposures in a selected class
  route together.
- Metamath 16K eligibility selected on 2026-08-04:
  - compute exact fixed-Qwen `text + EOS` length on the final rendered row;
  - drop the whole row when length exceeds 16,384 **before** exposure counting,
    classifier compatibility, and tail-class selection;
  - never truncate or split a proof;
  - persist and cryptographically bind every dropped ID, theorem, token length,
    native row hash, typed reason, tokenizer seal, and aggregate/root;
  - reselect exactly 500 represented eligible statement classes after filtering.
  The accepted local result is 65,122 train, 952 builder eval, and 960 exact
  overlength drops; final generation accounts for all 67,034 source occurrences.
- Evaluator JSONL/manifests use a separate immutable release, linked from the
  packed-token release by exact version and manifest SHA-256.
- No separate evaluator selector is added to the platform form. The evaluator
  release is resolved transitively from the selected token release's pinned
  dependency.

For the current full-run data-readiness scope, evaluator-release platform
integration is deferred and is not a prerequisite for publishing or training
from the packed-token corpus.

## User-approved full-run data-only scope — 2026-08-03

The immediate target is the original **full 13-epoch dense/split training run**,
not a reduced local pilot.

Our controlled deliverable is the data release uploaded through
`s3://edullm-landing` and promoted into `s3://edullm-data`:

- source-backed six-family train/validation rows;
- packed uint32 little-endian token shards under
  `tokens/<family>/{train,val}-NNNNN.u32le.bin`;
- exact manifests, counts, hashes, partitions, and descriptive provenance.

The model tokenizer is fixed and must not change:
`tokenizer/qwen25-vendored/v1`, the published Qwen2.5 tokenizer required by the
pretrained Qwen2.5-0.5B embeddings. Do not retrain or republish a tokenizer.

The platform owns dataset selection, reader resolution, image construction,
run manifests, compute profiles, checkpoints, retries, and S3 output locations.
Do not expand this data-readiness task into:

- new `edullm-data` reader APIs or profiles;
- evaluator publisher/validator/reader infrastructure;
- platform or submission-workflow changes;
- additional checkpoint/export/`run_eval` provenance plumbing;
- changes to OLMo-core library behavior unrelated to producing the token shards.

Training remains the approved 13 complete loader epochs. After fresh
tokenization, recompute the exact rows, tokens, complete batches per epoch,
steps, fact exposures, saturation, and dense/split supervised fractions from the
new bytes. Never reuse v2 counts.

## User-approved operational boundaries — 2026-08-04

- The stale v2 release is not hard-denied. If selected, the P3 entrypoint emits
  one clear rank-zero warning; final scientific runs must select the newly
  repaired release.
- Metamath validity is accepted only through the reviewed tri-state API and
  hash-verified source snapshot. Missing source files make validity unavailable
  rather than silently falling back to a boolean or canonical hidden context.
- License/about/source descriptions are optional, backfillable dataset metadata
  under the eduLLM dataset contract. Never fabricate a license. Whether external
  human/legal approval is required is outside the validator contract.
- Complete local corpus building, tokenization, and real-payload validation, then
  pause before any S3 upload for manual review. Source license evidence may be
  inventoried, but this ledger makes no legal determination.
- Publish only through `edullm_data.publish()` to `s3://edullm-landing`; never
  write `s3://edullm-data` directly. Promotion and cataloging are validator-owned.
- The publish source must contain only audited
  `tokens/<family>/{train,val}-NNNNN.u32le.bin` payloads. Internal done markers,
  caches, control manifests, and preserved v2 bytes are forbidden.
- The already published `tokenizer/qwen25-vendored/v1` is the corpus's sole
  external dataset dependency. Do not republish or replace it.
- Production generation must run and persist the exact P3 deep verifier before
  tokenization. Generic transaction-v2 payloads are intentionally consumable by
  the generic tokenizer, but are not authorized P3 inputs without this gate.
- Production generation must consume the persisted pooled-MML contract rather
  than silently replanning it, and must bind the approved recovered-Mizar,
  low-tier-ENIGMA, and Metamath-overlength acceptance roots.
- Until the platform registry exposes v3, the submission form selects Data:
  None and the command supplies immutable v3 dataset ID/version/tokenizer
  flags. Training still resolves through `edullm_data`, never a raw/local path.

## Fixed constants that may not drift silently

- Sequence length: 16,384.
- Epochs: 13.
- Seed: 42.
- Global batch: 262,144 tokens.
- Rank microbatch: 16,384 tokens.
- LR: 2e-5.
- Warmup: 2,400.
- Hardware: two independent 8xA100 or 8xH100 jobs; the exact fused-loss image
  must pass a 100-step smoke on the chosen final profile.
- Loss implementation: `fused_linear` (Liger Kernel 0.7.0).
- Metamath snapshot:
  `metamath/set.mm@82830c78861b96e906d9868c30c35dbd98be5db5`.

## Current immutable artifacts

- Tokenizer v1 is usable and final unless the user explicitly changes model.
- Corpus `pretrain/formal-proof-premises-500m/v2` is immutable and reproducible,
  but **must not be used for final scientific training or conclusions**.
- v2 exact values (stale after rebuild):
  - packed train tokens: 494,862,336;
  - packed val tokens: 11,862,016;
  - complete batches per loader epoch: 1,887;
  - exact 13-epoch steps: 24,531;
  - nominal consumed tokens: 6,430,654,464;
  - simple `13 × artifact tokens` (not actually consumed): 6,433,210,368;
  - dense supervised fraction: 99.915%;
  - split supervised fraction: 83.803%.
- Recalculate every value after rebuilding. Never copy v2 values into v3.
- Independently accepted local source candidates, not yet published:
  - Metamath: `.p3-work/full13/metamath-16k-v1`, 65,122 train /
    952 builder eval / 960 overlength drops, with exactly 500 represented held
    classes and zero downstream tokenizer drops;
  - recovered direct Mizar: `.p3-work/full13/mizar`, 55,353 rows;
  - prf2: 24,797 rows;
  - ENIGMA low tier: 29,166 rows = 27,079 preserved + 2,087 alternatives;
  - Isabelle: 16,576 train / 590 eval.
- No physical pooled MML v7 artifact, final six-family transaction, fresh token
  release, or promoted replacement dataset exists yet.

## Current code/data state

- OLMo-core branch `edullm/p3-math-split` is pushed at
  `4b5b58b5ad010df3848816317d32177bcd54ca9f`; the minimal
  `compare_arms.py` indentation repair is local and uncommitted.
- Corpus-side P3 builders/docs/artifacts are local and not version-controlled;
  they need a Git identity before a final release can be called reproducible.
- Family-local builders and candidates are accepted. The generation coordinator
  independently binds persisted MML v7, recovered Mizar, low-tier ENIGMA, the
  fixed tokenizer, and the Metamath 16K drop ledger.
- Tokenization/profile code is ready but fresh real shards do not exist.
- Active critical path:
  1. materialize and verify pooled MML v7;
  2. build and deeply verify one immutable six-family transaction;
  3. tokenize and run the real `pretrain-tokens/v1` gate;
  4. pause for manual S3 review.

## Known v2 blockers

- Metamath: local `$e` pushes and `(reuse)` pollute targets.
- ATP: heldout statement/proof leakage and lossy dependencies.
- Mizar: theorem/proof regex crosses later theorem headers.
- thproofs: incomplete/canceled premises.
- Isabelle: tactic-only target instead of approved tactic+state-after content.

## Open design decisions — do not infer

- Original proof-level seen-fact vs new-fact axis is not implemented.
- No user-approved minimum effect/equivalence/non-inferiority margin.
- Random-mask control not approved.
- Fact-block dropout not approved.
- Strong dense facts-present > facts-absent load-bearing gate not approved.
- Under equal-stream Run A, split losing is confounded by fewer supervised
  labels; do not state that it proves the null without qualification.
- New repaired dataset ID/version is not selected.
- Aggregate corpus license metadata remains honestly unresolved but is
  backfillable and not a technical generation/publication gate.
- The eager persisted-MML loader has a known P2 memory/scalability risk
  (at least 3.653 GiB of payload bytes plus parsed copies); no correctness
  failure has been demonstrated and no redesign is approved for the first run.

## Required terminal gates before final submission

1. Every mechanical finding independently verified before fixing.
2. Failing regression observed for every fix.
3. Rebuild all repaired families into a separate versioned directory.
4. Full prompt-completeness, heldout-isolation, alias, gold, source, and
   train/eval accounting sweeps pass.
5. Retokenize and verify artifact bytes, separators, packing, EOS/pad, masks,
   counts, and exposure statistics.
6. Publish a new immutable corpus release; update platform dropdown.
7. Build/security-scan the exact OLMo commit.
8. A10G config dry-run and 8xH100 100-step runtime smoke pass.
9. Only then submit final dense/split jobs.
