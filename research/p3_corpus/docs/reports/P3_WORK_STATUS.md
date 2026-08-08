# P3 formal-proof split vs dense — work status

**Parent-owned status document.** Last reconciled: 2026-08-04 14:43 CDT.

This document explains what has happened in the long-running P3 repair effort:
what each specialist investigated, which defects were confirmed, what is
accepted, what is still under review, and what must happen before training.

Use these documents together:

- `P3_DECISION_LEDGER.md` — user-approved scientific decisions and constants.
- `P3_WORK_STATUS.md` — current implementation, audit, and subagent status.
- `checklist.md` — execution checklist; several old entries still need
  reconciliation against this document before they can be trusted.

The parent agent owns this file. Specialist reports are evidence, not authority:
a change is marked accepted only after an independent review and parent check.

## 1. Executive status

**Overall: NO-GO for final dense/split training.**

The old published `pretrain/formal-proof-premises-500m/v2` is byte-reproducible
but scientifically stale. It must not be used for final training or conclusions.

**Full 13-epoch run remains the target; repaired data is not ready.** Current
work is restricted to the corpus and packed-token release we control. The
published Qwen2.5 tokenizer remains fixed. All family-local candidates are now
accepted, including the 16K-eligible Metamath rebuild. The critical path is the
pooled MML artifact, then immutable six-family generation, tokenization, and the
real payload gate.

### Accepted locally

- Core training path:
  - Qwen weights load after train-module/FSDP initialization.
  - Loss normalization is correct under eight-way data parallelism.
  - Runtime separator masking handles CPU loader batches and CUDA model state.
  - Dense no longer scores impossible packed-document transitions.
  - The loader runs exactly 13 complete epochs, not nine batches of epoch 14.
  - Supervised-token diagnostics aggregate every microbatch.
- ATP source parsing and rendering:
  - complete parent lists and source annotations;
  - local/bookkeeping inputs;
  - final-refutation and topological-closure checks;
  - alternate-proof/alias isolation;
  - ordered prf2-before-ENIGMA exact deduplication;
  - syntax-aware statement identity and quoted atom rendering.
- Evaluator/comparator mechanics:
  - EOS-inclusive per-example sufficient statistics;
  - explicit micro/macro endpoints and denominators;
  - strict paired cohorts, IDs, booleans, hashes, and provenance;
  - ATP local inputs and Isabelle-transition prompt rendering;
  - no unapproved equivalence/non-inferiority verdict.
- Model/tokenizer/checkpoint provenance and export:
  - pinned Qwen revision and weight SHA-256;
  - sealed tokenizer artifact;
  - remote-safe checkpoint discovery and local unsharding;
  - BF16-preserving HF export;
  - evaluator-compatible export metadata;
  - eight-rank enforcement for non-dry runs.
- Atomic generation transaction and six-family production orchestration:
  - exact recursive builder inventories and closed commands;
  - signed input preflight and recomputed sidecar/family checks;
  - descriptor-safe writes, immutable generations, atomic `CURRENT`;
  - complete stage/copy fault matrix.
- Exact source archives are now available and checksum-verified for ATP,
  Mizar/thproofs/current semantic HTML, and Isabelle.
- Persistent family-local candidates are independently accepted:
  - Metamath 16K candidate: 65,122 train, 952 builder eval, and 960 exact
    overlength drops;
  - recovered direct Mizar: 55,353 source-backed rows;
  - prf2: 24,797 rows;
  - ENIGMA: 29,166 rows, including the approved 2,087-row low tier;
  - Isabelle: 16,576 train and 590 eval rows.
- The network-free tokenization path passed 104 focused tests and a 24-shard
  synthetic packed-artifact audit. It is waiting only for the final immutable
  corpus transaction.

### Not yet accepted

- The physical pooled 1,000-class MML artifact. Planning against the recovered
  55,353-row Mizar and 29,166-row ENIGMA roots completed, but the latest local
  attempt stopped before staging on a stale harness assertion that expected
  55,343 rather than the planner's 55,353 direct-Mizar trajectories. No pooled
  candidate was promoted.
- A real persisted-MML production dry run. Fixture/code-level persisted v7
  ingestion and mutation rejection are independently accepted; the real run
  awaits the artifact above.
- Resumable token-shard v3/v4 cache/shard/CURRENT seals, the canonical Qwen
  tokenizer seal, and shard-only publication staging are accepted; fresh
  artifacts do not yet exist.
- Real rebuilt corpus, retokenized artifacts, exact `pretrain-tokens/v1`
  validation, S3 publication/promotion, and the resulting submission-form
  dataset release.

### Resolved operational issues

Two independent read-only reviews previously confirmed four issues. Their
minimal fixes are complete:

- v2 remains warning-only as selected, but now emits one prominent rank-zero
  warning on dry and non-dry invocations; v3+ is silent;
- production and staging use one exact canonical Qwen tokenizer seal;
- publication staging copies only manifest-declared `.u32le.bin` shards into a
  fresh tree, while injected done controls still reject;
- generation documentation now treats license and Metamath validity as
  metadata/evaluation concerns rather than technical generation blockers;
- Metamath tri-state validity is accepted in evaluator schema `p3-eval-v8`
  behind hash-verified `--mm-dir`.

The tokenizer bytes themselves were never changed.

## 1.1 Why so many defects were reported

The running review history previously conflated three very different things.
They are separated here so a rejection is not automatically read as “the
research dataset is bad.”

### A. Upstream research data

The pinned Metamath, Mizar/MPTP, ENIGMA, and Magnushammer source artifacts are
not generally corrupt. Source hashes, official replay, theorem identities, and
large sampled/full joins have held up.

The important exception was not corrupt bytes: the first build paired legacy
MML-1147 `html2` with current MML-1493 `thproofs`. Those are valid research
artifacts from different releases and were incompatible with each other.

Research-paper publication also does not certify our downstream use. The papers
used these files for premise selection, neural conjecturing, or tactic/state
tasks. They did not promise:

- our exact oracle fact block;
- our content target;
- cross-corpus heldout isolation;
- Qwen 16k eligibility;
- one stable name-to-statement map across multiple formal systems;
- our packing, masking, evaluator, or eduLLM release contracts.

### B. Confirmed transformation/experiment defects

The small set of root data defects was in our conversion code:

- Metamath `$e`/`(reuse)` rendering and cross-database label collisions;
- ATP parent/input loss and alternate/duplicate holdout leakage;
- Mizar declaration/reference parsing and mismatched source join;
- Isabelle tactic-only target, clipping, omitted locals, and held-statement
  exposure;
- Mizar↔ATP semantic-class leakage.

These are real because each was reproduced against the pinned raw source and
either changed incorrect rows or exposed held facts. They do not imply the
paper datasets themselves were invalid.

The core training/evaluation defects were likewise our integration code:
pretrained-weight lifecycle, mask device, packed boundary labels, epoch horizon,
metrics, model provenance, and evaluator cohort checks.

### C. New release-tool hardening

Most late review cycles were not finding new mathematical-data corruption.
They were adversarially reviewing infrastructure written during this session:

- atomic generation and crash recovery;
- symlink/FIFO/hardlink/path-swap defenses;
- cache and shard resume hashes;
- re-signed manifest and fsync fault handling;
- separate evaluator-release schemas and platform profile design.

Those checks improve reproducibility, but many would not alter a single proof
row. Counting every failed hostile filesystem test as another “dataset bug”
materially overstates the data-quality problem.

### D. Findings that were refuted, stale, or overstated

The review process did produce illegitimate or overstated concerns:

- “Split must not mask Metamath local assumptions” was refuted: masking them is
  the approved intervention.
- “One seed blocks the experiment” was refuted: it supports a comparison
  conditional on the two seed-42 checkpoints, not a population-level claim.
- “Evaluation denominators are absent” was refuted; the actual issue was naming
  and conditioning.
- “Byte-identical shuffled prompts are a deterministic-order bug” was mostly
  refuted: one-fact rows and identity permutations are expected, though their
  effective denominator must be reported.
- Three-to-five seeds, cross-prover formal replay, and targeted corruption were
  recommendations, not approved requirements.
- “Sources are unavailable” became stale once exact archives were recovered.
- Several Mizar/Isabelle candidates were rejected only for lint, schema adapter,
  or publication wiring after their row contents had already passed.
- The stage-one “dataset-format” finding was framed incorrectly. The actual
  eduLLM-published layout `tokens/<family>/train-*.u32le.bin` is compliant with
  `pretrain-tokens/v1`. The real defect is only an internal producer/consumer
  mismatch: transaction roles/schemas versus tokenizer assumptions.

### E. Why the defect stream lasted hours

- The user explicitly requested a zero-error, adversarial, independently
  verified gate. Each repair therefore triggered another hostile review.
- Scope expanded from corpus parsing to training, evaluation, provenance,
  publication, crash atomicity, S3, and platform-profile design.
- Multiple agents edited shared files concurrently. Some reviews inspected
  intermediate hashes and were invalidated by later changes.
- The corpus project was grafted into a repository dominated by an older
  Wikidata/requery project. Most P3 corpus files are untracked, so there was no
  clean baseline diff or stable commit boundary.
- The OLMo worktree currently contains roughly 10,900 added lines across 23
  tracked files, plus new files. Large newly written surfaces naturally produce
  more review findings than a narrow patch.
- Some agents initially reasoned from summaries or hand-built internal
  contracts rather than the two eduLLM dataset skills and exact image-pinned
  `edullm-data` source. Persistent rules now require both skills first.

### F. Current confidence policy

- “Accepted” means source-backed or adversarial evidence, independent review,
  and parent confirmation.
- “Rejected” may mean row content is wrong, integration is incompatible, or
  release hardening is incomplete; the status must state which.
- Internal generation paths are not judged by eduLLM publication naming rules.
  Published token paths are.
- Active concurrent work is not promoted to accepted until its final hash is
  independently reviewed.

## 2. Current scientific contract

The unchanged core experiment is:

- pretrained Qwen2.5-0.5B;
- dense and split arms only, seed 42;
- identical packed token stream/order/compute;
- dense scores the complete pre-`---` block;
- split attends to that block but receives no loss on it;
- both score goal/derivation tokens;
- 16,384-token context;
- literal 13 loader epochs;
- one independent 8xH100 node per arm.

Recent approved data/release decisions:

- Isabelle target:
  `facts + state_before -> tactic + state_after`.
- Mizar, thproofs, prf2, and ENIGMA share one holdout namespace:
  **1,000 pooled semantic classes**, seed 20260801.
- MPTP scope remains the existing policy:
  explicit bookkeeping prefixes are local inputs; other stable named premises
  remain eligible global facts.
- Evaluator inputs will be a **separate immutable release**, linked from the
  token release by exact dataset version and manifest SHA-256.
- The existing token-release selection resolves that evaluator dependency
  transitively; no separate evaluator selector/UI is added.
- No additional dataset is approved; the independent scout/reviewer rejected
  every candidate.

## 3. Important corrected numbers

Values in old documents that say 24,540 steps are stale.

For published v2 only:

- packed rows: 30,204;
- global batch: 16 sequences;
- complete batches/epoch: `floor(30,204 / 16) = 1,887`;
- exact 13-epoch horizon: **24,531 steps**;
- nominal consumed tokens:
  **6,430,654,464** (`24,531 × 262,144`).

The pooled 1,000-class holdout was estimated on stale v2 text to remove:

- about **8.32–8.83M Qwen tokens per epoch**;
- about **108–115M token exposures over 13 epochs**;
- roughly **2%** of current training compute.

The final values must be recomputed from rebuilt v3 bytes.

## 4. Workstream status and defects

### 4.1 Core training — accepted

Confirmed defects and fixes:

- `_sep` was on CUDA while loader/mock IDs were still on CPU: first forward
  would fail. The separator is now CPU-resident and compared on the input device.
- Dense scored the next document's first token from the prior document's EOS
  state despite document-isolated attention. Both arms now mask that transition.
- The old step calculation entered epoch 14. Training now uses exact loader
  epoch semantics and validates the built loader horizon.
- Supervised fraction logged only the first of two rank microbatches. Metrics now
  sum live-token counts over all microbatches and reduce across ranks.
- Fixed-divisor/FSDP math, Qwen architecture, tied weights, and post-init DCP
  loading were independently confirmed.

Remaining evidence gap: a physical 8xH100 runtime smoke.

### 4.2 ATP — persistent production inputs accepted

Confirmed defects that were fixed:

- six-parent truncation and lossy inference annotations;
- missing external/bookkeeping inputs;
- numbered `ccN_`/`fcN_`/`rcN_` classification;
- `file(..., unknown)` provenance collapse;
- non-refutation traces and unresolved/late/cyclic parents;
- quoted step/parent/rule ambiguity;
- ENIGMA `#N` alternate-proof leakage;
- statement aliases and syntax-insensitive hash collisions;
- family-wide exact prf2/ENIGMA duplicates before fact counting;
- missing source/empty rebuilds leaving stale outputs active.

Independent final review accepted the ATP implementation. Exact source archives
for prf2 and four ENIGMA runs are downloaded and checksum-verified.

Fresh persistent recursive-source roots are independently accepted:

- prf2: 24,797 rows,
  `fdde1aececef6de1c88cac8e17945c7a55491bd7bb527784c26099f67f63ab3d`;
- ENIGMA: 29,166 rows = 27,079 byte-identical base rows + 2,087 approved
  alternatives; train SHA
  `7fddf832938404f6e76f33fae06a6e8731b923cde65d9c32795288ac4250a3f7`;
- ENIGMA additions: 9,655,618 text+EOS tokens; added SHA
  `8c50c325c7f42ec4d6f3f2abbfc2f779b07a6e73c19faa95edb197a69a84c5ad`.

All 53,963 accepted rows replay cleanly. The remaining eight traces reference undeclared
reserved E IDs and are correctly typed/excluded. Exact source archives, trees,
metadata, quality/schema roots, audit, and acceptance seals were independently
recomputed. These roots are now feeding the real pooled MML split.

Remaining work:

- complete pooled MML routing and transaction ingestion;
- run final source-backed deep verification.

### 4.3 Mizar and thproofs — candidate accepted

Confirmed defects found over multiple review cycles:

- theorem regex crossed later declarations;
- canceled statements/facts survived;
- nonstandard local labels such as `Def5a` were omitted;
- qualified `OTHER:Lm4` could bind the current article;
- fact values were not checked against source;
- source-mismatched joins could exit successfully;
- proof-completion parsing first over-accepted garbage, then rejected valid
  boundaries;
- temporary production floors did not match the real release.

Current-source recovery changed the path:

- legacy `html2` is MML 1147 and cannot be paired with current thproofs;
- current Mizar 8.1.15 semantic HTML has 91,114 theorem identities;
- all 76,696 current thproof names join it;
- a new `mizar_current_index.py` adapter provides canonical statements,
  contextual local-label resolution, source goals, and pinned tree hashes.

The semantic index is integrated into active builders and deep verification.
The earlier independently accepted isolated candidate contained:

- 50,743 rows (49,778 train / 965 temporary eval);
- 50,752 accepted before 9 exact duplicates;
- 58,658 complete of 69,698 explicit proof-bearing extracts;
- 76,696/76,696 names joined to 106,317 semantic statements;
- zero emitted goal, fact, citation, reference, or contextual-label discrepancies;
- clean deep verification and a clean shared Ruff/test gate.

Legacy html2 remains a nonproduction compatibility path. Remaining work is to
replace the temporary 500-fact split with the pooled 1,000-class split, publish
through the accepted transaction contract, and resolve redistribution licensing.

The direct-Mizar row candidate and shared `verify_corpus` path remain sound, but
post-generation production verification is currently rejected:
`build_p3_generation._verify_current_mizar_index()` still dispatches both
`mizar` and `thproofs` through the thproof resolver. Independent replay found
the same 94 disagreements among 50,114 direct-Mizar rows, including proof-local
`Lm2`; raw and manifest roots were unchanged. The isolated repair now dispatches
direct Mizar and thproofs through their family-native resolvers, reports zero
disagreements over all 50,114 rows, and preserves the accepted candidate roots.
Independent re-review reproduced zero native-resolver disagreements while the
old resolver retained exactly 94, and confirmed unchanged code/candidate roots.
The post-generation dispatch is accepted.

The user-approved conservative recovery is independently accepted persistently
under `.p3-work/full13/mizar`:

- 55,353 rows and 42,851,393 Qwen text+EOS tokens;
- the original 50,114 rows remain a byte-identical ordered prefix;
- exactly 5,239 unique-label/nearest-anchor additions;
- zero direct/thproof/recovered identity or text duplicates;
- raw SHA
  `54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f`;
- recovery identity/source-binding roots
  `048f47cf…8cd` / `790c86db…65c`.

All additions replay source/index/facts/citations/local context/goals/targets,
remain under 16K, and exclude broader contextual or inline recovery. The pooled
MML split must use this recovered root; any split from the 50,114-row candidate
is stale.

### 4.4 Isabelle — real corpus candidate accepted

The tactic-only implementation was replaced with adjacent transitions:

- input: full theorem + state-before + global/local premise block;
- target: tactic + state-after;
- aliases and local assumptions are preserved;
- ambiguous global names, malformed premises, unchanged/final/aborted
  transitions, high state carryover, duplicates, and overlength rows are dropped;
- source and tokenizer bytes are pinned;
- heldout splitting excludes entire related trajectories.

First real pinned-source build:

- 18,906 eligible rows;
- 16,578 train;
- 590 direct eval;
- 500 held facts;
- 14,020,370 train tokens/epoch;
- 50/50 raw-adjacent samples matched.

The first build was correctly rejected because two held statements remained in
train:

- `iso_assoc` through a local assumption;
- `Rel.substClosedSubset` through a `lemma `-prefixed theorem/goal.

Those paths were fixed and a second full source build was independently
accepted:

- 18,906 eligible rows;
- 16,576 train;
- 590 byte-identical direct eval rows;
- 1,738 typed sibling drops and 2 typed held-statement exposure drops;
- exact accounting: `16,576 + 590 + 1,738 + 2 = 18,906`;
- 500 held facts / 595 direct uses;
- zero held-name or held-statement exposure across every train field;
- train/eval trajectories disjoint;
- 50/50 raw adjacent-transition samples replayed;
- maximum Qwen text+EOS length 14,427.

Accepted candidate hashes:

- train:
  `7dc04aaa00b4bfae1d38756c207af5b06e371da67fb9b9bd05302aae025180f1`;
- eval:
  `7276e35777fab4c8152732188559dbad8e89be885feafff3c146efbf974906fb`;
- manifest:
  `78c03d731b1e5478c5755e023add4cf492760df8ac295b0cad9127b0eec10119`.

The current-schema rebuild at `.p3-work/full13/isabelle` is independently
accepted for generation ingestion:

- train with source metadata:
  `ea5f881afd84dd519de1a5bfc643d16dd89c794a4ac158680d95aa7cbad46b26`;
- eval with source metadata:
  `4dfa8182fb666936d15ff2c618efe42e5e791d68012570f5a146819b532621ba`;
- source root:
  `1d5dd6b6f7a58d4b4c6a6163f03014cd129712b8d3e96d786e0698f4bd37b995`;
- quality root:
  `ee441033ce8732fc9627918c0b09366e0e78e53d486bc335022903a991e3d7de`;
- schema root:
  `37d7eeaf65fae23efed6d37ca24c031d0c59a7dc75eb65723aa5a65c52fdf432`.

Projected row content excluding source metadata exactly recovers the previously
accepted train/eval hashes. Remaining work is transaction ingestion and final
packed-token validation; Apache-2.0 attribution is nonblocking metadata.

### 4.5 Metamath — training candidate accepted; validity metric separate

Earlier code was unsound:

- discarded `$d`;
- ignored generated `$f` typing;
- rebound concrete substituted tokens as metavariables;
- converted hypothesis-budget exhaustion to invalid;
- rejected/crashed on many valid source traces.

The replacement implements:

- source replay with `$f`, `$e`, and `$d`;
- typed, disjoint-aware generated trace checking;
- nonrecursive bounded matching;
- valid/invalid/unknown tri-state;
- pinned official `metamath-exe` oracle.

Reported implementation evidence:

- 69,800 pinned source proofs replayed with zero failures;
- 1,151 cleaned eval traces: 133 valid, 1,018 explicit unknown,
  zero invalid, zero crashes.

The cross-database identity repair is independently accepted:

- only the 394 labels with conflicting set/iset/nf statements are qualified;
- per-database fact rendering and verifier prefix checks are implemented;
- exact-statement alias scanning was added for heldout isolation;
- the implementation reports zero wrong statements and 69,800/69,800 replay.

The post-generation Metamath isolation verifier now independently scans exact
canonical statements in facts, goals, proof-step expressions, and local values,
with own-goal exposure typed as `heldout_own_proof`. Its focused tests and all
152 generation tests passed, but independent review rejected the end-to-end
fix: `_normalize_metamath_package()` still emitted name-only routes and therefore
rejected correctly split statement-exposure rows before the later verifier.
The repair now uses one shared context/classifier in normalization and
verification; its production-path regression, mutation gates, 163 generation
tests, synthetic generation, and Ruff pass. Independent re-review confirmed the
production normalizer and verifier share the statement-aware classifier, all
hostile route/drop mutations reject, and frozen hashes remained unchanged.
The old policy sampled rare labels and then isolated normalized statements.
Rare aliases therefore selected very common classes (`A = A`, `a1i`). The
accepted replacement samples exactly 500 normalized statement classes by total
visible row exposure and routes each selected class together.

The old 66,074-train/960-eval candidate is retained only as a historical,
byte-stable input. Exact Qwen tokenization showed that it relied on downstream
whole-row dropping: 960 rows exceeded 16,384 tokens, and dropping its 60
overlength eval rows would leave only 465 of 500 held classes represented.

The user therefore approved filtering whole rows by exact fixed-Qwen
`text + EOS` length **before** exposure counting and heldout selection. The
independently verified replacement at `.p3-work/full13/metamath-16k-v1`
contains:

- source: 67,034 rows / 141,077,915 text+EOS tokens;
- eligible: 66,074 rows / 113,086,656 tokens;
- train: 65,122 rows / 109,768,291 tokens;
- builder eval: 952 rows / 3,318,365 tokens;
- exact overlength ledger: 960 rows / 27,991,259 tokens;
- final generation accounting:
  `67,034 = 65,122 train + 494 eval + 458 heldout_own_proof + 960 overlength`;
- exactly 500 represented tail classes and zero selected-class exposure across
  every train field;
- 69,800/69,800 source proofs replayed with zero failures;
- maximum retained length: 16,384 train / 16,218 eval;
- zero downstream tokenizer overlength drops;
- train/eval SHAs `7ef07b9e…b171` / `aec4cd5d…17bd`;
- drop-ledger/heldout SHAs `835e9e93…407f` / `d309502b…5cfd`;
- canonical ledger and entries roots `40bf7758…46ff` /
  `7002b44d…b6da`.

The generation coordinator now requires and binds the exact ledger and fixed
tokenizer. Independent mutation checks confirmed complete 67,034-occurrence
accounting and fail-closed metadata.

Common classes `A = A`, `a1i`, and `|- QQ e. _V` have 13,300, 10,194, and
41 visible-row exposures respectively and are correctly ineligible for the
1–2-exposure tail. The prior two QQ local-assumption findings are superseded,
not leaks.

The generated-output verifier and evaluator integration are now independently
accepted:

- all 494 sealed gold traces validate, with zero invalid and zero unknown;
- deep recursion and cyclic syntax search fail conservatively to uncached
  `unknown`, never `valid`;
- public verifier schema is `p3-metamath-tristate-v1`;
- evaluator schema `p3-eval-v8` requires `--mm-dir`, verified source hashes,
  all three source databases, and reconciled per-example/aggregate tri-state
  counts;
- comparator schema `p3-comparison-v4` matches source provenance across arms,
  excludes unknowns from the decided denominator, and passes end-to-end
  print/write coverage;
- `facts_present` uses visible canonical facts, `facts_absent` uses canonical
  row facts withheld from the prompt, and `facts_corrupted` is explicitly
  excluded from validity.

The complete checker/evaluator/comparator suite passes 178 tests. Raw
generations remain preserved for later independent revalidation.

### 4.6 Shared MML semantic holdout — active

Legacy separate Mizar/ATP holdouts leaked across formal representations:

- 629 held semantic classes were exposed in sibling training data;
- 1,303 context-eligible eval rows were affected.

Approved mapping:

- `ARTICLE:N` ↔ `tN_article`;
- `ARTICLE:def_N` ↔ `dN_article`;
- no guessed mapping for local/generated/scheme-instance names.

The isolated pooled 1,000-class planner and typed contract loader are
independently accepted. They provide:

- deterministic class selection;
- ATP dedup before counting;
- exact cardinality;
- representation-scoped statement hashes;
- direct/own/alias/target exposure paths;
- native-byte partition plans and root-linked projections;
- complete shard/eval/drop/sidecar inventory verification;
- route-bound manifest roots and canonical path ordering;
- production-vs-test fail-closed policy.

The code-level production planner/loader and persisted-contract generation
boundary are accepted. Remaining work is to materialize the physical artifact,
then run the shared verifier against those persisted bytes and roots.

The isolated semantic module now has a quote-aware ordered delimiter stack and
rejects malformed ATP formulas/source structures before planning. Its updated
contract tuple is manifest v7, policy v8, mapping v2, statement v4, ATP dedup
v5, direct-Mizar/thproof dedup v1, compatibility v7, source policy v3, loader
v7, canonicalization v4, and tuple v5. The
implementation is independently accepted after hostile delimiter, wrapper,
native-replay, downgrade, and tuple probes; all 150 scoped tests and Ruff pass.
Generation now reloads and verifies the full persisted manifest-v7/tuple-v5
contract without replanning and passed independent mutation review. The
evaluator packager is separately accepted; OLMo `run_eval` expansion remains
outside the data-readiness scope.

The production policy has a deterministic finalization seam and drops all
50,103 direct-Mizar/thproof duplicate trajectories while retaining 640
thproof-only rows. It now pins the approved 55,353-row Mizar recovery and
29,166-row ENIGMA input. Persisted-contract ingestion is independently accepted
at the generation boundary, but no physical v7 pooled artifact exists yet.

The latest full planning attempt sealed manifest root
`7fc1bf055ad81d5dbe809740703246eb98e97eece48dd8096c0322daadc11f21`
and reported 55,353 direct-Mizar trajectories. An external harness assertion
still expected 55,343 and stopped the attempt before atomic staging. This is the
current immediate blocker; no partial candidate was promoted.

Manual-attention design risk: the final MML loader eagerly retains at least
3.653 GiB of artifact bytes plus split/parsed copies. Two read-only reviews
agreed on the memory mechanism and P2 severity, but not on a concrete runtime
failure; treat it as a scalability risk, not a confirmed correctness defect.

### 4.7 Evaluator/comparator — accepted mechanics, integrations pending

Accepted mechanics:

- exactly one EOS scored;
- combined BPE suffix accounting;
- per-example NLL sum/token/correct sufficient statistics;
- explicit token-micro and example-macro endpoints;
- paired ratio bootstrap;
- explicit source/context/evaluated/attempted/budget denominators;
- duplicate/cohort/config/provenance/hash checks;
- ATP local-input and Isabelle-transition rendering;
- global train-visibility classification;
- descriptive output only; no hidden 2-point equivalence verdict.

The sole comparator had a committed indentation error and could not import.
The parent applied the minimal indentation-only repair: all 78 comparator tests
pass, all 15 P3 runtime scripts parse, and a two-condition CLI smoke succeeds.
This local repair has not been committed.

The independent stage-two provenance re-review also confirmed, against unchanged
file snapshots:

- exported arm/provenance is checked before Transformers model loading;
- exact single-file and sharded safetensors inventories reject mutation,
  missing/extra shards, index changes, non-BF16 tensors, and broken ties;
- direct `mizar-proof-v2` evaluation reconstructs the exact training prompt
  without injecting a local-assumption block;
- dense/split comparison permits only the approved arm/output identity
  differences while independently binding each trained-weight root;
- non-dry training, export, evaluation, and comparison consistently require and
  propagate `source_commit` and available platform identity.

Evidence: 41 focused hostile tests, two real Mizar builder tests, independent
sharded mutation/index/missing/extra probes, and cross-repository prompt parity.

Pending integrations:

- sound, independently accepted Metamath tri-state;
- sealed evaluator release loader;
- authoritative MML semantic manifest root.

`run_eval` cannot complete the sealed-release integration within its local
files. The final image lacks the accepted reusable evaluator loader, and the
current `edullm-data` reader, `train_platform.py`, and `export_checkpoint.py`
preserve dataset ID/version but not the token corpus logical root plus evaluator
five pins. Consequently a checkpoint cannot yet prove which exact corpus and
evaluator release trained it. Closing this requires a deliberate scope
expansion across the reader metadata API, image/dependency packaging, training
config, export metadata, and evaluation; no `run_eval` files were changed by
the preflight.

Those evaluator/checkpoint provenance changes are outside the corrected
data-readiness scope and do not block publishing the pretraining token corpus.
The platform already records the selected dataset release, code commit/image,
command, compute profile, and output/checkpoint locations.

### 4.8 Model/tokenizer/export — accepted locally

Accepted:

- pushed OLMo branch commit
  `4b5b58b5ad010df3848816317d32177bcd54ca9f`;
- Qwen revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`;
- model weight SHA-256
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`;
- sealed tokenizer artifact/version and behavior;
- arbitrary post-YAML overrides rejected;
- one W&B crash record per distributed job;
- obsolete local trainer fails fast;
- remote checkpoint discovery/torn-checkpoint checks;
- BF16-preserving tied HF export;
- `p3-model-export-v1` metadata accepted by evaluator;
- eight ranks required for every non-dry run;
- exact publication-profile gate bound to
  `edullm-data@38bf831a6c3f445e394784018441fd59288b876c` / package 0.5.0.
  Its structural FakeS3 fixture passes nested labels, train/val partitions,
  uint32 format, counts, and shard names. The real producer-to-staging seal path
  now passes. The offline real-payload publish→validate gate is implemented and
  reports image-pinned local-256 and deployed-128 policy results separately;
  execution awaits fresh six-family shards and cannot authorize upload on skip.

Remaining live gates:

- exact-image build/security scan after commit;
- authenticated S3 export/unshard/upload;
- A10G published-artifact dry-run;
- 8xH100 runtime smoke.

### 4.9 Release integrity — active

Confirmed release-chain defects:

- old splitters could publish mixed sibling generations;
- unnamed positive drops could pass verification;
- same-size stale packed shards could be resumed;
- evaluator JSONL was not bound to the packed-token release;
- Isabelle v2 was not understood by the shared deep verifier.

Accepted release infrastructure:

- the accepted atomic six-family generation coordinator has physical occurrence
  IDs, typed drops, mandatory validators, immutable generations,
  descriptor-safe writes, atomic `CURRENT`, and complete stage fault coverage;
- accepted v3 token caches/shards with per-file SHA-256, exact transaction-v2
  input validation, CURRENT commit locking, strict resume, and fault cleanup.

The coordinator's real-builder adapters and provenance contracts are also
accepted after 200 generation, 171 semantic, and 18 boundary tests: Metamath and
thproof argv are satisfiable, builder-native source metadata is preserved,
outer family roots remain distinct from row-source roots, and license/Metamath
validity status no longer blocks token-data generation.

Subsequent independently verified integration work added:

- persisted MML v7 ingestion without replanning, with exact raw/inventory/root
  mutation rejection;
- mandatory ENIGMA low-tier acceptance/tokenizer binding;
- mandatory Metamath 16K drop-ledger/tokenizer binding and complete typed-drop
  accounting.

The separate `p3-evaluator-release-v1` package and token dependency contract are
independently accepted. The repaired loader derives canonical projections from
persisted authoritative MML bytes rather than mutable workspace policy/code.
Independent evidence includes 346 combined tests, 50 concurrent fresh-process
cycles across hash seeds, randomized mapping/list-order probes, persisted-byte
mutation rejection, exact schema parity, and unchanged reviewed hashes.

The evaluator packager already calls the semantic module's production
`load_holdout_contract()`, so the accepted manifest-v5/policy-v5/statement-v4/
dedup-v5/canonicalization-v4 tuple is enforced at that boundary. Remaining work
is the OLMo token/evaluation consumers and platform profile support.

The token producer is now data-only: it binds the corpus generation, fixed
Qwen2.5 tokenizer, packing/build contract, and six-family train/val inventory,
with zero evaluator references or accepted legacy evaluator arguments.
Production construction now emits the exact seal required by staging, and the
documented fresh staging procedure copies only manifest-declared shards.
Token/contract tests pass 97 cases plus mask tests; real staged-payload and
packed-artifact gates remain unavailable until fresh v3/v4 shards exist, and no
skip authorizes publication.

Platform action still needed:

- register `p3-evaluator-release-v1`; `text-corpus/v1` is insufficient.

### 4.10 Additional dataset search — closed

The scout found two near-pass candidates; independent review rejected both:

- Pile-of-Rocq: incomplete tactic/automation dependencies, weak joins, duplicate
  environments, copied states, and unresolved aggregate licensing.
- Kontroli/Dedukti HOL Light: generated positional low-level names and expansion
  overhead rather than stable theorem-level facts.

No dataset is recommended or awaiting manual consideration.

## 5. Specialist work ledger

### Data builders and holdouts

- [Metamath local assumptions implementation](d027324c-8d40-4489-84ee-71857a07cda3):
  implemented relevant `$e` assumptions and `(reuse)` removal.
- [ATP repair](63ace9d5-efe1-4568-ba63-88465b4ad602):
  implemented three TDD repair rounds; final code accepted.
- [ATP verifier](6d50e24c-e352-4e7b-8d48-76630099cddf):
  repeatedly rejected incomplete fixes, then confirmed final ATP integrity.
- [Mizar/thproof repair](1f3e29d6-5830-417c-aa99-da98e794e179):
  implemented declaration, reference, source, and completion repairs; now
  integrating current semantic index.
- [Mizar verifier](0ea4b6e6-c133-42e5-bab9-c0f34a2cfb5a):
  found omitted local labels, fact-source gaps, permissive/over-strict proof
  boundaries, and incompatible floors; latest verdict remains rejected pending
  current-index integration.
- [Isabelle intent audit](2d1c49a3-d8ad-4ee2-9f4e-bd85307714c5):
  reconstructed the intended Magnushammer next-state target.
- [Isabelle design verifier](4f999e4d-747a-47d6-9269-785cce6533e4):
  confirmed adjacent-state semantics and repeatedly checked builder/docs.
- [Isabelle builder repair](7f7cd5c6-9fea-4816-a396-584dbe69bf57):
  implemented streaming adjacent-transition builder and is fixing two real
  held-statement leaks.
- [Real Isabelle build](b13c6fe4-df5b-4a12-b13e-1c4a4c32a0d4):
  ran the pinned 2.3 GB source build, reconstructed all outputs, and rejected the
  candidate on two leaks.
- [Cross-family leakage verifier](c9c8e91a-cde8-45a3-ac1a-54bf00e35ae1):
  proved the 629-class Mizar↔ATP leak and adversarially reviews the pooled planner.
- [Pooled MML holdout implementation](faf1f33c-9098-4eff-bc9c-fecbec884624):
  built the isolated 1,000-class planner and is sealing its contract loader.
- [Holdout token-cost analysis](51207aa3-916d-4224-9e3a-6f72d716c99e):
  measured pooled versus two-strata training-token loss.

### Training, evaluator, and export

- [Training runtime audit](022f0562-4cba-4e3f-b319-33c9281baa7c):
  found separator-device, packed-label, epoch, metric, provenance, and export bugs.
- [Core training verifier](b7c5e8dc-0df9-46ab-bcfa-4eb05a3f0183):
  independently confirmed all four core repairs.
- [Core training repair](b79d4089-81d1-426e-bbdb-d3c6926fd88d):
  fixed device, boundary-label, epoch, and metric defects with TDD.
- [Evaluation validity audit](91a30a26-2253-4b29-9a30-8631fdf1e99d):
  found stale schemas, metric, verifier, cohort, statistics, and export blockers.
- [Evaluator verifier](f0895ada-986a-4f5f-905c-1ae4551afb67):
  repeatedly adversarially reviewed the evaluator; final mechanics accepted.
- [Evaluator repair](11787562-6a95-4933-a7fc-e745ed9971db):
  implemented EOS, sufficient statistics, cohorts, provenance, and strict
  reportability checks.
- [Metamath soundness audit](bc4df5bf-73a1-4a0b-88d5-5a12151f4222):
  proved the old checker unsound; its later re-review was aborted.
- [Sound Metamath verifier implementation](a37f3d6d-c72c-41ee-901c-0694a5bdcdf7):
  implemented typed/disjoint tri-state verification and official oracle.
- [Provenance/export verifier](b849facd-e6c8-4033-a188-e4c4a8c87785):
  found mutable model, obsolete entrypoint, S3, rank, and dtype bugs; confirmed
  the final local repair.
- [Provenance/export repair](f029d9a5-4f99-41aa-b0aa-c01e02a85cc1):
  implemented pinned model/tokenizer, sealed controls, remote export, BF16, and
  evaluator metadata.
- [Stage-two model/arm provenance review](c33fa4e3-857a-4fb8-bd0a-4bc4f9a9e2ba):
  independently confirmed findings 3, 4, 8, 9, and 10 against unchanged
  snapshots with 41 hostile tests, two real Mizar fixtures, sharded-weight
  mutation probes, and exact direct-Mizar prompt parity.

### Broad audits and release infrastructure

- [Master issue finder](a99b0846-adb8-4134-aeb0-15d1813872f8):
  produced the cross-system P0/P1/P2 issue ledger.
- [Corpus/artifact audit](20857ec8-cbd1-4cfd-b9e4-8d95e5b1c693):
  proved existing byte/packing integrity while rejecting scientific readiness.
- [Release pipeline audit](fa460722-9a05-4d6c-b707-77525dcf65a3):
  found mixed generations, stale resumes, schema, binding, and accounting gaps.
- [Release-chain verifier](3ae185ca-4394-4f5a-8e23-9d987d14fc8d):
  independently reproduced release defects, then accepted the final transaction
  and six-family production orchestration.
- [Atomic generation implementation](cb023fa7-2c45-4abc-9e4c-26564700fd8d):
  built the independently accepted descriptor-safe transaction layer.
- [Token-shard seal implementation](4ea405d0-e7e9-4539-9cf8-abb26175012b):
  implemented SHA-bound v3 resume.
- [Token-shard verifier](b62854b6-3ec1-4fd4-ad5d-1d33ed21bc48):
  independently accepted cache/shard/CURRENT resume and commit semantics;
  evaluator dependency remains.
- [Evaluator release implementation](11497466-32f4-4d80-8b15-612b845030c4):
  implemented separate linked evaluator package.
- [Evaluator release verifier](dd6fb339-d6c8-4183-a25e-61a196e0a085):
  independently reviewing package seals and dependency binding.
- [Source recovery](e9e2c3ff-129c-4565-bf10-ffe7a3269da6):
  located and hashed exact ATP, current Mizar/thproof, and Isabelle sources.
- [Current Mizar source adapter](1c16a9ab-0ddf-42be-84ac-4c241d50ab67):
  built semantic HTML index and source manifest APIs.
- [Persistent ATP rebuild](5fa8f18c-ab0c-4a9d-9850-005c8599d2ec):
  reproduced accepted prf2 and low-tier ENIGMA roots, hashes, source replay, and
  the 27,079-row ENIGMA byte prefix in persistent storage.
- [Tokenization preflight](de1ee329-a6fc-4b46-bb5c-82edd8706a6e):
  passed 104 network-free tests and the synthetic multi-shard packed/profile
  gates; real-byte execution awaits final generation.
- [Metamath 16K rebuild](7112d9ec-ac0f-40dd-8207-21599df8d2c9):
  filtered and ledgered 960 whole overlength rows before selecting 500
  represented classes, then built the accepted persistent candidate.
- [Metamath 16K verifier](153e4da3-d93d-48f0-a96a-d51e5cda8122):
  accepted source replay, isolation, exact token accounting, and zero downstream
  drops while identifying the stale generation adapter.
- [Metamath generation binding](fefc9269-96ca-498a-9fea-e7787bb46424):
  bound the exact drop ledger and tokenizer through production generation.
- [Metamath binding verifier](c4c32883-3402-457c-90b2-723e3e95b575):
  independently reproduced all 67,034 routes and mutation rejection.

### New-data search

- [Dataset scout](11112cf9-dfee-4c24-a555-2335f861e4c8):
  found only Pile-of-Rocq and Dedukti HOL Light as near-pass candidates.
- [Dataset recommendation verifier](27a72201-71c8-449a-861d-906c65f6acc1):
  independently rejected both; no recommendation remains.

## 6. Verified source inventory

- Metamath snapshot:
  `set.mm@82830c78861b96e906d9868c30c35dbd98be5db5`.
- Qwen:
  revision `060db6499f32faf8b98477b0a26969ef7d8b9987`,
  weights SHA-256
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- prf2 archive SHA-256:
  `e57c9e799132ee0c05bb8e956620f8c9e49346b78eb8d1eaa0bf3d210d89a7d5`.
- ENIGMA archive SHA-256:
  - mzr01 `06ddbf863ff7f6c3421d158f106a33b5932dc2b5dfee8d5ba0fb7bab027afcd0`;
  - mzr03 `4280e5ed25f5ec3052a53449c0959e8adae913ee8fb310afa4a8702ce9907dcd`;
  - mzr02 `5c02524146b90028712cda6fffae362ac5ea2d40f74b993ea4968edf1b4f06ba`;
  - mzr08 `8135d36c26e020f16d36a1dc7d1828d597fd92694bb538ef489fca568a42ff7d`.
- Mizar 8.1.15 archive SHA-256:
  `cfc32c3e05d5d93c595934e26d4d3b4e399f95a75da7df08359eb9ee73ae6e2e`.
- Current semantic HTML archive SHA-256:
  `e988481577e4e5cc25a5c96c4e86a7de612447088b20781a2680b0e6fc974eee`.
- Current thproofs archive SHA-256:
  `665b17fea382d23168998a4bd1fd91736baf59c1fa3927f8c656d9886fdc3433`.
- Isabelle source SHA-256:
  `aa71609de90fee138835cfdf9e954becb1b231a293ac19bd98951e6d8bec8e7d`.

All listed non-Metamath archives are in isolated `/tmp/p3-sources`.

## 7. Current active work

At the time of this snapshot:

- resolve the stale direct-Mizar trajectory assertion and materialize the
  already planned pooled MML v7 artifact;
- run the real persisted-MML generation dry run;
- rebuild a fresh immutable corpus generation and rerun deep isolation,
  source replay, mutation, and accounting gates;
- pack fresh uint32 train/val shards with the unchanged Qwen2.5 tokenizer;
- run the exact pinned `pretrain-tokens/v1` publisher/validator gate;
- publish only the token corpus through `edullm-landing`, wait for promotion,
  and record the new `dataset_release` value;
- recompute the exact 13-epoch step count and exposure statistics from the
  promoted bytes.

External operational prerequisites remain AWS broker authentication, any human
legal approval required before upload, validator promotion/cataloging, exposing
the repaired release in the submission form, and building/scanning the exact
OLMo commit used for training.

The Metamath soundness and reportable integration reviews are complete. Native
Mizar validity remains out of scope because the exact binary redistribution
terms/build pin are unresolved and the checker is operationally expensive.

Out of scope: evaluator platform profile registration, `edullm-data` reader API
changes, image dependency packaging, platform/submission changes, and additional
checkpoint/export/`run_eval` provenance.

## 8. Remaining user/platform decisions

- Select the repaired dataset release ID/version.
- No effect-size, equivalence, or non-inferiority margin has been approved.

Already decided:

- pooled 1,000-class MML holdout;
- existing MPTP fact-scope policy;
- unchanged six-family mix and full 13 loader epochs;
- fixed published tokenizer `tokenizer/qwen25-vendored/v1`;
- publish only the repaired packed-token corpus for this readiness task;
- initial evaluation reports versioned tri-state Metamath validity for
  `facts_present` and `facts_absent` when pinned source databases are supplied;
- v2 is warning-only rather than hard-denied, but is forbidden for final
  scientific training by operator policy;
- complete local build/tokenization/real-payload validation, then pause before
  S3 upload for manual review; no legal conclusion is encoded in the tooling.

## 9. Required order from here

1. Finish and independently accept the remaining data correctness findings.
2. Supply final source roots to the six-family generation coordinator.
3. Build every family into a fresh immutable generation.
4. Run deep verifier, mutation tests, exact accounting, and source replay.
5. Pack train/val shards with the unchanged Qwen2.5 tokenizer.
6. Run the exact pinned `pretrain-tokens/v1` gate against the actual six-family
   staged payload and reconcile local-256 versus deployed-128 diversity policy.
7. Pause for manual upload review, then publish through
   `edullm-landing` using
   `tokenizer/qwen25-vendored/v1`; confirm promotion and reader resolution.
8. Expose/select the repaired release in the submission form and build/scan the
   exact OLMo commit.
9. Recompute exact complete batches/epoch and set the run horizon to 13 loader
   epochs from the new promoted bytes.
10. Hand off the new `dataset_release` value for the two full training jobs.

## 10. Definition of done

The experiment is ready only when:

- every family is source-pinned and rebuilt;
- every train row is prompt-complete and touches no held semantic class;
- every eval row has a declared exposure path;
- all raw rows have exactly one train/eval/typed-drop disposition;
- all JSONL, sidecars, packed shards, and manifests are hash-bound;
- the published release resolves the fixed Qwen2.5 tokenizer and exact uint32
  train/val paths;
- dense/split configs and streams differ only in the approved arm field;
- exact new rows/tokens/batches/steps/exposures are recomputed for 13 epochs.

Until then the status remains **NO-GO**.
