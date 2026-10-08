**TL;DR:** This is the standalone implementation of Part 1 of the [spacing rerun PRD](../../plans/PRD_spacing_interleaving_reruns.md). Start with CPU preparation, then one bounded development acquisition screen on a FarmShare GPU. Confirmation stays blocked until acquisition, interference sensitivity, cost, replication and the source audits have been resolved and preregistered.

The active [October 8 assay amendment](../../plans/spacing_final_assay_amendment_20261008.md) freezes metadata-derived L=252 for the original confirmation role geometry, adds development acquisition scale/stability checks, and requires six paired final-span sensitivity bundles before precision planning. The completed L=84 five-arm run is a disclosed pilot. Independent agent audit artifacts will replace the preliminary proposed human-only audit method only through separately guarded protocol support; human review remains false and confirmation remains blocked.

## What the implementation runs

The pretrained checkpoint is `allenai/DataDecide-dolma1_7-300M` at `4b1b42ff7c5224a077c4f5624824dd7c8bbe98d7`. The loader uses the checkpoint's original `hf_olmo` architecture with full fine-tuning, fp32 parameters and AdamW states, and bf16 CUDA autocast. CPU execution exists for tests. Real jobs require CUDA and a recorded full code commit.

FictionalQA is pinned to `131cb74fdc3e601b5e896ed768ad9852ea35a8f9`. The preparation step checks the byte hashes of the QA, MCQ and blind-answer parquet files, including local cache overrides. It resolves duplicate-root chains to their final roots. The source has 7,500 rows and 1,797 final-root clusters. Intermediate `duplicate_root` values are not distinct final clusters. Different answer strings within a similarity cluster are not automatically accepted aliases.

The fixed outer split reserves 20 events for development and 80 for confirmation. Development uses 5 old, 7 new, 5 never-trained control and 3 QA-format events. Confirmation uses 20/30/20/10. Event assignment uses only style metadata and a separate split seed. The preparation step rejects cross-event duplicate/supporting-text links that would invalidate those counts. Cross-event entity aliases still require human audit before confirmation.

Every training sequence has exactly 36 loss-bearing tokens in the initial development configuration, right-padded to 96 input tokens. Complete statements and QA answers precede EOS. A deterministic segment from pinned WikiText fills the remaining loss budget. The statement is never truncated. QA prompt tokens and padding bear no loss. All filler tokens bear ordinary weight-one loss, and the manifest and realized logs report their counts. Generic filler can affect interference, so its substantial share of this initial recipe must be included in assay interpretation and calibration. The minimum viable budget depends on the longest complete statement, rather than on a target replay percentage.

`UNI`, `EXP` and `MASS` have identical complete packed-example multisets, optimizer updates, batch shapes and loss-token budgets. Their common new and QA streams retain their order. `GEN` replaces exactly the `UNI` old-example positions with generic examples; new and QA positions are unchanged. `NONE` replaces those positions with additional new examples. All arms have the same no-review buffer. The CPU compiler verifies exact per-fact doses and endpoint timing. It fails if capacity or the declared maximum stagger/span ratio is infeasible.

The initial development span is 84 updates, with four reviews at offsets `[0,28,56,84]`, `[0,12,36,84]`, or `[81,82,83,84]`. This is a calibration choice, not the frozen confirmation span. Increasing the old-fact count will generally require a longer span or larger valid cohort capacity. No runtime scheduler shifts, drops or inserts reviews.

Stage 1 exposes every old fact exactly E times. QA-format teaching uses separate events. Shared checkpoints contain model weights, all Adam moments and step counters, Python/NumPy/CPU/CUDA RNG states, exposure cursor, epoch position, learning-rate clock and per-fact last-exposure/token clocks. Arms resume the same shared state after warmup. Every resume and completed run compares realized training IDs, hashes and loss tokens with the compiled sequence.

## Initial run and limits

The checked-in configuration runs one LR candidate, `1e-4`, and tests E at `1,2,4,8,16,32,64`. It stops at the first milestone reaching 40% old-event-macro generated exact match after warmup. A score above 70% is a ceiling failure; failing to reach 40% by E=64 is an acquisition failure. Both produce a recorded decision and block the default arm command. No confirmation facts enter this screen.

The `calibrate` command performs acquisition only. If its decision is usable, submit development arm diagnostics from its shared checkpoint. Complete at least one five-arm development bundle and enough independent UNI/EXP pairs to estimate paired variance, within the PRD's initial limit of three LR candidates and four development bundles. That cross-job limit is an operator responsibility; this package does not start a grid automatically.

The first FarmShare screen at LR `1e-4` failed the acquisition gate. The completed-checkpoint diagnostic found 96.08% exact match on QA-teaching facts and 4.24% on old canonical questions, with old-statement loss of 0.388 nats per token. `configs/calibration-lr3e5.json` is the second bounded acquisition candidate. It changes only LR to `3e-5`, preserving every event role, seed, training statement, evaluation question, exposure milestone and acceptance threshold. The hypothesis is that a lower full-fine-tuning LR may preserve pretrained question-answer transfer while learning the facts. This is a development test, not an established fix. It starts from the same pretrained checkpoint in a fresh run directory and stops by E=64. Neither a weaker acquisition threshold nor a spacing result selects this candidate.

`configs/calibration-directional.json` is a separate development candidate at the original LR `1e-4`. It tests whether the direction of the supporting statement limits transfer. The [audited declaration bundle](data/development-directional-statements.json) contains all 348 selected records. Natural declarative restatements put the relation before the canonical answer where the source supports doing so, for example, `Jason Moorland was a coach for the Crestwood Eagles` becomes `A coach for the Crestwood Eagles was Jason Moorland.` Authoring used the source statement and its answer label, without evaluation question text. Both agents reviewed the applied old/new changes against their sources; this is not human verification. Every original statement remains in the prepared manifest, and no evaluation question or accepted answer changes.

Only old/new training statements change. QA-teaching examples and never-trained controls retain their existing forms, all event roles and seeds stay fixed, and every old fact still receives E complete record exposures. This candidate retains the 36-token loss budget, 96-token context, E=64 cap and 40–70% behavioral gate. It neither trains an evaluated old question nor substitutes a cloze question. One new-fact source does not state the extra specificity in its answer label, so its statement stays unchanged with an explicit audit exception. No fact is excluded. The loader verifies the complete record set, source hashes, roles, declaration form, answer position and documented exceptions before compilation. It refuses to use this development bundle for confirmation.

The first three development configurations count final-root record exposures. The source audit found exact supporting-statement duplicates across those IDs: 95 old probes correspond to 86 distinct normalized source statements, and 127 new probes to 102 statements. Those legacy runs remain readable development diagnostics. They cannot enter confirmation or be pooled with the corrected unit recipe.

`configs/calibration-units-directional.json` applies the audited directional declarations with `rehearsal_unit_policy: source_statement_v1`. A unit ID hashes the event and normalized original source statement, before any rewrite. Every canonical question and answer remains an evaluation probe. The compiler trains one complete statement per unit at each exposure, and Stage 1 gives each old unit exactly E exposures. UNI/EXP/MASS each review every old unit exactly four times. QA teaching retains its separate question-answer stream.

Within each unit, the lexicographically first probe supplies the training declaration. If probes have different canonical answer labels, the unit retains its full original statement so an answer-directed rewrite cannot discard another target. If two different source units acquire identical rewritten text, both revert to their source statements. The manifest retains the original text, each candidate rewrite, the chosen representation, every probe-to-unit link and the reason for each fallback. No question or answer is filtered. This rule removes exact normalized statement duplication; semantic overlap between differently worded source statements still needs audit.

The corrected recipe requires a fresh acquisition screen because unit counts, filler keys, QA ordering and schedule length change. Its success cannot be inferred from a legacy screen. It retains the original LR, 36-token budget, E=64 cap and 40–70% gate. The unit-only configuration has been prepared and tested locally but has not been selected for a GPU rerun.

The unit configuration enables `qa_teaching_diagnostics`. Behavior checkpoints also record canonical QA-teaching loss and generated exact match in a separate diagnostic section, with its own timing. These are in-sample teaching questions. Their scores never enter the old-fact acquisition gate or the primary spacing contrasts.

`configs/calibration-transfer-v2.json` is the next bounded transfer candidate. It combines source-statement rehearsal units and the audited directional declarations with all 225 source question-answer rows from the same three QA-teaching events. Every teaching row retains its own question and natural answer, including when a similarity cluster has different answer labels. The canonical 64 QA probes remain unchanged for debugging. Old/new/control evaluation questions, aliases and MCQ inputs also remain unchanged. Preparation verifies complete membership, source-field hashes, event roles and absence of exact held-out evaluation questions from the teaching stream. It refuses the expanded teaching policy in confirmation until that recipe is audited and frozen.

The candidate increases QA teaching from two to four positions in each 16-example update. Stage 1 has eight updates per complete exposure of the 86 old units, capped at 512 updates for E=64. The 36-token loss budget and ordinary weight-one filler remain unchanged. All arms share the same QA stream, rate and positions; the review cap of eight still fits beside four QA positions. This is a combined assay revision addressing source duplication and question-form coverage. Its results cannot isolate the causal contribution of each change. It starts fresh from the pinned pretrained model and uses the same 40–70% acquisition gate. No spacing-arm outcome selects it.

Expanded Stage-1 teaching follows one shuffled, continuously cycling list of source questions. Its cursor advances by the number of QA positions, including across partial old-unit batches and checkpoint resumes. At E=64 the 2,048 teaching positions cover every one of the 225 questions nine or ten times. Legacy configurations retain their original stream indexing for exact restart compatibility.

Transfer-v2 completed its bounded screen and failed the 40% acquisition gate. At E=64, old generated exact match was 16.15% under unit/event aggregation and 12.63% under the common probe-micro sensitivity; canonical QA-teaching exact match was 92.90%. This improves on some earlier diagnostic outcomes but does not establish a usable spacing assay.

`configs/calibration-grounded.json` addresses missing source context. The [reviewed grounding bundle](data/development-grounded-declarations.json) contains every one of the 188 original old/new statement units. Its declarations use the original `fict`, canonical answer labels and the pinned event factsheet. Evaluation questions and model outcomes are excluded from the authoring packet. For example, an unnamed society becomes the Vivacious Writers' Society when the factsheet supplies that identity. Every added context claim has exact factsheet excerpts and source hashes. Preparation checks the actual factsheet parquet against its pinned byte hash. Linked fictions are not an authoring authority; a known timeline discrepancy is retained as an audit note.

The source assertion remains authoritative for its original claim. An explicit conflict with the factsheet preserves that assertion and its unit, with a flag; it does not trigger filtering or a gold-answer change. Five such cases remain, alongside the separate Town Hall venue-scope ambiguity. Three answer labels are nonliteral descriptions of their grounded declarations, and those exceptions remain visible. Agent review covered all declarations, factsheets and grouping decisions. This is not human verification or an assertion that every source claim is correct.

The bundle groups 26 sets of equivalent grounded assertions into a total of 157 training units: 73 old and 84 new. Each stable group ID hashes its original source-unit IDs. All 188 original units map exactly once, and every one of the 348 evaluation probes remains. Twenty sets of related but non-equivalent assertions are listed without merging; sharing an answer is not sufficient for a merge. The loader rejects duplicate group membership, identical ungrouped declarations, altered evidence and changes to preserved conflicts. It stores both the original and grounded registries. Semantic overlap outside the reviewed groups remains an audit limitation.

The longest grounded declaration has 28 tokens, so this candidate retains 36 loss tokens per example and a 96-token input context. It also retains the 225-question teaching pool, four QA positions per update, LR `1e-4`, E≤64 and the 40–70% old behavioral gate. Stage 1 now takes seven updates per complete old-unit exposure, at most 448 updates. It is a fresh development assay revision, combining source construction and grouping changes. It cannot identify each change's separate effect or authorize confirmation.

Each process stops after 20 hours by default and writes a restart checkpoint. The Slurm launcher requests 22 hours, one GPU, four CPU cores and 32 GB RAM, and excludes `wheat-01`. Its early signal reaches Python. Exit code 75 means a checkpointed time limit, not experiment completion. Resubmit the same command and output path to resume; each output directory allows at most three process attempts. Ordinary exceptions, divergence and nonfinite loss do not cause automatic retries. Stage 1 writes checkpoints every 50 updates, at acquisition milestones, at completion and on a time/signal stop. It does not write one at every non-milestone epoch. Resuming a checkpoint in the middle of an epoch restores the exact shuffled exposure sequence and optimizer/RNG state.

Use scratch storage for the model cache, virtual environment and outputs. Each full optimizer checkpoint is several GB. The runner keeps the shared Stage-1 state, a current restart state and per-fact evaluation JSON; it does not retain optimizer snapshots at every evaluation. GPU allocation elapsed from `sacct`, CPU preparation, failures, downloads and storage must be added to process timing before forecasting the full study.

## Commands

Run from the repository root with Python 3.11 or 3.12. Install the pinned requirements into a dedicated environment. All network downloads in preparation use immutable public revisions; no cloud credentials or W&B connection are needed.

```bash
python -m pip install -r experiments/spacing_rerun/requirements.txt
export PYTHONPATH="$PWD/experiments/spacing_rerun"
python -m pytest experiments/spacing_rerun/tests -q
python -m spacing_rerun prepare \
  --config experiments/spacing_rerun/configs/calibration.json \
  --output /scratch/users/USER/spacing/dev-001/prepared
python -m spacing_rerun validate \
  --prepared /scratch/users/USER/spacing/dev-001/prepared
```

Push the code before GPU submission. A repository archive is supported; set `SPACING_CODE_COMMIT` to the full pushed commit. The operator must verify that the archive and recorded commit agree. The Slurm file deliberately does not authenticate to GitHub or create a connection.

```bash
export SPACING_ROOT=/scratch/users/USER/spacing/code
export SPACING_VENV=/scratch/users/USER/spacing/venv
export SPACING_CODE_COMMIT=FULL_PUSHED_COMMIT
sbatch experiments/spacing_rerun/farmshare.sbatch calibrate \
  --prepared /scratch/users/USER/spacing/dev-001/prepared \
  --output /scratch/users/USER/spacing/dev-001/stage1
```

After a usable acquisition decision, the same launcher can run each development arm. Independent jobs reuse the complete Stage-1 checkpoint. Submit concurrency according to the live account QoS and other user jobs.

```bash
sbatch experiments/spacing_rerun/farmshare.sbatch arm \
  --prepared /scratch/users/USER/spacing/dev-001/prepared \
  --output /scratch/users/USER/spacing/dev-001/UNI \
  --stage1-checkpoint /scratch/users/USER/spacing/dev-001/stage1/stage1.pt \
  --arm UNI
```

Repeat with `NONE`, `EXP`, `MASS` and `GEN`. New independent development bundles get fresh split/order/train/eval seeds but preserve `outer_seed`. Changing a recipe requires a new prepared directory. Prepared JSON files are immutable and hash-checked on every start. Do not edit them to resume a different experiment.

```bash
python -m spacing_rerun analyze \
  --bundle /scratch/users/USER/spacing/dev-001 \
  --primary-delay 84 --output /scratch/users/USER/spacing/dev-001/summary.json
python -m spacing_rerun power --sd 0.02 --sd 0.04 --sd 0.08 \
  --output /scratch/users/USER/spacing/power-sensitivity.json
```

Those SD values illustrate sensitivity inputs. Replace them with measured paired variability and conservative larger scenarios. The power command simulates finite-sample paired t intervals under normal paired differences; it does not prove that this assumption holds or automatically choose n.

## Diagnosing failed acquisition

The opt-in `configs/calibration-sourceqa.json` changes shared acquisition to
`declaration_plus_source_qa_v1`. It pairs each old declaration exposure with one
source-derived QA exposure per distinct answer target, rotating between two
independently reviewed question forms across epochs. The authoring input contains
old source assertions, their answer labels and pinned event factsheets. It excludes
evaluation questions and model outcomes. Preparation binds both source input and
question bundle hashes, requires independent review, retains every evaluation
probe, and rejects any exact normalized canonical or paraphrase question shared
with old/new/control evaluation.

This recipe currently covers 73 old declaration groups and 78 answer targets.
Each epoch has 151 mixed content examples in 13 updates, with 12 content slots
and four separate-event teaching slots per update. Every declaration and answer
target receives exactly E exposures; the selected acquisition question variant
is `epoch % 2`. Teaching uses its continuous stream cursor, including the last
partial batch. This increases the separate-event teaching dose relative to the
grounded declaration-only recipe and changes Stage 1 total tokens and updates.
The 36-token loss budget, answer-only QA mask, 40–70% acquisition gate and frozen
evaluation remain unchanged. Milestones are E1/2/3/4/6/8/12/16/32/64.

Stage 2 remains declaration rehearsal. No acquisition QA examples enter any
review arm or the shared buffer, and all five continuation streams retain their
previous example multisets and ordering. Exposure logs and the acquisition
decision record declaration doses and source QA target doses separately. Source
QA tokens have their own cumulative clock and do not enter the new/teaching-token
interference clock. Shared checkpoints preserve these counts across arm forks.

The result measures retention after shared declaration-plus-QA acquisition,
followed by declaration review. It does not establish statement-only acquisition
or retrieval-practice effects. The recipe remains development-only; it does not
relax confirmation audit or preregistration requirements. A failed or excessive
acquisition gate still prevents continuation.

After the acquisition process terminates, a separate diagnostic can read its final `stage1.pt` without changing that checkpoint or its outputs. Run this command inside a short GPU allocation. The standard `farmshare.sbatch` invokes the main module, so use an explicit diagnostic batch command with the same environment, one GPU and `--exclude=wheat-01`.

```bash
python -m spacing_rerun.diagnose \
  --prepared /scratch/users/USER/spacing/dev-001/prepared \
  --checkpoint /scratch/users/USER/spacing/dev-001/stage1/stage1.pt \
  --output /scratch/users/USER/spacing/dev-001-diagnostic.json \
  --max-runtime-seconds 900
```

The diagnostic records a SHA-256 of the source checkpoint, its byte count, manifest, epoch/global step, code commit, load/hash/evaluation times and peak GPU memory. It scores all QA-teaching questions with loss and actual generation, all old questions with the same measures, and the old training statements with filler and EOS excluded from the primary content loss. An answer-span diagnostic scores the canonical answer only where it appears uniquely in its training statement and aligns with tokenizer offsets; every exclusion and the coverage denominator are retained. Statement reconstruction and answer-span likelihood are contextual memorization diagnostics, not proof of factual QA ability.

Strong statement fitting with weak old-question performance can indicate failure to transfer trained content into QA. Weak performance even on QA-teaching questions motivates examining the teaching recipe. The comparisons use different contexts and sometimes different facts, so they do not identify a causal mechanism. Choose subsequent development settings from acquisition stability, transfer and general-language degradation. Do not select settings because EXP outperforms UNI.

## Evaluation and inference

Teacher-forced scoring uses the correct next-token shift, masks the question, and reports answer-token loss excluding EOS. EOS-inclusive loss is separate. With source-statement units, each question contributes its answer-token mean, questions average within units, units average within events, and events average within each training replicate. Exact match and MCQ use the same hierarchy. Probe counts remain under `facts`; distinct units have a separate count. The outputs retain fact-micro and unit-micro sensitivities.

Evaluations record `spacing-metrics-v2-unit-macro` for the corrected recipe. Legacy development uses `spacing-metrics-v1-fact-macro`, which averages canonical root probes directly within events. Analysis refuses to pool different unit policies or metric schemas. Exposure logs and timing clocks use the trained unit IDs, and each evaluation carries the mapping from every old probe to its exposure unit.

The grounded recipe uses `grounded_assertion_v1` and `spacing-metrics-v3-grounded-unit-macro`. The same probe/unit/event hierarchy now averages within reviewed assertion groups. It cannot be pooled with source-unit or legacy results. Common probe-micro results support descriptive comparisons across recipe revisions while keeping the changed training and aggregation definitions explicit.

Behavioral scoring uses actual greedy generation with a 32-token cap and EOS/newline stopping. Raw predictions, token IDs and termination flags are saved. Frozen normalization applies Unicode casefolding, whitespace collapse and trailing punctuation removal. Only the canonical answer is accepted unless a reviewed alias file supplies alternatives. All old/new/control facts are evaluated in the initial development recipe.

MCQ scoring uses the source's fixed candidate texts with correct positions balanced over the ordered fact list. Every arm shares that arrangement. The frozen prompt lists the candidates and requests full answer text. Candidate scores sum answer-token log probabilities without EOS; mean-token scoring is a sensitivity outcome. Strict ties count as failures, and margins against the best distractor are reported. Verified question paraphrases have separate scores when supplied; this initial development run has no human-verified paraphrases and makes no transfer claim.

Expensive behavior/MCQ runs occur at baseline, acquisition milestones, shared Stage-1 end, buffer onset and declared buffer delays. Stage-2 interior checkpoints score loss. A pinned WikiText validation sample measures general language loss at Stage-1 end and final evaluation. Generation and MCQ costs are logged separately from loss scoring.

Analysis requires complete five-arm bundles. It reports paired replicate differences, 90% and 95% t intervals, strict loss-equivalence bounds, meaningful directional bounds, and Holm correction for H1/H3/HG. A single development bundle produces descriptive estimates only. Stage-2 trapezoidal AUC excludes the buffer. Raw control trajectories and the old-minus-control change sensitivity remain available. No behavioral-equivalence claim is implemented automatically. Event-resampling/mixed-model sensitivities and publication figures remain analysis work after data collection.

## Historical confirmation gate

The following describes the initial source-unit protocol. The active [final assay amendment](../../plans/spacing_final_assay_amendment_20261008.md) requires grounded source-QA acquisition, actual independent audit coverage, human review recorded as false, and guarded confirmation implementation before launch. This historical protocol cannot authorize the current main run.

Preparation with `mode: confirmation` requires an audit JSON containing `cross_event_alias_audit_complete: true`, `answer_alias_audit_complete: true`, and a verified `paraphrase` for every old canonical fact under `facts[FACT_ID]`. Optional `aliases` must be reviewed. Booleans record a human audit, not an automated proof.

Both `stage1 --exposures E` and `arm` require `--preregistration` for confirmation. The config must use `rehearsal_unit_policy: source_statement_v1`; legacy record-based confirmation is rejected. The preregistration must contain `frozen_utc`, fixed `n`, all `replicate_manifest_sha256` values, E, `primary_delay`, a positive `loss_margin`, `hardware`, `measured_cost`, `power_scenarios`, `assay_usable`, `maximum_attempts`, `secondary_holm_family` and `missing_pair_rule`. It must also bind `rehearsal_unit_policy: source_statement_v1` and `metric_schema: spacing-metrics-v2-unit-macro`. It binds manifests and checks the frozen endpoint and attempts. The experimenter must justify the assay, margins, precision and measured budget; filling fields is not scientific review.

No confirmation config or preregistration is checked in as complete. The initial screen may fail to acquire facts or may show inadequate interference. Either result calls for development diagnosis. It does not authorize a large experiment or support a spacing conclusion.

## Tests and remaining checks

Unit tests additionally cover multiple-answer source preservation, rewritten-text collisions, missing-probe corruption, exact unit doses, the probe/unit/event aggregation hierarchy and rejection of legacy confirmation. A tiny actual `hf_olmo` integration exercises unit-based Stage 1, optimizer-state fork, a full UNI continuation, realized doses and exposure clocks.

Grounding tests check complete source/probe retention, exact per-group E and four-review doses, pinned factsheet mismatches, missing review, altered evidence, duplicated membership, changed source conflicts and runtime probe-to-group corruption. Natural relative clauses are allowed; complete held-out question forms and question/answer training markers are rejected. Both expanded source teaching and grounded assertion policies remain development-only until their remaining scientific audits and confirmation protocol are implemented and frozen.

The CPU tests cover transitive duplicate clusters, cross-event leakage, outer split isolation, exact schedules after JSON roundtrips, endpoint/dose corruption, generic budgets, context overflow, event aggregation, MCQ ties, answer/EOS alignment, full optimizer/RNG resume with dropout, gradient accumulation, and restart-log truncation. A mid-epoch interruption test verifies all subsequent shuffled epochs, exact fact doses and bitwise-equal final model/optimizer state. Diagnostic tests cover QA-role aggregation, statement masks and audited answer-span exclusions. The runtime and diagnostic have also been exercised with a tiny instance of the actual `hf_olmo` model class. A numerical A/A checkpoint check on the actual GPU/backend is still required before confirmation. CUDA acquisition, full-run memory, elapsed time and scientific assay quality are measured by the first jobs, not inferred from CPU tests.

## Frozen acquisition grid

The prospective policy core in `configs/acquisition-grid-policy-core-20261008.json` binds six normal and three acquisition-only scale trials, seeds, role hashes and source provenance. Normal trajectories launch with `calibrate --acquisition-policy-core CORE`. Each completes E2/3/4/6/8 and retains immutable full-state dose checkpoints; none stops at the first passing milestone.

After both scale source chunks receive actual independent review and all nine manifests are prepared, `bind-acquisition-grid-manifests --policy-core CORE --prepared PATH` repeated for all nine, with `--output binding.json`, records the actual binding time. Scale calibration also requires `--acquisition-manifest-binding binding.json` before model initialization. Scale probes cannot run arms or enter spacing-effect analysis.

After all nine trajectories terminate, `choose-acquisition-grid-checkpoint --prepared NORMAL --grid-output PATH` repeated nine times, with `--selection-rule binding.json --output selection.json`, selects a common E only if all nine score within40–70%. The scaled mean closest to55% chooses among valid doses, ties to the lower E. An empty common gate records failure. Normal arms require `--acquisition-selection selection.json` and its exact immutable dose checkpoint. The selector revalidates policy, manifest, decision and checkpoint hashes.

The grid changes trajectory handling and provenance without changing numerical Session methods from the actual GPU A/A commit `fb27597f`. The complete implementation suite has77 passing tests plus27 mutation subtests; independent root verification of29 acquisition/grid/policy tests and27 subtests also passed.

## Final-span development evidence

The prospective planning policy in `configs/final-span-planning-policy-20261008.json` was frozen at actual UTC2026-10-08T07:13:53+00:00, before any final-span arm outcomes. `python -m spacing_rerun.final_span --bundle PATH` repeated for the six normal bundles, with `--allocation-accounting FILE --output report.json`, validates the nine-grid common-dose proof, full arm checkpoints, original evaluations, ordered streams and realized exposure clocks. Source QA cannot resume after the fork. The report recomputes all frozen NONE/UNI/GEN sensitivity gates. EXP is excluded from eligibility.

Precision planning uses the one-sided80% upper bound on EXP-minus-UNI paired SD from all six bundles. It simulates100,000 finite-t experiments with fixedseed2026100809 for each n2–32, true differences0/±.01/±.04 and unchanged margin.02. All five planning scenarios must have one-sided95% exact binomial Monte Carlo lower bounds at least80%. Otherwise the policy chooses n32 with an estimation claim and expected95% CI half-width. The normality assumption and uncertainty from only six pairs remain explicit. Allocation costs require actual accounting, including preparation, calibration, retries and staging; process time is a lower bound. No final-span evidence report exists yet.
