**TL;DR:** This is the standalone implementation of Part 1 of the [spacing rerun PRD](../../plans/PRD_spacing_interleaving_reruns.md). Start with CPU preparation, then one bounded development acquisition screen on a FarmShare GPU. Confirmation stays blocked until acquisition, interference sensitivity, cost, replication and the source audits have been resolved and preregistered.

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

Teacher-forced scoring uses the correct next-token shift, masks the question, and reports answer-token loss excluding EOS. EOS-inclusive loss is separate. Each fact contributes a token mean; facts average within events, then events average within each training replicate. Fact-micro outcomes are also retained.

Behavioral scoring uses actual greedy generation with a 32-token cap and EOS/newline stopping. Raw predictions, token IDs and termination flags are saved. Frozen normalization applies Unicode casefolding, whitespace collapse and trailing punctuation removal. Only the canonical answer is accepted unless a reviewed alias file supplies alternatives. All old/new/control facts are evaluated in the initial development recipe.

MCQ scoring uses the source's fixed candidate texts with correct positions balanced over the ordered fact list. Every arm shares that arrangement. The frozen prompt lists the candidates and requests full answer text. Candidate scores sum answer-token log probabilities without EOS; mean-token scoring is a sensitivity outcome. Strict ties count as failures, and margins against the best distractor are reported. Verified question paraphrases have separate scores when supplied; this initial development run has no human-verified paraphrases and makes no transfer claim.

Expensive behavior/MCQ runs occur at baseline, acquisition milestones, shared Stage-1 end, buffer onset and declared buffer delays. Stage-2 interior checkpoints score loss. A pinned WikiText validation sample measures general language loss at Stage-1 end and final evaluation. Generation and MCQ costs are logged separately from loss scoring.

Analysis requires complete five-arm bundles. It reports paired replicate differences, 90% and 95% t intervals, strict loss-equivalence bounds, meaningful directional bounds, and Holm correction for H1/H3/HG. A single development bundle produces descriptive estimates only. Stage-2 trapezoidal AUC excludes the buffer. Raw control trajectories and the old-minus-control change sensitivity remain available. No behavioral-equivalence claim is implemented automatically. Event-resampling/mixed-model sensitivities and publication figures remain analysis work after data collection.

## Confirmation gate

Preparation with `mode: confirmation` requires an audit JSON containing `cross_event_alias_audit_complete: true`, `answer_alias_audit_complete: true`, and a verified `paraphrase` for every old canonical fact under `facts[FACT_ID]`. Optional `aliases` must be reviewed. Booleans record a human audit, not an automated proof.

Both `stage1 --exposures E` and `arm` require `--preregistration` for confirmation. The preregistration must contain `frozen_utc`, fixed `n`, all `replicate_manifest_sha256` values, E, `primary_delay`, a positive `loss_margin`, `hardware`, `measured_cost`, `power_scenarios`, `assay_usable`, `maximum_attempts`, `secondary_holm_family` and `missing_pair_rule`. It binds manifests and checks the frozen endpoint and attempts. The experimenter must justify the assay, margins, precision and measured budget; filling fields is not scientific review.

No confirmation config or preregistration is checked in as complete. The initial screen may fail to acquire facts or may show inadequate interference. Either result calls for development diagnosis. It does not authorize a large experiment or support a spacing conclusion.

## Tests and remaining checks

The CPU tests cover transitive duplicate clusters, cross-event leakage, outer split isolation, exact schedules after JSON roundtrips, endpoint/dose corruption, generic budgets, context overflow, event aggregation, MCQ ties, answer/EOS alignment, full optimizer/RNG resume with dropout, gradient accumulation, and restart-log truncation. A mid-epoch interruption test verifies all subsequent shuffled epochs, exact fact doses and bitwise-equal final model/optimizer state. Diagnostic tests cover QA-role aggregation, statement masks and audited answer-span exclusions. The runtime and diagnostic have also been exercised with a tiny instance of the actual `hf_olmo` model class. A numerical A/A checkpoint check on the actual GPU/backend is still required before confirmation. CUDA acquisition, full-run memory, elapsed time and scientific assay quality are measured by the first jobs, not inferred from CPU tests.
