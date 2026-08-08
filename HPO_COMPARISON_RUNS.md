# HPO comparison smoke

This is a functional comparison between the fixed OLMo2-190M recipe and the
FT-PFN/ifBO + BTT + IPBT controller. It is deliberately small enough to catch
image, data, checkpoint, evaluator, and controller failures before a scientific
HPO campaign. It is **not** a conclusive HPO benchmark.

Everything must be committed and pushed on an `edullm/**` branch before these
commands describe the image that will run. `edullm check` never dispatches a GPU
job; only `edullm submit` does.

## Frozen comparison contract

- Dataset release: `reservoir-dolma2-v1`.
- Published dataset: `pretrain/reservoir-dolma2/v1`.
- Profile: `pretrain-tokens/v1`.
- Tokenizer: `tokenizer/dolma2-bpe`.
- Model: OLMo2-190M.
- Sequence length: 2,048.
- Global batch: 32,768 tokens.
- Data seed: 210007.
- Initialization seed: 110007.
- Held-out metric: two deterministic batches from the sealed validation split.
- Aggregate training budget: 1,146,880 tokens, or 35 optimizer steps.

The default arm spends all 35 steps on one fixed recipe. The HPO arm first
screens four one-GPU lineages for five steps each, then emits an IPBT generation
or BTT-triggered restart and uses FT-PFN/ifBO to grant only three more five-step
allocations. In the nominal healthy path this chooses three of four available
candidates; a BTT restart may change that candidate composition. Each promoted
inherited lineage can reach at most ten steps. The HPO arm therefore tests
anytime search efficiency, not equal final-model training duration.

The HPO smoke uses real FT-PFN/ifBO, BTT, and the IPBT population shell. Sol is
excluded because this repository does not yet contain a provider-backed Sol
transport. Calling the deterministic smoke recorder “Sol” would make the test
look more complete while testing no model at all.

The search validation split is reused for the smoke summary. Each evaluation is
only 8,192 held-out tokens (four 2,048-token sequences), and HPO selects the
minimum of several measurements on that same tiny split. This positively biases
the HPO result. It is not an untouched final evaluation. A publishable
comparison needs a third, preregistered split and at least three paired seeds.

Compilation is disabled in both arms so a short functional smoke measures the
controller and training path rather than four independent Inductor compilations.

## Dataset reader contract

Both arms resolve the platform-provided dataset id, exact version, and tokenizer
through `edullm_data.read.dataset_paths()`. No object path or hand-written
manifest appears in the code.

The target image pins `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c`. Its live profile registry at that
commit contains:

- `eval-results/v1`
- `pretrain-tokens/v1`
- `sft-conversations/v1`
- `token-order/v1`
- `tokenizer/v1`

This setup follows both `edullm-dataset-design` and `edullm-datasets`: the
existing validated corpus is read through its registered profile, with train and
held-out paths kept separate.

## Non-dispatching checks

Run these independently and read JSON from stdout without combining stderr.
Exit 0 stands, exit 1 is a refusal on the merits, exit 2 is a malformed command,
and only exit 3 is retryable. Match refusal codes, never prose.

```bash
edullm check --json \
  --experiment hpo-default-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-hpo-comparison-baseline.yaml \
  --compute gpu-1xa10g \
  --hours 1 \
  --attempts 1
```

```bash
edullm check --json \
  --experiment hpo-hybrid-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-hpo-comparison-hybrid.yaml \
  --compute gpu-4xa10g \
  --hours 1 \
  --attempts 1
```

The check output is the sole authority for cost, runtime bounds, approval class,
image state, and current capacity. Do not copy those values into this document.

## Dispatch only after both checks stand

```bash
edullm submit \
  --experiment hpo-default-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-hpo-comparison-baseline.yaml \
  --compute gpu-1xa10g \
  --hours 1 \
  --attempts 1
```

```bash
edullm submit \
  --experiment hpo-hybrid-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-hpo-comparison-hybrid.yaml \
  --compute gpu-4xa10g \
  --hours 1 \
  --attempts 1
```

## What constitutes a pass

The default arm must complete 35 steps, write a checkpoint, and print a finite
`heldout_ce`. The HPO arm must allocate four trials, emit an IPBT transition at
the first boundary, grant exactly three second-round allocations, make at least
one FT-PFN-conditioned decision, write the controller event log, produce a
full-fidelity winner, and print a finite search-validation result. Both a
generation transition and a BTT-triggered restart are valid smoke outcomes. The
HPO entrypoint fails nonzero when `require_final_winner` is not satisfied.

Compare:

1. best held-out CE,
2. aggregate training tokens,
3. whether either arm failed or produced no finite metric.

Do not claim convergence superiority from this functional smoke. Use it to
decide whether a larger, paired-seed comparison is safe to run.

Before expecting either check to stand, commit every run dependency, including
`src/olmo_core/hpo/comparison.py`, both YAML specs, the hybrid JSON, the baseline
runner, the comparison tests, this runbook, and the shared HPO entrypoint. The
platform can identify directly named untracked files but cannot discover every
untracked Python module imported transitively.
