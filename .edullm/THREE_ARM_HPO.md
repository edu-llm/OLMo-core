# HPO arm contracts

The historical preregistered study has exactly three arms:

1. `full_acronym_soup`: FT-PFN + ifBO + IPBT/BTT, 30% `gpt-5.6-sol`
   multi-action Centaur, the same-depth 16-layer width-reduced u-μP proxy, and embeddings plus
   blocks 0–7 frozen.
2. `no_centaur`: identical model, freezing, and controller stack, with Centaur disabled.
3. `no_proxy`: FT-PFN + ifBO + IPBT/BTT and the same 30% Centaur policy as Arm 1, but the
   conventional stock 12-layer `olmo2_190M`, fully trainable. It has no u-μP metadata,
   width-transfer behavior, or layer freezing.

The later curriculum arms are separate extensions, not members of the historical proxy-evidence
cohort. Both preserve `no_proxy`'s stock model, exact fidelity, controller stack, searchable
hyperparameters, and fixed quadratic-warmup MTLD curriculum:

- `curriculum_quadratic_mtld` keeps the 30% Centaur policy;
- `curriculum_quadratic_mtld_no_centaur` is identical except that Centaur is disabled.

Both use only these sealed inputs:

- parent `pretrain/opt-with-synthetic-10b/v1`, including its published train and validation
  partitions;
- order `curriculum/opt-with-synthetic-10b/v1`, group `mtld`.

Their coordinated controller specs are `.edullm/hpo-curriculum-quadratic-mtld.json` and
`.edullm/hpo-curriculum-quadratic-mtld-no-centaur.json`. Neither consumes or produces
`proxy-evidence.json`. Ordinary population members remain isolated one-GPU workers; once the
controller has a single resumed winner, its continuation uses all eight GPUs through the explicit
8-rank finalist path while preserving the winner's global batch.

## Paired first-rung evidence

Arms 1–2 remain reporting-only until a common 16-configuration cohort compares the complete
u-μP-plus-freezing bundle against Arm 3 at exactly 50,003,968 tokens per configuration. The
cohort runner recomputes rank correlation, top-k recall, uncertainty, and measured compute
savings from raw worker observations. It rejects missing IDs, extra IDs, token mismatches,
non-finite metrics, invalid accelerator time, changed contracts, and persisted decisions that
do not match local recomputation.

The runner is:

```bash
python .edullm/hpo_on_corpus.py "$EDULLM_RUN_ID" \
  --run-proxy-cohort \
  --proxy-spec .edullm/hpo-full-acronym-soup.json \
  --reference-spec .edullm/hpo-no-proxy.json
```

It writes the evidence path preregistered by the frozen specs. A failed admission result is
persisted for diagnosis and then stops the run.

## Official u-μP integration

`unit-scaling==0.3.5` is the latest published release. Its experimental
`transforms.unit_scale()` path cannot transform OLMo: the generated FX graph emits both a
positional and keyword `constraint` for the scalar epsilon addition in RMSNorm:

```text
add_9 = unit_scaling_functional_add(variance, 1e-05, None, constraint=None)
```

The release documentation labels that transform experimental and recommends explicit
substitution with public `unit_scaling.functional` operations as the standard path. Arms 1–2
therefore install explicit unit-scaled embedding, linear/readout, RMSNorm, SwiGLU, attention,
residual split/add, and cross-entropy operations. They also use the official transformer residual
rule, unit-normal initialization, parameter metadata, and AdamW learning-rate scaling. The
integration preserves OLMo module identity for meta-device construction and FSDP, and fails
closed on model features that have not been explicitly adapted. Arm 3 remains stock OLMo with
none of these substitutions.

## Platform submission

Each job is one arm (or the proxy cohort) on `gpu-8xa100` with team `pre-training` and
dataset `regmix-10b-v1` (`pretrain/regmix-10b` v1). Expected wall time is about
1–1.5 hours per arm; pass `--hours 4` as the hard runtime bound and `--attempts 2`
(the `olmo-core-train` maximum).

Runtime secrets:

| Secret | Proxy cohort | Arm 1 / 3 | Arm 2 (`no_centaur`) |
|--------|--------------|-----------|----------------------|
| `WANDB_API_KEY` | required | required | required |
| `OPENAI_API_KEY` | not used | required (30% Centaur) | not used |

The curriculum pair follows the same split: `curriculum_quadratic_mtld` requires
`OPENAI_API_KEY`, while `curriculum_quadratic_mtld_no_centaur` does not.

On the platform, inject these as job environment variables. Locally, unset keys
fall back to `~/.wandb_api_key` and `~/.openai_api_key`. On RunPod, copy the
same `/workspace/wandb-session.env` file used by the curriculum adapter.

Centaur defaults to the TrueFoundry AI Gateway
(`OPENAI_BASE_URL=https://gateway.truefoundry.ai/v1`, route model
`openai-group/gpt-5.6-sol`) while keeping the Brainlift logical model id
`gpt-5.6-sol`.

HPO artifacts mirror to W&B project `hpo-probe` (not the platform `wandb_project`).

Proxy cohort first:

```bash
edullm check --json \
  --experiment hpo-proxy-cohort \
  --dataset regmix-10b-v1 \
  --team pre-training \
  --spec .edullm/run-hpo-proxy-cohort.yaml \
  --compute gpu-8xa100 \
  --hours 4 \
  --attempts 2
```

Then each arm separately (distinct `EDULLM_RUN_ID`, set `EDULLM_HPO_SPEC` to the arm JSON):

```bash
export EDULLM_HPO_SPEC=.edullm/hpo-no-proxy.json

edullm check --json \
  --experiment hpo-three-arm-no-proxy \
  --dataset regmix-10b-v1 \
  --team pre-training \
  --compute gpu-8xa100 \
  --hours 4 \
  --attempts 2
```

## RunPod curriculum arms

The RunPod adapter stages the curriculum release pair into one local manifest. Preparation is
bounded to the initial download: `stage_inputs.py` accepts one short-lived credential file,
verifies object sizes, writes `/workspace/edullm-inputs/hpo-probe/ready.json`, and deletes the
credential file. The training entrypoint refuses any remaining AWS credential file or environment
variable.

After copying and bootstrapping `.edullm/runpod` as described in its README, stage inputs once:

```bash
cd /workspace/OLMo-core
PYTHONPATH="$PWD/src:$PWD/.edullm" \
  python3 .edullm/runpod/stage_inputs.py \
    --credentials-file /workspace/aws-session.env \
    --release-set curriculum
```

The resulting schema-v2 manifest contains the curriculum parent train/validation objects and the
exact MTLD order group, but not the unused RegMix corpus. No dataset is rebuilt or published.
Use a fresh 500 GB workspace; launch preflight requires at least 300 GiB free after staging.

The shared controller and worker path carries the morning-run repairs into both curriculum arms:

- JSON `transition: null` is treated as no transition override;
- inherited trials may load the historical donor checkpoint named by their allocation;
- recovery reconstructs pending batches from durable allocation events newer than a snapshot;
- W&B heartbeats continue after the backend's latest history step;
- `retry-startup` uses `resume=allow`, while a checkpointed run uses strict `resume=must`.

The u-μP loss and DTensor initialization failures are not applicable: both curriculum arms use
stock, exact `olmo2_190M` rather than the historical proxy model.

Run a launch preflight without creating a run identity or starting the controller:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main DRY_RUN=1 \
  bash .edullm/runpod/launch.sh
```

Launch a fresh, independent controller slot:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main \
  bash .edullm/runpod/launch.sh
```

For the matching no-Centaur ablation, use a distinct slot:

```bash
MODE=curriculum_quadratic_mtld_no_centaur RUN_SLOT=main \
  bash .edullm/runpod/launch.sh
```

Resume that exact checkpoint/controller lineage and W&B identity:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main RECOVERY_MODE=resume \
  bash .edullm/runpod/launch.sh
```

Fresh mode refuses existing state under
`/workspace/edullm-runs/hpo-probe/curriculum_quadratic_mtld/<slot>/`; choose a new slot for an
independent study. The persisted `run.env` binds resumes to the original `EDULLM_RUN_ID` and
`WANDB_RUN_ID`. Runs and publication remain in W&B project `hpo-probe`.

On controller success, `publish_outputs.py` reads the final-evaluation event rather than guessing
from trial rank, then uploads that selected checkpoint as an `hpo-winner-checkpoint` artifact.
Use the artifact metadata and `study-result.json` as the winner handoff to a separately reviewed
long-run recipe. The existing `launch_final_validation.sh` vectors remain the historical
RegMix/370M validations and are not a curriculum-arm handoff.
