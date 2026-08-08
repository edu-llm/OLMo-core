# Using the platform

Everything true whichever repository you work in. For what to actually run, see [training a model](olmo-core.md), [running an evaluation](olmo-eval-full.md) or [validating a corpus](edullm-data.md).

## Your first run

Open [Submit a run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml), press **Run workflow**, and fill in six fields. Everything else has a correct default.

| Field | Value |
| --- | --- |
| `repository` | `OLMo-core` |
| `commit_sha` | `main` |
| `workload_profile` | `olmo-core-check` |
| `compute_profile` | `cpu-32vcpu` |
| `team` | `scratch` |
| `experiment` | Anything, such as `onboarding` |
| `wandb_project` | Your Weights and Biases project |

**Type `main` and do not go looking for a branch.** The field takes a branch, a tag or a full commit, but your code has to have been built into a container image before a run can name it, and on your first run you have not built anything. `main` always has one. When you come to run your own work, *Building your image* in your repository's guide is the step that gets you there.

Leave `command` alone. Its default runs a one-line check needing no code of your own, which is the point of a first run: you are testing the path from the form to a machine and back.

Use `scratch` the first time. It is the bin for work nobody intends to keep, so a first run costs no research group anything.

## Access

Two lists, and you need both. They are separate systems, which is the most confusing thing about getting started here.

| List | What it gates | If you are missing |
| --- | --- | --- |
| `team-members` GitHub team | Write access, and therefore whether the **Run workflow** button exists at all | The page has no button. Nothing you type will help |
| `members` in `config/organization.yaml` | Admission | Your submission is refused when it compiles, before any lead is asked, naming the file |
| A W&B account on the `eduLLM` team, recorded on the roster | Attribution only | The run works and logs as the platform rather than as you. Nothing warns you |

Ask for all three through the [access request](https://github.com/edu-llm/platform/issues/new?template=access-request.yml) template.

## The submission form

| Field | What it decides |
| --- | --- |
| `repository`, `commit_sha` | What code runs. The commit must already have an image — see *Building your image* in your repository's guide |
| `workload_profile` | The time limit, the retry limit and the checkpoint contract, together. Not the machine |
| `compute_profile` | The machine. Required, and the field that decides what your run costs |
| `team` | Who reviews it, and the S3 prefix your output lands under. Closed dropdown: `platform`, `memory-split`, `input-core`, `pre-training`, `post-training`, `data-prep`, `eval-inference`, `scratch` |
| `experiment` | Groups related runs — a sweep, an ablation, a week. Free text, registers nothing, and what the cost view groups by |
| `dataset_release` | The corpus your job reads. `none` if it reads none |
| `command` | What the container runs. No shell: the first word must be a program name, and the line must not be wrapped in outer quotes |
| `wandb_project` | Where the run reports |
| `image_digest`, `maximum_runtime_hours`, `maximum_attempts`, the fan-out fields | Advanced. Leave them. The image comes from your commit, and the two bounds come from the workload profile |

## Choosing a machine

`compute_profile` is a closed dropdown of every shape with a queue behind it, and it is the most expensive field on the form by two orders of magnitude — $0.53 an hour at one end and $55.04 at the other. Nothing infers it from what you are running, and nothing refuses a small job on a large machine.

| You are | Pick |
| --- | --- |
| Doing anything for the first time | `cpu-32vcpu` |
| Checking your code sees a GPU | `gpu-1xa10g` |
| Training, one device | `gpu-1xa10g` |
| Training, several devices | A `4x` or `8x` shape, and start one process per device — see your repository's guide |

**This used to say to leave the field alone, and that advice was wrong the whole time it was there.** The workload profile appeared to name a machine and the form outranked it silently, so a run labelled `olmo-core-train-1gpu` could land on eight H100s and nothing anywhere said so. The catalog no longer claims a machine and this field is required, so you are asked once rather than defaulted somewhere you did not choose.

Your image is unaffected by this field. It is built from your commit for one architecture and runs on every shape here, so picking a bigger machine changes what you are billed and nothing about what runs.

## Approval

Every run waits for a person. Any of the eight team leads can release any group's run, so you are not blocked on one individual — but nobody is paged, so if a run has been waiting, ask.

If you are approving, it is not a formality. Before you release a run you are shown its cost, its machine, the team it is booked to, whether the submitter will be attributed, and whether it waived any check.

## The run id

Every run gets an id like `run_019fbce3-…`. That one string is the name of the job on AWS, the folder your output lands in, and the name of your run in Weights and Biases. It is stable across retries, and it is the only thing anyone needs from you when something goes wrong.

## Guards and waivers

The platform refuses commands that contradict the run you asked for. Each guard exists because it cost somebody a real run — asking for four GPUs and starting one process trains on a quarter of the machine, bills for all of it, and exits zero.

Override by putting a token in the command, which records the decision rather than routing around the check:

```
bash -lc 'EDULLM_LAUNCH_CHECK=waived python benchmarks/memory.py --batch 64'
```

There are two of them today and they are spelled the same way on purpose: `EDULLM_LAUNCH_CHECK=waived` for a multi-GPU machine running one process, and `EDULLM_CHECKPOINT_CHECK=waived` for a run that promised a checkpoint and saves somewhere the platform will not look. A command can carry both.

A waiver lands in the run's manifest and the approving lead is told which check was waived. What it does not do is make the underlying thing work — a waived checkpoint run that loses its machine starts from nothing.

## The corpora

| `dataset_release` | Train tokens | Objects |
| --- | --- | --- |
| `refhq-regmix-5p5b-v2` | 5.5B | 24 |
| `regmix-10b-v1` | 10.0B | 41 |
| `olmo-original-30b-v1` | 31.3B | 120 |
| `olmo-127b-v1` | 126.5B | 474 |
| `olmo-150b-dolma2-v1` | 157.2B | 6,851 |

All use the dolma2 tokenizer, all are frozen, and nothing you run can write to them.

**Size costs nothing up front.** Shards are memory-mapped from S3 as the loader reaches them, so a 157B corpus starts as quickly as a 5B one and reads only what your step count needs. Pick by what you are training, not by what you can afford to download.

## Container environment

Set for you. You never supply these, and a value you set yourself is one the record will disagree with.

| Variable | What it holds |
| --- | --- |
| `EDULLM_RUN_ID` | This run's id, stable across retries |
| `EDULLM_CHECKPOINT_DIR` | Where checkpoints go |
| `EDULLM_OUTPUT_PREFIX` | Where everything else this run writes goes |
| `EDULLM_TEAM` | The team you claimed |
| `EDULLM_COMMIT_SHA` | The commit that was resolved and built |
| `EDULLM_DATASET_RELEASE` | The dataset you named |
| `EDULLM_DATASET_ID`, `EDULLM_DATASET_VERSION`, `EDULLM_DATASET_TOKENIZER` | Which published corpus that resolves to, and what tokenised it. Absent when you picked `none` |
| `WANDB_PROJECT`, `WANDB_ENTITY` | Read by the W&B client directly |

The three dataset variables come from the registry entry behind the field you picked rather than from anything you typed.

## Looking at a run, and stopping one

[Look at a run, or stop it](https://github.com/edu-llm/platform/actions/workflows/cancel-run.yml) does both. Nothing changes unless you tick **stop**.

| You want to | Do this |
| --- | --- |
| See your latest run | Leave `run_id` blank |
| See a specific run | Give it the run id. You can look at anybody's |
| Stop a run | Tick **stop** and give a reason. Your own, any time; admins can stop anyone's |

You get the status, why it is not running if it is not, the exit code, and the CloudWatch log stream name. Batch reports `RUNNABLE` both for a job waiting on a machine and for one asking for more machine than exists — the reason beside the status is what tells those apart. A queued job bills nothing, but nobody is watching the queue, so ask if yours has not started within an hour.

**Pressing cancel on Submit a run is not this.** That stops the workflow and leaves the job running. Come here with the run id instead.

The reason you give is recorded, so the run's history says it was cancelled rather than that it failed. Anything already written stays, checkpoints included — so stopping a run to fix its command does not throw away the hours it already did.
