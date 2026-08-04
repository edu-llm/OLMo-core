# submit-run.yml payloads

Exact `gh workflow run submit-run.yml -R edu-llm/platform --json < <file>` inputs for every
submission this implementation made. Kept because reconstructing one by hand is error-prone —
the `command` field is ~400 characters with two kinds of quote and a shell variable that must
arrive *unexpanded* in the JSON but *expanded* inside `bash -lc`.

| file | profile | $ ceiling | outcome |
|---|---|---|---|
| `impl3x5-job0-gate.json` | gpu-4xl40s | 41.97 | ran; all stages passed, then exit 1 on the appended `aws s3 cp` — **the AWS CLI is not in the image** |
| `impl3x5-jobB-train-selfcontained.json` | gpu-4xl40s | 83.94 | **SUCCEEDED** — 4 arms, 22 ckpts each, ped_nll |
| `impl3x5-job2-math.json` | gpu-4xl40s | 41.97 | cancelled — `g6e.12xlarge` capacity exhausted in all 4 AZs |
| `impl3x5-job2-math-a10g.json` | gpu-4xa10g | 22.69 | queued — `g5.12xlarge` also capacity-exhausted |
| `impl3x5-job2-math-1gpu.json` | gpu-1xa10g | 8.05 | the one that should actually get hardware |
| `impl3x5-gateA-1gpu.json` | gpu-1xl40s | 3.35 | never submitted; would have been AUTOMATIC (no approver) |
| `impl3x5-cancel-l40s.json` | — | — | `cancel-run.yml` input; note `stop` must be the **string** `"true"` |

## Three things that will bite whoever edits these

**`EDULLM_LAUNCH_CHECK=waived`** is required on any multi-GPU profile, because the platform reads
"no launcher" as one process and refuses a 4-GPU shape. Do **not** replace it with `torchrun` —
HF `Trainer` detects `WORLD_SIZE`/`RANK` and switches to DDP, multiplying the effective batch and
destroying the batch shape these runs hold fixed against their baseline. The 1-GPU payload needs
no waiver and does not carry it.

**`"$EDULLM_CHECKPOINT_DIR"` must stay inside the `bash -lc` string.** The container execs with
no shell, so without the wrapper the variable arrives as 22 literal characters and the guard
refuses the submission.

**Capacity is about instance SIZE, not AZ breadth.** Both `.12xlarge` 4-GPU sizes went dry;
`g6e` is offered in 4 AZs and `g5` in 5, and it made no difference. Single-GPU `.xlarge` sizes
are the plentiful ones. Diagnose with `autoscaling describe-scaling-activities` — Batch shows
none of this on the job record, it just sits `RUNNABLE` forever.
