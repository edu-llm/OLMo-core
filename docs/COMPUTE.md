# Where to run: Colab A100 vs the edullm platform

## The premise needs correcting first

**There is no H100 option on the platform.** The `compute_profile` dropdown in the
submission workflow carries 14 options and both H100 shapes were removed, so the
form cannot express one; the CLI refuses it. Provisioning a new shape is a
four-edit infrastructure change (compute environment + queue + job definition, a
row in the execution targets, an entry in `CONTAINER_SHAPES`, and a `provisioned`
flag). A profile present in the catalog but absent from `CONTAINER_SHAPES` is
rejected at admission.

**But A100 is available**, as `gpu-8xa100` ($21.9576/hr, 8 GPUs, provisioned). There
is no single-GPU A100 shape, so using A100s means one independent arm per GPU on an
8-GPU node. Shapes, costs and the submission plan live in `RUNBOOK.md`; this file is
the choice-of-backend argument only.

The headline correction: **for wall-clock, Colab is the worst option, not the best.**
One interactive session runs 12 arms in series (~26 h); the platform runs them in
parallel (~1.7 h on two 8xA100 nodes, ~3.0 h on twelve L40S). Per-GPU speed is not the
binding constraint -- concurrency is.

## OLMo integration is not required

This was the other open question, and the answer is no. `edullm-p1-train` is
precedent for exactly our situation: a custom non-OLMo entrypoint that execs its
own `torchrun` internally, taking the launcher waiver because the platform's
text-reading guard cannot see inside the program:

```
bash -lc 'EDULLM_LAUNCH_CHECK=waived python experiments/.../platform_array_entrypoint.py \
    --checkpoint-prefix "$EDULLM_CHECKPOINT_DIR"'
```

`.edullm/train.py` here matches the scaffold's auto-generated signature (positional
run id, `--save-folder`), so `edullm check` should produce a valid first spec
without hand-editing. We are single-GPU at these model sizes, so no launcher
waiver is needed at all.

One caveat worth knowing: in the platform's recorded evidence, every command that
has actually run is either `python -c '...'` or `python -m olmo_core.train`. The
custom-entrypoint path is architecturally supported and configured, but we would
be **proving that path rather than following it**. Budget a check-profile job for
that.

## Throughput and cost

Measured anchors, all single-GPU, ctx=1024, micro-batch 32, compiled:

| | d8m | d40m | d160m |
|---|---:|---:|---:|
| L40S (measured) | 462,611 tok/s | 184,671 tok/s | ~132,000 tok/s |
| A100 40GB (measured) | — | — | ~185,000 tok/s @ ctx=2048 |

At these sizes training is memory-bandwidth bound rather than FLOP bound, which
is why the A100 beat the L40S by 1.40x at d160m *despite running twice the
context*. Scaling by bandwidth (A100 1555 GB/s, L40S 864, L4 300):

| d40m, 2B tokens | tok/s | hours/run | 12 runs | cost |
|---|---:|---:|---:|---:|
| Colab A100 | ~258,000 (est.) | 2.2 | 26 h | subscription |
| Platform A10G | ~128,000 (est.) | 4.3 | 52 h wall-parallel | **~$52** |

At the longer 8.17B budget the crowding stage needs: A100 ~105 h total, A10G ~177 h
= **~$179**.

**Both non-L40S numbers are estimated by bandwidth scaling, not measured** (A10G
600 GB/s, A100 1555, L40S 864, against the measured L40S `d40m` figure of 184,671
tok/s). Replace them with a measurement before committing the matrix -- Job 0 in the
runbook is a ~5-minute, ~$0.10 submission that reports `tok_s`, since walltime is a
cap billed for actual time rather than a reservation.

## Recommendation: platform for the matrix, Colab for iteration

The 4x per-GPU speed advantage is real and does not matter much here, because:

* The matrix is 12 independent single-GPU runs. Twelve jobs queued in parallel
  beat one interactive session that is 4x faster, and the dollar cost is ~$84.
* **Colab loses runs.** The previous generation's split arm died at 0.87B of 1.0B
  tokens when a session ended, and the 0.797B snapshot was then compared against
  the dense arm's 0.996B one. That is not a hypothetical failure mode; it is the
  reason one of the two headline anchors is not iso-token.
* The platform enforces a checkpoint contract (`resume_required: true` on every
  train profile but one), so a preempted attempt resumes rather than restarting.
  Our `checkpoint_io.ResumeGuard` raises if a second attempt finds nothing to load,
  which is the failure a sibling repo shipped: it gated its load on
  `os.path.exists()` against an `s3://` URI -- always false -- so every retry
  silently repeated the previous attempt at full price.
* Cost is visible per run and manifests are hash-pinned, so the matrix is
  auditable after the fact. The previous line could not reproduce Experiment 1
  from its own tree.

Use Colab for: the calibration gate (it needs **no GPU at all** -- pure CPU, and
it is the thing that must run first), interactive debugging, and single pilot runs
where turnaround matters more than durability.

### Resolved: concurrency is not a constraint

`gpu-1xa10g` supports **96 concurrent single-GPU jobs**, and 12 runs is 48 of the
768-vCPU G quota. G and P are separate pools, so P jobs cannot starve us. There is
no per-user or per-team job cap anywhere in the config. This was the one open input
that could have flipped the recommendation; it doesn't.

Two related facts worth carrying: **jobs are not preemptible** (every compute
environment is `Type: EC2`, not SPOT, so retries fire only for a lost host), which
means resume is insurance against a walltime overrun rather than the normal path.
And an *unplaceable* job sits in `RUNNABLE` under a state the submitter cannot read,
with a **1800-second auto-cancel** -- so a submission that disappears after ~30
minutes was over-quota, not failed.

## Practical notes

* Checkpoints go to `$EDULLM_CHECKPOINT_DIR`, an S3 prefix under
  `s3://sbsandbox-intern-edullm-outputs/teams/`. Never hardcode a bucket; the
  literal that appears in older fixtures points at a bucket that is no longer the
  convention.
* Node Python is 3.12.13; egress is limited to ports 443/80, so dependencies can
  be installed but plan on the image rather than runtime downloads. `tiktoken` must
  be in the image -- the byte-level fallback here is for tests, and
  `require_production_tokenizer` refuses to write a trainable corpus with it.
* Multi-GPU launchers are matched textually from a fixed list; wrapper scripts
  (`./scripts/train.sh`) are refused because the guard cannot see into them.
* `experiment` groups runs in the cost view but is not part of the hashed lineage
  record, so use it freely for organisation.

## Suggested sequence

1. Calibration gate locally or on Colab CPU: `scripts/calibrate_nhop.py`. No GPU,
   must pass before anything trains.
2. One platform **check** profile job (1 h) running ~50 steps of `d40m` to measure
   real `tok_s` and prove the custom-entrypoint path end to end.
3. Re-cost the matrix from that measurement; submit the 12-run depth matrix.
4. Keep Colab for the mechanistic probes, which are interactive and short.
