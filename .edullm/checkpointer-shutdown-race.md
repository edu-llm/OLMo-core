# The checkpointer outliving its process group

A run that fails for any reason after `pre_train` can die reporting a checkpointing fault
it does not have, and leave a half-written checkpoint directory behind. This is what that
is, because it cost hours of looking in the wrong place once already.

## What happens

`CheckpointerCallback.pre_train` does two things. It creates a Gloo process group for async
checkpointing, and at step 0 it launches a save on a background thread and returns without
waiting:

```
Creating new process group for checkpointing (needed for async checkpointing)
```

The callback has `priority = 1`, so it runs early and every other callback's `pre_train`
runs after the save is already in flight.

`post_train` is the only place that awaits that save. When any later callback raises,
`Trainer.fit` takes its `except BaseException` branch, calls `_shutdown(gracefully=False)`
and re-raises. `post_train` is outside the `try` and never runs. Nothing awaits the save.

`.edullm/train_on_corpus.py` tears the environment down in a `finally`, so
`teardown_training_environment` runs on the way out and calls `destroy_process_group`. The
DCP writer thread is still inside a collective on the group being destroyed.

## What it looks like

Two things, and neither points at the real fault.

The process dies with `ValueError: Group ... is not registered`, raised out of the
checkpointer. The actual failure was in whatever callback raised first, and its traceback
is the one that matters. This is what sent the eduLLM work chasing the checkpointer for
hours when the failure was `wandb.init()` refusing a bad credential.

The checkpoint directory is torn. The trainer state is written synchronously and the model
and optimizer shards are not, so what survives is `step0/train/rank0.pt` and nothing else.
Eight runs in `sbsandbox-intern-edullm-outputs` are in exactly that state, 15317 bytes each
against roughly 3.2 GB for a real step of a 190M model. A resume from one starts at step
zero. Two of the eight died with `RuntimeError: cannot schedule new futures after
interpreter shutdown`, which is the same teardown seen from the other side.

## What was done about it

`CheckpointerCallback.close` now awaits the in-flight save. `close` is called from
`_shutdown` on both paths, so it covers the error path that `post_train` cannot reach, and
it is a no-op on the success path because `post_train` has already cleared the future.

The wait is bounded by `shutdown_timeout`, 300 seconds by default. `_shutdown(gracefully=
False)` exists so that ranks already in an inconsistent state do not deadlock on further
collectives. If one rank raised and the others did not, the save can never complete, and an
open-ended wait would turn a crash into a hang that runs to the job's time limit. On one
GPU, where every rank fails together, the wait completes.

## What has not been done

**The image is not rebuilt.** `sha256:6e0dd353` is the digest the proven twelve-hour
training run used, and repinning the GPU job definition while that run or another like it
is in flight is not worth the fix. Rebuild through
`.github/workflows/build-research-image.yml` in the platform repository and repin
deliberately, when no long run depends on the current digest.

**Not tested on more than one rank.** The bounded wait is the branch that matters for
multi-rank, and reaching it needs a real multi-GPU failure. The unit tests cover the future
resolving, the bound expiring, and a save that failed, none of which need a GPU.

**Not sent upstream.** The file is identical to `upstream/main`, so the defect is upstream's
too and the fix applies there unchanged. It is worth contributing.

## Not the SIGTERM path

Cancellation does not come through here. `_handle_os_signal` calls `cancel_run`, the loop
exits normally, and `post_train` runs and awaits. The resume story is unaffected by this
change in either direction.
