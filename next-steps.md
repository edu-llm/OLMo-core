# Next steps (deferred work)

Items intentionally deferred from the Mamba-3 training-readiness plan
(`.cursor/plans/mamba-3_training_readiness_5cea65bb.plan.md`). Not required to train
the SISO Mamba-3 hybrid at the target scale.

## Tensor parallelism (TP) for the Mamba-3 mixer
Deferred. FSDP/HSDP + CP already cover multi-node training for the target model sizes; TP is
only needed for models too large to fit with FSDP alone. `Mamba3Mixer.apply_tp` stays a
`NotImplementedError` for now.

When revisited (Megatron-style head sharding, feasible - Mamba-2 does it):
- Shard `nheads` across the TP mesh: colwise-parallelize the `in_proj` slices feeding
  `x/z/dt/A/trap`; rowwise-parallelize `out_proj` (partial sums all-reduced).
- Slice the per-head params (`D`, `mimo_x/z/o`, gated-norm weights) to the local head range.
- Enforce `ngroups % tp_size == 0` so each rank gets whole B/C groups (shared B/C stays with
  its heads).
- Call the fused kernel on the local head slice (the official kernel is per-head).
- Validate single-GPU vs 2-GPU parity (SISO first). Requires >=2 GPUs.

## Checkpoint / weight interop with the official mamba-ssm / HuggingFace layout
Deferred (ignored for now). Only needed to (a) load official Mamba-3 pretrained weights, or
(b) export our trained model to HuggingFace (`AutoModelForCausalLM`). Both require aligning our
parameter names/shapes/layout (combined `in_proj` split order, B/C group layout,
`mimo_x/z/o`) with the official module, plus the P2 HF-conversion key maps. Not required to
train from scratch in-house.

Decision to make before finalizing param names IF interop later matters: mirror the official
`mamba_ssm.modules.mamba3.Mamba3` layout so checkpoints map cleanly in both directions.
