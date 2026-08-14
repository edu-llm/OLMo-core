# Final 370M / 10B validation

`final_validation.py` transfers each completed 190M probe winner to an otherwise stock
OLMo2-370M run on the sealed `pretrain/regmix-10b/v1` corpus.

## Fixed contract

- Model: `TransformerConfig.olmo2_370M`, Dolma2 tokenizer, 2,048-token sequences.
- Data: `pretrain/regmix-10b/v1`.
- Budget: the largest whole winner batch not exceeding 10B tokens
  (`9,999,990,784` tokens at the probe-default 16 Ki batch → `610,351` steps; at the
  256 Ki batch used for eduLLM / RunPod validation → `38,146` steps).
- Runtime: eight-rank HSDP, bf16 parameters, fp32 reductions, compilation enabled.
- Stock optimization behavior retained: `SkipStepAdamWConfig`, embedding weight-decay
  exemption, z-loss `1e-5`, seed `12,536`, and float8 disabled.
- Overrides: only the nine fields in `final-validation-vectors.json` (`lr`, weight decay,
  Adam beta2 gap/epsilon, global batch multiplier, WSD warmup/decay/terminal ratio, and
  max gradient norm).
- Durability/evaluation: 21 endpoint-inclusive, equally spaced checkpoints (step 0,
  19 interior points, final step). Every checkpoint runs the complete 20-label OLMES BPB
  task-loss suite and logs the result artifact to W&B. The final checkpoint is also
  uploaded as a W&B model artifact.

The two vector names are:

- `no-proxy-winner` — W&B probe run `904ea39d368dfe412048a6063c1600df`, trial `t9_0`.
- `no-centaur-winner` — W&B probe run `06e12699f744b8d2e562e78afa003b7f`, trial `t8_0`.

## RunPod

Stage `regmix-10b` once using the existing HPO staging command. The final-validation
launcher consumes the same sealed local manifest:

```bash
VECTOR=no-proxy-winner \
RUN_SLOT=no-proxy-v1 \
bash /workspace/OLMo-core/.edullm/runpod/launch_final_validation.sh
```

```bash
VECTOR=no-centaur-winner \
RUN_SLOT=no-centaur-v1 \
GLOBAL_BATCH_TOKENS=262144 \
bash /workspace/OLMo-core/.edullm/runpod/launch_final_validation.sh
```

eduLLM submissions use `.edullm/run-final-validation-no-centaur.yaml`, which passes
`--global-batch-tokens 262144` to match the completed no-proxy validation batch.

The first launch for a new `RUN_SLOT` must use the default `RECOVERY_MODE=fresh`. Do not
use `retry-startup` for that initial attempt: it can reuse startup identity while the
step-0 evaluator is still waiting for the pre-train checkpoint. Use `resume` only after
the run has created checkpoint state.

Resume the same checkpoint and W&B identities after interruption:

```bash
VECTOR=no-proxy-winner RUN_SLOT=no-proxy-v1 RECOVERY_MODE=resume \
bash /workspace/OLMo-core/.edullm/runpod/launch_final_validation.sh
```

The image must include `.edullm/requirements-task-loss-eval.txt`; the repository Dockerfile
installs and verifies that evaluator stack.

## OLMo-ladder control arm

`run-olmo-ladder-control.yaml` defines a separate no-curriculum control on the same
`pretrain/regmix-10b/v1` corpus and stock OLMo2-370M architecture. It is not one of the two
historical probe-winner vectors above. Its fixed optimizer contract is AdamW at
`7.78548e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, and gradient clipping
at `1.0`. The schedule linearly warms up for 0.5% of training and then cosine-decays to
`7.78548e-5`.

The 256 Ki global batch is split evenly across eight A100s as a 32 Ki rank microbatch. The
largest whole-batch run below 10B tokens is 38,146 steps (`9,999,745,024` tokens), with 191
warmup steps. It retains the 21-point checkpoint and OLMES task-loss evaluation ladder so its
W&B curve is directly comparable to the existing 370M validations.

Validate the platform submission with:

```bash
edullm check --json \
  --experiment olmo-ladder-control-370m \
  --dataset regmix-10b-v1 \
  --team pre-training \
  --spec .edullm/run-olmo-ladder-control.yaml \
  --compute gpu-8xa100 \
  --hours 10 \
  --attempts 2
```
