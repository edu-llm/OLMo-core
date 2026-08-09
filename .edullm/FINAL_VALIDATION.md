# Final 370M / 10B validation

`final_validation.py` transfers each completed 190M probe winner to an otherwise stock
OLMo2-370M run on the sealed `pretrain/regmix-10b/v1` corpus.

## Fixed contract

- Model: `TransformerConfig.olmo2_370M`, Dolma2 tokenizer, 2,048-token sequences.
- Data: `pretrain/regmix-10b/v1`.
- Budget: the largest whole winner batch not exceeding 10B tokens
  (`9,999,990,784` tokens, `610,351` steps for both registered winners).
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
bash /workspace/OLMo-core/.edullm/runpod/launch_final_validation.sh
```

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
