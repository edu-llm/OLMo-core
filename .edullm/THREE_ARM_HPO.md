# Three-arm HPO contract

The preregistered study has exactly three arms:

1. `full_acronym_soup`: FT-PFN + ifBO + IPBT/BTT, 30% `gpt-5.6-sol`
   multi-action Centaur, the same-depth 16-layer width-reduced u-μP proxy, and embeddings plus
   blocks 0–7 frozen.
2. `no_centaur`: identical model, freezing, and controller stack, with Centaur disabled.
3. `no_proxy`: FT-PFN + ifBO + IPBT/BTT and the same 30% Centaur policy as Arm 1, but the
   conventional stock 12-layer `olmo2_190M`, fully trainable. It has no u-μP metadata,
   width-transfer behavior, or layer freezing.

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
dataset `reservoir-dolma2-v1`. Expected wall time is about 1–1.5 hours per arm; pass
`--hours 4` as the hard runtime bound and `--attempts 2` (the `olmo-core-train` maximum).

Runtime secrets:

| Secret | Proxy cohort | Arm 1 / 3 | Arm 2 (`no_centaur`) |
|--------|--------------|-----------|----------------------|
| `WANDB_API_KEY` | required | required | required |
| `OPENAI_API_KEY` | not used | required (30% Centaur) | not used |

HPO artifacts mirror to W&B project `hpo-probe` (not the platform `wandb_project`).

Proxy cohort first:

```bash
edullm check --json \
  --experiment hpo-proxy-cohort \
  --dataset reservoir-dolma2-v1 \
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
  --dataset reservoir-dolma2-v1 \
  --team pre-training \
  --compute gpu-8xa100 \
  --hours 4 \
  --attempts 2
```
