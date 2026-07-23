# Worked-examples → W&B metrics smoke (eduLLM)

## Goal

Confirm training logs metrics to **W&B entity `eduLLM`** before a scientific
MetaMath CPT request. Use the platform **generic-smoke** profile (policy-approved).

## Metrics to expect (from selected code)

`src/examples/llm/train.py` enables `WandBCallback` via ORCD
`src/scripts/orcd/run_generic_smoke.sh`. Observed metric namespaces include:

- `train/CE loss` (primary)
- train throughput / tokens metrics (when emitted)
- `optim/*` LR / grad-norm style metrics (when emitted)

These are the names you may list under **Success metrics** for a generic-smoke
request (comma-separated), e.g. `train/CE loss`.

If you need a custom scientific metric that is **not** emitted, stop and use
`/weights-and-biases` to wire it in a reviewed commit first (that skill is
documented in the ORCD design docs; implement on a feature branch, never on `main`).

## How to request the smoke

1. Push this feature branch (clean tree).
2. Run `/submit-edullm-job`.
3. When asked for intent, you may choose the **generic-smoke** fixture values
   from `.claude/skills/submit-edullm-job/request-reference.md`.
4. Operator runs `edullm run` after Actions marks the Issue ready.

## Not this smoke

- Loading `allenai/OLMo-Ladder-760M-0.5xC`
- Training on `hiyasvyas/worked-examples-metamath-v0`

Those are the scientific CPT follow-up after the engineering path works and
policy allows a dedicated entrypoint.
