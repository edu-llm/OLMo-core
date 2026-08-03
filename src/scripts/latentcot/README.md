# latentcot scripts

Runnable training/eval scripts for the latent chain-of-thought experiments
(CODI continuous-thought substrate + superposition / distributional-shift study).

Pre-registered design and build checklist: `docs/latent-cot/latent-cot-superposition-prd.md`.

Planned (per the PRD build checklist, section 8):

- `train_codi.py` — adapted from `src/scripts/train/template.py` for the CODI train
  module; primary rung `TransformerConfig.olmo2_370M`.
- `eval.py` — gates A/B (superposition slope; vocab-regularization effect) + probing harness.

Package code lives in `src/olmo_core/latentcot/`.
