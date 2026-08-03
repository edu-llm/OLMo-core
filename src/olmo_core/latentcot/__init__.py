"""
Latent chain-of-thought reasoning for OLMo-core (CODI substrate + superposition study).

Experiment code for the pre-registered PRD at ``docs/latent-cot/latent-cot-superposition-prd.md``.
Everything for the continuous-thought / superposition experiments lives under this
package so it stays isolated from the rest of OLMo-core and does not conflict on ``main``.

Primary model rung: :meth:`olmo_core.nn.transformer.TransformerConfig.olmo2_370M`.

Planned modules (see the PRD build checklist, section 8):

- ``data/``            : synthetic directed-graph reachability generator + tokenization
- ``cot.py``           : continuous-thought forward loop (feeds the last hidden state
                         back as the next input embedding via ``Transformer.forward(..., input_embeddings=...)``)
- ``train_module.py``  : ``CodiTransformerTrainModule`` (dual-branch CODI + feature
                         distillation + vocabulary-manifold regularizer)
- ``probes.py``        : logit-lens / linear / causal-intervention probing for superposition
"""
