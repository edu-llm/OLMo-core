---
type: "query"
date: "2026-07-25T21:44:40.213239+00:00"
question: "what forces d_state=128 across the presets, and what would break if the NC1 ablation used 96 instead"
contributor: "graphify"
outcome: "useful"
source_nodes: ["mamba3_hybrid_like", "mamba3_olmo3_370M", "_validate_dims", "test_mamba3_olmo3_370M_default_d_state_cannot_express_block_size_3", "test_mamba3_olmo3_370M_is_parameter_matched", "build_a5_model"]
---

# Q: what forces d_state=128 across the presets, and what would break if the NC1 ablation used 96 instead

## Answer

Expanded from original query via graph vocab: [state, rotation, block, size, divisible, indivisible, mixer, preset, hybrid, olmo, mamba, abelian]. The graphify query CLI collapsed to 4 nodes (matched the torch 'Size' label), so traversal was done inline with explicit start nodes. Finding: mamba3_hybrid_like() (config.py:164) is the single funnel that declares d_state=128 (config.py:176); mamba3_hybrid_190M, mamba3_hybrid_1B, mamba3_olmo3_370M AND build_a5_model all call it, and each preset re-declares d_state=128 in its own signature. Nothing forces 128 architecturally - it is an overridable default and config.py:293/312 say so explicitly. d_state=96 is a supported, tested path: model_test.py:179 pins that b=3 at the default d_state fails loudly, and model_test.py:257 asserts d_state=96 + rotation_block_size=3 builds. What actually breaks is the experimental design, not the model: bc_out = n_groups*mimo_rank*d_state so in_B/in_C shrink 25 percent at 96, which breaks test_mamba3_olmo3_370M_is_parameter_matched, and it breaks the invariant behind test_mamba3_olmo3_370M_switches_block_size_without_touching_anything_else (rationale: 'Flipping rotation_block_size must be the only difference between the arms'). Three-way tension: OLMo-3 parameter matching, single-variable ablation, and b=3 trainability - pick two. Note the A5 harness build_a5_model already dodges this by hardcoding d_state=48, which divides by 2,3,4,6,8.

## Outcome

- Signal: useful

## Source Nodes

- mamba3_hybrid_like
- mamba3_olmo3_370M
- _validate_dims
- test_mamba3_olmo3_370M_default_d_state_cannot_express_block_size_3
- test_mamba3_olmo3_370M_is_parameter_matched
- build_a5_model