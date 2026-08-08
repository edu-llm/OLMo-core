---
type: "query"
date: "2026-07-25T21:48:42.503537+00:00"
question: "what does test_mamba3_olmo3_370M_is_parameter_matched compare against, and could a wider d_state still be parameter-matched"
contributor: "graphify"
outcome: "useful"
source_nodes: ["test_mamba3_olmo3_370M_is_parameter_matched", "mamba3_olmo3_370M", "mamba3_hybrid_like", "test_mamba3_olmo3_370M_default_d_state_cannot_express_block_size_3", "build_a5_model"]
---

# Q: what does test_mamba3_olmo3_370M_is_parameter_matched compare against, and could a wider d_state still be parameter-matched

## Answer

Traced from test_mamba3_olmo3_370M_is_parameter_matched. It compares config.num_active_non_embedding_params against TransformerConfig.olmo3_370M's same metric (model_test.py:133-139), NOT total params: reference is 371.26M active-non-emb vs 474.0M total, because both leave the LM head untied. Tolerance is 3 percent deliberately. Measured: d_state=128 gives 362.99M = 2.23 percent off, exactly the 2.2 percent the docstring claims; the gap is mimo_rank=1 (bc_out = n_groups*mimo_rank*d_state, 4->1 removes 787k/layer x 12 = 9.45M). Confirmed the rejected alternative: n_groups=4 at d_state=128 gives 374.79M = 0.95 percent, as documented. KEY RESULT: the Mamba preset sits BELOW the reference, so widening d_state closes the gap. d_state=192 gives 364.96M = 1.70 percent, strictly better than 128 on parameter match AND admits b in 2,3,4,6 instead of only 2,4. So 192 dominates 128 on both axes; 96 is worse on params (2.49 percent). The remaining real trade is that no power of two is divisible by 3 (verified), so any b=3 arm pays padding in the official kernel: _padded(192)=256, 25 percent of QK lanes are zeros, numerically neutral, official path only, and partly moot since rotation preprocessing is ~70 percent of b>=3 mixer time.

## Outcome

- Signal: useful

## Source Nodes

- test_mamba3_olmo3_370M_is_parameter_matched
- mamba3_olmo3_370M
- mamba3_hybrid_like
- test_mamba3_olmo3_370M_default_d_state_cannot_express_block_size_3
- build_a5_model