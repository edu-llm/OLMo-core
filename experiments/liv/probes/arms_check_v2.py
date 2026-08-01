from olmo_core.nn.transformer.liv_arms import (
    ARMS, VOCAB_SIZE, L0_PARAM_TARGET, _count_params, build_arm,
)
from olmo_core.data import TokenizerConfig

pad = TokenizerConfig.gpt2().padded_vocab_size()
print("VOCAB_SIZE          :", VOCAB_SIZE)
print("gpt2 padded_vocab   :", pad)
assert pad == VOCAB_SIZE, "vocab disagrees with OLMo-core's own gpt2 padding"

l0 = _count_params(build_arm("L0"))
print("L0 target           : {:,}".format(L0_PARAM_TARGET))
print("L0 built            : {:,}".format(l0))
assert l0 == L0_PARAM_TARGET

print()
print("PILOT ARMS")
for a in ("L0", "F-r128", "G-grouped", "N-narrow"):
    n = _count_params(build_arm(a))
    print("  {:<10} {:>13,}  ({:+.4f}% vs L0)".format(a, n, 100 * (n - l0) / l0))

print()
print("BUILD + LAYER TYPES (meta device)")
for a in ("L0", "F-r128", "G-grouped", "N-narrow"):
    m = build_arm(a).build(init_device="meta")
    kinds = [type(b.attention).__name__ for b in m.blocks.values()]
    print("  {:<10} ShortConv={:<3} Attention={:<3} flops@4K={:.3e}".format(
        a, kinds.count("ShortConv"), kinds.count("Attention"),
        m.num_flops_per_token(4096)))
