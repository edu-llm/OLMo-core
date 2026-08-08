import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.residual_stream import HyperConnectionConfig
from olmo_core.nn.transformer import (
    BlockReuseConfig,
    BlockReusePattern,
    TransformerBlockType,
    TransformerConfig,
)

VOCAB_SIZE = 128
D_MODEL = 64
SEQ_LEN = 16
N_LAYERS = 8


def build_config(
    *,
    block_reuse: BlockReuseConfig | None = None,
    hyper_connections: HyperConnectionConfig | None = None,
) -> TransformerConfig:
    block_name = (
        TransformerBlockType.hyper_connection_reordered_norm
        if hyper_connections is not None
        else TransformerBlockType.reordered_norm
    )
    config = TransformerConfig.llama_like(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=4,
        block_name=block_name,
        qk_norm=True,
    )
    assert not isinstance(config.block, dict)
    config.block.hyper_connections = hyper_connections
    config.block_reuse = block_reuse
    config.__post_init__()
    return config


@pytest.mark.parametrize(
    "n_unique, expected",
    [
        (8, [0, 1, 2, 3, 4, 5, 6, 7]),
        (4, [0, 1, 2, 3, 0, 1, 2, 3]),
        (2, [0, 1, 0, 1, 0, 1, 0, 1]),
        (1, [0] * 8),
    ],
)
def test_cycle_pattern(n_unique: int, expected: list):
    reuse = BlockReuseConfig(n_unique_blocks=n_unique)
    assert reuse.execution_order(N_LAYERS) == expected


def test_middle_cycle_keeps_the_ends_unique():
    reuse = BlockReuseConfig(n_unique_blocks=4, pattern=BlockReusePattern.middle_cycle)
    order = reuse.execution_order(N_LAYERS)
    assert order == [0, 1, 2, 1, 2, 1, 2, 3]
    assert order[0] == 0 and order[-1] == 3


def test_reuse_shares_parameters_at_matched_effective_depth():
    """
    The tied and untied arms have to be matched on effective depth rather than on physical
    depth, so that the comparison is about weight sharing and not about compute.
    """
    untied = build_config()
    tied = build_config(block_reuse=BlockReuseConfig(n_unique_blocks=4))

    untied_model = untied.build()
    tied_model = tied.build()

    assert len(untied_model.blocks) == 8
    assert len(tied_model.blocks) == 4
    assert len(tied_model.block_execution_order) == N_LAYERS

    # Same FLOPs, roughly half the block parameters.
    assert untied_model.num_flops_per_token(SEQ_LEN) == tied_model.num_flops_per_token(SEQ_LEN)
    untied_blocks = untied.num_params - untied.d_model * untied.vocab_size
    tied_blocks = tied.num_params - tied.d_model * tied.vocab_size
    assert tied_blocks < untied_blocks
    assert tied.num_params == tied_model.num_params


def test_reused_blocks_run_more_than_once():
    model = build_config(block_reuse=BlockReuseConfig(n_unique_blocks=2)).build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ_LEN)

    calls: list = []
    for key, block in model.blocks.items():
        block.register_forward_hook(lambda *_args, key=key: calls.append(key))

    model(torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)))
    assert calls == ["0", "1"] * 4


def test_reuse_composes_with_hyper_connections():
    """
    Arms 10 and 11 are this crossing: whether widening the stream is worth more when the same
    parameters are applied repeatedly than when every layer has its own.
    """
    hc = HyperConnectionConfig(n_lanes=4, output_init_exponent=0.0)
    reuse = BlockReuseConfig(n_unique_blocks=4)

    tied_baseline = build_config(block_reuse=reuse).build()
    tied_hc = build_config(block_reuse=reuse, hyper_connections=hc).build()
    for model in (tied_baseline, tied_hc):
        model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ_LEN)
        model.eval()

    assert tied_hc.residual_lanes == 4
    assert len(tied_hc.blocks) == 4

    # Initialization equivalence has to survive reuse too, or arm 10 starts somewhere arm 11
    # does not.
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    with torch.no_grad():
        torch.testing.assert_close(
            tied_hc(input_ids), tied_baseline(input_ids), atol=1e-4, rtol=1e-4
        )


def test_reuse_rejects_a_pattern_that_does_not_fit():
    with pytest.raises(OLMoConfigurationError, match="between 1 and n_layers"):
        BlockReuseConfig(n_unique_blocks=99).execution_order(N_LAYERS)
    with pytest.raises(OLMoConfigurationError, match="at least 3 unique blocks"):
        BlockReuseConfig(n_unique_blocks=2, pattern=BlockReusePattern.middle_cycle).execution_order(
            N_LAYERS
        )


def test_reuse_rejects_block_overrides():
    config = build_config()
    assert not isinstance(config.block, dict)
    config.block_reuse = BlockReuseConfig(n_unique_blocks=4)
    config.block_overrides = {0: config.block}
    with pytest.raises(OLMoConfigurationError, match="cannot be combined"):
        config.__post_init__()


def test_no_reuse_leaves_the_execution_order_alone():
    model = build_config().build()
    assert model.block_execution_order == list(range(N_LAYERS))
    assert model.block_reuse is None
