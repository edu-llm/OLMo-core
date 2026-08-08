from dataclasses import replace

import pytest
import torch

from olmo_core.nn.residual_stream import (
    HyperConnectionConfig,
    HyperConnectionMode,
    HyperConnectionStream,
    expand_residual_lanes,
    output_init_scale,
    reduce_residual_lanes,
    sinkhorn_knopp,
)
from olmo_core.nn.transformer import (
    HyperConnectionReorderedNormTransformerBlock,
    TransformerBlockType,
    TransformerConfig,
)

D_MODEL = 64
VOCAB_SIZE = 128
N_LAYERS = 4
SEQ_LEN = 16


def build_config(
    *,
    hyper_connections: HyperConnectionConfig | None = None,
    d_model: int = D_MODEL,
    n_layers: int = N_LAYERS,
    n_heads: int = 4,
    **kwargs,
) -> TransformerConfig:
    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=VOCAB_SIZE,
        n_layers=n_layers,
        n_heads=n_heads,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        **kwargs,
    )
    if hyper_connections is not None:
        assert not isinstance(config.block, dict)
        config.block = replace(
            config.block,
            name=TransformerBlockType.hyper_connection_reordered_norm,
            hyper_connections=hyper_connections,
        )
    return config


def build_model(config: TransformerConfig, *, seed: int = 0) -> torch.nn.Module:
    config.init_seed = seed
    model = config.build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ_LEN)
    model.eval()
    return model


@pytest.mark.parametrize("n_lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", [HyperConnectionMode.full, HyperConnectionMode.output])
@pytest.mark.parametrize("dynamic", [True, False])
def test_init_is_equivalent_to_the_residual_stack_it_replaces(
    n_lanes: int, mode: HyperConnectionMode, dynamic: bool
):
    """
    Eq. 14 is chosen so that a freshly initialized hyper-connection model computes exactly what
    the ordinary residual model computes. If that does not hold, every arm is measuring an
    initialization difference on top of the mechanism and none of them mean anything.

    The output-init correction is off here because it deliberately changes the initialization.
    """
    hc = HyperConnectionConfig(
        n_lanes=n_lanes, mode=mode, dynamic=dynamic, output_init_exponent=0.0
    )
    baseline = build_model(build_config())
    treated = build_model(build_config(hyper_connections=hc))

    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    with torch.no_grad():
        expected = baseline(input_ids)
        actual = treated(input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n_lanes", [2, 4])
def test_mhc_init_is_also_equivalent(n_lanes: int):
    """
    The Birkhoff projection reads ``A_r`` as logits, so the identity has to survive a round trip
    through Sinkhorn-Knopp for arm 9 to start from the same place as the others.
    """
    hc = HyperConnectionConfig(n_lanes=n_lanes, doubly_stochastic=True, output_init_exponent=0.0)
    baseline = build_model(build_config())
    treated = build_model(build_config(hyper_connections=hc))

    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    with torch.no_grad():
        expected = baseline(input_ids)
        actual = treated(input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_lanes_are_identical_at_init_and_differentiate_once_trained():
    """
    The rehearsal's fail-closed guard rests on this: identical lane norms mean the mechanism is
    inert. At init they are identical by construction, so the guard has to be read after the
    parameters have moved.
    """
    hc = HyperConnectionConfig(n_lanes=4, output_init_exponent=0.0)
    model = build_model(build_config(hyper_connections=hc))

    lanes = {}

    def capture(_module, _args, output):
        lanes["value"] = output

    handle = model.blocks["2"].register_forward_hook(capture)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    at_init = lanes["value"]
    assert at_init.shape == (2, SEQ_LEN, 4, D_MODEL)
    per_lane = at_init.norm(dim=-1).mean(dim=(0, 1))
    torch.testing.assert_close(per_lane, per_lane.roll(1), atol=1e-5, rtol=1e-5)

    # Move the static matrices off their initialization the way a few steps of training would.
    with torch.no_grad():
        for block in model.blocks.values():
            for stream in (block.attention_residual_stream, block.feed_forward_residual_stream):
                stream.hc_static_alpha_r.add_(torch.randn_like(stream.hc_static_alpha_r) * 0.1)
                stream.hc_static_beta.add_(torch.randn_like(stream.hc_static_beta) * 0.1)

    handle = model.blocks["2"].register_forward_hook(capture)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    per_lane = lanes["value"].norm(dim=-1).mean(dim=(0, 1))
    assert per_lane.std() > 1e-4, "lanes did not differentiate after perturbing the mixing matrix"


def test_parameter_count_matches_the_papers_formula():
    """
    Eq. 25 with OLMo-1B's dimensions has to come back as the 394,048 the paper reports for
    DHC x4, which pins down that the static and dynamic pieces are the right shapes.
    """
    d_model, n_lanes, n_layers = 2048, 4, 16
    per_stream = HyperConnectionStream.expected_num_params(
        d_model=d_model, n_lanes=n_lanes, dynamic=True
    )
    assert per_stream * 2 * n_layers == 394_048

    stream = HyperConnectionStream(d_model=d_model, n_lanes=n_lanes, block_idx=0)
    assert stream.num_params() == per_stream


def test_output_mode_drops_the_input_side_mapping():
    full = HyperConnectionStream(
        d_model=D_MODEL, n_lanes=4, block_idx=1, mode=HyperConnectionMode.full
    )
    output_only = HyperConnectionStream(
        d_model=D_MODEL, n_lanes=4, block_idx=1, mode=HyperConnectionMode.output
    )

    assert full.hc_static_alpha_m is not None
    assert output_only.hc_static_alpha_m is None
    assert output_only.hc_dynamic_w_m is None
    assert output_only.num_params() < full.num_params()


def test_static_and_dynamic_parameters_are_separable_by_name():
    """
    ByteDance exclude the static component from weight decay and keep it on the dynamic one,
    which is done here with two globs, so the names have to partition cleanly.
    """
    stream = HyperConnectionStream(d_model=D_MODEL, n_lanes=4, block_idx=0)
    names = [n for n, _ in stream.named_parameters()]
    static = [n for n in names if n.startswith("hc_static_")]
    dynamic = [n for n in names if n.startswith("hc_dynamic_")]
    assert sorted(static + dynamic) == sorted(names)
    assert static and dynamic


@pytest.mark.parametrize("n_lanes", [2, 4, 8])
def test_sinkhorn_returns_a_doubly_stochastic_matrix(n_lanes: int):
    logits = torch.randn(3, n_lanes, n_lanes)
    projected = sinkhorn_knopp(logits, num_iters=64)

    assert (projected >= 0).all()
    ones = torch.ones(3, n_lanes)
    torch.testing.assert_close(projected.sum(dim=-2), ones, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(projected.sum(dim=-1), ones, atol=1e-4, rtol=1e-4)

    # This is mHC's whole argument: the spectral radius is pinned at 1, so the composite across
    # depth cannot blow up the way Tencent's diverging runs did.
    eigenvalues = torch.linalg.eigvals(projected[0])
    assert eigenvalues.abs().max().item() == pytest.approx(1.0, abs=1e-4)


def test_expand_and_reduce_round_trip():
    h = torch.randn(2, SEQ_LEN, D_MODEL)
    lanes = expand_residual_lanes(h, 4)
    assert lanes.shape == (2, SEQ_LEN, 4, D_MODEL)
    torch.testing.assert_close(reduce_residual_lanes(lanes), h * 4)
    torch.testing.assert_close(reduce_residual_lanes(lanes, average=True), h)


@pytest.mark.parametrize(
    "n_lanes, exponent, expected",
    [(4, 0.5, 0.5), (4, 1.0, 0.25), (4, 0.0, 1.0), (1, 0.5, 1.0), (8, 0.5, 8**-0.5)],
)
def test_output_init_scale(n_lanes: int, exponent: float, expected: float):
    assert output_init_scale(n_lanes, exponent) == pytest.approx(expected)


def test_output_init_scaling_shrinks_the_output_modules():
    hc = HyperConnectionConfig(n_lanes=4, output_init_exponent=0.5)
    scaled = build_model(build_config(hyper_connections=hc))
    unscaled = build_model(build_config(hyper_connections=replace(hc, output_init_exponent=0.0)))

    torch.testing.assert_close(
        scaled.blocks["0"].feed_forward.w2.weight,
        unscaled.blocks["0"].feed_forward.w2.weight * 0.5,
    )
    torch.testing.assert_close(
        scaled.blocks["0"].attention.w_out.weight,
        unscaled.blocks["0"].attention.w_out.weight * 0.5,
    )


def test_added_flops_keep_the_arms_iso_flop():
    """
    Every arm has to be iso-FLOP with the baseline for the comparison to be legal. At the 370M
    shape the paper's Table 8 puts the overhead near 0.2%; anything approaching a percent would
    mean the arms are not matched on compute.
    """
    hc = HyperConnectionConfig(n_lanes=4)
    treated = build_model(build_config(hyper_connections=hc, d_model=1024, n_heads=16, n_layers=1))
    baseline = build_model(build_config(d_model=1024, n_heads=16, n_layers=1))

    assert isinstance(treated.blocks["0"], HyperConnectionReorderedNormTransformerBlock)
    base_flops = baseline.blocks["0"].num_flops_per_token(4096)
    hc_flops = treated.blocks["0"].num_flops_per_token(4096)
    assert hc_flops > base_flops
    assert (hc_flops - base_flops) / base_flops < 0.005


def test_config_reports_the_added_parameters():
    hc = HyperConnectionConfig(n_lanes=4)
    config = build_config(hyper_connections=hc)
    assert not isinstance(config.block, dict)
    baseline_block = build_config().block
    assert not isinstance(baseline_block, dict)

    added = config.block.num_params(D_MODEL) - baseline_block.num_params(D_MODEL)
    assert added == hc.num_params(D_MODEL)
    assert added == 2 * (D_MODEL * (4 + 2) + 4 * (4 + 2) + 2)


def run_fsdp_shards_the_hyper_connection_parameters():
    """
    The hyper-connection parameters are the smallest in the model -- ``B`` is ``(n,)`` and the
    dynamic scales are one element each -- and FSDP shards on dim 0. A four-element tensor
    across four ranks, or a one-element tensor across any of them, is exactly the case that
    breaks, and it would break eleven hours into a run rather than at construction.
    """
    from olmo_core.distributed.parallel import (
        DataParallelConfig,
        DataParallelType,
        build_world_mesh,
    )
    from olmo_core.distributed.utils import get_full_tensor
    from olmo_core.utils import get_default_device

    mesh = build_world_mesh(dp=DataParallelConfig(name=DataParallelType.fsdp))
    config = build_config(hyper_connections=HyperConnectionConfig(n_lanes=4))
    model = config.build(init_device="meta")
    model.apply_fsdp(mesh)
    model.init_weights(max_seq_len=SEQ_LEN, device=get_default_device())

    stream = model.blocks["0"].attention_residual_stream
    torch.testing.assert_close(get_full_tensor(stream.hc_static_alpha_r.detach()), torch.eye(4))
    torch.testing.assert_close(get_full_tensor(stream.hc_static_beta.detach()), torch.ones(4))
    torch.testing.assert_close(
        get_full_tensor(stream.hc_dynamic_scale_beta.detach()), torch.tensor([0.01])
    )

    model(torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))).sum().backward()
    for name, param in model.named_parameters():
        if "hc_" in name:
            assert param.grad is not None, f"no gradient for {name}"


def test_fsdp_shards_the_hyper_connection_parameters():
    from olmo_core.testing import run_distributed_test

    run_distributed_test(
        run_fsdp_shards_the_hyper_connection_parameters,
        backend="gloo",
        start_method="spawn",
        world_size=4,
    )


def test_gradients_reach_every_hyper_connection_parameter():
    hc = HyperConnectionConfig(n_lanes=4)
    model = build_model(build_config(hyper_connections=hc))
    model.train()

    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    model(input_ids).sum().backward()

    stream = model.blocks["1"].attention_residual_stream
    for name, param in stream.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
