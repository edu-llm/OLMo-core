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


@pytest.mark.parametrize("block_idx", [0, 1, 5])
def test_output_mode_keeps_the_staggered_one_hot_read(block_idx: int):
    """
    Only the *learned* input map goes. The read stays on eq. 14's ``e_{k mod n}``, which is the
    one asymmetry in the whole construction -- see ``HyperConnectionMode.output``.
    """
    full = HyperConnectionStream(
        d_model=D_MODEL, n_lanes=4, block_idx=block_idx, mode=HyperConnectionMode.full
    )
    output_only = HyperConnectionStream(
        d_model=D_MODEL, n_lanes=4, block_idx=block_idx, mode=HyperConnectionMode.output
    )

    expected = torch.zeros(4)
    expected[block_idx % 4] = 1.0
    assert output_only.hc_fixed_alpha_m is not None
    torch.testing.assert_close(output_only.hc_fixed_alpha_m, expected)
    torch.testing.assert_close(full.hc_static_alpha_m.detach(), expected)

    # The read vector is a constant of the config, so it is not a parameter, does not take a
    # gradient, does not reach an optimizer group and is not in the checkpoint.
    assert "hc_fixed_alpha_m" not in dict(output_only.named_parameters())
    assert "hc_fixed_alpha_m" not in output_only.state_dict()

    hidden = torch.randn(2, SEQ_LEN, 4, D_MODEL)
    alpha_m, _, _ = output_only.coefficients(hidden)
    torch.testing.assert_close(alpha_m, expected)


def train_lanes(mode: HyperConnectionMode, *, steps: int = 60, seed: int = 0) -> float:
    """
    Take a small model a few dozen AdamW steps and report the median lane dispersion over its
    blocks, which is the statistic
    :class:`~olmo_core.train.callbacks.HyperConnectionMonitorCallback` guards on.
    """
    from olmo_core.optim import AdamWConfig

    torch.manual_seed(seed)
    hc = HyperConnectionConfig(n_lanes=4, mode=mode, output_init_exponent=0.0)
    model = build_model(build_config(hyper_connections=hc, n_layers=4), seed=seed)
    model.train()
    optim = AdamWConfig(
        lr=1e-2, weight_decay=0.01, group_overrides=hc.optim_group_overrides(weight_decay=0.01)
    ).build(model)

    for _ in range(steps):
        input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
        logits = model(input_ids)
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE), input_ids[:, 1:].reshape(-1)
        ).backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    captured: dict = {}
    handles = [
        block.register_forward_hook(lambda _m, _a, out, key=key: captured.__setitem__(key, out))
        for key, block in model.blocks.items()
    ]
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN)))
    for handle in handles:
        handle.remove()

    dispersions = []
    for lanes in captured.values():
        lanes = lanes.float()
        lane_mean_norm = lanes.mean(dim=-2).norm(dim=-1)
        about_mean = lanes.norm(dim=-1).pow(2).mean(dim=-1) - lane_mean_norm.pow(2)
        dispersions.append(
            (about_mean.clamp_min(0).sqrt() / lane_mean_norm.clamp_min(1e-12)).mean().item()
        )
    return sorted(dispersions)[len(dispersions) // 2]


def test_output_mode_lanes_differentiate_as_far_as_the_faithful_arms_do():
    """
    The arm is only a control if its lanes carry different things, and asserting that two
    parameters are ``None`` does not check that. A uniform-mean read passes every structural
    test in this file and still leaves the model in the permutation-symmetric subspace, where
    an elementwise optimizer keeps it forever: it measured 8e-05 dispersion against `full`'s
    2e-01, with ``A_r``'s diagonal and ``B`` not having moved at all. So this trains.
    """
    output_only = train_lanes(HyperConnectionMode.output)
    faithful = train_lanes(HyperConnectionMode.full)

    assert faithful > 1e-2, f"the faithful arm itself did not separate: {faithful:.3g}"
    assert output_only > faithful / 10, (
        f"output-mode lanes reached {output_only:.3g} against the faithful arm's "
        f"{faithful:.3g}, an order of magnitude short. The arm is degenerate rather than "
        "crippled and would report a null about nothing."
    )


def test_output_mode_leaves_the_mixing_matrices_free_to_move():
    """
    The tell of the symmetric subspace, and the sharper statement than lane dispersion: in it,
    ``A_r``'s diagonal and ``B`` are held equal across lanes by the symmetry itself, so both
    spreads read exactly zero however long the model trains.
    """
    from olmo_core.optim import AdamWConfig

    torch.manual_seed(0)
    hc = HyperConnectionConfig(n_lanes=4, mode=HyperConnectionMode.output, output_init_exponent=0.0)
    model = build_model(build_config(hyper_connections=hc, n_layers=2))
    model.train()
    optim = AdamWConfig(lr=1e-2, group_overrides=hc.optim_group_overrides(0.01)).build(model)

    for _ in range(30):
        input_ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
        model(input_ids).sum().backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    stream = model.blocks["1"].attention_residual_stream
    assert stream.hc_static_alpha_r.detach().diagonal().std() > 1e-4
    assert stream.hc_static_beta.detach().std() > 1e-4


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


@pytest.mark.parametrize("n_lanes", [2, 4, 8])
@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0])
def test_sinkhorn_at_the_shipped_sweep_count_is_column_stochastic_under_drift(
    n_lanes: int, sigma: float
):
    """
    The test above runs 64 sweeps on unit-scale logits, which is neither what ships nor where
    ``A_r`` sits once it has drifted. At the shipped 8 the rows are not normalized in any useful
    sense -- the largest residual here reaches 4.6e-01 at sigma=4 -- so this asserts what is
    actually guaranteed, and the property mHC's argument rests on.
    """
    torch.manual_seed(0)
    iters = HyperConnectionConfig().sinkhorn_iters
    projected = sinkhorn_knopp(torch.randn(256, n_lanes, n_lanes) * sigma, num_iters=iters)

    assert (projected >= 0).all()
    torch.testing.assert_close(
        projected.sum(dim=-2), torch.ones(256, n_lanes), atol=1e-5, rtol=1e-5
    )

    radii = torch.stack([torch.linalg.eigvals(m).abs().max() for m in projected])
    torch.testing.assert_close(radii, torch.ones(256), atol=1e-5, rtol=1e-5)

    # And the rows are not, once the logits are as far apart as a trained ``A_r``'s. At sigma=1
    # eight sweeps do get there, which is why the 64-sweep test above could not see this.
    if sigma >= 2.0:
        assert (projected.sum(dim=-1) - 1).abs().max() > 1e-2


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


def _residual_rms_by_depth(model: torch.nn.Module, input_ids: torch.Tensor) -> dict:
    """
    RMS of the residual stream each block actually reads. The reordered-norm block normalizes
    after its sublayer rather than before, so the sequence mixer receives the stream itself.
    """
    seen: dict = {}
    handles = [
        block.attention.register_forward_pre_hook(
            lambda _mod, args, key=key: seen.__setitem__(
                key, float(args[0].float().pow(2).mean().sqrt())
            )
        )
        for key, block in model.blocks.items()
    ]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def test_output_init_scaling_reaches_the_forward_pass():
    """
    ``test_output_init_scaling_shrinks_the_output_modules`` checks two weight tensors and stops,
    so it would pass on an implementation whose scaling never reached the forward pass. Arm 4
    exists to separate this scaling from the mechanism and only buys something if the scaling
    moves the model, so the size of the move is worth pinning.

    It moves it a long way, and not in the direction the correction is named for. The lane sum
    it compensates for is followed by a scale-invariant final norm, so with the correction off
    the logits are already the baseline's -- which is what
    ``test_init_is_equivalent_to_the_residual_stack_it_replaces`` asserts. What the correction
    does instead is hold the residual stream well below the baseline's through the early blocks.
    """
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    hc = HyperConnectionConfig(n_lanes=4)

    baseline_rms = _residual_rms_by_depth(build_model(build_config()), input_ids)
    unscaled_rms = _residual_rms_by_depth(
        build_model(build_config(hyper_connections=replace(hc, output_init_exponent=0.0))),
        input_ids,
    )
    scaled_rms = _residual_rms_by_depth(
        build_model(build_config(hyper_connections=replace(hc, output_init_exponent=0.5))),
        input_ids,
    )

    # Off, the stream is the baseline's block for block.
    for key, value in unscaled_rms.items():
        assert value == pytest.approx(baseline_rms[key], rel=1e-4), key

    # On, it is not, and the gap is far too large to be read as a scale the final norm absorbs.
    assert scaled_rms["0"] == pytest.approx(baseline_rms["0"], rel=1e-4)
    assert scaled_rms["1"] < 0.75 * baseline_rms["1"]


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


def run_fsdp_shards_the_hyper_connection_parameters(mode: HyperConnectionMode):
    """
    The hyper-connection parameters are the smallest in the model -- ``B`` is ``(n,)`` and the
    dynamic scales are one element each -- and FSDP shards on dim 0. A four-element tensor
    across four ranks, or a one-element tensor across any of them, is exactly the case that
    breaks, and it would break eleven hours into a run rather than at construction.

    The one-hot read is the other half of it: it is position-dependent, so a rank holding rows
    2 and 3 of a sharded copy would get the wrong ones. ``output`` mode holds it as a buffer,
    which FSDP replicates rather than shards, and this is where that gets checked.
    """
    from olmo_core.distributed.parallel import (
        DataParallelConfig,
        DataParallelType,
        build_world_mesh,
    )
    from olmo_core.distributed.utils import get_full_tensor
    from olmo_core.utils import get_default_device

    mesh = build_world_mesh(dp=DataParallelConfig(name=DataParallelType.fsdp))
    config = build_config(hyper_connections=HyperConnectionConfig(n_lanes=4, mode=mode))
    model = config.build(init_device="meta")
    model.apply_fsdp(mesh)
    model.init_weights(max_seq_len=SEQ_LEN, device=get_default_device())

    stream = model.blocks["1"].attention_residual_stream
    torch.testing.assert_close(get_full_tensor(stream.hc_static_alpha_r.detach()), torch.eye(4))
    torch.testing.assert_close(get_full_tensor(stream.hc_static_beta.detach()), torch.ones(4))
    torch.testing.assert_close(
        get_full_tensor(stream.hc_dynamic_scale_beta.detach()), torch.tensor([0.01])
    )

    # Block 1's attention stream is layer k=2 of eq. 14, so it reads lane 2.
    read = torch.tensor([0.0, 0.0, 1.0, 0.0])
    if mode == HyperConnectionMode.full:
        torch.testing.assert_close(get_full_tensor(stream.hc_static_alpha_m.detach()), read)
    else:
        torch.testing.assert_close(get_full_tensor(stream.hc_fixed_alpha_m), read)

    model(torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))).sum().backward()
    for name, param in model.named_parameters():
        if "hc_" in name:
            assert param.grad is not None, f"no gradient for {name}"


@pytest.mark.parametrize("mode", [HyperConnectionMode.full, HyperConnectionMode.output])
def test_fsdp_shards_the_hyper_connection_parameters(mode: HyperConnectionMode):
    from olmo_core.testing import run_distributed_test

    run_distributed_test(
        run_fsdp_shards_the_hyper_connection_parameters,
        backend="gloo",
        start_method="spawn",
        world_size=4,
        func_args=(mode,),
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
