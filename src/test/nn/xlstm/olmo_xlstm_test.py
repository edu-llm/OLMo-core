import pytest
import torch
from torch import nn

from olmo_core.nn.transformer import InitMethod
from olmo_core.nn.xlstm import XLSTMMixer, XLSTMMixerConfig
from olmo_core.testing import run_distributed_test

D_MODEL = 64
N_HEADS = 4
BLOCK_IDX = 1
NUM_BLOCKS = 3
SEED = 1234
STD = 0.02


def _mixer_config(**overrides) -> XLSTMMixerConfig:
    options: dict = dict(
        n_heads=N_HEADS,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        conv_size=4,
        input_gate_bias=-10.0,
        forget_gate_bias_min=3.0,
        forget_gate_bias_max=6.0,
    )
    options.update(overrides)
    return XLSTMMixerConfig(**options)


def _build_mixer(**overrides) -> XLSTMMixer:
    return _mixer_config(**overrides).build(D_MODEL, layer_idx=BLOCK_IDX, n_layers=NUM_BLOCKS)


def _initialize(mixer: XLSTMMixer, *, init_method: InitMethod = InitMethod.normal) -> XLSTMMixer:
    mixer.init_weights(
        init_method=init_method,
        d_model=D_MODEL,
        block_idx=BLOCK_IDX,
        num_blocks=NUM_BLOCKS,
        std=STD,
        generator=torch.Generator().manual_seed(SEED),
    )
    return mixer


def _intended_forget_gate_bias(mixer: XLSTMMixer, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(
        mixer.forget_gate_bias_min,
        mixer.forget_gate_bias_max,
        steps=mixer.n_heads,
        dtype=dtype,
    )


@torch.no_grad()
def _initialize_without_routing_the_slice_writes(
    mixer: XLSTMMixer,
    *,
    init_method: InitMethod,
    generator: torch.Generator,
) -> XLSTMMixer:
    """
    Transcription of the initialization as it was written before the slice writes were
    routed through ``_apply_init``.

    Indexing a parameter is only correct while it is an ordinary tensor, so this is the
    single-rank reference and nothing else. It exists so the refactor can be held to
    bit-identity rather than to a distribution.
    """
    from olmo_core.nn.transformer.init import init_linear

    std = STD
    if init_method == InitMethod.normalized:
        std = D_MODEL**-0.5

    for projection in (mixer.w_qk, mixer.w_vo):
        init_linear(projection, std=std, generator=generator)
    mixer.w_vo.weight[mixer.value_dim :].zero_()
    init_linear(mixer.conv1d, std=std, generator=generator)
    mixer.w_if.weight.zero_()
    assert mixer.w_if.bias is not None
    mixer.w_if.bias[: mixer.n_heads].fill_(mixer.input_gate_bias)
    mixer.w_if.bias[mixer.n_heads :].copy_(
        torch.linspace(
            mixer.forget_gate_bias_min,
            mixer.forget_gate_bias_max,
            steps=mixer.n_heads,
            dtype=mixer.w_if.bias.dtype,
            device=mixer.w_if.bias.device,
        )
    )
    mixer.o_norm.weight.fill_(1.0)

    if init_method == InitMethod.llama:
        std = std / (2 * NUM_BLOCKS) ** 0.5
    elif init_method == InitMethod.llama_depth:
        std = std / (2 * (BLOCK_IDX + 1)) ** 0.5
    elif init_method == InitMethod.normalized:
        std = std / (2 * NUM_BLOCKS) ** 0.5
    init_linear(mixer.w_out, std=std, generator=generator)
    return mixer


def _run_sharded_initialization(world_size: int) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, init_device_mesh

    from olmo_core.distributed.utils import get_full_tensor

    # Built inside the rank rather than handed in, so the comparison depends on neither a
    # model built in the parent process nor tensors crossing the process boundary.
    reference = {
        name: parameter.detach().clone()
        for name, parameter in _initialize(_build_mixer()).named_parameters()
    }
    assert reference["w_qk.weight"].std().item() == pytest.approx(STD, rel=0.2)

    mixer = _build_mixer()
    fully_shard(mixer, mesh=init_device_mesh("cpu", (world_size,)))

    # The defect only exists once the parameters the hand-written slice writes address are
    # sharded along the dimension those slices index, which is exactly the state training
    # leaves them in before it initializes anything.
    assert mixer.w_if.bias is not None
    for parameter in (mixer.w_vo.weight, mixer.w_if.bias):
        assert isinstance(parameter, DTensor)
        assert parameter.to_local().shape[0] < parameter.shape[0]

    _initialize(mixer)

    w_vo = get_full_tensor(mixer.w_vo.weight.detach())
    bias = get_full_tensor(mixer.w_if.bias.detach())
    output_gate = w_vo[mixer.value_dim :]

    torch.testing.assert_close(output_gate, torch.zeros_like(output_gate))
    torch.testing.assert_close(
        bias[: mixer.n_heads],
        torch.full((mixer.n_heads,), mixer.input_gate_bias, dtype=bias.dtype),
    )
    torch.testing.assert_close(bias[mixer.n_heads :], _intended_forget_gate_bias(mixer, bias.dtype))
    torch.testing.assert_close(
        get_full_tensor(mixer.w_if.weight.detach()),
        torch.zeros(2 * mixer.n_heads, D_MODEL),
    )
    torch.testing.assert_close(
        get_full_tensor(mixer.o_norm.weight.detach()),
        torch.ones(mixer.n_heads, mixer.head_v_dim),
    )
    # Every draw is made against the whole parameter before a shard is taken out of it, so
    # two ranks owe the same answer as one rather than merely the same distribution.
    for name, parameter in mixer.named_parameters():
        torch.testing.assert_close(
            get_full_tensor(parameter.detach()),
            reference[name],
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_sharded_initialization_zeroes_the_output_gate_and_pins_both_gate_biases():
    # Spawned rather than forked. The xLSTM test modules leave the pytest process
    # multi-threaded, and a second fork out of it deadlocks before the ranks connect.
    run_distributed_test(
        _run_sharded_initialization,
        world_size=2,
        backend="gloo",
        start_method="spawn",
        func_args=(2,),
    )


@pytest.mark.parametrize(
    "init_method",
    [InitMethod.normal, InitMethod.llama, InitMethod.llama_depth, InitMethod.normalized],
)
def test_single_rank_initialization_is_bit_identical_to_the_unrouted_slice_writes(init_method):
    expected = _initialize_without_routing_the_slice_writes(
        _build_mixer(),
        init_method=init_method,
        generator=torch.Generator().manual_seed(SEED),
    )
    actual = _initialize(_build_mixer(), init_method=init_method)

    reference = dict(expected.named_parameters())
    assert set(reference) == set(name for name, _ in actual.named_parameters())
    for name, parameter in actual.named_parameters():
        assert torch.equal(parameter, reference[name]), name


def test_single_rank_initialization_pins_the_gates_it_is_supposed_to_pin():
    mixer = _initialize(_build_mixer())
    assert mixer.w_if.bias is not None

    weight = mixer.w_vo.weight.detach()
    bias = mixer.w_if.bias.detach()
    torch.testing.assert_close(
        weight[mixer.value_dim :], torch.zeros_like(weight[mixer.value_dim :])
    )
    torch.testing.assert_close(
        bias[: mixer.n_heads],
        torch.full((mixer.n_heads,), mixer.input_gate_bias, dtype=bias.dtype),
    )
    torch.testing.assert_close(bias[mixer.n_heads :], _intended_forget_gate_bias(mixer, bias.dtype))
    torch.testing.assert_close(mixer.w_if.weight.detach(), torch.zeros(2 * mixer.n_heads, D_MODEL))
    torch.testing.assert_close(
        mixer.o_norm.weight.detach(), torch.ones(mixer.n_heads, mixer.head_v_dim)
    )
    assert weight[: mixer.value_dim].abs().max().item() > 0.0


def test_initialization_leaves_no_parameter_holding_its_construction_time_value():
    """
    Every parameter has to be written, whatever its shape or where it is sharded.

    A sentinel that no initializer produces makes a discarded write visible as itself
    rather than as a plausible number.
    """
    sentinel = 12345.0
    mixer = _build_mixer()
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.fill_(sentinel)
    _initialize(mixer)

    for name, parameter in mixer.named_parameters():
        assert not (parameter == sentinel).any(), name


def test_initialization_rejects_fan_in():
    with pytest.raises(NotImplementedError, match="fan_in"):
        _initialize(_build_mixer(), init_method=InitMethod.fan_in)


def test_gate_projection_is_a_plain_linear_whose_bias_halves_are_addressed_by_head():
    mixer = _build_mixer()
    assert isinstance(mixer.w_if, nn.Linear)
    assert mixer.w_if.bias is not None
    assert mixer.w_if.bias.shape == (2 * mixer.n_heads,)
    assert mixer.w_vo.weight.shape == (2 * mixer.value_dim, D_MODEL)
