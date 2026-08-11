import shutil
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    InitMethod,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.nn.xlstm import SLSTMMixer, SLSTMMixerConfig, olmo_slstm
from olmo_core.nn.xlstm.olmo_slstm import _FlashRNNPersistentSLSTMLayer
from olmo_core.testing import run_distributed_test

# flashrnn 1.0.6 turns each dtype field of its config into one compile-time pointer type
# (``-DFLASHRNN_DTYPE*``), so the kernel only builds when they all agree.
CUDA_POINTER_TYPE = {
    "float32": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}

# flashrnn 1.0.6 compiles the fused kernel for a fixed batch and rounds every launch up to
# it, discarding the batch it was handed whenever the kernel dtype is float32.
FLOAT32_FUSED_BATCH_SIZE = 16
FUSED_BATCH_MULTIPLE = 8


def round_to_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@dataclass
class RecordedFlashRNNConfig:
    """Stand-in for ``flashrnn.FlashRNNConfig``, with its unset-dtype inheritance."""

    head_dim: int
    num_heads: int
    batch_size: int
    function: str
    backend: str
    input_shape: str
    output_shape: str
    recurrent_shape: str
    bias_shape: str
    dtype: str = "bfloat16"
    dtype_b: str | None = None
    dtype_r: str | None = None
    dtype_w: str | None = None
    dtype_g: str | None = None
    dtype_s: str | None = None
    dtype_a: str | None = None

    def __post_init__(self) -> None:
        # An unset role inherits another role rather than the kernel dtype, so naming only
        # some of them is how a config ends up describing two different kernels at once.
        self.dtype_b = self.dtype_b or self.dtype
        self.dtype_a = self.dtype_a or self.dtype_b
        self.dtype_r = self.dtype_r or self.dtype
        self.dtype_w = self.dtype_w or self.dtype
        self.dtype_s = self.dtype_s or self.dtype_w
        self.dtype_g = self.dtype_g or self.dtype_r
        # A float32 fused kernel is only ever built for batch 16, whatever it was asked for.
        if self.backend == "cuda_fused" and self.dtype == "float32":
            self.batch_size = FLOAT32_FUSED_BATCH_SIZE

    @property
    def launch_batch_size(self) -> int:
        """The batch a launch actually runs at, which only ``cuda_fused`` rounds up.

        THE ROUNDING IS THE REASON THIS ARM MOVED OFF ``cuda_fused``. That generator pads the
        inputs and the initial state up to the multiple with ``torch.ones`` and, on the way
        back, slices only the input and state gradients to the real batch -- the recurrent
        weight and bias gradients are shared and keep the fabricated streams' contribution.
        ``FlashRNNFuncGenerator``, the non-fused one, contains no padding at all, so its launch
        batch is the batch it was handed.
        """
        if self.backend != "cuda_fused":
            return self.batch_size
        if self.dtype == "float32":
            return FLOAT32_FUSED_BATCH_SIZE
        return round_to_multiple(self.batch_size, FUSED_BATCH_MULTIPLE)

    @property
    def role_dtypes(self) -> tuple[str, ...]:
        return (
            self.dtype,
            self.dtype_b,
            self.dtype_r,
            self.dtype_w,
            self.dtype_g,
            self.dtype_s,
            self.dtype_a,
        )

    @property
    def compiled_pointer_types(self) -> set[str]:
        return {CUDA_POINTER_TYPE[dtype] for dtype in self.role_dtypes}


class RecordingFlashRNN:
    """Record what the wrapper hands the persistent kernel, without compiling one."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def __call__(self, gate_inputs, recurrent, bias, **kwargs):
        seq_len, batch_size, num_heads, head_dim, num_gates = gate_inputs.shape
        config = kwargs.get("config")
        launched = gate_inputs
        if config is not None and batch_size % config.launch_batch_size != 0:
            padded_batch_size = round_to_multiple(batch_size, config.launch_batch_size)
            launched = torch.ones(
                (seq_len, padded_batch_size, num_heads, head_dim, num_gates),
                dtype=gate_inputs.dtype,
                device=gate_inputs.device,
            )
            launched[:, :batch_size] = gate_inputs
        self.calls.append(
            SimpleNamespace(gate_inputs=launched, recurrent=recurrent, bias=bias, kwargs=kwargs)
        )
        states = torch.zeros(
            4,
            launched.shape[1],
            num_heads,
            seq_len,
            head_dim,
            dtype=launched.dtype,
            device=launched.device,
        )
        # The padding is the kernel's own business: it is sliced back off before returning.
        states = states[:, :batch_size]
        return states, states[:, :, :, -1:]


def persistent_layer(
    *,
    kernel_dtype: str,
    num_heads: int,
    head_dim: int,
    param_dtype: torch.dtype,
    flashrnn: RecordingFlashRNN,
) -> _FlashRNNPersistentSLSTMLayer:
    layer = _FlashRNNPersistentSLSTMLayer.__new__(_FlashRNNPersistentSLSTMLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(num_heads=num_heads, head_dim=head_dim)
    layer.kernel_dtype = kernel_dtype
    layer._flashrnn = flashrnn
    layer.slstm_cell = nn.Module()
    layer.slstm_cell._recurrent_kernel_ = nn.Parameter(
        torch.randn(num_heads, 4 * head_dim, head_dim, dtype=param_dtype)
    )
    layer.slstm_cell._bias_ = nn.Parameter(torch.randn(4 * num_heads * head_dim, dtype=param_dtype))
    return layer


def unsharded_parameter(values: torch.Tensor) -> tuple[nn.Parameter, torch.Tensor]:
    """One persistent Parameter aliasing a reusable all-gather buffer, as FSDP2 keeps one."""
    buffer = torch.zeros(2 * values.numel(), dtype=values.dtype)
    buffer[: values.numel()] = values.reshape(-1)
    parameter = nn.Parameter(torch.as_strided(buffer, values.shape, values.stride(), 0))
    return parameter, buffer


def all_gather_into(
    parameter: nn.Parameter,
    buffer: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Repoint a persistent unsharded Parameter at freshly gathered values, as FSDP2 does.

    FSDP2 holds the version counter fixed across the swap, so the parameter comes out of an
    all-gather with the same id, version, device and dtype it went in with, and only its
    storage has moved.
    """
    offset = buffer.numel() // 2
    with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(parameter):
        buffer[offset : offset + values.numel()] = values.reshape(-1)
        parameter.set_(torch.as_strided(buffer, values.shape, values.stride(), offset))


def parameter_identity(parameter: nn.Parameter) -> tuple:
    return (id(parameter), parameter._version, parameter.device, parameter.dtype)


@pytest.mark.parametrize("kernel_dtype", ["bfloat16", "float32"])
def test_bfloat16_parameters_compile_one_flashrnn_pointer_type(monkeypatch, kernel_dtype):
    monkeypatch.setattr(olmo_slstm, "_PREFLIGHTED_FLASHRNN_CONFIG", RecordedFlashRNNConfig)
    flashrnn = RecordingFlashRNN()
    num_heads, head_dim, batch_size, seq_len = 4, 256, 2, 8
    layer = persistent_layer(
        kernel_dtype=kernel_dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        param_dtype=torch.bfloat16,
        flashrnn=flashrnn,
    )
    gate_inputs = torch.zeros(seq_len, batch_size, num_heads, head_dim, 4, dtype=torch.bfloat16)

    states, _ = layer._run_flashrnn(gate_inputs, None)

    (call,) = flashrnn.calls
    config = call.kwargs["config"]
    assert config.role_dtypes == (kernel_dtype,) * 7
    assert config.compiled_pointer_types == {CUDA_POINTER_TYPE[kernel_dtype]}
    assert call.kwargs["dtype"] == kernel_dtype
    # The kernel dereferences these as the type it was compiled with, and it is also the
    # config flashrnn infers for itself when the wrapper passes none.
    for tensor in (call.gate_inputs, call.recurrent, call.bias):
        assert tensor.dtype is getattr(torch, kernel_dtype)
    torch.testing.assert_close(
        call.recurrent,
        layer.slstm_cell._recurrent_kernel_.detach()
        .view(num_heads, 4, head_dim, head_dim)
        .permute(0, 2, 1, 3)
        .to(call.recurrent.dtype),
    )
    torch.testing.assert_close(
        call.bias,
        layer.slstm_cell._bias_.detach()
        .view(4, num_heads, head_dim)
        .permute(1, 2, 0)
        .to(call.bias.dtype),
    )
    # NEITHER IS ROUNDED NOW, BECAUSE THE ARM RUNS `FLASHRNN_BACKEND` AND THAT IS NOT THE FUSED
    # KERNEL. Under `cuda_fused` this read `16 if float32 else batch_size` for the compiled
    # batch and `16 if float32 else 8` for the launch, pinning the pad that silently fed six
    # fabricated sequences into the shared recurrent-weight and bias gradients at this arm's
    # batch of 2. The non-fused generator does no padding, so both are the real batch.
    fused = olmo_slstm.FLASHRNN_BACKEND == "cuda_fused"
    expected_compiled_batch_size = 16 if fused and kernel_dtype == "float32" else batch_size
    expected_launch_batch_size = config.launch_batch_size
    assert (config.batch_size, config.num_heads, config.head_dim) == (
        expected_compiled_batch_size,
        num_heads,
        head_dim,
    )
    assert call.gate_inputs.shape[1] == expected_launch_batch_size
    # The padding stays inside the kernel: the caller gets its own batch back.
    assert states.shape == (4, batch_size, num_heads, seq_len, head_dim)
    assert (config.input_shape, config.output_shape) == ("TBHDG", "SBHTD")
    assert (config.recurrent_shape, config.bias_shape) == ("HDGP", "HDG")


@pytest.mark.gpu
def test_bfloat16_flashrnn_kernel_compiles_and_backpropagates():
    pytest.importorskip("flashrnn")
    if not torch.cuda.is_available():
        pytest.skip("the persistent sLSTM kernel needs a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) < (8, 0):
        pytest.skip("the persistent sLSTM kernel needs compute capability >= 8.0")
    if shutil.which("nvcc") is None:
        pytest.skip("FlashRNN JIT-compiles its persistent kernel and needs nvcc")

    num_heads, head_dim, batch_size, seq_len = 1, 64, 2, 8
    olmo_slstm._preflight_flashrnn()
    olmo_slstm._prewarm_flashrnn(
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=num_heads,
        head_dim=head_dim,
        kernel_dtype="bfloat16",
        device=device,
    )
    gate_inputs = torch.randn(
        seq_len,
        batch_size,
        num_heads,
        head_dim,
        4,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    recurrent = torch.randn(
        num_heads, head_dim, 4, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    bias = torch.randn(
        num_heads, head_dim, 4, dtype=torch.bfloat16, device=device, requires_grad=True
    )

    states, _ = olmo_slstm._flashrnn_opaque(
        olmo_slstm._PREFLIGHTED_FLASHRNN,
        gate_inputs,
        recurrent,
        bias,
        None,
        "bfloat16",
        olmo_slstm._PREFLIGHTED_FLASHRNN_CONFIG,
    )

    assert states.shape == (4, batch_size, num_heads, seq_len, head_dim)
    states[0].float().square().sum().backward()
    for tensor in (gate_inputs, recurrent, bias):
        assert tensor.grad is not None
        assert tensor.grad.dtype is torch.bfloat16
        assert torch.isfinite(tensor.grad.float()).all()


def test_flashrnn_parameters_convert_vanilla_storage_with_gradients():
    num_heads, head_dim = 2, 3
    recurrent_storage = torch.arange(
        num_heads * 4 * head_dim * head_dim,
        dtype=torch.float64,
    ).reshape(num_heads, 4 * head_dim, head_dim)
    bias_storage = torch.arange(
        101,
        101 + 4 * num_heads * head_dim,
        dtype=torch.float64,
    )

    layer = _FlashRNNPersistentSLSTMLayer.__new__(_FlashRNNPersistentSLSTMLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(num_heads=num_heads, head_dim=head_dim)
    layer.slstm_cell = nn.Module()
    layer.slstm_cell._recurrent_kernel_ = nn.Parameter(recurrent_storage.clone())
    layer.slstm_cell._bias_ = nn.Parameter(bias_storage.clone())

    recurrent, bias = layer._flashrnn_parameters()

    assert recurrent.shape == (num_heads, head_dim, 4, head_dim)
    assert bias.shape == (num_heads, head_dim, 4)
    torch.testing.assert_close(
        recurrent,
        recurrent_storage.view(num_heads, 4, head_dim, head_dim).permute(0, 2, 1, 3),
    )
    torch.testing.assert_close(
        bias,
        bias_storage.view(4, num_heads, head_dim).permute(1, 2, 0),
    )

    # Nothing is carried between calls, so converting again yields a separate tensor with
    # the same values rather than a buffer whose freshness some key has to vouch for.
    repeated_recurrent, repeated_bias = layer._flashrnn_parameters()
    assert repeated_recurrent.data_ptr() != recurrent.data_ptr()
    assert repeated_bias.data_ptr() != bias.data_ptr()
    torch.testing.assert_close(repeated_recurrent, recurrent)
    torch.testing.assert_close(repeated_bias, bias)

    recurrent_native_grad = torch.arange(
        1,
        1 + recurrent.numel(),
        dtype=recurrent.dtype,
    ).reshape(recurrent.shape)
    bias_native_grad = torch.arange(
        211,
        211 + bias.numel(),
        dtype=bias.dtype,
    ).reshape(bias.shape)
    torch.autograd.backward(
        (recurrent, bias),
        (recurrent_native_grad, bias_native_grad),
    )

    torch.testing.assert_close(
        layer.slstm_cell._recurrent_kernel_.grad,
        recurrent_native_grad.permute(0, 2, 1, 3).reshape(
            num_heads,
            4 * head_dim,
            head_dim,
        ),
    )
    torch.testing.assert_close(
        layer.slstm_cell._bias_.grad,
        bias_native_grad.permute(2, 0, 1).reshape(4 * num_heads * head_dim),
    )

    with torch.no_grad():
        layer.slstm_cell._recurrent_kernel_.add_(1000)
        layer.slstm_cell._bias_.add_(2000)
    updated_recurrent, updated_bias = layer._flashrnn_parameters()
    assert updated_recurrent.data_ptr() != recurrent.data_ptr()
    assert updated_bias.data_ptr() != bias.data_ptr()
    torch.testing.assert_close(
        updated_recurrent,
        layer.slstm_cell._recurrent_kernel_.view(num_heads, 4, head_dim, head_dim).permute(
            0, 2, 1, 3
        ),
    )
    torch.testing.assert_close(
        updated_bias,
        layer.slstm_cell._bias_.view(4, num_heads, head_dim).permute(1, 2, 0),
    )

    replacement_recurrent = nn.Parameter(recurrent_storage.clone().add_(3000))
    replacement_bias = nn.Parameter(bias_storage.clone().add_(4000))
    layer.slstm_cell._recurrent_kernel_ = replacement_recurrent
    layer.slstm_cell._bias_ = replacement_bias
    replaced_recurrent, replaced_bias = layer._flashrnn_parameters()
    assert replaced_recurrent.data_ptr() != updated_recurrent.data_ptr()
    assert replaced_bias.data_ptr() != updated_bias.data_ptr()
    torch.testing.assert_close(
        replaced_recurrent,
        replacement_recurrent.view(num_heads, 4, head_dim, head_dim).permute(0, 2, 1, 3),
    )
    torch.testing.assert_close(
        replaced_bias,
        replacement_bias.view(4, num_heads, head_dim).permute(1, 2, 0),
    )


def test_flashrnn_parameters_track_fsdp_rematerialized_all_gather_buffer(monkeypatch):
    monkeypatch.setattr(olmo_slstm, "_PREFLIGHTED_FLASHRNN_CONFIG", RecordedFlashRNNConfig)
    num_heads, head_dim, batch_size, seq_len = 2, 8, 16, 3
    flashrnn = RecordingFlashRNN()
    layer = persistent_layer(
        kernel_dtype="float32",
        num_heads=num_heads,
        head_dim=head_dim,
        param_dtype=torch.float32,
        flashrnn=flashrnn,
    )

    first_recurrent = torch.randn(num_heads, 4 * head_dim, head_dim)
    first_bias = torch.randn(4 * num_heads * head_dim)
    recurrent_parameter, recurrent_buffer = unsharded_parameter(first_recurrent)
    bias_parameter, bias_buffer = unsharded_parameter(first_bias)
    layer.slstm_cell._recurrent_kernel_ = recurrent_parameter
    layer.slstm_cell._bias_ = bias_parameter

    gate_inputs = torch.zeros(seq_len, batch_size, num_heads, head_dim, 4)
    layer._run_flashrnn(gate_inputs, None)

    identities = (parameter_identity(recurrent_parameter), parameter_identity(bias_parameter))
    storages = (recurrent_parameter.data_ptr(), bias_parameter.data_ptr())

    second_recurrent = first_recurrent + 1000.0
    second_bias = first_bias - 500.0
    all_gather_into(recurrent_parameter, recurrent_buffer, second_recurrent)
    all_gather_into(bias_parameter, bias_buffer, second_bias)

    # The premise of the defect: an all-gather moves the storage and leaves every piece of
    # parameter metadata a cache could key on exactly as it was.
    assert (
        parameter_identity(recurrent_parameter),
        parameter_identity(bias_parameter),
    ) == identities
    assert (recurrent_parameter.data_ptr(), bias_parameter.data_ptr()) != storages

    expected_recurrent = second_recurrent.view(num_heads, 4, head_dim, head_dim).permute(0, 2, 1, 3)
    expected_bias = second_bias.view(4, num_heads, head_dim).permute(1, 2, 0)

    recurrent, bias = layer._flashrnn_parameters()
    torch.testing.assert_close(recurrent, expected_recurrent)
    torch.testing.assert_close(bias, expected_bias)

    layer._run_flashrnn(gate_inputs, None)
    call = flashrnn.calls[-1]
    torch.testing.assert_close(call.recurrent, expected_recurrent)
    torch.testing.assert_close(call.bias, expected_bias)

    recurrent_native_grad = torch.randn_like(recurrent)
    bias_native_grad = torch.randn_like(bias)
    torch.autograd.backward((recurrent, bias), (recurrent_native_grad, bias_native_grad))

    torch.testing.assert_close(
        recurrent_parameter.grad,
        recurrent_native_grad.permute(0, 2, 1, 3).reshape(num_heads, 4 * head_dim, head_dim),
    )
    torch.testing.assert_close(
        bias_parameter.grad,
        bias_native_grad.permute(2, 0, 1).reshape(4 * num_heads * head_dim),
    )


MIXER_D_MODEL = 64
MIXER_N_HEADS = 4
MIXER_LAYER_IDX = 0
MIXER_N_LAYERS = 2
MIXER_SEED = 1234
MIXER_SENTINEL = 12345.0

# Every layer flavour `_build_slstm_layer` can produce without a GPU. Production runs the
# first of them.
MIXER_FLAVOURS = [("cuda_fused", True), ("vanilla", True), ("vanilla", False)]


def slstm_mixer(*, backend: str = "cuda_fused", fused: bool = True) -> SLSTMMixer:
    """Build the production sLSTM mixer shape on CPU."""
    mixer = SLSTMMixerConfig(
        n_heads=MIXER_N_HEADS,
        conv_size=4,
        backend=backend,
        batch_size=2,
        kernel_dtype="bfloat16",
        fuse_input_projections=fused,
    ).build(
        MIXER_D_MODEL,
        layer_idx=MIXER_LAYER_IDX,
        n_layers=MIXER_N_LAYERS,
        init_device="cpu",
    )
    # The mixer's own `init_weights` never reaches the convolution -- upstream leaves it
    # with whatever `nn.Conv1d` built -- so stamping it is how a test can tell a preserved
    # parameter from an overwritten one.
    with torch.no_grad():
        for parameter in (mixer.layer.conv1d.conv.weight, mixer.layer.conv1d.conv.bias):
            parameter.copy_(
                torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            )
    return mixer


def initialize_slstm_mixer(mixer: SLSTMMixer) -> SLSTMMixer:
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=MIXER_D_MODEL,
        block_idx=MIXER_LAYER_IDX,
        num_blocks=MIXER_N_LAYERS,
        generator=torch.Generator().manual_seed(MIXER_SEED),
    )
    return mixer


def _run_sharded_slstm_initialization(world_size: int) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, init_device_mesh

    from olmo_core.distributed.utils import get_full_tensor

    # Built inside the rank rather than handed in, so the comparison depends on neither a
    # model built in the parent process nor tensors crossing the process boundary.
    reference = {
        name: parameter.detach().clone()
        for name, parameter in initialize_slstm_mixer(slstm_mixer()).named_parameters()
    }
    # Every draw is made against the whole parameter, so two ranks owe the same answer as
    # one rather than merely the same distribution.
    assert reference["layer.conv_gate_weight"].std().item() > 0.0

    mixer = slstm_mixer()
    fully_shard(mixer, mesh=init_device_mesh("cpu", (world_size,)))

    # The defect only exists once the parameters the per-head writes address are sharded on
    # the dimension those writes index, which is the state training leaves them in before it
    # initializes anything.
    local_shapes = {}
    for name, parameter in mixer.named_parameters():
        assert isinstance(parameter, DTensor), name
        assert parameter.to_local().shape[0] < parameter.shape[0], name
        local_shapes[name] = parameter.to_local().shape

    initialize_slstm_mixer(mixer)

    for name, parameter in mixer.named_parameters():
        # The cell writes through a proxy that finishes by reassigning `.data`, so a
        # parameter that stopped being its own shard is its own failure.
        assert isinstance(parameter, DTensor), name
        assert parameter.to_local().shape == local_shapes[name], name
        torch.testing.assert_close(
            get_full_tensor(parameter.detach()),
            reference[name],
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_sharded_initialization_matches_one_rank_parameter_for_parameter():
    pytest.importorskip("xlstm")
    # Spawned rather than forked. The xLSTM test modules leave the pytest process
    # multi-threaded, and a second fork out of it deadlocks before the ranks connect.
    run_distributed_test(
        _run_sharded_slstm_initialization,
        world_size=2,
        backend="gloo",
        start_method="spawn",
        func_args=(2,),
    )


def test_single_rank_initialization_is_bit_identical_to_the_unrouted_reset():
    pytest.importorskip("xlstm")
    expected = slstm_mixer()
    with torch.no_grad():
        olmo_slstm._reset_parameters_with_generator(
            expected.layer, torch.Generator().manual_seed(MIXER_SEED)
        )
    actual = initialize_slstm_mixer(slstm_mixer())

    reference = dict(expected.named_parameters())
    assert set(reference) == {name for name, _ in actual.named_parameters()}
    for name, parameter in actual.named_parameters():
        assert torch.equal(parameter, reference[name]), name


def test_initialization_writes_the_cell_and_gates_and_leaves_the_convolution_alone():
    pytest.importorskip("xlstm")
    mixer = slstm_mixer()
    convolution = {
        "layer.conv1d.conv.weight": mixer.layer.conv1d.conv.weight.detach().clone(),
        "layer.conv1d.conv.bias": mixer.layer.conv1d.conv.bias.detach().clone(),
    }
    with torch.no_grad():
        for name, parameter in mixer.named_parameters():
            if name not in convolution:
                parameter.fill_(MIXER_SENTINEL)

    initialize_slstm_mixer(mixer)

    for name, parameter in mixer.named_parameters():
        if name in convolution:
            # `init_weights` alone does not reach the convolution; the bare sweep in
            # `Transformer.init_weights` is what writes it. Rewriting it here would be a
            # change of numerics wearing a bug fix's clothes.
            assert torch.equal(parameter, convolution[name]), name
        else:
            assert not (parameter == MIXER_SENTINEL).any(), name


def bare_reset_parameters_sweep(module: nn.Module) -> None:
    """Verbatim transcription of the sweep in ``Transformer.init_weights``."""
    for submodule in module.modules():
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()  # type: ignore[operator]


def surviving_sentinel(mixer: SLSTMMixer) -> dict[str, int]:
    """Count, per parameter, the elements no initializer ever wrote."""
    from olmo_core.distributed.utils import get_full_tensor

    counts = {}
    for name, parameter in mixer.named_parameters():
        full = get_full_tensor(parameter.detach())
        left = int((full == MIXER_SENTINEL).sum().item())
        if left:
            counts[name] = left
    return counts


def transformer_init_sequence(mixer: SLSTMMixer) -> SLSTMMixer:
    """
    Reproduce what ``Transformer.init_weights`` does to a mixer, in its order.

    ``to_empty`` first, then the bare ``reset_parameters`` sweep over every module, and
    only then the mixer's own ``init_weights``. The sentinel stands in for the
    uninitialized memory ``to_empty`` hands back, because uninitialized memory that is
    never written is otherwise indistinguishable from a plausible number.
    """
    # `Transformer.init_weights` carries `@torch.no_grad()`, and whether the parameters are
    # leaves that require grad decides which in-place writes are even legal.
    with torch.no_grad():
        mixer.to_empty(device=torch.device("cpu"))
        for parameter in mixer.parameters():
            parameter.fill_(MIXER_SENTINEL)
        bare_reset_parameters_sweep(mixer)
    return initialize_slstm_mixer(mixer)


def _run_sharded_transformer_init_sequence(world_size: int, backend: str, fused: bool) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, init_device_mesh

    from olmo_core.distributed.utils import get_full_tensor

    # The bare sweep draws from the ambient generator rather than a seeded one, so the two
    # runs are only comparable from the same ambient seed.
    torch.manual_seed(MIXER_SEED)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in transformer_init_sequence(
            slstm_mixer(backend=backend, fused=fused)
        ).named_parameters()
    }

    torch.manual_seed(MIXER_SEED)
    mixer = slstm_mixer(backend=backend, fused=fused)
    fully_shard(mixer, mesh=init_device_mesh("cpu", (world_size,)))
    for name, parameter in mixer.named_parameters():
        assert isinstance(parameter, DTensor), name
        assert parameter.to_local().shape[0] < parameter.shape[0], name

    transformer_init_sequence(mixer)

    assert surviving_sentinel(mixer) == {}
    for name, parameter in mixer.named_parameters():
        torch.testing.assert_close(
            get_full_tensor(parameter.detach()),
            reference[name],
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize(("backend", "fused"), MIXER_FLAVOURS)
def test_sharded_transformer_init_sequence_writes_every_parameter(backend, fused):
    pytest.importorskip("xlstm")
    # Spawned rather than forked. The xLSTM test modules leave the pytest process
    # multi-threaded, and a second fork out of it deadlocks before the ranks connect.
    run_distributed_test(
        _run_sharded_transformer_init_sequence,
        world_size=2,
        backend="gloo",
        start_method="spawn",
        func_args=(2, backend, fused),
    )


@pytest.mark.parametrize(("backend", "fused"), MIXER_FLAVOURS)
def test_single_rank_transformer_init_sequence_writes_every_parameter(backend, fused):
    pytest.importorskip("xlstm")
    mixer = transformer_init_sequence(slstm_mixer(backend=backend, fused=fused))
    assert surviving_sentinel(mixer) == {}


def slstm_transformer_config() -> TransformerConfig:
    """The smallest whole model carrying the production sLSTM mixer."""
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    return TransformerConfig(
        d_model=MIXER_D_MODEL,
        vocab_size=64,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=SLSTMMixerConfig(
                n_heads=MIXER_N_HEADS,
                conv_size=4,
                backend="cuda_fused",
                batch_size=2,
                kernel_dtype="bfloat16",
                fuse_input_projections=True,
            ),
            layer_norm=norm,
            feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False),
    )


def _run_sharded_transformer_init_weights(world_size: int) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, init_device_mesh

    from olmo_core.distributed.utils import get_full_tensor

    # Meta construction, then shard, then `init_weights` -- the order production uses, and
    # the reason nothing the constructor writes survives to be initialized.
    mesh = init_device_mesh("cpu", (world_size,))
    model = slstm_transformer_config().build(init_device="meta")
    for block in model.blocks.values():
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)

    model.init_weights(device=torch.device("cpu"))

    mixer = model.blocks["0"].attention
    assert isinstance(mixer, SLSTMMixer)
    for name, parameter in mixer.named_parameters():
        assert isinstance(parameter, DTensor), name
        assert torch.isfinite(get_full_tensor(parameter.detach())).all(), name


def test_sharded_real_transformer_init_weights_reaches_the_slstm_mixer():
    pytest.importorskip("xlstm")
    # The transcription above is only worth what its fidelity is worth, so this one calls
    # the function itself rather than a copy of it.
    run_distributed_test(
        _run_sharded_transformer_init_weights,
        world_size=2,
        backend="gloo",
        start_method="spawn",
        func_args=(2,),
    )
