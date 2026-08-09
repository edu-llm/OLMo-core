"""
sLSTM sequence mixer for the paper-style xLSTM[7:1] hybrid.

The implementation wraps the Apache-2.0 ``xlstm==2.0.5`` sLSTM layer while
adapting it to OLMo-core's :class:`SequenceMixer` lifecycle. The optional
``cuda_fused`` path uses FlashRNN's persistent kernel and fails closed when
the dependency or required CUDA capabilities are unavailable.
"""

import importlib
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import metadata
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


FLASHRNN_VERSION = "1.0.6"
_PREFLIGHTED_FLASHRNN = None
_PREFLIGHTED_FLASHRNN_CONFIG = None
_PREWARMED_FLASHRNN_SHAPES = set()


def _reject_optimizer_state(optimizer_state_dict: Mapping[str, Any] | None) -> None:
    if optimizer_state_dict is not None:
        raise NotImplementedError(
            "optimizer-state conversion is unsupported because projection packing changes "
            "parameter identity and cardinality; convert model state only and initialize a "
            "new optimizer"
        )


def _sorted_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: state_dict[key] for key in sorted(state_dict)}


def _reset_parameters_with_generator(
    module: nn.Module,
    generator: torch.Generator | None,
) -> None:
    if generator is None:
        module.reset_parameters()  # type: ignore[attr-defined]
        return

    device = torch.device(generator.device)
    generator_state = generator.get_state()
    if device.type == "cpu":
        ambient_state = torch.random.get_rng_state()
        try:
            torch.random.set_rng_state(generator_state)
            module.reset_parameters()  # type: ignore[attr-defined]
            generator.set_state(torch.random.get_rng_state())
        finally:
            torch.random.set_rng_state(ambient_state)
    elif device.type == "cuda":
        ambient_state = torch.cuda.get_rng_state(device)
        try:
            torch.cuda.set_rng_state(generator_state, device)
            module.reset_parameters()  # type: ignore[attr-defined]
            generator.set_state(torch.cuda.get_rng_state(device))
        finally:
            torch.cuda.set_rng_state(ambient_state, device)
    else:
        raise NotImplementedError(
            f"sLSTM initialization with a generator on '{device.type}' is not supported"
        )


@torch.no_grad()
def _reset_parameters_on_unsharded_copies(
    module: nn.Module,
    reset: Callable[[], None],
) -> None:
    """
    Run ``reset`` against unsharded copies of ``module``'s parameters, then copy each
    rank's own shard back.

    The cell initializes one head and one gate at a time, and its ``ParameterProxy``
    finishes every such write by reassigning ``.data``. Once FSDP2 has made those
    parameters sharded :class:`~torch.distributed.tensor.DTensor`\\ s, indexing one yields
    a redistributed copy rather than a view and reassigning ``.data`` would put a whole
    tensor where a shard belongs, so the initialization either lands nowhere or raises.

    This is :func:`olmo_core.nn.transformer.init._apply_init` widened from one tensor to a
    module: the unmodified upstream code runs against full-size local tensors seeded with
    the values the parameters already hold, and then every rank copies its own shard in.
    Because every rank draws the whole tensor, the result matches a single-rank run
    element for element rather than varying with the number of shards.

    Nothing is staged when no parameter is sharded, so single-rank numerics are untouched
    and nesting one of these inside another costs nothing.

    Grad is off throughout. A parameter's shard is reached through ``to_local``, whose
    output autograd treats as a view of a custom function's, and writing into that is
    forbidden while grad is on. Initialization has no business being recorded anyway, and
    doing it here rather than relying on the caller keeps one rank and many alike.

    :param module: The module whose parameters ``reset`` writes.
    :param reset: The initialization to run. Takes no arguments.
    """
    if not any(isinstance(parameter, DTensor) for parameter in module.parameters()):
        reset()
        return

    from olmo_core.distributed.utils import (
        distribute_like,
        get_full_tensor,
        get_local_tensor,
    )

    owners = [
        (submodule, name, parameter)
        for submodule in module.modules()
        for name, parameter in submodule.named_parameters(recurse=False)
    ]
    unsharded = [
        nn.Parameter(get_full_tensor(parameter.detach()).clone(), requires_grad=False)
        for _, _, parameter in owners
    ]
    # Swapping the registrations rather than the parameter objects leaves every reference
    # FSDP holds pointing at the same objects it sharded.
    for (submodule, name, _), replacement in zip(owners, unsharded):
        submodule._parameters[name] = replacement
    try:
        reset()
    finally:
        for submodule, name, parameter in owners:
            submodule._parameters[name] = parameter
    for (_, _, parameter), replacement in zip(owners, unsharded):
        get_local_tensor(parameter).copy_(
            get_local_tensor(distribute_like(parameter, replacement.detach()))
        )


class _ShardingSafeReset:
    """
    A module's own ``reset_parameters``, staged onto unsharded copies before it writes.

    ``Transformer.init_weights`` sweeps ``reset_parameters`` over every module it can
    reach and only afterwards dispatches to each mixer's ``init_weights``, so every one of
    these has to survive being called on its own rather than only when reached through
    :meth:`SLSTMMixer.init_weights`. That includes the upstream ``sLSTMCell``, which the
    sweep reaches directly, so the staging is installed per instance: subclassing the
    ``xlstm`` modules would move the checkpoint keys they own.

    :param module: The module the reset belongs to.
    :param reset: The unbound ``reset_parameters`` it would otherwise have run.
    """

    def __init__(self, module: nn.Module, reset: Callable[..., None]):
        self.module = module
        self.reset = reset

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        _reset_parameters_on_unsharded_copies(
            self.module,
            lambda: self.reset(self.module, *args, **kwargs),
        )


def _make_reset_parameters_sharding_safe(module: nn.Module) -> None:
    """
    Route ``reset_parameters`` on ``module`` and every submodule through unsharded staging.

    :param module: Root of the subtree to make safe.
    """
    for submodule in module.modules():
        reset = getattr(type(submodule), "reset_parameters", None)
        if reset is None:
            continue
        submodule.reset_parameters = _ShardingSafeReset(submodule, reset)  # type: ignore[assignment]


def convert_slstm_official_to_packed_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack official sLSTM gate projections into the OLMo/FlashRNN model layout."""
    _reject_optimizer_state(optimizer_state_dict)
    marker = "layer.fgate.weight"
    prefixes = sorted({key[: -len(marker)] for key in state_dict if key.endswith(marker)})
    if not prefixes:
        packed = sorted(
            key
            for key in state_dict
            if key.endswith(("layer.conv_gate_weight", "layer.raw_gate_weight"))
        )
        if packed:
            raise ValueError(
                f"expected official sLSTM checkpoint layout, found packed keys: {packed}"
            )
        raise ValueError("no official sLSTM projection layout was found")
    converted = dict(state_dict)
    for prefix in prefixes:
        official_suffixes = (
            "layer.fgate.weight",
            "layer.igate.weight",
            "layer.zgate.weight",
            "layer.ogate.weight",
        )
        required = {f"{prefix}{suffix}" for suffix in official_suffixes}
        missing = sorted(required.difference(converted))
        if missing:
            raise ValueError(f"incomplete official sLSTM layout at '{prefix}': missing {missing}")
        packed_keys = {
            f"{prefix}layer.conv_gate_weight",
            f"{prefix}layer.raw_gate_weight",
        }
        collisions = sorted(packed_keys.intersection(converted))
        if collisions:
            raise ValueError(f"mixed sLSTM checkpoint layouts at '{prefix}': found {collisions}")
        updates = {
            f"{prefix}layer.conv_gate_weight": torch.stack(
                (
                    converted[f"{prefix}layer.fgate.weight"],
                    converted[f"{prefix}layer.igate.weight"],
                ),
                dim=1,
            ),
            f"{prefix}layer.raw_gate_weight": torch.stack(
                (
                    converted[f"{prefix}layer.zgate.weight"],
                    converted[f"{prefix}layer.ogate.weight"],
                ),
                dim=1,
            ),
        }
        for key in required:
            del converted[key]
        converted.update(updates)
    return _sorted_state_dict(converted)


def convert_slstm_packed_to_official_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Unpack OLMo/FlashRNN sLSTM gate projections into official model keys."""
    _reject_optimizer_state(optimizer_state_dict)
    marker = "layer.conv_gate_weight"
    prefixes = sorted({key[: -len(marker)] for key in state_dict if key.endswith(marker)})
    if not prefixes:
        official = sorted(
            key
            for key in state_dict
            if key.endswith(
                (
                    "layer.fgate.weight",
                    "layer.igate.weight",
                    "layer.zgate.weight",
                    "layer.ogate.weight",
                )
            )
        )
        if official:
            raise ValueError(
                f"expected packed sLSTM checkpoint layout, found official keys: {official}"
            )
        raise ValueError("no packed sLSTM projection layout was found")
    converted = dict(state_dict)
    for prefix in prefixes:
        packed_keys = {
            f"{prefix}layer.conv_gate_weight",
            f"{prefix}layer.raw_gate_weight",
        }
        missing = sorted(packed_keys.difference(converted))
        if missing:
            raise ValueError(f"incomplete packed sLSTM layout at '{prefix}': missing {missing}")
        official_keys = {
            f"{prefix}layer.fgate.weight",
            f"{prefix}layer.igate.weight",
            f"{prefix}layer.zgate.weight",
            f"{prefix}layer.ogate.weight",
        }
        collisions = sorted(official_keys.intersection(converted))
        if collisions:
            raise ValueError(f"mixed sLSTM checkpoint layouts at '{prefix}': found {collisions}")
        conv_gates = converted[f"{prefix}layer.conv_gate_weight"]
        raw_gates = converted[f"{prefix}layer.raw_gate_weight"]
        if conv_gates.ndim < 2 or conv_gates.shape[1] != 2:
            raise ValueError(
                f"packed sLSTM tensor '{prefix}layer.conv_gate_weight' must have gate dimension 2"
            )
        if raw_gates.ndim < 2 or raw_gates.shape[1] != 2:
            raise ValueError(
                f"packed sLSTM tensor '{prefix}layer.raw_gate_weight' must have gate dimension 2"
            )
        updates = {
            f"{prefix}layer.fgate.weight": conv_gates[:, 0],
            f"{prefix}layer.igate.weight": conv_gates[:, 1],
            f"{prefix}layer.zgate.weight": raw_gates[:, 0],
            f"{prefix}layer.ogate.weight": raw_gates[:, 1],
        }
        for key in packed_keys:
            del converted[key]
        converted.update(updates)
    return _sorted_state_dict(converted)


class _FusedInputSLSTMLayer(nn.Module):
    """Official sLSTM layer with its four head-wise input projections fused into two."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.config = layer.config
        self.conv1d = layer.conv1d
        self.conv_act_fn = layer.conv_act_fn
        self.slstm_cell = layer.slstm_cell
        self.group_norm = layer.group_norm
        self.dropout = layer.dropout

        # The first pair consumes convolved input; the second consumes the original input. Keep
        # those as two contractions while folding each pair's launch and input read together.
        self.conv_gate_weight = nn.Parameter(
            torch.stack((layer.fgate.weight, layer.igate.weight), dim=1)
        )
        self.raw_gate_weight = nn.Parameter(
            torch.stack((layer.zgate.weight, layer.ogate.weight), dim=1)
        )

    def _project_pair(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        head_input = x.view(*shape[:-1], self.config.num_heads, -1)
        pair = torch.einsum("...hi,hgoi->...hgo", head_input, weight)
        first = pair[..., 0, :].reshape(*shape)
        second = pair[..., 1, :].reshape(*shape)
        return first, second

    def _project_gates(
        self,
        x: torch.Tensor,
        x_conv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i, f = self._project_pair(x_conv, self.conv_gate_weight)
        z, o = self._project_pair(x, self.raw_gate_weight)
        return i, f, z, o

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        slstm_state: torch.Tensor | None = None,
        return_last_state: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = x.shape
        if return_last_state:
            x_conv, conv_state = self.conv1d(
                x,
                conv_state,
                return_last_state=True,
            )
        else:
            x_conv = self.conv1d(x, conv_state, return_last_state=False)
        x_conv = self.conv_act_fn(x_conv)

        i, f, z, o = self._project_gates(x, x_conv)
        y, slstm_state = self.slstm_cell(
            torch.cat((i, f, z, o), dim=-1),
            state=slstm_state,
        )
        y = self.dropout(y)
        output = self.group_norm(y).transpose(1, 2).reshape(batch_size, seq_len, -1)
        if return_last_state:
            return output, {
                "conv_state": conv_state,
                "slstm_state": slstm_state,
            }
        return output

    def reset_parameters(self) -> None:
        from xlstm.components.init import small_init_init_

        self.slstm_cell.reset_parameters()
        self.group_norm.reset_parameters()
        small_init_init_(self.conv_gate_weight, dim=self.config.embedding_dim)
        small_init_init_(self.raw_gate_weight, dim=self.config.embedding_dim)


def _load_flashrnn():
    global _PREFLIGHTED_FLASHRNN_CONFIG
    try:
        installed_version = metadata.version("flashrnn")
        module = importlib.import_module("flashrnn")
        flashrnn = module.flashrnn
        _PREFLIGHTED_FLASHRNN_CONFIG = module.FlashRNNConfig
    except (AttributeError, ImportError, metadata.PackageNotFoundError) as exc:
        raise ImportError(
            f"the sLSTM cuda_fused backend requires flashrnn=={FLASHRNN_VERSION}"
        ) from exc
    if installed_version != FLASHRNN_VERSION:
        raise ImportError(
            "the sLSTM cuda_fused backend requires "
            f"flashrnn=={FLASHRNN_VERSION}, found {installed_version}"
        )
    return flashrnn


def _preflight_flashrnn():
    """Resolve and cache the exact FlashRNN callable before model construction."""
    global _PREFLIGHTED_FLASHRNN
    _PREFLIGHTED_FLASHRNN = _load_flashrnn()
    return _PREFLIGHTED_FLASHRNN


def _prewarm_flashrnn(
    *,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    kernel_dtype: str,
    device: torch.device,
) -> str:
    """JIT the exact FlashRNN production shape once and return its cache identity."""
    if _PREFLIGHTED_FLASHRNN is None:
        raise RuntimeError("FlashRNN must be preflighted before exact-shape prewarm")
    try:
        dtype = getattr(torch, kernel_dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported FlashRNN kernel dtype '{kernel_dtype}'") from exc
    capability = torch.cuda.get_device_capability(device)
    torch_version = torch.__version__
    cuda_build = torch.version.cuda or "none"
    sm = f"sm_{capability[0]}{capability[1]}"
    cache_key = (
        FLASHRNN_VERSION,
        torch_version,
        cuda_build,
        capability,
        device.type,
        device.index,
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        kernel_dtype,
    )
    identity = (
        f"flashrnn=={FLASHRNN_VERSION}:slstm:cuda_fused:torch={torch_version}:"
        f"cuda={cuda_build}:{sm}:dtype={kernel_dtype}:B{batch_size}:T{seq_len}:"
        f"H{n_heads}:D{head_dim}"
    )
    if cache_key in _PREWARMED_FLASHRNN_SHAPES:
        return identity

    gate_inputs = torch.zeros(
        seq_len,
        batch_size,
        n_heads,
        head_dim,
        4,
        dtype=dtype,
        device=device,
    )
    recurrent = torch.zeros(
        n_heads,
        head_dim,
        4,
        head_dim,
        dtype=dtype,
        device=device,
    )
    bias = torch.zeros(n_heads, head_dim, 4, dtype=dtype, device=device)
    with torch.no_grad():
        states, _ = _flashrnn_opaque(
            _PREFLIGHTED_FLASHRNN,
            gate_inputs,
            recurrent,
            bias,
            None,
            kernel_dtype,
            _PREFLIGHTED_FLASHRNN_CONFIG,
        )
    expected_shape = (4, batch_size, n_heads, seq_len, head_dim)
    if states.shape != expected_shape:
        raise RuntimeError(
            f"FlashRNN exact-shape prewarm returned {tuple(states.shape)}, "
            f"expected {expected_shape}"
        )
    _PREWARMED_FLASHRNN_SHAPES.add(cache_key)
    return identity


def _validate_flashrnn_device(x: torch.Tensor, *, head_dim: int) -> None:
    if not x.is_cuda:
        raise RuntimeError("the sLSTM cuda_fused backend requires a CUDA input")
    capability = torch.cuda.get_device_capability(x.device)
    if capability < (8, 0):
        raise RuntimeError(
            "the sLSTM cuda_fused backend requires CUDA compute capability >= 8.0; "
            f"found {capability[0]}.{capability[1]}"
        )
    if head_dim % 8 != 0:
        raise RuntimeError(
            "the sLSTM cuda_fused backend requires a head dimension divisible by 8; "
            f"found {head_dim}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(
            "the sLSTM cuda_fused backend supports bfloat16, float16, or float32 inputs; "
            f"found {x.dtype}"
        )
    if shutil.which("nvcc") is None:
        raise RuntimeError(
            "the sLSTM cuda_fused backend JIT-compiles its persistent kernel and requires nvcc"
        )


def _flashrnn_opaque(
    flashrnn,
    gate_inputs: torch.Tensor,
    recurrent: torch.Tensor,
    bias: torch.Tensor,
    flash_state: torch.Tensor | None,
    kernel_dtype: str,
    config_class=None,
):
    """Execute the external persistent kernel eagerly inside compiled surrounding regions."""
    try:
        dtype = getattr(torch, kernel_dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported FlashRNN kernel dtype '{kernel_dtype}'") from exc
    # FlashRNN compiles one pointer type per tensor role, and an unnamed role inherits
    # another role instead of the kernel dtype. Describing the roles by the dtype the
    # tensors happen to carry -- bfloat16 once FSDP casts the parameters -- builds a kernel
    # whose own arguments do not fit its signature, so the kernel dtype names every role
    # and the tensors are cast to it.
    gate_inputs = gate_inputs.to(dtype)
    recurrent = recurrent.to(dtype)
    bias = bias.to(dtype)
    if flash_state is not None:
        flash_state = flash_state.to(dtype)
    kwargs = {
        "states": flash_state,
        "function": "slstm",
        "backend": "cuda_fused",
        "dtype": kernel_dtype,
    }
    if config_class is not None:
        kwargs["config"] = config_class(
            head_dim=gate_inputs.shape[3],
            num_heads=gate_inputs.shape[2],
            batch_size=gate_inputs.shape[1],
            function="slstm",
            backend="cuda_fused",
            dtype=kernel_dtype,
            dtype_w=kernel_dtype,
            dtype_r=kernel_dtype,
            dtype_b=kernel_dtype,
            dtype_g=kernel_dtype,
            dtype_s=kernel_dtype,
            dtype_a=kernel_dtype,
            input_shape="TBHDG",
            output_shape="SBHTD",
            recurrent_shape="HDGP",
            bias_shape="HDG",
        )
    return flashrnn(gate_inputs, recurrent, bias, **kwargs)


class _FlashRNNPersistentSLSTMLayer(nn.Module):
    """Official sLSTM parameters evaluated by FlashRNN's persistent CUDA kernel."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        batch_size: int,
        kernel_dtype: str,
    ):
        super().__init__()
        self.config = replace(layer.config, backend="cuda_fused")
        self.batch_size = batch_size
        self.kernel_dtype = kernel_dtype
        self.backend_identity = (
            f"flashrnn=={FLASHRNN_VERSION}:function=slstm:backend=cuda_fused:dtype={kernel_dtype}"
        )
        self._flashrnn = _PREFLIGHTED_FLASHRNN
        self.conv1d = layer.conv1d
        self.conv_act_fn = layer.conv_act_fn
        self.slstm_cell = layer.slstm_cell
        self.group_norm = layer.group_norm
        self.dropout = layer.dropout

        # The official layer names the projection modules after their source gates, while its
        # forward maps fgate -> i and igate -> f. Keep that exact canonical [i, f, z, o] order.
        self.conv_gate_weight = nn.Parameter(
            torch.stack((layer.fgate.weight, layer.igate.weight), dim=1)
        )
        self.raw_gate_weight = nn.Parameter(
            torch.stack((layer.zgate.weight, layer.ogate.weight), dim=1)
        )

    def _project_gates(self, x: torch.Tensor, x_conv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        source_inputs = torch.stack((x_conv, x), dim=0).view(
            2,
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        )
        gates = torch.einsum(
            "sbthi,shgoi->tbhosg",
            source_inputs,
            torch.stack((self.conv_gate_weight, self.raw_gate_weight), dim=0),
        )
        return gates.reshape(
            seq_len,
            batch_size,
            self.config.num_heads,
            self.config.head_dim,
            4,
        )

    def _flashrnn_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reconverted on every call rather than cached across calls. Under FSDP2 the
        # unsharded parameter is one persistent object whose storage is repointed at each
        # all-gather buffer with its version counter held fixed, so its id, version, device
        # and dtype are the same on every step and no key drawn from them can tell a
        # rematerialized value from a stale one.
        recurrent = (
            self.slstm_cell._recurrent_kernel_.view(
                self.config.num_heads,
                4,
                self.config.head_dim,
                self.config.head_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        bias = (
            self.slstm_cell._bias_.view(4, self.config.num_heads, self.config.head_dim)
            .permute(1, 2, 0)
            .contiguous()
        )
        return recurrent, bias

    @torch.compiler.disable
    def _run_flashrnn(
        self,
        gate_inputs: torch.Tensor,
        flash_state: torch.Tensor | None,
    ):
        recurrent, bias = self._flashrnn_parameters()
        return _flashrnn_opaque(
            self._flashrnn,
            gate_inputs,
            recurrent,
            bias,
            flash_state,
            self.kernel_dtype,
            _PREFLIGHTED_FLASHRNN_CONFIG,
        )

    def _normalize_and_pack(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = hidden.shape
        if not hasattr(self.group_norm, "eps"):
            return (
                self.group_norm(hidden)
                .transpose(1, 2)
                .reshape(
                    batch_size,
                    seq_len,
                    self.config.embedding_dim,
                )
            )
        hidden = torch.nn.functional.layer_norm(
            hidden.transpose(1, 2),
            (self.config.head_dim,),
            weight=None,
            bias=None,
            eps=self.group_norm.eps,
        )
        weight = self.group_norm.weight_proxy
        if weight is not None:
            hidden = hidden * weight.view(
                1,
                1,
                self.config.num_heads,
                self.config.head_dim,
            )
        if self.group_norm.bias is not None:
            hidden = hidden + self.group_norm.bias.view(
                1,
                1,
                self.config.num_heads,
                self.config.head_dim,
            )
        return hidden.reshape(batch_size, seq_len, self.config.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        slstm_state: torch.Tensor | None = None,
        return_last_state: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, _seq_len, _ = x.shape
        if batch_size != self.batch_size:
            raise RuntimeError(
                "the sLSTM cuda_fused backend was configured for batch size "
                f"{self.batch_size}, but received {batch_size}"
            )
        if self._flashrnn is None:
            raise RuntimeError(
                "the sLSTM cuda_fused backend was not preflighted; "
                "resolve FlashRNN before model construction"
            )

        if return_last_state:
            x_conv, conv_state = self.conv1d(
                x,
                conv_state,
                return_last_state=True,
            )
        else:
            x_conv = self.conv1d(x, conv_state, return_last_state=False)
        x_conv = self.conv_act_fn(x_conv)
        gate_inputs = self._project_gates(x, x_conv)

        flash_state = None
        if slstm_state is not None:
            expected_shape = (4, batch_size, self.config.embedding_dim)
            if slstm_state.shape != expected_shape:
                raise ValueError(
                    f"expected sLSTM state shape {expected_shape}, found {slstm_state.shape}"
                )
            flash_state = slstm_state.view(
                4,
                batch_size,
                self.config.num_heads,
                1,
                self.config.head_dim,
            )

        states, last_state = self._run_flashrnn(gate_inputs, flash_state)
        hidden = states[0]
        hidden = self.dropout(hidden)
        output = self._normalize_and_pack(hidden)
        if return_last_state:
            return output, {
                "conv_state": conv_state,
                "slstm_state": last_state[:, :, :, -1].reshape(
                    4,
                    batch_size,
                    self.config.embedding_dim,
                ),
            }
        return output

    def reset_parameters(self) -> None:
        from xlstm.components.init import small_init_init_

        self.slstm_cell.reset_parameters()
        self.group_norm.reset_parameters()
        # Match the official reset order: igate, fgate, zgate, ogate.
        small_init_init_(self.conv_gate_weight[:, 1], dim=self.config.embedding_dim)
        small_init_init_(self.conv_gate_weight[:, 0], dim=self.config.embedding_dim)
        small_init_init_(self.raw_gate_weight[:, 0], dim=self.config.embedding_dim)
        small_init_init_(self.raw_gate_weight[:, 1], dim=self.config.embedding_dim)


def _build_slstm_layer(
    *,
    d_model: int,
    n_heads: int,
    conv_size: int,
    backend: str,
    batch_size: int,
    layer_idx: int,
    n_layers: int,
    kernel_dtype: str,
    init_device: str,
    fuse_input_projections: bool,
) -> nn.Module:
    try:
        from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig
    except ImportError as exc:
        raise ImportError("sLSTM requires xlstm==2.0.5") from exc

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "the official CUDA sLSTM backend was requested but CUDA is unavailable; "
            "choose backend='vanilla' explicitly for the control implementation"
        )
    official_backend = "vanilla" if backend == "cuda_fused" else backend
    actual_kernel_dtype = kernel_dtype if official_backend == "cuda" else "float32"

    config = sLSTMLayerConfig(
        embedding_dim=d_model,
        num_heads=n_heads,
        conv1d_kernel_size=conv_size,
        backend=official_backend,
        batch_size=batch_size,
        dtype=actual_kernel_dtype,
        dtype_b="float32",
        enable_automatic_mixed_precision=official_backend == "cuda",
        _block_idx=layer_idx,
        _num_blocks=n_layers,
    )
    with torch.device(init_device):
        layer = sLSTMLayer(config)
        if backend == "cuda_fused":
            if not fuse_input_projections:
                raise ValueError(
                    "the sLSTM cuda_fused backend requires direct fused input projections"
                )
            layer = _FlashRNNPersistentSLSTMLayer(
                layer,
                batch_size=batch_size,
                kernel_dtype=kernel_dtype,
            )
        elif fuse_input_projections:
            layer = _FusedInputSLSTMLayer(layer)
    _make_reset_parameters_sharding_safe(layer)
    return layer


class SLSTMMixer(SequenceMixer):
    """
    Scalar-memory recurrent xLSTM sequence mixer.

    :param d_model: Model hidden size.
    :param n_heads: Number of sLSTM heads.
    :param conv_size: Causal depthwise-convolution kernel size.
    :param backend: sLSTM backend: FlashRNN ``"cuda_fused"`` or official xLSTM
        ``"cuda"`` / ``"vanilla"``.
    :param batch_size: Per-rank sequence batch size compiled into the CUDA kernel.
    :param layer_idx: Layer position, used by block-dependent forget-bias initialization.
    :param n_layers: Total model depth, used by forget-bias initialization.
    :param kernel_dtype: Internal sLSTM kernel dtype.
    :param fuse_input_projections: Fuse the four input-side head-wise projections into two
        contractions without changing the recurrent cell.
    :param init_device: Parameter initialization device.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 4,
        conv_size: int = 4,
        backend: str = "cuda",
        batch_size: int = 16,
        layer_idx: int,
        n_layers: int,
        kernel_dtype: str = "bfloat16",
        fuse_input_projections: bool = True,
        init_device: str = "cpu",
    ):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model")
        if conv_size <= 0:
            raise ValueError("conv_size must be positive")
        if backend not in ("cuda_fused", "cuda", "vanilla"):
            raise ValueError(f"unknown sLSTM backend '{backend}'")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.conv_size = conv_size
        self.backend = backend
        self.batch_size = batch_size
        self.fuse_input_projections = fuse_input_projections
        self.layer = _build_slstm_layer(
            d_model=d_model,
            n_heads=n_heads,
            conv_size=conv_size,
            backend=backend,
            batch_size=batch_size,
            layer_idx=layer_idx,
            n_layers=n_layers,
            kernel_dtype=kernel_dtype,
            init_device=init_device,
            fuse_input_projections=fuse_input_projections,
        )
        self.backend_identity = getattr(self.layer, "backend_identity", f"xlstm.{backend}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        local_keys = {key[len(prefix) :] for key in state_dict if key.startswith(prefix)}
        has_packed = any(
            key in local_keys for key in ("layer.conv_gate_weight", "layer.raw_gate_weight")
        )
        has_unpacked = any(
            key in local_keys
            for key in (
                "layer.fgate.weight",
                "layer.igate.weight",
                "layer.zgate.weight",
                "layer.ogate.weight",
            )
        )
        if self.fuse_input_projections and has_unpacked:
            raise RuntimeError(
                "sLSTM checkpoint layout mismatch: configured packed input projections, "
                "but the checkpoint uses the unpacked official layout. "
                "Call convert_slstm_official_to_packed_state_dict explicitly, or select "
                "--unfused-slstm-input-projections. Optimizer-state conversion is unsupported."
            )
        if not self.fuse_input_projections and has_packed:
            raise RuntimeError(
                "sLSTM checkpoint layout mismatch: configured unpacked input projections, "
                "but the checkpoint uses the packed layout. "
                "Call convert_slstm_packed_to_official_state_dict explicitly, or select "
                "--fused-slstm-input-projections. Optimizer-state conversion is unsupported."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply scalar-memory recurrent sequence mixing.

        :param x: Input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Unsupported until segmented sLSTM state resets exist.
        :returns: Output with the same shape as ``x``.
        """
        if cu_doc_lens is not None:
            raise RuntimeError(
                "sLSTM document boundaries are not supported until segmented state resets exist"
            )
        del kwargs
        output = self.layer(x)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for SLSTMMixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        raise NotImplementedError("Context parallelism is not implemented for SLSTMMixer")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod

        del d_model, block_idx, num_blocks, std
        if init_method != InitMethod.normal:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for SLSTMMixer"
            )
        # The layer's `reset_parameters` stages its own writes whenever the parameters are
        # sharded, so this is one call at one rank and at many.
        _reset_parameters_with_generator(self.layer, generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return 6 * sum(param.numel() for param in self.parameters())


@SequenceMixerConfig.register("slstm")
@dataclass
class SLSTMMixerConfig(SequenceMixerConfig[SLSTMMixer]):
    """Configuration for :class:`SLSTMMixer`."""

    n_heads: int = 4
    conv_size: int = 4
    backend: str = "cuda"
    batch_size: int = 16
    kernel_dtype: str = "bfloat16"
    fuse_input_projections: bool = True
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        # Four head-wise input projections and four recurrent projections.
        params = 8 * d_model * d_model // self.n_heads
        # Depthwise convolution (weight+bias), cell biases, and output norm.
        params += d_model * (self.conv_size + 6)
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: BufferCache | None = None,
    ) -> SLSTMMixer:
        del cache
        return SLSTMMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            conv_size=self.conv_size,
            backend=self.backend,
            batch_size=self.batch_size,
            layer_idx=layer_idx,
            n_layers=n_layers,
            kernel_dtype=self.kernel_dtype,
            fuse_input_projections=self.fuse_input_projections,
            init_device=init_device,
        )
