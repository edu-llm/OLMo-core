import functools
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate

from ..config import DType, StrEnum
from ..doc_utils import beta_feature
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize
from .quantization import QuantConfig, QuantLinear
from .utils import get_tp_wrappers

__all__ = [
    "ActivationFunction",
    "FeedForwardType",
    "FeedForwardConfig",
    "FeedForward",
    "NormalizedFeedForward",
    "MAPLE_SWIGLU_LIMIT",
]


MAPLE_SWIGLU_LIMIT: float = 7.0
"""
The SwiGLU outlier clamp Maple-Preview carries: gate ``max=7.0``, up ``[-7.0, 7.0]``.

Semantically identical to ``transformers/models/gpt_oss/modeling_gpt_oss.py`` (``self.limit =
7.0``), **including the asymmetry** -- which is the tell that it was copied rather than derived.
Measured layer-0 MoE pre-activation RMS in the released model is ~0.136, so 7.0 is roughly
**52x RMS**: the clamp is inert at inference and almost certainly never fires during training
either. It is included as a faithfulness detail. Do not "tune" it -- a clamp that never fires
costs nothing, and changing it is the kind of well-meaning edit that makes a replication stop
being one.
"""


class ActivationFunction(StrEnum):
    """
    An enumeration of the supported activation functions for feed-forward modules.
    """

    silu = "silu"
    """
    SiLU/Swish activation function, used for SwiGLU.
    """

    gelu_tanh = "gelu_tanh"
    """
    GELU with tanh approximation, used for GeGLU.
    """

    def build(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == ActivationFunction.silu:
            return F.silu
        elif self == ActivationFunction.gelu_tanh:
            return functools.partial(F.gelu, approximate="tanh")
        else:
            raise NotImplementedError(self)


class FeedForwardType(StrEnum):
    """
    An enumeration of the different feed-forward / MLP implementations.
    """

    default = "default"
    """
    ➡️ :class:`FeedForward`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedFeedForward`
    """


@dataclass
class FeedForwardConfig(ModuleConfig):
    """
    A config for building :class:`FeedForward` modules.
    """

    hidden_size: int
    name: FeedForwardType = FeedForwardType.default
    """
    The name of the implementation.
    """
    bias: Optional[bool] = None
    dtype: Optional[DType] = None
    activation: ActivationFunction = ActivationFunction.silu
    """
    The activation function to use. See :class:`ActivationFunction` for options.
    """
    quant: Optional[QuantConfig] = None
    """
    Ternary QAT on the three projections. ``None`` builds stock :class:`torch.nn.Linear`.

    See :mod:`olmo_core.nn.quantization`. Note that ``QuantConfig(enabled=False)`` is *not* the
    same as ``None``: it builds :class:`~olmo_core.nn.quantization.QuantLinear` with the
    quantizer bypassed, which is bitwise identical in the forward pass but keeps the module
    graph and state-dict keys of the ternary arm. That is what makes bf16-vs-ternary a paired
    comparison. Use ``enabled=False`` for the control arm, not ``None``.
    """
    swiglu_limit: Optional[float] = None
    """
    Clamp the gate (``w1``) output to ``max=limit`` and the up (``w3``) output to
    ``[-limit, limit]`` before the gated product.

    Set to :data:`MAPLE_SWIGLU_LIMIT` (7.0) for Maple faithfulness. This is an *architecture*
    detail, not a quantization one -- Maple carries it in both cases and so must both arms of
    the precision comparison, so it is deliberately a separate knob from ``quant``. The
    asymmetry (one-sided on gate, two-sided on up) is intentional and copied.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        bias = self.bias if self.bias is not None else self.name != FeedForwardType.normalized

        params = 0

        params += 3 * d_model * self.hidden_size
        if bias:
            params += 2 * self.hidden_size + d_model

        # w1 + w3 scaling factors
        if self.name == FeedForwardType.normalized:
            params += 2 * self.hidden_size

        return params

    def build(
        self, d_model: int, *, dtype: Optional[torch.dtype] = None, init_device: str = "cpu"
    ) -> "FeedForward":
        """
        Build the corresponding feed-forward module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        # `as_dict` recurses by default, so a nested `QuantConfig` arrives here as a plain
        # dict. Put the real object back -- silently passing `{"enabled": True}` would make
        # `quant.enabled` an AttributeError at first forward, i.e. at step 0 of a queued run.
        if self.quant is not None:
            kwargs["quant"] = self.quant

        try:
            if self.name == FeedForwardType.default:
                return FeedForward(**kwargs)
            elif self.name == FeedForwardType.normalized:
                activation = kwargs.get("activation", ActivationFunction.silu)
                if activation != ActivationFunction.silu:
                    raise OLMoConfigurationError(
                        f"NormalizedFeedForward only supports 'silu' activation, got '{activation}'"
                    )
                if self.quant is not None:
                    raise OLMoConfigurationError(
                        "ternary QAT is not supported with NormalizedFeedForward: nGPT "
                        "re-normalizes the weight matrices after every optimizer step, which "
                        "fights the TWN threshold for control of the weight scale"
                    )
                if self.swiglu_limit is not None:
                    raise OLMoConfigurationError(
                        "'swiglu_limit' is not supported with NormalizedFeedForward"
                    )
                return NormalizedFeedForward(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class FeedForward(nn.Module):
    """
    Basic feed-forward module with gated activation (SwiGLU or GeGLU).
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
        quant: Optional[QuantConfig] = None,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.activation_fn = activation.build()
        self.quant = quant
        self.swiglu_limit = swiglu_limit

        # QuantLinear subclasses nn.Linear and, with `enabled=False`, has an identical forward
        # and an identical state dict. So the control arm of the paired comparison builds the
        # same class as the ternary arm, and `init_linear` / `apply_tp` / `normalize_matrices`
        # keep working on both without a special case.
        def linear(in_features: int, out_features: int) -> nn.Linear:
            if quant is None:
                return nn.Linear(
                    in_features, out_features, bias=bias, dtype=dtype, device=init_device
                )
            return QuantLinear(
                in_features,
                out_features,
                bias=bias,
                enabled=quant.enabled,
                backend=quant.backend,
                ste_policy=quant.ste_policy,
                fallback_to_fake_quant=quant.fallback_to_fake_quant,
                dtype=dtype,
                device=init_device,
            )

        self.w1 = linear(d_model, hidden_size)
        self.w2 = linear(hidden_size, d_model)
        self.w3 = linear(d_model, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit is not None:
            # Asymmetric on purpose: gate is clamped above only, up on both sides. This is
            # gpt-oss's shape, copied. See MAPLE_SWIGLU_LIMIT.
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.w2(self.activation_fn(gate) * up)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        # `w2` is row-wise sharded, i.e. sharded on its input-feature axis -- the exact axis TWN
        # reduces over. Under TP each rank would compute `mean|W|` from its own shard only,
        # producing a per-row alpha derived from a fraction of the row. That is a different
        # quantizer, and it would train without complaint, so refuse instead.
        for w in (self.w1, self.w2, self.w3):
            if isinstance(w, QuantLinear):
                w.assert_no_tensor_parallel()

        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        return 6 * sum(p.numel() for p in self.parameters())


@beta_feature
class NormalizedFeedForward(FeedForward):
    """
    An nGPT feed-forward implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: ActivationFunction = ActivationFunction.silu,
    ):
        if activation != ActivationFunction.silu:
            raise OLMoConfigurationError(
                f"NormalizedFeedForward only supports 'silu' activation, got '{activation}'"
            )
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            bias=False,
            activation=activation,
        )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sw3 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sqrt_d_model = math.sqrt(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        nn.init.ones_(self.sw3)
        with torch.no_grad():
            self.sw1.mul_(self.sw_init_scaling)
            self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized feed-forward variant"
        )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
