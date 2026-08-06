import logging
import math
import warnings
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Placement, Shard, distribute_tensor

from olmo_core.distributed.parallel import get_device_mesh_info
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.quantization import QuantConfig, twn_quantize_ste
from olmo_core.utils import log_once

try:
    import grouped_gemm  # type: ignore

    gmm = grouped_gemm.ops.gmm
except ImportError:
    gmm = None

__all__ = ["MoEMLP", "DroplessMoEMLP"]


log = logging.getLogger(__name__)


class MoEMLPBase(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        quant: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.quant = quant

        self.num_local_experts = num_experts
        self.hidden_sharding_degree = 1
        self.ep_mesh: Optional[DeviceMesh] = None
        self.ep_pg: Optional[dist.ProcessGroup] = None

    def maybe_quantize(self, w: torch.Tensor, *, in_dim: int) -> torch.Tensor:
        """
        Apply ternary QAT to a stacked expert weight, or return it untouched.

        Expert weights here are bare stacked :class:`torch.nn.Parameter` tensors of shape
        ``(num_experts, a, b)``, not :class:`torch.nn.Linear` submodules, so
        :class:`~olmo_core.nn.quantization.QuantLinear` does not apply and the quantizer is
        called on the tensor directly. This is a single hook at the top of ``forward`` rather
        than a rewrite of the matmul sequence, which keeps the ``grouped_gemm``-versus-``bmm``
        question entirely in L3's hands.

        ``in_dim`` must be the axis the *forward pass* treats as input features, which is not
        the same axis for every weight here -- see the table in
        :func:`~olmo_core.nn.quantization.twn_quantize`. Reducing over the wrong one gives a
        per-input-row alpha: a different quantizer that trains happily.

        ``quant=None`` and ``QuantConfig(enabled=False)`` both return ``w`` **by identity**, so
        the control arm is not merely numerically close to the unquantized path, it *is* the
        unquantized path.
        """
        if self.quant is None or not self.quant.enabled:
            return w
        return twn_quantize_ste(w, in_dim=in_dim)

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.

        :param ep_mesh: A 1D device mesh to shard experts over.
        """
        if ep_mesh.ndim != 1:
            raise RuntimeError("expert parallel mesh must be 1 dimensional")
        self._shard_experts(ep_mesh)

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        """
        Apply expert parallelism.

        :param tp_mesh: A 1D device mesh to shard experts over.
        """
        # NOTE (ternary QAT): unlike `FeedForward.apply_tp` and `Attention.apply_tp`, this needs
        # no `assert_no_tensor_parallel` guard -- but only because it delegates to
        # `_shard_experts`, which shards flat axis 0 (the EXPERT axis), never a feature axis. So
        # TWN's per-output-row reduction still sees whole rows. That is load-bearing: if a
        # feature-axis placement is ever added below, each rank would derive alpha from a
        # fraction of each row and the quantizer would change silently, with no guard to catch
        # it. Same caveat for `hidden_sharding_degree`, which is hardcoded to 1 -- raising it
        # would shard the hidden axis, which IS the reduction axis for w1/w3.
        del float8_enabled  # TODO
        if tp_mesh.ndim != 1:
            raise RuntimeError("tensor parallel mesh must be 1 dimensional")
        self._shard_experts(tp_mesh)

    def _shard_experts(self, mesh: DeviceMesh):
        num_shards = mesh.size()
        if self.num_experts % num_shards != 0:
            raise OLMoConfigurationError(
                f"'num_experts' ({self.num_experts}) must be divisible by the expert parallel shard degree ({num_shards})."
            )

        self.ep_mesh = mesh
        self.ep_pg = mesh.get_group()
        self.num_local_experts = self.num_experts // num_shards

        placements: List[Placement] = [Shard(0)]
        self.register_parameter("w1", nn.Parameter(distribute_tensor(self.w1, mesh, placements)))  # type: ignore
        self.register_parameter("w2", nn.Parameter(distribute_tensor(self.w2, mesh, placements)))  # type: ignore
        self.register_parameter("w3", nn.Parameter(distribute_tensor(self.w3, mesh, placements)))  # type: ignore

    def prepare_experts_for_fsdp(self, *, world_mesh: DeviceMesh, **kwargs):
        """
        Should be called before wrapping this module, or a parent module, with FSDP2.
        """
        # If expert/tensor parallel is not enabled then we don't need to do anything special here.
        if self.ep_mesh is None:
            return

        if self.ep_mesh.mesh_dim_names is None:
            raise RuntimeError("mesh must have named dimensions!")

        if (dim_names := world_mesh.mesh_dim_names) is None:
            raise RuntimeError("mesh must have named dimensions!")

        # If the experts are already sharded over a data parallel dimension, we need to shard them
        # over the other data parallel dimension, otherwise `fully_shard` called with the full DP
        # mesh won't handle this module correctly.
        if (ep_mesh_dim_name := self.ep_mesh.mesh_dim_names[0]).startswith("dp"):
            # Shard local experts over the adjacent DP dimension.
            dp_replicate_dim_name = dim_names[dim_names.index(ep_mesh_dim_name) - 1]
            dp_replicate_mesh = world_mesh[dp_replicate_dim_name]

            log_once(
                log, f"Sharding local experts over {get_device_mesh_info(dp_replicate_mesh)}..."
            )
            fully_shard(self, mesh=dp_replicate_mesh, **kwargs)

    def prepare_experts_for_ddp(self, *, world_mesh: DeviceMesh):
        """
        Should be called before wrapping this module, or a parent module, with FSDP2.
        """
        # TODO: do we need to do anything special here like with FSDP?
        del world_mesh
        pass


class MoEMLP(MoEMLPBase):
    """
    A basic expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        quant: Optional[QuantConfig] = None,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__(
            d_model=d_model, hidden_size=hidden_size, num_experts=num_experts, quant=quant
        )
        self.swiglu_limit = swiglu_limit
        # NOTE: these parameters need to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP, so we flatten the first 2 dimensions.
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def extra_repr(self):
        return f"num_experts={self.num_experts}, in_features={self.d_model}, hidden_size={self.hidden_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(num_local_experts, N, d_model)``.
        """
        og_dtype = x.dtype

        # Scale gradients and get local tensors (in case of expert parallelism).
        # shapes: w1, w3 -> (num_local_experts, d_model, hidden_size)
        #         w2     -> (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.w1.view(self.num_experts, self.d_model, self.hidden_size)),
            get_local_tensor(self.w2.view(self.num_experts, self.hidden_size, self.d_model)),
            get_local_tensor(self.w3.view(self.num_experts, self.d_model, self.hidden_size)),
        )

        # Ternary QAT, if enabled. `in_dim=1` for all three because `torch.bmm(x, w)` contracts
        # `w`'s axis -2 UNCONDITIONALLY -- it is forced by the operator, not by the fact that
        # w1/w3 are (d_model, hidden) while w2 is (hidden, d_model). So this is NOT fragile to
        # that transposition; it is fragile to a change of operator (bmm -> einsum, baddbmm with
        # a transposed operand, gmm), which would move the contracted axis.
        #
        # Applied after `get_local_tensor` so that under expert parallelism each rank quantizes
        # whole experts. That is safe because `_shard_experts` uses `Shard(0)` on the FLATTENED
        # (E*a, b) parameter and enforces `num_experts % num_shards == 0` -- the reduction axis
        # is interleaved inside flat axis 0, so a cut at a non-expert boundary would split rows.
        # The divisibility guard is what makes post-shard quantization equivalent to pre-shard,
        # not the shard axis alone. Verified under real DTensor, FSDP2, and stacked EP+FSDP2 in
        # `maple/agents/lanes/L4-ternary/verify/in-dim-orientation.md`.
        w1 = self.maybe_quantize(w1, in_dim=1)
        w2 = self.maybe_quantize(w2, in_dim=1)
        w3 = self.maybe_quantize(w3, in_dim=1)

        x = x.type_as(w1)

        # Compute the MLP.
        gate = torch.bmm(x, w1)
        up = torch.bmm(x, w3)
        if self.swiglu_limit is not None:
            # gpt-oss's asymmetric SwiGLU outlier guard; see feed_forward.MAPLE_SWIGLU_LIMIT.
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return torch.bmm(F.silu(gate) * up, w2).to(dtype=og_dtype)


class DroplessMoEMLP(MoEMLPBase):
    """
    A dropless expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        quant: Optional[QuantConfig] = None,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__(
            d_model=d_model, hidden_size=hidden_size, num_experts=num_experts, quant=quant
        )
        self.swiglu_limit = swiglu_limit
        # NOTE: these parameters need to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP, so we flatten the first 2 dimensions.
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )

        self._gmm = gmm
        if self._gmm is None:
            warnings.warn(
                "Grouped GEMM not available, so the MoE will be substantially slower. "
                "Please install the 'grouped_gemm' package if possible.\n"
                "https://github.com/tgale96/grouped_gemm"
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    @torch._dynamo.disable()
    def gmm(
        self, x: torch.Tensor, w: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool = False
    ) -> torch.Tensor:
        if self._gmm is not None:
            # grouped-gemm only accepts BF16
            return self._gmm(x.to(torch.bfloat16), w.to(torch.bfloat16), batch_sizes, trans_b=trans_b)  # type: ignore
        else:
            out = []
            start = 0
            for i, size in enumerate(batch_sizes.cpu().numpy()):
                rhs = w[i, :, :].t() if trans_b else w[i, :, :]
                out.append(x[start : start + size, :] @ rhs)
                start += size
            return torch.cat(out)

    def forward(self, x: torch.Tensor, batch_size_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(*, d_model)``.
        :param batch_size_per_expert: Specifies how many items/tokens go to each expert. Should be a
            1-D ``LongTensor``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.w1.view(self.num_experts, self.hidden_size, self.d_model)),
            get_local_tensor(self.w2.view(self.num_experts, self.hidden_size, self.d_model)),
            get_local_tensor(self.w3.view(self.num_experts, self.hidden_size, self.d_model)),
        )

        # Ternary QAT, if enabled. Note `in_dim` differs between w1/w3 and w2 even though all
        # three tensors have the SAME shape (num_experts, hidden_size, d_model). The axis that
        # counts is the one the matmul consumes as input features, and `gmm` is called with
        # `trans_b=True` for w1/w3 but not for w2:
        #   w1, w3 : x @ w[i].T  -> w[i].T is (d_model, hidden), input features are d_model, axis 2
        #   w2     : x @ w[i]    -> w[i]   is (hidden, d_model), input features are hidden,  axis 1
        # Using 2 for w2 would compute alpha across d_model for a matmul whose output rows are
        # indexed by d_model -- a per-input-row scale, i.e. a different quantizer that trains
        # without complaint. If L3 changes a `trans_b`, this must change with it.
        w1 = self.maybe_quantize(w1, in_dim=2)
        w3 = self.maybe_quantize(w3, in_dim=2)
        w2 = self.maybe_quantize(w2, in_dim=1)

        # Compute the MLP.
        x1 = self.gmm(x, w1, batch_size_per_expert, trans_b=True)
        x2 = self.gmm(x, w3, batch_size_per_expert, trans_b=True)
        if self.swiglu_limit is not None:
            # gpt-oss's asymmetric SwiGLU outlier guard; see feed_forward.MAPLE_SWIGLU_LIMIT.
            x1 = x1.clamp(max=self.swiglu_limit)
            x2 = x2.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        x1 = F.silu(x1) * x2
        return self.gmm(x1, w2, batch_size_per_expert)
