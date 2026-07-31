"""
Gated short convolution sequence mixer, matching Liquid AI's LFM2 "LIV" block.

The operator is pinned to the released Apache-2.0 implementation
(``transformers`` v5.0.0rc1, ``models/lfm2/modeling_lfm2.py``, class ``Lfm2ShortConv``)::

    BCx     = in_proj(x)                    # (B, T, 3d), chunked as (B, C, x)
    Bx      = B * x                         # pre-gate
    z       = depthwise_causal_conv(Bx)     # kernel_size taps, no bias, no activation
    out     = out_proj(C * z)               # post-gate

Three properties of the released block are load-bearing and easy to get wrong:

1. **The chunk order is** ``(B, C, x)`` — pre-gate, post-gate, value. Permuting these still
   trains, just worse, so it is a silent failure.
2. **There is no activation anywhere in the conv path.** ``Lfm2ShortConv`` passes
   ``activation=None``. Note :class:`~olmo_core.nn.convolution.CausalConv1d` defaults to
   ``activation="silu"`` *inside* the fused kernel, so reusing it unchanged implements a
   different operator.
3. **There is no normalization inside the block.** The RMSNorm (``operator_norm``) is owned by
   the decoder layer and applied before this module is called.

The gate projections additionally support two cheap structures, for the low-rank-gate study:

* ``gate_rank=r`` factorizes both gates through a shared ``d -> 2r`` down-projection.
* ``gate_groups=g`` makes both gates block-diagonal with ``g`` blocks.

At matched cost these are *not* nested — block-diagonal has full rank but no cross-block
mixing, while low-rank mixes all channels through an ``r``-dimensional bottleneck.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = ["ShortConv", "ShortConvConfig", "GateStructure"]


GateStructure = Literal["dense", "lowrank", "grouped"]


class _GateProj(nn.Module):
    """
    The ``d -> 3d`` input projection, with optionally-structured gate blocks.

    The value block is always dense: only the two gates are candidates for compression, and
    keeping the value stream dense is what distinguishes this from a plain narrow model.

    :param d_model: The model dimensionality.
    :param structure: ``"dense"``, ``"lowrank"``, or ``"grouped"``.
    :param rank: The bottleneck rank, required when ``structure="lowrank"``.
    :param groups: The number of blocks, required when ``structure="grouped"``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        structure: GateStructure = "dense",
        rank: Optional[int] = None,
        groups: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.structure = structure
        self.rank = rank
        self.groups = groups
        kwargs = {"dtype": dtype, "device": init_device}

        # The value stream is dense in every variant.
        self.value_proj = nn.Linear(d_model, d_model, bias=False, **kwargs)

        if structure == "dense":
            self.gate_proj = nn.Linear(d_model, 2 * d_model, bias=False, **kwargs)
        elif structure == "lowrank":
            if rank is None or rank <= 0:
                raise ValueError("'rank' must be a positive int when structure='lowrank'")
            # One shared d -> 2r down-projection rather than two d -> r. Same parameter count,
            # one fewer kernel launch per layer at decode time.
            self.gate_down = nn.Linear(d_model, 2 * rank, bias=False, **kwargs)
            self.gate_up_pre = nn.Linear(rank, d_model, bias=False, **kwargs)
            self.gate_up_post = nn.Linear(rank, d_model, bias=False, **kwargs)
        elif structure == "grouped":
            if groups is None or groups <= 0:
                raise ValueError("'groups' must be a positive int when structure='grouped'")
            if d_model % groups != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by groups ({groups})")
            bs = d_model // groups
            self.gate_blocks_pre = nn.Parameter(torch.empty(groups, bs, bs, **kwargs))
            self.gate_blocks_post = nn.Parameter(torch.empty(groups, bs, bs, **kwargs))
            # Self-initialize, matching nn.Linear: a module must be usable before
            # init_weights() runs, or tests and probes silently operate on uninitialized
            # memory (which is often all zeros, making a broken module look merely inert).
            if init_device != "meta":
                for p in (self.gate_blocks_pre, self.gate_blocks_post):
                    nn.init.kaiming_uniform_(p, a=5**0.5)
        else:
            raise ValueError(f"unknown gate structure '{structure}'")

    def _grouped(self, x: torch.Tensor, blocks: nn.Parameter) -> torch.Tensor:
        g, bs = self.groups, self.d_model // self.groups  # type: ignore[operator]
        lead = x.shape[:-1]
        v = x.reshape(-1, g, bs).transpose(0, 1)  # (g, N, bs)
        out = torch.bmm(v, blocks).transpose(0, 1)  # (N, g, bs)
        return out.reshape(*lead, self.d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: Input of shape ``(batch_size, seq_len, d_model)``.

        :returns: The ``(pre_gate, post_gate, value)`` streams, each
            ``(batch_size, seq_len, d_model)``.
        """
        value = self.value_proj(x)
        if self.structure == "dense":
            pre, post = self.gate_proj(x).chunk(2, dim=-1)
        elif self.structure == "lowrank":
            h = self.gate_down(x)
            pre = self.gate_up_pre(h[..., : self.rank])
            post = self.gate_up_post(h[..., self.rank :])
        else:
            pre = self._grouped(x, self.gate_blocks_pre)
            post = self._grouped(x, self.gate_blocks_post)
        return pre, post, value


class ShortConv(SequenceMixer):
    """
    A gated short causal convolution, as used by LFM2's non-attention layers.

    :param d_model: The model dimensionality.
    :param kernel_size: Number of convolution taps. LFM2 ships ``3``.
    :param gate_structure: ``"dense"`` (as released), ``"lowrank"``, or ``"grouped"``.
    :param gate_rank: The bottleneck rank when ``gate_structure="lowrank"``.
    :param gate_groups: The number of blocks when ``gate_structure="grouped"``.
    :param bias: Whether the projections and convolution use biases. LFM2 uses ``False``.
    :param use_fla: Use the fused ``fla`` convolution kernel when available. The fallback is a
        plain :class:`torch.nn.Conv1d`, which is what the reference implementation uses and
        which runs on CPU.
    """

    def __init__(
        self,
        *,
        d_model: int,
        kernel_size: int = 3,
        gate_structure: GateStructure = "dense",
        gate_rank: Optional[int] = None,
        gate_groups: Optional[int] = None,
        bias: bool = False,
        use_fla: bool = True,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.gate_structure = gate_structure
        self.gate_rank = gate_rank
        self.gate_groups = gate_groups
        self.use_fla = use_fla
        self.cp_enabled = False

        self.in_proj = _GateProj(
            d_model,
            structure=gate_structure,
            rank=gate_rank,
            groups=gate_groups,
            dtype=dtype,
            init_device=init_device,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        # Depthwise, causal via left padding, and deliberately activation-free.
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=bias,
            padding=kernel_size - 1,
            dtype=dtype,
            device=init_device,
        )

    def _conv(self, x: torch.Tensor, cu_doc_lens: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Depthwise causal convolution over ``(batch_size, seq_len, d_model)``.

        When ``cu_doc_lens`` is given, each document is convolved independently. A ``k``-tap
        filter that reads across a document boundary is a *different operator* from the one
        being studied, and at a ~622-token median document length a 4K sequence holds several
        documents, so this is not a small effect.
        """
        if cu_doc_lens is not None:
            if x.shape[0] != 1:
                raise RuntimeError("cu_doc_lens requires batch_size == 1")
            if self.use_fla and has_fla() and x.is_cuda:
                from olmo_core.nn.attention.flash_linear_attn_api import (
                    dispatch_causal_conv1d,
                )

                out = dispatch_causal_conv1d(
                    x=x,
                    weight=self.conv.weight.squeeze(1),
                    bias=self.conv.bias,
                    activation=None,  # never 'silu' -- see module docstring
                    cu_seqlens=cu_doc_lens,
                )
                return out[0] if isinstance(out, tuple) else out
            # Reference path: one conv per document, so no filter spans a boundary.
            bounds = cu_doc_lens.tolist()
            segments = [
                self._conv_dense(x[:, s:e]) for s, e in zip(bounds[:-1], bounds[1:]) if e > s
            ]
            return torch.cat(segments, dim=1)

        if self.use_fla and has_fla() and x.is_cuda:
            from olmo_core.nn.attention.flash_linear_attn_api import (
                dispatch_causal_conv1d,
            )

            out = dispatch_causal_conv1d(
                x=x,
                weight=self.conv.weight.squeeze(1),
                bias=self.conv.bias,
                activation=None,
            )
            return out[0] if isinstance(out, tuple) else out
        return self._conv_dense(x)

    def _conv_dense(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        z = self.conv(x.transpose(-1, -2))[..., :seq_len]  # trim the right padding
        return z.transpose(-1, -2)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        :param x: Input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths, a 1D :class:`torch.int32` tensor with
            one more element than there are documents, starting at ``0``. Requires
            ``batch_size == 1``.

        :returns: Output of shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs
        pre_gate, post_gate, value = self.in_proj(x)
        z = self._conv(pre_gate * value, cu_doc_lens)
        return self.out_proj(post_gate * z)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for ShortConv")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        # A depthwise conv needs the k-1 tokens preceding each shard, so sequence-sharded CP
        # would require a halo exchange that is not implemented. Channel-parallel CP would
        # work (channels are independent in a depthwise conv) but the gate projections mix
        # channels, so it cannot be done at this granularity either.
        raise NotImplementedError("Context parallelism is not yet implemented for ShortConv")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Initialize projections, and the convolution as a near-identity (last tap only).

        .. important::
            For the low-rank variant the two factors are initialized so that the *product* has
            the same output variance as the dense projection it replaces. With
            ``Var(y) = d * r * sigma_A^2 * sigma_B^2``, using ``std`` for both factors is
            24-48x too small, and the error is **monotone in r** — so a fixed-std rank sweep
            produces a smooth, plausible "higher rank is better" curve that is really an
            init-scale curve. Tests assert step-0 gate variance parity against dense.
        """
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(self.in_proj.value_proj, std=std, generator=generator)

        if self.gate_structure == "dense":
            init_linear(self.in_proj.gate_proj, std=std, generator=generator)
        elif self.gate_structure == "lowrank":
            assert self.gate_rank is not None
            # Var(y) = d * r * s_A^2 * s_B^2 must equal the dense d * std^2, so
            # s_A * s_B = std / sqrt(r). Split the geometric mean evenly between factors.
            factor_std = (std / self.gate_rank**0.5) ** 0.5
            init_linear(self.in_proj.gate_down, std=factor_std, generator=generator)
            init_linear(self.in_proj.gate_up_pre, std=factor_std, generator=generator)
            init_linear(self.in_proj.gate_up_post, std=factor_std, generator=generator)
        else:
            for p in (self.in_proj.gate_blocks_pre, self.in_proj.gate_blocks_post):
                nn.init.trunc_normal_(
                    p, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
                )

        # Identity-like conv: pass the current token through, zero the history. Keeps the
        # block close to a pure gated linear map at step 0 regardless of kernel width, so
        # the k3/k5/k9/k15 arms start from the same function.
        self.conv.weight.zero_()
        self.conv.weight[:, :, -1] = 1.0
        if self.conv.bias is not None:
            self.conv.bias.zero_()

        out_std = std
        if init_method == InitMethod.llama:
            out_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            out_std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            out_std = std / (2 * num_blocks) ** 0.5
        init_linear(self.out_proj, std=out_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        FLOPs per token: projections, the depthwise convolution, and the two gate multiplies.

        Note this is independent of ``seq_len`` — unlike attention, a short convolution has no
        term that grows with context. That asymmetry is exactly why arms must be matched on
        ``num_flops_per_token`` rather than on parameter count alone.
        """
        del seq_len
        d = self.d_model
        params = sum(p.numel() for p in self.in_proj.parameters())
        params += self.out_proj.weight.numel()
        linear_flops = 2 * params
        conv_flops = 2 * self.kernel_size * d
        gate_flops = 2 * d  # pre-gate and post-gate elementwise multiplies
        return int(linear_flops + conv_flops + gate_flops)


@dataclass
class ShortConvConfig(SequenceMixerConfig):
    """
    Configuration for :class:`ShortConv`, the LFM2-style gated short convolution.
    """

    kernel_size: int = 3
    """
    Number of convolution taps. LFM2 ships ``3``; larger values widen the receptive field.
    """
    gate_structure: GateStructure = "dense"
    """
    Structure of the two gate projections: ``"dense"`` (as released), ``"lowrank"``, or
    ``"grouped"``. The value stream is always dense.
    """
    gate_rank: Optional[int] = None
    """
    Bottleneck rank, required when ``gate_structure="lowrank"``. Note that ``rank >= d/2``
    saves no parameters, since ``2 * d * r >= d^2``.
    """
    gate_groups: Optional[int] = None
    """
    Number of diagonal blocks, required when ``gate_structure="grouped"``.
    """
    bias: bool = False
    """
    Whether projections and the convolution use biases. LFM2 uses ``False``.
    """
    use_fla: bool = True
    """
    Use the fused ``fla`` convolution kernel when available, falling back to
    :class:`torch.nn.Conv1d`.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of parameters this mixer will have once built.

        :param d_model: The model dimensionality.
        """
        params = d_model * d_model  # value_proj
        if self.gate_structure == "dense":
            params += 2 * d_model * d_model
        elif self.gate_structure == "lowrank":
            assert self.gate_rank is not None
            params += d_model * (2 * self.gate_rank)  # shared down-projection
            params += 2 * self.gate_rank * d_model  # two up-projections
        elif self.gate_structure == "grouped":
            assert self.gate_groups is not None
            params += 2 * (d_model * d_model // self.gate_groups)
        else:
            raise ValueError(f"unknown gate structure '{self.gate_structure}'")

        params += d_model * d_model  # out_proj
        params += self.kernel_size * d_model  # depthwise conv
        if self.bias:
            params += 2 * d_model + d_model  # in_proj gates + value, out_proj, conv
            params += d_model
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ShortConv:
        """
        Build the :class:`ShortConv` module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize parameters on, e.g. ``"cpu"``, ``"meta"``.
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return ShortConv(
            d_model=d_model,
            kernel_size=self.kernel_size,
            gate_structure=self.gate_structure,
            gate_rank=self.gate_rank,
            gate_groups=self.gate_groups,
            bias=self.bias,
            use_fla=self.use_fla,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


SequenceMixerConfig.register("short_conv")(ShortConvConfig)
