"""
LFM2-style *gated* short convolution, packaged so it can be dropped into a recurrent mixer.

WHY THIS IS A SEPARATE FILE AND A SEPARATE CLASS
    :class:`~olmo_core.nn.convolution.CausalConv1d` is depended on by
    :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`,
    :class:`~olmo_core.nn.attention.recurrent.KimiDeltaAttention` and
    :class:`~olmo_core.nn.attention.recurrent.KimiDeltaHouseholder`, and its ``activation``
    argument defaults to ``"silu"``. The LFM2 "LIV" block passes ``activation=None``, so a
    subclass that merely flipped that default would change the operator every existing caller
    gets. Everything here is additive: no existing module's behaviour moves.

WHAT THE OPERATOR IS
    Liquid AI's released block (``transformers`` v5.0.0rc1, ``Lfm2ShortConv``) is::

        B, C, v = in_proj(x)                    # three streams
        z       = depthwise_causal_conv(B * v)   # no bias, NO ACTIVATION
        out     = out_proj(C * z)

    Inside a recurrent mixer there is no ``out_proj`` to fold into -- the convolution's output
    feeds the delta-rule kernel directly. So the transplantable part is the pair of
    multiplicative gates around the convolution::

        out = post_gate(s) * conv(pre_gate(s) * u)

    where ``u`` is the stream being convolved (KDA's q, k or v) and ``s`` is whatever the gate
    reads. See :class:`GateSource` for the two choices.

.. warning::
    **A CONSTANT PER-CHANNEL GATE IS A VACUOUS REPARAMETERIZATION, NOT A NEW OPERATOR.**

    The convolution here is *depthwise*: channel ``c`` of the output depends only on channel
    ``c`` of the input, through that channel's own ``kernel_size`` taps. So a learned constant
    vector ``a`` applied as ``conv(a * u)`` is identically ``conv'(u)`` with
    ``w'[c] = a[c] * w[c]``, and a constant ``b`` applied as ``b * conv(u)`` is the same trick on
    the other side. **Both are absorbed exactly.** A "gate" of that form adds parameters, trains
    stably, and measures nothing -- it is the same function class as the plain convolution, so
    the honest expected effect is precisely zero and any observed difference is optimizer noise.

    The gates below are therefore **input-varying and nonlinear**, which is the whole content of
    the name "Linear Input-Varying". :func:`gate_is_absorbable` states the property that must
    hold, and ``test_gated_convolution.py`` asserts a real forward-pass difference against the
    plain convolution rather than trusting this comment.

THE INIT IS EXACTLY NEUTRAL AND STILL ALIVE, WHICH TOOK TWO TRIES
    The gate is ``2 * sigmoid(z)``, so at ``z = 0`` it is exactly ``1.0`` and the module computes
    a bit-identical result to an ungated convolution with the same weights. That is what makes the
    gated and plain arms an ablation rather than two unrelated models.

    Getting the pre-activation to zero **without killing the branch** is the part that is easy to
    get wrong. ``d/dz [2*sigmoid(z)] = 0.5`` at ``z = 0``, so the gate always passes gradient --
    but if the parameters *producing* ``z`` are arranged so their own gradients are identically
    zero, the branch is dead forever, the run trains stably, and it reports a clean replicable
    null.

    The first version of this module zeroed every gate parameter. That is correct for
    ``"depthwise"``, whose pre-activation is ``a[c] * u[b,t,c]`` so ``d/da`` carries the nonzero
    ``u``. It is **dead for ``"lowrank"``**, whose pre-activation is a product of two zeroed
    factors, giving both of them exactly zero gradient. ``test_gate_gradient_is_alive_at_init``
    caught it. :meth:`GatedCausalConv1d.init_gate_weights` now draws the shared down-projection
    and zeroes only the two up-projections -- the LoRA convention, which exists for this reason.

    One consequence: ``"lowrank"`` **consumes randomness** and ``"depthwise"`` does not, so a
    ``"lowrank"`` arm cannot share a random stream with the plain arm parameter-for-parameter.
    ``init_gate_weights`` returns whether it drew, so a caller can report that rather than assume
    it.

NOT HUGGING FACE CONVERTIBLE, DELIBERATELY
    ``nn/hf/convert.py`` maps ``attention.q_conv1d.weight``; a gated convolution's key is
    ``attention.q_conv1d.conv.weight`` and its gate parameters have no counterpart in any released
    architecture. So a gated checkpoint cannot round-trip to HF, and no mapping is added -- there is
    nothing on the other side to map to, and a fabricated entry would silently drop the gates. If a
    gated arm ever needs exporting, that is a new HF architecture, not a key rename.

MEMORY IS THE REAL COST, AND IT IS NOT IN THE PARAMETER COUNT
    Gating is nearly free in parameters and expensive in activation memory: each gate retains
    stream-sized tensors for the backward pass. At KDA's geometry (``d_model=2048``,
    ``n_heads=16``, ``head_dim=128``, ``expand_v=1.0``, so all three streams are 2048 channels),
    bf16, and **8192 tokens per rank** -- the microbatch KDA's throughput was actually measured at
    -- one stream tensor is 32 MiB, so :func:`gate_activation_bytes` gives

    =====================================  ==============================
    2 gates x 3 tensors x 32 MiB            192 MiB per convolution (eager)
    x 3 convolutions (q, k, v)              **576 MiB per layer**
    x 28 layers                             **15.75 GiB**
    =====================================  ==============================

    KDA's measured peak on ``gpu-8xa100`` (40 GiB cards) is **5.169 GiB**, so eager gating is
    roughly **4x the peak**. Compiled it is nearer 10.5 GiB, i.e. 3x. Either way this is not a
    rounding error: it fits on a 40 GiB card and would not fit at ``seq_len=32768``.

    **Size any run from a measured peak on the gated arm, not from the parameter table and not
    from this estimate.** The parameter delta for the depthwise gate is 12,288 per layer, about
    0.07%, and it invites exactly the wrong conclusion about cost. The estimate itself carries
    three assumptions (eager, bf16, no activation checkpointing) that each move it by 2x --
    :func:`gate_activation_bytes` documents them.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.nn.attention.flash_linear_attn_api import has_fla

__all__ = [
    "GatedCausalConv1d",
    "GateStructure",
    "GateSource",
    "gate_activation_bytes",
    "gate_param_count",
    "gate_is_absorbable",
]


GateStructure = Literal["depthwise", "lowrank"]
"""
How the two gates are parameterized.

``"lowrank"`` -- **the faithful one.**
    Both gates are projections of the *layer input* through a shared bottleneck:
    ``d_model -> gate_rank -> hidden_size``. This is LFM2's structure, where both gates come off
    one ``in_proj`` applied to ``x``, and it is also KDA's own ``f_proj``/``g_proj`` idiom. Costs
    ``d_model * gate_rank + 2 * gate_rank * hidden_size`` per convolution.

    The bottleneck is not a compromise made for this experiment: the P1 study measured that
    cutting an LFM2 gate to rank 128 at ``d_model=1024`` costs **nothing measurable** in held-out
    CE (``[-0.0097, +0.0020]`` nats, n=12 paired). A low-rank gate is the *established*
    parameterization, not a weakened one.

``"depthwise"`` -- **cheapest, NOT LIV's gate, and its pre-gate is a SiLU in disguise.**
    One scalar per channel per gate, applied to the stream: ``pre = 2*sigmoid(a * u)``. Costs
    ``2 * hidden_size`` per convolution, about 0.07% of a KDA layer's projections.

    .. warning::
        **THE PRE-GATE IS EXACTLY A PER-CHANNEL-TEMPERATURE SiLU, NOT A GATE.** Since
        ``silu(y) = y * sigmoid(y)``, substituting ``y = a*u`` gives the identity

        .. code-block:: text

            2 * sigmoid(a*u) * u  ==  (2/a) * silu(a*u)        for every a != 0

        verified to machine precision (``8.9e-16`` in float64). The amplitude ``2/a_c`` is
        **constant per channel**, so it folds straight into that channel's convolution taps. What
        the pre-gate actually buys is *one* number per channel: the slope of a SiLU, applied
        **before** the convolution instead of after, initialized at ``a = 0`` where it is the
        identity.

        So ``gated_conv=True`` with ``activation=None`` is **not activation-free** under this
        structure. It moves an activation from after the convolution to before it and makes its
        slope learnable. Only the *post* gate is a genuinely new term, because it reads position
        ``t`` while the convolution output mixes ``t-j`` -- that is 1 real degree of freedom per
        channel, not 2.

        Consequence for experiment design: contrasting ``"depthwise"`` against the shipped
        silu-after-conv arm varies the activation's position, its learnability, **and** the post
        gate, all at once. Isolating the gate needs an arm with **no** activation at all, which is
        what :attr:`~olmo_core.nn.attention.recurrent.KimiDeltaAttentionConfig.conv_activation`
        exists for.

        This gate also cannot mix channels or read anything but its own channel of its own stream,
        where LIV's gates are channel-mixing projections of the layer input.

        Use ``"depthwise"`` as a cheap floor or a mechanism control. Use ``"lowrank"`` for any
        claim about LIV-style gating -- it gates on ``x`` rather than on ``u``, so
        ``2*sigmoid(W h(x)) * u`` is not of the form ``f(u)*u`` and the identity above does not
        apply to it.

A third option, a full dense ``d_model -> hidden_size`` projection per gate, is deliberately
**not** implemented. At KDA's geometry (``d_model=2048``, three 2048-channel streams, two gates
each) it is ``2 * 3 * 2048^2 = 25,165,824`` parameters against the layer's own **17,887,376** --
it does not cost 60%, it costs **140.7%**, more than doubling the layer. That confounds the
mechanism under test with raw capacity, which is the exact error ``KDA/HANDOFF.md:587`` records
for the R sweep.

For reference at that geometry, per layer:

===================  ==============  =====================
gate                 parameters      share of the layer
===================  ==============  =====================
``"depthwise"``      12,288          **0.069%**
``"lowrank"`` r=128  2,359,296       **13.2%**
dense (not built)    25,165,824      140.7%
===================  ==============  =====================
"""

GateSource = Literal["stream", "input"]
"""
What the gate reads.

``"stream"``
    The post-projection stream being convolved (KDA's q, k or v). Required by
    ``"depthwise"``, whose parameters are per convolution channel.

``"input"``
    The mixer's input ``x``. This is what LFM2 does -- its ``in_proj`` produces all three streams
    from ``x`` -- and it is the only choice available to ``"lowrank"``.
"""


def gate_param_count(
    *,
    hidden_size: int,
    structure: GateStructure,
    d_model: Optional[int] = None,
    gate_rank: Optional[int] = None,
) -> int:
    """
    Parameters the two gates add to one convolution.

    Kept as a module-level function so a configuration's ``num_params`` and the module's actual
    ``state_dict`` can be checked against *the same* expression, and so the arithmetic can be
    tested without building anything on a GPU.

    :param hidden_size: The convolution's channel count.
    :param structure: The gate structure.
    :param d_model: The mixer's model dimension. Required for ``"lowrank"``.
    :param gate_rank: The bottleneck width. Required for ``"lowrank"``.

    :returns: The number of added parameters.

    :raises ValueError: If ``structure`` is unknown, or a required argument is missing.
    """
    if structure == "depthwise":
        return 2 * hidden_size
    if structure == "lowrank":
        if d_model is None or gate_rank is None:
            raise ValueError("'d_model' and 'gate_rank' are required when structure='lowrank'")
        if gate_rank <= 0:
            raise ValueError(f"'gate_rank' must be positive, got {gate_rank}")
        # One shared down-projection, then one up-projection per gate.
        return d_model * gate_rank + 2 * gate_rank * hidden_size
    raise ValueError(f"unknown gate structure '{structure}'")


def gate_activation_bytes(
    *,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    bytes_per_element: int = 2,
    tensors_per_gate: int = 3,
) -> int:
    """
    Extra activation bytes one gated convolution holds for its backward pass.

    The parameter delta is negligible and the *memory* delta is not, so this is reported alongside
    it rather than left implicit.

    .. important::
        **This is an estimate with three assumptions, each of which can move it by 2x. Size a run
        from a measured peak, not from this number.**

        *Eager, and this is the default.* Walking what autograd retains for
        ``post * conv(pre * u)`` with ``pre = 2*sigmoid(a*u)``: ``sigmoid`` saves its own output,
        the scalar multiply's result is retained as an operand, and the product ``pre * u`` is
        retained as the convolution's input. That is **3 stream-sized tensors per gate** in eager,
        which is the default here. Under :mod:`torch.compile` inductor fuses the
        ``sigmoid -> mul -> mul`` chain and the count drops to about 2, so the honest figure is
        compile-dependent rather than a fixed property of the operator.

        *dtype.* ``bytes_per_element=2`` assumes the gate chain runs in bf16. ``pre_scale * u`` is
        a parameter-times-activation product and ``mul`` type-promotes rather than being on
        autocast's cast list, so with a float32 parameter dtype and no ``param_dtype`` override the
        whole chain lands in **fp32** and this figure doubles.

        *Activation checkpointing.* With an ``ac_config`` set, the gate chain is recomputed in
        backward and costs approximately nothing at peak, making this figure wildly pessimistic.

    :param hidden_size: The convolution's channel count.
    :param batch_size: The batch size.
    :param seq_len: The sequence length.
    :param bytes_per_element: 2 for bf16/fp16, 4 for fp32.
    :param tensors_per_gate: Stream-sized tensors retained per gate. Defaults to 3, which is the
        eager count. Pass 2 for a compiled run, 1 for a recompute-in-backward implementation, 0
        for a fully fused kernel.

    :returns: The number of extra bytes.
    """
    per_tensor = batch_size * seq_len * hidden_size * bytes_per_element
    return 2 * tensors_per_gate * per_tensor


def gate_is_absorbable(structure: GateStructure) -> bool:
    """
    Whether this gate collapses into the depthwise convolution's weights.

    A depthwise convolution is per-channel, so any gate that is *constant across positions* can
    be folded into that channel's taps, making the gated module the same function class as the
    plain one. Such a gate cannot produce a real effect, and an experiment built on it would
    measure noise while looking healthy.

    Every structure here returns ``False``. The function exists so that the property is
    asserted by a test rather than argued in a comment, and so that a future structure has an
    obvious place to declare itself unsafe.

    :param structure: The gate structure.

    :returns: ``True`` if the gate is absorbable, and so scientifically vacuous.

    :raises ValueError: If ``structure`` is unknown.
    """
    if structure not in ("depthwise", "lowrank"):
        raise ValueError(f"unknown gate structure '{structure}'")
    # Both are nonlinear functions of a position-dependent input, so neither is a constant
    # per-channel rescale.
    return False


class GatedCausalConv1d(nn.Module):
    """
    A depthwise causal convolution wrapped in two multiplicative, input-varying gates.

    ``out = post_gate * conv(pre_gate * u)``, matching LFM2's LIV block, with the gate structure
    chosen for parameter economy. See the module docstring for the operator, the vacuity
    argument, and the memory cost.

    .. note::
        ``activation`` defaults to ``None``, the **opposite** of
        :class:`~olmo_core.nn.convolution.CausalConv1d`. LFM2's block has no activation anywhere
        in the convolution path; ``CausalConv1d`` applies ``silu`` inside the fused kernel.
        Passing ``activation="silu"`` here gives the gate-and-silu combination, which is the arm
        that separates "gating helps" from "removing silu helps".

    :param hidden_size: Number of channels. The convolution is depthwise, so input and output
        channel counts are equal.
    :param kernel_size: Number of convolution taps.
    :param gate_structure: ``"depthwise"`` or ``"lowrank"``. See :data:`GateStructure`.
    :param gate_source: ``"stream"`` or ``"input"``. See :data:`GateSource`. Leave ``None`` to
        derive it from ``gate_structure``, which is the only coherent choice in each case; passing
        a conflicting value **raises** rather than being silently overridden.
    :param d_model: The mixer's model dimension, required for ``"lowrank"``.
    :param gate_rank: The bottleneck width, required for ``"lowrank"``.
    :param bias: Whether the convolution has a bias. LFM2 uses ``False``.
    :param backend: Fused-kernel backend, passed through to ``fla``.
    :param activation: Activation applied inside the convolution. ``None`` matches LFM2.
    :param use_fla: Use the fused ``fla`` kernel when available and the input is on CUDA. The
        fallback is a plain :class:`torch.nn.Conv1d`, which runs on CPU and is what makes this
        module testable without a GPU.
    :param dtype: Parameter dtype.
    :param init_device: Device to initialize parameters on, e.g. ``"cpu"``, ``"meta"``.

    :raises ValueError: If the gate structure or source is unknown, or a required argument for
        the chosen structure is missing.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        kernel_size: int,
        gate_structure: GateStructure = "depthwise",
        gate_source: Optional[GateSource] = None,
        d_model: Optional[int] = None,
        gate_rank: Optional[int] = None,
        bias: bool = False,
        backend: Literal["triton", "cuda"] = "triton",
        activation: Optional[Literal["silu", "swish"]] = None,
        use_fla: bool = True,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        if gate_structure not in ("depthwise", "lowrank"):
            raise ValueError(f"unknown gate structure '{gate_structure}'")
        # Derived from the structure when not given, because only one value is coherent with each.
        implied: GateSource = "stream" if gate_structure == "depthwise" else "input"
        if gate_source is None:
            gate_source = implied
        elif gate_source not in ("stream", "input"):
            raise ValueError(f"unknown gate source '{gate_source}'")
        # VALIDATED HERE, because the fused kernel does NOT validate it. fla's 'causal_conv1d'
        # applies swish only when the activation string is exactly 'swish' or 'silu' and otherwise
        # runs activation-free WITHOUT ERROR. So "Silu", "SILU" or "gelu" raises on the reference
        # path (which is what a CPU test exercises) and silently computes the activation-free
        # operator on GPU -- collapsing the gated-silu arm onto the gated arm while the config, the
        # logs and the arm name all say otherwise.
        if activation not in (None, "silu", "swish"):
            raise ValueError(
                f"unsupported activation {activation!r}; use None, 'silu' or 'swish'. "
                "The fused kernel matches these exactly and silently ignores anything else."
            )
        # A contradiction, rather than a silent override. 'gate_source' is determined by
        # 'gate_structure' (depthwise gates are per-channel on the stream; lowrank gates project
        # the layer input), so accepting a conflicting value and then overwriting it would let a
        # config request one operator and get another.
        if gate_structure == "depthwise" and gate_source != "stream":
            raise ValueError("gate_structure='depthwise' requires gate_source='stream'")
        if gate_structure == "lowrank" and gate_source not in ("input",):
            raise ValueError("gate_structure='lowrank' requires gate_source='input'")

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.gate_structure: GateStructure = gate_structure
        self.backend = backend
        self.activation = activation
        self.use_fla = use_fla
        self.cp_enabled = False
        self._cp_channel_slice: Optional[slice] = None

        kwargs = {"dtype": dtype, "device": init_device}

        if gate_structure == "depthwise":
            # Per-channel scale on the stream, so the gate is defined on the convolution's own
            # channels and 'input' would be a dimension mismatch whenever key_dim != d_model.
            self.gate_source: GateSource = "stream"
            self.d_model = None
            self.gate_rank = None
            self.pre_scale = nn.Parameter(torch.zeros(hidden_size, **kwargs))  # type: ignore[arg-type]
            self.post_scale = nn.Parameter(torch.zeros(hidden_size, **kwargs))  # type: ignore[arg-type]
        else:
            if d_model is None or gate_rank is None:
                raise ValueError("'d_model' and 'gate_rank' are required when structure='lowrank'")
            if gate_rank <= 0:
                raise ValueError(f"'gate_rank' must be positive, got {gate_rank}")
            # A projection-shaped gate must read the mixer input to be faithful to LFM2, where
            # both gates come off the same 'in_proj' applied to x.
            self.gate_source = "input"
            self.d_model = d_model
            self.gate_rank = gate_rank
            self.gate_down = nn.Linear(d_model, gate_rank, bias=False, **kwargs)  # type: ignore[arg-type]
            self.gate_up_pre = nn.Linear(gate_rank, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
            self.gate_up_post = nn.Linear(gate_rank, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]

        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            **kwargs,  # type: ignore[arg-type]
        )

    def num_gate_params(self) -> int:
        """
        Parameters the gates add, from :func:`gate_param_count`.

        :returns: The number of gate parameters.
        """
        return gate_param_count(
            hidden_size=self.hidden_size,
            structure=self.gate_structure,
            d_model=self.d_model,
            gate_rank=self.gate_rank,
        )

    def _gates(
        self, u: torch.Tensor, gate_input: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The ``(pre, post)`` gate values, each broadcastable onto ``u``.

        ``2 * sigmoid(z)`` rather than ``sigmoid(z)`` so that the zero-initialized gate is
        exactly ``1.0`` and the module reproduces the ungated convolution bit-for-bit at step 0.
        """
        if self.gate_structure == "depthwise":
            pre_scale, post_scale = self.pre_scale, self.post_scale
            if self.cp_enabled:
                assert self._cp_channel_slice is not None
                pre_scale = pre_scale[self._cp_channel_slice]
                post_scale = post_scale[self._cp_channel_slice]
            return (
                2.0 * torch.sigmoid(pre_scale * u),
                2.0 * torch.sigmoid(post_scale * u),
            )

        if gate_input is None:
            raise RuntimeError(
                "structure='lowrank' reads the mixer input, so 'gate_input' must be passed. "
                "Passing None would silently gate on the stream instead, which is a different "
                "operator."
            )
        h = self.gate_down(gate_input)
        pre_w, post_w = self.gate_up_pre.weight, self.gate_up_post.weight
        if self.cp_enabled:
            assert self._cp_channel_slice is not None
            pre_w = pre_w[self._cp_channel_slice]
            post_w = post_w[self._cp_channel_slice]
        return (
            2.0 * torch.sigmoid(F.linear(h, pre_w)),
            2.0 * torch.sigmoid(F.linear(h, post_w)),
        )

    def _conv(self, u: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> torch.Tensor:
        weight = self.conv.weight
        bias = self.conv.bias
        if self.cp_enabled:
            assert self._cp_channel_slice is not None
            weight = weight[self._cp_channel_slice]
            if bias is not None:
                bias = bias[self._cp_channel_slice]

        if self.use_fla and has_fla() and u.is_cuda:
            from olmo_core.nn.attention.flash_linear_attn_api import (
                dispatch_causal_conv1d,
            )

            out = dispatch_causal_conv1d(
                x=u,
                weight=weight.squeeze(1),
                bias=bias,
                activation=self.activation,
                backend=self.backend,
                cu_seqlens=cu_seqlens,
            )
            return out[0] if isinstance(out, tuple) else out

        # Reference path. Convolving each document separately is not a nicety: a 'kernel_size'
        # filter that reads across a document boundary is a different operator, and at a ~622
        # token median document length a 4096-token sequence holds several documents.
        if cu_seqlens is not None:
            if u.shape[0] != 1:
                raise RuntimeError("'cu_seqlens' requires batch_size == 1")
            bounds = cu_seqlens.tolist()
            segments = [
                self._conv_dense(u[:, s:e], weight, bias)
                for s, e in zip(bounds[:-1], bounds[1:])
                if e > s
            ]
            return torch.cat(segments, dim=1)
        return self._conv_dense(u, weight, bias)

    def _conv_dense(
        self, u: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        seq_len = u.shape[1]
        z = F.conv1d(
            u.transpose(-1, -2),
            weight,
            bias,
            padding=self.kernel_size - 1,
            groups=weight.shape[0],
        )[..., :seq_len]
        z = z.transpose(-1, -2)
        if self.activation in ("silu", "swish"):
            z = F.silu(z)
        elif self.activation is not None:
            raise RuntimeError(f"unsupported activation '{self.activation}'")
        return z

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        gate_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: The stream to convolve, shape ``(batch_size, seq_len, hidden_size)``. Named
            ``x`` to match :meth:`~olmo_core.nn.convolution.CausalConv1d.forward`, so this
            module is call-compatible at every existing call site.
        :param cu_seqlens: Cumulative sequence lengths, shape ``(num_seqs + 1,)``. Requires
            ``batch_size == 1``.
        :param gate_input: The mixer input, shape ``(batch_size, seq_len, d_model)``. Required
            when ``gate_structure="lowrank"`` and ignored otherwise.

        :returns: Output of shape ``(batch_size, seq_len, hidden_size)``.

        :raises RuntimeError: If ``gate_input`` is needed and not given, or ``cu_seqlens`` is
            given with a batch size above 1.
        """
        pre, post = self._gates(x, gate_input)
        return post * self._conv(pre * x, cu_seqlens)

    def apply_cp(self, cp_mesh: DeviceMesh):
        """
        Configure for Ulysses-style (channel-parallel) context parallelism.

        Mirrors :meth:`~olmo_core.nn.convolution.CausalConv1d.apply_cp`: keep the full
        parameters on every rank and slice to the local ``C/CP`` channels in the forward pass,
        rather than sharding them, which would conflict with FSDP.

        The gates are sliced on the same channel axis as the convolution. A gate left unsliced
        would broadcast the wrong channels onto the local shard and produce a wrong result
        rather than an error.

        :param cp_mesh: The context parallel device mesh.

        :raises NotImplementedError: If ``gate_structure="lowrank"``, whose shared bottleneck
            reads the full ``d_model``; splitting it correctly needs a decision about whether
            the down-projection is replicated or sharded, and guessing would be silently wrong.
        """
        if cp_mesh.size() == 1:
            return
        if self.gate_structure == "lowrank":
            raise NotImplementedError(
                "Context parallelism is not implemented for gate_structure='lowrank'"
            )
        if self.hidden_size % cp_mesh.size() != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by the CP world size "
                f"({cp_mesh.size()})"
            )
        local_channels = self.hidden_size // cp_mesh.size()
        start = cp_mesh.get_local_rank() * local_channels
        self._cp_channel_slice = slice(start, start + local_channels)
        self.cp_enabled = True

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """
        Put the gates in the neutral state, so the module is usable before ``init_weights`` runs.

        :meth:`~olmo_core.nn.transformer.model.Transformer.init_weights` calls ``to_empty(device)``
        and then sweeps every submodule's ``reset_parameters``. Without this method the gate
        parameters would be **uninitialized memory** for anything that builds the module and never
        calls a mixer's ``init_weights`` -- a probe, a benchmark, a direct instantiation. The
        ``torch.zeros`` in ``__init__`` does not protect against that, because ``to_empty``
        discards it.

        That is not a hypothetical: ``short_conv.py`` documents having shipped exactly this on the
        MQAR calibration, where a grouped arm ran at a fraction of dense's activation scale because
        the probe built the module directly. Uninitialized memory is often all zeros, which makes a
        broken module look merely inert.

        Zeroing is also the *correct* neutral state here, so this is not a placeholder: the gate
        becomes exactly ``1.0``. The convolution's own ``reset_parameters`` is inherited from
        :class:`torch.nn.Conv1d` and left alone.
        """
        self.conv.reset_parameters()
        if self.gate_structure == "depthwise":
            self.pre_scale.zero_()
            self.post_scale.zero_()
        else:
            self.gate_down.reset_parameters()
            self.gate_up_pre.weight.zero_()
            self.gate_up_post.weight.zero_()

    @torch.no_grad()
    def init_gate_weights(
        self, *, std: float = 0.02, generator: Optional[torch.Generator] = None
    ) -> bool:
        """
        Initialize the gates so the module starts as an ungated convolution, but not dead.

        Both structures put the gate's **pre-activation** at exactly zero, so
        ``2 * sigmoid(0) = 1`` and the module reproduces the ungated convolution bit-for-bit at
        step 0. How they get there differs, and the difference is the whole point:

        ``"depthwise"``
            Both scales go to zero. The pre-activation is ``a[c] * u[b,t,c]``, so
            ``d/da = 0.5 * sum_bt u * upstream`` -- **nonzero at ``a = 0``**, because ``u`` is
            not zero. Alive on step 1. Consumes no randomness.

        ``"lowrank"``
            The shared down-projection is **drawn**, and only the two up-projections are zeroed.
            Zeroing both factors of the product ``W_up @ (W_down @ x)`` would make *both*
            gradients identically zero -- the branch would be dead forever while the run trained
            stably and reported a clean, replicable null. This is the LoRA convention (draw A,
            zero B) and it exists for exactly that reason. Gradient reaches ``W_up`` on step 1
            because ``h = W_down @ x`` is nonzero, and reaches ``W_down`` from step 2 once
            ``W_up`` has moved. ``test_gate_gradient_is_alive_at_init`` and
            ``test_lowrank_down_projection_wakes_up_after_one_step`` assert both halves.

        The convolution's own weight is **not** touched here -- it is drawn by the owning mixer,
        exactly as the ungated convolution's is, so the two arms share that draw.

        :param std: Standard deviation for the drawn down-projection. Ignored by ``"depthwise"``.
        :param generator: The random generator. Ignored by ``"depthwise"``, which draws nothing.

        :returns: ``True`` if this call consumed randomness from ``generator``. The caller needs
            to know: a gated arm that advanced the shared generator would change every *later*
            parameter in the model, and that confound is invisible in a loss curve. See
            :func:`~olmo_core.nn.attention.recurrent._init_short_conv`.
        """
        from olmo_core.nn.transformer.init import _apply_init, init_linear

        def _zero(t: torch.Tensor) -> None:
            t.zero_()

        # Routed through '_apply_init' because under FSDP every parameter is a DTensor and an
        # in-place 'zero_()' on the parameter object has no guaranteed sharding strategy. An
        # indexed assignment of this shape killed run_019fbf9f, and a CPU test cannot catch it
        # because a single-process build never produces a DTensor.
        if self.gate_structure == "depthwise":
            for p in (self.pre_scale, self.post_scale):
                _apply_init(_zero, p)
            return False

        init_linear(self.gate_down, std=std, generator=generator)
        for p in (self.gate_up_pre.weight, self.gate_up_post.weight):
            _apply_init(_zero, p)
        return True
