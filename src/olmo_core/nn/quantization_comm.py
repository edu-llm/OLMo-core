"""
Ternary-compressed FSDP2 all-gather: send 2-bit trits + per-row ``alpha`` instead of bf16.

**EVERYTHING IN THIS MODULE IS UNTESTED ON HARDWARE.** It was written with no torch import and
no distributed run available. Nothing here has executed. Do not enable it on a funded run until
``src/test/nn/quantization_comm_test.py`` passes.

Why this exists
---------------
``nn/quantization.py``'s header states the cost model correctly: ternary QAT is **not** cheaper
to train, because ``twn_quantize_ste`` produces a *dequantized* ``alpha * sign(W) * 1[|W| >
delta]`` tensor in the compute dtype which then feeds an ordinary ``F.linear``/``bmm``. That
quantization happens **after** ``get_local_tensor``, i.e. **post-gather**. So FSDP2 all-gathers
bf16 shards and the ternary structure is discarded before it can save a single byte on the wire.

There is exactly one place where ternary could pay during *training*: the all-gather itself. Each
weight is one of ``{-1, 0, +1}`` times a per-output-row ``alpha``. At 2 bits per trit that is a
**8x** reduction in all-gather bytes, with a tiny ``alpha`` vector alongside. The sibling ``moe/``
track measured MFU **36.6% on NVSwitch vs 4.66% on PCIe** at E=8/k=2 -- an ~8x gap that is
essentially all communication -- so on a comms-bound fabric this is a first-order win. On NVSwitch
it may be worth nothing. Both are valid findings.

The correctness question, which outranks the bytes
--------------------------------------------------
Moving quantization from post-gather to pre-gather **can** change what the model computes,
because ``alpha`` and ``delta`` are per-output-row statistics over the **full** ``in_dim``. If a
rank holds only part of a row, its local ``mean|W|`` is not the global ``mean|W|`` -- a silently
different quantizer that trains happily. This module therefore refuses rather than approximates.

The algebra, worked out per tensor family (see :class:`TernaryCommSpec`):

* **Attention / dense projections** -- ``nn.Linear.weight`` is ``(out_features, in_features)``
  and TWN reduces over ``in_features`` (``in_dim=-1``). FSDP2's default placement is
  ``Shard(0)`` (``torch/distributed/fsdp/_fully_shard/_fsdp_param.py:274`` at v2.9.0), which
  cuts ``out_features``. **Every row stays whole on one rank, at every world size.**
  Unconditionally exact.

* **Stacked MoE expert weights** -- these are stored **flattened to 2-D**, ``(E * a, b)``
  (``nn/moe/mlp.py:183-208``: *"these parameters need to have a large enough first dimension ...
  so we flatten the first 2 dimensions"*), and viewed back to ``(E, a, b)`` inside ``forward``.
  For ``MoEMLP`` all three reduce over the **3-D axis 1**, which is ``a`` -- and ``a`` is
  **folded inside flat axis 0**, the very axis FSDP cuts. So a TWN row for ``w1`` is the column
  segment ``w[e, :, h]``: ``a`` consecutive flat rows at one column. It survives only if the
  shard boundary lands on an **expert** boundary.

  ``nn/moe/mlp.py:244-250`` already states this invariant for expert parallelism: *"the
  reduction axis is interleaved inside flat axis 0, so a cut at a non-expert boundary would
  split rows. The divisibility guard is what makes post-shard quantization equivalent to
  pre-shard, not the shard axis alone."* Expert parallelism gets that guarantee from
  ``_shard_experts``'s ``num_experts % num_shards == 0`` check (``mlp.py:113-116``). **FSDP2 has
  no such guard** -- it calls ``torch.chunk(param_data, world_size, dim=0)``
  (``_fsdp_common.py:113-119`` at v2.9.0) on ``E * a``, and nothing forces that cut to be
  expert-aligned.

  So for these tensors exactness is **conditional**, and the condition is
  ``local_flat_dim0 % prod(trailing_shape[:-1]) == 0`` -- equivalently, under even chunking,
  ``world_size`` divides ``num_experts``. :meth:`TernaryCommSpec.assert_exact` enforces it and
  **raises** otherwise. It is never approximated, never all-reduced into agreement, and never
  waved through as "close enough".

* **DroplessMoEMLP w1/w3** reduce over 3-D axis **2** (``gmm`` with ``trans_b=True``), which is
  the flat param's own axis 1 -- always whole. Unconditionally exact. Its ``w2`` reduces over
  axis 1 and is conditional like ``MoEMLP``'s.

What stays full precision, unchanged
------------------------------------
Embeddings, ``lm_head``, **the router**, and all norms. This module only ever attaches itself to
tensors that ``nn/quantization.py`` already quantizes, so ``audit_quantization``'s carve-out
assertion is untouched by construction -- there is no path here that can reach a router.

The one residual empirical question
-----------------------------------
The *algebra* above is exact. Bitwise identity additionally requires that a float32 reduction
over the same elements in the same order returns the same bits when the tensor's **outer** size
differs (``(E_local, a, b)`` vs ``(E, a, b)``). That is true for any sane reduction kernel and is
what the test suite checks; it is not something this module can prove by reading code. **That is
why the flag defaults off and why the hardware test is the gate.**
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from ..exceptions import OLMoConfigurationError
from .quantization import QuantLinear, twn_threshold_and_scale

__all__ = [
    "TRITS_PER_BYTE",
    "TernaryCommSpec",
    "TernaryCommTensor",
    "pack_trits",
    "unpack_trits",
    "trits_and_alpha",
    "reconstruct_from_trits",
    "apply_ternary_comm",
    "ternary_comm_bytes_report",
]


TRITS_PER_BYTE: int = 4
"""
Trits packed per byte: **2 bits each**, not 1.58.

Chosen deliberately over the information-theoretic 5-trits-per-byte (``log2(3) = 1.585`` bits,
``3**5 = 243 <= 256``) packing, which would give ~10x instead of 8x. The reasons, in order:

1. **2-bit is a pure shift-and-mask.** Pack is three shifts and three ors; unpack is one
   broadcasted shift and one and. 1.58-bit needs base-3 modular arithmetic -- five
   divisions/remainders per byte on unpack -- on the critical path of every all-gather, which is
   the thing being optimized. The extra 25% of bytes is very likely cheaper than the extra
   arithmetic, and if it is not, that is a measurement, not a guess.
2. **Byte alignment survives sharding.** A group of 4 divides evenly into every real hidden
   size, ``d_model`` and expert-flattened row length in the ladder (all multiples of 128). A
   group of 5 does not divide 1536, 1024 or 384, so every shard boundary would need a partial
   group -- and a partial group at a shard boundary is exactly where a silent off-by-one becomes
   a wrong weight rather than a crash.
3. **8x already dominates the decision.** The question this lane answers is whether cutting
   all-gather traffic by ~an order of magnitude helps at all. 8x vs 10x does not change that
   answer; if 8x is a loss, 10x is a loss too.

Revisit only if a measurement shows the all-gather is still the bottleneck **after** 8x.
"""

_CODE_OFFSET: int = 1
"""Trit ``t`` is stored as the 2-bit code ``t + 1``, so ``{-1, 0, +1} -> {0, 1, 2}``, and ``3``
is never emitted. An unpacked ``3`` therefore means corruption, which
:func:`unpack_trits` can and does assert on."""


# ===================================================================================
# Packing
# ===================================================================================


def pack_trits(trits: torch.Tensor) -> torch.Tensor:
    """
    Pack a flat tensor of trits in ``{-1, 0, +1}`` into 2-bit codes, 4 per ``uint8``.

    :param trits: Any integer or floating tensor whose values are exactly ``-1``, ``0`` or
        ``+1``. Flattened before packing, so the caller owns the layout convention.
    :returns: A 1-D ``uint8`` tensor of ``numel // 4`` bytes, little-end first: element ``4k+j``
        occupies bits ``2j..2j+1`` of byte ``k``.

    :raises ValueError: if ``numel`` is not a multiple of :data:`TRITS_PER_BYTE`. This refuses
        rather than padding, because a pad at a shard boundary is indistinguishable from a real
        zero trit on unpack and would silently change ``alpha`` on the next round trip.
    """
    flat = trits.reshape(-1)
    if flat.numel() % TRITS_PER_BYTE != 0:
        raise ValueError(
            f"pack_trits needs numel divisible by {TRITS_PER_BYTE}, got {flat.numel()}. "
            "Refusing to pad: a pad byte is indistinguishable from a real zero trit on unpack."
        )
    codes = (flat.to(torch.int16) + _CODE_OFFSET).to(torch.uint8)
    groups = codes.view(-1, TRITS_PER_BYTE)
    packed = groups[:, 0].clone()
    for j in range(1, TRITS_PER_BYTE):
        packed |= groups[:, j] << (2 * j)
    return packed.contiguous()


def unpack_trits(packed: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Inverse of :func:`pack_trits`.

    :param packed: 1-D ``uint8`` tensor as produced by :func:`pack_trits`.
    :param dtype: Output dtype. Defaults to float32 so the subsequent ``trit * alpha`` product
        happens in the same precision the post-gather path uses.
    :returns: A flat tensor of ``4 * packed.numel()`` values in ``{-1.0, 0.0, +1.0}``.

    :raises ValueError: if any 2-bit field decodes to ``3``, which :data:`_CODE_OFFSET` never
        emits. That is the corruption tripwire: it makes a mangled byte stream **fail** instead
        of quietly producing a weight of ``2 * alpha``.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"unpack_trits expects uint8, got {packed.dtype}")
    shifts = torch.arange(0, 2 * TRITS_PER_BYTE, 2, device=packed.device, dtype=torch.uint8).view(
        1, TRITS_PER_BYTE
    )
    codes = (packed.reshape(-1, 1) >> shifts) & 3
    if bool((codes == 3).any()):
        raise ValueError(
            "unpack_trits decoded the reserved 2-bit code 3, which pack_trits never emits. "
            "The packed buffer is corrupt or was not produced by pack_trits."
        )
    return codes.reshape(-1).to(dtype) - float(_CODE_OFFSET)


# ===================================================================================
# The eligibility / exactness spec
# ===================================================================================


@dataclass(frozen=True)
class TernaryCommSpec:
    """
    How one eligible parameter is laid out, and whether pre-gather TWN is exact for it.

    A parameter reaching this path is always **2-D as stored**, ``(flat_dim0, trailing[-1])``,
    and is *interpreted* as ``(-1, *trailing)``. ``in_dim`` indexes that interpreted shape.

    ==============================  ===================  ==========  =======  ================
    tensor                          stored shape         trailing    in_dim   exact?
    ==============================  ===================  ==========  =======  ================
    ``nn.Linear.weight``            ``(out, in)``        ``(in,)``   1        always
    ``MoEMLP.w1`` / ``.w3``         ``(E*d, h)``         ``(d, h)``  1        iff expert-aligned
    ``MoEMLP.w2``                   ``(E*h, d)``         ``(h, d)``  1        iff expert-aligned
    ``DroplessMoEMLP.w1`` / ``w3``  ``(E*h, d)``         ``(h, d)``  2        always
    ``DroplessMoEMLP.w2``           ``(E*h, d)``         ``(h, d)``  1        iff expert-aligned
    ==============================  ===================  ==========  =======  ================

    The ``in_dim`` column is copied from the orientation table in
    :func:`~olmo_core.nn.quantization.twn_quantize` and the call sites in ``nn/moe/mlp.py:251``
    and ``:404``. **It must not be re-derived from the shapes** -- ``DroplessMoEMLP``'s ``w1``
    and ``w2`` have *identical shapes* and *different* ``in_dim``, because ``gmm`` is called with
    ``trans_b=True`` for one and not the other. Getting it wrong yields a per-input-row alpha: a
    different quantizer that trains happily.
    """

    trailing: Tuple[int, ...]
    """The interpreted trailing dimensions. The stored parameter is ``(-1, trailing[-1])`` and is
    viewed as ``(-1, *trailing)``."""

    in_dim: int
    """Index into the interpreted shape ``(-1, *trailing)`` of the axis TWN reduces over. Always
    ``>= 1``: axis 0 is the sharded/concatenated axis and is never a reduction axis."""

    @property
    def fold(self) -> int:
        """
        Product of the interpreted axes that are **folded inside stored flat axis 0**.

        For ``(out, in)`` this is 1 -- nothing is folded. For a flattened expert weight
        ``(E*a, b)`` viewed ``(E, a, b)`` it is ``a``. This is the granularity FSDP's ``Shard(0)``
        cut must respect, and it is the whole content of the exactness condition.
        """
        return int(math.prod(self.trailing[:-1])) if len(self.trailing) > 1 else 1

    @property
    def reduces_over_folded_axis(self) -> bool:
        """
        Does TWN reduce over an axis folded inside flat axis 0?

        ``False`` means the reduction axis is the stored parameter's own last axis, so every row
        is contiguous within one stored row and **no** shard of ``Shard(0)`` can split it --
        unconditionally exact. ``True`` means the row spans ``fold`` consecutive stored rows and
        exactness depends on where the cut falls.
        """
        return self.in_dim < len(self.trailing)

    def interpreted_shape(self, flat_dim0: int) -> Tuple[int, ...]:
        """The ``(-1, *trailing)`` view of a stored tensor with ``flat_dim0`` rows."""
        if flat_dim0 % self.fold != 0:
            raise ValueError(f"flat dim0 {flat_dim0} is not a multiple of fold {self.fold}")
        return (flat_dim0 // self.fold, *self.trailing)

    def assert_exact(self, *, local_flat_dim0: int, param_name: str, world_size: int) -> None:
        """
        Refuse unless pre-gather TWN on a shard of this size is **exactly** post-gather TWN.

        This is a hard gate, not a warning, and it is deliberately not rescuable. The two ways
        out of a failure here would be (a) an extra all-reduce of row statistics or (b) accepting
        the deviation -- and both **change the quantizer**, which changes the model. Maple
        faithfulness outranks all-gather bytes: a transport optimization that alters the weights
        is not a transport optimization.

        :raises OLMoConfigurationError: if the shard can split a TWN row.
        """
        if not self.reduces_over_folded_axis:
            return
        if local_flat_dim0 % self.fold != 0:
            raise OLMoConfigurationError(
                f"ternary-compressed all-gather would change the quantizer for {param_name!r}: "
                f"TWN reduces over interpreted axis {self.in_dim}, whose extent {self.fold} is "
                f"folded inside stored flat axis 0, and this rank's shard holds "
                f"{local_flat_dim0} rows, which is NOT a multiple of {self.fold}. Each rank "
                f"would derive alpha and delta from a FRACTION of every output row -- a "
                f"different quantizer that trains without error.\n"
                f"Cause: FSDP2 chunks flat axis 0 into {world_size} pieces with torch.chunk and "
                f"has no expert-alignment guard (unlike MoEMLPBase._shard_experts, which "
                f"enforces num_experts % num_shards == 0). Fix the world size so it divides the "
                f"expert count, or leave ternary_comm off. This will NOT be approximated."
            )

    def alpha_shape(self, flat_dim0: int) -> Tuple[int, ...]:
        """Shape of the per-row ``alpha`` for a stored tensor with ``flat_dim0`` rows."""
        shape = list(self.interpreted_shape(flat_dim0))
        shape[self.in_dim] = 1
        return tuple(shape)


def trits_and_alpha(w: torch.Tensor, spec: TernaryCommSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a stored 2-D weight into its trits and its per-row ``alpha``.

    This is :func:`~olmo_core.nn.quantization.twn_quantize` **factored**, not reimplemented: it
    calls the same :func:`~olmo_core.nn.quantization.twn_threshold_and_scale` and forms the same
    ``sign(w) * (|w| > delta)``. The only difference is that it returns the two factors instead
    of their product, so the product can be deferred until after the wire.

    :returns: ``(trits, alpha)`` -- ``trits`` float32 in ``{-1, 0, +1}`` with the interpreted
        shape, ``alpha`` float32 with :meth:`TernaryCommSpec.alpha_shape`.
    """
    viewed = w.reshape(spec.interpreted_shape(w.shape[0]))
    delta, alpha = twn_threshold_and_scale(viewed, in_dim=spec.in_dim)
    w32 = viewed.detach().to(torch.float32)
    trits = torch.sign(w32) * (w32.abs() > delta)
    return trits, alpha


def reconstruct_from_trits(
    trits: torch.Tensor, alpha: torch.Tensor, *, out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Rebuild the dequantized weight from its factors.

    ``trits * alpha`` in float32 then cast is **bitwise** what
    :func:`~olmo_core.nn.quantization.twn_quantize` computes as
    ``sign(w32) * (|w32| > delta) * alpha`` then cast, because ``sign * mask`` *is* ``trits``
    elementwise and float32 multiplication by ``+-1`` and ``0`` is exact. So the deferral costs
    nothing numerically -- which is the property the whole scheme rests on.
    """
    return (trits.to(torch.float32) * alpha).to(out_dtype)


# ===================================================================================
# The tensor subclass carrying the FSDP2 all-gather extension
# ===================================================================================

_OPS_TO_PRESERVE_SUBCLASS = {
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.new_empty.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.narrow.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.copy_.default,
    torch.ops.aten.zero_.default,
    torch.ops.aten.t.default,
}
"""Ops after which the result must still be a :class:`TernaryCommTensor`.

FSDP2 reaches the sharded parameter through ``torch.chunk`` -> ``new_zeros`` -> ``narrow`` ->
``copy_`` -> ``view`` (``_fsdp_param.py:362-385`` at v2.9.0), and it locates the hooks with
``hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather")``
(``_fsdp_param.py:678-680``). If any op in that chain drops the subclass, the hooks become
invisible and FSDP silently falls back to the plain bf16 all-gather -- a run that reports itself
as compressed and is not. Mirrors ``torchao/float8/fsdp_utils.py``'s
``_ops_to_preserve_subclass``, which exists for the same reason."""


class TernaryCommTensor(torch.Tensor):
    """
    A latent weight that all-gathers as packed trits plus ``alpha`` instead of as bf16.

    **UNTESTED.** Written against torch **2.9.0** (the pin asserted in ``.edullm/Dockerfile:292``),
    reading the installed source at that tag. The extension point is *private and unversioned*:
    nothing in ``torch.distributed.fsdp``'s public API mentions these hooks, they are discovered
    purely by ``hasattr``, and the signature already has a v1/v2 fork carrying a
    ``"keep for BC for now"`` comment (``_fsdp_param.py:690``). Treat a torch bump as
    breaking-by-default for this file.

    Verified call contract at v2.9.0, quoted from
    ``torch/distributed/fsdp/_fully_shard/_fsdp_param.py``:

    * ``_init_extensions`` (**line 422-433**) asserts *both* hooks exist if either does.
    * ``all_gather_inputs`` (**line 674-737**) inspects the signature and accepts **1 or 5**
      parameters, calling the 5-parameter form as
      ``fsdp_pre_all_gather(shard_mesh_from_root, _orig_size, _contiguous_orig_stride,
      _module_info.module, mp_policy)`` (**line 711-717**).
    * ``init_unsharded_param`` (**line 450-496**) calls ``fsdp_post_all_gather(all_gather_outputs,
      metadata, param_dtype)`` on the first gather and again with ``out=self._unsharded_param``
      on every later one (**line 475-480**).

    Two traps that are load-bearing and both silent:

    * ``all_gather_inputs`` and ``init_unsharded_param`` are both guarded by
      ``not compiled_autograd_enabled()`` (**lines 678, 466, 484**). Under compiled autograd the
      hooks are **skipped entirely** and FSDP gathers the raw latent weight -- which, since this
      scheme moves quantization out of the forward, would train the model **unquantized** while
      every log line looks ordinary. :func:`apply_ternary_comm` refuses in that case.
    * ``all_gather_inputs`` raises ``NotImplementedError`` for
      ``ShardedState.SHARDED_POST_FORWARD`` when the hooks are present (**line 744-748**). That
      state is only reached when ``reshard_after_forward`` is an **int** (reshard to a smaller
      mesh); OLMo-core passes ``True``/``False`` (``nn/transformer/model.py:932``), so it is
      unreachable on our path -- but a future ``reshard_after_forward=<int>`` would hit it.

    :param tensor: The latent full-precision weight. Kept intact: this class changes *transport*,
        not storage, so the master weight and the optimizer state are untouched.
    :param spec: How to interpret and reduce the tensor. See :class:`TernaryCommSpec`.
    """

    _tensor: torch.Tensor
    _spec: TernaryCommSpec

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, spec: TernaryCommSpec):
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, spec: TernaryCommSpec):
        self._tensor = tensor
        self._spec = spec

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"TernaryCommTensor(shape={tuple(self.shape)}, spec={self._spec})"

    # -- subclass plumbing ----------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        spec: Optional[TernaryCommSpec] = None

        def unwrap(t: "TernaryCommTensor") -> torch.Tensor:
            nonlocal spec
            if spec is None:
                spec = t._spec
            elif spec != t._spec:
                raise AssertionError(f"mixed TernaryCommSpec in one op: {spec} vs {t._spec}")
            return t._tensor

        args, kwargs = pytree.tree_map_only(cls, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _OPS_TO_PRESERVE_SUBCLASS:
            return out
        assert spec is not None
        held = spec
        return pytree.tree_map_only(torch.Tensor, lambda x: cls(x, held), out)

    def __tensor_flatten__(self):
        return ["_tensor"], {"spec": self._spec}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        del outer_size, outer_stride
        return TernaryCommTensor(inner_tensors["_tensor"], flatten_spec["spec"])

    # -- the extension point --------------------------------------------------------

    def fsdp_pre_all_gather(
        self,
        mesh: Any,
        outer_size: torch.Size,
        outer_stride: Tuple[int, ...],
        module: nn.Module,
        mp_policy: Any,
    ) -> Tuple[Tuple[torch.Tensor, ...], Any]:
        """
        Quantize this rank's shard and hand FSDP **packed trits + alpha** to put on the wire.

        The cast to ``mp_policy.param_dtype`` happens **first**, before any statistic is taken.
        That is what makes this bitwise comparable to the current path: today the quantizer sees
        the bf16 *gathered* weight (FSDP casts in ``_to_dtype_if_needed`` at
        ``_fsdp_param.py:743``), not the fp32 master, and ``twn_threshold_and_scale`` accumulates
        in fp32 *from those bf16 values*. Quantizing the fp32 master here instead would be a
        different -- arguably better, definitely not identical -- quantizer.

        :returns: ``((packed_uint8, alpha_fp32), metadata)``. Heterogeneous dtypes are fine:
            ``_get_all_gather_input_metadatas`` (``_fsdp_collectives.py:649-669`` at v2.9.0)
            promotes the wire dtype to ``uint8`` and bitcasts both inputs, then
            ``foreach_all_gather_copy_out`` splits them back by recorded numel and dtype.
        """
        del mesh, outer_stride, module
        spec = self._spec
        local = self._tensor
        param_dtype = getattr(mp_policy, "param_dtype", None) or local.dtype

        spec.assert_exact(
            local_flat_dim0=int(local.shape[0]),
            param_name=type(self).__name__,
            world_size=max(1, int(outer_size[0]) // max(1, int(local.shape[0]))),
        )

        trits, alpha = trits_and_alpha(local.to(param_dtype), spec)
        packed = pack_trits(trits)
        return (packed, alpha.reshape(-1).to(torch.float32)), (
            tuple(spec.interpreted_shape(int(local.shape[0]))),
            param_dtype,
        )

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        """
        Unpack the gathered trits, re-apply each rank's ``alpha``, and rebuild the full weight.

        The all-gather concatenates each rank's contribution along the flat axis, so the packed
        buffer is ``world_size`` blocks of this rank's byte count, in rank order, and likewise
        for ``alpha``. Because FSDP's placement is ``Shard(0)``, block ``r`` unpacks to rank
        ``r``'s slice of interpreted axis 0 and the blocks concatenate in the same order -- which
        is exactly the layout ``torch.as_strided(unsharded_tensor, _orig_size, ...)`` expects
        (``_fsdp_param.py:502-507``).
        """
        packed, alpha_flat = all_gather_outputs
        local_shape, _ = metadata
        packed = packed.reshape(-1)
        alpha_flat = alpha_flat.reshape(-1)

        per_rank_bytes = int(math.prod(local_shape)) // TRITS_PER_BYTE
        if per_rank_bytes == 0 or packed.numel() % per_rank_bytes != 0:
            raise AssertionError(
                f"gathered {packed.numel()} bytes is not a whole multiple of the "
                f"{per_rank_bytes} bytes each rank contributed"
            )
        world_size = packed.numel() // per_rank_bytes

        alpha_shape = list(local_shape)
        alpha_shape[self._spec.in_dim] = 1
        per_rank_alpha = int(math.prod(alpha_shape))

        chunks: List[torch.Tensor] = []
        for r in range(world_size):
            trits = unpack_trits(packed[r * per_rank_bytes : (r + 1) * per_rank_bytes]).reshape(
                local_shape
            )
            alpha = alpha_flat[r * per_rank_alpha : (r + 1) * per_rank_alpha].reshape(alpha_shape)
            chunks.append(reconstruct_from_trits(trits, alpha, out_dtype=param_dtype))

        full = torch.cat(chunks, dim=0).reshape(-1)
        if out is not None:
            out.reshape(-1).copy_(full)
            return None
        return full, (full,)


torch.serialization.add_safe_globals([TernaryCommTensor, TernaryCommSpec])


# ===================================================================================
# Wiring it onto a built model
# ===================================================================================


def _linear_spec(mod: nn.Linear) -> TernaryCommSpec:
    return TernaryCommSpec(trailing=(int(mod.weight.shape[1]),), in_dim=1)


def _expert_specs(mod: nn.Module) -> Dict[str, TernaryCommSpec]:
    """
    Specs for a stacked-expert MLP's three weights.

    ``in_dim`` is taken from the ``maybe_quantize`` call sites, **not** re-derived: ``mlp.py:251``
    uses ``in_dim=1`` for all three of ``MoEMLP``'s (``torch.bmm`` contracts ``w``'s axis -2
    unconditionally), while ``mlp.py:404`` uses ``in_dim=2`` for ``DroplessMoEMLP``'s ``w1``/``w3``
    and ``1`` for its ``w2`` (``gmm`` with ``trans_b=True`` for the first two only). The dropless
    weights all have the *same shape* and do *not* all have the same ``in_dim``, so shape is not
    a safe source of truth here.
    """
    d_model = int(getattr(mod, "d_model"))
    hidden = int(getattr(mod, "hidden_size"))
    dropless = type(mod).__name__ == "DroplessMoEMLP"
    if dropless:
        return {
            "w1": TernaryCommSpec(trailing=(hidden, d_model), in_dim=2),
            "w3": TernaryCommSpec(trailing=(hidden, d_model), in_dim=2),
            "w2": TernaryCommSpec(trailing=(hidden, d_model), in_dim=1),
        }
    return {
        "w1": TernaryCommSpec(trailing=(d_model, hidden), in_dim=1),
        "w3": TernaryCommSpec(trailing=(d_model, hidden), in_dim=1),
        "w2": TernaryCommSpec(trailing=(hidden, d_model), in_dim=1),
    }


def apply_ternary_comm(model: nn.Module) -> List[str]:
    """
    Swap every enabled ternary weight for a :class:`TernaryCommTensor`.

    **Must be called before ``fully_shard``**, for the same reason torchao's float8 conversion
    must be (``float8/__init__.py:73-76``): FSDP2 reads the hooks off the *sharded local tensor*
    it builds at ``fully_shard`` time, so a swap afterwards is invisible.

    Only tensors ``nn/quantization.py`` already quantizes are touched -- enabled
    :class:`~olmo_core.nn.quantization.QuantLinear` weights and stacked expert weights whose
    ``quant`` is enabled. Embeddings, ``lm_head``, the router and every norm are unreachable from
    here, so :func:`~olmo_core.nn.quantization.audit_quantization`'s carve-out assertion holds by
    construction rather than by a second check.

    :returns: The fully-qualified names of the parameters converted, so a caller can assert the
        set is non-empty and covers what it expected. An empty list means the flag did nothing,
        which is a configuration error at the call site, not a no-op to shrug at.

    :raises OLMoConfigurationError: if compiled autograd is enabled, which would silently bypass
        the hooks and train the model **unquantized**.
    """
    try:
        from torch._dynamo.compiled_autograd import compiled_autograd_enabled

        enabled = (
            compiled_autograd_enabled()
            if callable(compiled_autograd_enabled)
            else compiled_autograd_enabled
        )
        if enabled:
            raise OLMoConfigurationError(
                "ternary-compressed all-gather is incompatible with compiled autograd: at torch "
                "2.9.0 both FSDPParam.all_gather_inputs and init_unsharded_param guard the "
                "extension hooks with `not compiled_autograd_enabled()`, so the hooks would be "
                "skipped, the raw latent weight would be gathered, and the model would train "
                "UNQUANTIZED while reporting itself as the ternary arm."
            )
    except ImportError:  # pragma: no cover - torch internal moved
        pass

    converted: List[str] = []

    def _convert(owner: nn.Module, attr: str, spec: TernaryCommSpec, fqn: str) -> None:
        param = getattr(owner, attr)
        if isinstance(param, TernaryCommTensor):
            return
        if param.ndim != 2:
            raise OLMoConfigurationError(
                f"{fqn} has ndim {param.ndim}; ternary_comm expects the stored 2-D layout "
                f"(expert weights are flattened to (E*a, b) at nn/moe/mlp.py:185)"
            )
        spec.interpreted_shape(int(param.shape[0]))
        if int(param.shape[1]) != spec.trailing[-1]:
            raise OLMoConfigurationError(
                f"{fqn} stored last dim {param.shape[1]} != spec trailing[-1] "
                f"{spec.trailing[-1]}; the spec does not describe this tensor"
            )
        new = nn.Parameter(
            TernaryCommTensor(param.detach(), spec), requires_grad=param.requires_grad
        )
        owner.register_parameter(attr, new)
        converted.append(fqn)

    for fqn, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            if mod.quant_enabled:
                _convert(mod, "weight", _linear_spec(mod), f"{fqn}.weight")
            continue
        quant = getattr(mod, "quant", None)
        if quant is None or not getattr(quant, "enabled", False):
            continue
        if not all(isinstance(getattr(mod, a, None), nn.Parameter) for a in ("w1", "w2", "w3")):
            continue
        for attr, spec in _expert_specs(mod).items():
            _convert(mod, attr, spec, f"{fqn}.{attr}")

    return converted


# ===================================================================================
# Cost / benefit
# ===================================================================================


def ternary_comm_bytes_report(
    *,
    quantized_numel: int,
    full_precision_numel: int,
    n_all_gathers_per_step: int = 2,
    bf16_bytes: int = 2,
) -> Dict[str, float]:
    """
    Bytes on the all-gather wire per optimizer step, with and without compression.

    ``n_all_gathers_per_step`` is **2**, not 1: with ``reshard_after_forward=True`` (OLMo-core's
    default at ``nn/transformer/model.py:932``) every unit is gathered once in forward and again
    in backward. ``lm_head`` is the exception -- it is sharded with ``reshard_after_forward=False``
    (``model.py:959``) -- but ``lm_head`` is a full-precision carve-out and so contributes
    nothing to the compressed term either way.

    Compressed cost is ``numel/4`` bytes of trits plus ``4 * numel/fold_len`` bytes of fp32
    ``alpha``; the ``alpha`` term is negligible at real shapes (one scalar per output row of
    length >= 384) but is counted rather than waved away.
    """
    baseline = (quantized_numel + full_precision_numel) * bf16_bytes
    compressed_q = quantized_numel / TRITS_PER_BYTE
    compressed = compressed_q + full_precision_numel * bf16_bytes
    return {
        "baseline_bytes_per_gather": float(baseline),
        "compressed_bytes_per_gather": float(compressed),
        "baseline_bytes_per_step": float(baseline * n_all_gathers_per_step),
        "compressed_bytes_per_step": float(compressed * n_all_gathers_per_step),
        "saved_bytes_per_step": float((baseline - compressed) * n_all_gathers_per_step),
        "ratio": float(baseline) / float(compressed) if compressed else float("inf"),
    }
