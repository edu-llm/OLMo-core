"""The MoE grouped-GEMM substitution, kept apart from everything that needs the platform.

This module imports ``torch`` and nothing else on purpose. ``train_on_corpus`` reaches
``olmo_core``, ``edullm_data``, ``rich`` and ``wandb`` before it reaches line 200, so a
verification that had to import it could only run somewhere the whole training image exists --
which is not where a GPU is easiest to get hold of. Splitting the twenty lines that matter out
means the check can run on any machine with a card and a torch.

WHAT IS BEING SUBSTITUTED AND WHY. ``grouped_gemm`` is absent from the training image, observed
by running the container. Without it ``olmo_core.nn.moe.mlp.DroplessMoEMLP`` falls back to a
Python loop over local experts: one GEMM per expert, a ``batch_sizes.cpu()`` per call -- which
is a device-to-host synchronisation the whole stream waits on -- and a ``torch.cat`` to put the
pieces back together, all inside a method decorated ``@torch._dynamo.disable()``.

``torch._grouped_mm`` does the same arithmetic in one call. It is not a replacement package; it
is a kernel torch already ships, and on an H100 it is the same class of CUTLASS grouped GEMM a
``grouped_gemm`` build would have compiled -- ``bf16bf16_grouped_gemm_impl_sm90_sm100<Sm90>`` is
in the image's own ``libtorch_cuda.so``.

WHAT THIS IS WORTH, HONESTLY. Less than it sounds. At expert-parallel degree 8 each rank holds
four local experts rather than 32, so the loop runs four iterations and each GEMM is large
enough to be compute-bound -- 16,384 x 2,048 x 2,048 is roughly 960 FLOP/byte against an H100
ridge point near 295. The synchronisations the library's warning complains about are under 1%
of step time at this shape. The estimate is 1.1x-1.35x, not the order of magnitude the same
fallback would cost at degree 1.
"""

from __future__ import annotations

import torch

__all__ = ["grouped_mm_gmm", "install_grouped_mm"]


def grouped_mm_gmm(a, b, batch_sizes, trans_b: bool = False):
    """A drop-in for ``grouped_gemm.ops.gmm``, backed by ``torch._grouped_mm``.

    Same signature and same result, so it can be substituted for the module-level ``gmm`` that
    :class:`DroplessMoEMLP` captures into ``self._gmm`` in its constructor.

    ``batch_sizes`` is the token count per *local* expert; ``torch._grouped_mm`` wants the
    inclusive cumulative sum of those counts as int32 on the activations' device. Computing it
    with ``torch.cumsum`` is most of why this is faster than the loop -- the loop calls
    ``batch_sizes.cpu()``, and that copy is a synchronisation.

    ``trans_b`` transposes each expert's weight. The transposed view is passed straight through
    rather than copied: it is not contiguous, and a kernel that refused it would have cost a
    33 MB copy on each of the 192 calls a step makes. That the kernel accepts it was checked
    against the real image rather than assumed.
    """
    offsets = torch.cumsum(batch_sizes, dim=0).to(device=a.device, dtype=torch.int32)
    return torch._grouped_mm(a, b.transpose(-2, -1) if trans_b else b, offs=offsets)


def install_grouped_mm(*, enabled: bool = True) -> str:
    """Route the dropless MoE through ``torch._grouped_mm``. Returns what it decided, for the log.

    WHY A MONKEYPATCH AND NOT AN EDIT TO ``mlp.py``. On the block only ``.edullm/`` is read from
    the branch clone; ``import olmo_core`` resolves to the copy baked into the image. A change
    under ``src/olmo_core/`` would sit in the repository looking applied and never run. This is
    not a shortcut around review -- it is the only place the change can be made without
    rebuilding the image, and rebuilding the image hours before a window that cannot be repeated
    is the larger risk.

    WHAT IT MUST NOT DO. Take effect when the fast package is present. ``mlp.gmm`` is ``None``
    exactly when ``grouped_gemm`` failed to import, so a future image carrying the package keeps
    using it and this returns without touching anything.

    CALL IT BEFORE THE MODEL IS BUILT. ``DroplessMoEMLP.__init__`` reads the module-level ``gmm``
    once, so a patch applied afterwards changes nothing while the log says it worked.
    """
    try:
        from olmo_core.nn.moe import mlp as moe_mlp
    except ImportError:
        return "no MoE module, nothing to patch"

    if getattr(moe_mlp, "gmm", None) is not None:
        return "grouped_gemm is present, left alone"
    if not enabled:
        return "disabled by --no-moe-grouped-mm, using the library's Python loop"
    if not hasattr(torch, "_grouped_mm"):
        return f"torch {torch.__version__} has no _grouped_mm, using the library's Python loop"

    moe_mlp.gmm = grouped_mm_gmm
    return "grouped_gemm absent, routed through torch._grouped_mm"
