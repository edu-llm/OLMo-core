"""
Does our sliding-window attention SAVE arithmetic, or spend extra?

At ``bcc05d6`` it spent extra, and nothing in a config dump said so. ``TorchAttentionBackend``
builds a dense ``(s, s)`` boolean mask for a windowed layer (``backend.py:416-448``) and then hands
it to SDPA under ``is_causal=attn_mask is None`` (``backend.py:371``) -- so the presence of a window
turns ``is_causal`` False, SDPA drops its causal fast path, and the layer computes the FULL ``s x s``
score matrix in order to throw away ``1 - w/s`` of it. Maple's 3:1 SWA:global layout then executes
about **1.75x** the attention arithmetic of plain global causal attention: the efficiency feature,
inverted.

The fix is a dependency, not a code change -- flash-attn 2 takes the window as a native
``window_size`` argument (``backend.py:634-648``) and never forms the masked-out blocks.

WHICH IS WHY THIS FILE CARRIES NO DOCKERFILE CHANGE. Installing the wheel is PR #67's subject
(``.edullm/Dockerfile``, the 2.8.3 ``cu12torch2.9cxx11abiTRUE-cp312`` release asset), and it
arrived there from a different symptom entirely: the fourteen ``olmo3_*`` factories hardcode
``attn_backend=flash_2`` and ``Attention.__init__`` calls ``assert_supported()``, so on a GPU
those configs cannot be INSTANTIATED. Same root cause, same file, one install -- so duplicating
the edit here would be a competing change to a file already under review.

Our stake is the part #67 does not measure. A passing image check proves the package imports; it
says nothing about whether the window reaches the kernel or whether the dense mask is still being
built alongside it. That is the difference between "the wheel installed" and "the saving is real",
and every way it can fail is silent:

* the wheel installs but ``_maple_uniform_attn_backend()`` still answers ``torch``
  -> ``test_maple_resolves_to_flash_2_when_the_package_is_importable``
* flash_2 is selected but the window never reaches the kernel, so a "SWA" layer is really global
  -> ``test_flash_2_swa_matches_torch_dense_mask`` (plus the non-vacuity test below)
* the dense mask is still materialized alongside the native window, so nothing was saved
  -> ``test_flash_2_swa_materializes_no_dense_mask``

Every test here needs a GPU and a real flash-attn, and skips otherwise, EXCEPT
``test_swa_saves_the_arithmetic_it_claims_to``, which is closed-form and runs anywhere. **A skipped
test proves nothing about the image** -- #67's build-time assertion is what covers the container,
and an A100 (sm_80) run is what covers the hardware. FarmShare is sm_89, so even a green run there
leaves the funded target unproven: the wheel's sm_80 cubin has never been executed. UNTESTED.
"""

from typing import Tuple

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.backend import TorchAttentionBackend
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.testing import requires_flash_attn_2, requires_gpu
from olmo_core.utils import seed_all

# Maple's real geometry, shrunk only where it costs nothing to shrink.
#
# ``head_dim`` and the GQA ratio are NOT shrunk, because they are the two things flash-attn could
# plausibly reject. m7b is n_heads=12 / n_kv_heads=3, and 3 is neither even nor a power of two --
# worth pinning, since ``flash_api.cpp:395`` requires only ``num_heads % num_heads_k == 0`` (12/3=4)
# and imposes no further constraint on n_kv itself. If a future flash-attn adds one, this fails here
# rather than on a paid A100 run.
HEAD_DIM = 128
N_HEADS = 12
N_KV_HEADS = 3

# ``s`` must be several multiples of ``w`` or the window is nearly the whole sequence and a kernel
# that ignored it entirely would still produce almost the right answer. At s=256, w=32 the window
# covers 12.5% of the keys, so "window ignored" and "window honoured" are far apart -- which is what
# ``test_swa_and_global_disagree_...`` measures rather than assumes.
SEQ_LEN = 256
WINDOW = 32

# Maple's own window and the sequence length the ladder trains at. Used only for the arithmetic
# statement, which needs no GPU.
MAPLE_WINDOW = 512
MAPLE_SEQ_LEN = 2048


def _window_size_tuple(window: int) -> Tuple[int, int]:
    """
    The same translation ``Attention.__init__`` does (``attention/__init__.py:505``): look ``w-1``
    tokens left and none right, so a query attends to ``w`` keys including itself. Duplicated
    deliberately -- these tests drive the backends directly, and reusing the production expression
    here would make a change to it invisible.
    """
    return (window - 1, 0)


def _qkv(seq_len: int, dtype: torch.dtype, seed: int = 0):
    seed_all(seed)
    return (
        torch.randn(2, seq_len, N_HEADS, HEAD_DIM, device="cuda", dtype=dtype),
        torch.randn(2, seq_len, N_KV_HEADS, HEAD_DIM, device="cuda", dtype=dtype),
        torch.randn(2, seq_len, N_KV_HEADS, HEAD_DIM, device="cuda", dtype=dtype),
    )


def _build(name: AttentionBackendName, window: int, cache=None):
    return name.build(
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        window_size=_window_size_tuple(window) if window > 0 else (-1, -1),
        cache=cache,
    )


@requires_gpu
@requires_flash_attn_2
def test_maple_resolves_to_flash_2_when_the_package_is_importable():
    """
    The switch is the import, and there is no config flag. Pin that.

    ``_maple_uniform_attn_backend`` probes ``has_flash_attn_2()`` and falls back to ``torch``
    otherwise -- a fallback with no exception and no config-visible trace. If someone adds a gate
    (a compute-capability check, an env var, an opt-in flag), an image carrying a perfectly good
    wheel would silently keep running the dense-mask path, and this is the test that notices.
    """
    from olmo_core.nn.transformer.config import _maple_uniform_attn_backend

    assert _maple_uniform_attn_backend() == AttentionBackendName.flash_2, (
        "flash-attn 2 imports on this host, but the Maple factory still chose the torch SDPA "
        "backend, whose SWA path builds a dense (s, s) mask and disables is_causal. That is the "
        "1.75x-attention-arithmetic regression, and it is invisible in a config dump."
    )


@requires_gpu
@requires_flash_attn_2
def test_flash_2_swa_materializes_no_dense_mask():
    """
    Assert the MECHANISM, not just the answer: no ``(s, s)`` tensor is created.

    Numerical agreement alone cannot distinguish "used the native window" from "built the dense mask
    and got the same number" -- and the second is the status quo we are trying to leave. So this
    watches the cache both backends were handed. ``TorchAttentionBackend.warmup_cache`` is the exact
    call the trainer makes, and it exists ONLY to precompute this mask.

    The torch half is the positive control. Without it, a flash_2 assertion that passed because the
    key was renamed would look like success.
    """
    torch_cache, flash_cache = BufferCache(), BufferCache()

    torch_backend = _build(AttentionBackendName.torch, WINDOW, cache=torch_cache)
    flash_backend = _build(AttentionBackendName.flash_2, WINDOW, cache=flash_cache)

    device = torch.device("cuda")
    torch_backend.warmup_cache(SEQ_LEN, device)
    flash_backend.warmup_cache(SEQ_LEN, device)

    def cached(cache: BufferCache):
        # `build` namespaces the sub-cache per backend (`backend.py:109`), so read every namespace
        # rather than assuming which one a backend chose.
        return [t for sub in cache._data.values() for t in sub.values()]

    torch_masks = cached(torch_cache)

    # POSITIVE CONTROL. If this fails, the dense mask moved or the cache API changed, and the
    # flash_2 assertion below has stopped measuring anything -- it would pass on an empty cache.
    assert any(tuple(t.shape[-2:]) == (SEQ_LEN, SEQ_LEN) for t in torch_masks), (
        f"the torch backend did not materialize an ({SEQ_LEN}, {SEQ_LEN}) mask, so this test is no "
        f"longer watching the thing it was written to watch and the flash_2 assertion below is "
        f"vacuous. Cached shapes: {[tuple(t.shape) for t in torch_masks]}"
    )

    offenders = [
        tuple(t.shape) for t in cached(flash_cache) if t.dim() >= 2 and t.shape[-1] >= SEQ_LEN
    ]
    assert not offenders, (
        f"the flash_2 backend materialized {offenders}. flash-attn takes the window as a native "
        f"`window_size` argument, so an (s, s) buffer means the dense-mask path is still live and "
        f"the SWA saving was not realized."
    )

    # And the native window is what the backend will actually pass down. ``forward`` hands
    # ``window_size=self.window_size`` straight to ``dispatch_flash_attn``
    # (``backend.py:634-648``), so this attribute IS the argument.
    assert flash_backend.window_size == _window_size_tuple(WINDOW), (
        f"flash_2 backend holds window_size={flash_backend.window_size}, expected "
        f"{_window_size_tuple(WINDOW)}. This value is forwarded verbatim to the kernel; (-1, -1) "
        f"here would mean the layer runs GLOBAL attention while the config says it is windowed."
    )

    assert not isinstance(flash_backend, TorchAttentionBackend)


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        # bf16 has 8 mantissa bits, so one rounding is ~2^-8 = 3.9e-3 relative. These are the
        # tree's own BF16_ATOL/BF16_RTOL from attention_test.py, NOT numbers fitted to whatever
        # passed -- reusing the established pair is what keeps them honest.
        pytest.param(torch.bfloat16, 1e-5, 5e-3, id="bf16"),
        # fp16 accumulates in fp32 in both kernels, so the gap is reduction ORDER only: flash-attn
        # tiles over keys and rescales running softmax statistics, SDPA reduces the full row. The
        # difference is O(sqrt(w) * eps_fp32) on the softmax weights, far under fp16's own 4.9e-4
        # resolution -- which is why the tolerance is fp16's epsilon and not a measured residual.
        pytest.param(torch.float16, 1e-5, 1e-3, id="fp16"),
    ],
)
@requires_gpu
@requires_flash_attn_2
def test_flash_2_swa_matches_torch_dense_mask(dtype: torch.dtype, rtol: float, atol: float):
    """
    The native window must compute the SAME function as the dense mask, or this is not a speedup --
    it is a different model that happens to be faster.

    The dense-mask path is the reference precisely because it is the slow one: it is what
    ``bcc05d6`` trains today, so agreement here is what makes the switch a pure efficiency change
    rather than a change in what gets learned.
    """
    q, k, v = _qkv(SEQ_LEN, dtype)

    flash_out = _build(AttentionBackendName.flash_2, WINDOW)((q, k, v))
    torch_out = _build(AttentionBackendName.torch, WINDOW)((q, k, v))

    assert flash_out.shape == torch_out.shape
    torch.testing.assert_close(flash_out, torch_out, rtol=rtol, atol=atol)


@requires_gpu
@requires_flash_attn_2
def test_swa_and_global_disagree_so_the_equivalence_test_is_not_vacuous():
    """
    THE GUARD. If the window were silently ignored, the test above would still pass.

    A kernel that dropped ``window_size`` computes GLOBAL causal attention. The dense-mask reference
    would then have to be wrong in the same way for them to agree -- unlikely -- but "unlikely" is
    not a test. The real risk is subtler: if global and windowed outputs were numerically CLOSE at
    this shape, the equivalence test would pass under a broken kernel and prove nothing.

    So pin the separation, in the same tolerance the equivalence test uses. Two directions:

    1. windowed != global by MUCH MORE than the equivalence tolerance -- the shape is discriminating;
    2. flash_2-windowed is closer to torch-windowed than to torch-GLOBAL -- it honoured the window
       rather than merely landing in the neighbourhood.

    Assertion 2 is the one that would catch a dropped ``window_size``, and it is a comparison of
    magnitudes rather than a bare inequality: the ratio has to be orders of magnitude, not 1.001.
    """
    dtype, atol = torch.bfloat16, 5e-3
    q, k, v = _qkv(SEQ_LEN, dtype)

    torch_swa = _build(AttentionBackendName.torch, WINDOW)((q, k, v))
    torch_global = _build(AttentionBackendName.torch, -1)((q, k, v))
    flash_swa = _build(AttentionBackendName.flash_2, WINDOW)((q, k, v))

    swa_vs_global = (torch_swa - torch_global).abs().max().item()

    # (1) The shape discriminates. w/s = 32/256 = 12.5% of the keys, so this should be enormous
    # relative to atol; 20x is a floor, not the expected value.
    assert swa_vs_global > 20 * atol, (
        f"windowed and global attention differ by only {swa_vs_global:.3e} at s={SEQ_LEN}, "
        f"w={WINDOW}, versus an equivalence tolerance of {atol:.3e}. At this separation a kernel "
        f"that IGNORED the window would still pass test_flash_2_swa_matches_torch_dense_mask, so "
        f"that test would be vacuous. Lower WINDOW or raise SEQ_LEN."
    )

    # (2) flash_2 honoured the window. Both distances are measured against the same reference
    # family, so the ratio is meaningful.
    to_swa = (flash_swa - torch_swa).abs().max().item()
    to_global = (flash_swa - torch_global).abs().max().item()

    assert to_swa < atol, f"flash_2 windowed output is {to_swa:.3e} from the dense-mask reference"
    assert to_global > 20 * to_swa, (
        f"flash_2's output sits {to_swa:.3e} from WINDOWED torch and {to_global:.3e} from GLOBAL "
        f"torch -- a ratio of only {to_global / max(to_swa, 1e-12):.1f}x. It should be far closer "
        f"to windowed. A ratio near 1 means flash-attn is computing global attention and the "
        f"`window_size` argument is not reaching the kernel: the layers would be CORRECT-looking "
        f"and full cost, which is the exact regression this lane exists to remove."
    )


def test_swa_saves_the_arithmetic_it_claims_to():
    """
    The size of the prize, in closed form, so the motivation is auditable rather than asserted.
    No GPU and no flash-attn: this is arithmetic about the mask, and it holds on any host.

    Score-matrix entries a causal layer must evaluate, for seq_len ``s`` and window ``w``:

        global   = s(s+1)/2
        windowed = s*w - w(w-1)/2        [each query sees min(i+1, w) keys]

    The dense-mask path evaluates ``s^2`` -- it loses ``is_causal``, so not even the causal half is
    skipped. That is the pessimization: MORE than global, for a layer that needs a fraction.
    Every constant below was computed from these two expressions, not transcribed from a report.
    Three separate hand-transcription errors are already on this project's record; the second
    ``s=4096`` block exists so the s-dependence is pinned rather than assumed, because the saving
    grows with ``s`` and a single row cannot show that.
    """

    def entries(s: int, w: int):
        global_ = s * (s + 1) // 2  # causal, every key up to i
        windowed = s * w - w * (w - 1) // 2  # sum_i min(i+1, w)
        dense = s * s  # is_causal lost: the FULL square
        # Cross-check `windowed` against an independently written form. Same quantity, different
        # decomposition (ramp + steady state), so a slip in either expression fails here.
        assert windowed == w * (w + 1) // 2 + (s - w) * w
        return global_, windowed, dense

    s, w = MAPLE_SEQ_LEN, MAPLE_WINDOW
    global_entries, windowed_entries, dense_mask_entries = entries(s, w)

    assert (global_entries, windowed_entries, dense_mask_entries) == (2_098_176, 917_760, 4_194_304)

    # The window is a real restriction at the ladder's sequence length: a windowed layer needs
    # 43.7% of a global layer's score entries at s=2048, w=512.
    assert windowed_entries < global_entries
    assert windowed_entries / global_entries == pytest.approx(0.4374, abs=0.001)

    # THE PESSIMIZATION, and it is the whole reason this lane exists. The dense-mask path costs
    # essentially 2x GLOBAL causal attention -- it loses `is_causal`, so it does not even skip the
    # upper triangle -- for a layer the config describes as cheap. 4.57x what it should cost.
    assert dense_mask_entries > global_entries
    assert dense_mask_entries / global_entries == pytest.approx(2.0, abs=0.01)
    assert dense_mask_entries / windowed_entries == pytest.approx(4.570, abs=0.01)

    # Maple's 3:1 layout: 3 windowed + 1 global per group of 4, versus a hypothetical all-global
    # model. This is the headline number.
    native_group = 3 * windowed_entries + global_entries
    dense_group = 3 * dense_mask_entries + global_entries
    all_global = 4 * global_entries

    # Status quo: our "efficient" 3:1 SWA costs 1.749x plain global causal attention. The feature
    # is not merely unrealized, it is INVERTED.
    assert dense_group / all_global == pytest.approx(1.749, abs=0.002)
    # With the native window it costs 57.8% of global, as intended.
    assert native_group / all_global == pytest.approx(0.578, abs=0.002)
    # So the fix is worth 3.03x of the attention term at s=2048.
    assert dense_group / native_group == pytest.approx(3.026, abs=0.01)

    # AND IT GROWS WITH s, which is why this matters more at longer context and why a single-row
    # assertion would understate it. At s=4096 the same fix is worth 4.11x.
    g4, w4, d4 = entries(2 * s, w)
    assert w4 / g4 == pytest.approx(0.2343, abs=0.001)
    assert (3 * d4 + g4) / (3 * w4 + g4) == pytest.approx(4.109, abs=0.01)
    # The pessimization ratio, by contrast, is s-invariant at ~1.75: dense/global -> 2 for any s,
    # so the 3:1 mix sits at (3*2 + 1)/4. That invariance is why the 1.75x claim needs no caveat.
    assert (3 * d4 + g4) / (4 * g4) == pytest.approx(1.749, abs=0.002)
