"""Dynamic filter generator for Exp-2 (the Dynamic-LIV synthetic mechanism study).

Owner: sub-agent A. Binding spec: ``docs/dynconv-review/build/exp2/SPEC.md``.

WHAT THIS IS
------------
LFM2's LIV block is ``out = out_proj(C * depthwise_conv(B * x))`` with a *static* depthwise
filter ``a`` of ``W`` taps. This module makes that filter input-dependent::

    z_t   = V h_t                       # (d -> R)      conditioning stream
    dw_t  = U z_t   reshaped (d, W)     # (R -> W*d)    per-position filter perturbation
    w_t   = a + alpha * dw_t            # (d, W)        the filter actually convolved at t

``h`` is the **normalized block input** (the tensor handed to the mixer), not the gated stream
``B_t * x_t``. SPEC §7 fixes this and gives three reasons: it matches the cited paper (allows
fusing the generator into the input projection), ``h`` is normalized whereas ``B_t * x_t`` is a
product of two unnormalized projections and LightConv's headline failure was that *unnormalized*
dynamic filters diverged, and conditioning on the gated stream would re-entangle the two
mechanisms this experiment exists to separate.

WHY ``W - 2`` AND NOT ``W``
---------------------------
``orch_verify_W_minus_2.py`` (reproduced analytically by R5 F1) establishes that the static tap
family ``kappa[t,k] = C_t * a_k * B_{t-k}`` has Jacobian rank *exactly* ``2T + W - 3``, so the
genuinely new degrees of freedom bought by a per-position filter are ``W - 2`` per position per
channel: one generated number is redundant with the post-gate ``C_t``, one with the ``B``
sequence. **At W=2 the dynamic block is an exact reparameterization of the static block** (max
log-residual 8.3e-16, a constructive realization). That makes W=2 the program's cheapest
falsification control, not merely a data point.

THREE TRAPS THIS FILE IS BUILT AROUND
-------------------------------------
1. **The dead-branch bug.** ``Delta_w = alpha * U(V h)`` with BOTH ``U = 0`` and ``alpha = 0``
   makes ``dL/dU = dL/dV = dL/dalpha == 0`` *forever*: an exact saddle. The run trains stably,
   every arm ties, and it reads as a clean replicable negative -- the most expensive possible
   failure because it looks like science. NVIDIA shipped exactly this
   (``dynamic_conv.py:80-84``). SPEC §3 is therefore non-negotiable: ``U = 0``, ``V`` random,
   ``alpha = 1`` and **learnable**. See :class:`DynamicFilterGen` and preflight check 5b.
2. **DTensor-unsafe init.** An indexed in-place write such as ``w[...] = x`` inside
   ``init_weights`` lowers to ``aten.fill_.Tensor``, which has no registered DTensor sharding
   strategy, and kills the job seconds into training under FSDP (this is what killed submitted
   run ``run_019fbf9f`` at ``TRAINING_ITSELF_FAILED``). **A single-process CPU test structurally
   cannot catch it**, because no ``DTensor`` is ever constructed. Every initialization here is
   therefore routed through ``olmo_core.nn.transformer.init._apply_init``, which materializes the
   full tensor, initializes *that*, and copies back this rank's shard.
3. **The 3-D fan-in trap.** ``nn.init.kaiming_uniform_`` on a 3-D parameter derives
   ``fan_in = size(1) * receptive_field``, which is **not** the true contraction. For our ``V`` the
   contraction is over ``d`` alone. We therefore spell the bound out against ``d`` rather than
   delegating to a helper that the shape can mislead (SPEC §5.3). A one-sided fan-in correction
   has already biased a contrast toward the hypothesis once in this repo (memory
   ``fan-in-correct-one-branch-only``), so the correction is applied on *every* branch.

INTERFACES OTHER FILES MAY RELY ON (sub-agent B: this is your surface)
----------------------------------------------------------------------
* :class:`DynamicFilterGen` -- the generator. ``forward(h) -> (B, T, d, W)`` perturbation, or
  ``None`` when ``alpha`` is overridden to exactly zero *and* ``force_kernel=False``.
* :class:`DynamicShortConv` -- ``ShortConv`` with the dynamic filter wired to the **sequence
  mixer** forward. Drop-in for ``ShortConv``; identical parameter *names* for every shared tensor.
* :class:`DynamicQKVConv` -- the S3 mechanism: dynamic depthwise conv on Q, K, V inside an
  attention block.
* :func:`engagement_report` -- per-layer ``E_l`` and input-dependence, never averaged over layers.
* :func:`set_alpha_override` / :func:`clear_alpha_override` -- the ablate-at-eval hook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.attention.short_conv import ShortConv
from olmo_core.nn.transformer.init import _apply_init

__all__ = [
    "DynamicFilterGen",
    "DynamicShortConv",
    "DynamicQKVConv",
    "PermuteMode",
    "EngagementStats",
    "engagement_report",
    "set_alpha_override",
    "clear_alpha_override",
    "reset_permutations",
    "depthwise_causal_conv_static",
    "depthwise_causal_conv_dynamic",
    "gen_param_count",
    "iter_generators",
    "dyn_param_names",
    "split_param_groups",
    "bf16_dead_zone_probe",
    "static_realizability_residual",
    "count_flops_proxy",
    "named_shared_params",
]


PermuteMode = Literal["full", "causal_prefix"]


# ---------------------------------------------------------------------------------------------
# The two kernels. Both are pure functions so preflight can call them as an independent
# reference, which is the point: preflight check 12 must be able to detect a rogue activation in
# the conv path *numerically*, not by reading an attribute someone forgot to set.
# ---------------------------------------------------------------------------------------------


def depthwise_causal_conv_static(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Reference depthwise causal conv with a position-independent filter.

    :param u: ``(B, T, d)`` input.
    :param a: ``(d, W)`` filter; ``a[:, -1]`` multiplies the *current* token (tap k=0 in the
        ``kappa[t,k]`` convention of R5, i.e. the last slot in torch's conv layout).
    :returns: ``(B, T, d)``.

    Deliberately activation-free. ``CausalConv1d.__init__`` in this fork defaults to
    ``activation="silu"`` (``olmo_core/nn/convolution.py:37``) while the released
    ``Lfm2ShortConv`` passes ``activation=None`` -- a *different operator* that still trains,
    just worse. That is a silent failure, so it gets a numerical check rather than a flag check.
    """
    W = a.shape[-1]
    ut = u.transpose(1, 2)  # (B, d, T)
    win = F.pad(ut, (W - 1, 0)).unfold(-1, W, 1)  # (B, d, T, W)
    return (win * a.unsqueeze(0).unsqueeze(2)).sum(-1).transpose(1, 2)


def depthwise_causal_conv_dynamic(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv with a per-position, per-channel filter.

    :param u: ``(B, T, d)`` input.
    :param w: ``(B, T, d, W)`` filter. ``w[b, t, :, -1]`` multiplies token ``t`` itself.
    :returns: ``(B, T, d)``.

    Reduction order over the ``W`` taps differs from ``nn.Conv1d``'s, so the ``alpha=0``
    equivalence to the static path is exact in fp64 and ~5e-8 relative in fp32 -- *not* bitwise.
    R7 check 3 anticipates this explicitly and sets the fp32 tolerance at 1e-6 for that reason.
    Do not "fix" it by tightening the tolerance.
    """
    W = w.shape[-1]
    ut = u.transpose(1, 2)  # (B, d, T)
    win = F.pad(ut, (W - 1, 0)).unfold(-1, W, 1)  # (B, d, T, W)
    return (win.permute(0, 2, 1, 3) * w).sum(-1)  # (B, T, d)


def gen_param_count(d_model: int, rank: int, width: int, n_streams: int = 1) -> int:
    """Exact integer parameter count of one :class:`DynamicFilterGen`.

    ``V: d -> R`` is ``d*R``; ``U: R -> n_streams*W*d`` is ``R*n_streams*W*d``; ``alpha`` is 1.

    The ``W`` factor in ``U`` is the one people drop. R7 check 1 exists because a generator wired
    ``R -> d`` instead of ``R -> W*d`` is off by a factor of ``W`` and **still trains**.
    """
    return d_model * rank + rank * n_streams * width * d_model + 1


@dataclass
class EngagementStats:
    """Per-layer engagement readout. Report per layer; never average over layers."""

    name: str
    engagement: float
    """``E_l = rms_{b,t} || alpha * Delta_w[b,t] ||_F  /  || a ||_F``.

    Dimensionless. ``||a||_F`` at the identity-tap init is exactly ``sqrt(d)`` (11.3137 at
    d=128), so ``E_l`` is a clean engagement ratio and reduces to the per-position ratio when
    ``Delta_w`` happens to be position-constant.

    **ABORT below 1e-3.** The floor is physical, not taste: ``2^-8 = 3.90625e-3`` is bf16's
    half-ulp at 1.0, which is the magnitude of the current-token tap ``a[:, -1] = 1``. Below
    ~1e-3 the perturbation provably cannot move the dominant tap at all, so the arm is the static
    arm carrying fossils.
    """
    input_dependence: float
    """``mean_{d,k} std_{b,t}( w )  /  mean_{b,t,d,k} |w|``.

    Zero means the realized filter is the same at every position: input-*dependence* has
    collapsed even if ``E_l`` is large (which happens when ``U`` learns a constant offset that
    ``a`` could have absorbed). ``E_l`` alone cannot see this.
    """
    u_norm: float
    """``||U||_F``. Check 10: a monotone decrease after the first 5% of training is the
    weight-decay signature -- ``{V, U, alpha}`` must be out of the decay group."""
    alpha: float
    n_positions: int


class DynamicFilterGen(nn.Module):
    """``Delta_w = alpha * U( [sigma] (V h) )``, the low-rank per-position filter generator.

    :param d_model: conditioning width and channel count.
    :param rank: generator bottleneck ``R``. SPEC §7 pins **R=16 at d=128**, i.e. ``R/d = 1/8``,
        a deliberate deviation from the 350M spec's ``R/d = 1/64`` (which would give R=2 here,
        degenerately small). The cited paper's own rank curve is still descending at R=128, so R
        is the steep axis; starving it at d=128 would test the wrong thing. Pre-registered, not
        tuned.
    :param width: kernel taps ``W``.
    :param n_streams: how many independent filters to generate from one conditioning stream.
        1 for a LIV block, 3 for the S3 Q/K/V mechanism.
    :param alpha_init: SPEC §3 fixes **1.0**, learnable.
    :param permute_z: the S2 control. See :meth:`_permute`.
    :param permute_mode: ``"full"`` (SPEC §1) or ``"causal_prefix"`` (a robustness variant).
    :param permute_seed: seed for the permutation stream, so a run is reproducible.
    :param nonlinear: SPEC §7's optional ablation, ``Delta_w = alpha * U * silu(V h)``.
        JetBlock's shipped generator is a nonlinear bottleneck MLP with SiLU, not a bare linear
        map; R5 F4 shows the *linear* generator is provably feature-equivalent to the B/C gates
        (regressing ``Delta_w`` on the gate pre-activations gives R^2 = 1.0000), so the gap it
        buys is structural rather than informational. The nonlinearity is the cheap way to test
        whether that matters. Default off: linear is the primary, per SPEC.

    .. note:: **No bias on V or U.** SPEC §7's formula is ``Delta_w = alpha * U(V h)`` and the
        static filter ``a`` already plays the affine role in ``w = a + alpha * Delta_w`` (R5 F4
        fix 2). Adding a bias to ``U`` would duplicate ``a`` and make the reparameterization at
        W=2 harder to reason about.
    """

    def __init__(
        self,
        *,
        d_model: int,
        rank: int,
        width: int,
        n_streams: int = 1,
        alpha_init: float = 1.0,
        permute_z: bool = False,
        permute_mode: PermuteMode = "full",
        permute_seed: int = 0,
        nonlinear: bool = False,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if n_streams <= 0:
            raise ValueError(f"n_streams must be positive, got {n_streams}")
        self.d_model = d_model
        self.rank = rank
        self.width = width
        self.n_streams = n_streams
        self.permute_z = permute_z
        self.permute_mode = permute_mode
        self.permute_seed = permute_seed
        self.nonlinear = nonlinear

        kw = {"dtype": dtype, "device": init_device}
        self.V = nn.Linear(d_model, rank, bias=False, **kw)
        self.U = nn.Linear(rank, n_streams * width * d_model, bias=False, **kw)
        # alpha is a real 0-dim Parameter, NOT a python float and NOT a buffer. R7 check 2:
        # a parameter created after the optimizer is built never updates while every other
        # check passes, so preflight asserts identity-membership in the optimizer's param
        # groups. A float would make alpha silently non-learnable, which is SPEC §3 row 4
        # ("alpha = 0 fixed") in disguise even at alpha = 1.
        self.alpha = nn.Parameter(torch.empty((), **kw))
        self._alpha_init = float(alpha_init)

        # Ablate-at-eval / preflight hook. None = use the learned parameter.
        self._alpha_override: Optional[float] = None
        # Force the dynamic kernel even when alpha == 0. Preflight check 3 REQUIRES this: if we
        # short-circuited to the static path at alpha == 0 the check would be testing nothing.
        self.force_kernel: bool = True

        # Detached per-forward summary. Scalars only -- keeping Delta_w itself would be
        # B*T*d*W floats (17 MB at B=8, T=512, d=128, W=8) on every layer of every step.
        self._last: Optional[EngagementStats] = None
        self.capture_delta: bool = False
        self._last_delta: Optional[torch.Tensor] = None

        self._perm_gen: Dict[Tuple[str, int], torch.Generator] = {}

        if init_device != "meta":
            # A module must be usable before init_weights() runs, or a test silently operates on
            # uninitialized memory (which is often zeros, making a broken module look inert
            # rather than broken). Self-initialize to the spec state.
            self.init_weights()

    # -- init ---------------------------------------------------------------------------------

    @torch.no_grad()
    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        """SPEC §3: ``U = 0``, ``V`` random against the TRUE contraction, ``alpha = 1`` learnable.

        Every write goes through ``_apply_init``. **Why ``w[...] = x`` is forbidden here:** under
        FSDP each parameter is a ``DTensor``, and an indexed in-place assignment lowers to
        ``aten.fill_.Tensor``, for which no sharding strategy is registered ::

            NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy
            registered.

        That killed submitted run ``run_019fbf9f`` at ``TRAINING_ITSELF_FAILED`` after every
        local test passed, because a single-process CPU build never produces a ``DTensor`` at
        all. ``_apply_init`` materializes the full tensor, initializes that, and copies back the
        local shard -- the library's own answer to the same problem
        (``short_conv.py:394-411``).
        """
        # V: fan-in is d_model, the TRUE contraction of `V @ h`.
        #
        # NOT kaiming_uniform_, WHICH GETS THE FAN-IN WRONG ON A 3-D PARAMETER and would get it
        # wrong here the moment anyone reshapes V to (R, d, 1) or folds the W factor into V.
        # torch derives fan_in = size(1) * receptive_field. Spelled out against d_model so the
        # shape cannot mislead the helper (SPEC §5.3). nn.Linear's own default,
        # kaiming_uniform_(a=sqrt(5)), reduces to U(-1/sqrt(fan_in), +1/sqrt(fan_in)); we
        # reproduce that closed form rather than calling it.
        bound = 1.0 / math.sqrt(self.d_model)
        _apply_init(nn.init.uniform_, self.V.weight, a=-bound, b=bound, generator=generator)

        # U = 0 exactly. This is the LoRA-style warm start: Delta_w == 0 at step 0, so the arm
        # *is* the static arm at init, but dL/dU != 0 immediately (measured 3.08e-02 by R5), so
        # U moves at step 1 and V starts receiving gradient at step 2. ||V.grad|| == 0 at step 0
        # is CORRECT and expected -- see preflight check 5b, which asserts both halves.
        _apply_init(nn.init.zeros_, self.U.weight)

        # alpha = 1, learnable. NOT zero: zeroing BOTH legs is the exact saddle.
        _apply_init(nn.init.constant_, self.alpha, val=self._alpha_init)

    # -- alpha plumbing -----------------------------------------------------------------------

    @property
    def effective_alpha(self) -> torch.Tensor:
        if self._alpha_override is None:
            return self.alpha
        return torch.as_tensor(
            self._alpha_override, dtype=self.alpha.dtype, device=self.alpha.device
        )

    def set_alpha_override(self, value: Optional[float]) -> None:
        self._alpha_override = None if value is None else float(value)

    # -- the S2 control -----------------------------------------------------------------------

    def _permute(self, z: torch.Tensor) -> torch.Tensor:
        """Shuffle the conditioning stream along the sequence axis. This IS arm S2.

        S2 has **identical parameters, identical FLOPs and the identical kernel** to S4; the only
        difference is that ``z`` no longer refers to the token whose filter it produces. It is
        therefore the only arm that can separate "input-dependent local composition" from "one
        more multiplicative degree of freedom" -- which is the live confound, not capacity
        (SPEC §1.1). The decision rule is pre-registered: **if S4 beats S1 but does not beat S2,
        the hypothesis is unsupported.**

        Two design decisions, both load-bearing:

        **(a) The permutation is INDEPENDENT PER SEQUENCE (per batch element).** A single
        permutation shared across the batch would still be content-free, but a per-sequence draw
        is strictly stronger and costs nothing.

        **(b) The permutation is REDRAWN EVERY FORWARD, and this is not a stylistic choice.**
        A *fixed* permutation is a learnable positional code: ``pi`` is then a deterministic
        map, so position ``t`` learns that it always reads position ``pi(t)``, and the arm
        recovers position-dependent -- indeed acausally position-dependent -- filters. That is
        the opposite of a control.

        Redrawing also neutralizes the one real objection to this control, which is that a full
        permutation is **acausal**: ``z'[b,t]`` can come from ``t' > t``, so in principle the
        filter at ``t`` sees a future token and MQAR's answer could leak into the residual
        stream. The leak is not exploitable *because* ``pi`` is redrawn: the model cannot know
        which future position it is reading, ``P(pi(t) = t+1) = 1/T`` (~1/512 at our primary
        operating point), and the identity of the source varies independently every batch. So
        the future content arrives as noise with no learnable decoder, not as signal. With a
        fixed ``pi`` the same leak would be fully exploitable. Redrawing is what makes the
        control sound.

        ``permute_mode="causal_prefix"`` is available as a belt-and-braces variant that samples
        ``j <= t`` with replacement (strictly causal, but not a permutation -- no strict
        permutation of ``0..T-1`` can satisfy ``pi(t) <= t`` for all ``t``, since that forces
        the identity by induction). Default is ``"full"``, matching SPEC §1.
        """
        B, T, _ = z.shape
        g = self._generator(z.device)
        if self.permute_mode == "full":
            # argsort of uniform noise: one vectorized kernel, a valid permutation per row, and
            # device-agnostic (torch.randperm is per-row and CPU-generator-bound).
            idx = torch.argsort(torch.rand(B, T, generator=g, device=z.device), dim=1)
        elif self.permute_mode == "causal_prefix":
            lim = torch.arange(1, T + 1, device=z.device, dtype=z.dtype).view(1, T)
            idx = (torch.rand(B, T, generator=g, device=z.device) * lim).long().clamp_(0, T - 1)
        else:
            raise ValueError(f"unknown permute_mode '{self.permute_mode}'")
        return z.gather(1, idx.unsqueeze(-1).expand(B, T, z.shape[-1]))

    def _generator(self, device: torch.device) -> torch.Generator:
        key = (device.type, device.index if device.index is not None else -1)
        g = self._perm_gen.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.permute_seed)
            self._perm_gen[key] = g
        return g

    def reset_permutation(self) -> None:
        """Rewind the permutation stream to its seed.

        Needed by any check that forwards the SAME batch twice and compares, because for arm S2
        the permutation is redrawn every forward by design (see :meth:`_permute`), so two forwards
        of one batch legitimately differ. Preflight check 9b measured that difference as a
        3.48e-05 relative residual on S2 -- correct behaviour, but it makes a naive reversibility
        assertion flap. Rewinding makes the comparison deterministic *without* weakening the
        control, since training never calls this.
        """
        self._perm_gen.clear()

    # -- forward ------------------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        """:param h: ``(B, T, d)`` **normalized block input**.

        :returns: ``(B, T, n_streams, d, W)`` when ``n_streams > 1``, else ``(B, T, d, W)``.
            ``None`` only when ``alpha`` is exactly 0 *and* ``force_kernel`` is False.
        """
        alpha = self.effective_alpha
        if not self.force_kernel and float(alpha.detach()) == 0.0:
            self._last = None
            return None

        z = self.V(h)
        if self.nonlinear:
            z = F.silu(z)
        if self.permute_z:
            z = self._permute(z)
        delta = self.U(z)  # (B, T, n_streams*W*d)
        B, T = h.shape[0], h.shape[1]
        if self.n_streams == 1:
            delta = delta.view(B, T, self.d_model, self.width)
        else:
            delta = delta.view(B, T, self.n_streams, self.d_model, self.width)
        return alpha * delta

    # -- readout ------------------------------------------------------------------------------

    @torch.no_grad()
    def record(self, scaled_delta: Optional[torch.Tensor], a_norm: float, name: str = "") -> None:
        """Store the detached per-forward engagement summary. Called by the owning mixer."""
        if scaled_delta is None:
            self._last = EngagementStats(
                name=name,
                engagement=0.0,
                input_dependence=0.0,
                u_norm=float(self.U.weight.norm()),
                alpha=float(self.effective_alpha),
                n_positions=0,
            )
            self._last_delta = None
            return
        sd = scaled_delta.detach()
        flat = sd.reshape(sd.shape[0], sd.shape[1], -1)
        n_pos = flat.shape[0] * flat.shape[1]
        # rms over positions of the per-position Frobenius norm, over ||a||_F
        eng = float(flat.pow(2).sum(-1).mean().sqrt()) / a_norm if a_norm > 0 else float("inf")
        # w = a + alpha*Delta_w; the std over positions of w equals the std of alpha*Delta_w,
        # since a is position-independent. The denominator needs |w|, so it is supplied by the
        # mixer via a_norm's companion -- here we use the perturbation's own scale, which is the
        # quantity that must be nonzero for the filter to be input-DEPENDENT rather than merely
        # input-shifted. See EngagementStats.input_dependence.
        std_pos = sd.reshape(-1, *sd.shape[2:]).std(dim=0, unbiased=False).mean()
        mean_abs = sd.abs().mean()
        dep = float(std_pos / mean_abs) if float(mean_abs) > 0 else 0.0
        self._last = EngagementStats(
            name=name,
            engagement=eng,
            input_dependence=dep,
            u_norm=float(self.U.weight.norm()),
            alpha=float(self.effective_alpha),
            n_positions=n_pos,
        )
        self._last_delta = sd if self.capture_delta else None

    @property
    def last(self) -> Optional[EngagementStats]:
        return self._last

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, rank={self.rank}, width={self.width}, "
            f"n_streams={self.n_streams}, permute_z={self.permute_z}, "
            f"permute_mode={self.permute_mode}, nonlinear={self.nonlinear}"
        )


class DynamicShortConv(ShortConv):
    """``ShortConv`` with the depthwise filter made input-dependent.

    Subclasses the **released-parity** ``ShortConv`` (float64 parity to ``Lfm2ShortConv``, 89
    tests) rather than reimplementing it, so ``in_proj`` / ``out_proj`` / ``conv`` keep their
    exact parameter *names* and shapes. That is what makes preflight check 6 (shared parameters
    bit-identical across arms under ``torch.equal``) expressible at all: S1's
    ``blocks.3.sequence_mixer.in_proj.gate_proj.weight`` and S4's are the same key.

    :param conv_activation: **exists only so preflight check 12 has something real to catch.**
        The released LFM2 conv path has no activation; this fork's ``CausalConv1d`` defaults to
        ``activation="silu"``, which is a different operator that trains happily. Leave it
        ``None``. Check 12 detects a rogue activation *numerically*, by comparing the mixer's own
        conv against an independent reference, so it also catches a silu introduced somewhere
        this flag does not reach.
    """

    def __init__(
        self,
        *,
        d_model: int,
        kernel_size: int = 3,
        rank: int = 16,
        alpha_init: float = 1.0,
        permute_z: bool = False,
        permute_mode: PermuteMode = "full",
        permute_seed: int = 0,
        nonlinear: bool = False,
        conv_activation: Optional[str] = None,
        gate_structure: str = "dense",
        gate_rank: Optional[int] = None,
        gate_groups: Optional[int] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model,
            kernel_size=kernel_size,
            gate_structure=gate_structure,  # type: ignore[arg-type]
            gate_rank=gate_rank,
            gate_groups=gate_groups,
            bias=bias,
            use_fla=False,  # plain nn.Conv1d: the correct operator, and it runs on CPU
            dtype=dtype,
            init_device=init_device,
        )
        self.conv_activation = conv_activation
        self.dyn = DynamicFilterGen(
            d_model=d_model,
            rank=rank,
            width=kernel_size,
            n_streams=1,
            alpha_init=alpha_init,
            permute_z=permute_z,
            permute_mode=permute_mode,
            permute_seed=permute_seed,
            nonlinear=nonlinear,
            dtype=dtype,
            init_device=init_device,
        )

    @property
    def static_filter(self) -> torch.Tensor:
        """``(d, W)``. ``self.conv.weight`` is ``(d, 1, W)`` for a depthwise conv."""
        return self.conv.weight.view(self.d_model, self.kernel_size)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """:param x: ``(B, T, d)`` -- the **normalized block input**, which is also ``h``."""
        del kwargs
        if cu_doc_lens is not None:
            # A k-tap filter reading across a document boundary is a different operator. MQAR
            # sequences are single documents, so rather than half-implement the segmented
            # dynamic kernel we refuse.
            raise NotImplementedError("cu_doc_lens is not supported by DynamicShortConv")
        pre_gate, post_gate, value = self.in_proj(x)
        u = pre_gate * value
        a = self.static_filter
        delta = self.dyn(x)
        if delta is None:
            z = depthwise_causal_conv_static(u, a)
        else:
            z = depthwise_causal_conv_dynamic(u, a.unsqueeze(0).unsqueeze(0) + delta)
        self.dyn.record(delta, a_norm=float(a.detach().norm()))
        if self.conv_activation is not None:
            z = F.silu(z)  # WRONG OPERATOR ON PURPOSE -- see conv_activation's docstring.
        return self.out_proj(post_gate * z)

    @torch.no_grad()
    def init_weights(self, **kwargs) -> None:  # type: ignore[override]
        """Initialize the static block exactly as S1 does, then the generator separately.

        ``generator`` is consumed by ``ShortConv.init_weights`` in the same draw order as S1's,
        which is why the shared tensors come out bit-identical. The generator's own tensors draw
        from a **separate** stream (``dyn_generator``), because a single sequential RNG stream
        diverges at the first new tensor and every subsequent draw in the arm is then misaligned
        -- which is R7 FP1 and voids the paired-seed power analysis. See ``arms.py``.
        """
        dyn_generator = kwargs.pop("dyn_generator", None)
        super().init_weights(**kwargs)
        self.dyn.init_weights(generator=dyn_generator)


class DynamicQKVConv(nn.Module):
    """Arm S3: a dynamic depthwise causal conv on Q, K and V inside an attention block.

    This is the *ungated* slot -- the one the cited paper actually measured (a residual conv on
    Q/K/V), where the filter keeps all ``W`` degrees of freedom instead of surrendering 2 to the
    B/C gates. R5 F2(ii) is explicit that LFM2's gated slot is strictly *less* favourable to a
    dynamic filter than the slot the published effect was measured in, so S3 is the arm that
    tests the mechanism where the literature says it works.

    Two deliberate deviations from the paper, both to protect preflight check 3:

    * **Identity-tap, not residual.** The paper writes ``q <- q + conv(q)``. With an identity-tap
      init that would give ``q <- 2q`` at step 0, so S3 would NOT reduce to S1 at ``alpha = 0``
      and check 3 would be untestable for this arm. We use ``q <- conv(q)`` with
      ``a[:, -1] = 1``, which *is* exactly the identity at init.
    * **One shared conditioning stream ``V: d -> R``, three filters from one ``U: R -> 3*W*d``.**
      Cheaper, and it keeps the "one generator per mechanism site" accounting that check 7 uses.

    :param d_model: model width. Q/K/V are all ``(B, T, d)`` before head reshaping, so one
        depthwise filter per stream over ``d`` channels.
    """

    def __init__(
        self,
        *,
        d_model: int,
        kernel_size: int = 3,
        rank: int = 16,
        alpha_init: float = 1.0,
        permute_z: bool = False,
        permute_mode: PermuteMode = "full",
        permute_seed: int = 0,
        nonlinear: bool = False,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        kw = {"dtype": dtype, "device": init_device}
        # Three static filters, one per stream, stored as a single (3, d, W) parameter so the
        # identity-tap init is one _apply_init call.
        self.filters = nn.Parameter(torch.empty(3, d_model, kernel_size, **kw))
        self.dyn = DynamicFilterGen(
            d_model=d_model,
            rank=rank,
            width=kernel_size,
            n_streams=3,
            alpha_init=alpha_init,
            permute_z=permute_z,
            permute_mode=permute_mode,
            permute_seed=permute_seed,
            nonlinear=nonlinear,
            dtype=dtype,
            init_device=init_device,
        )
        if init_device != "meta":
            self.init_weights()

    @torch.no_grad()
    def init_weights(self, generator: Optional[torch.Generator] = None, **kwargs) -> None:
        """Identity-tap filters, via ``_apply_init``.

        ``w[:, :, -1] = 1.0`` inside ``init_weights`` is *exactly* the pattern that lowers to
        ``aten.fill_.Tensor`` and killed ``run_019fbf9f`` under FSDP, so the closure is handed to
        ``_apply_init`` and applied to a materialized full tensor instead.
        """
        dyn_generator = kwargs.pop("dyn_generator", generator)

        def _identity_tap(w: torch.Tensor) -> None:
            w.zero_()
            w[:, :, -1] = 1.0

        _apply_init(_identity_tap, self.filters)
        self.dyn.init_weights(generator=dyn_generator)

    def forward(
        self, h: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:param h: ``(B, T, d)`` normalized block input -- the conditioning signal.
        :param q, k, v: ``(B, T, d)`` projections, pre head-reshape.
        """
        delta = self.dyn(h)  # (B, T, 3, d, W) or None
        a = self.filters  # (3, d, W)
        outs = []
        for i, s in enumerate((q, k, v)):
            if delta is None:
                outs.append(depthwise_causal_conv_static(s, a[i]))
            else:
                outs.append(
                    depthwise_causal_conv_dynamic(
                        s, a[i].unsqueeze(0).unsqueeze(0) + delta[:, :, i]
                    )
                )
        self.dyn.record(delta, a_norm=float(a.detach().norm()))
        return outs[0], outs[1], outs[2]


# ---------------------------------------------------------------------------------------------
# Model-level readout helpers
# ---------------------------------------------------------------------------------------------


def iter_generators(model: nn.Module) -> List[Tuple[str, DynamicFilterGen]]:
    """Every :class:`DynamicFilterGen` in ``model``, in registration order, with its qualified
    name. This is what preflight check 7c counts -- and the reason it counts *modules* rather
    than trusting a parameter total is that an exact total can hide two offsetting errors."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DynamicFilterGen)]


def engagement_report(model: nn.Module) -> List[EngagementStats]:
    """Per-generator engagement from the most recent forward.

    **Report per layer, never averaged.** Depth-scaled ``out_proj`` init means late layers start
    smaller, so a mean over layers can sit above the abort floor while most layers are dead.
    """
    out: List[EngagementStats] = []
    for name, gen in iter_generators(model):
        st = gen.last
        if st is None:
            out.append(
                EngagementStats(
                    name=name,
                    engagement=float("nan"),
                    input_dependence=float("nan"),
                    u_norm=float(gen.U.weight.norm()),
                    alpha=float(gen.effective_alpha),
                    n_positions=0,
                )
            )
        else:
            out.append(
                EngagementStats(
                    name=name,
                    engagement=st.engagement,
                    input_dependence=st.input_dependence,
                    u_norm=st.u_norm,
                    alpha=st.alpha,
                    n_positions=st.n_positions,
                )
            )
    return out


def set_alpha_override(model: nn.Module, value: Optional[float]) -> int:
    """Force ``alpha`` on every generator. Returns how many were touched.

    Two users: preflight check 3 (``alpha = 0`` must reproduce the static path) and the
    ablate-at-eval readout ``Delta_loss = loss(alpha=0) - loss(alpha_hat)``, which is the only
    measurement in the program that separates "bug" from "redundant" from "harmful".
    """
    n = 0
    for _, gen in iter_generators(model):
        gen.set_alpha_override(value)
        n += 1
    return n


def clear_alpha_override(model: nn.Module) -> int:
    return set_alpha_override(model, None)


def reset_permutations(model: nn.Module) -> int:
    """Rewind every generator's permutation stream. See
    :meth:`DynamicFilterGen.reset_permutation`. Only for checks that re-forward one batch."""
    n = 0
    for _, gen in iter_generators(model):
        gen.reset_permutation()
        n += 1
    return n


def dyn_param_names(model: nn.Module) -> List[str]:
    """Qualified names of every parameter belonging to a generator.

    Used to build the no-weight-decay group. R7 FN3: with ``U`` starting at exactly 0 and
    ``alpha`` a scalar, weight decay is a race the mechanism can lose -- a rising-then-falling
    ``||U||`` with ``E_l`` falling is the unambiguous signature (check 10).
    """
    names: List[str] = []
    for gname, gen in iter_generators(model):
        for pname, _ in gen.named_parameters():
            names.append(f"{gname}.{pname}" if gname else pname)
    return names


def split_param_groups(
    model: nn.Module, weight_decay: float = 0.1
) -> List[Dict[str, object]]:
    """Two optimizer groups: everything, and ``{V, U, alpha}`` with ``weight_decay = 0``.

    Both groups are built from the SAME model object, so preflight check 2's identity test
    (``id(alpha) in {id(p) for g in groups for p in g['params']}``) is meaningful.
    """
    dyn = set(dyn_param_names(model))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if n in dyn else decay).append(p)
    groups: List[Dict[str, object]] = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def bf16_dead_zone_probe(
    d_model: int, width: int, magnitude: float
) -> Dict[str, float]:
    """Preflight check 4: characterise, do not fail.

    ``3.90625e-3 = 2^-8`` is bf16's exact half-ulp at 1.0. At the identity-tap init the
    current-token tap is ``a[:, -1] = 1.0``, so a perturbation smaller than that **rounds away
    entirely** in bf16 and the mechanism cannot move the dominant tap. That is where the
    engagement floor of 1e-3 comes from; it is a physical threshold, not taste.
    """
    a = torch.zeros(d_model, width)
    a[:, -1] = 1.0
    delta = torch.full((d_model, width), magnitude)
    w32 = a + delta
    w16 = (a.to(torch.bfloat16) + delta.to(torch.bfloat16)).float()
    cur_unchanged = bool((w16[:, -1] == 1.0).all())
    hist = w32[:, :-1]
    hist16 = w16[:, :-1]
    rel_hist = (
        float((hist16 - hist).norm() / hist.norm()) if width > 1 and float(hist.norm()) > 0 else 0.0
    )
    return {
        "magnitude": magnitude,
        "bf16_half_ulp_at_1": 2.0**-8,
        "current_tap_unchanged": float(cur_unchanged),
        "history_rel_err": rel_hist,
    }


def static_realizability_residual(
    T: int, width: int, seed: int = 0, dynamic: bool = True
) -> float:
    """Cheap structural version of preflight check 13 (the W=2 theorem).

    The static effective-coefficient family is ``kappa[t,k] = C_t * a_k * B_{t-k}``, which is
    linear in logs, so "is this dynamic realization reachable by some static block" is a linear
    least-squares problem. Returns ``max |log residual|``.

    Verified in ``orch_verify_W_minus_2.py``: **0 at W=2** (an exact reparameterization, 8.3e-16
    -- a constructive realization, not a fit) and **large at W>=3**. Therefore a W=2
    dynamic-vs-static difference exceeding seed noise is a bug, not a result. The lead owns the
    standalone falsification script; this is the structural copy preflight carries.
    """
    g = torch.Generator().manual_seed(seed)
    w_free = torch.rand(T, width, generator=g, dtype=torch.float64) + 0.5
    if not dynamic:
        w_free = (torch.rand(1, width, generator=g, dtype=torch.float64) + 0.5).expand(T, width)
    Bd = torch.rand(T, generator=g, dtype=torch.float64) + 0.5
    Cd = torch.rand(T, generator=g, dtype=torch.float64) + 0.5
    rows, rhs = [], []
    for t in range(T):
        for k in range(width):
            if t - k < 0:
                continue
            r = torch.zeros(T + width + T, dtype=torch.float64)
            r[t] = 1.0  # log C'_t
            r[T + k] = 1.0  # log a'_k
            r[T + width + (t - k)] = 1.0  # log B'_{t-k}
            rows.append(r)
            rhs.append(torch.log(Cd[t] * w_free[t, k] * Bd[t - k]))
    A = torch.stack(rows)
    b = torch.stack(rhs)
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    return float((A @ sol - b).abs().max())


def count_flops_proxy(model: nn.Module) -> Dict[str, int]:
    """A parameter-based FLOP proxy, only to assert S2 == S4 exactly.

    S2 differs from S4 by a gather on ``z``, which moves memory but performs no multiply-add. If
    these two ever diverge, the control is no longer matched and the primary contrast is void.
    """
    total = sum(p.numel() for p in model.parameters())
    dyn = sum(p.numel() for _, g in iter_generators(model) for p in g.parameters())
    return {"params_total": total, "params_dynamic": dyn, "flops_proxy_6x": 6 * total}


def named_shared_params(a: nn.Module, b: nn.Module) -> Iterable[str]:
    """Parameter names present in BOTH models with the same shape -- check 6's domain."""
    sa = {n: p.shape for n, p in a.named_parameters()}
    sb = {n: p.shape for n, p in b.named_parameters()}
    return sorted(n for n in sa.keys() & sb.keys() if sa[n] == sb[n])
