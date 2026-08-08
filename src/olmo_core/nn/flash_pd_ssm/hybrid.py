"""Additive helpers for selecting Flash PD inside an existing Mamba-3 hybrid shell."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.mamba3 import Mamba3Config, Mamba3MixerConfig

from .mamba3_flash import Mamba3FlashPDSSMMixerConfig
from .mixer import FlashPDSSMMixerConfig

__all__ = [
    "StateTracker",
    "StateTrackerConfig",
    "mamba3_flash_pd_mixed_olmo3_370m",
    "mamba3_flash_pd_olmo3_370m",
    "mamba3_olmo3_370m_with_state_tracker",
    "replace_mamba3_with_fused_flash_pd",
    "replace_state_tracker",
]


class StateTracker(StrEnum):
    """State-tracking mixer selected for named recurrent slots."""

    mamba3 = "mamba3"
    """Preserve the existing Mamba-3 config and mixer behavior exactly."""

    flash_pd = "flash_pd"
    """Replace only explicitly named Mamba-3 slots with Flash PD-SSM."""

    hybrid = "hybrid"
    """Retain Mamba-3 and replace every third recurrent occurrence with Flash PD-SSM."""


def _mirror_mamba_mixer(
    mixer: Mamba3MixerConfig,
    *,
    d_model: int,
) -> FlashPDSSMMixerConfig:
    square_root = math.isqrt(d_model)
    n_heads = square_root if square_root * square_root == d_model else mixer.n_heads
    head_state = d_model // n_heads
    return FlashPDSSMMixerConfig(
        n_heads=n_heads,
        d_state=head_state,
        dictionary_size=n_heads,
        dtype=mixer.dtype,
    )


def replace_state_tracker(
    config: Mamba3Config,
    *,
    state_tracker: StateTracker | str = StateTracker.mamba3,
    recurrent_slots: Sequence[str] = ("mamba3",),
    flash_pd_config: Optional[FlashPDSSMMixerConfig] = None,
) -> Mamba3Config:
    """
    Select a state tracker without modifying the supplied Mamba-3 config.

    The default returns ``config`` itself, not a reconstructed approximation, so config JSON,
    hashes, parameter/state names, kernel choices, and checkpoint behavior are exactly the
    pre-existing Mamba-3 behavior. Selecting ``flash_pd`` deep-copies the config and changes only
    ``sequence_mixer`` in the named recurrent block entries.

    The outer :class:`~olmo_core.nn.mamba3.Mamba3Config`, block pattern, norms, feed-forwards,
    attention blocks, LM head, checkpoint policy, and model class remain unchanged. Mixer weights
    are necessarily state-dict incompatible: Flash PD has dictionary/selector/complex-diagonal
    parameters where Mamba-3 has SSD projections and rotation parameters.

    :param config: Existing Mamba-3 hybrid config.
    :param state_tracker: ``"mamba3"`` (default), ``"flash_pd"``, or ``"hybrid"``.
    :param recurrent_slots: Named entries in ``config.block`` eligible for replacement.
    :param flash_pd_config: Explicit replacement config. When omitted, square model widths use
        the Flash PD ``H=N=K=sqrt(d_model)`` scaling; other widths retain the Mamba head count.

    :returns: The unchanged input for ``mamba3``, or an isolated copied config for ``flash_pd``.

    :raises OLMoConfigurationError: If a named slot is missing or is not a Mamba-3 mixer.
    """
    tracker = StateTracker(state_tracker)
    if tracker == StateTracker.mamba3:
        return config
    if not isinstance(config.block, dict):
        raise OLMoConfigurationError(
            "Flash PD replacement requires a named block dictionary and block_pattern"
        )
    if not recurrent_slots:
        raise OLMoConfigurationError("recurrent_slots must contain at least one named block")

    replaced = config.copy()
    assert isinstance(replaced.block, dict)
    blocks = dict(replaced.block)
    flash_slots: dict[str, str] = {}
    for slot in recurrent_slots:
        if slot not in blocks:
            raise OLMoConfigurationError(
                f"recurrent slot '{slot}' is absent; available slots are {sorted(blocks)}"
            )
        block = blocks[slot]
        mixer = block.sequence_mixer
        if not isinstance(mixer, Mamba3MixerConfig):
            raise OLMoConfigurationError(
                f"recurrent slot '{slot}' contains {type(mixer).__name__}, not Mamba3MixerConfig"
            )
        replacement = (
            flash_pd_config.copy()
            if flash_pd_config is not None
            else _mirror_mamba_mixer(mixer, d_model=config.d_model)
        )
        if tracker == StateTracker.flash_pd:
            blocks[slot] = block.replace(sequence_mixer=replacement)
        else:
            flash_slot = "flash_pd" if slot == "mamba3" else f"{slot}_flash_pd"
            if flash_slot in blocks:
                raise OLMoConfigurationError(
                    f"generated Flash PD slot '{flash_slot}' already exists"
                )
            blocks[flash_slot] = block.replace(sequence_mixer=replacement)
            flash_slots[slot] = flash_slot

    if tracker == StateTracker.flash_pd:
        return replaced.replace(block=blocks)

    if replaced.block_pattern is None:
        raise OLMoConfigurationError("hybrid state tracking requires an explicit block_pattern")
    counts = {slot: 0 for slot in recurrent_slots}
    pattern = list(replaced.block_pattern)
    for idx, slot in enumerate(pattern):
        if slot not in counts:
            continue
        counts[slot] += 1
        if counts[slot] % 3 == 0:
            pattern[idx] = flash_slots[slot]
    if not any(count >= 3 for count in counts.values()):
        raise OLMoConfigurationError(
            "hybrid state tracking needs at least three recurrent block occurrences"
        )
    return replaced.replace(block=blocks, block_pattern=pattern)


@dataclass
class StateTrackerConfig(Config):
    """
    Serializable additive selector for a Mamba-3 hybrid's named recurrent slots.

    :param state_tracker: Existing Mamba-3, Flash PD replacement, or mixed hybrid.
    :param recurrent_slots: Named recurrent block entries to replace.
    :param flash_pd_config: Optional explicit Flash PD mixer config.
    """

    state_tracker: StateTracker = StateTracker.mamba3
    recurrent_slots: tuple[str, ...] = ("mamba3",)
    flash_pd_config: Optional[FlashPDSSMMixerConfig] = None

    def apply(self, config: Mamba3Config) -> Mamba3Config:
        """
        Apply this selection to a Mamba-3 config.

        :param config: Existing hybrid config.

        :returns: The selected hybrid config.
        """
        return replace_state_tracker(
            config,
            state_tracker=self.state_tracker,
            recurrent_slots=self.recurrent_slots,
            flash_pd_config=self.flash_pd_config,
        )


def mamba3_olmo3_370m_with_state_tracker(
    vocab_size: int,
    *,
    state_tracker: StateTracker | str = StateTracker.mamba3,
    recurrent_slots: Sequence[str] = ("mamba3",),
    flash_pd_config: Optional[FlashPDSSMMixerConfig] = None,
    **mamba3_kwargs,
) -> Mamba3Config:
    """
    Mirror ``Mamba3Config.mamba3_olmo3_370M`` with an explicit state-tracker switch.

    With the default ``state_tracker="mamba3"`` this is exactly the existing factory output.
    Flash PD replacement is opt-in and changes only the named recurrent slots.

    :param vocab_size: Vocabulary size passed to the existing Mamba-3 preset.
    :param state_tracker: Existing Mamba-3, Flash PD replacement, or mixed hybrid.
    :param recurrent_slots: Named recurrent block entries to replace.
    :param flash_pd_config: Optional explicit replacement config.
    :param mamba3_kwargs: Unmodified keyword arguments for the existing preset.

    :returns: A Mamba-3/OLMo-3 hybrid shell with the selected recurrent mixer.
    """
    base = Mamba3Config.mamba3_olmo3_370M(vocab_size=vocab_size, **mamba3_kwargs)
    return replace_state_tracker(
        base,
        state_tracker=state_tracker,
        recurrent_slots=recurrent_slots,
        flash_pd_config=flash_pd_config,
    )


def mamba3_flash_pd_mixed_olmo3_370m(
    vocab_size: int,
    *,
    flash_pd_config: Optional[FlashPDSSMMixerConfig] = None,
    **mamba3_kwargs,
) -> Mamba3Config:
    """
    Build the baseline 370M ``[Mamba-3, Mamba-3, Flash PD, attention]`` mixed model.

    This is retained only as an ablation. The target fused implementation is
    :func:`mamba3_flash_pd_olmo3_370m`.
    """
    mamba3_kwargs.setdefault("rotation_block_size", 3)
    mamba3_kwargs.setdefault("rotation_scan_impl", "quaternion")
    if flash_pd_config is None:
        # At d_model=1024 this is within 1% of one 370M Mamba-3 mixer's parameter count
        # (3,804,320 vs 3,769,184) while keeping N within the one-warp Triton limit.
        flash_pd_config = FlashPDSSMMixerConfig(
            n_heads=16,
            d_state=29,
            dictionary_size=16,
        )
    return mamba3_olmo3_370m_with_state_tracker(
        vocab_size,
        state_tracker=StateTracker.hybrid,
        flash_pd_config=flash_pd_config,
        **mamba3_kwargs,
    )


def mamba3_flash_pd_olmo3_370m(
    vocab_size: int,
    *,
    fused_config: Optional[Mamba3FlashPDSSMMixerConfig] = None,
    **mamba3_kwargs,
) -> Mamba3Config:
    """
    Build the 370M shell with one fused Mamba-3 + Flash-PD recurrent mixer type.

    Every recurrent slot uses Flash-PD's collision-preserving sparse transition together with
    Mamba-3's complex phase, exponential-trapezoidal input update, and rank-4 MIMO projections.
    Separate Mamba and Flash-PD layers are not interleaved.
    """
    base = Mamba3Config.mamba3_olmo3_370M(vocab_size=vocab_size, **mamba3_kwargs)
    return replace_mamba3_with_fused_flash_pd(base, fused_config=fused_config)


def replace_mamba3_with_fused_flash_pd(
    config: Mamba3Config,
    *,
    fused_config: Optional[Mamba3FlashPDSSMMixerConfig] = None,
) -> Mamba3Config:
    """Replace every Mamba recurrent slot with the single fused Flash-PD mixer type."""
    if not isinstance(config.block, dict) or config.block_pattern is None:
        raise OLMoConfigurationError(
            "fused Mamba-3 + Flash-PD requires named blocks and an explicit block_pattern"
        )
    if "mamba3" not in config.block:
        raise OLMoConfigurationError("base Mamba-3 config has no 'mamba3' recurrent block")
    if fused_config is None:
        # Re-tuned with the learned MIMO write/read factors included: N=20 is the closest
        # one-warp width and differs by 0.265% from the existing 370M Mamba-3 mixer.
        fused_config = Mamba3FlashPDSSMMixerConfig(
            n_heads=8,
            head_dim=128,
            d_state=20,
            n_groups=1,
            mimo_rank=4,
            dictionary_size=16,
        )

    blocks = dict(config.block)
    blocks["mamba3_flash_pd"] = blocks["mamba3"].replace(sequence_mixer=fused_config.copy())
    pattern = [
        "mamba3_flash_pd" if block_name == "mamba3" else block_name
        for block_name in config.block_pattern
    ]
    return config.replace(block=blocks, block_pattern=pattern)
