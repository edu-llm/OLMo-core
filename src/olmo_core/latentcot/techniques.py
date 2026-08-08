"""
The latent-reasoning post-training techniques, as a named catalog.

The five experiment arms A0–A4 were *comparisons*. This module re-presents the same recipes as
**techniques you select for a real fine-tune**, so that when the experiment says which one wins,
naming it is a flag rather than a code change.

Where this sits in the pipeline: **post-training**, at the SFT stage. Every technique here starts
from a pretrained checkpoint and fine-tunes it; none of them touch pretraining.

.. important::
   **CODI needs reasoning traces in the data.** The ``codi-*`` techniques train a teacher branch on
   the written-out chain of thought and align the latent thoughts to its ``<distill>`` hidden state,
   so each training example needs *both* views. Post-training data with no CoT annotation gives the
   distillation term nothing to distill from, and the latent techniques degrade to a slower
   ``no-cot``. ``explicit-cot`` and ``no-cot`` are the two that work on ordinary SFT data.

The catalog is deliberately wider than the five arms, because two combinations were implemented and
wired to no arm — R1's anti-collapse entropy floor, and the R2 nearest-embedding regularizer. Both
are reachable here by name.

===================  ====  ===========================================================
Technique            Arm   What it does
===================  ====  ===========================================================
``explicit-cot``     A0    CE on the written-out CoT. No latent reasoning at all.
``no-cot``           A1    CE on the direct ``question -> answer`` view.
``codi``             A2    ``K`` continuous thoughts, unconstrained.
``codi-r1``          A3    ``codi`` + the vocabulary-manifold regularizer.
``codi-r1-entropy``  --    ``codi-r1`` with R1's entropy floor on (anti-collapse).
``codi-r2``          --    ``codi`` + pull to the nearest token embedding.
``codi-l2``          A4    ``codi`` + plain L2 on thought norm.
===================  ====  ===========================================================

Every technique converts to the :class:`~olmo_core.latentcot.arms.Arm` the loss and training loop
already take (:func:`as_arm`), so selecting one changes no code downstream.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional

from .arms import DEFAULT_GAMMA, DEFAULT_K, Arm
from .loss import VocabReg

__all__ = [
    "Technique",
    "TECHNIQUES",
    "LATENT_TECHNIQUES",
    "get_technique",
    "as_arm",
    "describe_techniques",
]


@dataclass(frozen=True)
class Technique:
    """
    One selectable post-training recipe.

    Holds every field that defines the *method*. Everything else about a run — learning rate,
    steps, batch size, the base checkpoint — is a property of the run, not the technique, and
    stays on the command line.
    """

    name: str
    """The catalog key, e.g. ``"codi-r1"``."""

    arm_mode: str
    """``"explicit_cot"``, ``"no_cot"`` or ``"codi"`` — which loss path runs."""

    num_continuous_thoughts: int
    """``K``. Ignored by the two non-latent techniques."""

    vocab_reg: VocabReg
    """Which thought regularizer, or ``"none"``."""

    vocab_reg_weight: float
    """Its weight. ``0.0`` with ``vocab_reg="none"``."""

    vocab_reg_entropy_floor: float
    """R1's anti-collapse floor in nats; ``0.0`` = off."""

    distill_weight: float
    """Weight on the teacher->student ``<distill>`` alignment. Only used by ``codi`` modes."""

    summary: str
    """One line, for ``--list-techniques``."""

    arm: Optional[str] = None
    """Which experiment arm this was, or ``None`` if it was never one."""

    @property
    def is_latent(self) -> bool:
        """Whether this technique actually does latent reasoning."""
        return self.arm_mode == "codi"

    @property
    def needs_cot_data(self) -> bool:
        """
        Whether the training data must carry explicit reasoning traces.

        True for every latent technique (the teacher branch trains on the written CoT and is what
        the thoughts are distilled from) and for ``explicit-cot`` (which trains on it directly).
        Only ``no-cot`` runs on data with no traces.
        """
        return self.arm_mode in ("codi", "explicit_cot")


def _t(
    name: str,
    arm_mode: str,
    vocab_reg: VocabReg,
    vocab_reg_weight: float,
    summary: str,
    *,
    entropy_floor: float = 0.0,
    arm: Optional[str] = None,
    k: int = DEFAULT_K,
    distill_weight: float = 1.0,
) -> Technique:
    return Technique(
        name=name,
        arm_mode=arm_mode,
        num_continuous_thoughts=k,
        vocab_reg=vocab_reg,
        vocab_reg_weight=vocab_reg_weight,
        vocab_reg_entropy_floor=entropy_floor,
        distill_weight=distill_weight,
        summary=summary,
        arm=arm,
    )


TECHNIQUES: Dict[str, Technique] = {
    t.name: t
    for t in (
        _t(
            "explicit-cot",
            "explicit_cot",
            "none",
            0.0,
            "Cross-entropy on the written-out chain of thought. No latent reasoning; the "
            "readable upper anchor, and the fallback if no latent technique wins.",
            arm="A0",
        ),
        _t(
            "no-cot",
            "no_cot",
            "none",
            0.0,
            "Cross-entropy on the direct question->answer view. No reasoning of either kind; "
            "the only technique here that needs no CoT annotation in the data.",
            arm="A1",
        ),
        _t(
            "codi",
            "codi",
            "none",
            0.0,
            "K continuous thoughts, self-distilled from the explicit-CoT teacher branch, with "
            "no constraint on where the thoughts live. The latent substrate on its own.",
            arm="A2",
        ),
        _t(
            "codi-r1",
            "codi",
            "R1",
            DEFAULT_GAMMA,
            "codi + the vocabulary-manifold regularizer: each thought is pulled toward a soft "
            "mixture of real token embeddings, so it stays in a region the pretrained weights "
            "were fit on. The primary hypothesis of the study.",
            arm="A3",
        ),
        _t(
            "codi-r1-entropy",
            "codi",
            "R1",
            DEFAULT_GAMMA,
            "codi-r1 with R1's entropy floor on, which stops the mixture collapsing onto a "
            "single token. Use if thoughts are seen to lose their superposition.",
            entropy_floor=1.0,
        ),
        _t(
            "codi-r2",
            "codi",
            "R2",
            DEFAULT_GAMMA,
            "codi + a pull toward the single nearest token embedding. The hard-assignment "
            "counterpart of R1: keeps thoughts on the manifold but gives up superposition, so "
            "it is the sharper test of whether the mixture is what matters.",
        ),
        _t(
            "codi-l2",
            "codi",
            "L2",
            DEFAULT_GAMMA,
            "codi + plain L2 on the thought norm. Regularizes without any vocabulary direction, "
            "so it isolates how much of R1's effect is the manifold rather than shrinkage.",
            arm="A4",
        ),
    )
}
"""Every selectable technique, by name."""

LATENT_TECHNIQUES = tuple(name for name, t in TECHNIQUES.items() if t.is_latent)
"""The subset that actually does latent reasoning."""


def get_technique(
    name: str,
    *,
    num_continuous_thoughts: Optional[int] = None,
    vocab_reg_weight: Optional[float] = None,
    vocab_reg_entropy_floor: Optional[float] = None,
    distill_weight: Optional[float] = None,
) -> Technique:
    """
    Look up a technique by name, with optional per-run overrides of its knobs.

    The overrides exist because ``K`` and the regularizer weights are the things a real fine-tune
    will want to tune once a technique is chosen; the *shape* of the technique is not negotiable.

    :param name: A key of :data:`TECHNIQUES`.
    :param num_continuous_thoughts: Override ``K``.
    :param vocab_reg_weight: Override the regularizer weight.
    :param vocab_reg_entropy_floor: Override R1's entropy floor (in nats; ``0.0`` disables).
    :param distill_weight: Override the teacher->student alignment weight.

    :returns: The technique, with any overrides applied.

    :raises KeyError: On an unknown name, listing what is available — a typo here would
        otherwise silently train the wrong method.
    :raises ValueError: If ``K < 1`` is requested for a latent technique.
    """
    if name not in TECHNIQUES:
        raise KeyError(f"unknown technique {name!r}. Available: {', '.join(sorted(TECHNIQUES))}")
    technique = TECHNIQUES[name]
    changes: Dict[str, object] = {}
    if num_continuous_thoughts is not None:
        if technique.is_latent and num_continuous_thoughts < 1:
            raise ValueError(
                f"{name} is a latent technique, so K must be >= 1, got {num_continuous_thoughts}"
            )
        changes["num_continuous_thoughts"] = num_continuous_thoughts
    if vocab_reg_weight is not None:
        changes["vocab_reg_weight"] = vocab_reg_weight
    if vocab_reg_entropy_floor is not None:
        changes["vocab_reg_entropy_floor"] = vocab_reg_entropy_floor
    if distill_weight is not None:
        changes["distill_weight"] = distill_weight
    return replace(technique, **changes) if changes else technique  # type: ignore[arg-type]


def as_arm(technique: Technique) -> Arm:
    """
    Convert a technique to the :class:`~olmo_core.latentcot.arms.Arm` the loss and training loop
    take, so selecting a technique changes nothing downstream.

    ``distill_weight`` is deliberately not carried across: ``Arm`` is the set of *arm-defining*
    fields the confound check whitelists, and the training loop takes the distill weight as its
    own argument. Read it off the technique at the call site.

    :param technique: From :data:`TECHNIQUES` or :func:`get_technique`.
    :returns: An equivalent ``Arm``, named after the technique.
    """
    return Arm(
        name=technique.name,
        arm_mode=technique.arm_mode,
        num_continuous_thoughts=technique.num_continuous_thoughts,
        vocab_reg=technique.vocab_reg,
        vocab_reg_weight=technique.vocab_reg_weight,
        vocab_reg_entropy_floor=technique.vocab_reg_entropy_floor,
    )


def describe_techniques() -> str:
    """
    The catalog as a printable table, for ``--list-techniques``.

    :returns: One block of text, techniques in catalog order.
    """
    lines = []
    for name, t in TECHNIQUES.items():
        provenance = f"arm {t.arm}" if t.arm else "not an arm"
        detail = f"mode={t.arm_mode}"
        if t.is_latent:
            detail += f" K={t.num_continuous_thoughts} reg={t.vocab_reg}"
            if t.vocab_reg != "none":
                detail += f"@{t.vocab_reg_weight:g}"
            if t.vocab_reg_entropy_floor:
                detail += f" entropy_floor={t.vocab_reg_entropy_floor:g}"
        lines.append(f"{name:18s} [{provenance:11s}] {detail}")
        lines.append(f"{'':18s}  {t.summary}")
        if t.needs_cot_data:
            lines.append(f"{'':18s}  requires reasoning traces in the training data")
    return "\n".join(lines)
