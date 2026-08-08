"""
Masked (absorbing-state) diffusion training for a :class:`~olmo_core.nn.transformer.Transformer`.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT. Training a diffusion language model is not
a different training loop. It is the same loop over a different pair of ``(input_ids, labels)``:
some fraction of the tokens are replaced by a ``MASK`` id, the labels are the *unshifted*
originals at exactly those positions and ``ignore_index`` everywhere else, and the loss is the
same cross entropy. So :class:`DiffusionTransformerTrainModule` corrupts the batch and then hands
it to :class:`~.train_module.TransformerTrainModule`, rather than reimplementing micro-batching,
FSDP, the optimizer step or the metrics around it.

Three things about this file are decisions rather than mechanics, and each one has a paper behind
it.

**The labels are not shifted.** Autoregression predicts token ``i+1`` from a prefix, so
:func:`~olmo_core.data.utils.get_labels` shifts left. Diffusion predicts token ``i`` *in place*
from a corrupted whole sequence. Shifting here would train the model to denoise one position to
the left of the mask it can see, which trains something, converges, and is not a diffusion model.
``get_labels`` is bypassed by putting ``labels`` in the batch ourselves, which is the one hook
``TransformerTrainModule.train_batch`` already offers.

**The loss is plain cross entropy, with no ELBO reweighting.** MDLM's objective weights each
sequence's loss by ``1/t``. `Scaling Beyond Masked Diffusion Language Models
<https://arxiv.org/abs/2602.15014>`_ (ICML 2026) reports that replacing that weighting with plain
cross entropy makes the same models 12% more compute efficient, and it is also the only variant
that reuses this repository's existing loss path untouched. Quokka disagrees at the margin -- it
finds the reweighted ELBO ahead at end-of-training and MaskGIT-style unweighted CE ahead early --
so this is a real disagreement in the literature and not settled by us. ``loss_weighting`` exists
to run the other arm; the default follows the more recent and more directly measured result.

**Absorbing corruption on a linear schedule.** Quokka's ablations at 1B/96B tokens find
absorbing-mask corruption beats uniform "by a wide margin", and the linear schedule
``alpha = 1 - t`` both strongest and lowest-variance, with cosine worst. Neither is a free
parameter here for that reason.

ON THE MEANING OF ``t``. Here ``t`` is the probability that a token is masked, so ``t -> 1`` is
fully corrupted and ``alpha = 1 - t`` is the probability a token survives. DeltaFlow uses the
opposite convention (its ``t = 1`` is clean). The models' noise-conditioning projections are
zero-initialised and learned, so nothing is numerically wrong under either reading, but the two
papers cannot be compared without knowing which is which.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.config import Config, DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from ..config import validate_precision_support
from .config import TransformerTrainModuleConfig
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = [
    "DiffusionSchedule",
    "DiffusionLossWeighting",
    "MaskedDiffusionConfig",
    "DiffusionTransformerTrainModule",
    "DiffusionTransformerTrainModuleConfig",
]


class DiffusionSchedule(StrEnum):
    """
    The noise schedule, which maps a draw ``t`` to the probability that a token is masked.
    """

    linear = "linear"
    """
    ``alpha = 1 - t``. Quokka's strongest and lowest-variance schedule, and the default.
    """
    poly2 = "poly2"
    """
    ``alpha = 1 - t^2``. Quokka reports this ahead on HellaSwag specifically while losing overall.
    """
    cosine = "cosine"
    """
    ``alpha = cos(pi t / 2)``. Included to be able to reproduce the comparison; Quokka finds it
    the worst of the three, so do not reach for it as a default.
    """

    def mask_probability(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert a uniform draw into a per-sequence masking probability.

        :param t: Uniform draws in ``(0, 1)``, shape ``(batch_size,)``.

        :returns: Masking probabilities in ``(0, 1)``, shape ``(batch_size,)``.
        """
        if self == DiffusionSchedule.linear:
            return t
        elif self == DiffusionSchedule.poly2:
            return t.square()
        elif self == DiffusionSchedule.cosine:
            return 1.0 - torch.cos(t * torch.pi / 2)
        else:
            raise NotImplementedError(self)


class DiffusionLossWeighting(StrEnum):
    """
    How each sequence's masked-token losses are weighted before being summed.
    """

    none = "none"
    """
    Unweighted cross entropy over masked positions, i.e. the MaskGIT-style objective. 12% more
    compute efficient per `arXiv:2602.15014 <https://arxiv.org/abs/2602.15014>`_, and the only
    option that needs no change to the existing loss path.
    """
    elbo = "elbo"
    """
    The MDLM diffusion ELBO, which weights a sequence's loss by ``1 / t``. Not implemented: it
    needs per-sequence weights, which means ``loss_reduction="none"`` and therefore the
    non-fused loss path plus a reduction written here. Declared so that the default is a choice
    on the record rather than the only thing that occurred to anybody.
    """


@dataclass
class MaskedDiffusionConfig(Config):
    """
    Configuration for the masked-diffusion corruption applied to each batch.
    """

    mask_token_id: int
    """
    The token id to write at masked positions.

    There is no need to grow the embedding matrix for this.
    :meth:`~olmo_core.data.tokenizer.TokenizerConfig.padded_vocab_size` rounds the vocabulary up
    to a multiple of 128, so for dolma2-bpe the ids from 100278 to 100351 are already allocated,
    already trained as unreachable rows, and free. Pass the tokenizer's ``vocab_size``.
    """
    schedule: DiffusionSchedule = DiffusionSchedule.linear
    """
    The noise schedule. See :class:`DiffusionSchedule`.
    """
    loss_weighting: DiffusionLossWeighting = DiffusionLossWeighting.none
    """
    How to weight each sequence's loss. See :class:`DiffusionLossWeighting`.
    """
    min_mask_probability: float = 1e-3
    """
    Lower clamp on the masking probability.

    At exactly zero a sequence contributes no loss at all, and its whole forward and backward pass
    is wasted work. DiffuMamba uses the same 1e-3 floor.
    """
    max_mask_probability: float = 1.0
    """
    Upper clamp on the masking probability. At exactly 1.0 a sequence is entirely masked, which is
    a valid draw -- the model is being asked to generate unconditionally -- so this is not clamped
    below 1 by default.
    """
    antithetic_sampling: bool = True
    """
    Draw ``t`` antithetically: the second half of the batch gets ``1 - t`` of the first half.

    Free variance reduction on the schedule draw, which is otherwise a significant part of the
    gradient noise at small batch sizes. DiffuMamba enables it.
    """

    def __post_init__(self):
        if not 0.0 <= self.min_mask_probability < self.max_mask_probability <= 1.0:
            raise OLMoConfigurationError(
                "expected 0 <= min_mask_probability < max_mask_probability <= 1, got "
                f"{self.min_mask_probability} and {self.max_mask_probability}"
            )
        if self.loss_weighting != DiffusionLossWeighting.none:
            raise OLMoConfigurationError(
                f"loss_weighting='{self.loss_weighting}' is not implemented; see "
                "DiffusionLossWeighting for what implementing it involves"
            )

    def sample_mask_probability(
        self, batch_size: int, *, device: torch.device, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Draw one masking probability per sequence.

        :param batch_size: How many sequences are in the batch.
        :param device: The device to draw on.
        :param generator: Optional generator, for tests that need a fixed draw.

        :returns: Masking probabilities, shape ``(batch_size,)``.
        """
        if self.antithetic_sampling and batch_size > 1:
            half = (batch_size + 1) // 2
            u = torch.rand(half, device=device, generator=generator)
            u = torch.cat([u, 1.0 - u])[:batch_size]
        else:
            u = torch.rand(batch_size, device=device, generator=generator)

        return self.schedule.mask_probability(u).clamp(
            min=self.min_mask_probability, max=self.max_mask_probability
        )

    def corrupt(
        self,
        input_ids: torch.Tensor,
        *,
        scoreable: Optional[torch.Tensor] = None,
        label_ignore_index: int = -100,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply absorbing-state corruption to a batch.

        :param input_ids: The clean token ids, shape ``(batch_size, seq_len)``.
        :param scoreable: Optional boolean mask of positions that may be masked and scored, shape
            ``(batch_size, seq_len)``. Positions that are ``False`` -- padding, or instances a
            filter has dropped -- are left clean and never contribute to the loss.
        :param label_ignore_index: The label value at positions that are not scored.
        :param generator: Optional generator, for tests that need a fixed draw.

        :returns: ``(corrupted_input_ids, labels, mask_probability)``, where ``labels`` is
            *unshifted* and ``mask_probability`` has shape ``(batch_size,)``.
        """
        batch_size = input_ids.shape[0]
        p = self.sample_mask_probability(batch_size, device=input_ids.device, generator=generator)

        masked = torch.rand(
            input_ids.shape, device=input_ids.device, generator=generator
        ) < p.unsqueeze(1)
        if scoreable is not None:
            masked = masked & scoreable

        corrupted = torch.where(masked, self.mask_token_id, input_ids)
        # Unshifted, and `ignore_index` wherever the model can already see the answer. The loss is
        # therefore over exactly the positions that were corrupted, which is what makes the
        # trainer's own `batch_num_tokens_for_loss` the correct denominator without changing it.
        labels = torch.where(masked, input_ids, label_ignore_index)

        return corrupted, labels, p


class DiffusionTransformerTrainModule(TransformerTrainModule):
    """
    A :class:`~.train_module.TransformerTrainModule` that trains a masked diffusion objective.

    Everything about parallelism, micro-batching, checkpointing and metrics is inherited. What
    this adds is the corruption of each batch, and the noise level the model's recurrent mixers
    need to read it.

    .. seealso::
        The module docstring of :mod:`olmo_core.train.train_module.transformer.diffusion` for why
        the labels are unshifted and the loss is unweighted.
    """

    def __init__(self, *args, diffusion: MaskedDiffusionConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        log.info(
            "masked diffusion training: schedule=%s, mask_token_id=%d, antithetic=%s",
            diffusion.schedule,
            diffusion.mask_token_id,
            diffusion.antithetic_sampling,
        )

    def _scoreable_positions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Which positions are eligible to be masked and scored.

        Mirrors what :func:`~olmo_core.data.utils.get_labels` excludes, minus the shift: padding
        via ``attention_mask``, explicitly unlabelled spans via ``label_mask``, and whole
        instances an instance filter has dropped via ``instance_mask``.
        """
        scoreable: Optional[torch.Tensor] = None

        def restrict(m: torch.Tensor) -> None:
            nonlocal scoreable
            scoreable = m if scoreable is None else (scoreable & m)

        if (label_mask := batch.get("label_mask")) is not None:
            restrict(label_mask.bool())
        if (attention_mask := batch.get("attention_mask")) is not None:
            restrict(attention_mask != 0.0)
        if (instance_mask := batch.get("instance_mask")) is not None:
            restrict(instance_mask.bool().unsqueeze(-1).expand_as(batch["input_ids"]))

        return scoreable

    def _corrupt_batch(self, batch: Dict[str, Any]) -> None:
        """
        Replace ``input_ids`` with a corrupted copy in place, and add ``labels`` and
        ``noise_level`` to the batch.

        A batch that already carries ``labels`` is left alone, so that an evaluator which has
        built its own corruption is not silently corrupted twice.
        """
        if "labels" in batch:
            return

        corrupted, labels, p = self.diffusion.corrupt(
            batch["input_ids"],
            scoreable=self._scoreable_positions(batch),
            label_ignore_index=self.label_ignore_index,
        )
        batch["input_ids"] = corrupted
        batch["labels"] = labels
        # Read by noise-conditioned sequence mixers. `Transformer._prepare_inputs` forwards this
        # to every block, and `split_batch` splits it along the batch dimension with everything
        # else, so it stays aligned with its own sequences across micro-batches.
        batch["noise_level"] = p

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._corrupt_batch(batch)
        return super().train_batch(batch, dry_run=dry_run)

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None):
        # An in-loop evaluator measures the same objective the model is trained on, so it needs
        # the same corruption. The draw is random per call, which makes eval CE noisier than an
        # autoregressive run's -- it is a Monte Carlo estimate of an expectation over `t`, not a
        # deterministic quantity. Average over steps before reading anything into a change.
        if labels is None:
            self._corrupt_batch(batch)
        return super().eval_batch(batch, labels=labels)


@dataclass
class DiffusionTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """
    A configuration class for building :class:`DiffusionTransformerTrainModule`.

    Identical to :class:`~.config.TransformerTrainModuleConfig` except for the required
    ``diffusion`` field, and it refuses the pipeline-parallel path rather than appearing to
    support it.
    """

    diffusion: Optional[MaskedDiffusionConfig] = None

    def build(  # type: ignore[override]
        self,
        model,
        device: Optional[torch.device] = None,
    ) -> DiffusionTransformerTrainModule:
        """
        Build the :class:`DiffusionTransformerTrainModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.

        :raises OLMoConfigurationError: If ``diffusion`` is unset, or pipeline parallelism is
            configured.
        """
        if self.diffusion is None:
            raise OLMoConfigurationError(
                "'diffusion' is required by DiffusionTransformerTrainModuleConfig"
            )
        if self.pp_config is not None:
            # `TransformerPipelineTrainModule` is a separate class with its own `train_batch`, so
            # it would silently train autoregressively here.
            raise OLMoConfigurationError(
                "pipeline parallelism is not supported for diffusion training"
            )

        # This mirrors `TransformerTrainModuleConfig.build`'s marshalling rather than calling it,
        # because that method chooses the class to instantiate and there is no seam to pass a
        # different one. The pieces worth not losing to a copy are called, not inlined:
        # `validate_precision_support` is the last cheap place to find a card with no bfloat16,
        # before the model is placed, parallelised or stepped.
        validate_precision_support(self, model)

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        diffusion = cast(MaskedDiffusionConfig, kwargs.pop("diffusion"))
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        return DiffusionTransformerTrainModule(
            model=model,
            device=device,
            diffusion=diffusion,
            **kwargs,
        )
