"""Fixed loss divisor — the control that makes the dense/split comparison mean something.

OLMo-core divides the summed cross-entropy by the number of *live* (unmasked) tokens
in the batch (``train/train_module/transformer/train_module.py:356-358``, passed as
``loss_div_factor`` at line 408). Under that default the two arms divide by different
numbers, because the split arm's denominator excludes the fact block. Every proof
token in the split arm would then receive roughly 1/(1 - mask_fraction) ≈ 1.3-1.5x the
gradient weight it receives in the dense arm — the arms would differ in effective
learning rate on the shared tokens as well as in the mask, and the experiment would
not isolate the mask.

``loss_div_factor`` is a local in ``train_batch``, not a config field, but it reaches
the model through ``model_forward()`` — which is small and overridable. So the fix is
ten lines rather than a fork.

The divisor used is ``global_batch_size_tokens``: constant, identical across arms, and
constant across steps, so it cannot interact with the LR schedule either.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from olmo_core.train import ReduceType
from olmo_core.train.train_module import TransformerTrainModule

log = logging.getLogger(__name__)


class FixedDivisorTransformerTrainModule(TransformerTrainModule):
    """``TransformerTrainModule`` with a constant loss denominator.

    :param fixed_loss_div_factor: the constant to divide summed CE by. Pass
        ``global_batch_size_tokens`` (= global_batch_size_sequences x sequence_length).
        Must be identical in both arms; ``train_sft.py`` derives it from the config
        both arms share, so it cannot drift.
    """

    def __init__(self, *args: Any, fixed_loss_div_factor: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if fixed_loss_div_factor <= 0:
            raise ValueError("fixed_loss_div_factor must be positive")
        self.fixed_loss_div_factor = float(fixed_loss_div_factor)
        log.info(
            "loss divisor pinned to %.1f (OLMo-core default would be the per-batch "
            "live-token count, which differs between the dense and split arms)",
            self.fixed_loss_div_factor,
        )

    def model_forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # Only override when the caller actually asked for a divisor. train_batch does
        # (train_module.py:408); eval_batch does not, and we leave that alone so
        # validation loss stays a readable per-token mean.
        if kwargs.get("loss_div_factor") is not None:
            live_tokens = kwargs["loss_div_factor"]
            kwargs["loss_div_factor"] = self.fixed_loss_div_factor
            self._record_divisor_diagnostics(live_tokens)
        return super().model_forward(input_ids, labels=labels, **kwargs)

    def _record_divisor_diagnostics(self, live_tokens: Any) -> None:
        """Log what the default *would* have divided by.

        Two arms with the same fixed divisor produce loss curves on different scales
        (dense sums over more tokens), so the raw number is not comparable by eye.
        This ratio is: it is the supervised fraction of the batch, and it should be
        ~1.0 for dense and ~0.7-0.85 for split. If they are equal, the mask is not
        being applied and the run is invalid — check this metric first, before
        reading anything into the loss curves.

        Kept as a tensor so recording it costs no device sync.
        """
        if isinstance(live_tokens, torch.Tensor):
            value: Any = live_tokens.detach() / self.fixed_loss_div_factor
        else:
            try:
                value = float(live_tokens) / self.fixed_loss_div_factor
            except (TypeError, ValueError):  # pragma: no cover
                return
        self.record_metric("train/supervised token fraction", value, ReduceType.mean)
