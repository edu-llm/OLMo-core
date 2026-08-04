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

The configured control is ``global_batch_size_tokens``: constant, identical across
arms, and constant across steps. Each rank divides by its nominal share of that
global batch because FSDP averages gradients across the data-parallel group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from olmo_core.distributed.utils import get_world_size
from olmo_core.train import MetricMergeStrategy, ReduceType
from olmo_core.train.train_module import TransformerTrainModule, TransformerTrainModuleConfig

log = logging.getLogger(__name__)


def _data_parallel_world_size(module: TransformerTrainModule) -> int:
    """The number of rank losses that FSDP averages.

    A process group may be initialized for unrelated reasons while the model has
    no data-parallel wrapper, so the global process count is not a safe fallback.
    """
    if module.world_mesh is None:
        return 1
    return get_world_size(module.dp_process_group)


class FixedDivisorTransformerTrainModule(TransformerTrainModule):
    """``TransformerTrainModule`` with a constant loss denominator.

    :param fixed_loss_div_factor: the global nominal token count. It must be
        identical in both arms; this module divides it by the data-parallel world
        size to get the rank-local denominator used before FSDP gradient averaging.
    """

    def __init__(self, *args: Any, fixed_loss_div_factor: float, **kwargs: Any) -> None:
        if fixed_loss_div_factor <= 0:
            raise ValueError("fixed_loss_div_factor must be positive")
        super().__init__(*args, **kwargs)
        self.global_fixed_loss_div_factor = float(fixed_loss_div_factor)
        dp_world_size = _data_parallel_world_size(self)
        if dp_world_size < 1:  # pragma: no cover - distributed API invariant
            raise RuntimeError(f"invalid data-parallel world size: {dp_world_size}")
        self.fixed_loss_div_factor = self.global_fixed_loss_div_factor / dp_world_size
        log.info(
            "global loss divisor %.1f / %d DP ranks = %.1f rank-local tokens "
            "(OLMo-core's live-token default differs between dense and split)",
            self.global_fixed_loss_div_factor,
            dp_world_size,
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


class DerivedMaskTrainModule(FixedDivisorTransformerTrainModule):
    """Fixed divisor, plus a fact-block mask recomputed from the token stream.

    The masks are not shipped with the corpus. `weights-sidecar/v1` is not a
    registered profile so they have nowhere to live, and a derived mask is the better
    artifact anyway: it cannot fall out of alignment with the tokens it describes,
    which is the defect `mask_alignment_test.py` exists to catch.

    The boundary is the separator `\\n---\\nGOAL ` that every example carries. With
    documents PACKED several to a sequence there is one fact block per document, not
    one per sequence, so a single "find the first separator" does not work. Instead:

        a document opens at position 0 and after every EOS
        a fact block closes one position after a separator run ends
        a position is supervised when as many blocks have closed as documents opened

    A document whose separator never appears — only possible if it was truncated —
    stays masked for its whole length. That is the safe direction: the split arm can
    lose supervision on a proof, but it can never gain supervision on a fact.

    :param arm: ``"dense"`` supervises everything; ``"split"`` masks fact blocks.
    :param separator_ids: token ids of the separator under the corpus's tokenizer.
    :param eos_token_id: document terminator, used to find packed document starts.
    :param pad_token_id: never supervised, in either arm.
    """

    def __init__(
        self,
        *args: Any,
        arm: str,
        separator_ids: list[int],
        eos_token_id: int,
        pad_token_id: int,
        **kwargs: Any,
    ) -> None:
        if arm not in ("dense", "split"):
            raise ValueError(f"arm must be 'dense' or 'split', got {arm!r}")
        if not separator_ids:
            raise ValueError("separator_ids is empty; the mask cannot be derived")
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        # TrainModule is a Stateful protocol, not torch.nn.Module, so it has no
        # register_buffer() lifecycle. Inputs from the numpy loader (including the
        # trainer's mock batch) are still on CPU here; Transformer._prepare_inputs
        # moves them only after this module derives the mask. Keep the tiny immutable
        # constant on CPU and copy it to the actual input device at comparison time.
        self._sep = torch.tensor(separator_ids, dtype=torch.long)
        log.info(
            "arm=%s; fact block derived from a %d-token separator, packing-aware",
            arm,
            len(separator_ids),
        )

    def supervised_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """True where the split arm should score. Handles packed sequences.

        Compares the position of the most recent block-close against the most recent
        document-open: supervision is on exactly when a separator has been seen more
        recently than a document boundary.

        Counting opens against closes instead — which is the obvious implementation —
        is wrong, and wrong in a way that hides. One document missing its separator
        leaves the counters permanently off by one, so every LATER document in the
        same packed sequence stays masked too. Comparing positions is per-document by
        construction, so a malformed document costs only itself.
        """
        b, t = input_ids.shape
        k = int(self._sep.numel())
        idx = torch.arange(t, device=input_ids.device).expand(b, t)

        # a separator run ended at i-1, so supervision may begin at i
        closes = torch.zeros((b, t), dtype=torch.bool, device=input_ids.device)
        if t > k:
            separator = self._sep.to(device=input_ids.device)
            starts = (input_ids.unfold(1, k, 1) == separator).all(dim=-1)
            closes[:, k:] = starts[:, : t - k]

        # a document begins at i (sequence start, or just after an EOS)
        opens = torch.zeros((b, t), dtype=torch.bool, device=input_ids.device)
        opens[:, 0] = True
        opens[:, 1:] = input_ids[:, :-1] == self.eos_token_id

        last_close = torch.cummax(torch.where(closes, idx, -torch.ones_like(idx)), dim=1)[0]
        last_open = torch.cummax(torch.where(opens, idx, -torch.ones_like(idx)), dim=1)[0]
        return last_close > last_open

    def padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """True only on padding, preserving genuine EOS predictions.

        Qwen2.5 uses one id for EOS and padding. Every real document ends in EOS,
        while fixed-width padding is the repeated-EOS tail after the final document.
        Masking `input_ids == pad_token_id` would therefore remove every real EOS
        target and teach neither arm to stop.
        """
        is_pad = input_ids == self.pad_token_id
        if self.pad_token_id != self.eos_token_id:
            return is_pad
        previous_is_eos = torch.zeros_like(is_pad)
        previous_is_eos[:, 1:] = input_ids[:, :-1] == self.eos_token_id
        return is_pad & previous_is_eos

    def label_supervision_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """True at label positions whose *target token* should be scored.

        OLMo's ``get_labels`` shifts IDs left before ``model_forward``:
        ``labels[i] == input_ids[i + 1]``. The fact/padding masks are naturally
        expressed over token positions, so they must be shifted left the same way.
        Applying an unshifted mask skips the first goal token and scores the first
        padding token instead—a plausible one-token boundary error on every proof.
        """
        token_live = ~self.padding_mask(input_ids)
        if self.arm == "split":
            token_live &= self.supervised_mask(input_ids)
        label_live = torch.zeros_like(token_live)
        label_live[:, :-1] = token_live[:, 1:]
        # Intra-document attention defines each segment through its EOS token. The
        # label at that EOS input position would predict the next packed document's
        # first token from a state that cannot attend to that document. Keep the
        # preceding label (the document's genuine EOS target), but never train the
        # EOS state across a packed boundary.
        label_live &= input_ids != self.eos_token_id
        return label_live

    def _record_divisor_diagnostics(self, live_tokens: Any) -> None:
        """Suppress the parent metric, whose count precedes the runtime mask.

        ``TransformerTrainModule.train_batch`` counts labels before model_forward,
        so its `live_tokens` includes fact tokens for both arms. model_forward records
        the correct post-mask fraction below instead.
        """
        del live_tokens

    def model_forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            labels = labels.clone()
            label_live = self.label_supervision_mask(input_ids)
            labels[~label_live] = -100
            if kwargs.get("loss_div_factor") is not None:
                self.record_metric(
                    "train/supervised token fraction",
                    label_live.detach().sum() / self.fixed_loss_div_factor,
                    ReduceType.mean,
                    merge_strategy=MetricMergeStrategy.sum,
                )
        return super().model_forward(input_ids, labels=labels, **kwargs)


@dataclass
class DerivedMaskTrainModuleConfig(TransformerTrainModuleConfig):
    """Config that builds :class:`DerivedMaskTrainModule` instead of the base module.

    ``TransformerTrainModuleConfig.build`` names ``TransformerTrainModule`` directly,
    so a subclass of the module needs a subclass of the config. Overriding ``build``
    here keeps the change in this experiment's own file — nothing under
    ``src/olmo_core/`` is touched, which is what the platform guide asks for.

    The extra fields ride through ``as_dict()`` into the module's constructor.
    """

    arm: str = "dense"
    separator_ids: Optional[list] = None
    eos_token_id: int = 0
    pad_token_id: int = 0
    fixed_loss_div_factor: float = 1.0

    def build(self, model, device=None):  # type: ignore[override]
        if self.pp_config is not None:
            raise ValueError(
                "pipeline parallelism is not supported here: the derived mask assumes "
                "whole sequences on one rank"
            )
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        for k in ("autocast_precision", "state_dict_save_opts", "state_dict_load_opts"):
            kwargs.pop(k, None)
        return DerivedMaskTrainModule(model=model, device=device, **kwargs)
