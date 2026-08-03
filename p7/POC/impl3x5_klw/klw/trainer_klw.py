"""The weighted trainer: Impl 5's trainer with a per-token multiplier on the pedagogy loss.

Everything is inherited from ``impl4.trainer.sequential_trainer_cls()`` — the
``SequentialSampler`` that keeps the 24/8 block layout a per-step constraint, and through it
Impl 2's LoRA/LR/masking. This module adds ``compute_loss`` and a collator, and nothing else.

## The one hard requirement

``m_t ≡ 1`` must produce **bitwise** the loss the stock trainer produces, because arm
``bT451`` exists to reproduce D4 and it cannot do that if the weighted path normalises
differently from the unweighted one. That is harder than it looks, for a reason
``impl4_ssd/probe_loss_norm.py`` documents at length:

    transformers>=4.48 normalises the loss by ``num_items_in_batch`` = total unmasked label
    tokens across the whole accumulation group [...] But that fix depends on
    ``Trainer.model_accepts_loss_kwargs``, resolved by inspecting the forward signature, and it
    can fall back to per-micro-batch mean when the model is PEFT-wrapped.

So there are two live normalisations and which one applies is a property of the installed
transformers plus the PEFT wrapping — not something to assume. Both are implemented here and
selected by ``loss_denom``:

``global``
    ``Σ_t m_t · ce_t / num_items_in_batch``. Matches ``fixed_cross_entropy`` with
    ``reduction="sum"``; ``training_step`` then does **not** rescale by the accumulation count.
``microbatch``
    ``Σ_t m_t · ce_t / (unmasked tokens in this micro-batch)``. Matches ``reduction="mean"``;
    ``training_step`` **does** rescale.
``auto``
    the ``model_accepts_loss_kwargs`` rule, which is what the stock path itself branches on.

``training_step`` is deliberately *not* overridden, so whichever rescaling the installed
version applies is inherited rather than reimplemented. ``acceptance_checks_klw.py`` check W1
settles the choice empirically — it runs ``m ≡ 1`` through this class and through the stock
class at ``lr=0`` and requires the logged losses to agree — and the run driver then passes the
winner explicitly. ``auto`` is a default, not an answer.

## Alignment

The multiplier for label position ``j`` travels in a ``weights`` tensor of the same shape as
``labels``, padded with 0.0, and is shifted by exactly the same slice the labels are. Masked
positions carry 0.0 rather than 1.0 so that a padding or shift bug removes a contribution
instead of quietly inventing a plausible one. Check W3 verifies this end-to-end.

**``remove_unused_columns`` must be False** or HF strips the ``weights`` column *before* the
collator and every arm silently trains unweighted at full strength. That failure mode produces
no error and a perfectly plausible loss curve, so :func:`weighted_trainer_cls` also counts the
batches it saw weights on and :meth:`assert_weighting_ran` fails the run if the count is zero.
"""

from __future__ import annotations

IGNORE = -100
WEIGHT_COLUMN = "weights"
LOSS_DENOM_CHOICES = ("auto", "global", "microbatch")


def weighted_token_loss(logits, labels, weights):
    """``(Σ_t m_t · ce_t, unmasked token count)`` for one micro-batch. No normalisation.

    Split out of ``compute_loss`` so acceptance check W3 can verify the alignment directly:
    the numerator is a single scalar, so a batch whose weights are all zero except one label
    position must return exactly that position's cross-entropy. Nothing else pins the causal
    shift and the padding together.

    Mirrors transformers' ``ForCausalLMLoss``: logits upcast to fp32, shifted so that position
    ``j-1`` predicts ``labels[j]``, ``ignore_index=-100`` contributing zero.
    """
    import torch.nn.functional as F

    logits = logits.float()
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    shift_w = weights[..., 1:]

    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE, reduction="none")
    total = (ce * shift_w.reshape(-1).to(ce.dtype)).sum()
    n_unmasked = (flat_labels != IGNORE).sum()
    return total, n_unmasked


def make_collator(tokenizer, weighted: bool):
    """Impl 5's collator, plus a 0.0-padded ``weights`` tensor for rows that carry one.

    **Whether to emit weights is decided per batch, from the rows themselves**, not from the
    ``weighted`` flag alone. The Trainer uses one collator for both dataloaders, and the eval
    dataset deliberately has no ``weights`` column — held-out loss stays unweighted so it stays
    comparable to D4's. A collator that unconditionally indexed the column would train fine and
    then die at the first ``eval_steps`` boundary, hundreds of steps in.

    A batch where *some* rows have weights and some do not is a real bug — the two datasets got
    mixed — so that raises rather than picking a default.

    A separate function rather than a closure inside the trainer so the acceptance checks can
    build batches the same way training does.
    """
    import torch

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        pad = tokenizer.pad_token_id
        have = [WEIGHT_COLUMN in x and x[WEIGHT_COLUMN] is not None for x in batch]
        if any(have) and not all(have):
            raise ValueError(
                f"{sum(have)} of {len(batch)} rows in this batch carry {WEIGHT_COLUMN!r} — a "
                f"weighted and an unweighted dataset have been mixed into one batch")
        emit = weighted and all(have)

        ii, ll, aa, ww = [], [], [], []
        for x in batch:
            n = maxlen - len(x["input_ids"])
            ii.append(x["input_ids"] + [pad] * n)
            ll.append(x["labels"] + [IGNORE] * n)
            aa.append(x["attention_mask"] + [0] * n)
            if emit:
                w = x[WEIGHT_COLUMN]
                if len(w) != len(x["input_ids"]):
                    raise ValueError(f"weights length {len(w)} != input_ids length "
                                     f"{len(x['input_ids'])}")
                ww.append(list(w) + [0.0] * n)
        out = {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
               "attention_mask": torch.tensor(aa)}
        if emit:
            out[WEIGHT_COLUMN] = torch.tensor(ww, dtype=torch.float32)
        return out

    return collate


def weighted_trainer_cls(sequential_trainer_cls):
    """``WeightedTrainer`` over the caller's sequential ``Trainer`` subclass.

    Takes the base class as an argument rather than importing it, so this module has no import
    of ``impl4`` and can be unit-tested against a stub.
    """
    import torch

    base_cls = sequential_trainer_cls()

    class WeightedTrainer(base_cls):
        def __init__(self, *a, loss_denom: str = "auto", require_weights: bool = True, **kw):
            super().__init__(*a, **kw)
            if loss_denom not in LOSS_DENOM_CHOICES:
                raise ValueError(f"loss_denom must be one of {LOSS_DENOM_CHOICES}")
            self.loss_denom = loss_denom
            self.require_weights = require_weights
            self.n_weighted_batches = 0
            self.n_unweighted_batches = 0
            self.denom_used: str | None = None

        # -- normalisation ---------------------------------------------------
        def _use_global_denom(self, num_items_in_batch) -> bool:
            if self.loss_denom == "global":
                return num_items_in_batch is not None
            if self.loss_denom == "microbatch":
                return False
            # auto: branch on the same flag the stock path branches on.
            return (bool(getattr(self, "model_accepts_loss_kwargs", False))
                    and num_items_in_batch is not None)

        # -- the loss --------------------------------------------------------
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,
                         **kwargs):
            weights = inputs.pop(WEIGHT_COLUMN, None)
            if weights is None:
                # Eval batches carry no weights by design: held-out loss stays unweighted so
                # it is comparable to D4's. Train batches reaching here mean the column was
                # stripped — assert_weighting_ran() turns that into a failure.
                self.n_unweighted_batches += 1
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch, **kwargs)

            self.n_weighted_batches += 1
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            total, n_unmasked = weighted_token_loss(outputs.logits, labels, weights)

            if self._use_global_denom(num_items_in_batch):
                denom = num_items_in_batch
                if not torch.is_tensor(denom):
                    denom = torch.tensor(denom, device=total.device, dtype=total.dtype)
                denom = denom.to(device=total.device, dtype=total.dtype)
                self.denom_used = "global"
            else:
                denom = n_unmasked.to(total.dtype)
                self.denom_used = "microbatch"
            loss = total / denom.clamp(min=1)

            return (loss, outputs) if return_outputs else loss

        # -- the guard -------------------------------------------------------
        def assert_weighting_ran(self) -> None:
            """Fail loudly if no training batch ever carried weights.

            The silent-unweighted failure this catches (``remove_unused_columns=True``
            stripping the column) leaves no trace in the loss curve, so it has to be checked
            explicitly rather than noticed.
            """
            if self.require_weights and self.n_weighted_batches == 0:
                raise RuntimeError(
                    "no training batch carried per-token weights — the arm trained UNWEIGHTED. "
                    "The usual cause is TrainingArguments(remove_unused_columns=True), which "
                    f"strips the {WEIGHT_COLUMN!r} column before the collator. This run is not "
                    "a weighted run; discard it."
                )

        def weighting_report(self) -> dict:
            return {"weighted_batches": self.n_weighted_batches,
                    "unweighted_batches": self.n_unweighted_batches,
                    "loss_denom_requested": self.loss_denom,
                    "loss_denom_used": self.denom_used,
                    "model_accepts_loss_kwargs": bool(
                        getattr(self, "model_accepts_loss_kwargs", False))}

    return WeightedTrainer
