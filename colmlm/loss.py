"""Co-LMLM training objective: masked next-token prediction + bidirectional InfoNCE.

Implements the joint loss from Co-LMLM (arXiv:2607.07707), Section 3.2:

    L = L_NTP + lambda * L_CL          (lambda = 0.25 by default)

where

* ``L_NTP`` is next-token prediction with the fact-span content masked out. The masked set ``M``
  is the tokens strictly between ``<FACT>`` and ``</FACT>``, *inclusive of the closing* ``</FACT>``
  but *excluding the opening* ``<FACT>`` (the model must still learn to emit ``<FACT>`` to trigger
  retrieval). In OLMo-core terms this is exactly a boolean ``label_mask`` that is ``False`` on
  ``M`` and ``True`` elsewhere (see ``olmo_core.data.utils.get_labels``).

* ``L_CL`` is the symmetric InfoNCE over fact/question query pairs (eqs. 1-2 in the paper). For a
  batch of paired ``(f_i, q_i)`` L2-normalized query vectors,

      l_{f->q}^i = log softmax_i( f_i . q_. / tau )
      l_{q->f}^i = log softmax_i( q_i . f_. / tau )
      L_CL       = -(1 / 2B) * sum_i ( l_{f->q}^i + l_{q->f}^i )

  which equals the mean of the two cross-entropies with the identity as targets.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

#: Contrastive temperature (paper Table 6).
DEFAULT_TEMPERATURE = 0.07

#: Weight on the contrastive term in the joint loss (paper: lambda = 0.25).
DEFAULT_CONTRASTIVE_WEIGHT = 0.25

#: Value used to mark positions that should not contribute to the NTP loss.
IGNORE_INDEX = -100


def bidirectional_info_nce(
    fact_queries: torch.Tensor,
    question_queries: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Symmetric InfoNCE between paired fact and question query vectors.

    :param fact_queries: ``(P, d)`` document-side query vectors (``<FACT>`` hidden states).
        Assumed L2-normalized.
    :param question_queries: ``(P, d)`` question-side query vectors (``<FACT-q>`` hidden states),
        row-aligned so that ``question_queries[i]`` is the positive for ``fact_queries[i]``.
        Assumed L2-normalized.
    :returns: A scalar loss. Returns ``0.0`` if there are no pairs.
    """
    if fact_queries.numel() == 0 or fact_queries.shape[0] == 0:
        return fact_queries.new_zeros(())
    if fact_queries.shape != question_queries.shape:
        raise ValueError(
            f"fact/question query shapes must match, got {tuple(fact_queries.shape)} vs "
            f"{tuple(question_queries.shape)}"
        )

    # (P, P) similarity logits; row i over all questions (and its transpose for the other direction).
    logits = (fact_queries @ question_queries.t()) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_f2q = F.cross_entropy(logits, targets)
    loss_q2f = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_f2q + loss_q2f)


def masked_ntp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    shift: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Next-token-prediction cross-entropy with ``ignore_index`` masking.

    :param logits: ``(B, T, V)`` logits.
    :param labels: ``(B, T)`` targets, with ``ignore_index`` on masked (fact-content) positions.
    :param shift: If ``True``, shift so ``logits[t]`` predicts ``labels[t+1]`` (use for raw,
        unshifted labels). If ``False`` (default), ``labels`` are assumed already shifted, which
        is OLMo-core's convention (``get_labels`` shifts and pads).
    """
    if shift:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def colmlm_joint_loss(
    ntp_loss: torch.Tensor,
    fact_queries: torch.Tensor,
    question_queries: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    contrastive_weight: float = DEFAULT_CONTRASTIVE_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine a precomputed NTP loss with the contrastive loss: ``L = L_NTP + lambda * L_CL``.

    ``ntp_loss`` is taken as an argument (rather than computed here) so this composes with
    OLMo-core's LM head, which already computes the (masked) next-token loss. Use
    :func:`masked_ntp_loss` to compute it directly when not going through the head.

    :returns: ``(total_loss, metrics)`` where ``metrics`` holds detached ``ntp``, ``contrastive``,
        and ``total`` scalars for logging.
    """
    cl = bidirectional_info_nce(fact_queries, question_queries, temperature=temperature)
    total = ntp_loss + contrastive_weight * cl
    metrics = {
        "ntp": ntp_loss.detach(),
        "contrastive": cl.detach(),
        "total": total.detach(),
    }
    return total, metrics
