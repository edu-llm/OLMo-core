"""Extract Co-LMLM retrieval query vectors from an OLMo-core transformer.

The Co-LMLM retrieval query is the final-layer hidden state at a special-token position, taken
*after* the final norm and L2-normalized -- i.e. exactly the tensor ``lm_head.norm(h)`` that the
LM head feeds into its output projection (this is what the released model indexes at ``<FACT>``
positions). Rather than reimplement the model's forward, we grab that tensor with a forward hook
on ``lm_head.norm`` so all of OLMo-core's attention / RoPE / parallelism plumbing is reused.
"""

import contextlib
from typing import Dict, Iterator

import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@contextlib.contextmanager
def capture_final_hidden_states(model) -> Iterator[Dict[str, torch.Tensor]]:
    """Context manager that captures the post-final-norm hidden states of a forward pass.

    Usage::

        with capture_final_hidden_states(model) as captured:
            out = model(input_ids, labels=labels)   # normal OLMo-core forward
        hidden = captured["hidden_states"]           # (B, T, d), == lm_head.norm(h)

    The captured tensor keeps its graph, so gradients from the contrastive loss flow back through
    the whole model. Requires the model's LM head to have a ``norm`` (true for the SmolLM2 /
    Co-LMLM configs, which use a final RMSNorm before the tied output projection).
    """
    lm_head = getattr(model, "lm_head", None)
    norm = getattr(lm_head, "norm", None) if lm_head is not None else None
    if norm is None:
        raise ValueError(
            "capture_final_hidden_states requires model.lm_head.norm to exist; the Co-LMLM query "
            "is the post-final-norm hidden state. This model has no final norm on its LM head."
        )

    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden_states"] = output

    handle = norm.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def gather_query_vectors(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Gather (and L2-normalize) query vectors at the given ``(batch_idx, seq_idx)`` positions.

    :param hidden_states: ``(B, T, d)`` hidden states (e.g. from ``capture_final_hidden_states``).
    :param positions: ``(P, 2)`` long tensor; each row is ``(batch_idx, seq_idx)``.
    :param normalize: L2-normalize the gathered vectors (queries are used as normalized keys).
    :returns: ``(P, d)`` query vectors.
    """
    if positions.numel() == 0:
        return hidden_states.new_zeros((0, hidden_states.shape[-1]))
    vecs = hidden_states[positions[:, 0], positions[:, 1]]
    return l2_normalize(vecs) if normalize else vecs
