"""
Continuous-thought forward path for latent chain-of-thought (PRD Phase 3).

The model "thinks in latent space": instead of decoding a token, its last-layer
hidden state at the final position is fed back in as the next input *embedding*
for ``K`` steps (Coconut-style). This needs two things of the model: to accept
embeddings in place of token ids, and to hand back the residual stream.

The first is ``input_embeddings=...``, already a parameter of
:meth:`olmo_core.nn.transformer.Transformer.forward`, which bypasses the embedding lookup
so a mix of real-token embeddings and continuous thoughts can be fed through.

The second is read out with a **forward hook on the last block** rather than by adding a
parameter to ``Transformer.forward`` — see :func:`_capture_last_block`. That function is the
whole of this module's coupling to the core model, and it changes no shared code, which
matters because ``Transformer.forward`` is the most contended function in the repository.

Gradients flow through the whole chain (each thought depends on the previous
forward), so a downstream loss trains the thought-generating computation.

**Thought scale.** The last block's output is the *pre*-final-norm residual
stream, because this model keeps the final norm inside the LM head
(:class:`olmo_core.nn.lm_head.LMHead`). Fed back unnormalized, a thought's magnitude
grows with both depth and ``K`` — measured on the ``olmo2_370M``/``olmo3_370M`` rung at
``K=10``, RMS 5.8 -> 52 against a real-token embedding RMS of 1.0, and training amplifies
it further. Two problems: the SwiGLU feed-forward sees the *unnormalized* residual under
``reordered_norm`` blocks, so the pretrained weights get pushed off their operating point
exactly at the thought positions; and the A3/A4 regularizers (which pull toward the
embedding manifold / penalize thought norm) incidentally suppress that drift while the
unregularized A2 does not — an arm-dependent artifact in a controlled comparison. So
:func:`final_norm` is applied before each feedback step, matching the ``hidden_states[-1]``
(post-final-norm) convention Coconut/CODI feed, and pinning thought scale to the
embedding scale for every arm alike.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

__all__ = ["embed_tokens", "final_norm", "run_continuous_thoughts", "student_forward"]


def embed_tokens(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Embed token ids exactly as :meth:`Transformer.forward` does its own lookup
    (applying ``embed_scale`` and ``embedding_norm`` if the model uses them), so
    real-token embeddings and continuous thoughts live in the same space.

    :param model: A built :class:`~olmo_core.nn.transformer.Transformer`.
    :param input_ids: Token ids of shape ``(batch, seq)``.
    :returns: Embeddings of shape ``(batch, seq, d_model)``.
    """
    h = model.embeddings(input_ids.to(model.device))
    embed_scale = getattr(model, "embed_scale", None)
    if embed_scale is not None:
        h = h * embed_scale
    embedding_norm = getattr(model, "embedding_norm", None)
    if embedding_norm is not None:
        h = embedding_norm(h)
    return h


def final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
    """
    Apply the model's final norm (the one living inside the LM head) to a hidden state.

    The last block's output is the residual stream *before* this norm, so a thought must pass
    through it to land in the same numeric range as the token embeddings it is spliced next
    to — see the module docstring.

    Falls back to returning ``hidden`` unchanged when the model has no final norm to apply
    (no LM head under pipeline parallelism, or a head like
    :class:`~olmo_core.nn.lm_head.NormalizedLMHead` that is built with ``layer_norm=None``).

    :param model: A built :class:`~olmo_core.nn.transformer.Transformer`.
    :param hidden: Hidden states of shape ``(..., d_model)``.
    :returns: The normalized hidden states, same shape.
    """
    lm_head = getattr(model, "lm_head", None)
    norm = getattr(lm_head, "norm", None) if lm_head is not None else None
    return hidden if norm is None else norm(hidden)


@contextmanager
def _capture_last_block(model) -> Iterator[Dict[str, Any]]:
    """
    Capture the output of the model's last block for the duration of the block.

    This is how a thought is read out of the residual stream, and it is deliberately a hook
    rather than a ``return_hidden_states=True`` parameter on
    :meth:`~olmo_core.nn.transformer.Transformer.forward`. The value is the same either way —
    ``forward`` assigns ``h = block(h, ...)`` in a loop and the LM head is the very next
    statement, so the last block's output *is* the post-block hidden state — but a hook adds no
    line to a function that a dozen other workstreams are editing concurrently.

    Blocks return a plain tensor
    (:meth:`~olmo_core.nn.transformer.block.TransformerBlock.forward`), so no tuple unwrapping
    is needed. The hook is removed on the way out even if the forward raises, and the captured
    tensor stays attached to the autograd graph — gradients flow through the thought chain
    exactly as they would from a returned value.

    :param model: A built :class:`~olmo_core.nn.transformer.Transformer`.
    :returns: A dict that holds the hidden states under ``"h"`` once a forward has run.
    """
    captured: Dict[str, Any] = {}

    def hook(_module, _args, output: torch.Tensor) -> None:
        captured["h"] = output

    # `blocks` is a ModuleDict keyed by stringified index, and under pipeline parallelism a
    # stage holds only its own slice, so take the largest key rather than assuming `n_layers-1`.
    last_key = max(model.blocks.keys(), key=int)
    handle = model.blocks[last_key].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _forward_hidden(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Run the model on pre-computed embeddings and return post-block hidden states."""
    batch, seq = inputs_embeds.shape[:2]
    # input_ids only supplies the shape (positions/RoPE); values are ignored because
    # input_embeddings overrides the embedding lookup.
    dummy_ids = torch.zeros((batch, seq), dtype=torch.long, device=model.device)
    with _capture_last_block(model) as captured:
        # `logits_to_keep=1` makes the LM head we do not need as cheap as it can be: it slices
        # the hidden states to one position *before* the vocab projection, so the discarded work
        # is a single (d_model x vocab) matmul per thought step instead of one per position.
        model(dummy_ids, input_embeddings=inputs_embeds, logits_to_keep=1)
    if "h" not in captured:
        raise RuntimeError(
            "the forward hook on the last block never fired, so no hidden state was captured; "
            "the model ran no blocks (an empty pipeline stage?)"
        )
    return captured["h"]


def run_continuous_thoughts(
    model, prefix_embeds: torch.Tensor, num_thoughts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ``K`` continuous thoughts from a prefix of embeddings.

    At each step the model is run on the running embedding sequence; the last-layer
    hidden state at the final position is passed through :func:`final_norm` and becomes
    the next continuous thought, appended to the sequence. The norm keeps a thought at
    the same scale as a token embedding no matter how large ``num_thoughts`` is — without
    it the magnitude compounds every step (see the module docstring).

    :param model: A built transformer.
    :param prefix_embeds: The prefix embeddings ``(batch, prefix_len, d_model)``
        (e.g. ``question <bot>``).
    :param num_thoughts: ``K`` — how many continuous thoughts to generate.
    :returns: ``(thoughts, embeds)`` where ``thoughts`` is ``(batch, K, d_model)`` and
        ``embeds`` is the prefix followed by the thoughts, ``(batch, prefix_len + K, d_model)``.
    """
    if num_thoughts < 1:
        raise ValueError(f"num_thoughts must be >= 1, got {num_thoughts}")
    embeds = prefix_embeds
    thoughts = []
    for _ in range(num_thoughts):
        hidden = _forward_hidden(model, embeds)
        # (batch, 1, d_model) — last position, through the final norm so the thought
        # lands in the same numeric range as a token embedding.
        thought = final_norm(model, hidden[:, -1:, :])
        thoughts.append(thought)
        embeds = torch.cat([embeds, thought], dim=1)
    return torch.cat(thoughts, dim=1), embeds


def student_forward(
    model,
    prefix_ids: torch.Tensor,
    num_thoughts: int,
    suffix_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    z_loss_multiplier: Optional[float] = None,
):
    """
    Full student forward: embed the prefix, generate ``K`` continuous thoughts, splice
    them in, append the suffix embeddings, and run one final forward.

    Layout of the assembled sequence: ``prefix (question <bot>) | K thoughts | suffix
    (<eot> <distill> answer)``.

    :param model: A built transformer.
    :param prefix_ids: Prefix token ids ``(batch, prefix_len)`` — ``question <bot>``.
    :param num_thoughts: ``K``.
    :param suffix_ids: Suffix token ids ``(batch, suffix_len)`` — ``<eot> <distill> answer``.
    :param labels: Optional next-token-aligned labels for the *full* assembled sequence
        (``-100`` where unsupervised); if given, the model returns an ``LMOutputWithLoss``.
    :param z_loss_multiplier: Optional z-loss multiplier passed through to the LM head.
    :returns: ``(output, thoughts)`` — ``output`` is logits (labels ``None``) or an
        ``LMOutputWithLoss`` (labels given); ``thoughts`` is ``(batch, K, d_model)``.
    """
    prefix_embeds = embed_tokens(model, prefix_ids)
    thoughts, embeds = run_continuous_thoughts(model, prefix_embeds, num_thoughts)
    suffix_embeds = embed_tokens(model, suffix_ids)
    full = torch.cat([embeds, suffix_embeds], dim=1)

    batch, seq = full.shape[:2]
    dummy_ids = torch.zeros((batch, seq), dtype=torch.long, device=model.device)
    output = model(
        dummy_ids,
        input_embeddings=full,
        labels=labels,
        z_loss_multiplier=z_loss_multiplier,
    )
    return output, thoughts
