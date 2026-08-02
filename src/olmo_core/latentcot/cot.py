"""
Continuous-thought forward path for latent chain-of-thought (PRD Phase 3).

The model "thinks in latent space": instead of decoding a token, its last-layer
hidden state at the final position is fed back in as the next input *embedding*
for ``K`` steps (Coconut-style). This uses two mechanisms already in
:meth:`olmo_core.nn.transformer.Transformer.forward`:

- ``return_hidden_states=True`` returns the post-block hidden states (Phase 3.1);
- ``input_embeddings=...`` bypasses the embedding lookup so a mix of real-token
  embeddings and continuous thoughts can be fed through the model.

Gradients flow through the whole chain (each thought depends on the previous
forward), so a downstream loss trains the thought-generating computation.
"""

from typing import Optional, Tuple

import torch

__all__ = ["embed_tokens", "run_continuous_thoughts", "student_forward"]


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


def _forward_hidden(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Run the model on pre-computed embeddings and return post-block hidden states."""
    batch, seq = inputs_embeds.shape[:2]
    # input_ids only supplies the shape (positions/RoPE); values are ignored because
    # input_embeddings overrides the embedding lookup.
    dummy_ids = torch.zeros((batch, seq), dtype=torch.long, device=model.device)
    return model(dummy_ids, input_embeddings=inputs_embeds, return_hidden_states=True)


def run_continuous_thoughts(
    model, prefix_embeds: torch.Tensor, num_thoughts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ``K`` continuous thoughts from a prefix of embeddings.

    At each step the model is run on the running embedding sequence; the last-layer
    hidden state at the final position becomes the next continuous thought and is
    appended to the sequence.

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
        thought = hidden[:, -1:, :]  # (batch, 1, d_model) — last position
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
