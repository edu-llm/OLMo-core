"""
Probing utilities for latent reasoning (PRD Phase 6.3 / experiment L8).

These read what the continuous thoughts encode:

- **logit-lens** — decode a hidden state through the LM head to a vocabulary distribution.
- **decodability** — how vocab-like a thought is (top-1 logit-lens mass); the quantity the
  vocab-manifold regularizer (gate B) should increase.
- **superposition mass** — how much logit-lens probability a thought places on a set of
  frontier-node tokens at once (the superposition signature).
- **linear probe** — whether a target property is *linearly present* in the activations,
  with a shuffled-label control (correlational).
- **causal ablation** — remove a direction from the thoughts and measure the change in the
  answer margin (causal, per the 2026 "observable patterns aren't explanations" caveat).
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "logit_lens",
    "decodability",
    "superposition_mass",
    "linear_probe_accuracy",
    "causal_ablation_margin_change",
]


def logit_lens(model, hidden: torch.Tensor) -> torch.Tensor:
    """Decode hidden states ``(..., d_model)`` to a vocabulary distribution via the LM head."""
    logits = model.lm_head(hidden)  # labels=None -> returns logits
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def decodability(model, thoughts: torch.Tensor) -> float:
    """Mean top-1 logit-lens probability of the thoughts (higher = closer to the vocab manifold)."""
    return float(logit_lens(model, thoughts).max(dim=-1).values.mean())


@torch.no_grad()
def superposition_mass(
    model, thoughts: torch.Tensor, frontier_token_ids: Sequence[int]
) -> torch.Tensor:
    """
    Per-thought total logit-lens probability on a set of frontier-node token ids.

    A superposition state should place mass on *several* frontier candidates at once.

    :returns: A tensor of shape ``thoughts.shape[:-1]`` (mass in ``[0, 1]``).
    """
    probs = logit_lens(model, thoughts)
    ids = torch.as_tensor(list(frontier_token_ids), device=probs.device, dtype=torch.long)
    return probs.index_select(-1, ids).sum(dim=-1)


def linear_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: Optional[int] = None,
    shuffle_labels: bool = False,
    seed: int = 0,
    steps: int = 300,
    lr: float = 0.1,
    test_frac: float = 0.3,
) -> float:
    """
    Train a linear (logistic-regression) probe and return held-out accuracy.

    Pass ``shuffle_labels=True`` for the control: if the real probe does not beat the
    shuffled-label probe, the property is not linearly decodable from the features.
    """
    generator = torch.Generator().manual_seed(seed)
    features = features.detach().float()
    labels = labels.detach().long()
    n, d = features.shape
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    if shuffle_labels:
        labels = labels[torch.randperm(n, generator=generator)]

    perm = torch.randperm(n, generator=generator)
    n_test = max(1, int(n * test_frac))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    weight = torch.zeros(d, num_classes, requires_grad=True)
    bias = torch.zeros(num_classes, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = features[train_idx] @ weight + bias
        F.cross_entropy(logits, labels[train_idx]).backward()
        opt.step()

    with torch.no_grad():
        pred = (features[test_idx] @ weight + bias).argmax(dim=-1)
        return float((pred == labels[test_idx]).float().mean())


@torch.no_grad()
def causal_ablation_margin_change(
    answer_margin_fn, thoughts: torch.Tensor, direction: torch.Tensor
) -> float:
    """
    Measure the causal effect of a direction in the continuous thoughts on the answer.

    ``answer_margin_fn(thoughts) -> float`` recomputes the answer margin (e.g. yes-logit
    minus no-logit) from a (possibly modified) thoughts tensor. This projects ``direction``
    out of every thought and returns ``|margin_before - margin_after|``.

    :returns: The absolute change in the answer margin (0 = the direction is not used).
    """
    unit = direction / (direction.norm() + 1e-8)
    before = answer_margin_fn(thoughts)
    coeff = (thoughts * unit).sum(dim=-1, keepdim=True)  # (..., 1)
    ablated = thoughts - coeff * unit
    after = answer_margin_fn(ablated)
    return abs(before - after)
