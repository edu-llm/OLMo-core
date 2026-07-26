"""
State-tracking harness for the A_5 word problem (Phase 0 of the NC^1 plan).

Trains a **pure** Mamba-3 stack to predict the running prefix product of a sequence of A_5
generators, then evaluates it at longer lengths than it saw in training. This is the experiment
that distinguishes an abelian transition monoid from a non-solvable one: at
``rotation_block_size=2`` the cumulative rotation is a cumulative *sum* of angles, which is in
TC^0 and provably cannot track a non-solvable group at length, however well it fits the
training distribution.

Two design choices are load-bearing and easy to get wrong:

- **Pure Mamba-3, no attention.** The default hybrid inserts attention every fourth block, and
  attention can memorize short sequences outright -- it would solve the training lengths with no
  state tracking at all and quietly invalidate the comparison.
- **Score the final position only.** Early positions are trivial (position 0 has only as many
  possible labels as there are generators), so averaging over positions inflates the number with
  prefixes that need no real tracking.

The expected result, per the design note: ``b=2`` stays near chance at length 256 while
``b >= 3`` clears 90%. Before believing either number, check the negative control -- an
untrained model must score at ~1/60.

Usage::

    python a5_harness.py --rotation-block-size 3 --steps 2000
    python a5_harness.py --rotation-block-size 2 --steps 2000   # negative control
"""

import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F

from olmo_core.nn.mamba3 import Mamba3Config, admissible_block_sizes

from a5_group import ELEMENTS, make_word_problem  # isort: skip

__all__ = ["build_a5_model", "train_a5", "evaluate_a5", "N_CLASSES"]

#: |A_5|. Also the model's vocabulary size: inputs only ever use the first ``len(GENERATORS)``
#: token ids, but sharing one table keeps the LM head's output width equal to the class count.
N_CLASSES = len(ELEMENTS)


def build_a5_model(
    *,
    rotation_block_size: int,
    n_layers: int = 2,
    d_model: int = 128,
    n_heads: int = 4,
    d_state: int = 48,
    mimo_rank: int = 1,
    n_groups: Optional[int] = None,
    a_log_init_max: float = 0.1,
    init_device: str = "cpu",
    seed: int = 0,
):
    """
    Build a pure Mamba-3 stack sized for the A_5 task.

    :param rotation_block_size: The axis under test. ``2`` is the abelian control.
    :param d_state: Must admit ``rotation_block_size``. The default 48 covers the whole sweep --
        ``admissible_block_sizes(48) == (2, 3, 4, 6, 8)`` -- unlike ``DEFAULT_D_STATE``, which
        omits 3 and so cannot express the smallest non-solvable block. Both values answer the
        same question, so the check below asks
        :func:`~olmo_core.nn.mamba3.admissible_block_sizes` rather than restating the rule;
        ``constraint_test.py`` pins the relationship between the two defaults.
    :param mimo_rank: Left at 1. MIMO widens the read/write rank but leaves the transition
        monoid untouched, so it buys no state-tracking power while making the rotation more
        expensive to apply.
    :param n_groups: ``(B, C)`` groups; defaults to ``n_heads`` so each head gets its own
        rotation schedule. The library default of 1 shares a single schedule across all heads,
        which limits state tracking for reasons unrelated to the block size.
    :param a_log_init_max: Kept small so the decay horizon covers the evaluation lengths. At the
        library default of 16 the state has decayed to ~1e-9 by position 256 and the gradient
        that would teach the model to hold on is ~1e-9 with it.
    :param seed: Seed for weight initialization.
    """
    admissible = admissible_block_sizes(d_state)
    if rotation_block_size not in admissible:
        # Fail here rather than deep inside `build()`, and say which `b` this `d_state` *can*
        # take -- the sweep is the whole point of the harness, so an unusable pair is a setup
        # error worth naming precisely.
        raise ValueError(
            f"d_state ({d_state}) cannot express rotation_block_size ({rotation_block_size}); "
            f"it admits {admissible}"
        )

    config = Mamba3Config.mamba3_hybrid_like(
        d_model=d_model,
        vocab_size=N_CLASSES,
        n_layers=n_layers,
        n_heads=n_heads,
        intermediate_size=4 * d_model,
        mamba_n_heads=n_heads,
        d_state=d_state,
        n_groups=n_heads if n_groups is None else n_groups,
        mimo_rank=mimo_rank,
        rotation_block_size=rotation_block_size,
        a_log_init_max=a_log_init_max,
        # Pure Mamba-3: no attention blocks anywhere in the stack.
        block_pattern=["mamba3"],
    )
    model = config.build(init_device=init_device)
    # `Transformer.init_weights` seeds from the global RNG and returns its generator rather
    # than accepting one, so reproducibility has to be arranged from the outside.
    torch.manual_seed(seed)
    model.init_weights(device=torch.device(init_device))
    return model


def _logits_and_labels(model, seq_len: int, batch_size: int, seed: int, device: torch.device):
    inputs, labels = make_word_problem(batch_size=batch_size, seq_len=seq_len, seed=seed)
    logits = model(inputs.to(device))
    return logits, labels.to(device)


def train_a5(
    model,
    *,
    train_len: int = 40,
    train_lengths: Optional[List[int]] = None,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    device: Optional[torch.device] = None,
    log_every: int = 0,
) -> List[float]:
    """
    Train on words, returning the per-step loss.

    Each step draws a fresh batch, so there is no fixed training set to overfit -- the model
    either learns the group operation or it does not.

    :param train_len: Fixed training word length, used when ``train_lengths`` is ``None``.
    :param train_lengths: A **length curriculum** -- if given, each step samples its length
        uniformly from this list instead of using ``train_len``. This is what lets the model
        generalize to lengths it was not trained at: a fixed short ``train_len`` teaches a
        solution tuned to that one length's accumulated-state distribution, which does not carry
        to much longer sequences even though the recurrence itself is length-agnostic. Exposing a
        range of lengths forces a length-robust solution. Non-solvable tracking (``b >= 3``)
        needs this to reach long evaluation lengths; ``b == 2`` cannot track ``A_5`` at length
        regardless, so the curriculum does not rescue it (it is not a capacity lever).

    :returns: One cross-entropy value per step.
    """
    device = device or torch.device("cpu")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    rng = torch.Generator().manual_seed(seed)
    losses: List[float] = []
    for step in range(steps):
        if train_lengths:
            length = train_lengths[int(torch.randint(len(train_lengths), (1,), generator=rng))]
        else:
            length = train_len
        logits, labels = _logits_and_labels(
            model, length, batch_size, seed=seed * 1_000_003 + step, device=device
        )
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:>5}/{steps}  loss {loss.item():.4f}", flush=True)
    return losses


@torch.no_grad()
def evaluate_a5(
    model,
    *,
    seq_len: int,
    batch_size: int = 256,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> float:
    """
    Accuracy at the **final** position of each sequence.

    The final position is the only one that requires tracking the whole prefix, so it is the
    only honest measure of length extrapolation. Chance is ``1 / 60``.
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    logits, labels = _logits_and_labels(model, seq_len, batch_size, seed=seed, device=device)
    predicted = logits[:, -1, :N_CLASSES].argmax(dim=-1)
    return (predicted == labels[:, -1]).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--rotation-block-size", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=48)
    parser.add_argument("--a-log-init-max", type=float, default=0.1)
    parser.add_argument("--train-len", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=[40, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_a5_model(
        rotation_block_size=args.rotation_block_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_state=args.d_state,
        a_log_init_max=args.a_log_init_max,
        seed=args.seed,
    )

    # The negative control's control. If this is not near 1/60 the harness is leaking and
    # nothing measured afterwards means anything.
    baseline = evaluate_a5(model, seq_len=max(args.eval_lengths), batch_size=256, device=device)
    print(
        f"untrained accuracy @ {max(args.eval_lengths)}: {baseline:.1%} (chance = {1 / N_CLASSES:.1%})"
    )
    if baseline > 0.15:
        print("WARNING: untrained model is well above chance -- the harness is leaking.")

    print(
        f"training b={args.rotation_block_size} for {args.steps} steps at length {args.train_len}"
    )
    losses = train_a5(
        model,
        train_len=args.train_len,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        log_every=max(1, args.steps // 20),
    )
    print(f"final training loss: {losses[-1]:.4f}")

    print("accuracy at the final position:")
    for length in args.eval_lengths:
        accuracy = evaluate_a5(model, seq_len=length, batch_size=256, device=device)
        marker = " (train length)" if length == args.train_len else ""
        print(f"  length {length:>4}: {accuracy:6.1%}{marker}")


if __name__ == "__main__":
    main()
