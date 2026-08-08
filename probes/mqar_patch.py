"""Add MQAR (multi-query associative recall) to the probe task suite.

Run this from the ``probes/`` directory; it edits ``tasks.py`` and ``train_probe.py`` in place
and is idempotent (re-running is a no-op).

WHY THIS TASK
-------------
The existing tasks (parity, S3/S4/S5 words, mod_arith) are all *running-product* problems, where
the optimal policy is to never forget anything. That makes them structurally unable to test KDA's
per-channel forget gate -- both KDA and GDN want their gate pinned at "no decay," and KDA's extra
per-channel flexibility is pure liability there (which is what the parity@512 result showed).

MQAR rewards *selective* retention: hold D key->value associations across a span of distractors,
then answer queries. That is the regime the per-channel gate exists for.

DESIGN: the two knobs are deliberately decoupled
------------------------------------------------
MQAR difficulty has two independent axes, and varying both at once confounds them:
  * number of pairs D   -> how much must be held (a CAPACITY limit on the fixed-size state)
  * retention distance  -> how long it must be held (what a FORGET GATE governs)

To isolate the gate, D is **capped** (``MQAR_MAX_PAIRS``) and extra sequence length is spent on
distractor filler. So going from length 128 to 2048 holds the memory load fixed and stretches the
distance 16x. Below the cap D grows with length, so short training sequences remain well-posed.

LAYOUT (causal, single pass, answers strictly after all pairs)
-------------------------------------------------------------
    k1 v1 k2 v2 ... kD vD  <filler ...>  SEP  q1 q2 ... qD
    ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^   ^^^  ^^^^^^^^^^^^
    2D tokens              distractors        D queries

The target at each query position is the *value* token paired with that key; every other position
is ``-100`` (ignored). Keys, values and distractors occupy disjoint vocabulary ranges, and keys
are sampled **without replacement** so the mapping is unambiguous.

``-100`` is already ``torch.nn.functional.cross_entropy``'s default ``ignore_index``, so the
training loss needs no change. Accuracy *does* -- unmasked it would score the ignored positions as
wrong, which is why ``evaluate()`` is patched below.
"""

import re
from pathlib import Path

TASKS_ADDITION = '''

# --- MQAR (multi-query associative recall) -----------------------------------------------------
# Disjoint vocabulary ranges: 0 is the separator, then keys, then values, then distractors.
MQAR_NUM_KEYS = 32
MQAR_NUM_VALUES = 32
MQAR_NUM_DISTRACTORS = 32
# Cap on pairs per sequence. Holding this fixed while sequence length grows is what turns the
# length sweep into a pure RETENTION-DISTANCE test rather than a capacity test. Must be
# <= MQAR_NUM_KEYS for keys to be sampled without replacement.
MQAR_MAX_PAIRS = 8
MQAR_SEP = 0
MQAR_KEY_BASE = 1
MQAR_VALUE_BASE = MQAR_KEY_BASE + MQAR_NUM_KEYS
MQAR_DISTRACTOR_BASE = MQAR_VALUE_BASE + MQAR_NUM_VALUES
MQAR_VOCAB = MQAR_DISTRACTOR_BASE + MQAR_NUM_DISTRACTORS
MQAR_IGNORE = -100


def mqar_num_pairs(length: int) -> int:
    """Number of key-value pairs to use at a given sequence length.

    Grows with length until :data:`MQAR_MAX_PAIRS`, then stays flat so that additional length
    becomes distractor filler (retention distance) rather than extra memory load (capacity).

    :param length: Sequence length.
    :returns: Number of pairs, at least 1.
    """
    return max(1, min(MQAR_MAX_PAIRS, (length - 1) // 3))


def make_mqar(
    batch: int,
    length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-query associative recall.

    See the module docstring of ``mqar_patch.py`` for the layout and rationale. Keys are drawn
    without replacement within a sequence, so each key has exactly one correct value.

    :param batch: Batch size.
    :param length: Sequence length. Must be at least 4.
    :param generator: RNG for reproducibility.
    :returns: ``(inputs, targets)``, both ``[batch, length]`` int64. ``targets`` is
        :data:`MQAR_IGNORE` everywhere except the query positions.
    """
    if length < 4:
        raise ValueError(f"mqar needs length >= 4, got {length}")
    D = mqar_num_pairs(length)

    inputs = torch.empty(batch, length, dtype=torch.int64)
    targets = torch.full((batch, length), MQAR_IGNORE, dtype=torch.int64)

    # Keys without replacement per row; values may repeat (many keys may share a value).
    key_perm = torch.argsort(torch.rand(batch, MQAR_NUM_KEYS, generator=generator), dim=1)
    keys = key_perm[:, :D] + MQAR_KEY_BASE
    values = (
        torch.randint(0, MQAR_NUM_VALUES, (batch, D), generator=generator) + MQAR_VALUE_BASE
    )

    inputs[:, 0 : 2 * D : 2] = keys
    inputs[:, 1 : 2 * D : 2] = values

    # Filler: distractors only, so it can never be mistaken for a key, value or separator.
    n_query = D
    sep_at = length - n_query - 1
    n_filler = sep_at - 2 * D
    if n_filler > 0:
        inputs[:, 2 * D : sep_at] = (
            torch.randint(
                0, MQAR_NUM_DISTRACTORS, (batch, n_filler), generator=generator
            )
            + MQAR_DISTRACTOR_BASE
        )
    inputs[:, sep_at] = MQAR_SEP

    # Query the D keys in a fresh random order, so answer position carries no information about
    # which key is being asked and the model cannot exploit an ordering shortcut.
    order = torch.argsort(torch.rand(batch, D, generator=generator), dim=1)
    inputs[:, sep_at + 1 :] = torch.gather(keys, 1, order)
    targets[:, sep_at + 1 :] = torch.gather(values, 1, order)
    return inputs, targets
'''

TASKS_ENTRY = '''    "mqar": {
        "fn": make_mqar,
        "in_vocab": MQAR_VOCAB,
        "out_vocab": MQAR_VOCAB,
    },
'''


def patch_tasks(path: Path) -> bool:
    """Insert the MQAR generator and register it in ``TASKS``.

    :returns: ``True`` if the file was modified.
    """
    src = path.read_text()
    if "make_mqar" in src:
        print("tasks.py: already patched")
        return False

    anchor = "\nTASKS = {"
    idx = src.index(anchor)
    src = src[:idx] + TASKS_ADDITION + src[idx:]

    # Register in TASKS by appending after the last existing entry. The mod_arith entry ends with
    # its own '    },' line; insert immediately after that, i.e. still INSIDE the dict. Anchoring
    # on a brace search is fragile -- it is easy to land past the dict's own closing '}'.
    anchor_entry = '        "out_vocab": MOD_ARITH_P,\n    },\n'
    if anchor_entry not in src:
        raise SystemExit("tasks.py: could not find the mod_arith entry to insert after")
    src = src.replace(anchor_entry, anchor_entry + TASKS_ENTRY, 1)
    path.write_text(src)
    print("tasks.py: patched (make_mqar + TASKS entry)")
    return True


def patch_self_check(path: Path) -> bool:
    """Teach ``tasks._self_check`` about ignore-indexed targets.

    Its existing ``targets.min() >= 0`` assertion is *correct* for every running-product task and
    correctly rejects MQAR, whose targets are ``-100`` outside query positions. Rather than weaken
    the assertion, restrict it to the scored positions.

    :returns: ``True`` if the file was modified.
    """
    src = path.read_text()
    if "scored = targets[targets != MQAR_IGNORE]" in src:
        print("tasks.py self-check: already patched")
        return False

    old = """        assert int(targets.min()) >= 0 and int(targets.max()) < spec["out_vocab"], (
            name,
            int(targets.max()),
            spec["out_vocab"],
        )"""
    new = """        # Ignore-indexed positions (-100) are excluded: they are deliberately outside the
        # vocabulary and are dropped by both cross_entropy and the accuracy computation.
        scored = targets[targets != MQAR_IGNORE]
        assert scored.numel() > 0, f"{name}: every position is ignored"
        assert int(scored.min()) >= 0 and int(scored.max()) < spec["out_vocab"], (
            name,
            int(scored.max()),
            spec["out_vocab"],
        )"""
    if old not in src:
        raise SystemExit("tasks.py: self-check assertion not found -- inspect manually")
    src = src.replace(old, new, 1)
    path.write_text(src)
    print("tasks.py self-check: patched (excludes ignored positions)")
    return True


def patch_train_probe(path: Path) -> bool:
    """Mask ignored positions out of the accuracy computation.

    The training loss needs no change: ``-100`` is already ``cross_entropy``'s default
    ``ignore_index``. Accuracy would otherwise count every ignored position as a miss, which for
    MQAR is ~90% of the sequence.

    :returns: ``True`` if the file was modified.
    """
    src = path.read_text()
    if "valid = y != -100" in src:
        print("train_probe.py: already patched")
        return False

    old = '        out[length] = (logits.float().argmax(-1) == y).float().mean().item()'
    new = """        # Tasks may mark positions as ignored with -100 (MQAR scores only query positions).
        # Averaging over all positions would make an unsolved MQAR look ~90% correct.
        correct = logits.float().argmax(-1) == y
        valid = y != -100
        out[length] = (correct & valid).sum().item() / max(1, int(valid.sum()))"""
    if old not in src:
        raise SystemExit("train_probe.py: accuracy line not found -- inspect manually")
    src = src.replace(old, new)
    path.write_text(src)
    print("train_probe.py: patched (masked accuracy)")
    return True


def self_check() -> None:
    """Validate the generator: shapes, disjoint ranges, key uniqueness, recoverable answers."""
    import torch  # noqa: F401  (imported by the patched tasks module)

    import tasks

    gen = torch.Generator().manual_seed(0)
    for length in (4, 16, 40, 128, 512, 2048):
        x, y = tasks.make_mqar(6, length, gen)
        D = tasks.mqar_num_pairs(length)
        assert x.shape == y.shape == (6, length), (length, x.shape)
        assert int(x.min()) >= 0 and int(x.max()) < tasks.MQAR_VOCAB, length
        n_valid = int((y != tasks.MQAR_IGNORE).sum())
        assert n_valid == 6 * D, (length, n_valid, 6 * D)

        # Every answer must be recoverable from the pairs shown earlier in the same row.
        for b in range(6):
            pairs = {
                int(x[b, 2 * i]): int(x[b, 2 * i + 1]) for i in range(D)
            }
            assert len(pairs) == D, f"duplicate key at length={length} row={b}"
            qpos = (y[b] != tasks.MQAR_IGNORE).nonzero().flatten()
            for p in qpos.tolist():
                assert pairs[int(x[b, p])] == int(y[b, p]), (length, b, p)
            # The separator must sit immediately before the first query.
            assert int(x[b, int(qpos[0]) - 1]) == tasks.MQAR_SEP, (length, b)
        print(f"  length={length:>5}: D={D:>2}  queries/row={D:>2}  vocab_max={int(x.max()):>3}  OK")

    # A random-guess baseline, so we can tell "learned nothing" from "learned something".
    print(f"  chance accuracy = 1/{tasks.MQAR_NUM_VALUES} = {1/tasks.MQAR_NUM_VALUES:.4f}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    patch_tasks(here / "tasks.py")
    patch_self_check(here / "tasks.py")
    patch_train_probe(here / "train_probe.py")
    print("\nself-check:")
    self_check()
    print("\nOK")
