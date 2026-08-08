"""Make MQAR difficulty configurable, and register harder variants.

WHY
---
The first MQAR calibration was far too easy: `kda` scored **100% at every length out to 512**.
A ceiling cannot discriminate between arms, so that grid would have produced 40 runs of "1.000"
and answered nothing.

The cause: difficulty in MQAR is set by the memory load *relative to the recurrent state
capacity*, and 8 pairs drawn from a 32-value vocabulary is nothing for a state of
K*V = 64*64 = 4096 floats per head x 4 heads. The Zoology/Based MQAR benchmark uses a vocabulary
in the thousands for exactly this reason.

This patch makes the pair count and vocabulary size parameters of the generator, then registers
graded variants so a calibration sweep can locate the operating point where accuracy is *between*
chance and ceiling -- the only regime where a contrast between arms is measurable.

    mqar        D<=8    vocab 32     (kept: the original, now known to be at ceiling)
    mqar_d32    D<=32   vocab 256
    mqar_d64    D<=64   vocab 512
    mqar_d128   D<=128  vocab 512

Run from ``probes/``. Idempotent.
"""

from pathlib import Path

OLD_CONSTANTS = """MQAR_NUM_KEYS = 32
MQAR_NUM_VALUES = 32
MQAR_NUM_DISTRACTORS = 32"""

NEW_CONSTANTS = """# Defaults for the original (easy) variant. make_mqar takes these as parameters so that harder
# variants can be registered without duplicating the generator; see MQAR difficulty note below.
MQAR_NUM_KEYS = 32
MQAR_NUM_VALUES = 32
MQAR_NUM_DISTRACTORS = 32"""


def patch_generator(path: Path) -> bool:
    """Rewrite ``make_mqar`` and helpers to take difficulty parameters.

    :returns: ``True`` if modified.
    """
    src = path.read_text()
    if "def mqar_vocab(" in src:
        print("tasks.py: generator already parameterised")
        return False

    # Replace the fixed-arity helpers with parameterised ones.
    old_pairs = '''def mqar_num_pairs(length: int) -> int:
    """Number of key-value pairs to use at a given sequence length.

    Grows with length until :data:`MQAR_MAX_PAIRS`, then stays flat so that additional length
    becomes distractor filler (retention distance) rather than extra memory load (capacity).

    :param length: Sequence length.
    :returns: Number of pairs, at least 1.
    """
    return max(1, min(MQAR_MAX_PAIRS, (length - 1) // 3))'''

    new_pairs = '''def mqar_num_pairs(length: int, max_pairs: int = MQAR_MAX_PAIRS) -> int:
    """Number of key-value pairs to use at a given sequence length.

    Grows with length until ``max_pairs``, then stays flat so that additional length becomes
    distractor filler (retention distance) rather than extra memory load (capacity).

    :param length: Sequence length.
    :param max_pairs: Cap on the pair count.
    :returns: Number of pairs, at least 1.
    """
    return max(1, min(max_pairs, (length - 1) // 3))


def mqar_vocab(n_keys: int, n_values: int, n_distractors: int) -> int:
    """Total vocabulary size for an MQAR variant (separator + keys + values + distractors)."""
    return 1 + n_keys + n_values + n_distractors'''

    if old_pairs not in src:
        raise SystemExit("tasks.py: mqar_num_pairs not found -- was the first patch applied?")
    src = src.replace(old_pairs, new_pairs, 1)

    # Parameterise the generator itself.
    start = src.index("def make_mqar(")
    end = src.index("\n\nTASKS = {")
    new_fn = '''def make_mqar(
    batch: int,
    length: int,
    generator: torch.Generator,
    max_pairs: int = MQAR_MAX_PAIRS,
    n_keys: int = MQAR_NUM_KEYS,
    n_values: int = MQAR_NUM_VALUES,
    n_distractors: int = MQAR_NUM_DISTRACTORS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-query associative recall.

    Layout (causal, answers strictly after all pairs)::

        k1 v1 k2 v2 ... kD vD  <distractors ...>  SEP  q1 q2 ... qD

    The target at each query position is the value paired with that key; every other position is
    :data:`MQAR_IGNORE` (``-100``, ``cross_entropy``'s default ``ignore_index``). Keys, values and
    distractors occupy disjoint vocabulary ranges, and keys are drawn **without replacement**
    within a sequence so each has exactly one correct value.

    Difficulty is governed by ``max_pairs`` relative to the model's recurrent state capacity, and
    by ``n_values`` (which sets chance accuracy at ``1 / n_values``). Holding ``max_pairs`` fixed
    while sequence length grows turns a length sweep into a pure retention-distance test rather
    than a capacity test.

    :param batch: Batch size.
    :param length: Sequence length; must leave room for ``2D + 1 + D`` tokens.
    :param generator: RNG for reproducibility.
    :param max_pairs: Cap on pairs per sequence. Must be ``<= n_keys``.
    :param n_keys: Size of the key range.
    :param n_values: Size of the value range; chance accuracy is ``1 / n_values``.
    :param n_distractors: Size of the filler range.
    :returns: ``(inputs, targets)``, both ``[batch, length]`` int64.
    """
    assert max_pairs <= n_keys, f"max_pairs={max_pairs} exceeds n_keys={n_keys}"
    D = mqar_num_pairs(length, max_pairs)
    # Need 2D pair tokens + 1 separator + D queries.
    if length < 3 * D + 1:
        raise ValueError(f"mqar needs length >= {3 * D + 1} for D={D}, got {length}")

    key_base = MQAR_KEY_BASE
    value_base = key_base + n_keys
    distractor_base = value_base + n_values

    inputs = torch.empty(batch, length, dtype=torch.int64)
    targets = torch.full((batch, length), MQAR_IGNORE, dtype=torch.int64)

    key_perm = torch.argsort(torch.rand(batch, n_keys, generator=generator), dim=1)
    keys = key_perm[:, :D] + key_base
    values = torch.randint(0, n_values, (batch, D), generator=generator) + value_base

    inputs[:, 0 : 2 * D : 2] = keys
    inputs[:, 1 : 2 * D : 2] = values

    sep_at = length - D - 1
    n_filler = sep_at - 2 * D
    if n_filler > 0:
        inputs[:, 2 * D : sep_at] = (
            torch.randint(0, n_distractors, (batch, n_filler), generator=generator)
            + distractor_base
        )
    inputs[:, sep_at] = MQAR_SEP

    # Fresh random query order, so answer position carries no information about which key is asked.
    order = torch.argsort(torch.rand(batch, D, generator=generator), dim=1)
    inputs[:, sep_at + 1 :] = torch.gather(keys, 1, order)
    targets[:, sep_at + 1 :] = torch.gather(values, 1, order)
    return inputs, targets

'''
    src = src[:start] + new_fn + src[end:]
    path.write_text(src)
    print("tasks.py: generator parameterised")
    return True


VARIANTS = '''    # Graded MQAR variants. The base "mqar" (D<=8, vocab 32) was measured at 100% accuracy out to
    # length 512 -- at ceiling, hence useless for comparing arms. These raise the memory load
    # relative to the recurrent state so that accuracy lands strictly between chance and ceiling.
    "mqar_d32": {
        "fn": partial(make_mqar, max_pairs=32, n_keys=256, n_values=256, n_distractors=256),
        "in_vocab": mqar_vocab(256, 256, 256),
        "out_vocab": mqar_vocab(256, 256, 256),
    },
    "mqar_d64": {
        "fn": partial(make_mqar, max_pairs=64, n_keys=512, n_values=512, n_distractors=512),
        "in_vocab": mqar_vocab(512, 512, 512),
        "out_vocab": mqar_vocab(512, 512, 512),
    },
    "mqar_d128": {
        "fn": partial(make_mqar, max_pairs=128, n_keys=512, n_values=512, n_distractors=512),
        "in_vocab": mqar_vocab(512, 512, 512),
        "out_vocab": mqar_vocab(512, 512, 512),
    },
'''


def patch_variants(path: Path) -> bool:
    """Register the graded MQAR variants in ``TASKS``.

    :returns: ``True`` if modified.
    """
    src = path.read_text()
    if '"mqar_d32"' in src:
        print("tasks.py: variants already registered")
        return False
    anchor = '''    "mqar": {
        "fn": make_mqar,
        "in_vocab": MQAR_VOCAB,
        "out_vocab": MQAR_VOCAB,
    },
'''
    if anchor not in src:
        raise SystemExit("tasks.py: base mqar TASKS entry not found")
    src = src.replace(anchor, anchor + VARIANTS, 1)
    path.write_text(src)
    print("tasks.py: variants registered (mqar_d32, mqar_d64, mqar_d128)")
    return True


def self_check() -> None:
    """Validate every MQAR variant: shapes, disjoint ranges, key uniqueness, recoverability."""
    import torch

    import tasks

    for name in ("mqar", "mqar_d32", "mqar_d64", "mqar_d128"):
        spec = tasks.TASKS[name]
        gen = torch.Generator().manual_seed(0)
        # Pick a length long enough for the largest D this variant reaches.
        for length in (64, 256, 512, 2048):
            try:
                x, y = spec["fn"](4, length, gen)
            except ValueError as exc:
                print(f"  {name:>10} len={length:>5}: skipped ({exc})")
                continue
            assert x.shape == y.shape == (4, length)
            assert int(x.min()) >= 0 and int(x.max()) < spec["in_vocab"], (name, length)
            scored = y[y != tasks.MQAR_IGNORE]
            assert int(scored.min()) >= 0 and int(scored.max()) < spec["out_vocab"]
            n_q = int((y[0] != tasks.MQAR_IGNORE).sum())
            for b in range(4):
                D_row = int((y[b] != tasks.MQAR_IGNORE).sum())
                pairs = {int(x[b, 2 * i]): int(x[b, 2 * i + 1]) for i in range(D_row)}
                assert len(pairs) == D_row, f"{name}: duplicate key"
                for p in (y[b] != tasks.MQAR_IGNORE).nonzero().flatten().tolist():
                    assert pairs[int(x[b, p])] == int(y[b, p]), (name, length, b, p)
            print(
                f"  {name:>10} len={length:>5}: D={n_q:>3}  vocab={spec['in_vocab']:>5}  "
                f"chance={1/(spec['in_vocab']//3):.4f}  OK"
            )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    patch_generator(here / "tasks.py")
    patch_variants(here / "tasks.py")
    print("\nself-check:")
    self_check()
    print("\nOK")
