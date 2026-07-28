"""
Argv pre-parsing for the Mamba-3 smoke tests.

The smoke tests hand ``sys.argv`` to :func:`olmo_core.internal.experiment.main`, which owns the
``dry_run``/``launch`` subcommands and the dotted ``--key=value`` override syntax. Knobs that have
to be known *while* the config is being built cannot go through that override syntax: overrides
merge into the :class:`ExperimentConfig` only after ``build_experiment_config`` has returned, so a
value read from the config during the build (the sentinel's expected rotation block size, for
instance) would see the pre-override value and disagree with the model that actually runs.

Popping the option out of ``argv`` first sidesteps that ordering entirely -- the value is in hand
before the config exists. This mirrors ``_pop_opt`` in
``src/scripts/train/OLMo3/OLMo3-370M-mamba3.py``, which solves the same problem for the real runs.
"""

from typing import List, Optional, Sequence, Tuple

__all__ = ["pop_opt", "pop_int_opt", "pop_flag", "popped_tokens"]


def popped_tokens(before: Sequence[str], after: Sequence[str]) -> List[str]:
    """
    The tokens the ``pop_*`` helpers removed, in their original order.

    A Beaker launch rebuilds the remote command from whatever is left in ``sys.argv``, so a flag
    popped here would not reach the job -- it would run the defaults while the launching command
    line said otherwise. Hand these back to ``build_launch_config`` so the remote command matches
    what was actually typed.

    ``after`` must be a subsequence of ``before``, which holds because popping only ever removes.
    """
    remaining = iter(after)
    nxt = next(remaining, None)
    removed: List[str] = []
    for token in before:
        if token == nxt:
            nxt = next(remaining, None)
        else:
            removed.append(token)
    return removed


def pop_opt(
    argv: Sequence[str], name: str, default: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    """
    Pull ``--name VALUE`` / ``--name=VALUE`` out of ``argv``.

    :param argv: Arguments to scan; not mutated.
    :param name: Long option name, including the leading ``--``.
    :param default: Returned when the option is absent.

    :returns: The option's value and the remaining arguments.

    :raises SystemExit: If ``--name`` is given as the final argument, with no value after it.
    """
    value = default
    rest: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return value, rest


def pop_flag(argv: Sequence[str], name: str) -> Tuple[bool, List[str]]:
    """
    Pull a valueless ``--name`` out of ``argv``.

    :returns: Whether the flag was present, and the remaining arguments.
    """
    present = name in argv
    return present, [arg for arg in argv if arg != name]


def pop_int_opt(
    argv: Sequence[str], name: str, default: Optional[int] = None
) -> Tuple[Optional[int], List[str]]:
    """
    :func:`pop_opt` for an integer-valued option.

    :raises SystemExit: If the value is not an integer. Rejecting it here rather than letting
        ``int()`` raise keeps a typo a usage error instead of a traceback.
    """
    raw, rest = pop_opt(argv, name, None)
    if raw is None:
        return default, rest
    try:
        return int(raw), rest
    except ValueError:
        raise SystemExit(f"{name} expects an integer, got {raw!r}")
