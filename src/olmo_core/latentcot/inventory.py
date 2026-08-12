"""
Discover which arm checkpoints a finished (or dead) run actually wrote.

``eval.py`` needs every arm spelled out as ``--arm A2=/path``, which assumes somebody knows what
was written. After a run that died mid-training, nobody does: the arms that reached a save
interval have checkpoints and the rest have nothing, and which is which decides whether the gates
can be computed at all. This module answers that from the checkpoint root alone.

Importable (rather than living in the eval script) so the selection and path rules below can be
unit-tested without S3 -- the same reason :mod:`olmo_core.latentcot.train_driver` is a module.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "ARM_NAMES",
    "arm_prefixes",
    "select_latest",
    "step_of",
    "steps_available",
    "select_common_step",
    "apply_common_step",
    "discover_arm",
    "take_inventory",
    "describe_inventory",
    "missing_gate_notes",
]

ARM_NAMES: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")

# `train_driver._save_rolling` writes stepN.pt; `train_codi.py` writes model.pt after the last step.
_STEP_RE = re.compile(r"^step(\d+)\.pt$")

# Recorded in the inventory but never selected. "Latest" means the last weights written, and
# best.pt is by construction an *earlier* step that happened to score better.
_NOTED_FILES: Tuple[str, ...] = ("best.pt", "best.json", "metrics.json", "train.log")


def arm_prefixes(root: str, arm: str, seed: int) -> Tuple[str, str]:
    """
    The two spellings an arm's checkpoint directory may have under ``root``, in probe order.

    The double slash is neither a typo nor cosmetic. ``$EDULLM_CHECKPOINT_DIR`` already ends in
    ``/`` and the training command appended ``/A$i``, so ``train_codi.py`` built
    ``…/checkpoints//A0/A0-seed1``; :func:`~olmo_core.io.upload` strips only *leading and
    trailing* slashes, so the interior ``//`` survives into the literal S3 key. S3 keys are opaque
    byte strings, so ``checkpoints//A0`` and ``checkpoints/A0`` name **different objects** and
    only one of them exists. Which one depends on whether the run that wrote them carried the
    defect, so both are probed rather than guessed at.

    :param root: The run's checkpoint root, with or without a trailing slash.
    :param arm: Arm name, e.g. ``"A0"``.
    :param seed: The ``--seed`` the arm trained under; it names the leaf directory.

    :returns: ``(double_slash_spelling, single_slash_spelling)``.
    """
    base = str(root).rstrip("/")
    leaf = f"{arm}/{arm}-seed{seed}"
    return f"{base}//{leaf}", f"{base}/{leaf}"


def select_latest(files: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Pick the latest checkpoint from a mapping of basename to full path.

    ``model.pt`` wins outright -- ``train_codi.py`` writes it after the final step, so it is later
    than any ``stepN.pt``. Otherwise the highest ``N`` wins, compared **numerically**: sorting
    those basenames as strings ranks ``step500.pt`` above ``step2000.pt``.

    :param files: Basenames to full paths, as :func:`discover_arm` returns.
    :returns: ``(basename, full_path)`` of the latest checkpoint, or ``None`` if there is none.
    """
    if "model.pt" in files:
        return "model.pt", files["model.pt"]
    steps = [(int(match.group(1)), name) for name in files if (match := _STEP_RE.match(name))]
    if not steps:
        return None
    _, name = max(steps)
    return name, files[name]


def step_of(name: str) -> Optional[int]:
    """
    The step number in a ``stepN.pt`` basename, or ``None`` for anything else.

    :param name: A checkpoint basename.
    :returns: ``N``, or ``None`` for ``model.pt``/``best.pt``/a non-checkpoint.
    """
    match = _STEP_RE.match(name)
    return int(match.group(1)) if match else None


def steps_available(names: Iterable[str]) -> List[int]:
    """
    The sorted step numbers among a collection of checkpoint basenames.

    :param names: Basenames -- a list, or the keys of a basename-to-path mapping.
    :returns: Ascending step numbers; empty if none are ``stepN.pt``.
    """
    return sorted(step for name in names if (step := step_of(name)) is not None)


def select_common_step(inventory: Dict[str, dict]) -> Optional[int]:
    """
    The largest ``stepN.pt`` that **every** arm in ``inventory`` wrote.

    Arms cost wildly different amounts per step -- A0/A1 run one forward per example, A2-A4 run
    ``K + 2`` -- so on a shared wall-clock budget they stop at different steps. Comparing them as
    they stand puts a difference in optimization budget straight into the gates, which is a
    confound on exactly the comparison gate A is defined on. Evaluating every arm at the same step
    removes it, and the dense checkpoint ladder is what makes that recoverable after the fact
    rather than something that had to be predicted before the run.

    :param inventory: The mapping from :func:`take_inventory`.
    :returns: The common step, or ``None`` if any arm has no ``stepN.pt`` (in which case there is
        no matched budget to be had and the caller should say so rather than silently compare
        mismatched arms).
    """
    if not inventory:
        return None
    per_arm = [set(steps_available(entry["files"])) for entry in inventory.values()]
    if any(not steps for steps in per_arm):
        return None
    common = set.intersection(*per_arm)
    return max(common) if common else None


def apply_common_step(inventory: Dict[str, dict], step: int) -> Dict[str, dict]:
    """
    Re-point every arm's selection at ``stepN.pt`` for the given step.

    :param inventory: The mapping from :func:`take_inventory`. Not mutated.
    :param step: The step to select, normally from :func:`select_common_step`.

    :returns: A new inventory whose ``selected``/``selected_path`` name that step. An arm lacking
        it keeps ``selected = None`` rather than falling back to a different step, because a silent
        fallback is the confound this function exists to remove.
    """
    name = f"step{step}.pt"
    out: Dict[str, dict] = {}
    for arm, entry in inventory.items():
        new = dict(entry)
        if name in entry["files"] and entry["prefix"] is not None:
            new["selected"] = name
            new["selected_path"] = f"{entry['prefix']}/{name}"
        else:
            new["selected"] = None
            new["selected_path"] = None
        out[arm] = new
    return out


def discover_arm(root: str, arm: str, seed: int) -> Tuple[Optional[str], Dict[str, str]]:
    """
    List one arm's checkpoint directory, trying both slash spellings.

    Only the arm's own leaf directory is listed, never the checkpoint root. Listing the root would
    be actively misleading: ``_s3_list_directory`` strips the slashes off the common prefixes it
    yields, so a root carrying the ``//`` defect reports a child whose name equals the root
    itself, and anything that recurses or filters on ``startswith`` misbehaves.

    :param root: The run's checkpoint root.
    :param arm: Arm name.
    :param seed: The seed naming the leaf directory.

    :returns: ``(prefix_that_matched, {basename: full_path})``, or ``(None, {})`` if neither
        spelling holds a file.
    """
    from olmo_core.io import list_directory

    for prefix in arm_prefixes(root, arm, seed):
        files: Dict[str, str] = {}
        try:
            for path in list_directory(prefix, include_dirs=False):
                files[str(path).rsplit("/", 1)[-1]] = str(path)
        except FileNotFoundError:
            # A missing *local* prefix raises; a missing S3 prefix merely yields nothing.
            continue
        if files:
            return prefix, files
    return None, {}


def take_inventory(root: str, arms: List[str], seed: int) -> Dict[str, dict]:
    """
    Discover every arm under ``root`` and describe what was found.

    :param root: The run's checkpoint root.
    :param arms: Arm names to look for.
    :param seed: The seed naming each leaf directory.

    :returns: ``{arm: {"prefix", "files", "selected", "selected_path", "notes"}}``. An arm with no
        checkpoint is still present, with ``selected`` set to ``None`` -- absence is the finding
        the caller is usually after, so it is recorded rather than dropped.
    """
    inventory: Dict[str, dict] = {}
    for arm in arms:
        prefix, files = discover_arm(root, arm, seed)
        latest = select_latest(files)
        inventory[arm] = {
            "prefix": prefix,
            "files": sorted(files),
            "selected": latest[0] if latest else None,
            "selected_path": latest[1] if latest else None,
            "notes": [name for name in _NOTED_FILES if name in files],
        }
    return inventory


def missing_gate_notes(found: set) -> List[str]:
    """
    What the arms in ``found`` cannot answer.

    Gate A comes from A2 minus A0 and gate B from A2/A3/A4, so an inventory holding only the
    A0/A1 anchors supports **no gate at all**. Saying so is the whole point of taking an
    inventory before spending a machine on one.

    :param found: Arm names that have a checkpoint.
    :returns: Human-readable warning lines; empty when every gate is computable.
    """
    notes: List[str] = []
    if not {"A2", "A0"} <= found:
        need = ", ".join(sorted({"A2", "A0"} - found))
        notes.append(f"gate A CANNOT be computed: needs both A2 and A0; missing {need}.")
    if not {"A2", "A3", "A4"} <= found:
        need = ", ".join(sorted({"A2", "A3", "A4"} - found))
        notes.append(f"gate B will be PARTIAL: needs A2, A3, A4; missing {need}.")
    if found and not {"A2", "A3", "A4"} & found:
        notes.append("No CODI arm is present, so this run measures only the anchors.")
    return notes


def describe_inventory(inventory: Dict[str, dict]) -> str:
    """
    Render an inventory, and what it implies for the gates, as a printable block.

    :param inventory: The mapping from :func:`take_inventory`.
    :returns: A multi-line string.
    """
    lines = ["", "=" * 78, "CHECKPOINT INVENTORY", "=" * 78]
    for arm, entry in inventory.items():
        if entry["selected"] is None:
            lines.append(f"  {arm}: NO CHECKPOINT FOUND")
            continue
        extra = f"  (also: {', '.join(entry['notes'])})" if entry["notes"] else ""
        lines.append(f"  {arm}: {len(entry['files'])} file(s), latest = {entry['selected']}{extra}")
        lines.append(f"      {entry['selected_path']}")

    found = {arm for arm, entry in inventory.items() if entry["selected"] is not None}
    lines.append("-" * 78)
    # Only arms that were actually looked for. Listing every arm in ARM_NAMES would report an arm
    # the caller excluded with --arms as "missing", which is a different claim about the bucket.
    missing = [arm for arm in inventory if arm not in found]
    if missing:
        lines.append(f"  MISSING: {', '.join(missing)}")
    lines.extend(f"  ** {note}" for note in missing_gate_notes(found))
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)
