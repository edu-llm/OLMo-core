"""Pick a HuggingFace export out of a prefix a run is still writing into.

WHY THIS IS NOT ``Checkpointer.latest_checkpoint`` WITH A DIFFERENT ARGUMENT. That function
already solves this problem for core checkpoints, correctly: :meth:`Checkpointer.find_checkpoints`
drops every directory failing :meth:`Checkpointer.dir_is_checkpoint` and the latest is the
highest of what survives, so a save in progress is skipped rather than half-read. Anything
consuming ``checkpoints/step{N}`` should call it and stop reading here.

It is the wrong test for ``hf/step{N}``, and not marginally. ``dir_is_checkpoint`` looks for
``.metadata``, ``train/rank0.pt`` and ``.metadata.json``; a HuggingFace export has none of
them, so it answers False for a complete one just as readily as for a torn one. A lane that
took the rule literally would poll a prefix filling up with exports and never read any of
them -- which at least fails visibly, and is the better of the two ways to be wrong here.

WHAT A TORN EXPORT LOOKS LIKE, WHICH IS WHY A LISTING IS NOT ENOUGH. ``step{N}`` exists from
the moment its first object lands, and on S3 there is no directory to be absent in the
meantime. ``save_pretrained`` writes ``config.json`` first, then one shard at a time, then
``model.safetensors.index.json``; the 7B MoE at bfloat16 is 13.27 GiB, so the gap between the
first object and the last is minutes long and a consumer taking the highest step number off a
listing lands in it. What it gets is a model whose config parses, whose index is absent or
whose shards are short, and which loads into some tensors that were never written.

So the rule that holds for exports is the same shape as the one for checkpoints -- the highest
step that is whole -- with a different test for whole: the config is there and so is every
weight file the export says it has.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Generator, List, Optional, Tuple

from olmo_core.io import file_exists, list_directory, normalize_path, resource_path

__all__ = [
    "EXPORT_DIR",
    "export_weight_files",
    "export_is_complete",
    "find_exports",
    "latest_complete_export",
]

log = logging.getLogger(__name__)

#: How an export directory is named, which is ``Checkpointer.checkpoint_dirname`` because
#: ``HFConverterCallback`` names the export after the step it converted.
EXPORT_DIR = re.compile(r"^step(\d+)$")

_CONFIG = "config.json"
_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"


def export_weight_files(directory: str) -> Optional[List[str]]:
    """The weight files this export says it consists of, or ``None`` if it does not say yet.

    ``None`` is a real answer rather than an error: an export mid-write has no index and one
    shard on disk, and that is indistinguishable from a complete single-file export except by
    asking. Callers get the distinction from :func:`export_is_complete`, which treats a missing
    index and a missing ``model.safetensors`` as the same incomplete state.

    Read out of the index rather than inferred from the ``model-00001-of-00009`` naming, which
    is a transformers implementation detail and has changed spelling before.
    """
    directory = normalize_path(directory)
    if file_exists(f"{directory}/{_INDEX}"):
        try:
            index = json.loads(resource_path(directory, _INDEX).read_text())
        except (OSError, ValueError) as exc:
            # A short or unparseable index is the index being written right now.
            log.debug("%s/%s could not be read as an index: %s", directory, _INDEX, exc)
            return None
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return None
        return sorted(set(weight_map.values()))
    if file_exists(f"{directory}/{_SINGLE}"):
        return [_SINGLE]
    return None


def export_is_complete(directory: str) -> bool:
    """Whether every object a consumer of this export would open is already there.

    The counterpart of ``Checkpointer.dir_is_checkpoint`` for ``hf/step{N}``, and it is
    deliberately the same shape: a predicate over one directory, cheap enough to call on every
    candidate in a listing, and answering about the directory rather than about the run.

    It does not read the weights. Whether the tensors are the ones the model was trained with
    is a different question and a much more expensive one -- ``.edullm/verify_hf_export.py``
    is where that is asked, and this is the check that decides there is anything worth asking
    it about.
    """
    directory = normalize_path(directory)
    if not file_exists(f"{directory}/{_CONFIG}"):
        return False
    weight_files = export_weight_files(directory)
    if weight_files is None:
        return False
    return all(file_exists(f"{directory}/{name}") for name in weight_files)


def find_exports(directory: str) -> Generator[Tuple[int, str], None, None]:
    """Every complete export under a prefix, as ``(step, path)``.

    Named and shaped after ``Checkpointer.find_checkpoints`` so that the two read as the pair
    they are, and so that a reader who knows one does not have to learn the other.
    """
    directory = normalize_path(directory)
    for path in list_directory(directory, include_files=False):
        match = EXPORT_DIR.match(os.path.basename(normalize_path(path)))
        if match is None:
            continue
        if not export_is_complete(path):
            log.debug("%s is not a whole export yet, so it is not a candidate", path)
            continue
        yield int(match.group(1)), path


def latest_complete_export(directory: str) -> str:
    """The highest step under this prefix that is whole.

    THE HIGHEST *COMPLETE* ONE AND NOT THE HIGHEST ONE, WHICH IS THE ENTIRE POINT. A consumer
    polling a prefix while the run writes into it sees the step being written as soon as its
    first object lands. Taking the maximum off the listing returns that one, every time, which
    is the one directory guaranteed to be unfinished.

    :raises FileNotFoundError: If nothing under the prefix is a whole export. A prefix holding
        only a torn export raises rather than returning it, so a poller waits and tries again
        instead of reading it.
    """
    directory = normalize_path(directory)
    latest: Optional[Tuple[int, str]] = None
    for step, path in find_exports(directory):
        if latest is None or step > latest[0]:
            latest = (step, path)
    if latest is None:
        raise FileNotFoundError(f"No complete HuggingFace export found in '{directory}'")
    return latest[1]
