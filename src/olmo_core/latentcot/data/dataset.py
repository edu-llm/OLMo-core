"""
In-memory dataset + collation for the latent-CoT experiments (PRD Phase 2.2).

Reads a JSONL of :class:`Example` records (produced by
``src/scripts/latentcot/gen_graph_data.py``) and yields the teacher/student token
views from :func:`encode_example`. ``collate`` right-pads a list of items into
batched tensors; padding positions are excluded from the loss because ``label_mask``
is padded with ``False`` (see :func:`olmo_core.data.utils.get_labels`).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset

from ..tokens import TOKENIZER_CONFIG
from .encode import encode_example
from .graph_gen import Example

__all__ = ["LatentCotDataset", "collate", "codi_collate"]


class LatentCotDataset(Dataset):
    """A map-style dataset over graph-reachability instances stored as JSONL."""

    def __init__(self, path: Union[str, Path], num_continuous_thoughts: int):
        """
        :param path: Path to a ``.jsonl`` of :meth:`Example.to_dict` records.
        :param num_continuous_thoughts: ``K`` latent slots in the student view.
        """
        self.num_continuous_thoughts = num_continuous_thoughts
        with Path(path).open() as f:
            self.examples: List[Example] = [Example.from_dict(json.loads(line)) for line in f]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return encode_example(self.examples[index], self.num_continuous_thoughts)


def codi_collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for the CODI/arm train module: return the per-example dicts as a list under
    ``"examples"``. The continuous-thought student is processed per example, so no padded
    tensor batch is built here (see :mod:`olmo_core.latentcot.loss`).
    """
    return {"examples": list(items)}


def _pad(seqs: List[List[int]], value: int, dtype: torch.dtype) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs], dtype=dtype)


def collate(
    items: List[Dict[str, Any]], pad_id: int = TOKENIZER_CONFIG.pad_token_id
) -> Dict[str, Any]:
    """
    Right-pad a list of :func:`encode_example` dicts into batched tensors.

    Student and teacher views are padded independently. ``label_mask`` pads with
    ``False`` so padding never contributes to the loss. Variable-length metadata
    (e.g. ``frontiers``) is returned as a Python list for downstream probing.
    """
    return {
        "input_ids": _pad([it["input_ids"] for it in items], pad_id, torch.long),
        "label_mask": _pad([it["label_mask"] for it in items], 0, torch.bool),
        "teacher_input_ids": _pad([it["teacher_input_ids"] for it in items], pad_id, torch.long),
        "teacher_label_mask": _pad([it["teacher_label_mask"] for it in items], 0, torch.bool),
        "bot_pos": torch.tensor([it["bot_pos"] for it in items], dtype=torch.long),
        "distill_pos": torch.tensor([it["distill_pos"] for it in items], dtype=torch.long),
        "teacher_distill_pos": torch.tensor(
            [it["teacher_distill_pos"] for it in items], dtype=torch.long
        ),
        "num_continuous_thoughts": items[0]["num_continuous_thoughts"],
        "reachable": torch.tensor([it["reachable"] for it in items], dtype=torch.bool),
        "depth": torch.tensor([it["depth"] for it in items], dtype=torch.long),
        "frontiers": [it["frontiers"] for it in items],
        "target": [it["target"] for it in items],
    }
