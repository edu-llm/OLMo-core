"""
Open a checkpoint: find it, rebuild the corpus that produced it, load the weights.

Everything downstream needs this and nothing else does, so it is one module with one job.

Three facts about the platform's checkpoints shape it, all verified against a real run rather than
assumed:

- They are **sharded**. A cell trained on four ranks writes sixteen ``.distcp`` files, and
  :func:`~olmo_core.distributed.checkpoint.load_model_and_optim_state` reshards them into a single
  unsharded model in one process with no process group. So scoring is a plain single-process job.
- Their ``config.json`` carries the **cell**, not the model config. ``train_cell.py`` records the cell
  spec and the corpus fingerprints; the model architecture is then re-derived from the cell, which is
  the authoritative source anyway -- a row implies exactly one width.
- The corpus is **generated, not stored**. So rebuilding it is the only way to know what the model was
  trained on, and the fingerprints are how we know the rebuild is right. :func:`load` verifies them by
  default and refuses on a mismatch, because a scorer that silently scored the wrong corpus would
  produce numbers that look entirely reasonable.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from olmo_core.exceptions import OLMoConfigurationError

from .. import cells as cell_module
from ..corpus.build import BuiltCorpus
from ..ladder import sizes as sizes_module

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

RECORD_KEY = "factcrowd"
"""Top-level key ``train_cell.py`` writes its cell record under in ``config.json``."""


@dataclass(frozen=True)
class CheckpointRef:
    """
    Where a checkpoint is and which step it holds.

    :param step: Optimizer step the checkpoint was written at.
    :param path: Directory holding ``config.json`` and ``model_and_optim/``. Local or ``s3://``.
    """

    step: int
    path: str

    @property
    def model_dir(self) -> str:
        """The sub-directory the loader wants: ``load_model_and_optim_state`` does not append it."""
        return f"{str(self.path).rstrip('/')}/model_and_optim"


def find_checkpoints(prefix: str) -> Tuple[CheckpointRef, ...]:
    """
    Every complete checkpoint under a run prefix, in step order.

    Delegates to :meth:`olmo_core.train.checkpoint.Checkpointer.find_checkpoints`, which already
    validates that ``train/rank0.pt``, ``model_and_optim/.metadata`` and ``.metadata.json`` all exist --
    so a checkpoint that was still uploading when the job died is skipped rather than half-read. Globbing
    for ``step*`` would not know the difference.

    :param prefix: A run's checkpoint directory, e.g.
        ``s3://.../runs/<run id>/cell-3/checkpoints``.

    :returns: The checkpoints, ascending by step.
    """
    from olmo_core.train.checkpoint import Checkpointer

    found = [
        CheckpointRef(step=step, path=str(path))
        for step, path in Checkpointer.find_checkpoints(prefix)
    ]
    return tuple(sorted(found, key=lambda ref: ref.step))


def read_record(path: str) -> Dict[str, Any]:
    """
    Read the factcrowd block out of a checkpoint's ``config.json``.

    :param path: The checkpoint directory.

    :returns: The record: ``cell``, ``resolved``, ``fingerprints`` and ``checkpoint_steps``.

    :raises OLMoConfigurationError: If the file or the block is missing. A checkpoint without it cannot
        be scored -- the corpus is generated, so without the cell there is nothing to rebuild.
    """
    from olmo_core.io import file_exists, get_bytes_range, get_file_size, join_path

    config_path = str(join_path(path, "config.json"))
    if not file_exists(config_path):
        raise OLMoConfigurationError(f"no config.json at {config_path}")
    raw = get_bytes_range(config_path, 0, get_file_size(config_path))
    parsed = json.loads(raw.decode("utf-8"))
    if RECORD_KEY not in parsed:
        raise OLMoConfigurationError(
            f"{config_path} has no '{RECORD_KEY}' block, so the cell that produced it is unknown. "
            f"Checkpoints written before that record was added cannot be scored."
        )
    return parsed[RECORD_KEY]


@dataclass
class LoadedCheckpoint:
    """
    A checkpoint, opened and ready to score.

    :param ref: Where it came from.
    :param cell: The cell that produced it, from the saved record.
    :param record: The whole saved record, including the corpus fingerprints.
    :param corpus: The rebuilt corpus, with reasoning tasks on the **eval** split.
    :param model: The loaded model, in eval mode on the requested device.
    """

    ref: CheckpointRef
    cell: cell_module.CellSpec
    record: Dict[str, Any]
    corpus: BuiltCorpus
    model: Any

    @property
    def resolved(self) -> cell_module.ResolvedCell:
        """The cell resolved against the rebuilt vocabulary, so both parameter bases are real."""
        return self.cell.resolve(vocab_size=self.corpus.vocabulary.padded_size())


def load(
    ref: CheckpointRef,
    *,
    work_dir: Path,
    device: str = "cpu",
    verify: bool = True,
    with_model: bool = True,
    dtype: Optional[Any] = None,
    corpus: Optional[BuiltCorpus] = None,
) -> LoadedCheckpoint:
    """
    Rebuild the corpus and load the weights for one checkpoint.

    :param ref: Which checkpoint.
    :param work_dir: Local scratch for the rebuilt entity table and for staging remote shards.
    :param device: Where to put the model.
    :param verify: Check the rebuilt corpus against the saved fingerprints. Leave this on; it is the
        only thing standing between a wrong rebuild and a plausible-looking score.
    :param dtype: Cast the built model to this ``torch.dtype`` before loading. ``None`` leaves it at the
        config's own, which is float32.
    :param with_model: Load the weights. ``False`` rebuilds the corpus alone, which is enough for tests
        and for inspecting what a run trained on.
    :param corpus: A corpus already built for this cell, to reuse across the cell's checkpoints. Only the
        weights differ between them, and rebuilding the entity table for each one was costing more than
        the scoring did -- an audit measured the fact stream alone at 4.7s against 0.3s without it, paid
        ten times per cell. Reusing it is checked, not assumed: the cell id must match.

    :returns: The opened checkpoint.

    :raises OLMoConfigurationError: If the record is missing, or a fingerprint disagrees.
    """
    record = read_record(ref.path)
    cell = cell_module.CellSpec.from_dict(dict(record["cell"]))
    resolved = cell.resolve()
    if corpus is not None and corpus.spec_cell_id != cell.cell_id:
        raise OLMoConfigurationError(
            f"a corpus for cell '{corpus.spec_cell_id}' was offered for a checkpoint of "
            f"'{cell.cell_id}'. Reusing the wrong corpus would score every checkpoint against a "
            f"different one than it was trained on."
        )
    # The eval split, and none of the packed volumes. Scoring reads the tasks and the renderer, both of
    # which are built either way; the offset index over billions of fact tokens is pure cost here. It was
    # being built anyway until an audit measured it at 4.7s a checkpoint.
    if corpus is None:
        corpus = BuiltCorpus(resolved, work_dir, split="eval", with_streams=False)
        if verify:
            verify_fingerprints(corpus, record)
    elif verify:
        # Still verified per checkpoint: reuse saves the rebuild, not the check, and two checkpoints of
        # one run could in principle carry different records.
        verify_fingerprints(corpus, record)

    model = None
    if with_model:
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        model_config = sizes_module.build(
            cell.ladder_row,
            corpus.vocabulary.padded_size(),
            tie_word_embeddings=True,
            init_seed=cell.init_seed,
        )
        # init_device="cpu" rather than "meta": a meta model cannot be copied into, and the checkpoint
        # overwrites every parameter anyway, so there is nothing to gain by deferring allocation.
        built = model_config.build(init_device=device)
        if dtype is not None:
            # Stated rather than defaulted. Training set bfloat16 through FSDP's `param_dtype`, which is a
            # mixed-precision setting and not the dtype the shards hold, so scoring has always run in
            # float32 and nothing said so. The platform's precision guard reads the *text* of a command and
            # cannot see a precision chosen in code, so a submission that does not name one can be admitted
            # onto a card whose hardware lacks it -- and then dies on the first kernel that needs it.
            built = built.to(dtype=dtype)
        model = built
        load_model_and_optim_state(
            ref.model_dir,
            model,
            # Staged rather than read through ranged GETs: one pass over the shards instead of a request
            # per tensor, which matters when the shards are in S3 and there are ten steps per cell.
            pre_download=True,
            work_dir=str(work_dir / "shards"),
        )
        model.eval()

    return LoadedCheckpoint(ref=ref, cell=cell, record=record, corpus=corpus, model=model)


def verify_fingerprints(corpus: BuiltCorpus, record: Dict[str, Any]) -> None:
    """
    Check a rebuilt corpus against what the run recorded.

    :param corpus: The rebuild.
    :param record: The saved record.

    :raises OLMoConfigurationError: On any disagreement, naming which one. A mismatch means the cell
        resolved differently than it did at training time -- a changed default, a changed word list, a
        changed template -- and every score from that rebuild would be measuring a different corpus.
    """
    saved = dict(record.get("fingerprints") or {})
    checks = [
        ("schema", corpus.corpus_schema.schema.fingerprint()),
        ("vocabulary", corpus.vocabulary.fingerprint()),
    ]
    # The renderer decides where value tokens land, so a changed seed or template set moves every span the
    # bit counter charges. Schema and vocabulary are both blind to it.
    if corpus.renderer is not None:
        checks.append(("renderer", corpus.renderer.fingerprint()))
    for name, rebuilt in checks:
        expected = saved.get(name)
        if expected is None:
            continue
        if expected != rebuilt:
            raise OLMoConfigurationError(
                f"the rebuilt {name} does not match the one this checkpoint was trained with "
                f"(saved {expected[:16]}, rebuilt {rebuilt[:16]}). Scoring it would measure a "
                f"different corpus from the one the model saw."
            )

    # The tasks are checked on their *structure* rather than their stream digest: measurement generates
    # the eval split, so it cannot reproduce the train digest a run recorded. Structure is what matters
    # anyway -- an expression length or a domain token that changed alters every item scored.
    saved_structure = dict(saved.get("reasoning_structure") or {})
    for task in corpus.tasks:
        expected = saved_structure.get(task.name)
        if expected is None:
            continue
        rebuilt = task.structure_fingerprint()
        if expected != rebuilt:
            raise OLMoConfigurationError(
                f"the rebuilt '{task.name}' endpoint does not match the one this checkpoint was trained "
                f"with (saved {expected[:16]}, rebuilt {rebuilt[:16]}). Its item shape has changed, so "
                f"every score would be on a different task."
            )
    for name in saved_structure:
        if name not in {task.name for task in corpus.tasks}:
            raise OLMoConfigurationError(
                f"this checkpoint was trained with a '{name}' endpoint that the rebuild does not carry, "
                f"so that endpoint cannot be scored and the others may not be comparable"
            )


def forward_fn(model: Any, *, device: str = "cpu"):
    """
    Wrap a loaded model as the ``forward`` callable the scorers take.

    One place where torch appears in the measurement path. ``ce_loss`` comes back as ``(batch,
    sequence)`` with zeros wherever the label was ignored, and labels are built by OLMo-core's own
    :func:`~olmo_core.data.utils.get_labels` rather than shifted here -- the shift is subtle enough that
    restating it would be a second place to get it wrong.

    :param model: A loaded model in eval mode.
    :param device: Where to run.

    :returns: A callable taking ``(batch, sequence)`` ids and returning ``(ce_loss, logits)``.
    """
    import torch

    from olmo_core.data.utils import get_labels

    def forward(batch: "np.ndarray"):
        input_ids = torch.from_numpy(batch.astype("int64")).to(device)
        labels = get_labels({"input_ids": input_ids})
        with torch.no_grad():
            out = model(
                input_ids,
                labels=labels,
                ignore_index=-100,
                loss_reduction="none",
                return_logits=True,
            )
        return out.ce_loss.float().cpu().numpy(), out.logits.float().cpu().numpy()

    return forward
