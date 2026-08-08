"""What ``olmo_core.nn.hf.exports`` refuses to hand a consumer, and why each case is reachable.

Built out of files rather than out of a real export. Every one of these states is a moment
part-way through ``save_pretrained``, and reproducing them by interrupting a 13.27 GiB write
would test the same predicate at a thousand times the cost.
"""

import json
from pathlib import Path

import pytest

from olmo_core.nn.hf import (
    export_is_complete,
    export_weight_files,
    find_exports,
    latest_complete_export,
)
from olmo_core.train.checkpoint import Checkpointer


def whole_export(directory: Path, shards: int = 3) -> Path:
    """One export as it looks after the last object lands."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    names = [f"model-{n + 1:05}-of-{shards:05}.safetensors" for n in range(shards)]
    for name in names:
        (directory / name).write_bytes(b"weights")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"layer.{n}": names[n % shards] for n in range(6)}})
    )
    return directory


def test_a_whole_export_is_whole(tmp_path: Path):
    assert export_is_complete(str(whole_export(tmp_path / "step2400")))


def test_a_single_file_export_needs_no_index(tmp_path: Path):
    """Below the shard threshold transformers writes one file and no index at all.

    So "the index is missing" cannot be the test for torn on its own -- a small model would
    never pass it.
    """
    directory = tmp_path / "step2400"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"weights")

    assert export_weight_files(str(directory)) == ["model.safetensors"]
    assert export_is_complete(str(directory))


def test_the_config_alone_is_not_an_export(tmp_path: Path):
    """``save_pretrained`` writes config.json first, so this is the first observable state."""
    directory = tmp_path / "step2400"
    directory.mkdir()
    (directory / "config.json").write_text("{}")

    assert export_weight_files(str(directory)) is None
    assert not export_is_complete(str(directory))


def test_shards_present_but_no_index_yet_is_not_an_export(tmp_path: Path):
    """The index is written last, so every multi-shard export passes through this state."""
    directory = tmp_path / "step2400"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model-00001-of-00003.safetensors").write_bytes(b"weights")

    assert not export_is_complete(str(directory))


def test_an_index_naming_a_shard_that_is_not_there_is_not_an_export(tmp_path: Path):
    """The case a listing cannot see, and the one a consumer would load and get wrong.

    Reachable on a resumed or retried export rather than on a straight write: the index from an
    earlier attempt is already in the prefix and names shards this attempt has not rewritten.
    """
    directory = whole_export(tmp_path / "step2400")
    (directory / "model-00002-of-00003.safetensors").unlink()

    assert not export_is_complete(str(directory))


def test_an_unparseable_index_is_being_written_rather_than_an_error(tmp_path: Path):
    directory = whole_export(tmp_path / "step2400")
    (directory / "model.safetensors.index.json").write_text('{"weight_map": {"a": ')

    assert export_weight_files(str(directory)) is None
    assert not export_is_complete(str(directory))


def test_the_latest_is_the_latest_whole_one_and_not_the_latest_one(tmp_path: Path):
    """THE DEFECT THIS MODULE EXISTS FOR, WRITTEN AS THE STATE A POLLER ACTUALLY MEETS.

    Two exports are done and the third is being written right now. Every consumer that took
    ``max`` over the step numbers in a listing would read step 7200 -- which is the one
    directory here guaranteed to be incomplete, because it is the one still being written.
    """
    whole_export(tmp_path / "step2400")
    whole_export(tmp_path / "step4800")
    torn = tmp_path / "step7200"
    torn.mkdir()
    (torn / "config.json").write_text("{}")

    assert sorted(step for step, _ in find_exports(str(tmp_path))) == [2400, 4800]
    assert latest_complete_export(str(tmp_path)).endswith("step4800")


def test_a_prefix_holding_only_a_torn_export_raises_rather_than_returning_it(tmp_path: Path):
    """A poller must wait, and a return value it has to test for is one it can forget to."""
    torn = tmp_path / "step2400"
    torn.mkdir()
    (torn / "config.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="No complete HuggingFace export"):
        latest_complete_export(str(tmp_path))


def test_directories_that_are_not_steps_are_not_candidates(tmp_path: Path):
    whole_export(tmp_path / "step800")
    whole_export(tmp_path / "latest")
    whole_export(tmp_path / "step800-hf")

    assert [step for step, _ in find_exports(str(tmp_path))] == [800]


def test_the_checkpoint_rule_rejects_a_whole_export_which_is_why_this_module_exists(
    tmp_path: Path,
):
    """The premise that ``Checkpointer.dir_is_checkpoint`` is the rule for everything.

    It is the right rule for ``checkpoints/step{N}`` and it answers False for a complete
    ``hf/step{N}``, so a lane applying it to the export prefix would poll for ever and read
    nothing. If this test ever fails, the two rules have converged and ``exports.py`` should
    be deleted rather than kept in step.
    """
    export = whole_export(tmp_path / "step2400")

    assert export_is_complete(str(export))
    assert not Checkpointer.dir_is_checkpoint(str(export))
