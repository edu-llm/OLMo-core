"""Which corpus layout ``colmlm/train.py`` reads, and what it refuses to start without.

``colmlm`` sits at the repository root rather than under ``src``, so the root goes on the path
here. The alternative was to leave the resolution untested, and the thing it resolves is which
bytes a twenty-billion-token run reads.

The two layouts are ``prepare_data.py``'s own -- a root ``manifest.json`` listing one shard per
annotate worker, with masks beside the tokens -- and the one the eduLLM dataset library seals,
which is a group manifest under ``tokens/`` and, for ``fineweb-edu-750m/v2``, no masks at all.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from colmlm import train  # noqa: E402


def native_corpus(root: Path, workers: int = 3, masks: Optional[int] = None) -> Path:
    """A corpus in the shape ``prepare_data.py`` writes, masks included unless asked otherwise."""
    (root / "tokens").mkdir(parents=True, exist_ok=True)
    for worker in range(workers):
        (root / "tokens" / f"train-{worker:05d}.bin").write_bytes(b"\x00\x00")
    for worker in range(workers if masks is None else masks):
        (root / "masks").mkdir(parents=True, exist_ok=True)
        (root / "masks" / f"train-{worker:05d}.mask.bin").write_bytes(b"\x01")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": "HuggingFaceTB/SmolLM2-135M",
                "dtype": "uint16",
                "byte_order": "little",
                "header_bytes": 0,
                "shards": [{"worker": worker, "tokens": 1} for worker in range(workers)],
                "total_tokens": 1_000_000,
            }
        )
    )
    return root


def entry(name: str, split: str, dtype: str = "uint16") -> dict:
    return {
        "bytes": 2,
        "count": {"unit": "tokens", "value": 1},
        "format": {
            "byte_order": "little",
            "codec": "none",
            "container": "raw",
            "dtype": dtype,
            "header_bytes": 0,
        },
        "path": f"tokens/{name}",
        "sha256": "0" * 64,
        "split": split,
    }


def edullm_data_corpus(root: Path, entries: Optional[List[dict]] = None) -> Path:
    """A corpus in the shape the dataset library seals: group manifest, sealed paths, no masks.

    The default is ``fineweb-edu-750m/v2`` in miniature -- train shards plus one held-out val
    shard, all ``.u16le.bin``, and nothing under ``masks/``.
    """
    if entries is None:
        entries = [entry(f"train-{n:05d}.u16le.bin", "train") for n in range(3)]
        entries.append(entry("val-00000.u16le.bin", "val"))
    (root / "tokens").mkdir(parents=True, exist_ok=True)
    for item in entries:
        (root / item["path"]).write_bytes(b"\x00\x00")
    (root / "tokens" / "manifest.json").write_text(
        json.dumps(
            {
                "bytes": 2 * len(entries),
                "entries": entries,
                "group": "tokens",
                "objects": len(entries),
                "schema_version": "edullm-dataset/v2",
            }
        )
    )
    return root


def masks_beside(root: Path, names: List[str]) -> Path:
    directory = root / "held-out-masks"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\x01")
    return directory


def config_for(*args: str) -> train.ExperimentConfig:
    opts, overrides = train.build_parser().parse_known_args(list(args))
    opts.overrides = overrides
    return train.build_config(opts)


def test_his_own_layout_resolves_to_the_paths_it_always_did(tmp_path):
    """The hard requirement. Nothing about the promoted corpus may move his local workflow.

    These are the exact strings ``_shard_paths`` built from the worker list, so a run resumed
    against a checkpoint written before this change reads the same shards in the same order.
    """
    corpus = train.read_corpus(str(native_corpus(tmp_path)))

    assert corpus.token_paths == [
        f"{tmp_path}/tokens/train-00000.bin",
        f"{tmp_path}/tokens/train-00001.bin",
        f"{tmp_path}/tokens/train-00002.bin",
    ]
    assert corpus.dtype == "uint16"
    assert corpus.total_tokens == 1_000_000
    assert corpus.manifest == train.NATIVE_MANIFEST


def test_a_promoted_corpus_resolves_to_the_paths_the_manifest_sealed(tmp_path):
    """Taken from ``entries[].path`` rather than rebuilt from an index.

    The published name carries the width -- ``train-00000.u16le.bin`` -- so a filename
    reconstruction would need to know the dtype before reading the manifest that declares it,
    and would break on the next corpus that spells its shards differently.
    """
    corpus = train.read_corpus(str(edullm_data_corpus(tmp_path)))

    assert corpus.token_paths == [
        f"{tmp_path}/tokens/train-00000.u16le.bin",
        f"{tmp_path}/tokens/train-00001.u16le.bin",
        f"{tmp_path}/tokens/train-00002.u16le.bin",
    ]
    assert corpus.dtype == "uint16"
    assert corpus.total_tokens == 3
    assert corpus.manifest == train.EDULLM_DATA_MANIFEST


def test_the_held_out_shard_is_not_handed_back_as_training_data(tmp_path):
    """Mutation: take every entry, which is what ignoring ``split`` amounts to.

    His own manifest has no splits, so nothing in the old code had an opinion. A promoted corpus
    declares one, and training on the val shard is not a longer run -- it is a run whose held-out
    loss is measured on text the model was fitted to.
    """
    corpus = train.read_corpus(str(edullm_data_corpus(tmp_path)))

    assert not [path for path in corpus.token_paths if "val" in path]


def test_a_corpus_with_no_trainable_split_is_refused_rather_than_run_on_nothing(tmp_path):
    edullm_data_corpus(tmp_path, entries=[entry("val-00000.u16le.bin", "val")])

    with pytest.raises(SystemExit, match="no 'train' entries"):
        train.read_corpus(str(tmp_path))


def test_a_directory_with_neither_manifest_says_where_it_looked(tmp_path):
    with pytest.raises(SystemExit, match="no corpus manifest"):
        train.read_corpus(str(tmp_path))


def test_pointing_data_dir_at_the_tokens_group_says_so_rather_than_reporting_no_dtype(tmp_path):
    """The mistake that produced the original report, and it does not look like one.

    The group manifest is also called ``manifest.json``, so ``--data-dir .../v2/tokens`` finds a
    file where the root one belongs, reads no ``dtype`` off it and refuses -- naming the width as
    the problem when the problem is one directory. Its entry paths are relative to the dataset
    root, so resolving them from here would put every shard a level too deep as well.
    """
    edullm_data_corpus(tmp_path)

    with pytest.raises(SystemExit, match="is the 'tokens' group"):
        train.read_corpus(str(tmp_path / "tokens"))


def test_the_width_comes_from_whichever_manifest_is_there_and_uint32_still_stops_the_run(tmp_path):
    """The failure that started this: the promoted manifest keeps dtype per entry, not at the root.

    Reading the root for a key that layout does not have reported ``dtype None`` and refused --
    correctly, but for the wrong reason, and it refused ``--mode base`` too. The refusal has to
    survive; a uint32 corpus read as uint16 is the silent halving this check exists for.
    """
    entries = [entry(f"train-{n:05d}.u32le.bin", "train", dtype="uint32") for n in range(2)]
    edullm_data_corpus(tmp_path, entries=entries)

    with pytest.raises(SystemExit, match="expected uint16 corpus, manifest says 'uint32'"):
        config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")


def test_base_reads_a_promoted_corpus_and_passes_no_masks(tmp_path):
    """Both halves of the case that used to die before the first step.

    ``base`` never wanted masks, so the corpus having none is not its problem -- and
    ``label_mask_paths`` staying None is what keeps ``base`` the arm it is meant to be.
    """
    edullm_data_corpus(tmp_path)

    config = config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")

    assert len(config.dataset.paths) == 3
    assert config.dataset.label_mask_paths is None
    assert config.dataset.dtype == "uint16"


def test_rank_coordination_and_checkpoint_writes_are_synchronous(tmp_path):
    """The two async Gloo groups deadlocked after checkpoints on both 8xA100 arms."""
    edullm_data_corpus(tmp_path)

    config = config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")

    assert config.trainer.async_bookkeeping is False
    assert config.trainer.callbacks["checkpointer"].save_async is False


def test_resume_requires_full_state_but_writes_to_the_new_run(tmp_path):
    """A replacement Batch run loads the finalized old prefix and saves under its own run ID."""
    edullm_data_corpus(tmp_path)
    old_checkpoints = tmp_path / "old-run" / "checkpoints"
    new_checkpoints = tmp_path / "new-run" / "checkpoints"

    config = config_for(
        "a-run",
        "--mode=base",
        f"--data-dir={tmp_path}",
        f"--resume-from={old_checkpoints}",
        f"--save-folder={new_checkpoints}",
        "--no-wandb",
    )

    assert config.trainer.load_path == str(old_checkpoints)
    assert config.trainer.save_folder == str(new_checkpoints)
    assert config.trainer.load_strategy == train.LoadStrategy.always
    assert config.trainer.load_trainer_state is True
    assert config.trainer.load_optim_state is True
    assert config.trainer.max_duration.value == 305_176


def test_a_fresh_run_keeps_checkpoint_loading_optional(tmp_path):
    """Ordinary runs and same-run Batch retries keep the existing if-available behavior."""
    edullm_data_corpus(tmp_path)

    config = config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")

    assert config.trainer.load_path is None
    assert config.trainer.load_strategy == train.LoadStrategy.if_available
    assert config.trainer.load_trainer_state is None
    assert config.trainer.load_optim_state is None


def test_split_against_a_corpus_with_no_masks_refuses_instead_of_training_a_second_base(tmp_path):
    """The catastrophic one, and the reason any of this is checked before the trainer is built.

    Nothing downstream notices an absent mask. Every fact token stays in the loss, the split arm
    becomes a second control, and the experiment reports that masking fact spans changed nothing.
    ``fineweb-edu-750m/v2`` publishes no ``masks/`` group, so this is the ordinary way to get here.
    """
    edullm_data_corpus(tmp_path)

    with pytest.raises(SystemExit, match="holds none"):
        config_for("a-run", "--mode=split", f"--data-dir={tmp_path}", "--no-wandb")


def test_split_refuses_a_mask_set_that_is_not_the_size_of_the_corpus(tmp_path):
    """Mutation: check only that each expected mask exists.

    His masks were built over nineteen annotate workers and the promoted corpus was sealed as
    fifteen train shards. Per-file existence passes on that overlap and masks fifteen shards with
    spans computed for other ones, which reads as a split arm that is merely a bit worse.
    """
    edullm_data_corpus(tmp_path)
    directory = masks_beside(tmp_path, [f"train-{n:05d}.mask.bin" for n in range(5)])

    with pytest.raises(SystemExit, match="holds 5 masks and this corpus has 3 token shards"):
        config_for(
            "a-run",
            "--mode=split",
            f"--data-dir={tmp_path}",
            f"--mask-dir={directory}",
            "--no-wandb",
        )


def test_split_refuses_masks_named_for_other_shards(tmp_path):
    # Right count, wrong shards. A mask set renumbered by a rebuild pairs file-for-file with the
    # corpus and masks none of the spans it was computed for.
    edullm_data_corpus(tmp_path)
    directory = masks_beside(tmp_path, [f"train-{n:05d}.mask.bin" for n in (7, 8, 9)])

    with pytest.raises(SystemExit, match="are not under"):
        config_for(
            "a-run",
            "--mode=split",
            f"--data-dir={tmp_path}",
            f"--mask-dir={directory}",
            "--no-wandb",
        )


def test_split_pairs_a_promoted_shard_with_the_mask_named_after_it(tmp_path):
    """``train-00000.u16le.bin`` and ``train-00000.mask.bin`` are the same shard.

    The mask name comes from the token shard's name up to its first suffix, which is what lets one
    rule cover both layouts and what keeps the pairing right when the published shards are
    renamed by the promotion.
    """
    edullm_data_corpus(tmp_path)
    directory = masks_beside(tmp_path, [f"train-{n:05d}.mask.bin" for n in range(3)])

    config = config_for(
        "a-run", "--mode=split", f"--data-dir={tmp_path}", f"--mask-dir={directory}", "--no-wandb"
    )

    assert config.dataset.label_mask_paths == [
        f"{directory}/train-00000.mask.bin",
        f"{directory}/train-00001.mask.bin",
        f"{directory}/train-00002.mask.bin",
    ]
    assert list(config.dataset.paths) == [
        f"{tmp_path}/tokens/train-00000.u16le.bin",
        f"{tmp_path}/tokens/train-00001.u16le.bin",
        f"{tmp_path}/tokens/train-00002.u16le.bin",
    ]


def test_split_on_his_own_layout_still_finds_the_masks_beside_the_tokens(tmp_path):
    """No ``--mask-dir``, so the default has to be the directory his corpus already has."""
    native_corpus(tmp_path)

    config = config_for("a-run", "--mode=split", f"--data-dir={tmp_path}", "--no-wandb")

    assert config.dataset.label_mask_paths == [
        f"{tmp_path}/masks/train-00000.mask.bin",
        f"{tmp_path}/masks/train-00001.mask.bin",
        f"{tmp_path}/masks/train-00002.mask.bin",
    ]
    assert config.mask_dir == f"{tmp_path}/masks"


def test_split_on_his_own_layout_refuses_a_half_written_mask_directory(tmp_path):
    # An interrupted prepare_data.py run leaves fewer masks than shards, which is the same
    # unmasked-training outcome arriving from his own machine rather than from a promotion.
    native_corpus(tmp_path, workers=3, masks=2)

    with pytest.raises(SystemExit, match="holds 2 masks and this corpus has 3 token shards"):
        config_for("a-run", "--mode=split", f"--data-dir={tmp_path}", "--no-wandb")


def test_base_on_his_own_layout_does_not_care_that_the_masks_are_gone(tmp_path):
    # base passed label_mask_paths=None before and passes it now; a missing mask directory is not
    # a reason to refuse a run that was never going to read one.
    native_corpus(tmp_path, workers=3, masks=0)

    config = config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")

    assert config.dataset.label_mask_paths is None
    assert config.mask_dir == ""


def test_a_corpus_that_would_be_read_from_the_wrong_offset_is_refused(tmp_path):
    entries = [entry(f"train-{n:05d}.u16le.bin", "train") for n in range(2)]
    for item in entries:
        item["format"]["header_bytes"] = 128
    edullm_data_corpus(tmp_path, entries=entries)

    with pytest.raises(SystemExit, match="header bytes"):
        config_for("a-run", "--mode=base", f"--data-dir={tmp_path}", "--no-wandb")


def test_a_corpus_whose_shards_do_not_agree_on_a_format_is_refused(tmp_path):
    entries = [
        entry("train-00000.u16le.bin", "train"),
        entry("train-00001.u32le.bin", "train", dtype="uint32"),
    ]
    edullm_data_corpus(tmp_path, entries=entries)

    with pytest.raises(SystemExit, match="different shard"):
        train.read_corpus(str(tmp_path))


def test_epochs_uses_the_token_count_the_promoted_manifest_carries_per_entry(tmp_path):
    # total_tokens is a root key of his manifest and a sum over entries of the other one. Reading
    # the root key off a promoted corpus was a KeyError rather than a message.
    entries = [entry(f"train-{n:05d}.u16le.bin", "train") for n in range(2)]
    for item in entries:
        item["count"] = {"unit": "tokens", "value": 65_536}
    edullm_data_corpus(tmp_path, entries=entries)

    config = config_for(
        "a-run",
        "--mode=base",
        f"--data-dir={tmp_path}",
        "--epochs=3",
        "--global-batch-size=65536",
        "--no-wandb",
    )

    assert config.trainer.max_duration.value == 6


def masked_corpus(root: Path, tokens: List[int]) -> Path:
    """One promoted-layout shard, small enough to read back whole, and a place for its mask."""
    edullm_data_corpus(root, entries=[entry("train-00000.u16le.bin", "train")])
    (root / "tokens" / "train-00000.u16le.bin").write_bytes(
        np.asarray(tokens, dtype="<u2").tobytes()
    )
    return masks_beside(root, [])


def test_the_mask_reaches_the_dataset_and_excludes_the_spans_it_names(tmp_path):
    """The whole chain, on shards small enough to check by hand.

    Manifest -> shard paths -> ``NumpyFSLDataset`` -> the ``label_mask`` the trainer turns into
    ``-100``. Resolving the right paths is worth nothing on its own; what makes ``split`` a
    different arm from ``base`` is that False positions arrive at the loss, and that is what this
    reads back rather than assumes.
    """
    tokens = list(range(8))
    keep = [True, True, False, False, True, True, True, True]
    directory = masked_corpus(tmp_path, tokens)
    (directory / "train-00000.mask.bin").write_bytes(np.asarray(keep, dtype=np.bool_).tobytes())

    config = config_for(
        "a-run",
        "--mode=split",
        f"--data-dir={tmp_path}",
        f"--mask-dir={directory}",
        "--sequence-length=8",
        f"--work-dir={tmp_path / 'work'}",
        "--no-wandb",
    )
    dataset = config.dataset.build()
    dataset.prepare()
    item = dataset[0]

    assert item["input_ids"].tolist() == tokens
    assert item["label_mask"].tolist() == keep


def test_a_mask_that_is_not_the_length_of_its_shard_stops_the_run(tmp_path):
    """The library's own pairing check, reached because the paths resolve far enough to hit it.

    A mask one token short of its shard would slide every span by one from that point on. This is
    ``NumpyFSLDataset``'s refusal rather than one of ours, and it is here to record that the
    resolution above still delivers the pair it needs to compare.
    """
    directory = masked_corpus(tmp_path, list(range(8)))
    (directory / "train-00000.mask.bin").write_bytes(
        np.asarray([True] * 7, dtype=np.bool_).tobytes()
    )

    config = config_for(
        "a-run",
        "--mode=split",
        f"--data-dir={tmp_path}",
        f"--mask-dir={directory}",
        "--sequence-length=8",
        f"--work-dir={tmp_path / 'work'}",
        "--no-wandb",
    )

    with pytest.raises(RuntimeError, match="mismatch between size of source file"):
        config.dataset.build().prepare()
