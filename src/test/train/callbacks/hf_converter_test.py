"""Tests for HFConverterCallback."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoConfig

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.hf_converter import HFConverterCallback
from olmo_core.train.checkpoint import Checkpointer


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def transformer_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    return TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size(), n_layers=2)


def test_post_train_converts_checkpoint(
    tmp_path: Path,
    transformer_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """End-to-end: the callback converts a saved checkpoint to HF format."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    model = transformer_config.build()
    save_model_and_optim_state(checkpoint_path / "model_and_optim", model)

    # Simulate what ConfigSaverCallback writes to the checkpoint.
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(
            {
                "model": transformer_config.as_config_dict(),
                "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
            },
            f,
        )

    trainer = Mock()
    trainer.train_module.model = model
    checkpointer = CheckpointerCallback()
    checkpointer._latest_checkpoint_path = str(checkpoint_path)
    checkpointer._trainer = trainer
    trainer.callbacks = {"checkpointer": checkpointer}

    output_folder = tmp_path / "hf_output"
    callback = HFConverterCallback(
        enabled=True,
        output_folder=str(output_folder),
        validate=False,
    )
    callback._trainer = trainer
    callback.post_train()

    assert output_folder.exists()
    hf_config = AutoConfig.from_pretrained(output_folder)
    assert hf_config.hidden_size == transformer_config.d_model
    assert hf_config.num_hidden_layers == transformer_config.n_layers


# ---------------------------------------------------------------------------------------
# Converting while the run is still going
# ---------------------------------------------------------------------------------------


def attached(callback: HFConverterCallback, tmp_path: Path, step: int) -> Mock:
    """The callback wired to a trainer that reports ``step`` and saves under ``tmp_path``."""
    trainer = Mock()
    trainer.global_step = step
    trainer.save_folder = str(tmp_path / "checkpoints")
    trainer.checkpointer.checkpoint_dirname = Checkpointer.checkpoint_dirname
    trainer.callbacks = {}
    callback._trainer = trainer
    return trainer


def test_nothing_is_converted_mid_run_unless_an_interval_was_asked_for(tmp_path: Path):
    """The default is what it always was: one conversion, once training has finished.

    A conversion is a collective followed by every rank waiting while rank zero writes the
    model, so a run that never asked for one must not start paying for them.
    """
    callback = HFConverterCallback(output_folder=str(tmp_path / "hf"))
    attached(callback, tmp_path, step=100)
    callback._convert = Mock()  # type: ignore[method-assign]

    callback.post_step()

    callback._convert.assert_not_called()


def test_an_interval_converts_on_its_multiples_and_names_the_step(tmp_path: Path):
    """Mutation: write every conversion to ``output_folder`` itself.

    Each one would overwrite the last, and the reason to convert during a run at all is that
    something downstream reads the intermediate results. One directory per step under a
    single prefix is also what makes them findable by listing.
    """
    converted = []
    callback = HFConverterCallback(output_folder=str(tmp_path / "hf"), convert_interval=50)

    for step in (49, 50, 51, 100):
        attached(callback, tmp_path, step=step)
        callback._convert = lambda path: converted.append(  # type: ignore[method-assign]
            (callback.step, callback._output_path(path))
        )
        callback.post_step()

    assert converted == [
        (50, str(tmp_path / "hf" / "step50")),
        (100, str(tmp_path / "hf" / "step100")),
    ]


def test_the_final_conversion_is_skipped_when_the_last_step_was_already_converted(
    tmp_path: Path,
):
    """``post_step`` and ``post_train`` both see the final step, and one gather is enough.

    Doing it twice costs a second whole-fleet stall to write bytes that are already there,
    at the end of a run where the cost of a mistake is highest.
    """
    callback = HFConverterCallback(output_folder=str(tmp_path / "hf"), convert_interval=50)
    attached(callback, tmp_path, step=100)
    callback._convert = Mock()  # type: ignore[method-assign]

    callback.post_train()
    callback._convert.assert_not_called()

    # A final step that is not a multiple of the interval still gets its conversion.
    attached(callback, tmp_path, step=103)
    checkpointer = CheckpointerCallback()
    checkpointer._latest_checkpoint_path = str(tmp_path / "checkpoints" / "step100")
    callback.trainer.callbacks = {"checkpointer": checkpointer}
    callback.post_train()
    callback._convert.assert_called_once()


def test_a_failed_conversion_can_be_made_not_to_end_the_run(tmp_path: Path, caplog):
    """Mutation: keep re-raising, which is the right default and the wrong one here.

    ``post_step`` runs inside the training loop, so an exception out of it is the run. An
    export depends on things the training does not -- a tokenizer fetched over the network, a
    write to a prefix the checkpoints do not use -- and none of them is worth the run.
    """
    callback = HFConverterCallback(
        output_folder=str(tmp_path / "hf"),
        convert_interval=1,
        raise_on_failure=False,
        experiment_config={"model": {"this": "will not build"}, "dataset": {"tokenizer": {}}},
    )
    trainer = attached(callback, tmp_path, step=1)
    trainer.train_module.model = TransformerConfig.olmo2_190M(256, n_layers=2).build()

    with caplog.at_level(logging.ERROR):
        callback.post_step()

    assert "Failed to convert checkpoint" in caplog.text

    # And with the default it is the run, which is what the flag is opting out of.
    callback.raise_on_failure = True
    with pytest.raises(Exception):
        callback.post_step()


def test_the_config_can_be_handed_over_rather_than_read_back_out_of_the_checkpoint(
    tmp_path: Path,
    transformer_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """A run saving asynchronously has not necessarily written ``config.json`` yet.

    The default is to read it back from the checkpoint directory being converted, which for
    a periodic conversion is a directory the run is in the middle of producing. The process
    already holds the config, so nothing has to wait on the write.
    """
    callback = HFConverterCallback(
        output_folder=str(tmp_path / "hf"),
        convert_interval=1,
        validate=False,
        experiment_config={
            "model": transformer_config.as_config_dict(),
            "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
        },
    )
    trainer = attached(callback, tmp_path, step=1)
    trainer.train_module.model = transformer_config.build()

    # No checkpoint has been written at all, let alone a config.json inside one.
    assert not (tmp_path / "checkpoints").exists()
    callback.post_step()

    hf_config = AutoConfig.from_pretrained(tmp_path / "hf" / "step1")
    assert hf_config.hidden_size == transformer_config.d_model
