"""The packaged OLMES evaluation base config must agree with the tokenizer.

`Tokenizer.from_train_config` compares `config.model.vocab_size` against the
loaded tokenizer's own `vocab_size` for exact equality and raises
`OLMoConfigurationError: vocab size mismatch between config and tokenizer`
otherwise. The config had `vocab_size: 100352` -- the EMBEDDING width, i.e.
the tokenizer's 100,278 tokens padded to a multiple of 128 -- so every
task-loss evaluation was guaranteed to fail before scoring anything.

It took a GPU to find, because it is the first thing the smoke reaches after
writing the step-0 checkpoint: job 1723916 died earlier at the loader, 1724623
died earlier at the fresh-state guard, and 1724845 was the first run ever to
reach an evaluation. These assertions make the same failure free.

Stdlib plus yaml only; no torch, no ai2-olmo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

EDULLM = Path(__file__).resolve().parents[1]
if str(EDULLM) not in sys.path:
    sys.path.insert(0, str(EDULLM))

CONFIG = EDULLM / "task_loss" / "ladder_base_config.yaml"

# The dolma2 tokenizer's real token count, confirmed empirically: a config
# carrying this value passes from_train_config against
# `allenai/dolma2-tokenizer`, and 100352 does not.
DOLMA2_VOCAB_SIZE = 100_278
# That padded to a multiple of 128, which is the model's output width.
EMBEDDING_SIZE = 100_352


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_vocab_size_is_the_tokenizer_count_not_the_embedding_width() -> None:
    model = _config()["model"]
    assert model["vocab_size"] == DOLMA2_VOCAB_SIZE
    assert model["embedding_size"] == EMBEDDING_SIZE
    # The two must differ. If a future edit sets them equal again, the
    # tokenizer check fails and no evaluation runs at all.
    assert model["vocab_size"] != model["embedding_size"]


def test_evaluator_defaults_match_the_config() -> None:
    """The evaluator falls back to these when a key is absent, so a default
    that disagrees with the file means the failure depends on which keys
    happen to be present."""
    source = (EDULLM / "task_loss" / "eval_task_loss_olmo_core.py").read_text(
        encoding="utf-8"
    )
    assert 'vocab_size=int(model.get("vocab_size", 100_278))' in source
    assert "EMBEDDING_SIZE = 100_352" in source


def test_sequence_length_is_the_trained_length() -> None:
    """ai2-olmo's ModelConfig defaults max_sequence_length to 1024, which
    silently truncates 5-shot ICL context at half the trained length
    (audit finding 1e)."""
    assert _config()["model"]["max_sequence_length"] == 2048


def test_special_token_ids_are_pinned() -> None:
    model = _config()["model"]
    assert model["eos_token_id"] == 100_257
    assert model["pad_token_id"] == 100_277
    # Both must sit inside the real vocab, or the tokenizer cannot emit them.
    assert model["eos_token_id"] < DOLMA2_VOCAB_SIZE
    assert model["pad_token_id"] < DOLMA2_VOCAB_SIZE


def test_tokenizer_identifier_is_pinned() -> None:
    assert _config()["tokenizer"]["identifier"] == "allenai/dolma2-tokenizer"


def test_model_width_matches_olmo2_370m() -> None:
    """Inert for the measurement -- the model under test is built from
    curriculum_model, not from this file -- but a wrong width here was the
    original audit-1e finding (an OLMoE-1B-7B config left in place), so it is
    pinned rather than left to drift back."""
    model = _config()["model"]
    assert model["d_model"] == 1024
    assert model["n_layers"] == 16
    assert model["n_heads"] == 16
    # FFN hidden size 4096 / d_model 1024 = 4.
    assert model["mlp_ratio"] == 4
