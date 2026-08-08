import json
from pathlib import Path

import pytest
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from transformers import Olmo2Config

from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None


def test_get_hf_config():
    vocab_size = 4096
    model_config = TransformerConfig.olmo2_190M(vocab_size)
    model = model_config.build()

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, Olmo2Config)
    assert hf_config.hidden_size == model_config.d_model
    assert hf_config.intermediate_size == 3072
    assert hf_config.num_hidden_layers == model_config.n_layers


def test_get_hf_config_default_block():
    vocab_size = 4096
    model_config = TransformerConfig.llama2_271M(vocab_size)
    model = model_config.build()

    with pytest.raises(NotImplementedError):
        get_hf_config(model)


def test_get_hf_config_moe():
    vocab_size = 4096
    model_config = TransformerConfig.smallmoe(vocab_size)
    model = model_config.build()

    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, FlexOlmoConfig)
    assert hf_config.hidden_size == model_config.d_model
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    assert hf_config.intermediate_size == model_config.block.feed_forward_moe.hidden_size
    assert hf_config.num_hidden_layers == model_config.n_layers


@pytest.mark.parametrize("factory", ["olmo2_190M", "smallmoe"])
def test_the_context_window_this_builds_is_a_placeholder_and_not_a_length(factory):
    """A ``Transformer`` carries no sequence length, so -1 is the honest answer here.

    Recorded as a test rather than left implicit because the value is not inert: it is what
    vLLM reads as the context window, so every caller that writes one of these configs to disk
    owes it a real length from somewhere else. ``convert_checkpoint_to_hf`` takes
    ``max_sequence_length`` for exactly this, and ``.edullm/train_on_corpus.py`` fills that from
    ``--sequence-length``.
    """
    if factory == "smallmoe" and FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")
    model = getattr(TransformerConfig, factory)(4096).build()

    assert get_hf_config(model).max_position_embeddings == -1


def test_the_placeholder_reaches_config_json_when_nothing_replaces_it(tmp_path: Path):
    """THE HALF OF IT THAT MAKES THE PLACEHOLDER DANGEROUS RATHER THAN INTERNAL.

    ``save_hf_model`` does not take a length and does not invent one, so -1 is written out and
    survives on disk until something rewrites the file. Anything assembling an export from this
    function alone -- rather than through ``convert_checkpoint_to_hf``, which rewrites the
    config afterwards -- publishes a model declaring -1 tokens of context.
    """
    model = TransformerConfig.olmo2_190M(256).build()
    state = dist_cp_sd.get_model_state_dict(model)

    save_hf_model(tmp_path / "hf", state, model, vocab_size=128)

    written = json.loads((tmp_path / "hf" / "config.json").read_text())
    assert written["max_position_embeddings"] == -1
