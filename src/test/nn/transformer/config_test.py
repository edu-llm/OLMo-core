import json

import pytest
from cached_path import cached_path

from olmo_core.nn.attention.recurrent import KimiDeltaHouseholderConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

# OLMO3_7B_CHECKPOINT = "https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage1/step0"
OLMO3_7B_CHECKPOINT = "https://storage.googleapis.com/ai2-llm/checkpoints/OLMo25/step0"


def test_load_olmo3_7b_config():
    """Verify that old checkpoint configs with a single block (not a dict) still load correctly."""
    config_path = cached_path(f"{OLMO3_7B_CHECKPOINT}/config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = TransformerConfig.from_dict(config_dict["model"])

    assert config.d_model == 4096
    assert config.n_layers == 32
    assert config.vocab_size == 100352
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.name == "reordered_norm"

    # Round-trip through as_config_dict / from_dict should be lossless.
    roundtripped = TransformerConfig.from_dict(config.as_config_dict())
    assert roundtripped.as_config_dict() == config.as_config_dict()


VOCAB_SIZE = 100352


def test_kda_householder_130M_uses_the_householder_mixer():
    """The factory exists so ``--model-factory`` can reach a mixer no other factory builds."""
    config = TransformerConfig.kda_householder_130M(vocab_size=VOCAB_SIZE)
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, KimiDeltaHouseholderConfig)
    assert config.block.sequence_mixer.num_householder == 2
    # Strict is the default: the reflection regime must be asked for explicitly, because it
    # is the treatment and a silent default would make every arm the same arm.
    assert config.block.sequence_mixer.allow_neg_eigval is False


@pytest.mark.parametrize("reflection", [False, True])
def test_kda_householder_130M_beta_regime_is_reachable_by_dotlist(reflection: bool):
    """The regime must survive ``merge``, because the platform has no other config surface.

    This is the whole reason the factory exists. Overriding a mixer's ``type`` by dotlist is
    a silent no-op --- ``Config.from_dict`` drops ``type`` when ``_CLASS_`` is present
    (``config.py:279``) --- so a run built that way trains softmax attention while reporting
    the arm it was asked for. Selecting the mixer by factory and the regime by scalar
    override is the combination that actually works, and this asserts it still does.
    """
    config = TransformerConfig.kda_householder_130M(vocab_size=VOCAB_SIZE).merge(
        [f"block.sequence_mixer.allow_neg_eigval={str(reflection).lower()}"]
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, KimiDeltaHouseholderConfig)
    assert config.block.sequence_mixer.allow_neg_eigval is reflection


def test_kda_householder_130M_regimes_are_capacity_and_compute_matched():
    """The two arms differ in one boolean and in nothing a comparison must hold fixed.

    ``beta = s * Sigmoid(z)`` with ``s in {1, 2}`` adds no parameter, so the ledgers are
    identical rather than merely close. If this ever fails, the beta contrast has acquired a
    capacity confound and the arms are no longer comparable.
    """
    strict = TransformerConfig.kda_householder_130M(vocab_size=VOCAB_SIZE, allow_neg_eigval=False)
    reflection = TransformerConfig.kda_householder_130M(
        vocab_size=VOCAB_SIZE, allow_neg_eigval=True
    )
    assert strict.num_params == reflection.num_params
    assert strict.num_non_embedding_params == reflection.num_non_embedding_params


def test_kda_householder_130M_round_trips():
    """A config that cannot be serialized cannot be checkpointed or resumed."""
    config = TransformerConfig.kda_householder_130M(vocab_size=VOCAB_SIZE)
    assert TransformerConfig.from_dict(config.as_config_dict()).as_config_dict() == (
        config.as_config_dict()
    )
