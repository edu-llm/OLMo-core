"""
Tests for the post-training technique catalog (``olmo_core.latentcot.techniques``) and for
building a model from a checkpoint's own saved config.

The catalog's job is that choosing a method for a real fine-tune is a flag, not a code change. So
what is worth pinning is that every technique reaches the loss path unchanged, that the five that
were experiment arms still mean exactly what those arms meant, and that a typo cannot silently
train the wrong method.
"""

import json

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.loss import arm_loss
from olmo_core.latentcot.techniques import (
    LATENT_TECHNIQUES,
    TECHNIQUES,
    as_arm,
    describe_techniques,
    get_technique,
)
from olmo_core.latentcot.tokens import PADDED_VOCAB_SIZE, assert_control_tokens_fit
from olmo_core.latentcot.train_driver import (
    build_model_from_config,
    read_model_config,
    train_arm,
)
from olmo_core.nn.transformer import TransformerConfig

from .test_train_driver import _tiny_model


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture
def dataset(tok, tmp_path):
    path = tmp_path / "conversations" / "train-00000.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        for s in range(6):
            ex = generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2))
            f.write(json.dumps(to_sft_record(ex)) + "\n")
    return LatentCotDataset(path, num_continuous_thoughts=2)


# --------------------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------------------


def test_every_technique_has_a_coherent_shape():
    for name, t in TECHNIQUES.items():
        assert t.name == name  # the key and the field cannot drift apart
        assert t.arm_mode in ("explicit_cot", "no_cot", "codi")
        assert t.summary  # a catalog entry with no explanation is not a catalog entry
        if t.vocab_reg == "none":
            assert t.vocab_reg_weight == 0.0
        else:
            assert t.vocab_reg_weight > 0.0, name  # a named regularizer at weight 0 is a lie
            assert t.is_latent, name  # only latent techniques have thoughts to regularize


def test_the_five_arms_are_all_represented_unchanged():
    """The study's arms must survive renaming exactly, or its results stop applying."""
    by_arm = {t.arm: t for t in TECHNIQUES.values() if t.arm}
    assert set(by_arm) == set(ARMS)
    for arm_key, technique in by_arm.items():
        arm = ARMS[arm_key]
        assert technique.arm_mode == arm.arm_mode
        assert technique.vocab_reg == arm.vocab_reg
        assert technique.vocab_reg_weight == arm.vocab_reg_weight
        assert technique.num_continuous_thoughts == arm.num_continuous_thoughts
        assert technique.vocab_reg_entropy_floor == arm.vocab_reg_entropy_floor


def test_the_catalog_is_wider_than_the_arms():
    """R2 and the entropy floor were implemented and wired to no arm; they must be reachable."""
    never_arms = {name for name, t in TECHNIQUES.items() if t.arm is None}
    assert never_arms == {"codi-r1-entropy", "codi-r2"}
    assert TECHNIQUES["codi-r2"].vocab_reg == "R2"
    assert TECHNIQUES["codi-r1-entropy"].vocab_reg_entropy_floor > 0


def test_latent_techniques_are_exactly_the_codi_ones():
    assert set(LATENT_TECHNIQUES) == {n for n, t in TECHNIQUES.items() if t.arm_mode == "codi"}
    assert "no-cot" not in LATENT_TECHNIQUES
    assert "explicit-cot" not in LATENT_TECHNIQUES


def test_only_no_cot_runs_without_reasoning_traces():
    """The data requirement is a property of the technique and worth asserting explicitly."""
    assert TECHNIQUES["no-cot"].needs_cot_data is False
    assert all(t.needs_cot_data for name, t in TECHNIQUES.items() if name != "no-cot")


def test_get_technique_rejects_an_unknown_name_and_lists_the_options():
    with pytest.raises(KeyError, match="codi-r1"):  # the message must name what IS available
        get_technique("codi-r3")


@pytest.mark.parametrize(
    "kwargs,field,expected",
    [
        ({"num_continuous_thoughts": 4}, "num_continuous_thoughts", 4),
        ({"vocab_reg_weight": 0.5}, "vocab_reg_weight", 0.5),
        ({"vocab_reg_entropy_floor": 2.0}, "vocab_reg_entropy_floor", 2.0),
        ({"distill_weight": 0.25}, "distill_weight", 0.25),
    ],
)
def test_get_technique_applies_overrides(kwargs, field, expected):
    assert getattr(get_technique("codi-r1", **kwargs), field) == expected


def test_get_technique_leaves_the_catalog_unmutated():
    """Overrides must return a copy — a mutated catalog would leak across runs in one process."""
    before = TECHNIQUES["codi-r1"]
    get_technique("codi-r1", num_continuous_thoughts=99, vocab_reg_weight=9.0)
    assert TECHNIQUES["codi-r1"] == before


def test_get_technique_rejects_zero_thoughts_for_a_latent_technique():
    with pytest.raises(ValueError, match="K must be >= 1"):
        get_technique("codi", num_continuous_thoughts=0)


def test_describe_techniques_mentions_every_technique():
    described = describe_techniques()
    for name in TECHNIQUES:
        assert name in described


# --------------------------------------------------------------------------------------
# as_arm — the bridge that makes selection a no-op downstream
# --------------------------------------------------------------------------------------


def test_as_arm_round_trips_the_arm_defining_fields():
    arm = as_arm(TECHNIQUES["codi-r1"])
    assert arm.name == "codi-r1"
    assert (arm.arm_mode, arm.vocab_reg, arm.vocab_reg_weight) == ("codi", "R1", 0.01)


@pytest.mark.parametrize("name", sorted(TECHNIQUES))
def test_every_technique_computes_a_finite_loss(dataset, name):
    """The point of the catalog: every entry must actually reach the loss path."""
    technique = get_technique(name, num_continuous_thoughts=2)
    arm = as_arm(technique)
    torch.manual_seed(0)
    loss, metrics = arm_loss(
        _tiny_model(),
        [dataset[i] for i in range(2)],
        mode=arm.arm_mode,
        distill_weight=technique.distill_weight,
        vocab_reg=arm.vocab_reg,
        vocab_reg_weight=arm.vocab_reg_weight,
        vocab_reg_entropy_floor=arm.vocab_reg_entropy_floor,
    )
    assert torch.isfinite(loss), (name, metrics)


@pytest.mark.parametrize("name", ["codi-r2", "codi-r1-entropy"])
def test_the_never_armed_techniques_train(dataset, name):
    """These two were implemented but unreachable; check they really run, not just configure."""
    technique = get_technique(name, num_continuous_thoughts=2)
    torch.manual_seed(0)
    history = train_arm(
        _tiny_model(),
        as_arm(technique),
        dataset,
        steps=2,
        batch_size=2,
        warmup_steps=1,
        log_every=1,
        distill_weight=technique.distill_weight,
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)


# --------------------------------------------------------------------------------------
# Building from a checkpoint's own config — the arbitrary-pretrained-model path
# --------------------------------------------------------------------------------------


def _write_checkpoint_config(directory, model_config: TransformerConfig):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"model": model_config.as_config_dict(), "trainer": {"unrelated": 1}})
    )


def test_read_model_config_finds_the_saved_config(tmp_path):
    cfg = TransformerConfig.llama_like(d_model=64, n_layers=2, n_heads=4, vocab_size=1024)
    _write_checkpoint_config(tmp_path / "step100", cfg)
    assert read_model_config(str(tmp_path / "step100")) == cfg.as_config_dict()


def test_read_model_config_looks_one_level_up(tmp_path):
    """A caller may point at model_and_optim/ rather than the step directory."""
    cfg = TransformerConfig.llama_like(d_model=64, n_layers=2, n_heads=4, vocab_size=1024)
    _write_checkpoint_config(tmp_path / "step100", cfg)
    (tmp_path / "step100" / "model_and_optim").mkdir()
    assert read_model_config(str(tmp_path / "step100" / "model_and_optim")) is not None


def test_read_model_config_finds_the_config_beside_a_pt_file(tmp_path):
    """
    Regression: `--init-checkpoint` may name a `.pt` file directly. Appending "/../config.json"
    to a file path does not resolve (a file is not a directory), so this silently found nothing
    and the run fell back to `--rung`.
    """
    cfg = TransformerConfig.llama_like(d_model=64, n_layers=2, n_heads=4, vocab_size=1024)
    directory = tmp_path / "step100"
    _write_checkpoint_config(directory, cfg)
    (directory / "model.pt").write_bytes(b"not really a checkpoint")
    assert read_model_config(str(directory / "model.pt")) == cfg.as_config_dict()


def test_read_model_config_is_none_when_absent(tmp_path):
    (tmp_path / "bare").mkdir()
    assert read_model_config(str(tmp_path / "bare")) is None


def test_read_model_config_is_none_on_unreadable_json(tmp_path):
    """A corrupt config must degrade to the --rung fallback, not crash the run."""
    directory = tmp_path / "step100"
    directory.mkdir()
    (directory / "config.json").write_text("{not json")
    assert read_model_config(str(directory)) is None


def test_build_model_from_config_reproduces_the_architecture():
    """
    The whole point: a strict load needs the built architecture to match the weights, and this is
    the only description guaranteed to. Checked by state_dict equality of shapes.
    """
    cfg = TransformerConfig.llama_like(d_model=64, n_layers=2, n_heads=4, vocab_size=1024)
    original = cfg.build(init_device="cpu")
    rebuilt = build_model_from_config(cfg.as_config_dict(), device="cpu")
    assert {k: v.shape for k, v in rebuilt.state_dict().items()} == {
        k: v.shape for k, v in original.state_dict().items()
    }
    # And the weights actually load strictly, which is what the run does.
    rebuilt.load_state_dict(original.state_dict(), strict=True)


def test_build_model_from_config_reproduces_an_moe_architecture():
    """Builds on CPU (a forward would need CUDA), which is enough to pin the shapes."""
    cfg = TransformerConfig.llama_like_moe(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=1024,
        num_experts=4,
        top_k=2,
        expert_hidden_size=128,
        capacity_factor=2.0,
    )
    original = cfg.build(init_device="cpu")
    rebuilt = build_model_from_config(cfg.as_config_dict(), device="cpu")
    assert rebuilt.is_moe
    assert type(rebuilt).__name__ == "MoETransformer"
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    assert any("experts" in key for key in rebuilt.state_dict())


# --------------------------------------------------------------------------------------
# The control-token guard
# --------------------------------------------------------------------------------------


def test_control_tokens_fit_a_dolma2_sized_model():
    assert_control_tokens_fit(_tiny_model())  # built at PADDED_VOCAB_SIZE; must not raise


def test_control_tokens_are_rejected_on_too_small_a_vocab():
    """
    A checkpoint on a different tokenizer would otherwise index off the end of the embedding
    mid-training. Fail at load, naming the assumption.
    """
    small = TransformerConfig.llama_like(d_model=64, n_layers=2, n_heads=4, vocab_size=1024).build(
        init_device="cpu"
    )
    with pytest.raises(ValueError, match="control tokens"):
        assert_control_tokens_fit(small)


def test_control_token_guard_accepts_a_larger_vocab():
    """More rows than we need is fine — the control ids are still inside the embedding."""
    big = TransformerConfig.llama_like(
        d_model=64, n_layers=2, n_heads=4, vocab_size=PADDED_VOCAB_SIZE + 128
    ).build(init_device="cpu")
    assert_control_tokens_fit(big)
