"""Prove the OLMo-core port is Qwen2.5-0.5B before spending GPU hours on it.

An architecture port that is 99% right trains fine and produces a plausible number. The
only way to know is to run both implementations on the same input and compare logits.

Three levels:

  structural   parameter set matches exactly — no spurious attention output bias
               (see the module docstring), tying present or absent as configured
  conversion   every HF tensor is consumed, every OLMo-core parameter is filled
  numerical    max |logit difference| vs HF on fixed inputs, in float32

Needs the model downloaded; skips cleanly without network or transformers.

    pytest -v src/test/nn/transformer/qwen_test.py
"""

from __future__ import annotations

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.transformer import (
    QWEN2_0_5B_HF_REVISION,
    QWEN2_0_5B_HF_WEIGHTS_FILE,
    QWEN2_0_5B_HF_WEIGHTS_SHA256,
    QWEN2_0_5B_HF_WEIGHTS_SIZE,
)
from olmo_core.nn.transformer import qwen as qwen_module
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.nn.transformer.qwen import (
    HF_NUM_LAYERS,
    HF_TENSORS_PER_LAYER,
    QWEN2_0_5B_HF_ID,
    build_qwen2_0_5b,
    convert_hf_state_dict,
    export_to_hf_state_dict,
    hf_to_olmo_key_map,
    load_hf_weights,
    parameter_report,
    qwen2_0_5b_config,
)

# float32 throughout: this test is about whether the *architecture* matches, and bf16
# rounding would mask a real discrepancy behind numerical noise.
TOLERANCE = 2e-4


@pytest.fixture(scope="module")
def hf_state():
    transformers = pytest.importorskip("transformers")
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            QWEN2_0_5B_HF_ID,
            revision=QWEN2_0_5B_HF_REVISION,
            dtype=torch.float32,
        )
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"cannot fetch {QWEN2_0_5B_HF_ID}: {e}")
    state = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    del hf_model
    return state


@pytest.fixture(scope="module")
def olmo_model():
    return build_qwen2_0_5b(dtype=DType.float32, tie=True)


def test_no_attention_output_bias(olmo_model):
    """OLMo-core gives w_out a bias when bias=True; Qwen2 has none."""
    offenders = [
        name
        for name, block in olmo_model.blocks.items()
        if getattr(block.attention.w_out, "bias", None) is not None
    ]
    assert not offenders, (
        f"{len(offenders)} blocks still carry an attention output bias. These have no "
        f"counterpart in the HF checkpoint and would train as free parameters."
    )


def test_qkv_biases_are_present(olmo_model):
    """The other half of the same issue: q/k/v biases are pretrained and required."""
    for name, block in olmo_model.blocks.items():
        for proj in ("w_q", "w_k", "w_v"):
            assert getattr(block.attention, proj).bias is not None, (
                f"block {name}.{proj} has no bias; Qwen2 q/k/v biases carry pretrained "
                f"values and dropping them corrupts the model"
            )


def test_embeddings_are_tied(olmo_model):
    # Tying comes from TransformerConfig.tie_word_embeddings, so it survives to_empty()
    # and FSDP rather than being re-applied by hand after build.
    assert olmo_model.tie_word_embeddings
    assert olmo_model.lm_head.w_out.weight is olmo_model.embeddings.weight
    rep = parameter_report(olmo_model)
    assert rep.tied
    # 494M tied, ~630M untied. The gap is the 151936x896 matrix.
    assert 4.8e8 < rep.unique_params < 5.1e8, rep


def test_untied_build_is_larger():
    untied = build_qwen2_0_5b(dtype=DType.float32, tie=False)
    rep = parameter_report(untied)
    assert not rep.tied
    assert rep.unique_params > 6.0e8


def test_architecture_matches_published_config():
    cfg = qwen2_0_5b_config(dtype=DType.float32)
    assert cfg.d_model == 896
    assert cfg.n_layers == HF_NUM_LAYERS == 24
    assert cfg.vocab_size == 151936
    block = cfg.block
    assert isinstance(block, TransformerBlockConfig)
    attn = block.sequence_mixer
    assert (attn.n_heads, attn.n_kv_heads) == (14, 2)
    assert attn.rope.theta == 1_000_000
    assert attn.bias is True, "q/k/v biases are pretrained and required"
    # llama_like would compute 2560 here; Qwen uses 4864.
    assert block.feed_forward is not None
    assert block.feed_forward.hidden_size == 4864
    # head_dim is not passed (2.5.0-only field); the default must already be right.
    assert cfg.d_model // attn.n_heads == 64


def test_pretrained_source_is_fully_pinned_and_exported():
    assert QWEN2_0_5B_HF_ID == "Qwen/Qwen2.5-0.5B"
    assert QWEN2_0_5B_HF_REVISION == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert QWEN2_0_5B_HF_WEIGHTS_FILE == "model.safetensors"
    assert QWEN2_0_5B_HF_WEIGHTS_SHA256 == (
        "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
    )
    assert QWEN2_0_5B_HF_WEIGHTS_SIZE == 988_097_824


def test_pinned_weight_loader_prefers_local_snapshot_and_verifies_before_load(
    tmp_path, monkeypatch
):
    weights = tmp_path / "model.safetensors"
    payload = b"local-cache-fixture"
    weights.write_bytes(payload)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(tmp_path)

    def load_file(path, *, device):
        calls.append(("deserialize", path, device))
        return {"fixture": torch.ones(1)}

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("safetensors.torch.load_file", load_file)
    monkeypatch.setattr(qwen_module, "QWEN2_0_5B_HF_WEIGHTS_SIZE", len(payload))
    monkeypatch.setattr(
        qwen_module,
        "QWEN2_0_5B_HF_WEIGHTS_SHA256",
        __import__("hashlib").sha256(payload).hexdigest(),
    )

    state = qwen_module.load_pinned_hf_state_dict()

    assert set(state) == {"fixture"}
    assert calls[0][0] == "snapshot"
    assert calls[0][1] == {
        "repo_id": QWEN2_0_5B_HF_ID,
        "revision": QWEN2_0_5B_HF_REVISION,
        "allow_patterns": [QWEN2_0_5B_HF_WEIGHTS_FILE],
        "local_files_only": True,
    }
    assert calls[1] == ("deserialize", str(weights), "cpu")


def test_pinned_weight_loader_refuses_drift_before_deserialization(tmp_path, monkeypatch):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"drift")
    deserialized = False

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_: str(tmp_path),
    )

    def load_file(*_args, **_kwargs):
        nonlocal deserialized
        deserialized = True
        raise AssertionError("drifted bytes must not be deserialized")

    monkeypatch.setattr("safetensors.torch.load_file", load_file)
    monkeypatch.setattr(qwen_module, "QWEN2_0_5B_HF_WEIGHTS_SIZE", 5)
    monkeypatch.setattr(qwen_module, "QWEN2_0_5B_HF_WEIGHTS_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        qwen_module.load_pinned_hf_state_dict()

    assert not deserialized


def test_key_map_is_complete_and_injective():
    m = hf_to_olmo_key_map()
    # 2 global (embedding, final norm) + 12 per layer.
    assert len(m) == 2 + HF_TENSORS_PER_LAYER * HF_NUM_LAYERS
    assert len(set(m.values())) == len(m), "two HF tensors map to the same destination"


def test_every_hf_tensor_is_consumed(hf_state):
    converted = convert_hf_state_dict(hf_state, tied=True)
    assert "lm_head.w_out.weight" in converted
    assert converted["embeddings.weight"].shape == (151936, 896)
    # An unmapped tensor raises inside convert_state_dict; reaching here means none did.


def test_state_dict_loads_strictly(olmo_model, hf_state):
    converted = convert_hf_state_dict(hf_state, tied=True)
    olmo_model.load_state_dict(converted, strict=True)
    assert olmo_model.lm_head.w_out.weight is olmo_model.embeddings.weight


def test_distributed_state_dict_loads_after_train_module_initialization(monkeypatch):
    """The platform wraps/initializes first, then installs pretrained weights.

    ``set_model_state_dict(full_state_dict=True)`` is the supported bridge from a
    normal HuggingFace state dict into an FSDP2-sharded model. This tiny unsharded
    model exercises the same API without requiring a distributed GPU test.
    """

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w_out = torch.nn.Linear(2, 2, bias=False)

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(2, 2)
            self.lm_head = Head()
            self.lm_head.w_out.weight = self.embeddings.weight

    model = TinyQwen()
    expected = torch.full((2, 2), 0.375)
    monkeypatch.setattr(
        "olmo_core.nn.transformer.qwen.convert_hf_state_dict",
        lambda *_args, **_kwargs: {
            "embeddings.weight": expected.clone(),
            "lm_head.w_out.weight": expected.clone(),
        },
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_: (_ for _ in ()).throw(AssertionError("injected state must stay offline")),
    )

    load_hf_weights(
        model,
        hf_state_dict={"unused": torch.empty(0)},
        distributed_state_dict=True,
    )

    assert torch.equal(model.embeddings.weight, expected)
    assert model.lm_head.w_out.weight is model.embeddings.weight


def test_round_trip_to_hf_and_back(olmo_model, hf_state):
    """Export must be the exact inverse of import, or eval scores a different model."""
    converted = convert_hf_state_dict(hf_state, tied=True)
    back = export_to_hf_state_dict(converted, tied=True)
    for key, original in hf_state.items():
        if key == "lm_head.weight":
            continue  # tied: HF materialises it from the embedding
        assert key in back, f"{key} lost in the round trip"
        assert torch.equal(back[key], original), f"{key} changed in the round trip"


@pytest.mark.slow
def test_logits_match_huggingface(hf_state):
    """The one that actually settles it."""
    from transformers import AutoModelForCausalLM

    olmo = build_qwen2_0_5b(dtype=DType.float32, tie=True)
    olmo.load_state_dict(convert_hf_state_dict(hf_state, tied=True), strict=True)
    olmo.eval()

    hf = AutoModelForCausalLM.from_pretrained(
        QWEN2_0_5B_HF_ID,
        revision=QWEN2_0_5B_HF_REVISION,
        torch_dtype=torch.float32,
    )
    hf.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, 151_000, (2, 64))

    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids).logits
        olmo_logits = olmo(input_ids)
    if isinstance(olmo_logits, (tuple, list)):
        olmo_logits = olmo_logits[0]

    assert olmo_logits.shape == hf_logits.shape, (olmo_logits.shape, hf_logits.shape)
    diff = (olmo_logits - hf_logits).abs().max().item()
    assert diff < TOLERANCE, (
        f"max |logit diff| = {diff:.3g} > {TOLERANCE}. The port is not Qwen2.5-0.5B. "
        f"Check RoPE theta, GQA head grouping, RMSNorm eps, and the SwiGLU "
        f"w1/w2/w3 -> gate/down/up ordering before training on it."
    )
