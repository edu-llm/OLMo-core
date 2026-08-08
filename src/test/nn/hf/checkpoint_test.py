from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Olmo2Config

from olmo_core.nn.hf.checkpoint import (
    _fuse_moe_expert_weights,
    load_hf_model,
    save_hf_model,
)
from olmo_core.nn.moe.moe import MoEType
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None


def test_load_hf_model(tmp_path: Path):
    vocab_size = 200
    padded_vocab_size = 256
    model_config = TransformerConfig.olmo2_190M(padded_vocab_size)

    hf_config = Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=3072,
        num_hidden_layers=model_config.n_layers,
        num_attention_heads=12,
        rope_theta=500_000,
        rms_norm_eps=1e-6,
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.save_pretrained(tmp_path / "hf")

    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=padded_vocab_size,
    )
    model.load_state_dict(model_state_dict)

    rand_input = torch.randint(0, vocab_size, (2, 3))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=rand_input)

    assert hf_logits.shape[-1] == vocab_size
    assert logits.shape[-1] == padded_vocab_size
    torch.testing.assert_close(hf_logits, logits[..., :vocab_size])


# ---------------------------------------------------------------------------------------
# MoE: the layout transformers wants, which is not the one the mapping emits
# ---------------------------------------------------------------------------------------

D_MODEL, EXPERT_HIDDEN, NUM_EXPERTS, TOP_K, N_LAYERS = 64, 96, 32, 4, 2


def moe_config(vocab_size: int) -> TransformerConfig:
    """The 32x4 dropless MoE shape this repository trains, shrunk on the free dimensions.

    ``d_model``, ``n_layers`` and the expert width only cost memory. The expert count, the
    top-k and ``dropless`` are what the conversion branches on, so they are the real ones.
    """
    return TransformerConfig.llama_like_moe(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=4,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=EXPERT_HIDDEN,
        dropless=True,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )


def test_fusing_is_a_no_op_when_the_target_wants_one_module_per_expert():
    """Keyed on what the model in front of it asks for, not on a transformers version.

    The layout moved once and can move again, and a model type that never adopted the fused
    parameters has to keep working. Handed a set of expected keys with no fused names in it,
    this must return the state dict it was given.
    """
    per_expert = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(4, 2),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(4, 2),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(2, 4),
    }

    assert _fuse_moe_expert_weights(per_expert, set(per_expert)) is per_expert


def test_the_gate_and_the_up_projection_are_concatenated_in_the_order_hf_splits_them():
    """Mutation: concatenate up before gate, or stack on the wrong axis.

    ``FlexOlmoExperts.forward`` runs one linear over the fused parameter and takes the first
    half of its output as the gate. Reversed, the model loads without complaint, every shape
    matches, and it computes ``silu(up) * gate`` -- a model that is wrong by a permutation
    and reports no error at any point.
    """
    gate = torch.arange(8.0).reshape(4, 2)
    up = torch.arange(8.0).reshape(4, 2) + 100
    down = torch.arange(8.0).reshape(2, 4) + 200
    state = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": gate,
        "model.layers.0.mlp.experts.0.up_proj.weight": up,
        "model.layers.0.mlp.experts.0.down_proj.weight": down,
    }

    fused = _fuse_moe_expert_weights(
        state,
        {"model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"},
    )

    gate_up = fused["model.layers.0.mlp.experts.gate_up_proj"]
    assert gate_up.shape == (1, 8, 2)
    torch.testing.assert_close(gate_up[0][:4], gate)
    torch.testing.assert_close(gate_up[0][4:], up)
    torch.testing.assert_close(fused["model.layers.0.mlp.experts.down_proj"], down.unsqueeze(0))
    # The per-expert keys are gone rather than left beside the fused ones, which would reach
    # ``load_state_dict`` as unexpected keys.
    assert not any(".experts.0." in key for key in fused)


@pytest.mark.skipif(FlexOlmoConfig is None, reason="transformers has no FlexOlmo")
def test_the_exported_expert_weights_compute_what_the_olmo_core_experts_compute(tmp_path: Path):
    """The check no MoE test in this repository was making, and the reason one is needed.

    ``config_test`` checks the generated HF config and ``convert_test`` checks the state
    mapping. Both pass, because they agree with each other; neither asks whether
    ``transformers`` agrees with either, and it does not -- a MoE layer's experts are one 3D
    parameter per projection there and one module per expert here.

    This runs the OLMo-core expert MLP directly rather than the whole model. The model
    forward routes through ``olmo_core.ops.moe``, whose kernels are Triton and therefore
    CUDA-only, so a full logit comparison for a MoE cannot run without a GPU.
    ``DroplessMoEMLP.forward`` falls back to plain matmuls when grouped-gemm is absent, and
    it holds the weights the conversion is about.
    """
    torch.manual_seed(0)
    config = moe_config(vocab_size=128)
    model = config.build()
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.feed_forward_moe is not None
    assert config.block.feed_forward_moe.name == MoEType.dropless

    state = dist_cp_sd.get_model_state_dict(
        model, options=dist_cp_sd.StateDictOptions(cpu_offload=True)
    )
    # The reshape ``convert_checkpoint_to_hf`` applies to a dropless MoE before saving.
    for key, value in list(state.items()):
        if key.endswith(".experts.mlp.w1") or key.endswith(".experts.mlp.w3"):
            state[key] = (
                value.reshape(NUM_EXPERTS, EXPERT_HIDDEN, -1)
                .permute(0, 2, 1)
                .reshape(-1, EXPERT_HIDDEN)
            )

    save_hf_model(tmp_path / "hf", state, model, vocab_size=128)
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")

    tokens_per_expert = 2
    x = torch.randn(NUM_EXPERTS * tokens_per_expert, D_MODEL)
    counts = torch.full((NUM_EXPERTS,), tokens_per_expert, dtype=torch.long)

    for layer in range(N_LAYERS):
        experts = hf_model.model.layers[layer].mlp.experts
        expected = model.blocks[str(layer)].feed_forward_moe.experts.mlp(x, counts)

        got = []
        for expert in range(NUM_EXPERTS):
            chunk = x[expert * tokens_per_expert : (expert + 1) * tokens_per_expert]
            gate, up = F.linear(chunk, experts.gate_up_proj[expert]).chunk(2, dim=-1)
            got.append(F.linear(F.silu(gate) * up, experts.down_proj[expert]))

        torch.testing.assert_close(torch.cat(got), expected, rtol=0, atol=1e-5)


def test_save_hf_model(tmp_path: Path):
    vocab_size = 200
    padded_vocab_size = 256
    model_config = TransformerConfig.olmo2_190M(padded_vocab_size)
    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    save_hf_model(
        tmp_path / "hf",
        model_state_dict,
        model,
        vocab_size=vocab_size,
    )
    model.load_state_dict(model_state_dict)

    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")

    rand_input = torch.randint(0, vocab_size, (2, 3))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=rand_input)

    assert hf_logits.shape[-1] == vocab_size
    assert logits.shape[-1] == padded_vocab_size
    torch.testing.assert_close(hf_logits, logits[..., :vocab_size])
