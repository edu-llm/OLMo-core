"""
Qwen2.5 architecture support.

OLMo-core ships presets for OLMo, Llama, nGPT and MoE variants, but none for Qwen, and
:meth:`TransformerConfig.llama_like` cannot express Qwen2 for two reasons: it derives the
feed-forward hidden size (2560 for ``d_model=896``, where Qwen2.5-0.5B uses 4864) and it
hardcodes ``bias=False`` on attention.

One property of Qwen2 cannot be expressed through
:class:`~olmo_core.nn.transformer.config.TransformerConfig` and needs a fix-up after the model
is built: **attention bias placement**. :class:`~olmo_core.nn.attention.Attention` applies a
single ``bias`` flag to ``w_q``, ``w_k``, ``w_v`` *and* ``w_out``, but Qwen2 has biases on
q/k/v only. Building with ``bias=False`` discards pretrained q/k/v biases; building with
``bias=True`` adds an output-projection bias with no counterpart in the checkpoint, which would
then be trained as a free parameter. :func:`strip_attn_out_bias` removes it, and
:func:`build_qwen2_0_5b` calls it for you.

Weight tying needs no special handling here: Qwen2.5-0.5B sets ``tie_word_embeddings`` (its LM
head *is* its embedding matrix, around 136M of its 494M parameters) and
:class:`~olmo_core.nn.transformer.config.TransformerConfig` already supports that, including
re-tying after ``to_empty()`` and skipping the final ``w_out`` initialization.

Weight conversion lives here rather than in :mod:`olmo_core.nn.hf.convert` because that
converter has no bias mappings at all -- every architecture it currently supports (llama,
gemma3, qwen3, qwen3_5) is bias-free in attention, and Qwen2 is the exception. Teaching the
generic converter about biases is a larger change to shared infrastructure than this model
warrants; the map below is explicit, exhaustive, and round-trip tested instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

from .config import TransformerBlockConfig, TransformerBlockType, TransformerConfig

__all__ = [
    "QWEN2_0_5B_HF_ID",
    "build_qwen2_0_5b",
    "convert_hf_state_dict",
    "export_to_hf_state_dict",
    "hf_to_olmo_key_map",
    "ParameterReport",
    "parameter_report",
    "qwen2_0_5b_config",
    "qwen2_tokenizer_config",
    "strip_attn_out_bias",
]

log = logging.getLogger(__name__)

QWEN2_0_5B_HF_ID = "Qwen/Qwen2.5-0.5B"

#: Architecture constants, from https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json.
#: Kept as module constants so the converter and the parity tests assert against the same
#: numbers rather than repeating literals.
HF_HIDDEN_SIZE = 896
HF_INTERMEDIATE_SIZE = 4864
HF_NUM_LAYERS = 24
HF_NUM_ATTENTION_HEADS = 14
HF_NUM_KV_HEADS = 2
HF_VOCAB_SIZE = 151936
HF_RMS_NORM_EPS = 1e-6
HF_ROPE_THETA = 1_000_000
HF_EOS_TOKEN_ID = 151643
HF_HEAD_DIM = HF_HIDDEN_SIZE // HF_NUM_ATTENTION_HEADS

#: HF tensors per layer: 2 layer norms, q/k/v weight and bias (6), o_proj weight, and
#: gate/down/up (3).
HF_TENSORS_PER_LAYER = 12


def qwen2_tokenizer_config() -> TokenizerConfig:
    """
    Build the tokenizer config for Qwen2.5.

    ``vocab_size`` is the *embedding* width (151936), which exceeds the number of real tokens
    because Qwen pads the matrix. The model dimension is what matters here.

    :returns: A :class:`~olmo_core.data.TokenizerConfig` for the Qwen2.5 tokenizer.
    """
    return TokenizerConfig(
        vocab_size=HF_VOCAB_SIZE,
        eos_token_id=HF_EOS_TOKEN_ID,
        pad_token_id=HF_EOS_TOKEN_ID,
        bos_token_id=None,
        identifier=QWEN2_0_5B_HF_ID,
    )


def qwen2_0_5b_config(
    dtype: DType = DType.bfloat16,
    init_seed: int = 42,
    tie_word_embeddings: bool = True,
) -> TransformerConfig:
    """
    Build the Qwen2.5-0.5B architecture config.

    ``bias=True`` on attention gives q/k/v their pretrained biases. It also gives ``w_out`` a
    bias Qwen2 does not have, because OLMo-core has one flag for all four projections; call
    :func:`strip_attn_out_bias` on the built model to remove it.

    Not built through :meth:`TransformerConfig.llama_like` (which is how the ``qwen3_*``
    factories are written) for two reasons: it derives the feed-forward hidden size, and it
    hardcodes ``bias=False`` on attention, which is exactly the thing Qwen2 needs to differ on.

    :param dtype: Parameter dtype.
    :param init_seed: Seed for random initialization, pinned so both arms of a controlled
        comparison start identically even before pretrained weights are loaded over the top.
    :param tie_word_embeddings: Share the embedding matrix with the LM head, as the released
        model does. Setting this ``False`` yields a valid but larger (~630M) untied model.

    :returns: The :class:`~olmo_core.nn.transformer.config.TransformerConfig`.
    """
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms, eps=HF_RMS_NORM_EPS, bias=False, dtype=dtype
    )
    attention = AttentionConfig(
        name=AttentionType.default,
        n_heads=HF_NUM_ATTENTION_HEADS,
        n_kv_heads=HF_NUM_KV_HEADS,
        head_dim=HF_HEAD_DIM,
        bias=True,
        rope=RoPEConfig(name=RoPEType.default, theta=HF_ROPE_THETA),
        qk_norm=None,  # Qwen2 has no QK-norm; Qwen3 does.
        dtype=dtype,
    )
    block = TransformerBlockConfig(
        name=TransformerBlockType.default,  # pre-norm, matching Qwen2
        sequence_mixer=attention,
        feed_forward=FeedForwardConfig(
            # Explicit: llama_like would compute 2560 from d_model.
            hidden_size=HF_INTERMEDIATE_SIZE,
            bias=False,
            dtype=dtype,
        ),
        layer_norm=layer_norm,
    )
    return TransformerConfig(
        d_model=HF_HIDDEN_SIZE,
        vocab_size=HF_VOCAB_SIZE,
        n_layers=HF_NUM_LAYERS,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        dtype=dtype,
        init_seed=init_seed,
        tie_word_embeddings=tie_word_embeddings,
    )


def strip_attn_out_bias(model: nn.Module, expected_layers: int = HF_NUM_LAYERS) -> int:
    """
    Remove the attention output-projection bias that Qwen2 does not have.

    Assigning ``None`` over a registered parameter routes through ``nn.Module.__setattr__`` to
    ``register_parameter(name, None)``, which drops it from the module and from the state dict,
    leaving exactly Qwen2's parameter set. Must run before the model is sharded or compiled.

    :param model: A built :class:`~olmo_core.nn.transformer.model.Transformer`.
    :param expected_layers: How many biases must be removed. Defaults to Qwen2.5-0.5B's layer
        count; override it for a scaled-down model of the same shape.

    :returns: The number of biases removed.

    :raises RuntimeError: If the number removed does not match ``expected_layers``, which means
        the block or attention layout has changed.
    """
    removed = 0
    for block in model.blocks.values():
        attn = getattr(block, "attention", None)
        w_out = getattr(attn, "w_out", None) if attn is not None else None
        if w_out is not None and getattr(w_out, "bias", None) is not None:
            w_out.bias = None
            removed += 1

    if removed != expected_layers:
        raise RuntimeError(
            f"expected to strip {expected_layers} attention output biases, stripped {removed}; "
            f"the block or attention layout has changed"
        )
    log.info("stripped %d attention output-projection biases (Qwen2 has none)", removed)
    return removed


def build_qwen2_0_5b(
    *,
    dtype: DType = DType.bfloat16,
    init_device: str = "cpu",
    tie: bool = True,
    init_seed: int = 42,
) -> nn.Module:
    """
    Build Qwen2.5-0.5B, with the output-projection bias stripped.

    The strip must happen before the model is moved, sharded, or compiled. Weight tying is
    handled by the config, so it survives ``to_empty()`` and FSDP.

    :param dtype: Parameter dtype.
    :param init_device: Device to initialize parameters on.
    :param tie: Whether to tie the LM head to the embedding, as the released model does.
    :param init_seed: Seed for random initialization.

    :returns: The built model.
    """
    config = qwen2_0_5b_config(dtype=dtype, init_seed=init_seed, tie_word_embeddings=tie)
    model = config.build(init_device=init_device)
    strip_attn_out_bias(model)
    return model


@dataclass
class ParameterReport:
    """
    Parameter counts for a built model.

    :param unique_params: Parameters counted once even when shared, so a tied model reports
        its true size (~494M for Qwen2.5-0.5B rather than ~630M).
    :param total_params: The naive sum over ``model.parameters()``.
    :param tied: Whether the LM head shares the embedding matrix.
    """

    unique_params: int
    total_params: int
    tied: bool


def parameter_report(model: nn.Module) -> ParameterReport:
    """
    Summarize a built model's parameter counts.

    ``tied`` distinguishes the 494M released model from the 630M untied variant, which is the
    difference most worth noticing before a run rather than after.

    :param model: A built model.

    :returns: The counts as a :class:`ParameterReport`.
    """
    seen = set()
    unique = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            unique += p.numel()
    return ParameterReport(
        unique_params=unique,
        total_params=sum(p.numel() for p in model.parameters()),
        tied=model.lm_head.w_out.weight is model.embeddings.weight,
    )


def hf_to_olmo_key_map(n_layers: int = HF_NUM_LAYERS) -> Dict[str, str]:
    """
    Map HuggingFace Qwen2 tensor names to OLMo-core parameter names.

    OLMo-core's feed-forward computes ``w2(act(w1(x)) * w3(x))``, so ``w1`` is ``gate_proj``,
    ``w2`` is ``down_proj`` and ``w3`` is ``up_proj``.

    :param n_layers: Number of transformer layers.

    :returns: A dict mapping HF names to OLMo-core names.
    """
    mapping: Dict[str, str] = {
        "model.embed_tokens.weight": "embeddings.weight",
        "model.norm.weight": "lm_head.norm.weight",
    }
    for i in range(n_layers):
        hf, oc = f"model.layers.{i}", f"blocks.{i}"
        mapping.update(
            {
                f"{hf}.input_layernorm.weight": f"{oc}.attention_norm.weight",
                f"{hf}.post_attention_layernorm.weight": f"{oc}.feed_forward_norm.weight",
                f"{hf}.self_attn.q_proj.weight": f"{oc}.attention.w_q.weight",
                f"{hf}.self_attn.q_proj.bias": f"{oc}.attention.w_q.bias",
                f"{hf}.self_attn.k_proj.weight": f"{oc}.attention.w_k.weight",
                f"{hf}.self_attn.k_proj.bias": f"{oc}.attention.w_k.bias",
                f"{hf}.self_attn.v_proj.weight": f"{oc}.attention.w_v.weight",
                f"{hf}.self_attn.v_proj.bias": f"{oc}.attention.w_v.bias",
                f"{hf}.self_attn.o_proj.weight": f"{oc}.attention.w_out.weight",
                f"{hf}.mlp.gate_proj.weight": f"{oc}.feed_forward.w1.weight",
                f"{hf}.mlp.down_proj.weight": f"{oc}.feed_forward.w2.weight",
                f"{hf}.mlp.up_proj.weight": f"{oc}.feed_forward.w3.weight",
            }
        )
    return mapping


def convert_hf_state_dict(
    hf_state_dict: Dict[str, torch.Tensor],
    *,
    tied: bool,
    n_layers: int = HF_NUM_LAYERS,
) -> Dict[str, torch.Tensor]:
    """
    Remap a HuggingFace Qwen2 state dict onto OLMo-core parameter names.

    Exhaustive by construction: an unmapped source tensor or an unfilled destination raises,
    so a ``transformers`` release that renames something fails loudly rather than leaving a
    layer at random initialization.

    :param hf_state_dict: The HuggingFace state dict.
    :param tied: Whether the target model has a tied LM head.
    :param n_layers: Number of transformer layers.

    :returns: A state dict keyed by OLMo-core parameter names.

    :raises KeyError: If a source tensor is unmapped or a destination parameter has no source.
    :raises ValueError: If the checkpoint is tied but its LM head differs from its embedding.
    """
    key_map = hf_to_olmo_key_map(n_layers)

    source = dict(hf_state_dict)
    hf_lm_head = source.pop("lm_head.weight", None)
    for key in [k for k in source if k.endswith(".rotary_emb.inv_freq")]:
        source.pop(key)  # buffer, recomputed by OLMo-core

    out: Dict[str, torch.Tensor] = {}
    for hf_key, tensor in source.items():
        dest = key_map.get(hf_key)
        if dest is None:
            raise KeyError(
                f"unmapped HF tensor {hf_key!r}; update hf_to_olmo_key_map() rather than "
                f"dropping it silently"
            )
        out[dest] = tensor

    missing = set(key_map.values()) - set(out)
    if missing:
        raise KeyError(f"{len(missing)} OLMo-core params had no HF source: {sorted(missing)[:5]}")

    embed = out["embeddings.weight"]
    if hf_lm_head is not None:
        if tied and not torch.equal(hf_lm_head, embed):
            raise ValueError(
                "checkpoint declares tied embeddings but lm_head.weight differs from "
                "embed_tokens.weight; refusing to guess which is authoritative"
            )
        out["lm_head.w_out.weight"] = hf_lm_head
    else:
        out["lm_head.w_out.weight"] = embed.clone()
    return out


def export_to_hf_state_dict(
    olmo_state_dict: Dict[str, torch.Tensor],
    *,
    tied: bool,
    n_layers: int = HF_NUM_LAYERS,
) -> Dict[str, torch.Tensor]:
    """
    Map an OLMo-core state dict back to HuggingFace Qwen2 names.

    The inverse of :func:`convert_hf_state_dict`, for evaluating a trained checkpoint through
    ``transformers`` (KV cache, batched generation) or exporting it.

    :param olmo_state_dict: State dict keyed by OLMo-core parameter names.
    :param tied: Whether the model has a tied LM head, in which case ``lm_head.weight`` is
        omitted so HuggingFace materializes it from the embedding.
    :param n_layers: Number of transformer layers.

    :returns: A state dict keyed by HuggingFace tensor names.

    :raises KeyError: If a parameter has no HuggingFace counterpart, which for
        ``blocks.N.attention.w_out.bias`` means :func:`strip_attn_out_bias` was never applied.
    """
    reverse = {v: k for k, v in hf_to_olmo_key_map(n_layers).items()}
    out: Dict[str, torch.Tensor] = {}
    for olmo_key, tensor in olmo_state_dict.items():
        if olmo_key == "lm_head.w_out.weight":
            if not tied:
                out["lm_head.weight"] = tensor
            continue
        hf_key = reverse.get(olmo_key)
        if hf_key is None:
            raise KeyError(
                f"unmapped OLMo-core parameter {olmo_key!r}; if this is "
                f"'blocks.N.attention.w_out.bias' then strip_attn_out_bias() was not applied "
                f"and the model is not Qwen2"
            )
        out[hf_key] = tensor
    return out


def load_hf_weights(
    model: nn.Module,
    *,
    hf_id: str = QWEN2_0_5B_HF_ID,
    hf_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    distributed_state_dict: bool = False,
) -> None:
    """
    Load pretrained Qwen2.5 weights into a built OLMo-core model, in place.

    Loads strictly, so a model that never had :func:`strip_attn_out_bias` applied fails here on
    the spurious output biases. That is the intended failure: it means the model is not Qwen2.

    :param model: A model from :func:`build_qwen2_0_5b`.
    :param hf_id: HuggingFace model id, used when ``hf_state_dict`` is not supplied.
    :param hf_state_dict: An already-loaded HuggingFace state dict.
    :param distributed_state_dict: Load a normal full state dict into a model that
        may already be FSDP2-sharded. This must be used after train-module
        construction, since that lifecycle initializes model weights.

    :raises RuntimeError: If the state dict does not match the model exactly.
    """
    if hf_state_dict is None:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
        hf_state_dict = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
        del hf_model

    tied = model.lm_head.w_out.weight is model.embeddings.weight
    converted = convert_hf_state_dict(hf_state_dict, tied=tied)
    target_dtype = model.embeddings.weight.dtype
    converted = {k: v.to(dtype=target_dtype) for k, v in converted.items()}

    if distributed_state_dict:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            model,
            converted,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
    else:
        model.load_state_dict(converted, strict=True)
    if tied and model.lm_head.w_out.weight is not model.embeddings.weight:
        raise RuntimeError("loading broke the embedding tie")
    log.info("loaded %s weights: %s", hf_id, parameter_report(model))
