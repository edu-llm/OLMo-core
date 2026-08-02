"""Model + tokenizer loading shared across implementations.

Defaults target ``allenai/OLMo-2-0425-1B-Instruct`` (the PRD's base model / KL
reference pi_0 — the instruction-tuned checkpoint, NOT the pretrained base). To
post-train your own model instead, pass ``--base_model <path-or-hub-id>`` on any
entrypoint; if it lacks a chat template we borrow OLMo-2's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Shorthands accepted by the ``--start_from`` flag.
MODELS = {
    "base": "allenai/OLMo-2-0425-1B",             # pretrained (NOT the KL reference)
    "instruct": "allenai/OLMo-2-0425-1B-Instruct",  # PRD base model / KL reference pi_0
}
# Where to borrow a chat template from when the base model doesn't ship one.
TEMPLATE_SRC = "allenai/OLMo-2-0425-1B-Instruct"

# LoRA target modules for OLMo-2 (attention + MLP projections), PRD §2.6.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: list(LORA_TARGET_MODULES))


def resolve_dtype():
    """(dtype, bf16, fp16). bf16 on Ampere+ (A100/L40S/H100/B200); fp16 on T4; fp32 CPU."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    return dtype, bf16, fp16


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.chat_template is None:  # Instruct ships one; pretrained base does not.
        tok.chat_template = AutoTokenizer.from_pretrained(TEMPLATE_SRC).chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, *, dtype=None, for_training: bool):
    """Load a causal LM. ``for_training`` toggles use_cache (off for grad-ckpt training)."""
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype, _, _ = resolve_dtype()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = not for_training
    return model


def wrap_lora(model, lora: LoraSettings):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=lora.target_modules,
    )
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()  # needed for grad checkpointing + PEFT
    model.print_trainable_parameters()
    return model


def load_for_training(model_id: str, *, use_lora: bool, lora: LoraSettings | None = None):
    """Load model + tokenizer ready for SFT (masking done in ``common.chat``)."""
    dtype, bf16, fp16 = resolve_dtype()
    print(f"model={model_id} | bf16={bf16} fp16={fp16} | lora={use_lora}")
    tok = load_tokenizer(model_id)
    model = load_model(model_id, dtype=dtype, for_training=True)
    if use_lora:
        model = wrap_lora(model, lora or LoraSettings())
    return model, tok, bf16, fp16


def load_tokenizer_for(base_model: str, adapter_dir: str | None):
    """Tokenizer for an (optionally LoRA-adapted) model.

    Intermediate log-spaced checkpoints hold only adapter weights — no tokenizer files — so loading
    from the checkpoint dir dies with "Couldn't instantiate the backend tokenizer ... sentencepiece
    or tiktoken". LoRA never changes the tokenizer, so fall back to the base model's. A full
    fine-tune dir that *does* ship a tokenizer still takes precedence.
    """
    if adapter_dir:
        try:
            return load_tokenizer(adapter_dir)
        except Exception:
            pass
    return load_tokenizer(base_model)


def load_for_inference(base_model: str, adapter_dir: str | None = None, *, merge: bool = False):
    """Load a (LoRA-adapted or plain) model for generation / KL measurement."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype, _, _ = resolve_dtype()
    tok = load_tokenizer_for(base_model, adapter_dir)
    model = load_model(base_model, dtype=dtype, for_training=False)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge:
            model = model.merge_and_unload()
    return model.to(device).eval(), tok, device
