"""Ladder 370M hyperparameters and frontload-cl token budgets.

See ``EXPERIMENT-early-behavior-primer.md`` and ``DATASET-DESIGN-frontload-cl.md``.
"""

from __future__ import annotations

# Shared subsample / mix seed from the experiment design.
DATA_SEED = 42069666

# AI2 model-ladder 370M (Bhagia et al. / OLMo-ladder Table 1).
SEQ_LENGTH = 4096
GLOBAL_BATCH_SEQUENCES = 192
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SEQUENCES * SEQ_LENGTH  # 786_432 tokens
PEAK_LR = 7.8e-4
WARMUP_STEPS = 472
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
MODEL_FACTORY = "olmo2_370M"

# Token budgets (after Dolma2 BPE).
HQ_FINEWEB_MAIN = 8_360_000_000
HQ_FINEWEB_ANNEAL = 950_000_000
HQ_FINEWIKI_TOTAL = 490_000_000
HQ_FINEWIKI_ANNEAL = 50_000_000
HQ_FINEWIKI_MAIN = HQ_FINEWIKI_TOTAL - HQ_FINEWIKI_ANNEAL  # 440M
HQ_PRE_ANNEAL = HQ_FINEWEB_MAIN + HQ_FINEWIKI_MAIN  # 8.80B
HQ_ANNEAL = HQ_FINEWEB_ANNEAL + HQ_FINEWIKI_ANNEAL  # 1.00B
HQ_TOTAL = HQ_PRE_ANNEAL + HQ_ANNEAL  # 9.80B

SFT_LIKE_COSMOPEDIA = 80_000_000
SFT_LIKE_FINEMATH = 60_000_000
SFT_LIKE_OPENHERMES = 30_000_000
SFT_LIKE_NATURAL_REASONING = 30_000_000
SFT_LIKE_TOTAL = (
    SFT_LIKE_COSMOPEDIA
    + SFT_LIKE_FINEMATH
    + SFT_LIKE_OPENHERMES
    + SFT_LIKE_NATURAL_REASONING
)  # 200M

PRIMER_BLOCK = 100_000_000
PRIMER_DISPERSED = SFT_LIKE_TOTAL - PRIMER_BLOCK  # 100M

WARMUP_TOKENS = WARMUP_STEPS * GLOBAL_BATCH_SIZE  # ~371.2M
TOTAL_TRAIN_TOKENS = HQ_TOTAL + SFT_LIKE_TOTAL  # 10.0B
TOTAL_STEPS = TOTAL_TRAIN_TOKENS // GLOBAL_BATCH_SIZE  # 12_715

# Hardware defaults for 8×A100 @ seq=4096: fill the per-rank share of the global
# batch (192/8 = 24 sequences) so there is no grad accumulation. Lower via CLI if OOM.
DEFAULT_RANK_MICROBATCH_SIZE = 24 * SEQ_LENGTH  # 98_304 tokens
# Dao FlashAttention-2 (SM80+). Requires flash-attn in the image; see .edullm/Dockerfile.
DEFAULT_ATTN_BACKEND = "flash_2"
# Short GPU path-check: same model/microbatch/attn as a real run, few optimizer steps.
SMOKE_STEPS = 20

# Checkpointing: periodic + curriculum milestones (see milestone_checkpoint_steps).
DEFAULT_SAVE_INTERVAL = 1000
# Skip a milestone if it falls within this many steps of a periodic save.
CHECKPOINT_MILESTONE_PROXIMITY = 100

# Nominal step indices for curriculum boundaries (token budget // global batch).
# Actual phase lengths may differ by a few sequences after seq-length flooring.
STEPS_AFTER_WARMUP = WARMUP_STEPS  # 472
STEPS_AFTER_PRIMER_BLOCK = WARMUP_STEPS + (PRIMER_BLOCK // GLOBAL_BATCH_SIZE)  # ~599
STEPS_AT_ANNEAL_START = (HQ_PRE_ANNEAL + SFT_LIKE_TOTAL) // GLOBAL_BATCH_SIZE  # ~11_444


def milestone_checkpoint_steps(
    arm: str,
    *,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    proximity: int = CHECKPOINT_MILESTONE_PROXIMITY,
    total_steps: int = TOTAL_STEPS,
    milestones: list[int] | None = None,
) -> list[int]:
    """Curriculum milestone steps for ``CheckpointerCallback.fixed_steps``.

    Both arms: after LR warmup, at anneal start.
    Primer only: after the contiguous SFT-like block.

    Milestones within ``proximity`` of a multiple of ``save_interval`` (or of the
    final step, which ``post_train`` already saves) are dropped to avoid a redundant write.

    :param milestones: Optional override list (for tests); default is arm-specific boundaries.
    """
    if milestones is None:
        milestones = [STEPS_AFTER_WARMUP, STEPS_AT_ANNEAL_START]
        if arm == "primer":
            milestones.append(STEPS_AFTER_PRIMER_BLOCK)

    kept: list[int] = []
    for step in sorted(set(milestones)):
        if step <= 0 or step >= total_steps:
            continue
        nearest_periodic = round(step / save_interval) * save_interval
        if nearest_periodic == 0:
            nearest_periodic = save_interval
        if abs(step - nearest_periodic) <= proximity:
            continue
        if abs(step - total_steps) <= proximity:
            continue
        kept.append(step)
    return kept


# Source folder names under tokens/ in pretrain/frontload-cl-10b.
SOURCE_FINEWEB_MAIN = "fineweb-edu-main"
SOURCE_FINEWEB_ANNEAL = "fineweb-edu-anneal"
SOURCE_FINEWIKI = "finewiki"
SOURCE_COSMOPEDIA = "cosmopedia-v2"
SOURCE_FINEMATH = "finemath-4plus"
SOURCE_OPENHERMES_PT = "openhermes-pt"
SOURCE_NATURAL_REASONING = "natural-reasoning"

HQ_MAIN_SOURCES = (SOURCE_FINEWEB_MAIN, SOURCE_FINEWIKI)
HQ_ANNEAL_SOURCES = (SOURCE_FINEWEB_ANNEAL, SOURCE_FINEWIKI)
SFT_LIKE_SOURCES = (
    SOURCE_COSMOPEDIA,
    SOURCE_FINEMATH,
    SOURCE_OPENHERMES_PT,
    SOURCE_NATURAL_REASONING,
)

# Within SFT-like: 40 / 30 / 15 / 15.
SFT_LIKE_RATIOS = {
    SOURCE_COSMOPEDIA: SFT_LIKE_COSMOPEDIA / SFT_LIKE_TOTAL,
    SOURCE_FINEMATH: SFT_LIKE_FINEMATH / SFT_LIKE_TOTAL,
    SOURCE_OPENHERMES_PT: SFT_LIKE_OPENHERMES / SFT_LIKE_TOTAL,
    SOURCE_NATURAL_REASONING: SFT_LIKE_NATURAL_REASONING / SFT_LIKE_TOTAL,
}

# FineWiki is 5% of every HQ-bearing phase.
HQ_FINEWIKI_RATIO = 0.05
HQ_FINEWEB_RATIO = 1.0 - HQ_FINEWIKI_RATIO

DATASET_ID = "pretrain/frontload-cl-10b"
SFT_DATASET_ID = "sft/frontload-cl-chat-sft"
TOKENIZER_ID = "tokenizer/dolma2-bpe"

# Shared post-PT SFT (one epoch on sft/frontload-cl-chat-sft).
# Hparams follow official OLMo-core SFT scripts (Olmo-2-7B-SFT.py), scaled to seq=4096
# to match the ladder 370M pretrain; the experiment doc fixes the mix and "one epoch only".
SFT_SEQ_LENGTH = SEQ_LENGTH  # 4096
SFT_GLOBAL_BATCH_SEQUENCES = 64
SFT_GLOBAL_BATCH_SIZE = SFT_GLOBAL_BATCH_SEQUENCES * SFT_SEQ_LENGTH  # 262_144
SFT_PEAK_LR = 8e-5
SFT_WARMUP_FRACTION = 0.03
SFT_WEIGHT_DECAY = 0.0
SFT_EPOCHS = 1
SFT_SAVE_INTERVAL = 500
SFT_HF_TOKENIZER = "allenai/dolma2-tokenizer"
SFT_HF_TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
# Classic OLMo 2 / Tülu chat markers (not the ChatML template on dolma2-tokenizer alone).
SFT_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\\n' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '\\n' }}"
    "{% elif message['role'] in ['assistant', 'gpt'] %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
)
