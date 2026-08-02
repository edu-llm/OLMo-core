"""Co-LMLM modifications for OLMo-core (SmolLM2 backbone).

This package ports the modeling/training changes of **Co-LMLM: Continuous-Query Limited Memory
Language Models** (Feldman et al., 2026, arXiv:2607.07707; code: https://github.com/lil-lab/Co-LMLM)
onto the OLMo-core transformer stack, reusing the SmolLM2 architecture presets added to
:class:`olmo_core.nn.transformer.TransformerConfig` (``smollm2_135M`` / ``smollm2_360M``).

What Co-LMLM changes relative to a standard decoder-only LM (all verified against the released
``lil-lab/CoLMLM-360M-FW`` checkpoint, which is a plain ``LlamaForCausalLM`` with ``vocab_size``
49155 = 49152 + 3):

1. **Three special tokens** (``special_tokens``): ``<FACT>`` (retrieval trigger / query position),
   ``</FACT>`` (closing, appended mechanically), and ``<FACT-q>`` (query marker appended to
   questions). These are the *only* additions to the model — no extra parameters.

2. **The retrieval query is a hidden state, not decoded text.** The query for a fact is the
   final-layer hidden state (post final-norm, i.e. ``lm_head.norm`` output) at the ``<FACT>``
   position, L2-normalized. See ``hidden_states.capture_final_hidden_states``.

3. **A joint training objective** (``loss``): next-token prediction with the fact-span content
   masked out (``M`` = tokens strictly between ``<FACT>`` and ``</FACT>``, *including* the closing
   ``</FACT>`` but *excluding* the opening ``<FACT>``), plus a bidirectional InfoNCE contrastive
   loss aligning each ``<FACT>`` document query with its paired ``<FACT-q>`` question query
   (temperature ``0.07``), combined as ``L = L_NTP + lambda * L_CL`` with ``lambda = 0.25``.

Implemented here (with a CPU self-test in ``colmlm_selftest.py``): the special-token registry,
the Co-LMLM model-config factories, the joint loss, and the hidden-state (query) extraction.
Still to wire up (see the repo notes / TODOs): the annotated-corpus data collator that emits the
``label_mask`` + ``<FACT>``/``<FACT-q>`` position metadata, the train-module that adds the
contrastive term to OLMo-core's LM loss, and the FAISS index build + retrieval-augmented decoding
used only at inference time.
"""

from .loss import (
    DEFAULT_CONTRASTIVE_WEIGHT,
    DEFAULT_TEMPERATURE,
    bidirectional_info_nce,
    colmlm_joint_loss,
    masked_ntp_loss,
)
from .hidden_states import capture_final_hidden_states, gather_query_vectors, l2_normalize
from .model import colmlm_smollm2_135M, colmlm_smollm2_360M
from .special_tokens import (
    FACT_CLOSE,
    FACT_OPEN,
    FACT_QUERY,
    NUM_SPECIAL_TOKENS,
    SMOLLM2_BASE_VOCAB_SIZE,
    SPECIAL_TOKENS,
    SpecialTokenIds,
    colmlm_vocab_size,
    special_token_ids,
)

__all__ = [
    "FACT_OPEN",
    "FACT_CLOSE",
    "FACT_QUERY",
    "SPECIAL_TOKENS",
    "NUM_SPECIAL_TOKENS",
    "SMOLLM2_BASE_VOCAB_SIZE",
    "SpecialTokenIds",
    "special_token_ids",
    "colmlm_vocab_size",
    "colmlm_smollm2_135M",
    "colmlm_smollm2_360M",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_CONTRASTIVE_WEIGHT",
    "bidirectional_info_nce",
    "masked_ntp_loss",
    "colmlm_joint_loss",
    "l2_normalize",
    "gather_query_vectors",
    "capture_final_hidden_states",
]
