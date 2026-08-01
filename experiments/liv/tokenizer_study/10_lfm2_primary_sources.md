# LFM2 Tokenizer — Primary Source Ground Truth

Agent: child 10 of 5-agent tokenizer study. Written incrementally.
Date of investigation: 2026-08-01.
All execution on Stanford FarmShare login node (no GPU, no local Mac execution).
Scratch workspace: `/scratch/users/ericrcwu/liv/tok/`

---

## BOTTOM LINE (lift verbatim)

- **arXiv 2511.23404 RESOLVES.** *"LFM2 Technical Report"*, submitted 28 Nov 2025, 33 authors (Amini, Hasani, Rus, Labonne, Lechner, et al.), CC BY 4.0. **CONFIRMED / MEASURED.** The ID is real — a rare non-future-dated one in this project's notes. The model card's own HF tag `arxiv:2511.23404` independently corroborates it.
- **Tokenizer type: byte-level BPE**, `tokenizers` fast format, `tokenizer_class: PreTrainedTokenizerFast`. `normalizer: null`, **no unk token**, `byte_fallback: false`. Not SentencePiece, not Unigram. Paper §2.2 verbatim: *"We use a byte-level BPE tokenizer (Sennrich et al., 2016) with a 65,536-token vocabulary."* **MEASURED + CONFIRMED against the paper.**
- **65,536 is the padded embedding size, NOT the tokenizer.** The real tokenizer has **64,400 ids (0–64399)**; **1,136 embedding rows are permanently dead** (2^16 alignment pad). Usable **text** vocab = **63,893** (64,400 − 507 special). **Liquid never discloses this anywhere.** **MEASURED — new relative to all published sources.**
- **Embeddings ARE tied** in every LFM2 and LFM2.5 checkpoint — three independent proofs: `Lfm2Config` class default `tie_word_embeddings = True`; `LFM2.5-1.2B-Base` config carries the legacy key `"tie_embedding": true`; and **no `lm_head.weight` tensor exists** in any safetensors header. `LFM2-350M`/`LFM2-1.2B` carry **neither** tie key, so the True default applies. The paper is **completely silent** on tying. **CONFIRMED.**
- **Multilingual: 14,800 / 63,893 = 23.16% of usable vocab is non-ASCII.** Cyrillic 6.89%, CJK 5.24%, Latin-accented 5.16%, Japanese 2.21%, Korean 1.62%, Arabic 1.22%, + 791 partial-UTF-8 fragments. Indic/Hebrew/Thai ≈ absent. **MEASURED.** **Surprise: Russian/Cyrillic is the LARGEST non-English block, yet Russian is named nowhere in the paper, blog, or model card.**
- **Digits use `\p{N}{1,3}` — 3-digit left-aligned chunking, NOT individual-digit splitting.** `84721` → `847|21`; `" 1234"` → `' '|123|4`. **MEASURED, directly decision-relevant for passkey/phonebook probes** — any eval assuming one-token-per-digit will break.
- **License: LFM Open License v1.0** (Apache-2.0 text + a new §5 Commercial Use Limitation + §11 auto-termination). The **$10,000,000 annual-revenue Threshold is CONFIRMED and quoted**. Research/non-profit use expressly carved out (§5(c)); derivative works expressly granted (§2). **Tokenizer files are NOT separately licensed** — same LFM license as the weights, *unlike* HuggingFace's `modeling_lfm2.py` which is Apache-2.0. **CONFIRMED** (the project's code-vs-weights license note was right).
- **NOT gated.** HF API `gated: False`; every file fetched unauthenticated at HTTP 200 including `tokenizer.json`. **MEASURED.**
- **One tokenizer across the whole family.** `LFM2-350M` and `LFM2-1.2B` `tokenizer.json` are byte-identical (md5 `83703d34b17eca9dd512ca67668028fc`). LFM2.5 differs by 594 bytes — **only reserved-slot renames** (audio/vision tokens); the text BPE is unchanged. **MEASURED.**
- **Vocab was NOT part of the architecture search**, and **LFM2 is NOT a STAR product** — the paper calls STAR an *"earlier academic prototype"* whose proxies *"do not transfer reliably."* STAR = **arXiv:2411.17800, VERIFIED via arXiv API** (Thomas, Parnichkun, Amini, Massaroli, Poli; ICLR 2025).
- **Zero tokenizer ablations, zero fertility numbers, zero vocab-sizing rationale** in paper, blog, or card. Measured myself: ~4.5 bytes/token English prose, 3.8 plain English, 2.5 Python, 2.1 LaTeX — unremarkable.
- **THE DECISION-DRIVER:** pure-ASCII text tokens are **LFM2 = 48,302 vs GPT-2 = 49,383** (same measurement pipeline, §6). **LFM2's tokenizer gives you ~1,100 FEWER English-usable tokens than GPT-2**, despite a 30% larger nominal vocabulary. Retokenizing GPT-2 → LFM2 buys **no** English headroom, only a license encumbrance. If you want LFM2's *shape*, train your own 64K English BPE.

---

## SECTION 1 — arXiv 2511.23404 (LFM2 Technical Report)

### 1.0 Does the ID resolve? — YES

**MEASURED**, source `https://arxiv.org/abs/2511.23404` via WebFetch.

- Title: **"LFM2 Technical Report"**
- arXiv:2511.23404 [cs.LG], cross-listed cs.AI
- v1 submitted **Fri, 28 Nov 2025 17:56:35 UTC** (17,237 KB)
- 33 authors: Alexander Amini, Anna Banaszak, Harold Benoit, Arthur Böök, Tarek Dakhran, Song Duong, Alfred Eng, Fernando Fernandes, Marc Härkönen, Anne Harrington, Ramin Hasani, Saniya Karwa, Yuri Khrustalev, Maxime Labonne, Mathias Lechner, Valentine Lechner, Simon Lee, Zetian Li, Noel Loo, Jacob Marks, Edoardo Mosca, Samuel J. Paech, Paul Pak, Rom N. Parnichkun, Alex Quach, Ryan Rogers, Daniela Rus, Nayan Saxena, Bettina Schlager, Tim Seyde, Jimmy T.H. Smith, Aditya Tadimeti, Neehal Tumma
- arXiv page displays a **CC BY 4.0** license icon (for the *paper*, not the weights).

Abstract facts relevant to us (**MEASURED** from abstract):
- Family spans 350M / 700M / 1.2B / 2.6B dense + an MoE at 8.3B total / 1.5B active, all **32K context**.
- Backbone: "gated short convolutions with a small number of grouped query attention blocks".
- Architecture search done "with hardware in the loop under edge latency/memory limits".
- Pre-training: **10–12T tokens**.
- Objective: "tempered, decoupled Top-K knowledge distillation objective that avoids support mismatch" + curriculum learning ordered by difficulty.
- Variants: LFM2-VL, LFM2-Audio, **LFM2-ColBERT (described as a "low-latency multilingual retrieval encoder")** — multilinguality is an explicit product axis.

**The abstract says NOTHING about the tokenizer.** (Full-text extraction is in §1.1 below.)

---

## SECTION 2 — Released configs (exact values)

Source: unauthenticated `curl` from FarmShare login node against
`https://huggingface.co/<repo>/raw/main/<file>` — every request HTTP 200.

### 2.1 Repo existence check (HF API, **MEASURED**)

| repo | HTTP |
|---|---|
| `LiquidAI/LFM2-350M` | 200 |
| `LiquidAI/LFM2-700M` | 200 |
| `LiquidAI/LFM2-1.2B` | 200 |
| `LiquidAI/LFM2-2.6B` | 200 |
| `LiquidAI/LFM2-8B-A1B` | 200 |
| `LiquidAI/LFM2.5-1.2B-Base` | **200 — EXISTS** |
| `LiquidAI/LFM2.5-1.2B` | 401 (does not exist under that exact name; the instruct repo is `LFM2.5-1.2B-Instruct`) |

Other LFM2.5 base repos that exist: `LFM2.5-350M-Base`, `LFM2.5-230M-Base`, plus `LFM2.5-230M`, `LFM2.5-350M`, `LFM2.5-8B-A1B`, `LFM2.5-1.2B-Instruct`, `LFM2.5-1.2B-Thinking`, encoder/embedding/VL/audio lines.

### 2.2 `config.json` — the load-bearing fields (**MEASURED**, verbatim)

**`LiquidAI/LFM2-350M`** (`transformers_version: "4.54.0.dev0"`):
```
"vocab_size": 65536
"hidden_size": 1024, "block_dim": 1024, "block_ff_dim": 6656
"num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 8
"full_attn_idxs": [2, 5, 8, 10, 12, 14]
"bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0
"max_position_embeddings": 128000, "rope_theta": 1000000.0
```
**NO `tie_embedding` key. NO `tie_word_embeddings` key.**

**`LiquidAI/LFM2-1.2B`** (`transformers_version: "4.54.0.dev0"`): identical except
`block_dim`/`hidden_size` 2048, `block_ff_dim` 12288, `num_attention_heads`/`num_heads` 32.
`"vocab_size": 65536`. **Also NO tie key of either spelling.**

**`LiquidAI/LFM2.5-1.2B-Base`** (`transformers_version: "4.57.2"`):
```
"vocab_size": 65536
"tie_embedding": true          <-- EXPLICIT, legacy key as predicted
"layer_types": ["conv","conv","full_attention","conv","conv","full_attention",
                "conv","conv","full_attention","conv","full_attention","conv",
                "full_attention","conv","full_attention","conv"]
"bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0
"intermediate_size": 12288, "hidden_size": 2048, "num_hidden_layers": 16
```
Note LFM2.5 migrated from `full_attn_idxs` to the newer `layer_types` list; the attention
positions changed from `[2,5,8,10,12,14]` to `[2,5,8,10,12,14]` — same indices, just a
different encoding. Also `torch_dtype` → `dtype` rename.

### 2.3 The tie-embedding question — RESOLVED

**MEASURED**, source: `/scratch/users/ericrcwu/kda/venv/lib/python3.12/site-packages/transformers/models/lfm2/configuration_lfm2.py` (transformers **5.14.1** on FarmShare).

The class default and the remap, verbatim from source:

```python
class Lfm2Config(PreTrainedConfig):
    vocab_size: int = 65536
    hidden_size: int = 2560
    ...
    tie_word_embeddings: bool = True        # <-- DEFAULT IS TRUE

    def __post_init__(self, **kwargs):
        ...
        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        self.intermediate_size = kwargs.pop("block_ff_dim", self.intermediate_size)
```

Three confirmations of the parent's hypothesis:
1. The legacy key **`tie_embedding`** is indeed the one remapped in `__post_init__` — **CONFIRMED**.
2. `block_ff_dim` is remapped to `intermediate_size` the same way (bonus finding).
3. **The `Lfm2Config` default is `tie_word_embeddings = True`.** So `LFM2-350M`/`LFM2-1.2B`,
   which carry neither key, are **TIED** by default. **CONFIRMED / MEASURED.**

Also note the class default `vocab_size: int = 65536` is hardcoded into transformers itself —
65,536 is the canonical LFM2 vocab, not a per-checkpoint accident.

### 2.4 Independent evidence of tying via weight index

`https://huggingface.co/LiquidAI/LFM2-350M/raw/main/model.safetensors.index.json` → **HTTP 404,
body "Entry not found"**. Same for LFM2-1.2B. These are single-file checkpoints (no shard index),
so the index-based test is **not available**. See §2.5 for the direct safetensors header check.
**UNCLEAR by this route** — superseded by §2.5.

### 2.5 `special_tokens_map.json` / `generation_config.json` (**MEASURED**)

`special_tokens_map.json` is **byte-identical across LFM2-350M, LFM2-1.2B, and LFM2.5-1.2B-Base**:
```json
{"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>", "pad_token": "<|pad|>"}
```
(each with `normalized:false`, all other flags false)

`generation_config.json` (all three, modulo transformers_version):
```json
{"bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0}
```

**IMPORTANT INCONSISTENCY (MEASURED):** `eos_token_id` is **7** in config/generation_config,
which is `<|im_end|>` — but the *added-token id 2* is `<|endoftext|>`. So LFM2 ships with the
chat-style `<|im_end|>` as the generation EOS even in base repos. A from-scratch pretrain would
want id 2 `<|endoftext|>` as the document separator and should NOT blindly copy `eos_token_id: 7`.

`tokenizer_config.json` is byte-identical between LFM2-350M and LFM2-1.2B
(md5 `cfa180193a7ac629bfe202f5cf40d9af`, 91,565 bytes).

### 2.6 tokenizer.json identity across the family (**MEASURED**)

| file | bytes | md5 |
|---|---|---|
| `LFM2-350M/tokenizer.json` | 4,732,426 | `83703d34b17eca9dd512ca67668028fc` |
| `LFM2-1.2B/tokenizer.json` | 4,732,426 | `83703d34b17eca9dd512ca67668028fc` |
| `LFM2.5-1.2B-Base/tokenizer.json` | 4,733,020 | `910b7828a3926f1447b9a65df373f090` |

LFM2-350M and LFM2-1.2B ship the **exact same tokenizer artifact**. LFM2.5 differs by 594 bytes
— diffed in §3.

### 2.7 DECISIVE tying evidence — safetensors header (**MEASURED**)

I read the safetensors header directly via HTTP range requests (first 8 bytes = u64 LE header
length, then that many bytes of JSON) — no weight download.

| repo | n tensors | `lm_head.weight` present? | `model.embed_tokens.weight` shape |
|---|---|---|---|
| `LFM2-350M` | 148 | **NO** | `[65536, 1024]` |
| `LFM2-1.2B` | 148 | **NO** | `[65536, 2048]` |
| `LFM2.5-1.2B-Base` | 148 | **NO** | `[65536, 2048]` |

No `lm_head`-like or `output`-like key exists in any of the three checkpoints.
**Embeddings are TIED in all LFM2 and LFM2.5 models. CONFIRMED / MEASURED — three independent
lines of evidence (class default, explicit `tie_embedding: true` in LFM2.5, absent `lm_head`).**

Bonus: there is a `model.embedding_norm.weight` tensor — LFM2 applies a norm to the embedding
output (relevant if the parent is cloning the shape).

### 2.8 `tokenizer_config.json` top-level (**MEASURED**, verbatim, LFM2-350M == LFM2-1.2B)

```json
{
  "add_bos_token": true,
  "add_eos_token": false,
  "bos_token": "<|startoftext|>",
  "clean_up_tokenization_spaces": true,
  "eos_token": "<|im_end|>",
  "extra_special_tokens": {},
  "legacy": false,
  "model_input_names": ["input_ids", "attention_mask"],
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "<|pad|>",
  "sp_model_kwargs": {},
  "spaces_between_special_tokens": false,
  "tokenizer_class": "PreTrainedTokenizerFast",
  "use_default_system_prompt": false,
  "use_fast": true
}
```

- `tokenizer_class` = **`PreTrainedTokenizerFast`** (generic fast wrapper — no slow/sentencepiece
  counterpart; there is **no `tokenizer.model`/`spiece.model` file**).
- `model_max_length` = `1e30`-ish sentinel, i.e. **unset**. It does NOT encode the 32K/128K context.
- `sp_model_kwargs: {}` and `legacy: false` are **vestigial Llama-tokenizer-config fields** carried
  over by the conversion script; they do **not** mean SentencePiece is in use (the actual model is
  `BPE`, see §3). **INFERRED** but strongly supported: there is no sentencepiece proto in the repo.
- `add_bos_token: true`, `add_eos_token: false` — BOS `<|startoftext|>` (id 1) is auto-prepended.
  **Verified empirically:** `encode("hello world")` → `[1, 52572, 2031]` = `['<|startoftext|>','hello','Ġworld']`.
- **NO `chat_template`** in the LFM2-350M/1.2B *base* repos.
- `len(added_tokens_decoder)` = **507** in both LFM2 and LFM2.5.

### 2.9 LFM2 vs LFM2.5 tokenizer diff (**MEASURED**)

Only two differences:
1. `clean_up_tokenization_spaces`: `true` (LFM2) → `false` (LFM2.5).
2. LFM2.5 **repurposes reserved slots into real tokens** — the reserved pool is the escape hatch:
   - ids 128–132: `<|reserved_118..122|>` → `<|audio_start|>`, `<|text_start|>`, `<|text_end|>`, `<|mixed_start|>`, `<|mixed_end|>`
   - ids 396+: `<|reserved_386..|>` → `<image>`, `<|img_row_R_col_C|>` grid tokens (10x10 tiling)

The **text BPE vocabulary is unchanged between LFM2 and LFM2.5.** Same 64,400-entry BPE, same
merges. Only reserved-slot naming moved. **MEASURED** (byte diff of the two tokenizer.json is 594
bytes, all in the reserved-token names).

---

## SECTION 3 — `tokenizer.json`, the decisive artifact

Downloaded on FarmShare only: `/scratch/users/ericrcwu/liv/tok/lfm2_tokenizer.json`,
4,732,426 bytes (checked `content-length` first — 4.7 MB, well under the 50 MB limit),
md5 `83703d34b17eca9dd512ca67668028fc`. Parsed with plain `json.load` + `tokenizers` 0.22.x.

### 3.1 Model type — **byte-level BPE** (**MEASURED**)

```
model.type                     = "BPE"
model.dropout                  = None
model.unk_token                = None      <-- NO UNK. Byte-level ⇒ lossless, no OOV.
model.continuing_subword_prefix= None
model.end_of_word_suffix       = None
model.fuse_unk                 = False
model.byte_fallback            = False     <-- not needed; pre-tokenizer is already ByteLevel
model.ignore_merges            = False
```

**NOT Unigram. NOT SentencePiece. NOT WordPiece. It is GPT-2-family byte-level BPE.**

### 3.2 `normalizer` — **`null`** (**MEASURED**)

No normalization at all: no NFKC, no lowercasing, no space-prefix insertion. Byte-exact.

### 3.3 `pre_tokenizer` (**MEASURED**, verbatim)

```json
{"type": "Sequence", "pretokenizers": [
  {"type": "Split",
   "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
   "behavior": "Isolated", "invert": false},
  {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
]}
```

**This is the GPT-4 / `cl100k_base`-style split regex**, not GPT-2's. Diagnostic features:
- `(?i:'s|'t|...)` — case-insensitive contraction handling (GPT-2 uses lowercase-only `'s|'t|...`).
- `\p{L}` / `\p{N}` Unicode property classes (GPT-2 uses `\p{L}`/`\p{N}` too but without the
  `[^\r\n\p{L}\p{N}]?` leading-non-letter guard, and without `\s*[\r\n]+`).
- `ByteLevel` with `add_prefix_space: false` and `use_regex: false` (regex already applied by Split).

### 3.4 DIGIT SPLITTING — **the `\p{N}{1,3}` rule** (**MEASURED**) — DECISION-RELEVANT

There is **no `Digits` pre-tokenizer with `individual_digits`**. Instead the split regex contains
**`\p{N}{1,3}`**, which chunks digit runs into **groups of up to 3, left-to-right**. Empirically
verified by encoding:

| input | tokens | count |
|---|---|---|
| `"1234567890"` | `['123','456','789','0']` | 4 |
| `"The passkey is 84721."` | `['The',' pass','key',' is',' ','847','21','.']` | 8 |
| `"Call 415-555-0132"` | `['Call',' ','415','-','555','-','013','2']` | 8 |
| `"9"` / `"99"` / `"999"` | 1 token each | 1 |
| `"9999"` | `['999','9']` | 2 |
| `"99999"` | `['999','99']` | 2 |
| `" 1234"` | `[' ','123','4']` | 3 |
| `"3.14159265358979"` | `['3','.','141','592','653','589','79']` | 7 |
| `"1000000"` | `['100','000','0']` | 3 |

**Implications for passkey / phonebook endpoints (INFERRED from the MEASURED behavior):**
- Digits are **NOT individually split** (unlike Llama-3 / Gemma / DeepSeek which use
  `individual_digits`). A 5-digit passkey becomes **2 tokens**, not 5.
- Chunking is **left-aligned and length-dependent**: `84721` → `847|21` but `8472` → `847|2`.
  The same digit in different positions maps to different tokens. This makes digit-level
  copy/retrieval tasks **harder to score per-digit** and creates alignment hazards if the parent's
  eval assumes one-token-per-digit.
- **A leading space is NOT merged with the digits**: `" 1234"` → `' '`, `'123'`, `'4'`. The space
  is its own token before a number. This differs from GPT-2 (`" 1234"` → `' 12','34'`).
- If the parent's passkey/phonebook probes were designed against GPT-2 tokenization, **token counts
  and per-digit alignment will change** on retokenization. This must be re-derived, not assumed.

### 3.5 `post_processor` and `decoder` (**MEASURED**)

```json
post_processor = {"type":"Sequence","processors":[
  {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
  {"type":"TemplateProcessing",
   "single":[{"SpecialToken":{"id":"<|startoftext|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
   "pair":  [BOS, A, BOS, B],
   "special_tokens":{"<|startoftext|>":{"id":"<|startoftext|>","ids":[1],"tokens":["<|startoftext|>"]}}}]}

decoder = {"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]}
```

BOS auto-prepended by the template. Byte-level decode. **Note the `pair` template inserts BOS
between segments, not EOS** — a from-scratch pretrain would want its own document-separator scheme.

### 3.6 Vocab arithmetic — **65,536 is NOT the real vocab** (**MEASURED**) — DECISION-RELEVANT

```
len(model.vocab)                          = 64400
len(added_tokens)                         = 507
max id in model.vocab                     = 64399   (contiguous 0..64399)
tokenizer.get_vocab_size(with_added=True) = 64400
tokenizer.get_vocab_size(with_added=False)= 64400
config.json vocab_size                    = 65536
```

**The 507 added tokens are INSIDE `model.vocab`, not appended on top.** `len(model.vocab) +
len(added_tokens)` = 64,907 which does **NOT** equal 65,536 — because that sum double-counts.
The true, contiguous tokenizer vocabulary is **64,400 ids (0..64399)**.

Therefore:
- **`vocab_size: 65536` in config.json over-allocates the embedding matrix by 1,136 rows.**
  `model.embed_tokens.weight` is literally `[65536, d]` (verified in §2.7) but ids 64400..65535
  are **unreachable / never trained**. This is a deliberate pad to a power of two (65,536 = 2^16)
  for tensor-core / kernel alignment. **MEASURED + INFERRED (the "why").**
- **Usable text vocabulary** = 64,400 − 507 special = **63,893 text tokens**.
- So "LFM2 has a 65,536 vocab" is true only in the *embedding-matrix* sense. The tokenizer emits
  **64,400** distinct ids and only **63,893** of them are text. **The parent should not assume
  63.9K usable text tokens ≈ 65,536.** Headroom vs GPT-2's 50,257 (of which 50,256 are text +
  1 special) is 63,893 / 50,256 = **1.27x**, not 65,536/50,257 = 1.30x.

### 3.7 Special / added token layout (**MEASURED**) — **BOTTOM of the vocab**

Special tokens are allocated at the **LOW end (ids 0–500)** plus a small block at the **TOP
(64394–64399)**:

- **0–13 functional:** `<|pad|>`(0), `<|startoftext|>`(1), `<|endoftext|>`(2), `<|fim_pre|>`(3),
  `<|fim_mid|>`(4), `<|fim_suf|>`(5), `<|im_start|>`(6), `<|im_end|>`(7), `<|tool_list_start|>`(8),
  `<|tool_list_end|>`(9), `<|tool_call_start|>`(10), `<|tool_call_end|>`(11),
  `<|tool_response_start|>`(12), `<|tool_response_end|>`(13)
- **14–500 reserved:** `<|reserved_4|>` … `<|reserved_490|>` — **487 reserved slots**
- **64394–64399 (top):** `<|cot_start|>`, `<|cot_end|>`, `<|review_start|>`, `<|review_end|>`,
  `<|file_start|>`, `<|file_end|>` (6 tokens, appended later — clearly a post-hoc addition since
  they sit above the BPE range rather than in the reserved pool)
- **Two non-special added tokens (curiosity / MEASURED):** id **64011 = `'Mathias'`** and id
  **64014 = `'python'`**, both `special: false`. `Mathias` is co-author Mathias Lechner — an
  artifact of how the vocab was finalized. Harmless, but evidence the vocab was hand-touched.

**Layout takeaway for the parent:** if you clone this scheme, specials at the bottom means your
text-token ids all shift by ~500 vs a naive build; and the reserved pool is what let LFM2.5 add
audio + vision tokens **without changing the embedding matrix or the text BPE** (§2.9). That is a
genuinely good design worth copying: **reserve ~500 slots up front.**

### 3.8 MULTILINGUAL FRACTION — the key quantitative claim (**MEASURED**)

Method: decoded every `model.vocab` key through the GPT-2 byte-level unicode↔byte map back to raw
bytes, then UTF-8 decoded, then classified each character by Unicode name prefix. Script:
`/scratch/users/ericrcwu/liv/tok/tokml.py`. Zero entries failed byte-level decoding (confirming
pure byte-level BPE).

```
len(model.vocab)              = 64400
  special/added tokens        =   507
  pure-ASCII text tokens      = 48302
  non-ASCII text tokens       = 14800
  partial-UTF8 byte fragments =   791   (multi-byte pieces that aren't valid UTF-8 alone)
  not byte-level decodable    =     0
```

### **NON-ASCII = 14,800 / 63,893 non-special = 23.16% of the usable vocab**
### **= 22.58% of the nominal 65,536**

Script breakdown (% of the 63,893 non-special vocab):

| script | tokens | % of vocab | examples |
|---|---|---|---|
| **Cyrillic** | 4,402 | **6.89%** | `о` `а` `е` `и` `н` |
| **CJK (Chinese)** | 3,346 | **5.24%** | `的` `一` `人` `中` `年` |
| **Latin-accented** | 3,298 | **5.16%** | `é` `ó` `á` `í` `ä` |
| **Japanese** (kana) | 1,410 | **2.21%** | `の` `い` `で` `ー` `に` |
| **Korean** (hangul) | 1,038 | **1.62%** | `다` `이` `는` `에` `을` |
| **Arabic** | 779 | **1.22%** | `ا` `ل` `ي` `ال` `م` |
| Punct (typographic) | 324 | 0.51% | `’` `、` `。` `，` |
| Space/Ctrl (NBSP etc.) | 63 | 0.10% | `\xa0` ` ` `　` |
| Symbol | 54 | 0.08% | `°` `−` `±` `×` `°C` |
| Greek | 44 | 0.07% | `α` `ο` `β` `μ` `ν` |
| Devanagari | 6 | 0.01% | `ा` `्` `े` |
| Hebrew | 6 | 0.01% | `י` `ו` `ר` |
| Fullwidth / misc | ~25 | ~0.04% | `１` `½` `²` `º` |
| *partial UTF-8 fragments* | 791 | 1.24% | (byte pieces, mostly serving the above) |

**Interpretation (INFERRED from MEASURED counts):**
- LFM2's tokenizer is **genuinely multilingual and it is expensive**: ~**23%** of the vocabulary
  (≈14,800 slots, plus ~791 partial-byte helpers ⇒ ~15,600 ≈ **24.4%**) is spent on non-English.
- The language set matches Liquid's marketing exactly: **Russian, Chinese, Spanish/French/German
  (via Latin-accented), Japanese, Korean, Arabic**. Cyrillic is the single largest bucket,
  *larger than Chinese*.
- Devanagari/Hebrew/Thai/Indic are ~absent (≤6 tokens each) — this is a **7-language** tokenizer,
  not a 100-language one.
- **For an English-only from-scratch research model, adopting LFM2's tokenizer wastes ~15,000 of
  65,536 embedding rows (~23%) on scripts your corpus will never contain** — on top of the
  **1,136 structurally dead rows** from §3.6. Combined dead/wasted: ~**16,700 rows ≈ 25.5%** of the
  embedding matrix. At 350M/`d=1024` that is ~17M parameters (~5% of the model) of pure ballast;
  because embeddings are **tied**, it is also 17M rows of a softmax you never need.

### 3.9 Fertility, measured (**MEASURED**)

`tokenizers` 0.22.x, `add_special_tokens=False`:

| text type | chars | tokens | **bytes/token** | chars/token |
|---|---|---|---|---|
| encyclopedic English prose | 202 | 45 | **4.489** | 4.489 |
| plain literary English | 109 | 29 | **3.759** | 3.759 |
| Python code | 136 | 55 | **2.473** | 2.473 |
| LaTeX math | 111 | 54 | **2.056** | 2.056 |

~3.8–4.5 bytes/token on English is **unremarkable** — roughly GPT-2/cl100k territory, not a big
compression win. The multilingual spend does cost English fertility relative to what an
English-only 64K could achieve. (A head-to-head vs GPT-2 on identical text is another child's job;
these are the LFM2-side absolute numbers.)

### 3.10 Lineage (**MEASURED where stated**)

First 25 merges are **all newline runs**:
```
['Ċ','Ċ'] ['Ċ','ĊĊ'] ['ĊĊ','Ċ'] ['Ċ','ĊĊĊ'] ['ĊĊ','ĊĊ'] ['ĊĊĊ','Ċ'] ['Ċ','ĊĊĊĊ'] ...
```
i.e. `\n\n`, `\n\n\n`, … up to long runs, before any letter merge. **63,683 merges total.**

- **NOT GPT-2 lineage.** GPT-2's first merges are `Ġ t`, `Ġ a`, `h e`, `i n`, `r e`, … LFM2's are
  newline runs. The vocab sizes, the split regex, and the specials all differ. **REFUTED** as
  GPT-2-derived.
- **NOT a Llama/SentencePiece derivative** — no `▁` metaspace, no unigram scores, no `spiece.model`,
  `byte_fallback: false`. The `sp_model_kwargs`/`legacy` keys in `tokenizer_config.json` are
  leftovers from a conversion template, not evidence of SentencePiece. **REFUTED.**
- The **split regex is the GPT-4 `cl100k_base` family pattern**, so the *pre-tokenization scheme*
  was borrowed from the tiktoken lineage; the *merges themselves* were trained fresh on Liquid's
  own multilingual corpus. **INFERRED** (regex match is MEASURED; "trained fresh" is inferred from
  the merge order and the 7-language distribution, which matches no public tokenizer I can point
  to).
- Newline-runs-first merge order means the training corpus was heavily **code/structured text** at
  the start of BPE training, or newlines were simply the most frequent bigram. Either way it
  indicates a **fresh training run**, not a graft. **INFERRED.**
- **I could NOT positively identify a parent tokenizer.** No public tokenizer I checked has 64,400
  entries with this specials layout. Verdict: **bespoke, trained by Liquid, using a cl100k-style
  pre-tokenizer regex.** Lineage beyond that: **UNCLEAR.**

---

## SECTION 4 — License, gating, and can-we-use-it

### 4.1 Gating — **NOT GATED** (**MEASURED**)

Every single fetch in this report was an **unauthenticated `curl`** from a Stanford FarmShare login
node with **no HF token**, and every one returned **HTTP 200**: `config.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `tokenizer.json`,
`LICENSE`, and HTTP range reads of `model.safetensors`. HF API `gated` field is `None` for all
LiquidAI LFM2/LFM2.5 repos listed. **No terms-acceptance click-through, no token required.**

(`LiquidAI/LFM2.5-1.2B` returned 401 — that is a **non-existent repo name**, not gating; the real
names are `LFM2.5-1.2B-Instruct`, `LFM2.5-1.2B-Base`, `LFM2.5-1.2B-Thinking`.)

### 4.2 The license — **LFM Open License v1.0** (**MEASURED**, read in full)

Source: `https://huggingface.co/LiquidAI/LFM2-350M/raw/main/LICENSE` (HTTP 200).
It is **Apache-2.0's text, modified**, with two substantive changes: a new §5 Commercial Use
Limitation and a new §11 Termination.

**The revenue cap — CONFIRMED. Exact quotes:**

> `"Threshold" shall mean annual revenue of 10 million United States dollars ($10,000,000) or more.`

> `5. Commercial Use Limitation.`
> `(a) The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity not exceeding the Threshold.`
> `(b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the Threshold is not licensed under this Agreement.`
> `(c) The Threshold shall not apply to a Qualified Non-Profit Organization's use of the Work or a Derivative Work for Non-Commercial or Research Purposes.`

> `"Commercial Use" shall mean any use of the Work for direct or indirect commercial advantage or monetary compensation.`

> `"Non-Commercial or Research Purposes" shall mean purposes that do not involve any use of the Work or a Derivative Work for Commercial Use.`

**Derivative works are explicitly permitted (§2):**

> `each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.`

**Redistribution conditions (§4)** — if you *distribute*, you must: (a) include a copy of the
License, (b) **mark modified files with prominent change notices**, (c) retain copyright/patent/
trademark/attribution notices, (d) propagate any NOTICE file.

**§11 Termination** (not in Apache-2.0):
> `This License will terminate automatically and immediately if You fail to comply with any of its terms and conditions. Upon termination, You must cease all use of the Work and any Derivative Works and delete all copies in Your possession.`

**§7 Trademarks:** no permission to use Liquid AI names/marks beyond describing origin.

### 4.3 Are the tokenizer files separately licensed? — **NO** (**MEASURED / INFERRED**)

The repo ships **one** `LICENSE` covering the repo contents; there is no separate tokenizer license,
no `LICENSE-tokenizer`, no per-file SPDX header in `tokenizer.json` (it is pure JSON with no
license field — verified, top-level keys are only
`version, truncation, padding, added_tokens, normalizer, pre_tokenizer, post_processor, decoder, model`).
So **`tokenizer.json` is covered by the LFM Open License v1.0**, same as the weights. **INFERRED**
from the absence of any separate grant — but it is the only reading available.

**IMPORTANT DISTINCTION — CONFIRMS the project's note.** Three different licenses are in play and
they are routinely conflated:
1. **`modeling_lfm2.py` / `configuration_lfm2.py` in `transformers`** — **Apache-2.0**, header reads
   `# Copyright 2025 The HuggingFace Team. All rights reserved. # Licensed under the Apache License,
   Version 2.0`. **MEASURED** (read the file on FarmShare). This is HuggingFace's reimplementation.
   Using it is unencumbered.
2. **The weights AND the tokenizer files on the LiquidAI HF repos** — **LFM Open License v1.0**
   with the $10M revenue cap. **MEASURED.**
3. **The arXiv paper 2511.23404** — **CC BY 4.0**. **MEASURED.**

The project's note that "the code is Apache-2.0 via transformers, which is DIFFERENT from the
weights' license" is **CONFIRMED**, and the tokenizer sits on the **weights** side, not the code side.

### 4.4 VERDICT — can we tokenize our corpus with it and pretrain from scratch?

**YES for this project, with conditions. (INFERRED — legal reading, not legal advice.)**

- Using `tokenizer.json` to tokenize a corpus and train a *new* model is at worst preparing a
  **Derivative Work**, which §2 expressly grants ("prepare Derivative Works of ... and distribute").
- Stanford academic research is **Non-Commercial or Research Purposes** — no Commercial Use, so §5's
  Threshold is not even engaged. Additionally §5(c) exempts qualified non-profits (Stanford) for
  research regardless of revenue. **Doubly clear for this use.**
- **The catch is downstream, not now:** the license is **viral onto derivatives**. A model trained
  with LFM2's tokenizer arguably embeds the Work, so §5 travels with it. If this research ever gets
  commercialized by an entity with **≥$10M annual revenue, the license does not cover it** (§5(b)).
  For a capstone that is fine; for a spinout it is a real encumbrance.
- If we ever **release** the artifact, §4 requires shipping the LFM license text, marking modified
  files, and retaining attribution. And **§11 auto-terminates on any breach** with a delete-all-copies
  obligation — stricter than Apache-2.0.

**Practical escape hatch (INFERRED):** we do not have to *use* LFM2's tokenizer to *match* its
shape. Training our own 64K byte-level BPE on an English corpus is cheap, has no license
encumbrance at all, avoids the ~23% multilingual waste, and lets us pick the digit rule
deliberately. LFM2's tokenizer is best treated as a **reference design** (specials layout, reserved
pool, power-of-two padding), not a dependency.

---

## SECTION 1 (continued) — What the paper actually says

Full text obtained from `https://arxiv.org/html/2511.23404v1` (HTTP 200, 871,354 bytes),
downloaded on FarmShare to `/scratch/users/ericrcwu/liv/tok/lfm2.html`, stripped to
`lfm2.txt` (179,069 chars) and grepped. All quotes below are **MEASURED** (verbatim from that text).

### 1.1 THE tokenizer paragraph — Section 2.2, "Tokenizer and special tokens."

This is the **entire** treatment of the tokenizer in the paper. Quoted verbatim:

> **"Tokenizer and special tokens. We use a byte-level BPE tokenizer (Sennrich et al., 2016) with a
> 65,536-token vocabulary. We used the same pre-training dataset discussed in Section 3 to train the
> LFM2 tokenizer, with a focus on encoding efficiency of the English, Japanese, Arabic, Korean,
> Spanish, French, and German languages. We additionally include JSON and other code-like data to
> improve tokenization of structured formats. The tokenizer includes special tokens for
> fill-in-the-middle training objectives (Bavarian et al., 2022), tool calling, and the ChatML chat
> template."**

Four sentences. That is all.

Everything in it **CONFIRMS** the artifact analysis in §3:
- "byte-level BPE" ⟶ matches `model.type == "BPE"` + ByteLevel pre-tokenizer/decoder. **CONFIRMED.**
- "65,536-token vocabulary" ⟶ matches `config.vocab_size`. But the *paper's own number is the
  padded embedding size, not the tokenizer's real 64,400.* The paper **never mentions 64,400** and
  **never discloses the 1,136-row pad.** (§3.6 is therefore new information relative to the paper.)
- "trained on the same pre-training dataset" ⟶ that dataset is **75% English / 20% multilingual /
  5% code** (§1.3), which predicts the measured **~23% non-ASCII** vocabulary almost exactly.
  **CONFIRMED — and this is a striking match.** The tokenizer's script distribution is essentially
  a mirror of the corpus mixture.
- "encoding efficiency of the English, Japanese, Arabic, Korean, Spanish, French, and German
  languages" ⟶ **7 named languages.** My script counts found exactly these plus Cyrillic and CJK.
  **PARTIAL MISMATCH worth flagging:** the paper's list does **NOT include Russian or Chinese**, yet
  **Cyrillic is the single largest non-ASCII bucket (4,402 tokens, 6.89%)** and CJK is second
  (3,346, 5.24%). Both are larger than any of Japanese/Korean/Arabic. So the tokenizer carries
  substantially more Russian and Chinese than the stated design target — an artifact of training on
  the raw pre-training corpus rather than a curated language mixture. (§3.1 of the paper does add
  "additional base support for Chinese, Italian, and Portuguese"; **Russian is named nowhere in
  the paper**, yet it is the biggest non-English script in the vocab. **This is a genuine,
  measured discrepancy between the paper and the artifact.**)
- "special tokens for fill-in-the-middle ..., tool calling, and the ChatML chat template" ⟶ matches
  ids 3–5 (`fim_pre/mid/suf`), 8–13 (tool), 6–7 (`im_start`/`im_end`). **CONFIRMED.**

### 1.2 What the paper is SILENT on (all **MEASURED absences** — I grepped the full text)

| question | verdict |
|---|---|
| Tied vs untied embeddings | **SILENT.** Zero occurrences of "tied", "untied", "weight tying", "embedding matrix" anywhere in the paper. The only related phrase is "including options for shared weights and cache reuse" in the §2.1 *layout* search space — that refers to **layer/block weight sharing, not embeddings.** |
| Fertility / bytes-per-token / tokens-per-word | **SILENT.** No occurrence of "fertility", "compression ratio", "bytes per token", "tokens per word". No number of any kind quantifying tokenizer efficiency. |
| Comparison to other tokenizers | **SILENT.** The paper uses "identical tokenizer settings" as a *control* in throughput comparisons (§2.4) but never compares tokenizers as objects. |
| Ablation varying tokenizer or vocab size | **SILENT.** Table 1 (model hyperparameters: layers, d_model, FF dim, heads, conv k, experts, MoE) has **no vocab column**. No ablation anywhere varies vocabulary size. |
| Vocab-sizing rationale (why 65,536?) | **SILENT.** The number is stated, never justified. My inference (§3.6) that it is a 2^16 alignment pad over a 64,400 tokenizer is **not** stated by Liquid. |
| Tokenizer as a design decision / tradeoff discussion | **SILENT** beyond the one paragraph. No discussion of vocab-size-vs-embedding-cost, which is notable for a paper otherwise obsessed with on-device memory. |
| "SentencePiece" / "Unigram" | **Zero occurrences.** Consistent with §3.1. |

### 1.3 STAR and the search space — vocab was NOT searched (**MEASURED**)

Verbatim, §2.1 "Search space." — the enumerated families are:

> "**Local context and subquadratic blocks**: gated short convolution blocks with varying kernel
> sizes, sliding-window attention (Child et al., 2019), and a family of sub-quadratic sequence
> blocks including linear attention variants ..., Mamba ..., Mamba2 ...; Liquid-Time Constant
> networks such as CfC ..."

> "... **grouped-query attention** ... group counts and head dimensions, augmented with QK-Norm ...
> • **Position-wise blocks**: SwiGLU feed-forward blocks (Shazeer, 2020) with expansion ratios
> chosen by search. • **Layout**: interleaving patterns of local context blocks, global context
> blocks, position-wise blocks, and overall block counts under fixed parameter budgets, **including
> options for shared weights and cache reuse.** • **MoE options**: per-layer sparse FFNs with
> varying width and expert granularity."

**Vocabulary size and embeddings appear NOWHERE in the search space.** Note also the layout bullet's
"under **fixed parameter budgets**" — since embeddings are a large fixed chunk of a 350M model, the
vocab was necessarily a **fixed constant** the search worked around. The paper does not say this
explicitly, so: **vocab was NOT part of STAR/the LFM2 search — INFERRED (strong), from the MEASURED
absence in an otherwise exhaustive enumeration.**

**Important correction to a likely assumption:** the paper explicitly **distances LFM2 from STAR**:

> "Our earlier academic prototype (**STAR**) (Thomas et al., 2024) explored a specific design space
> of operator/layout choices with an evolutionary search heuristic optimized on proxy signals (i.e.,
> perplexity for quality, cache size for efficiency). **In practice, these proxies do not transfer
> reliably to downstream task scores or device-level latency and memory, limiting their utility as
> optimization objectives.**"

So **LFM2 was NOT produced by STAR.** STAR is described as a superseded "earlier academic
prototype" whose proxy objectives Liquid says failed. LFM2 used a **hardware-in-the-loop** search
measuring real TTFT / decode latency / peak RSS on a Galaxy S25 and a Ryzen HX 370, ranked by
"hypervolume improvement on the quality–latency–memory Pareto frontier". **CONFIRMED / MEASURED.**
If the parent's notes say "LFM2 came out of STAR", that is **REFUTED by the paper itself.**

### 1.4 STAR's arXiv ID — VERIFIED

Queried the arXiv API directly (`export.arxiv.org/api/query`, title search) rather than guessing:

- **arXiv:2411.17800v1** — "STAR: Synthesis of Tailored Architectures"
- Published 2024-11-26
- Authors: Armin W. Thomas, Rom Parnichkun, Alexander Amini, Stefano Massaroli, Michael Poli
- Venue per LFM2's bibliography: "The Thirteenth International Conference on Learning Representations" (ICLR 2025)

**MEASURED / CONFIRMED.** The ID **2411.17800** resolves and matches the LFM2 bibliography entry
("A. W. Thomas, R. Parnichkun, A. Amini, S. Massaroli, and M. Poli (2024) STAR: synthesis of
tailored architectures. In The Thirteenth International Conference on Learning Representations,
Cited by: §2.1"). This is a real, non-future-dated ID.

### 1.5 Training corpus composition (**MEASURED**, §3.1 verbatim)

> "The LFM2 dense models are pre-trained on a mixture comprising roughly **75% English text,
> 20% multilingual text, and 5% code**. We prioritize Japanese, Arabic, Korean, Spanish, French, and
> German for multilingual coverage, with additional base support for Chinese, Italian, and
> Portuguese. The MoE model LFM2-8B-A1B uses a similar mixture, but with a heavier emphasis on code
> (**60% English, 25% multilingual, 15% code**). For code data, 50% of examples use a
> fill-in-the-middle (FIM) objective."

> "The released dense LFM2 model checkpoints are pre-trained for **10T tokens at a context length of
> 4,096 tokens**. We then perform a mid-training phase on an additional **1T higher-quality
> tokens**..."

**Math is NOT broken out in pre-training** (it appears only in the SFT mixture: 7.4% dense /
26.2% MoE). SFT mixture is "**80% English**, with the remaining 20% uniformly distributed across
seven languages: Arabic, Chinese, French, German, Japanese, Korean, and Spanish."

**The number the parent needs:** the tokenizer was trained on a corpus that was **only 75%
English**, and the vocabulary reflects that (~23% non-ASCII, §3.8). Note the important asymmetry:
**20% multilingual *text* bought 23% of the *vocabulary*** — non-Latin scripts are
disproportionately expensive per unit of corpus because they share no subwords with English.

Also relevant (§7.1, LFM2-ColBERT):
> "The vocabulary size is 65,536 tokens, using the **same byte-level BPE tokenizer as the base LFM2
> models**." — and "We continued to pre-train LFM2-350M to **25T tokens**". Confirms one tokenizer
across the whole product line.

---

## SECTION 5 — Other Liquid publications on the tokenizer

### 5.1 LFM2 launch blog post (**MEASURED**)

`https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models`

**The tokenizer is NOT MENTIONED AT ALL.** No vocab size, no BPE, no tokenizer training data, no
tied/untied. The only tokenization-adjacent content is context length ("extended during pretraining
to 32k") and output budgets ("<4,096 tokens").

The blog **does** restate the corpus mix ("approximately 75% English, 20% multilingual, and 5% code
data"; focus on "Japanese, Arabic, Korean, Spanish, French, and German") and the license posture:

> models are under "an open license, which is based on Apache 2.0"; free for "academic and research
> purposes"; commercial use allowed for a "smaller company (under $10m revenue)"; larger orgs must
> "contact us (sales@liquid.ai) to obtain a commercial license."

**This independently CONFIRMS the $10M threshold** found in the LICENSE file (§4.2).

### 5.2 Model card (**MEASURED**)

`https://huggingface.co/LiquidAI/LFM2-350M/raw/main/README.md`. YAML front-matter:

```yaml
license: other
license_name: lfm1.0
license_link: LICENSE
language: [en, ar, zh, fr, de, ja, ko, es]
new_version: LiquidAI/LFM2.5-350M
```

Card table states **"Vocabulary size: 65,536"** for all four dense sizes, "Context length: 32,768",
"Training budget: 10 trillion tokens", **"License: LFM Open License v1.0"**. Parameter counts:
LFM2-350M = **354,483,968**; LFM2-1.2B = **1,170,340,608**.

> "**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, and Spanish."

**Note again: Russian is not listed as supported, yet Cyrillic is the largest non-ASCII block in the
vocabulary (6.89%).** The card is silent on the tokenizer beyond the vocab-size row — no BPE
mention, no tying, no fertility.

Sanity check on the embedding-cost claim: LFM2-350M has 354.5M params and a `[65536, 1024]`
embedding = **67.1M params = 18.9% of the whole model in the (tied) embedding table.** The ~25.5%
of that table which is dead-or-non-English (§3.8) is therefore ≈**17.1M params ≈ 4.8% of the model**.
**MEASURED arithmetic on MEASURED inputs.**

### 5.3 Repo file listing (**MEASURED**) — confirms what ships

`GET https://huggingface.co/api/models/LiquidAI/LFM2-350M` →
`gated: False, private: False, disabled: False`

Files: `.gitattributes`, `LICENSE`, `README.md`, `chat_template.jinja`, `config.json`,
`generation_config.json`, `model.safetensors`, `special_tokens_map.json`, `tokenizer.json`,
`tokenizer_config.json`.

- **`gated: False` — the repo is NOT gated. CONFIRMED by both the API field and by every
  unauthenticated fetch succeeding.**
- **No `tokenizer.model` / `spiece.model` / `vocab.json` / `merges.txt`** — confirming §3.1: this is
  a pure `tokenizers`-format fast BPE with no SentencePiece artifact and no GPT-2-style split files.
- Tags include `arxiv:2511.23404` — the model card itself links the technical report, an
  independent confirmation the arXiv ID is correct.
- `license: other`, `license_name: lfm1.0` (note: the *name* string is `lfm1.0`, the *file* says
  "LFM Open License v1.0").

### 5.4 What I looked for and did NOT find

- **A separate tokenizer license or tokenizer-specific terms** — does not exist (§4.3).
- **Any published fertility/compression benchmark by Liquid** — none, in paper, blog, or card.
- **Any statement by Liquid of tied embeddings** — none anywhere. The tying is only discoverable
  from the config default + the missing `lm_head` tensor (§2.7).
- **Any acknowledgement of the 64,400-vs-65,536 gap** — none. Liquid consistently says 65,536.
- **A dedicated tokenizer blog post / tokenizer repo** — none found.
- **LFM2.5 blog post** — I could not retrieve a distinct LFM2.5 launch post; `WebSearch` was
  unavailable this session (**HTTP 403 from the search backend — a tooling failure, not a
  finding**), so my coverage of Liquid's blog is limited to the LFM2 launch post I fetched
  directly. **UNCLEAR / INCOMPLETE — flagging honestly.** However, the LFM2.5 *artifacts* are
  decisive on the only question that matters: the LFM2.5 text BPE is **unchanged** from LFM2 (§2.9),
  so a hypothetical LFM2.5 blog cannot contain a different tokenizer.

---

## FINAL BOTTOM LINE (supersedes the placeholder at top)

1. **Type:** byte-level BPE (GPT-2/tiktoken family), `tokenizers` fast format, `tokenizer_class:
   PreTrainedTokenizerFast`. **No unk token, no byte_fallback, normalizer = null.** Not
   SentencePiece, not Unigram. Paper: *"We use a byte-level BPE tokenizer (Sennrich et al., 2016)
   with a 65,536-token vocabulary."* **MEASURED + CONFIRMED by primary text.**
2. **Vocab: 65,536 is a lie of convenience.** The real tokenizer has **64,400 ids (0–64399)**;
   `config.vocab_size: 65536` pads the embedding to 2^16, leaving **1,136 permanently dead rows**.
   Usable **text** vocab = **63,893** (64,400 − 507 special). Liquid never discloses this.
   **MEASURED — this is new relative to every published source.**
3. **Tied: YES**, in every LFM2 and LFM2.5 checkpoint. Three independent proofs: `Lfm2Config`
   class default `tie_word_embeddings = True`; `LFM2.5-1.2B-Base` config carries the legacy key
   `"tie_embedding": true`; and **no `lm_head.weight` tensor exists** in any safetensors header.
   The LFM2-350M/1.2B configs carry **neither** tie key — absence means the True default applies.
   **CONFIRMED.** The paper is **completely silent** on tying.
4. **Multilingual: ~23%.** **14,800 of 63,893 non-special tokens (23.16%) are non-ASCII.** Breakdown:
   Cyrillic 6.89%, CJK 5.24%, Latin-accented 5.16%, Japanese 2.21%, Korean 1.62%, Arabic 1.22%,
   plus 791 partial-UTF-8 fragments (1.24%). Indic/Hebrew/Thai ≈ absent. **MEASURED.**
   **Surprise: Russian/Cyrillic is the LARGEST non-English block, and Russian is named nowhere in
   the paper, blog, or model card.**
5. **Digits: `\p{N}{1,3}` — 3-digit left-aligned chunking, NOT individual digits.** `84721` →
   `847|21`. A leading space does not merge with digits. **MEASURED, decision-relevant for
   passkey/phonebook probes.**
6. **License: LFM Open License v1.0** (Apache-2.0 text + §5 Commercial Use Limitation + §11
   Termination). **$10,000,000 annual-revenue Threshold CONFIRMED and quoted.** Research/non-profit
   use is expressly carved out (§5(c)). Derivative works expressly granted (§2). The tokenizer files
   are **not** separately licensed — they fall under the same LFM license, **unlike** HuggingFace's
   `modeling_lfm2.py` which is Apache-2.0. **CONFIRMED.**
7. **Gating: NONE.** `gated: False`; every file fetched unauthenticated at HTTP 200. **MEASURED.**
8. **Vocab was NOT searched.** The §2.1 search space enumerates conv kernels, SWA, linear
   attention/SSM, GQA groups/head dims, SwiGLU ratios, layout, MoE granularity — **no vocab, no
   embeddings**, and operates "under fixed parameter budgets". **INFERRED (strong) from a MEASURED
   exhaustive absence.**
9. **LFM2 is NOT a STAR product.** The paper calls STAR an "earlier academic prototype" whose
   proxy objectives "do not transfer reliably". **STAR = arXiv:2411.17800 — VERIFIED via arXiv API**
   (Thomas, Parnichkun, Amini, Massaroli, Poli; ICLR 2025). **REFUTES** any note claiming LFM2 came
   out of STAR.
10. **Zero tokenizer ablations, zero fertility numbers, zero vocab-sizing rationale** in the paper,
    blog, or card. Measured myself: **~4.5 bytes/token on English prose, 3.8 on plain English,
    2.5 on Python, 2.1 on LaTeX.** Unremarkable — no compression advantage over GPT-2-class.

### THE SINGLE MOST DECISION-RELEVANT FINDING

**Copying LFM2's tokenizer to train an English-only research model buys 63,893 usable text tokens,
not 65,536 — and roughly a quarter of them (~15,600, incl. byte fragments) encode Russian, Chinese,
Japanese, Korean, Arabic, and accented Latin that an English corpus will never emit.** Add the
1,136 structurally dead rows and **~25.5% of the tied embedding matrix is ballast** — at 350M/d=1024
that is ~17M of 354M parameters (**~4.8% of the model**) that neither trains nor helps, and because
embeddings are tied it is also ~15,600 permanently-near-zero softmax logits distorting the output
distribution.

**The head-to-head, MEASURED (see §6):** pure-ASCII text tokens are **LFM2 = 48,302** vs
**GPT-2 = 49,383**. **LFM2's tokenizer has ~1,100 FEWER English-usable tokens than GPT-2 does**,
despite a nominal vocabulary 30% larger.

⇒ **Retokenizing to *LFM2's actual tokenizer* is not justified by vocabulary headroom — it would
slightly reduce English capacity while adding a license encumbrance.** If the parent wants
LFM2-*shaped* (65,536 embedding rows for 2^16 alignment, ~500 reserved specials at the bottom,
tied embeddings), the right move is to **train a fresh 64K byte-level BPE on the English corpus** —
same shape, ~25% more English capacity, no LFM Open License, and a digit rule chosen deliberately
rather than inherited.

---

## SECTION 6 — Head-to-head vs GPT-2 (the number the parent's decision turns on)

I did not want to assert "LFM2 buys you less English than GPT-2" from inference, so I measured it.
Downloaded `https://huggingface.co/openai-community/gpt2/raw/main/vocab.json` (HTTP 200,
1,042,301 bytes) on FarmShare and ran the **identical** byte-level-decode + script-classification
pipeline used in §3.8.

| | **GPT-2** | **LFM2 / LFM2.5** |
|---|---|---|
| nominal `vocab_size` in config | 50,257 | **65,536** |
| real tokenizer ids | 50,257 | **64,400** (1,136 dead rows) |
| special / added tokens | 1 (`<\|endoftext\|>`) | **507** |
| **pure-ASCII text tokens** | **49,383** | **48,302** |
| non-ASCII text tokens | 529 (**1.05%**) | **14,800 (23.16%)** |
| partial-UTF-8 byte fragments | 344 | 791 |

**MEASURED.** Both columns produced by the same script, same method, same session.

### The punchline

> **LFM2's 65,536-token vocabulary contains FEWER English-usable tokens (48,302) than GPT-2's
> 50,257-token vocabulary (49,383).**

The nominal vocabulary grew **+30.4%** (50,257 → 65,536) while English-usable capacity went
**−2.2%** (49,383 → 48,302). The entire nominal increase, and then some, was consumed by:
- **+14,271** non-ASCII tokens (529 → 14,800) — the multilingual spend,
- **+506** special/reserved tokens (1 → 507),
- **+1,136** structurally dead padding rows,
- **+447** extra partial-UTF-8 byte fragments.

GPT-2 is ~99% English-ASCII by construction; LFM2 is ~76%. **A 30% bigger vocabulary that is 23%
non-English nets out to slightly LESS English capacity.**

### Consequence for the retokenization decision (**INFERRED** from the above **MEASURED** numbers)

- **Retokenizing GPT-2 → LFM2's tokenizer would not improve English fertility.** With ~2% fewer
  English tokens available, English tokens-per-word should be flat-to-marginally-worse. (Direct
  fertility A/B on identical text is another child's assignment; my §3.9 gives LFM2's absolute
  numbers — ~4.5 bytes/token on prose — which are ordinary.)
- **The only real gains from moving to 65,536 are structural, not lexical:** 2^16 alignment,
  a 507-slot special/reserved block, and the LFM2-matching embedding shape. **All three are
  obtainable by training our own 64K English BPE**, which would additionally recover the ~15,600
  multilingual slots as English capacity — that *would* be a genuine fertility win, and it carries
  no LFM Open License obligation.
- **If the goal is architectural fidelity to LFM2** (so numbers are comparable to published LFM2
  results), use `vocab_size = 65536` + tied embeddings + ~500 reserved specials, but **populate it
  with our own English-trained merges.** Shape-identical, license-clean, ~30% more English capacity
  than either GPT-2 or LFM2's own tokenizer.
- **If the goal is to load LFM2 weights** (it is not — this is a from-scratch model), then the
  tokenizer is mandatory and none of this is optional.

---

## APPENDIX — reproduction

All artifacts live on FarmShare at `/scratch/users/ericrcwu/liv/tok/` (nothing under
`agent-runs/` was touched; no GPU used; total download ~11 MB of tokenizer/config JSON):

```
lfm2_tokenizer.json          4,732,426 B  md5 83703d34b17eca9dd512ca67668028fc  (LFM2-350M == LFM2-1.2B)
lfm2_12b_tokenizer.json      4,732,426 B  md5 83703d34b17eca9dd512ca67668028fc
lfm25_tokenizer.json         4,733,020 B  md5 910b7828a3926f1447b9a65df373f090  (LFM2.5-1.2B-Base)
gpt2_vocab.json              1,042,301 B
lfm2.html / lfm2.txt         arXiv 2511.23404v1 full text
LFM2-350M__*.json, LFM2-1.2B__*.json, LFM25-1.2B-Base__*.json
hdr_LFM2-350M.json, hdr_LFM2-1.2B.json, hdr_LFM2.5-1.2B-Base.json   (safetensors headers via HTTP range)
tokprobe.py  tokml.py  tok3.py                                       (analysis scripts)
```

Python: `/scratch/users/ericrcwu/kda/venv/bin/python` (transformers 5.14.1, tokenizers 0.22.x).

**Known gaps / honest limits:**
- `WebSearch` returned HTTP 403 all session (backend auth failure), so my sweep of Liquid's blog is
  limited to the one LFM2 launch post I fetched by direct URL. No LFM2.5-specific blog was read.
  Mitigated by the fact that the LFM2.5 tokenizer artifact is provably the same text BPE (§2.9).
- Tokenizer **lineage** could not be positively identified beyond "cl100k-style split regex,
  fresh-trained merges". Marked **UNCLEAR**.
- Cross-tokenizer **fertility A/B on a real corpus** was out of scope here (another child's task);
  §3.9 and §6 give LFM2's absolute numbers and the vocab-capacity comparison only.
