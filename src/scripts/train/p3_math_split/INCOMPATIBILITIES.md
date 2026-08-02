# OLMo-core ↔ Qwen2.5-0.5B: what the port had to work around

Verified against this repository's own source (OLMo-core 2.5.0). Every claim cites a file
and line so it can be re-checked after a rebase. Items marked **confirmed by execution**
are exercised by `smoke_test.py` (36 checks) or `src/test/nn/transformer/qwen_test.py`
(10 checks), not just read.

An earlier draft, written against upstream `allenai/OLMo-core`, listed a problem this fork
has already solved. It is recorded as resolved rather than deleted, because that
difference is why the port is smaller than expected.

---

## Summary

| # | Issue | Status |
|---|-------|--------|
| 1 | No Qwen2 preset; `llama_like` cannot express it | Handled — `qwen2_0_5b_config()` |
| 2 | Attention `bias` is one flag for all four projections | **Real, unavoidable** — `strip_attn_out_bias()` |
| 3 | Embedding weight tying | **Resolved by this fork** — `TransformerConfig.tie_word_embeddings` |
| 4 | Generic HF converter has no bias mappings | Real — standalone map in `qwen.py` |
| 5 | Loss divisor is the per-batch live token count | **Real, confounds the experiment** — `train_module.py` |
| 6 | Data must be `.npy`, not JSONL | Minor — `label_mask_paths` is native |

## 1. No Qwen2 preset

`TransformerConfig` has `qwen3_0_6B` through `qwen3_32B` and the `qwen3_5_*` family
(`nn/transformer/config.py:1304-1538`), but nothing for Qwen2. `llama_like` is the factory
those use, and it cannot express Qwen2 for two reasons:

- it derives the feed-forward hidden size as
  `ensure_multiple_of(int(8 * d_model / 3), 256)` → 2560 for `d_model=896`, where
  Qwen2.5-0.5B uses **4864**. Overridable by passing `feed_forward=...`.
- it hardcodes `bias=False` on the `AttentionConfig` it builds, which is issue #2 and is
  *not* overridable through its signature.

So `qwen2_0_5b_config()` writes the config out longhand. Everything else maps cleanly:
pre-norm blocks (`nn/transformer/block.py:152`), RMSNorm, SwiGLU with
`w2(silu(w1(x)) * w3(x))` (`nn/feed_forward.py:171`, so `w1=gate_proj`, `w2=down_proj`,
`w3=up_proj`), GQA with 14 query / 2 KV heads, `head_dim=64`, RoPE θ=1e6.

Note Qwen3 uses `LayerNormType.qwen_rms`; Qwen2 uses plain `rms`, which the logit-parity
test confirms empirically.

## 2. Attention bias — the one real blocker

**Confirmed by execution.** `Attention.__init__` takes a single `bias: bool` and applies it
to all four projections (`nn/attention/__init__.py:379-390`):

```python
self.w_q   = nn.Linear(d_model, n_heads * head_dim,    bias=bias, ...)
self.w_k   = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias, ...)
self.w_v   = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias, ...)
self.w_out = nn.Linear(n_heads * head_dim, d_model,    bias=bias, ...)
```

Qwen2 has bias on **q/k/v only**; `o_proj` has none. So:

- `bias=False` drops the q/k/v biases. Those hold *pretrained values*, not zeros. Dropping
  them corrupts the model.
- `bias=True` creates 24 spurious `w_out.bias` vectors (21,504 params) with no counterpart
  in the HF checkpoint. They would be randomly initialised and then trained, so the model
  silently drifts off the Qwen2 architecture.

Every other architecture in the repo — llama, gemma3, qwen3, qwen3_5 — is bias-free in
attention, which is why nothing else has hit this.

**Fix:** build with `bias=True`, then `w_out.bias = None`. Assigning `None` over a
registered parameter routes through `nn.Module.__setattr__` to
`register_parameter(name, None)`, which drops it from the module and from the state dict.
`strip_attn_out_bias()` does this and asserts the count.

## 3. Embedding tying — resolved by this fork

Qwen2.5-0.5B sets `tie_word_embeddings: true`; its LM head *is* its embedding matrix
(151936 × 896 ≈ 136M of ~494M parameters).

Upstream OLMo-core builds `embeddings` and `lm_head.w_out` independently with no tie
option, which would have meant tying by hand after build. **This fork added
`TransformerConfig.tie_word_embeddings`** (`nn/transformer/config.py:333`), wired through
`model.py` at `164`, `313` (re-ties after `to_empty()`), `366` (skips
`init_final_w_out`), `654`, `697`, and `903`.

That is strictly better than a hand-rolled tie, which would break on `to_empty()` and
would re-initialise the shared weight. `qwen2_0_5b_config(tie_word_embeddings=True)` uses
it, so the FSDP restriction an earlier draft imposed does not apply.

## 4. HF conversion

`nn/hf/convert.py` is a generic `StateMappingTemplate` system keyed on the HF `model_type`,
with entries for llama, gemma3, qwen3 and qwen3_5_text. Qwen2 could be added, but the
converter has **no `.bias` mappings at all** — every architecture it supports is bias-free
in attention. Teaching it about biases is a change to shared infrastructure that this one
model does not justify.

`qwen.py` therefore carries its own map: 2 global tensors + 12 per layer = 290 entries. It
is exhaustive by construction (an unmapped source or unfilled destination raises) and
round-trip tested, so a `transformers` release that renames something fails loudly rather
than leaving a layer at random init.

## 5. Loss divisor — the one that changes the result

**Confirmed by execution.** This is why the experiment needs code rather than config.

`TransformerTrainModule.train_batch` computes
(`train/train_module/transformer/train_module.py:356-358`):

```python
batch_num_tokens_for_loss = move_to_device(
    (batch["labels"] != self.label_ignore_index).sum(), self.device
)
```

and passes it as `loss_div_factor` (line 408), which `LMHead` divides the summed
cross-entropy by (`nn/lm_head.py:352-362`).

So the default is **token-mean over live (unmasked) tokens**. Under it the two arms divide
by different numbers, because the split arm's denominator excludes the fact block. Every
proof token in the split arm would receive roughly `1/(1 - mask_fraction)` ≈ **1.3–1.5×**
the gradient weight it receives in the dense arm. The arms would then differ in effective
learning rate on the shared tokens as well as in the mask, and the comparison would not
isolate the mask.

`loss_div_factor` is a local in `train_batch`, not a config field — but it reaches the
model through `model_forward()` (line 542), which is small and overridable:

```python
class FixedDivisorTransformerTrainModule(TransformerTrainModule):
    def model_forward(self, input_ids, labels=None, **kwargs):
        if kwargs.get("loss_div_factor") is not None:
            kwargs["loss_div_factor"] = self.fixed_loss_div_factor
        return super().model_forward(input_ids, labels=labels, **kwargs)
```

with `fixed_loss_div_factor = global_batch_size_tokens` — constant, identical in both arms,
and constant across steps, so it cannot interact with the LR schedule either. Ten lines, no
fork. See `train_module.py` in this directory.

The resulting loss values are smaller than a conventional SFT loss (dividing by all `B×S`
tokens, not just supervised ones), so the module also logs
`train/supervised token fraction` — ~1.0 for dense, ~0.7–0.85 for split. **If those are
equal, the mask is not being applied and the run is invalid.** Check it before reading
anything into the loss curves.

## 6. Data format

OLMo-core reads flat `.npy` token arrays, not JSONL. Loss masking is **natively supported**:
`NumpyFSLDatasetConfig` takes `label_mask_paths` (`data/numpy_dataset.py:2542`), parallel
`np.bool_` arrays read at `605-609` and applied in `get_labels` (`data/utils.py:598-599`).

Two consequences:

- **Mask semantics.** `get_labels` does `labels.masked_fill_(~label_mask, -100)` and *then*
  left-shifts (`utils.py:599,605`). So `label_mask[i] = False` removes token `i` from being
  **predicted** — the mask is indexed by the token whose prediction is scored, not by the
  token being conditioned on. `mask_alignment_test.py` pins this, and `smoke_test.py`
  verifies it against `get_labels` itself rather than restating it.
- **Padding.** `tokenize_corpus.py` pre-pads every example to exactly `sequence_length` and
  uses plain `NumpyFSLDataset`, so instance `i` is exactly example `i` with no
  cross-document packing. It wastes compute (~66% at seq 1024 on the pilot corpus) but
  makes "identical documents in identical order" a property you can assert byte-for-byte.

---

## Not verified

- `strip_attn_out_bias` under FSDP2 and `torch.compile` together. Single-GPU eager is
  covered by the parity test; the multi-GPU path is not exercised.
- The tied round trip through `save_model_and_optim_state`.
- `qwen2_0_5b_config` is checked at 24 layers; `strip_attn_out_bias(expected_layers=...)`
  is exercised at 2 layers by the smoke test.
