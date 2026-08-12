# Provenance

Where each module came from, and what changed. The previous tree had a
`RETAINED-RESULTS.md` whose rule was *"Nothing outside this file may be cited in
the paper. A number without an artifact behind it is withdrawn."* That discipline
is worth keeping, so: **no file here is a verbatim copy**, and every behavioural
change is recorded below rather than in a commit message nobody reads.

Source tree: `/workspace/edullm/memory-split`, refs `main` (`0201fc7`),
`p3-composition` (`1c7d3de`, identical to `origin/feat/ood-2-hop`),
`origin/feat/compose-finetune`, and `f875db1` (the withdrawal commit, whose
`docs/RETAINED-RESULTS.md` is the single most useful document in that repo).

| module | derived from | relationship |
|---|---|---|
| `tokenizer.py` | `160m-farmshare/train/tokenizer.py` | Same frozen special ids and `VOCAB_SIZE=50304`; same segment-independent `encode_segments` contract. **Added** a byte-level fallback and `require_production_tokenizer`, and a fifth control token (`<|db_fail|>`) that the old tokenizer declared in the paper but not in code. |
| `records.py` | `corpusgen/records.py` (`p3-composition`) | `lookup_segments` preserved exactly. **Changed** `Doc` from two pre-rendered arms to one stream plus a parallel role list, which is what makes the four-condition masker possible. |
| `bios.py` | `corpusgen/bios.py` | Same idea, new pools. **Changed**: pools enlarged so `split_pools` can partition them (the old `major` had 125 constructible values against 100 used -- no room to split at all); `birth_city`/`current_city` no longer share one list; `birth_date` dropped, so `bits_per_entity()` is 41.47 rather than 52.96 and is **computed, not a constant**. |
| `nhop.py` | `corpusgen/compose.py` | Rewritten. Depth was hardcoded at 13 sites; here it is a parameter. **Added** layered-DAG edges (the old `assign_bridges` self-intersects at depth >= 3), a shortcut gate, `eligible_starts` (so depth is orthogonal to entity identity), >= 10 templates per slot against the old 1, and `pn_table`. Kept: hop *k*+1 never names its subject in prose. |
| `masking.py` | `160m-farmshare/corpusgen/mask_ledger.py` | The best-designed file in the old tree and **never run** (its 15-config matrix has no results). Kept the one-stream/four-sidecar architecture and the `(span_length, position_bin)` matching key. **Fixed** a real bug: independently sampled control spans could overlap, so the masked union was smaller than the sum of demands while still looking count-correct. **Added** the scattered difficulty-matched control, cue-window exclusion, and `mask_restatements` as an ablatable axis. |
| `store.py` | `organizer/store.py` | Essentially verbatim; same `normalize` and JSONL format. |
| `model.py` | `160m-farmshare/train/model.py` | Same architecture (RMSNorm, RoPE, SwiGLU, untied head). **Changed** the loss to `sum / fixed divisor` and made the divisor mandatory. **Added** `flops_per_token` reporting both `N` conventions. Presets state `ctx` explicitly, because the old tree set `ctx=1024` for `d160m` in code while the paper said 2048. |
| `generate.py` | `evals/generate.py` | Same store protocol and `QUERY_TOKEN_CAP`. **Replaced** left-padded prefill with length-grouped decoding. **Split** `n_lookups` into `n_query_spans` and `n_lookups` so the store-detached addressing rate stays measurable. |
| `scorers.py` | `evals/scorers.py`, `evals/recall.py` | Same two conventions, now explicit `mode` values on one function. **Fixed** `parse_answer`, which used `re.DOTALL` so the first match swallowed later `Answer:` lines and `matches[-1]` silently read the *first* answer. **Added** best-constant, Wilson intervals, and `z_vs_chance` with a below-chance flag. |
| `calibration.py` | new; the idea is from the fact-crowding line's G4 gate | That gate was implemented and did correctly refuse its endpoint -- **after 32 cells had trained**, because a module constant made the depth sweep inexpressible. Here it needs no trained model, takes depth as a parameter, and accepts pluggable adversarial degenerate policies. |
| `metrics.py` | new | The old tree computed **no KL at all** (`grep -r kl_div` is empty) despite reporting "6.7 nats KL". Adds bounded JSD, both KL directions, rank shift, the seed-floor comparison H2 needs, and a `compute_to_threshold` that refuses unbracketed crossings. |

## Deliberately not ported

- `train/trainer.py` and `trainer_v2.py`. v1 is the arm-asymmetric estimator that
  produced every reported result; v2 fixes the micro-batch bias but still
  normalises by the split arm's smaller valid-token count, so it is not
  arm-symmetric either. A trainer wrapper here should use `model.forward`'s fixed
  divisor. Note also that the old `COMPOSE.md` mandated `--trainer v2` for the
  split arm while `trainer_v2.py`'s docstring said it had trained nothing, and the
  run configs record no trainer field -- so which estimator produced the published
  split result is genuinely unrecorded. Log the trainer variant and full CLI into
  the run config.
- `160m-farmshare/corpusgen/srgm_worlds.py` and the `<|graph_*|>` trace stack. It
  is hop-general to `MAX_HOPS = 6` and production quality, but uses an
  incompatible masking convention (role-tagged `TaggedSegment` vs `(text, bool)`).
  Reconciling the two conventions is a bigger job than calling `lookup_segments`
  in a loop, which is what `nhop.render_doc` does.
- The mechanistic battery. None of the paper's two-hop mechanistic results
  (bridge-overwrite, layer 7-8 patching crossover, logit lens, the layer-6 lookup
  head, the layer-11 copy MLP) exists as code on any ref -- they were Colab-only.
  Reusable machinery does exist upstream (`evals/interp.py:96` activation
  patching, `probe/splice.py` value-entry profiling, `probe/faithfulness.py:30`
  which is the right shape for bridge-overwrite but hardcoded to iGSM's regex).

## Numbers carried forward, and their status

Usable (teacher-forced or loss-based, explicitly surviving the withdrawal):
MIA AUC 0.99 vs 0.50; `bits_in_weights` 0.0 for split against 17,175-27,099 for
dense; per-attribute recall 0.992-0.996 store-on vs **0.000** store-off; seed
spread in final loss <= 0.0086 nats; measured throughput d8m 462,611 and d40m
184,671 tok/s on one L40S.

**Withdrawn -- do not cite**: deduction 65.0/66.7/68.2, bits-per-entity
33.1/0.25/0.20, fresh-entity lookup 0.996, fact-use 58.1/31.2, four-way
recognition 0.90, the 1B held-out accuracies, PopQA, and "memorization appears
between 49 and 196 exposures per fact". All six of the first group appear in the
current paper draft; the test that enforced the withdrawal was dropped from
`main`.
