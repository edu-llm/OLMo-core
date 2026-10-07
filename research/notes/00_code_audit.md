# Code audit of the P4 spacing experiment (configs/final.yaml)

## TL;DR

*The same code and config are on `edu-llm/OLMo-core`, branch `anshulm/fictionalqa-review`. The local copy used for this audit has been deleted.*

- **Reviews aren't per fact.** Each "review event" is one update of 16 old statements drawn at random from one FictionalQA document style, with styles taken in rotation.
  - Only 31–40 of the 80 evaluated old facts are ever reviewed, and only 11–17 are reviewed twice or more.
  - Only the global first (step 8) and last (step 165) reviews are matched. For three of the five styles (encyclopedia, news, social), the expanding arm's last review comes 45–57 updates earlier than the uniform arm's. Blog is 28 earlier, and only corporate is matched.
- **Exposure is uneven.** Old facts per style range from 28 (blog) to 180 (news), so Stage-1 exposures range from about 6.9 to 1.1 per fact. Between 4 and 8 evaluated "old" facts are never trained at all.
- **Train and test formats differ.** Training is `"Fictional fact: …"` with loss on every token. Testing is `"Question: …\nAnswer:"` with loss on the answer plus EOS. The answer appears verbatim in the training statement 93% of the time.
- **The evaluation set is small and fixed:** 16 items per style, 80 in total, the same for every seed and condition. Sequence length 64 truncates nothing.
- **Free fixes.** Exact match and per-item losses are already logged, so item-level reanalysis needs no retraining. FictionalQA has 40 unused events, which make a ready never-trained control set.


Method: re-built the exact data split from the pinned FictionalQA parquet (rev 131cb74) using the repo's own
`records_from_fictionalqa_rows`, reproduced the review schedules with `fixed_review_steps`, and re-simulated the
per-fact sampling with the same RNG seeding strings as `trainer.py` (`f"{seed}:{stage}:old:{skill}"`, pool
shuffles with `data.seed+11/+37`). Tokenizer: the pinned DataDecide-dolma1_7-300M tokenizer.json.
Caveat: simulation, not a re-run; it assumes record order in the jsonl equals the in-memory order (which the code implies).

## Facts the paper does not currently state
- Model: `allenai/DataDecide-dolma1_7-300M` (Dolma 1.7 recipe), rev 4b1b42f; d_model 1024, 16 layers, SwiGLU (mlp_ratio 8), untied 50,304-token embeddings. LoRA r=16, α=32, dropout 0.05 on att_proj/attn_out/ff_proj/ff_out = 327,680 params/layer × 16 = 5.24M (matches paper). fp16 autocast; AdamW β=(0.9,0.95), wd 0.1, grad-clip 1.0.
- Data: FictionalQA `fict_qa` (7,500 QA rows, 100 events). Duplicate question clusters reduced to their root. 20 events → old (386 facts), 40 events → new (689 facts). **40 events (40%) are unused.**
- "Skill" = FictionalQA document style (blog, corporate, encyclopedia, news, social). Old-fact pool sizes: blog 28, corporate 54, encyclopedia 41, news 180, social 83.
- Training text = `"Fictional fact: <fict statement>"`, loss on all tokens. Eval = `"Question: <question>\nAnswer:"` + ` <natural_answer>` + EOS, loss on answer tokens + EOS. The model never trains on the Q/A format. The answer string appears verbatim in the training statement for 1,003/1,075 facts (93%), so evaluation tests format transfer + recall of a seen span, not inference.
- Sequence lengths: training max 37 tokens (median 17); eval max 35. `sequence_length: 64` truncates nothing. Mean answer length 3.7 tokens, so the EOS token is ~20% of each item's scored tokens.
- Each update = 16 statements sampled **with replacement** from a single style; styles visited round-robin.
- Stage 1: 60 updates = 12 per style = 192 samples/style → expected exposures per fact range from 6.9 (blog) to 1.07 (news); 4–8 of the 80 evaluated old facts are never seen in Stage 1 at all (seed-dependent).
- Each "review event" = one update of 16 old statements from one style (round-robin). 12 events → blog & corporate get 3 review batches, the other styles 2.
- Schedules (Stage-2 step indices): uniform [8,22,37,51,65,79,94,108,122,136,151,165]; expanding (ratio 1.35) [8,10,13,17,22,29,38,51,68,91,123,165].
- Final evaluation: first 16 items per style after a fixed shuffle → **80 old-fact items, identical for all conditions**. Exact match (greedy) is computed and logged per item but not reported in the paper. Per-item losses are logged (`old_items`).

## Consequences for the paper's claims
1. **The schedule manipulation is not an item-level spacing manipulation.** Per evaluated fact, review exposure is 0–3 presentations: only 31–40 of 80 evaluated facts are reviewed at all, and only 11–17 are reviewed ≥2 times (seed-dependent; identical facts across arms because review sampling RNG is shared). The human spacing literature concerns repeated presentations of the *same item*; here "expanding vs uniform" mostly changes *when style-level batches* land.
2. **Per-style recency is not matched.** Only the global first (8) and last (165) review steps are matched. Last review per style — uniform: blog 151, corporate 165, encyclopedia 108, news 122, social 136 (mean 136.4); expanding: blog 123, corporate 165, encyclopedia 51, news 68, social 91 (mean 99.6). Expanding reviews of most facts therefore end ~37 steps earlier on average. The README's "same first and last review steps" is true globally but not per item.
3. **Review-vs-no-review benefit may be partly non-specific.** Since ~half the evaluated facts receive no review, the 4.3% average gain mixes fact-specific retention with style/format-level effects. This is testable from existing logs: compare reviewed vs never-reviewed facts within the review arms (dose–response on #reviews).
4. **Old-fact loss falls during new-fact training in every arm** (4.20 → 3.83 in no-review). Combined with the train/eval format mismatch, this indicates a large domain/format-adaptation component in the metric. A never-trained control set is free to build: the 40 unused FictionalQA events.
5. Exposure imbalance across styles (≈6× more exposures per blog fact than per news fact) while eval weights styles equally.

## Cheap fixes (no new training needed)
- Item-level reanalysis from logged `old_items`: mixed-effects model with fact and seed random effects; reviewed vs unreviewed facts; #reviews dose-response; per-fact time-since-last-review.
- Report exact match alongside loss.
- State every detail above in Methods.

## Fixes needing new runs (cheap: 300M LoRA, ~420 updates/run)
- Never-trained-fact control (unused events) evaluated at every checkpoint.
- Item-level schedules: every old fact gets exactly k reviews at fact-specific expanding vs uniform positions, with first and last exposure matched *per fact*; include a massed control and a contracting control.
- More seeds (≥10) and a larger eval set (all 386 old facts, not 80).
- Accuracy-based metric (e.g., FictionalQA's `gend_mcq_w_grades` multiple-choice config, or log-likelihood margin vs distractors).
