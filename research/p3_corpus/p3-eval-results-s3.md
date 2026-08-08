# P3 evaluation results in S3 — dense vs split (rerun-01)

Generated 2026-08-07. **Read-only**: the two `result.json` files and the two
checkpoint `config.json` files were downloaded for local scoring. No S3 object
was written, changed, copied, or deleted.

## Source objects (read-only)

```text
s3://sbsandbox-intern-edullm-outputs/teams/eval/vishnu-p3/rerun-01/dense/shard0/result.json
s3://sbsandbox-intern-edullm-outputs/teams/eval/vishnu-p3/rerun-01/split/shard0/result.json
s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints/step23166/config.json   (dense config)
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints/step23166/config.json (split config)
```

Each arm's run wrote a single shard (`shard0`) covering the full eval cohort;
both have `result.json`, `_READY`, and a top-level `box_READY`. Local copies and
the machine-readable comparison are in
`memorysplit-requery-exact/eval-rerun-01-compare/` (`dense.json`, `split.json`,
`comparison.json`).

## Status: reportable

`compare_arms.py` ran the full config-binding path (not `--skip-training-config-check`).
`training_config_equality_verified: true` — the dense and split checkpoint
configs are identical outside the arm switch, both declare `init_seed=42`, and
each result's model provenance binds back to its saved config.

- Comparison schema: `p3-comparison-v5`; result schema: `p3-eval-v9`.
- Decoding: **greedy** (`do_sample=false`); generation backend: **vLLM**.
- `context_length=16384`, `max_new_tokens=8192`, `nll_chunk_size=256`.
- Evaluator seed `20260801`; paired-bootstrap seed `20260801`, 10,000 resamples,
  percentile 95% CI, resampling unit = example ID within each family/condition.

## Provenance

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Initial weights SHA-256 | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` (identical both arms) |
| Training source commit | `d165d6cada69f8146df280d39ee7b46f5432f218` (identical both arms) |
| Checkpoint step | `23166` (both arms; init_seed 42) |
| Dense trained-weights root SHA-256 | `b61c687ed72badf120787af2232eeb1c8c503f35808303876bc8a2654ff79c31` |
| Split trained-weights root SHA-256 | `8289540c50c3deb5e3681e6e042316e3ef889cb40e7135e2fe382a1fa9d3c075` |
| Evaluator code SHA-256 | `657e9cafc4bf7f4c485ae00076a7ab497bad43e866d33e0945114805917340d6` |
| Corpus SHA-256 | `f76cdf2579c771003d76b2fffee2eb0984e6f3c75d4d0a08b92633357cef2d0a` |
| Tokenizer SHA-256 | `94e5b6293c02c537901028bcc4a5ca70caff6d04d78efd397b6bf8c6dba0d283` |

## What is measured

- **Target-token NLL** (teacher-forced) and **next-token accuracy** on the proof
  target only (premise/fact block is never in the loss). Reported token-micro
  (token-weighted) — example-macro is in `comparison.json` and agrees in sign.
- **Exact match**: whitespace-normalized whole-output match, only meaningful when
  the full proof fits the generation budget (`whole_proof_budget_eligible`).
- **Metamath validity**: sound tri-state verifier (`p3-metamath-tristate-v1`);
  rate is over *decided* (valid+invalid) pairs only.
- **Conditions**: `facts_present` (correct facts in context — the headline),
  `facts_absent` (header only), `facts_corrupted` (names kept, statements
  swapped). `facts_present` uses 100% of context-eligible examples; the two
  diagnostics use a deterministic 10% per-family sample.

All numbers below are **split − dense** with the paired 95% CI. Negative NLL =
split better; positive accuracy = split better.

## Headline: `facts_present` (both arms can read the correct facts)

Split matches or beats dense in **every family** — lower NLL and higher next-token
accuracy, with all split−dense CIs excluding 0. Base (untrained Qwen2.5-0.5B) is shown
as the floor; the Δ columns are always split−dense.

| Family | n | Base NLL (acc) | Dense NLL (acc) | Split NLL (acc) | ΔNLL split−dense [95% CI] | ΔAcc split−dense [95% CI] | Gold paste |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| enigma | 263 | 0.3103 (90.94%) | 0.0371 (98.67%) | 0.0322 (98.85%) | −0.0050 [−0.0053, −0.0047] | +0.19% [+0.17, +0.20] | 79.3% |
| isabelle | 590 | 0.7581 (84.17%) | 0.3652 (90.88%) | 0.3552 (91.21%) | −0.0099 [−0.0134, −0.0068] | +0.33% [+0.25, +0.42] | 82.1% |
| metamath | 494 | 0.3020 (91.82%) | 0.0715 (97.83%) | 0.0625 (98.08%) | −0.0091 [−0.0096, −0.0085] | +0.25% [+0.23, +0.28] | 93.0% |
| mizar | 2,485 | 0.8917 (76.90%) | 0.4764 (85.57%) | 0.4488 (86.39%) | −0.0276 [−0.0285, −0.0267] | +0.82% [+0.78, +0.85] | 65.5% |
| prf2 | 313 | 0.2978 (91.24%) | 0.0392 (98.60%) | 0.0342 (98.78%) | −0.0049 [−0.0052, −0.0047] | +0.18% [+0.17, +0.20] | 81.3% |
| thproofs | 46 | 0.9555 (76.72%) | 0.5412 (85.00%) | 0.5086 (85.91%) | −0.0326 [−0.0408, −0.0264] | +0.90% [+0.71, +1.18] | 66.6% |

**How the CIs are computed.** Every interval is a paired, example-level bootstrap
(`compare_arms.py`). For each family/condition the *paired* example set is resampled with
replacement 10,000 times (seed 20260801); on each resample the dense and split estimates
are recomputed and their difference (split − dense) recorded. The point estimate is the
difference on the full data; the 95% CI is the 2.5th–97.5th percentiles of the 10,000
resampled differences (percentile bootstrap). The resampling unit is the **example ID**
(whole examples move together, because tokens within a proof are not independent), and the
same resampled set is used for both arms, so the pairing cancels shared per-example
difficulty and tightens the interval. NLL endpoints use token-micro estimates
(Σ nll / Σ tokens); accuracy endpoints use token-micro accuracy. These are descriptive
intervals conditional on the two seed-42 checkpoints — not significance tests, no
multiple-comparison correction; "CI incl. 0" marks an interval spanning 0. Base has no Δ
column because it is a floor, not a paired arm.

Total `facts_present` cohort: 4,191 examples.

Gold paste-share = token-micro extractive coverage of the **gold reference target**
(the corpus proof, `row["target"]` — NOT any model output) at the **Qwen2.5 BPE token
level** (the granularity the NLL scores): fraction of gold-target tokens that fall in a
contiguous run also present verbatim in that condition's prompt. It is a property of the
dataset (identical for both arms), computed from `corpus-v3/eval/*.jsonl` with the
evaluator's own `build_prompt`. The distinct *model* (generation) paste-share — coverage
of each arm's generated proof — is not in this doc because rerun-01 did not persist
generations. (An earlier whitespace-token version is superseded: it read 0% on the
space-free `atp-v2` families, enigma/prf2, where a whole clause is one whitespace token;
BPE measures them correctly.)

### Paste-share by family (BPE)

At the token level the model actually predicts, targets are **copyable to varying
degrees** under `facts_present` (66–93% at any run length), so low per-token NLL/accuracy
is a weak indicator of reasoning — especially for the high-copy families:

- **metamath (93.0%)** — `metamath-proof-v2` targets re-emit, at every step, the full
  `|-` formula the cited step asserts after substitution; those formulas *are* the
  premise statements / goal (e.g. goal `|- ( -. A. x x = y -> A. z -. A. x x = y )`,
  premise `naev2 : |- ( -. A. x x = y -> A. z -. A. t t = u )`).
- **isabelle 82.1%, prf2 81.3%, enigma 79.3%** — the (negated) goal and premise atoms
  recur throughout the derivation. For the `atp-v2` families (enigma/prf2) the goal
  alone is a large recurring term, so even `facts_absent` stays ~76–78% copyable.
- **mizar 65.5%, thproofs 66.6%** — lower, and these two fall the most when the fact
  block is removed (mizar 65.5%→29.3%, thproofs 66.6%→21.7%): their copyable content
  lives in the premise statements, not just the goal.

Correction: an earlier whitespace-token computation reported enigma/prf2 at **0.0%** —
a metric artifact, since `atp-v2` clauses contain no spaces (a whole clause
`![X527]:(~v1_xboole_0(X527)|X527=k1_xboole_0)` is a single whitespace token and never
matches). At the BPE level they are ~79–81% copyable; the tables use BPE.

Crediting only verbatim runs of ≥ K BPE tokens (`facts_present`) shows the copying is
real long spans for metamath/isabelle/prf2/enigma, but mostly short-token overlap for
mizar/thproofs:

| Family | ≥1 | ≥2 | ≥4 | ≥8 | ≥16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamath | 93.0% | 85.4% | 72.8% | 49.7% | 30.3% |
| isabelle | 82.1% | 69.1% | 57.2% | 40.7% | 23.7% |
| prf2 | 81.3% | 65.3% | 50.3% | 34.8% | 14.3% |
| enigma | 79.3% | 62.6% | 45.8% | 30.2% | 12.0% |
| thproofs | 66.6% | 41.6% | 20.4% | 4.1% | 0.2% |
| mizar | 65.5% | 41.5% | 21.0% | 3.6% | 0.3% |

(% of target BPE tokens in a verbatim run ≥ K also present in the prompt. mizar/thproofs
collapse by ≥8, so their high ≥1 is short-match overlap — making them the least
copy-explainable families, and mizar shows the largest split−dense NLL gap.)

Why enigma/prf2 have ~0% exact match despite ~80% copyability: exact match is
whole-output free generation (≈ per-token-acc^N → ~0 for long proofs), and for `atp-v2`
it is ill-posed — the non-copyable ~20% are E-prover's run-specific names (`c_0_N`,
`X_N`) in non-unique derivations, which no model can string-match. Copyability drives the
per-token NLL/accuracy (highest of any family here), not whole-proof success.

**Model (generation) paste-share** — how much each arm's *own generated* proof
copies from the prompt, the more direct "is the win just copying?" probe — is
**not computable from the rerun-01 artifacts**: those dense/split runs were launched
without `--persist-generations`, so no generated text was stored (`generation_attempted`
is true, but there is no `generated` field). The base run in progress persists
generations; obtaining dense/split model paste-share requires a `--persist-generations`
re-run of those two arms.

Generation-based endpoints on **metamath** (the family whose full proofs fit the
budget), `facts_present`:

| Endpoint | base | dense | split | Δ split−dense [95% CI] |
| --- | ---: | ---: | ---: | --- |
| Exact match (evaluated) | 0.00% | 7.89% | 10.93% | +3.04% [+1.21, +5.06] |
| Exact match (budget-eligible) | 0.00% | 8.44% | 11.69% | +3.25% [+1.30, +5.41] |
| Metamath validity (decided) | 0.00% | 9.11% | 12.35% | +3.24% [+1.21, +5.26] |

Base scores **0 on all three** (0/494 exact; Metamath validity 0/494 valid — the verifier
decisively rejected all 494 base generations as *invalid*, 0 unknown), consistent with its
degenerate goal-repetition. So on the only sound end-to-end success metric, training moves
from 0% (base) to ~9–12% (dense/split).

Split produces more exact and formally **verified-correct** proofs than dense when
facts are present. But exact match is ~0 on the other families for reasons unrelated to
reasoning: for enigma/prf2 (`atp-v2`) it is **ill-posed** — the reference derivations
carry E-prover's run-specific internal names (clause ids `c_0_N`, variable indices
`X_N`) and are non-unique, so no model can string-match them (and 77% / 68% of those
proofs actually fit the budget, so it is not a budget effect). isabelle/mizar/thproofs
proofs mostly fit too (99–100%), but a 0.5B model rarely emits a whole exact proof
(isabelle 1.7%, mizar 2.7%, thproofs 0%). metamath is the only family with a sound,
naming-invariant checker, so its validity (≈12%) is the only trustworthy end-to-end
success signal — and it sits on a 93%-copyable target.

## Mechanism: `facts_absent` and `facts_corrupted`

These probe *where each arm stored the facts*. If dense memorized fact content
into weights it should degrade less when the context facts are removed/corrupted;
if split offloaded facts to context it should degrade more.

`facts_absent` (header only) — split − dense:

| Family | n | Base NLL (acc) | Dense NLL (acc) | Split NLL (acc) | ΔNLL split−dense [95% CI] | ΔAcc split−dense [95% CI] | Gold paste |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| enigma | 27 | 0.3080 (91.29%) | 0.0780 (97.84%) | 0.0835 (97.88%) | +0.0055 [+0.0036, +0.0077] | +0.04% [−0.03, +0.10] (CI incl. 0) | 76.2% |
| isabelle | 59 | 0.7679 (83.78%) | 0.4565 (88.92%) | 0.4462 (89.39%) | −0.0103 [−0.0206, −0.0002] | +0.46% [+0.20, +0.80] | 72.4% |
| metamath | 50 | 0.5020 (88.87%) | 0.2910 (93.22%) | 0.3720 (92.38%) | +0.0811 [+0.0676, +0.0987] | −0.83% [−1.03, −0.69] | 78.6% |
| mizar | 249 | 1.0685 (73.57%) | 0.7197 (80.06%) | 0.7527 (79.86%) | +0.0330 [+0.0256, +0.0421] | −0.20% [−0.33, −0.07] | 29.3% |
| prf2 | 32 | 0.3312 (90.59%) | 0.0883 (97.60%) | 0.0972 (97.56%) | +0.0089 [+0.0064, +0.0118] | −0.04% [−0.10, +0.01] (CI incl. 0) | 77.8% |
| thproofs | 5 | 0.9615 (78.39%) | 0.6666 (83.80%) | 0.6850 (84.20%) | +0.0184 [+0.0061, +0.1249] | +0.40% [−0.27, +0.68] (CI incl. 0) | 21.7% |

`facts_corrupted` (names kept, statements swapped) — split − dense:

| Family | n | Base NLL (acc) | Dense NLL (acc) | Split NLL (acc) | ΔNLL split−dense [95% CI] | ΔAcc split−dense [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| enigma | 27 | 0.3446 (90.29%) | 0.0793 (97.69%) | 0.0883 (97.71%) | +0.0090 [+0.0062, +0.0122] | +0.02% [−0.06, +0.09] (CI incl. 0) |
| isabelle | 59 | 1.0054 (79.37%) | 0.6061 (85.75%) | 0.6105 (85.69%) | +0.0044 [−0.0061, +0.0189] (CI incl. 0) | −0.05% [−0.28, +0.18] (CI incl. 0) |
| metamath | 50 | 0.3741 (90.43%) | 0.2533 (93.63%) | 0.3074 (92.88%) | +0.0541 [+0.0459, +0.0648] | −0.76% [−0.93, −0.62] |
| mizar | 249 | 1.0065 (75.12%) | 0.6111 (82.66%) | 0.6146 (82.80%) | +0.0035 [+0.0007, +0.0066] | +0.14% [+0.05, +0.24] |
| prf2 | 32 | 0.3186 (90.69%) | 0.0795 (97.62%) | 0.0851 (97.70%) | +0.0057 [+0.0035, +0.0082] | +0.08% [+0.03, +0.12] |
| thproofs | 5 | 1.3109 (69.77%) | 0.7563 (80.18%) | 0.7896 (80.23%) | +0.0332 [+0.0102, +0.0686] | +0.05% [−0.59, +0.65] (CI incl. 0) |

Absolute context-sensitivity, `facts_present → facts_corrupted` NLL (metamath):
dense 0.0715 → 0.2533 (~3.5×), split 0.0625 → 0.3074 (~4.9×). Both arms respond
to corruption, but split responds **more**.

Metamath validity in the diagnostics: `facts_absent` had decided pairs but 0%
valid for both arms; `facts_corrupted` had no eligible decided pairs. With facts
removed or wrong, neither arm produces verifiable proofs — as expected.

## Base control (untrained Qwen2.5-0.5B)

The base arm (pretrained Qwen2.5-0.5B, no fine-tuning) ran the identical eval
(`rerun-01/base`, commit `3ed7e796`, same corpus/controls, `--persist-generations`,
`p3-eval-v9`). It is the floor: anything the base already does is not a training gain.

Base NLL/accuracy for all three conditions are folded into the result tables above (the
`Base NLL (acc)` column). Training cut NLL 3–8× and raised accuracy 6–9 pp over the
untrained base on every family.

Base **model** (generation) paste-share vs gold (facts_present, BPE):

| Family | base model paste | gold paste | mean gen len |
| --- | ---: | ---: | ---: |
| enigma | 81.4% | 79.3% | 7,861 |
| isabelle | 83.2% | 82.1% | 8,030 |
| metamath | 86.2% | 93.0% | 5,708 |
| mizar | 85.5% | 65.5% | 7,910 |
| prf2 | 80.3% | 81.3% | 7,799 |
| thproofs | 90.4% | 66.6% | 7,597 |

The untrained base already copies ~80–90% of its generated output from the prompt (on
mizar/thproofs it *over*-copies vs gold — regurgitating context and running to the
8192-token cap). So copying is available to *any* model, yet base scores far worse than
the trained arms while copying just as much. **The training gain is therefore not
explained by copy ability** — the strongest evidence so far against "it's just copying."
(This still does not localize the gain to reasoning vs better format/copy modeling; the
copy-controlled metric would.)

**Why base per-token accuracy looks high (e.g., metamath 91.8%).** It is *teacher-forced*
next-token accuracy, which is copy-dominated: base's accuracy tracks the gold BPE
copyability almost exactly (metamath 91.8% acc vs 93.0% copyable), i.e. it predicts the
copyable tokens and misses most of the rest. It is **not** proof-solving — base's *free*
generation degenerately repeats the `GOAL` line (exact match 0%). Verified: the base run
loaded the pinned untrained weights (`model.safetensors` SHA `88c1425578…`, matching the
training configs), and accuracy is computed teacher-forced over target (proof) tokens
only. The trained arms score *above* copyability (metamath 97.8–98.1%), which is the part
that is not attributable to copying.

## Training-corpus premise exposure (memorization budget)

How often does the model see each premise during the 13-epoch run? A premise's
**exposure** = (number of training rows that cite it in the fact block) × 13
epochs. Computed over all 181,652 v3 training rows
(`corpus-v3/shards/*.jsonl`, held-out rows excluded; identity = premise name;
`local_inputs` typing lemmas excluded, matching the `DATASET-DESIGN.md` "fact
uses" definition).

![Premise exposure histogram: distribution over unique premises (left) and cumulative share of premise-uses by exposure threshold (right)](figures/premise-exposure-histogram.png)

- 120,377 unique premises across 1,517,520 premise-uses; the **median premise is
  seen just 26 times** (2 uses × 13), mean 164, max 239,226 — a heavy-tailed
  distribution where a small head carries most citations.

| Exposure threshold | Premises | Share of premises | Share of premise-uses |
| --- | ---: | ---: | ---: |
| ≥ 80 | 23,267 | 19.33% | 86.53% |
| ≥ 100 | 20,566 | 17.08% | 85.28% |
| ≥ 200 | 10,531 | 8.75% | 78.28% |
| ≥ 500 | 4,573 | 3.80% | 68.94% |

At the Allen-Zhu & Li (2024) memorization threshold of ~100 exposures, only
**17.1% of unique premises saturate, yet they account for 85.3% of all
premise-uses** — so the dense arm can plausibly memorize the premises behind most
citations, while the long tail (the other ~83% of premises, below the threshold)
is exactly the regime the supplied fact block is meant to carry. (An earlier v2
build reported 16.38% / 84.30% at ≥100; the v3 rebuild used here is 17.08% /
85.28%.)

Reproduce: `python3 scripts/premise_exposure_hist.py` (writes
`figures/premise-exposure-histogram.png` and `figures/premise-exposure-stats.json`).

## Interpretation

**Copyability caveat (BPE):** under `facts_present`, gold paste-share is **66–93%
across *all* families** (not just metamath), so the per-token NLL/accuracy figures
largely reflect copyable content and are a weak proxy for end-to-end reasoning. The
paired split−dense *delta* stays valid (copyable tokens are shared between arms and
cancel), but absolute "the model reasons" claims need a copy-controlled metric (score
only non-copyable tokens) or sound proof verification — which the current suite
provides only for metamath.

1. **Headline (`facts_present`) supports the hypothesis.** When both arms can read
   the correct facts, masking those facts out of the training loss (split) does
   **not** cost reasoning quality — it modestly improves it across all six
   families (NLL and accuracy CIs all exclude 0), and yields more exact and
   Metamath-verified proofs on metamath (+3.0–3.2 pp). Consistent with the idea
   that not spending capacity to memorize fact content leaves more for reasoning.
2. **Mechanism confirmed.** Split degrades more than dense when facts are absent
   or corrupted — clearest on the fact-heavy families **metamath** and **mizar**.
   Dense stored fact content in its weights (leans on them less-visibly); split
   offloaded facts to context (leans on them). The training manipulation did what
   it was designed to do.
3. **Magnitudes are small but tight.** Differences are hundredths of a nat and
   sub-1% accuracy on the headline, but per-example pairing plus large n give
   narrow CIs that exclude 0. This is a difference in *degree*, not a regime
   change.

## Scope / caveats

- Descriptive paired estimates **conditional on the two seed-42 checkpoints**.
  No equivalence or non-inferiority claim — no margin has been approved. One
  training seed per arm.
- Diagnostics (`facts_absent`, `facts_corrupted`) are 10% per-family samples;
  small families (thproofs n=5) are noisy.
- A `base` (untrained Qwen2.5-0.5B) control arm has been smoke-validated but its
  full run is not yet in `rerun-01`; it would slot in as a third column.

## Reproduction

```bash
cd memorysplit-requery-exact/eval-rerun-01-compare
python3 /home/vs/AlphaAI/eduLLM/OLMo-core/src/scripts/train/p3_math_split/evals/compare_arms.py \
  --dense dense.json --split split.json \
  --dense-config dense_config.json --split-config split_config.json \
  --out comparison.json
```

## History: the earlier failed run (resolved)

An initial fleet failed before scoring any example: vLLM 0.19.1 loaded
`libcudart.so.12` while the L4 DLAMI shipped CUDA 13.x / torch cu130
(`ImportError: libcudart.so.12`). This was an environment/wheel mismatch, not a
checkpoint, corpus, or evaluator-data problem. Fixes (committed to OLMo-core):
an isolated `bootstrap_vllm_env.sh` (Python 3.12.13, `torch 2.10.0+cu128`,
`transformers 5.7.0`, `vllm 0.19.1`, pinned `edullm-data`), a `preflight_vllm.py`
that imports `vllm._C` and the tokenizer reader before fleet launch,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `_FAILED` publication on early startup
errors, and a comparison gate requiring both arms to share a generation backend.
The `rerun-01` results above were produced after those fixes landed.
