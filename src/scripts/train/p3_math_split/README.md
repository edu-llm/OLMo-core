# P3 Math Split — dense vs. memory-split SFT on Qwen2.5-0.5B

Does a small model reason better when theorem facts come from an external store instead of
being learned through next-token supervision?

> When both models receive the same correct theorem statements in context, does a split model
> match or outperform a dense model on valid multi-step proofs?

Two arms, one manipulation:

| | fact block in context | loss on fact tokens | loss on goal + proof |
|---|---|---|---|
| **dense** | yes | **yes** | yes |
| **split** | yes | **no** | yes |

Held identical: initialization seed, architecture, tokenizer, input documents and their order,
batch size, optimizer, LR schedule, total input tokens, steps, and compute. The only difference
is which `label_mask_*.npy` the data loader reads.

Both arms read the **same** `tokens.npy`. That is a filesystem fact, not a claim about two
pipelines that agree — see `tokenize_corpus.py`.

The corpus is **Metamath** (`set.mm`), not Lean. `mm_verify.py` checks generated proofs by
replaying them; its module docstring states exactly what that check does and does not cover.
`INCOMPATIBILITIES.md` in this directory records the OLMo-core/Qwen findings the port had to
work around, with file:line citations.

## Budget

One GPU. A 0.5B model does not justify more.

| stage | time |
|---|---|
| bootstrap (torch, `set.mm`, HF download, gates) | 20–30 min |
| build corpus + tokenize (CPU) | ~2 min |
| **train, both arms** | **1.9–2.9 h** |
| export 2 checkpoints to HF | ~5 min |
| **eval** (2 arms × 4 conditions × 2 splits + probe) | **0.7–1.5 h** |

**≈6–8 h wall clock, ≈$6–10** on one `g5.xlarge` (1× A10G 24 GB, $1.006/hr), or ~$3–4 on spot.
Training is 130 PFLOPs per arm; the range above spans 20–30% MFU. Running each arm on its own
instance halves wall clock at the same total cost — `run_experiment.sh` is sequential on one
machine only because that removes one more thing that could differ between arms.

MFU and generation throughput are the two numbers not measured here (no GPU was available);
the first 50 steps will tell you the real MFU. Everything else below is computed from measured
data.

## Corpus size — read before choosing filters

`set.mm` has **47,663 provable theorems**, and one example is built per theorem, so **47,663 is
the hard ceiling**. At the default filters (3–40 steps, 2–8 cited facts, ≤6000 chars) the
measured keep rate is **40.6%**, giving **≈19,400 examples** across all splits — about 13,700
train instances at `sequence_length 1024`, or ~640 steps at 3 epochs.

Rejects are 28% step-count and 31% fact-count out of band. Widening `--min-steps`,
`--max-steps` and `--max-facts` moves toward the ceiling; going to ~45k roughly triples
training time (~7–9 h for both arms, ~$12–18 total). Decide before the run — it changes the
token budget you are holding constant.

## Run it

```bash
pip install -e '.[all]'          # transformers is needed for the tokenizer and eval

P3=src/scripts/train/p3_math_split
mkdir -p data && curl -sL --fail -o data/set.mm \
  https://raw.githubusercontent.com/metamath/set.mm/develop/set.mm

# 0. plumbing check: gradients move, and the two arms differ
python $P3/smoke_test.py

# 1. corpus, then the invariant gate
python $P3/build_corpus.py --db data/set.mm --out corpus
SHARD_PATH=corpus/train.jsonl HELDOUT_PATH=corpus/heldout.json \
  pytest -q src/test/scripts/p3_math_split/corpus_invariants_test.py

# ...and prove that gate bites: six corrupted shards must every one turn it red
python $P3/make_mutants.py --shard corpus/train.jsonl --out corpus/mutants
for m in corpus/mutants/*.jsonl; do
  out=$(SHARD_PATH="$m" HELDOUT_PATH=corpus/heldout.json \
        pytest -q src/test/scripts/p3_math_split/corpus_invariants_test.py 2>&1 | tail -1)
  case "$out" in
    *failed*) echo "  $(basename "$m" .jsonl)  red (correct)" ;;
    *)        echo "  $(basename "$m" .jsonl)  GREEN - THE GATE IS BROKEN"; exit 1 ;;
  esac
done

# 2. tokenize: one token array, two label masks
python $P3/tokenize_corpus.py --corpus corpus --out tokenized --suggest   # pick a length
for split in train eval_retrieval eval_iid; do
  python $P3/tokenize_corpus.py --corpus corpus --out tokenized \
    --split $split --sequence-length 1024
done
TOKENIZED_DIR=tokenized pytest -q src/test/scripts/p3_math_split/mask_alignment_test.py

# 3. train both arms
for arm in dense split; do
  torchrun --nproc-per-node=1 $P3/train.py \
    --arm $arm --config $P3/configs/$arm.yaml --data-dir tokenized --save-folder runs/$arm
done

# 4. evaluate and compare
for arm in dense split; do
  python $P3/export_checkpoint.py --run runs/$arm
  python $P3/run_eval.py --model runs/$arm/hf --arm $arm --corpus corpus \
    --split eval_retrieval --db data/set.mm --out results/${arm}_retrieval.json --probe
done
python $P3/compare_arms.py \
  --dense results/dense_retrieval.json --split results/split_retrieval.json \
  --dense-run runs/dense --split-run runs/split
```

`train.py --dry-run` prints the resolved plan (steps, total input tokens, grad accum,
epochs-equivalent) without touching a GPU. Repeat step 4 with `--split eval_iid` for the
control.

`build_corpus.py --max-theorems N` takes a seeded random subsample for a fast end-to-end
rehearsal (~3 s for 3,000 theorems). It samples across the database rather than truncating,
because `set.mm` is alphabetical and a prefix would be a biased slice of mathematics.

## Configuration

`configs/dense.yaml` and `configs/split.yaml` differ in exactly one line, and
`configs_test.py` enforces that. Everything else lives under a shared block.

`rank_microbatch_size_sequences` is **4**, not 8. The `default` LM loss materializes the full
logits tensor (`nn/lm_head.py:252`), and at Qwen's 151,936 vocab an 8,192-token microbatch
needs ~10 GB for logits, the cross-entropy upcast, and the gradient — on top of ~6 GB of model
and optimizer state and ~3.5 GB of activations, which does not fit comfortably in 24 GB. At
4,096 tokens the budget is ~12.7 GB. Global batch stays 64 sequences, so this only moves
gradient accumulation (8 → 16) and costs nothing scientifically. With liger-kernel installed,
`loss_implementation="fused_linear"` skips the logits tensor entirely and 8 fits.

## What the eval measures

`eval_retrieval` is the headline: every example cites at least one fact **neither arm was
supervised on**, so a correct proof requires reading the block. `eval_iid` is the matched
control.

Four conditions per split, because a bare win is ambiguous:

- **facts_present** — the real setting.
- **facts_absent** — header, no statements. Isolates what each arm stored in weights.
- **facts_corrupted** — names kept, statements swapped. A model that reads its context should
  collapse; one reciting from memory will not. This is how you tell them apart.
- **facts_shuffled** — order permuted; should be a no-op. If it is not, the model keys on
  position and every other number needs re-reading.

Plus a **fact-recall probe** (`--probe`): given a fact name alone, state the fact. If dense does
not lead split on train-visible facts, the mask did not do what it was supposed to and any
proof difference is coming from somewhere else.

A proof counts as `valid` only if every step cites a supplied rule, is a genuine substitution
instance of it, has its hypotheses discharged by earlier steps, and the last step is the goal.
Exact match against the gold trace is reported separately — a different correct route should
count, or the metric measures imitation.

## Held-out fact selection

Frequency-banded (`--heldout-min-freq 5`, `--heldout-max-freq 50`), and this is not cosmetic.
Citation frequency in `set.mm` is brutally skewed: on a 1.2k-example sample, `eqid` is cited by
12% of examples. Holding out one workhorse rule deletes a large slice of training data and
removes the most common inference patterns — nothing to do with the loss mask, but it would
move the result. `build_corpus.py` aborts if more than `--max-eval-share` (default 30%) of
examples end up citing a held-out fact.

## Guardrails

Each fails loudly rather than producing a plausible wrong number.

| check | catches |
|---|---|
| `smoke_test.py` | dead gradients; the two arms training identically; mask semantics inverted |
| `src/test/nn/transformer/qwen_test.py` | the port is not Qwen2.5 (spurious o_proj bias, wrong RoPE, untied head) |
| `corpus_invariants_test.py` | held-out fact leaks, degenerate targets, block order leaking step order |
| `make_mutants.py` | the invariant suite itself not biting |
| `mask_alignment_test.py` | token-space off-by-one; masks identical; padding supervised |
| `configs_test.py` | the two arm configs drifted apart |
| `train.py` startup | seq length vs. data mismatch, uneven grad accumulation, warmup ≥ total steps |
| `compare_arms.py` | any control differing between the two finished runs — **exits non-zero** |

While training, watch `train/supervised token fraction`: ~1.0 for dense, ~0.7–0.85 for split.
**If they are equal the mask is not being applied and the run is invalid.** Check that before
reading anything into the loss curves — the fixed divisor makes the raw loss values
non-comparable by eye between arms.

## Measured, not assumed

From `smoke_test.py` (36 checks, CPU), `qwen_test.py` (10 checks), and a 4,000-theorem corpus
rehearsal:

| | measured |
|---|---|
| Qwen port vs HuggingFace | max abs logit diff **< 2e-4** |
| parameter count | **494,032,768**, tied via `TransformerConfig.tie_word_embeddings` |
| gradients, dense vs split | differ on **26/26** tensors |
| corpus keep rate | **40.6%** of 47,663 theorems → ≈19,400 examples |
| fact-token fraction | **38%** — above the 17–30% design target, inside the 5–60% gate |
| padding waste @ seq 1024 | **66%** — median example 292 tokens, p99 1438 |

68 tests pass with a pilot corpus; 41 pass and 27 skip cleanly without one. isort, black, ruff
and mypy all pass on the code added here.

Two of those numbers are levers, not verdicts. A higher fact fraction means a *stronger*
manipulation; lower it with `--max-facts` or `--min-steps` if you want the stated band. The
padding is the price of one-example-per-sequence, which is what makes "identical documents in
identical order" checkable byte-for-byte; run `tokenize_corpus.py --suggest` on the full corpus
before choosing a length.

Re-measure both on the full corpus — the table above comes from a subsample.
