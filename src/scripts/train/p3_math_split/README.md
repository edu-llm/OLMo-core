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
replaying them, and its module docstring states exactly what that check does and does not
cover. `INCOMPATIBILITIES.md` in this directory records the OLMo-core/Qwen findings the port
had to work around.

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

# 2. tokenize: one token array, two label masks
python $P3/tokenize_corpus.py --corpus corpus --out tokenized --suggest   # pick a length
python $P3/tokenize_corpus.py --corpus corpus --out tokenized --sequence-length 1024
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

`--dry-run` on `train.py` prints the resolved plan (steps, total input tokens, grad accum,
epochs-equivalent) without touching a GPU.

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
not lead split on train-visible facts, the mask did not do what it was supposed to and any proof
difference is coming from somewhere else.

A proof counts as `valid` only if every step cites a supplied rule, is a genuine substitution
instance of it, has its hypotheses discharged by earlier steps, and the last step is the goal.
Exact match against the gold trace is reported separately — a different correct route should
count, or the metric measures imitation.

## Guardrails

Each fails loudly rather than producing a plausible wrong number.

| check | catches |
|---|---|
| `smoke_test.py` | dead gradients; the two arms training identically; mask semantics inverted |
| `src/test/nn/transformer/qwen_test.py` | the port is not Qwen2.5 (spurious o_proj bias, wrong RoPE, untied head) |
| `corpus_invariants_test.py` | held-out fact leaks, degenerate targets, block order leaking step order |
| `mask_alignment_test.py` | token-space off-by-one; masks identical; padding supervised |
| `configs_test.py` | the two arm configs drifted apart |
| `train.py` startup | seq length vs. data mismatch, uneven grad accumulation, warmup ≥ total steps |
| `compare_arms.py` | any control differing between the two finished runs — **exits non-zero** |
| `make_mutants.py` | the invariant suite itself not biting |

## Measured, not assumed

From `smoke_test.py` (36 checks, CPU) and a 3–4k-theorem corpus rehearsal:

| | measured |
|---|---|
| Qwen port vs HuggingFace | max abs logit diff **< 2e-4** |
| parameter count | **494,032,768**, tied (via `TransformerConfig.tie_word_embeddings`) |
| gradients, dense vs split | differ on **26/26** tensors |
| fact-token fraction | **38%** — above the 17–30% design target, inside the 5–60% gate |
| padding waste @ seq 1024 | **66%** — median example 292 tokens, p99 1438 |

Two of those are levers, not verdicts. A higher fact fraction means a *stronger* manipulation;
lower it with `--max-facts` or `--min-steps` if you want the stated band. The padding is the
price of one-example-per-sequence, which is what makes "identical documents in identical order"
checkable byte-for-byte; run `tokenize_corpus.py --suggest` on the full corpus before choosing.

Re-measure both on the full corpus — the numbers above come from a subsample.

All of `make checks` (isort, black, ruff, mypy) passes on the code added here.

## Held-out fact selection

Frequency-banded (`--heldout-min-freq 5`, `--heldout-max-freq 50`), and this is not cosmetic.
Citation frequency in `set.mm` is brutally skewed: on a 1.2k-example sample, `eqid` is cited by
12% of examples. Holding out one workhorse rule deletes a large slice of training data and
removes the most common inference patterns — nothing to do with the loss mask, but it would move
the result. `build_corpus.py` aborts if more than `--max-eval-share` of examples end up citing a
held-out fact.

`--max-theorems N` takes a seeded random subsample for a fast end-to-end rehearsal (~3 s for
3,000 theorems). It samples across the database rather than truncating, because `set.mm` is
alphabetical and a prefix would be a biased slice of mathematics.
