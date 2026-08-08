# corpus/

Built by the scripts in `../scripts/`, gated by `../tests/test_corpus_invariants.py`.
Every number below is measured from the files in this directory, not estimated.

## Layout

```
raw/        complete shards, no train/eval split   (rebuild input)
shards/     training examples                      (split arm reads these)
eval/       examples touching a held-out fact
heldout/    one manifest per FAMILY of shards
logs/       build logs
```

`raw/` is kept so the held-out draw can be redone without rebuilding: the
splitter is a pure function of `raw/` and the seed.

## What is in it

Tokens are Qwen 2.5 (`Qwen/Qwen2.5-0.5B`), the tokenizer this corpus is built for.
GPT-2 counts run ~17% higher and are not what the run will see.

| shard | examples | tokens | facts | avg target | novel 8-grams |
|---|---:|---:|---:|---:|---:|
| enigma | 26,837 | 199.4 M | 21,334 | 2,344 | 93% |
| prf2 | 24,350 | 197.0 M | 24,299 | 2,965 | 92% |
| metamath | 65,833 | 150.4 M | 44,784 | 80 | 53% |
| mizar | 45,626 | 37.1 M | 30,934 | 367 | 62% |
| isabelle | 78,094 | 28.4 M | 39,939 | 15 | 100% |
| thproofs | 17,576 | 14.6 M | 9,181 | 447 | 95% |
| **total** | **258,316** | **626.9 M** | **143,391** | | |

3,937 eval problems, 2,000 held-out facts. At the planned **13 epochs** that is
8.15 B tokens seen, where 19% of facts clear 80 exposures and 87% of premise uses
land on a fact seen 80+ times.

`name_echo` and `tiny_target` have been removed (27,587 Isabelle examples),
`goal_in_block` removed from the eval set only (53 problems), and the 18 examples
over 131,072 tokens dropped — they would be truncated mid-proof under any window
and one was 703,717 tokens. Everything removed
is in `corpus/dropped_raw/` and `corpus/dropped_eval/`; `filter_corpus.py
--restore` puts it back.

Tokens are counted with the GPT-2 tokenizer over a 1-in-37 sample, scaled by
bytes. Do not estimate them as bytes/2.2: compact TPTP runs at 1.46 bytes/token,
and that error understated the corpus by 156 M.

`enigma` draws on four E-prover runs — `mzr01`, `mzr03`, `mzr02`, `mzr08`, in that
order, which is also dedup priority. `mzr09` was measured and skipped: only 17% of
its proofs survive Jaccard dedup against the others, against 45-46% for `mzr02`
and `mzr08`. The two extra archives added 9,034 proofs and 84 M tokens while
introducing only 296 new facts, so they raise citations per fact rather than
lengthening the tail.

## Schema

One JSON object per line. `text` is what the model sees; `[mask_start, mask_end)`
is the prompt context before `---`, whose loss the split arm zeroes and the dense
arm keeps. For Metamath this span contains both the global fact block and a
separate `Local assumptions:` block.

```json
{"id": "9f2c1a4b7e03", "theorem": "XBOOLE_1:28",
 "facts": {"TARSKI:def_3": "for X, Y being set holds ( X c= Y iff ... )"},
 "cited": ["TARSKI:def_3"], "goal": "...", "target": "...",
 "text": "I know these mathematical statements:\n...\n---\nGOAL ...\n...",
 "mask_start": 0, "mask_end": 1183}
```

Metamath rows also carry a `local_assumptions` object. It contains only
theorem-local `$e` hypotheses actually pushed by the decoded proof. Those givens
are rendered before the unchanged `\n---\nGOAL` separator and omitted from the
target, as are internal `(reuse)` stack backreferences:

```json
{"theorem": "set:example", "facts": {"ax-mp": "|- ph & |- ( ph -> ps ) => |- ps"},
 "cited": ["ax-mp"], "local_assumptions": {"example.1": "|- ph"},
 "text": "I know these mathematical statements:\nax-mp : ...\nLocal assumptions:\nexample.1 : |- ph\n---\nGOAL |- ps\n  1  ax-mp ..."}
```

## Decisions that shaped the data

**Held-out is drawn per family, not per shard.** `mizar`+`thproofs` both cite the
MML and `prf2`+`enigma` both cite MPTP, so a fact withheld from one was being
trained on by its sibling. The first sweep caught 415 such facts. A fact counts as
held out only if nothing in the corpus trains on it.

**Both the citing proofs and the fact's own proof go to eval.** A fact's statement
is the goal of its own proof, so leaving that proof in training leaks the
statement the held-out condition is meant to remove.

**Fact blocks are shuffled** with a per-example seed. Listing premises in citation
order hands over the derivation sequence.

**One name denotes one statement, corpus-wide.** Enforced across all 148,923
facts. Three separate things had to be fixed to get there: `set.mm` and `iset.mm`
share theorem labels, E normalises the same definition differently per problem,
and prf2 writes spaced TPTP where ENIGMA writes it compact. Where prf2 and ENIGMA
genuinely disagree — different MML snapshots numbering generated Fraenkel terms
differently — the 31 affected examples are dropped rather than reconciled.

**Bookkeeping premises are dropped** from the ATP shards (`dt_`, `cc_`, `fc_`,
`rc_`, `redefinition_`, `fraenkel_`, `rq`, `spc`): 42% of prf2's citations, and
machine-generated typing conditions rather than mathematics.

**Isabelle keeps only paste share below 0.5** and drops `local.` premises. This
removes 96% of Magnushammer. What survives has 100% novel 8-grams; what it drops
is `by (simp add: assms(1) assms(2))`, which is the block echoed back.

**Unlabelled Mizar theorems count too.** In the MML the colon belongs to the
label — `theorem Th2: :: AFINSQ_1:2` against a bare `theorem :: AFINSQ_1:1` — and
two regexes demanded it unconditionally. That hid half the library: the fact
dictionary held 39,869 of 62,344 statements, so 38% of citations resolved to
nothing and the examples that made them were discarded as incomplete. Fixing it
took `mizar` from 16,576 examples to 45,626, cut its incomplete drops from 11,555
to 131, and raised its blocks from 3.1 to 4.3 facts each. `thproofs` shrinks in
return, because the html2 shard now covers theorems it used to supply alone.

**`proof` is a keyword, not a line.** It opens a block as often from the end of
the statement line as from its own line. Splitting on the line form lost 3,024
real proofs. Theorems discharged by a bare citation — `vars Non a = vars a by
Th99;` — are still dropped: the target would be the name of the one fact already
in the block.

**Metamath proofs are verified, not trusted.** Each compressed proof is replayed
and its final `|-` entry must equal the theorem's own statement. All 69,767 pass.
The source is pinned to `metamath/set.mm` commit
`82830c78861b96e906d9868c30c35dbd98be5db5`; exact file hashes are in
`metamath_sources.json`. That snapshot reproduces all 1,151 current eval targets
byte-for-byte and is required by deterministic evaluation.

## Two properties to know before training

**Fact exposure is head-heavy.** At 8 epochs:

| threshold | 8 ep | 12 ep | 20 ep |
|---|---:|---:|---:|
| facts over 80 exposures | 18,379 (12.0%) | 28,368 (18.4%) | 39,081 (25.4%) |
| citations landing on them | 82% | 86% | 89% |

41% of facts are cited exactly once and 61% twice or fewer, so the percentage of
facts that saturate stays low however many tokens are added — a new library
brings its own singleton tail. The number that matters is the second row: at 12
epochs, 86% of premise uses point at a fact the dense arm has seen 80+ times.
12 epochs over 744 M is 8.9 B tokens, near Chinchilla for 370M.

Adding proofs over the SAME fact set is the only way to move this. That is why
the extra ENIGMA archives were chosen for low Jaccard overlap rather than size.

**Single-token paste share reads high and means little here.** Metamath scores
0.94 because a proof step reuses `(`, `->` and `ph`; at 8-grams it is 53% novel
with an 18-token longest copied run. Judge targets by the n-gram column.

## Rebuild

**GATED ROUTING — not a complete all-family copy/paste recipe.** Follow
`../checklist.md` before rebuilding. For non-Isabelle families, builders first
stage complete rows under `corpus/raw/`; `scripts/split_heldout.py` performs the
later family split. Exact semantic MML-class/source commands remain pending the
shared repair, so none are presented here as executable production commands.

Isabelle must use the builder-local positive heldout path because the generic
splitter does not isolate whole trajectories:

```bash
python scripts/build_isabelle_shard.py --out corpus --heldout 500 --seed 20260801 --tokenizer-path tokenizers/qwen25-vendored
```

Its production outputs are
`corpus/shards/isabelle.jsonl`, `corpus/eval/isabelle.jsonl`, and
`corpus/heldout/isabelle.json`. Isabelle `--heldout 0` is debug/staging-only
and is not a scientifically complete split. Never pass Isabelle into the
generic family splitter.

Parser coverage is pinned separately in `tests/test_parsers.py`. The shard
invariants can only judge what was emitted; these check what was missed, which is
where both Mizar bugs lived.

## Quality flags

`corpus/flags/<shard>.jsonl` labels examples that may not be worth their tokens.
Nothing is removed; `scripts/filter_corpus.py --drop a,b` acts on chosen classes
and moves them to `corpus/dropped/` so `--restore` can undo it.

| flag | examples | tokens | what it is | act on it |
|---|---:|---:|---|---|
| `single_fact_short` | 66,725 | 26 M | one fact, target under 30 tokens | yes |
| `over_8k` | 27,009 | 418 M | longer than 8192 tokens | only under a small window |
| `over_4k` | 23,943 | 142 M | 4096–8192 tokens | only under a small window |
| `tiny_target` | 15,093 | 6 M | target under 12 tokens | yes |
| `name_echo` | 14,313 | 5 M | target is scaffolding plus names already in the block | yes |
| `high_copy` | 5,818 | 14 M | half the target's 16-grams appear in the prompt | **no — see below** |
| `goal_in_block` | 2,572 | 8 M | the goal appears verbatim inside a block fact | yes |
| `block_over_window` | 856 | 28 M | block alone exceeds 4096 tokens | only under a small window |
| `block_dominant` | 218 | 1 M | block is 90%+ of the example | yes |

**The size classes are a window decision, not a quality one.** Against a 32k
context only 1,058 examples (0.37%) overflow, holding 56.6 M tokens. The longest
single example is 703,717 tokens.

**`high_copy` is mostly a false positive and should not be acted on.** It fires
on Mizar and Metamath proof style rather than on copying. Mizar's flagged
examples have a median of 20 steps and none has fewer than three — they trip the
check because `::_thesis:` annotations restate the goal as the proof narrows it.
Metamath's have a median of 6 steps, with only 4% genuinely one-step. Judge these
two shards by n-gram novelty instead.

**`goal_in_block` is the one that matters most despite being the smallest.** All
2,572 were checked: in 1,469 the goal is character-identical to a whole fact in
the block (`set:con4` is handed `ax-3`, which is its own statement, and its proof
is one step citing it), and in the other 1,103 the goal is a proper sub-formula of
one block fact. In both the answer is sitting in the prompt, which is exactly how
the split arm could win for the wrong reason.

Cleaning costs little in tokens because degenerate examples are short:

| drop | examples left | tokens left |
|---|---:|---:|
| nothing | 285,908 | 745 M |
| `goal_in_block` + `name_echo` | 269,023 | 731 M |
| + `tiny_target` + `single_fact_short` | 210,169 | 708 M |
| + `block_dominant` | 210,139 | 707 M |

The last row removes 69% of `isabelle` (105,669 → 32,577) and almost nothing
else.

## Verify

```bash
python -m pytest tests/test_parsers.py -q            # parser coverage
python scripts/sweep_corpus.py --corpus corpus      # per-shard gate + cross-shard
python scripts/corpus_report.py --corpus corpus     # exposure and paste share
python scripts/measure_novelty.py --corpus corpus   # n-gram novelty
python scripts/audit_corpus.py --corpus corpus      # rewrite quality flags
```

The sweep must end `SWEEP CLEAN`. To confirm the gate still bites, corrupt a shard
with `scripts/make_mutants.py` and check every mutant fails — 6/6 are caught on
`metamath`, `isabelle` and `prf2`.
