# Corpus build plan — four machines, one shared contract

**Status:** reference implementation built and verified for Mizar. Three jobs remain.
**Verified:** the builder in §10.2 produced a shard that passes all 12 invariants in
§10.1, and each invariant was proven to fail on a deliberately corrupted shard
(mutation test, §7). Every file this plan needs is inlined in §10 — nothing external
is required.

---

---

## 0. Quickstart

**For the sample shard, use `setup_sample.sh` — it needs nothing else.** That script
carries the five Python files inline, so it runs in an empty directory with no copy of
this document present:

```bash
./setup_sample.sh 300     # ~10 s: writes the toolchain, fetches set.mm, builds, gates
```

It exits non-zero if any gate fails. Verified by sabotage: neutering the asserts makes
it report `GATE IS BROKEN` and exit 1.

**For the full four-corpus build, extract the toolchain from §10 of this document.** It
is the whole thing; drop it in an empty directory and run:

```bash
python3 - <<'EOF'
import re, pathlib
plan = pathlib.Path("CORPUS_BUILD_PLAN.md").read_text()
pat = re.compile(r"^### 10\.\d+ `([^`]+\.py)`\n(.*?)```python\n(.*?)\n```", re.S | re.M)
for m in pat.finditer(plan):
    dest = pathlib.Path(m.group(1)); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(m.group(3) + "\n"); print("wrote", dest)
EOF
```

That writes six files: `scripts/mm_expand.py`, `scripts/build_metamath_sample.py`,
`scripts/build_mizar_shard.py`, `scripts/make_mutants.py`, `scripts/inspect_shard.py`
and `tests/test_corpus_invariants.py`.

Then a cold-start smoke run on Metamath — one 51 MB download, about 30 seconds total:

```bash
python3 -m pip install pytest
mkdir -p data shards
curl -sL -o data/set.mm https://raw.githubusercontent.com/metamath/set.mm/82830c78861b96e906d9868c30c35dbd98be5db5/set.mm

python3 scripts/build_metamath_sample.py --db data/set.mm --out shards --limit 300

SHARD_PATH=shards/metamath_sample.jsonl pytest tests/test_corpus_invariants.py -q
python3 scripts/inspect_shard.py --shard shards/metamath_sample.jsonl --samples 3
python3 scripts/make_mutants.py --shard shards/metamath_sample.jsonl --out shards/mutants
for m in shards/mutants/*.jsonl; do
  SHARD_PATH=$m HELDOUT_PATH=shards/heldout.json pytest tests/test_corpus_invariants.py -q \
    && { echo "MUTANT $m PASSED — the gate is broken"; exit 1; }
done
echo "gate verified: green on the shard, red on every mutant"
```

Expected on a clean run: 300 examples, 0 proofs failing to reduce, ~4.0 facts per
example, ~46% masked, 17 invariants passing with 2 skipped. `make_mutants` will note
that it wrote a 2-fact stub `heldout.json` — the sample builder skips the held-out split
deliberately, and the stub exists only so the mutation tests can exercise I2.

Verified cold-start on 2026-08-01: extraction, build, gate and mutants all clean in a
directory containing nothing but this file.


## 1. Objective

Produce a masked/unmasked training corpus for the split-versus-dense experiment. Every
example is:

```
I know these mathematical statements:
<name> : <statement>          <- the masked span, one line per cited fact
---
GOAL <the proposition to derive>
<the derivation>
```

The fact block must be an **oracle**: every fact the derivation cites is present with
its full statement. Examples where the block cannot be completed are dropped, not
rendered partially — a partial block silently tests imperfect retrieval in a design
premised on perfect retrieval.

### Acceptance criteria

A shard ships only if it passes all **eighteen** tests in §10.1, grouped into eleven
invariants (the group is the idea; the count is the number of assertions):

| group | tests | Invariant | Why it exists |
|---|---|---|
| I1 | 3 | Every example has a non-empty block; every fact has a statement; every cited name is in the block | A missing statement turns an oracle into a noisy retriever |
| I2 | 3 | No training example cites a held-out fact, **is the proof of** a held-out fact, or contains its statement | 96.4% of facts are proved in-corpus, so the goal line is a second leak path |
| I3 | 1 | One name denotes exactly one statement corpus-wide | Otherwise the split arm has nothing stable to key on |
| I4 | 2 | Targets are non-empty, differ from the goal, and no single string exceeds 5% of the shard | 41% of LeanDojo's targets were the literal `no goals` |
| I5 | 3 | Exactly one mask span, covering the block and nothing else, at 5–60% of the text | The mask is the entire manipulation |
| I6 | 1 | No duplicate ids or byte-identical examples | A duplicate is seen twice per epoch |
| I7 | 1 | No eval example appears in train, by text **or by theorem** | The same result reached another way still leaks |
| I8 | 2 | No control characters, no U+FFFD, no truncated statements | A clipped statement makes the block a bad oracle |
| I9 | 1 | Goals are non-degenerate | Tested for targets; goals were not |
| I10 | 1 | The fact block is **not** in citation order | Otherwise the block hands over the step sequence for free |
| I11 | 1 | The held-out manifest matches the shared SHA-256 | Four machines must mask the same 500 facts |

---

## 2. Data sources

**Use exactly these six sources. Do not add LeanDojo, IsarStep, or any other corpus.**


| Source | Download | Extracted | Where | Compatibility |
|---|---|---|---|---|
| Metamath `set.mm`, `iset.mm`, `nf.mm` | 0.07 GB | 0.07 GB | `github.com/metamath/set.mm` (CC0) | independent |
| Legacy Mizar `html2.tar.gz` | 0.02 GB | 0.16 GB | `grid01.ciirc.cvut.cz/~mptp/nn_conj20/datasets/` | `nn_conj20`; legacy parser validation only, never combine with 8.1.15 thproofs |
| Current Mizar semantic HTML + MML + `thproofs.tar.gz` | 0.22 GB | 0.56 GB | Mizar/MPTP 8.1.15 / MML 5.94.1493 | recovered compatible triplet; requires verified semantic index |
| MPTP `prf2.tar.gz` | 0.05 GB | 0.66 GB | `grid01.ciirc.cvut.cz/~mptp/nn_conj20/datasets/` | pair only with its own `nn_conj20` source identities |
| ENIGMA `mzr01` + `mzr03` | 0.60 GB | 8.2 GB | `grid01.ciirc.cvut.cz/~mptp/enig1/` (HTTP only) | source identities must be verified independently |
| Magnushammer `all_data.json` | 2.33 GB | 2.33 GB | HF `Simontwice/premise_selection_in_isabelle` (gated, Apache-2.0) | independent |
| **Total** | **3.10 GB** | **11.5 GB** | ~6 GB if ENIGMA is filtered during extraction | |

**Mizar source compatibility blocker.** nn_conj20 html2 is incompatible with MML 8.1.15 thproofs;
these archives must not be combined. Current MML 8.1.15 semantic HTML and plain sources are recovered
and pinned with thproofs by `manifests/mizar-8.1.15_5.94.1493.json`
(`mizar-current-sources-v1`). The generated SQLite adapter is
`mizar-semantic-index-v1`; legacy `html2` can no longer authorize production
thproof output. Licensing notices are recorded in the manifest, but
redistribution rights remain unresolved and require legal review.

Exact recovered source pins:

- MML archive `cfc32c3e05d5d93c595934e26d4d3b4e399f95a75da7df08359eb9ee73ae6e2e`;
  tree `3d1af5b3e840aca5631541b42510b35c1b15dfa988af70ce463f58c899e88714`.
- Semantic HTML archive `e988481577e4e5cc25a5c96c4e86a7de612447088b20781a2680b0e6fc974eee`;
  tree `1f725c9943aeee2c21c6fe63484bc00336bdc442ec454ccfc810032d7de12781`.
- thproofs archive `665b17fea382d23168998a4bd1fd91736baf59c1fa3927f8c656d9886fdc3433`;
  tree `fce0eda226231de221ff2e7b3c9fa0699ec259d3e647e53eb9589b181dbf7877`.

Build the verified semantic index first, then the active builder (each command is
one line; use a fresh output path):

```bash
PYTHONPATH=scripts python scripts/mizar_current_index.py --manifest manifests/mizar-8.1.15_5.94.1493.json --mml /tmp/p3-source-audit/extract-mizar/mml --html /tmp/p3-source-audit/extract-html-current/html --thproofs /tmp/p3-source-audit/extract-thproofs/thproofs --sqlite /tmp/mizar-current-8.1.15.sqlite --jsonl /tmp/mizar-current-8.1.15.jsonl --report /tmp/mizar-current-8.1.15.report.json --mizar-archive /tmp/p3-sources/mizar-8.1.15_5.94.1493-i386-linux.tar --html-archive /tmp/p3-sources/html-abstr-8.1.15_5.94.1493.tar.gz --thproofs-archive /tmp/p3-sources/thproofs-8.1.15_5.94.1493.tar.gz
PYTHONPATH=scripts python scripts/build_thproofs_shard.py --semantic-index /tmp/mizar-current-8.1.15.sqlite --source-manifest manifests/mizar-8.1.15_5.94.1493.json --mml-root /tmp/p3-source-audit/extract-mizar/mml --html-root /tmp/p3-source-audit/extract-html-current/html --src /tmp/p3-source-audit/extract-thproofs/thproofs --mizar-archive /tmp/p3-sources/mizar-8.1.15_5.94.1493-i386-linux.tar --html-archive /tmp/p3-sources/html-abstr-8.1.15_5.94.1493.tar.gz --thproofs-archive /tmp/p3-sources/thproofs-8.1.15_5.94.1493.tar.gz --exclude "" --out /tmp/mizar-current-candidate
```

The isolated current-source measurement produced 50,752 accepted candidates
from 58,658 eligible complete proofs (86.522%); the immutable production floors
are 45,000 and 80%. The verified index hashes were SQLite
`8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835`
and JSONL
`1924067218e6875737260dda35e166d13aadc6c93261ead0d55c132cc3ee789a`.

**Disk budget.** Downloads 3.85 GB, extracted intermediates 20.8 GB (6 GB if ENIGMA is
filtered as it untars), final shards ~3.9 GB of JSONL. Peak usage is ~28 GB if nothing
is cleaned up, ~14 GB with ENIGMA filtered, and ~3.9 GB once intermediates are deleted.

JSONL runs **2.12× the text payload** because `facts`, `goal` and `target` repeat what
is already in `text` — measured 27.8 MB of text becoming 58.8 MB on disk. Keep the
redundancy while building; it makes the invariant tests readable. Drop the derived
fields before training and reconstruct from `text` plus the mask offsets to halve the
footprint.

ENIGMA expands ~12× because 76% of its files are failed proof attempts with no
derivation. Filter to files containing `SZS output start` as you untar.

---

## 3. Measured yields

Rendered, complete blocks only. **These are lower than raw-corpus byte counts** — the
raw figures include definitions, registrations and schemes that never become examples.

| Shard | Examples | Tokens | Rendering |
|---|---|---|---|
| Mizar html2 | 17,054 | **13 M** | theorem-level (measured) |
| Mizar thproofs | ~30,000 est. | ~25 M est. | theorem-level; **needs measuring** |
| Metamath set+iset+nf | 69,767 | ~101 M | theorem-level with full derivations (measured) |
| MPTP prf2 | 25,059 | ~299 M | one example per proof (measured) |
| ENIGMA `mzr01`+`mzr03`, deduped | ~18,400 | ~139 M | alternative derivations of theorems prf2 already covers |
| Magnushammer, paste < 0.5 | 456,083 | ~202 M | one per transition (measured) |
| **Total** | | **~880 M** | |

At **8 epochs** that is ~7.0 B tokens seen, close to Chinchilla-optimal for 370M
(~7.4 B), saturating the facts behind roughly 87–93% of premise uses.

---

## 4. Heldout architecture — historical rationale and current status

> **NON-EXECUTABLE / HISTORICAL.** Earlier drafts drew a Mizar-only tail,
> wrote a shared heldout file, and described merging independent family draws.
> That workflow is not authoritative and must not be used for a rebuild.

The surviving rationale is semantic: Mizar, thproofs, prf2, and ENIGMA can
refer to the same MML fact under different naming conventions. They therefore
require the pending authoritative semantic-class splitter before production
train/eval outputs can be approved. The pooled-versus-strata policy remains
unresolved; this plan deliberately does not choose one.

Metamath remains a separate fact namespace and split. Positive-heldout Isabelle
also remains separate because its builder performs family-local,
whole-trajectory isolation. Neither should be folded into the pending MML-class
decision.

For every eventual semantic heldout, citing rows and the held fact's own proof
must both be excluded from training. This is an invariant for the pending
splitter, not an executable workflow in this historical section.

---

## 5. Single-machine pipeline

Target hardware, measured: **12 cores, 8 GB RAM (6 available), 1 GB/s disk.** All four
jobs run concurrently on this box. Peak memory is not a constraint — measured 118 MB for
the Mizar build, 387 MB for Metamath, 41 MB for Magnushammer streaming — so four
processes together stay under 2 GB.

**The GPU does not help and should stay idle.** Every stage is regex parsing, JSON
serialisation and set arithmetic. There is no matrix work to offload, and fast
tokenisers are CPU-bound Rust. Save the RTX 5050 for training.

### Runtime

| phase | bound by | time |
|---|---|---|
| 1 — download 3.10 GB | network | 4–20 min depending on link |
| 2 — extract 11.5 GB | gzip, CPU | 3–5 min (parallel `tar`) |
| 3 — four build jobs | CPU, 12 cores | ~5 min wall (longest job) |
| 4 — gate, inspect, merge | CPU | ~2 min |
| **total** | | **~15–30 min** |

Measured components: Mizar build 6.8 s, Metamath sample 4.3 s, full Metamath
verification 25 s, ENIGMA scan 21 s per archive, Magnushammer full stream ~60 s.

### Orchestration

> **NON-EXECUTABLE / HISTORICAL.** The former phase 1–4 shell workflow has
> been removed because its source assumptions, family holdout semantics, and
> output layout are no longer authoritative. Do not reconstruct commands from
> the surrounding historical measurements.

Use the gated rebuild routing in `corpus/README.md` and the live repair gates in
`checklist.md`. For non-Isabelle families, the current architecture is complete
builder output under `corpus/raw/` followed by `scripts/split_heldout.py`.
Exact semantic MML-class/source commands are still pending shared repair, so
there is deliberately no copy/paste family invocation here.

Isabelle is the exception: its positive builder-local heldout already performs
whole-trajectory isolation and must not enter the generic splitter. The only
current Isabelle copy/paste command is:

```bash
python scripts/build_isabelle_shard.py --out "$OUT" --heldout 500 --seed 20260801 --tokenizer-path tokenizers/qwen25-vendored
```

It writes `$OUT/shards/isabelle.jsonl`, `$OUT/eval/isabelle.jsonl`, and
`$OUT/heldout/isabelle.json`.

### Intra-job parallelism

Each build is embarrassingly parallel over source files. With 12 cores:

```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=10) as ex:      # leave 2 for the OS
    for stmt, local in ex.map(parse_article, files, chunksize=16):
        index.update(stmt)
```

Worth it for ENIGMA (57,880 files per archive) and prf2 (25,060). Not worth it for
Metamath, where a single `set.mm` parse dominates and cannot be split.

### The four jobs

**Metamath** — `set.mm`, `iset.mm`, `nf.mm`, ~101 M tokens. **Verify while expanding:**
the final `|-` entry of each trace must equal the theorem's statement; 69,792 of 69,800
pass in 25 s, drop the 8 that do not. **Render facts with their hypotheses** —
`syl : |- ( ph -> ps ) & |- ( ps -> ch ) => |- ( ph -> ch )`, not the conclusion alone,
since 57.0% of cited facts carry `$e` hypotheses.

**Mizar** — current 8.1.15 semantic HTML, plain MML, and thproofs are a verified
source-compatible triplet behind `mizar-semantic-index-v1`. Legacy nnconj20
`html2` remains separate and nonproduction. The builder in §10.2 is retained
only for legacy parser validation. **Do not use step-level rendering:** the fact block inflates
10.34× while targets grow 1.20×, pushing the block to 79.5% of each example. For
`thproofs`, take the article from the filename (`t3_polyred` → `POLYRED`) since those
files carry no header.

**ATP** — `prf2` + ENIGMA `mzr01`/`mzr03`, ~438 M tokens. One parser for both. **Drop
bookkeeping premises** (`dt_`, `cc_`, `fc_`, `rc_`, `redefinition_`, `fraenkel_`, `rq`,
`spc`) — 62.7% of prf2's citations, 34.6% of ENIGMA's. **Deduplicate ENIGMA**: keep one
proof per theorem plus alternatives with clause-set Jaccard below 0.5.

**Isabelle** — Magnushammer, ~202 M tokens. Stream with `ijson`. **Drop `local.`
premises** (93.0% of the unqualified bucket) but keep `List.`, `Matrix.`, `axioms.`.
**Keep only transitions with paste share below 0.5** — 456,083 of 2,274,308.

## 6. Shard schema

One JSON object per line:

```json
{
  "id": "9f2c1a4b7e03",
  "theorem": "XBOOLE_1:28",
  "facts": {"TARSKI:def_3": "for X, Y being set holds ( X c= Y iff ... )"},
  "cited": ["TARSKI:def_3"],
  "goal": "X /\\ (Y \\/ Z) = (X /\\ Y) \\/ (X /\\ Z)",
  "target": "...the derivation...",
  "text": "I know these mathematical statements:\n...\n---\nGOAL ...\n...",
  "mask_start": 0,
  "mask_end": 214
}
```

`mask_start`/`mask_end` are character offsets into `text`, kept for the invariant tests.

**Do not build the training mask from those offsets.** Character offsets have to be
mapped through the tokenizer's `offset_mapping`, and a BPE merge can straddle the block
boundary — the last character of the block and the first of the separator can land in
one token, which then has no correct mask value. Tokenise the two spans separately and
concatenate instead, so the mask is correct by construction:

```python
block_ids = tok.encode(rec["text"][:rec["mask_end"]])   # or rebuild from rec["facts"]
body_ids  = tok.encode(rec["text"][rec["mask_end"]:])
input_ids = block_ids + body_ids
loss_mask = [0] * len(block_ids) + [1] * len(body_ids)  # split arm
```

This is why `facts`, `goal` and `target` are kept alongside `text` rather than dropped
to save disk. They cost 2.12× the payload and they remove an entire class of boundary
bug from the training integration.

---

## 7. The TDD gate — run on every machine, per shard

The governing rule: **a test that has never failed proves nothing.** Before trusting a
green run, corrupt the shard and confirm the suite goes red. §7A gives the full loop.

I10 and I6 were added after the first build passed twelve tests: 100% of multi-fact
blocks were emitted in citation order, and two examples were byte-identical. Neither
was visible in any aggregate. The builder now shuffles the block under a per-example
seed and drops duplicate texts.

**NON-EXECUTABLE / HISTORICAL example.** These flat Mizar paths predate the
current raw-to-semantic-split output contract.
<!-- NON-EXECUTABLE / HISTORICAL -->
```bash
# 1. green on the real shard
SHARD_PATH=$OUT/mizar.jsonl HELDOUT_PATH=$SHARED/heldout.json \
  pytest tests/test_corpus_invariants.py -q

# 2. red on mutants — required, not optional
python scripts/make_mutants.py --shard $OUT/mizar.jsonl --out $OUT/mutants
for m in $OUT/mutants/*.jsonl; do
  SHARD_PATH=$m HELDOUT_PATH=$SHARED/heldout.json \
    pytest tests/test_corpus_invariants.py -q  # each MUST fail
done
```

Verified mutation results on the reference shard:

| mutant | defect injected | invariant that caught it |
|---|---|---|
| `m1_empty_stmt` | one fact left without a statement | I1 — 2 failed |
| `m2_heldout_cited` | training example cites a held-out fact | I2 — 3 failed |
| `m3b_name_clash` | one name given two statements | I3 — 1 failed |
| `m4_degenerate_target` | target set equal to the goal | I4 — 1 failed |
| `m5_bad_mask` | mask truncated mid-block | I5 — 1 failed |
| `m6_heldout_proof` | proof of a held-out fact left in training | I2 — 1 failed |

I6–I11 were themselves discovered red: the first build failed `test_no_duplicate_examples`
and `test_fact_block_order_does_not_leak_the_proof` before the builder was corrected.

**A caught trap:** the first name-clash mutant passed. The cause was the mutation, not
the test — it altered a fact appearing in only one example, so no clash existed. Mutate
a *frequently occurring* fact (`TARSKI:def_1`, 223 occurrences) and I3 fires. When a
mutant fails to trigger, check the mutation before doubting the invariant.

---

## 7A. The TDD loop, adapted for data extraction

The code-oriented loop in `tdd-loop/SKILL.md` transfers to extraction with one
substitution: the unit under test is a **shard**, not a function, and the acceptance
criteria are the acceptance criteria in §1. The non-negotiable gate is unchanged — *you may not
write an extractor until you have watched a test fail.*

### Why a green suite is not evidence

An extractor bug and a broken test look identical from the outside: both produce a
passing run. Every invariant below exists because a real bug got through unnoticed
during the survey — a regex that silently returned zero definitions, a resolver whose
false positives inflated a store by 22%, a target field that was the constant string
`no goals` 41% of the time. None of those raised an error. They produced clean-looking
output that was wrong.

The only defence is to corrupt the shard deliberately and confirm the suite notices.

### The loop, per shard

**0. Setup (once per machine)**

Copy `heldout.json` from the shared location. Confirm `pytest` runs. Create a todo
list with: author invariant, watch it fail, write extractor, green on real, red on all
mutants.

**1. Plan — author the invariant and watch it fail**

State the acceptance criterion as a testable predicate over shard rows, then extend
`test_corpus_invariants.py`. Before writing any extractor code, run it against a
deliberately broken input and record the failure in your notes:

```bash
# a naive shard: 100 rows with empty fact blocks
python -c "
import json
print('\n'.join(json.dumps({'id':str(i),'theorem':'X:1','facts':{},'cited':[],
      'goal':'g','target':'t','text':'t','mask_start':0,'mask_end':0})
      for i in range(100)))" > /tmp/naive.jsonl

SHARD_PATH=/tmp/naive.jsonl HELDOUT_PATH=$SHARED/heldout.json \
  pytest tests/test_corpus_invariants.py -q     # MUST be red
```

If this is green, the invariant is not asserting anything. Fix the test before
continuing. **Do not proceed to step 2 without a recorded red.**

**2. Act — write the extractor**

Implement only what the invariant requires. Stay in scope: no extra fields, no
speculative filters. Do not weaken a test to make a shard pass — if an invariant is
genuinely wrong, change it deliberately in Plan with a stated reason, never silently
during Act.

**3. Check — green on real, red on every mutant**

```bash
python scripts/make_mutants.py --shard $OUT/shard.jsonl --out $OUT/mutants
SHARD_PATH=$OUT/shard.jsonl HELDOUT_PATH=$SHARED/heldout.json pytest -q   # green
for m in $OUT/mutants/*.jsonl; do
  SHARD_PATH=$m HELDOUT_PATH=$SHARED/heldout.json pytest -q               # each red
done
```

A mutant that stays green is a finding, not a formality. Diagnose it before shipping:
either the invariant is too weak, or the mutation is. Check the mutation first — this
is the common case, and it happened during development. The first name-clash mutant
altered a fact occurring in exactly one example, so no clash existed and the suite was
right to stay green. Mutating a frequent fact (`TARSKI:def_1`, 223 occurrences) made
I3 fire immediately.

**Escalation.** Two consecutive failed Checks means stop patching and re-plan with the
full failure history. Cap at roughly five attempts, then surface the blocker rather
than looping.

### Data-specific checks that no invariant can express

The invariant suite catches structural defects. It cannot catch a shard that is
structurally perfect and semantically wrong. Run `inspect_shard.py` (§10.4), which
automates four of these and exits non-zero on failure:

```bash
python scripts/inspect_shard.py --shard $OUT/shard.jsonl \
  --source-bytes <bytes of the raw source> \
  --candidates <source proofs considered>
```

| check | fails when | reference |
|---|---|---|
| 1 — print full examples | never fails; **you must read them** | every survey error was visible here and invisible in aggregates |
| 2 — rendered vs source bytes | ratio >5x (duplication) or <0.02x (silent dropping) | Mizar theorem-level 0.18x, step-level 0.89x |
| 3 — fact-block size histogram | >60% of rows share one size, or <3 distinct sizes | catches the file-dump defect that disqualified `mathlib4-state-change`, where every row in a file carried an identical 84-declaration block |
| 4 — drop rate | <1% (filter never ran) or >90% (over-rejecting) | Mizar drops 41.9% for incomplete blocks |
| 5 — paste share | mean >0.75 | Mizar 0.332, Metamath 0.503, LeanDojo 0.891 (dropped for this) |

Verified on the reference shard: all five pass (0.18x ratio, 22 distinct block sizes,
41.9% drop, 0.378 paste). Verified to fail on two semantic mutants — a shard given a
uniform 84-fact block trips check 3, and one whose target restates the goal trips
check 5. Both are structurally valid and pass all 12 invariants, which is exactly why
this second gate exists.

### Definition of done, per shard

Do not report a shard complete without all five:

1. A named invariant that failed before your extractor existed and passes now.
2. Green on the real shard.
3. Red on all six mutants.
4. Ten examples read by eye.
5. Row count, token count and drop rate recorded in the handoff note.

---

## 8. Merge and final verification

**NON-EXECUTABLE / HISTORICAL example.** Final composition is pending the
authoritative semantic-class splitter and current family output contracts.
<!-- NON-EXECUTABLE / HISTORICAL -->
```bash
cat $SHARED/shards/{metamath,mizar,atp,isabelle}.jsonl > $SHARED/corpus.jsonl
SHARD_PATH=$SHARED/corpus.jsonl HELDOUT_PATH=$SHARED/heldout.json \
  pytest tests/test_corpus_invariants.py -q
```

The merged run matters: **I3 is the only invariant that can fail on the union while
passing on every part.** If two shards give the same name different statements — likely
where Mizar and prf2 both reference MML facts under different conventions — it surfaces
only here.

Then confirm:
- held-out facts appear in `corpus.jsonl` zero times
- eval files total ~1,177 problems from Mizar, plus each other corpus's own draw
- total tokens ~880 M
- masked fraction per shard sits in 17–30%

---

## 9. Known limitations to record, not fix

- **Metamath names leak.** Facts sharing a four-character prefix have 0.640 statement
  overlap against 0.283 for random pairs; `mul`→`x.` lifts 18.7×. Accepted as realistic.
  Mizar's `ARTICLE:N` numbering does not leak, which is one reason to weight it.
- **Only cooldown A** is planned. Under A the split arm sees fewer supervised tokens, so
  a split *win* is strong evidence and a split *loss* is confounded with reduced
  supervision. Run B only if split loses.
- **Paste share varies by source**: Mizar 0.332, Metamath 0.503, Magnushammer filtered
  <0.5. The model copies part of every answer; this bounds how much reasoning is tested.
- **Proof validity is verified only for Metamath.** The invariant suite checks
  structure, isolation and non-degeneracy; it cannot tell a correct proof from a
  plausible-looking one. Per-corpus cost of closing that:

  | corpus | how | cost | in plan |
  |---|---|---|---|
  | Metamath | replay the compressed-proof stack machine, require each proof to reduce to its own statement | **25 s for all 69,792 theorems, 99.99% pass** | **yes — required** |
  | Mizar | invoke the Mizar verifier over the MML | hours, needs a Mizar install | no |
  | prf2 / ENIGMA | replay each refutation in E | minutes to hours per shard | no |
  | Magnushammer | invoke Isabelle | hours, needs a full AFP build | no |

  Metamath's check is free relative to a ~1 hour build and catches decoder bugs, which
  is the realistic failure mode — it found **8 theorems that do not reduce**; inspect
  and drop them. For the others, **extracted proofs are trusted because they come from
  verified libraries.** That assumption is sound for all six sources as specified and
  breaks the moment anything is generated rather than extracted, so re-open it if
  MetaGen or a prover run is ever added.
- **A custom tokenizer is required, not optional.** Measured on 25 Mizar articles,
  GPT-2's BPE uses **2.04× more tokens than there are Mizar symbols** (1,166,070 tokens
  for 571,101 symbols). `::_thesis:` costs 5 tokens, `c=` and `|-` cost 2 each. A BPE
  trained on the corpus approaches one token per symbol, halving sequence length for
  identical content — which halves attention cost and doubles effective context.
- **The perfect-retriever assumption removes premise search**, which is the component
  LMLM's retriever would have to perform. A positive result supports "given perfect
  retrieval, externalisation works for derivation" — the narrow claim, not the broad one.

---

## 9A. After extraction — training-time items to revisit

Out of scope for the build, but decided during design. Read this again before the
first training run:

1. **Run the memorisation probe first.** Train briefly, then present a fact name with
   no block and check whether the dense arm can produce the statement. The whole
   comparison assumes it can; confirm before spending the full budget.
2. **Fact-block dropout at a low rate.** Without it the no-store eval cell is
   out-of-distribution and measures distribution shift rather than fact access.
3. **Cooldown A only.** Run B only if split loses under A, since a split loss under A
   is confounded with reduced supervision.
4. **8 epochs**, giving ~7.0 B tokens seen against Chinchilla's ~7.4 B for 370M.
5. **Train the custom tokenizer before the corpus is frozen** — see §9.

---

## 9B. Worked output — hand this to the architecture team

A real record from `scripts/build_metamath_sample.py` (§10.5), which emits exactly the
production format. 500 examples build in ~13 s and pass the full gate.

**The record:**

```json
{
  "id": "36a8ac989bc2",
  "theorem": "0cxpd",
  "facts": {
    "syl2anc": "|- ( ph -> ps ) & |- ( ph -> ch ) & |- ( ( ps /\\ ch ) -> th ) => |- ( ph -> th )",
    "0cxp": "|- ( ( A e. CC /\\ A =/= 0 ) -> ( 0 ^c A ) = 0 )"
},
  "cited": ["0cxp", "syl2anc"],
  "goal": "|- ( ph -> ( 0 ^c A ) = 0 )",
  "target": "...the numbered derivation...",
  "text": "...the block, separator, goal and derivation concatenated...",
  "mask_start": 0,
  "mask_end": 183
}
```

**What `text` looks like — this is what the model sees:**

```
I know these mathematical statements:
syl2anc : |- ( ph -> ps ) & |- ( ph -> ch ) & |- ( ( ps /\ ch ) -> th ) => |- ( ph -> th )
0cxp : |- ( ( A e. CC /\ A =/= 0 ) -> ( 0 ^c A ) = 0 )
---
GOAL |- ( ph -> ( 0 ^c A ) = 0 )
  1  cxp0d.1      |- ( ph -> A e. CC )
  2  cxpefd.2     |- ( ph -> A =/= 0 )
  3  0cxp         |- ( ( A e. CC /\ A =/= 0 ) -> ( 0 ^c A ) = 0 )
  4  syl2anc      |- ( ph -> ( 0 ^c A ) = 0 )
```

**What the mask does.** `text[0:183]` is the fact block. The
split arm zeroes loss over exactly that span; the dense arm trains on everything. Both
arms see identical input tokens — only the loss weighting differs. Note the derivation
cites `0cxp` and `syl2anc` by name at steps 3 and 4, so the split arm still sees *which*
facts were used, just not *what they say*.

Note also that `syl2anc` is rendered with its hypotheses
(`|- ( ph -> ps ) & ... => |- ( ph -> th )`). Printing the conclusion alone would give
`|- ( ph -> th )`, which for an inference rule is meaningless — see Machine A in §5.

**This file is not the eval set.** It is a format demonstration: 500 training-shaped
examples with **no held-out split applied**. Do not train on it and do not evaluate on
it. The real files are `<corpus>.jsonl` (train) and `<corpus>_eval.jsonl` (eval), which
Machine A produces once it applies its own tail draw per §4.

**Sample statistics** (500 examples): 3.62 facts per example, 45.4% masked fraction,
0.27 MB of text in a 0.62 MB file (the 2.12× JSONL overhead), zero proofs failing to
reduce to their own statement.

**Smoke test for the loader:**

```bash
python scripts/build_metamath_sample.py --out ./shards --limit 500
SHARD_PATH=./shards/metamath_sample.jsonl pytest tests/test_corpus_invariants.py -q
python scripts/inspect_shard.py --shard ./shards/metamath_sample.jsonl --samples 5
```

The masked fraction here is 45.4% rather than the 17–30% design target because short
proofs cite proportionally more facts. It stays inside the 5–60% invariant band, but
expect the full Machine A shard to sit lower once long proofs are included.

---

## 10. Context — every referenced file, inlined

This plan is self-contained. The three files below are the complete implementation;
recreate them at the paths shown and nothing else is needed.

### 10.1 `tests/test_corpus_invariants.py`

```python
"""Invariants every corpus shard must satisfy before it enters training.

These are the acceptance criteria for the four extraction jobs. Each machine runs
this file against its own shard; a shard that fails any test does not ship.

The invariants exist because each one corresponds to a mistake already made during
the survey:

  I1 oracle completeness  — a fact block missing a cited statement silently turns a
                            perfect-retriever example into an imperfect one.
  I2 held-out isolation   — a held-out fact leaks if any training example cites it
                            OR if its own proof survives (96.4% of facts are proved
                            in-corpus, so the goal line is a second leak path).
  I3 name stability       — one name must denote one statement, or the split arm has
                            nothing stable to key on.
  I4 no degenerate targets— empty or unchanged targets teach nothing.
  I5 mask well-formedness — the masked span must be exactly the fact block.
"""

import json
import os
import re

import pytest

SHARD = os.environ.get("SHARD_PATH", "/tmp/dscount/shards/mizar.jsonl")
HELD = os.environ.get("HELDOUT_PATH", "/tmp/dscount/shards/heldout.json")
HDR = "I know these mathematical statements:"
SEP = "---"


def load(path, limit=None):
    if not os.path.exists(path):
        pytest.skip(f"shard not built yet: {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def rows():
    return load(SHARD, limit=200_000)


@pytest.fixture(scope="module")
def heldout():
    if not os.path.exists(HELD):
        pytest.skip(f"held-out set not built: {HELD}")
    return set(json.load(open(HELD))["facts"])


# ----------------------------------------------------------------- I1
def test_every_example_has_a_nonempty_fact_block(rows):
    bad = [r["id"] for r in rows if not r.get("facts")]
    assert not bad, f"{len(bad)} examples have an empty fact block, e.g. {bad[:3]}"


def test_every_fact_carries_a_statement(rows):
    bad = []
    for r in rows:
        for name, stmt in r["facts"].items():
            if not stmt or not stmt.strip():
                bad.append((r["id"], name))
    assert not bad, (f"{len(bad)} facts have a name but no statement — the block is "
                     f"not an oracle, e.g. {bad[:3]}")


def test_every_cited_name_appears_in_the_block(rows):
    """The derivation may only cite facts the block supplies."""
    bad = []
    for r in rows:
        missing = set(r.get("cited", [])) - set(r["facts"])
        if missing:
            bad.append((r["id"], sorted(missing)[:3]))
    assert not bad, (f"{len(bad)} examples cite a fact absent from their block, "
                     f"e.g. {bad[:3]}")


# ----------------------------------------------------------------- I2
def test_no_training_example_cites_a_heldout_fact(rows, heldout):
    bad = [(r["id"], sorted(set(r.get("cited", [])) & heldout)[:3])
           for r in rows if set(r.get("cited", [])) & heldout]
    assert not bad, f"{len(bad)} training examples cite a held-out fact: {bad[:3]}"


def test_no_training_example_proves_a_heldout_fact(rows, heldout):
    """The goal-line leak: a fact's own proof exposes its statement."""
    bad = [r["id"] for r in rows if r.get("theorem") in heldout]
    assert not bad, (f"{len(bad)} training examples ARE the proof of a held-out "
                     f"fact, leaking its statement as the goal: {bad[:3]}")


def test_no_heldout_statement_appears_verbatim_in_training(rows, heldout):
    """Catches leaks through paths I1/I2 miss, e.g. a restated goal."""
    if not rows:
        pytest.skip("empty shard")
    stmts = {}
    for r in rows:
        for n, s in r["facts"].items():
            if n in heldout:
                stmts[n] = s
    assert not stmts, (f"{len(stmts)} held-out statements appear in training fact "
                       f"blocks: {list(stmts)[:3]}")


# ----------------------------------------------------------------- I3
def test_one_name_denotes_one_statement(rows):
    seen = {}
    clashes = []
    for r in rows:
        for n, s in r["facts"].items():
            k = " ".join(s.split())
            if n in seen and seen[n] != k:
                clashes.append(n)
            seen.setdefault(n, k)
    uniq = sorted(set(clashes))
    assert not uniq, (f"{len(uniq)} names denote more than one statement — the store "
                      f"is not persistent: {uniq[:5]}")


# ----------------------------------------------------------------- I4
def test_target_is_nonempty_and_differs_from_the_goal(rows):
    bad = [r["id"] for r in rows
           if not r.get("target", "").strip()
           or " ".join(r["target"].split()) == " ".join(r.get("goal", "").split())]
    assert not bad, f"{len(bad)} examples have an empty or unchanged target: {bad[:3]}"


def test_no_constant_target_dominates(rows):
    """41% of LeanDojo's targets were the literal string 'no goals'."""
    from collections import Counter
    c = Counter(" ".join(r.get("target", "").split()) for r in rows)
    if not c:
        pytest.skip("empty shard")
    top, n = c.most_common(1)[0]
    share = n / len(rows)
    assert share < 0.05, (f"target {top[:40]!r} accounts for {share:.1%} of the "
                          f"shard — degenerate")


# ----------------------------------------------------------------- I5
def test_rendered_text_has_exactly_one_mask_span(rows):
    for r in rows[:5000]:
        t = r["text"]
        assert t.count(HDR) == 1, f"{r['id']}: fact-block header appears {t.count(HDR)}x"
        assert t.count(f"\n{SEP}\n") == 1, f"{r['id']}: separator is not unique"
        assert t.index(HDR) < t.index(f"\n{SEP}\n"), f"{r['id']}: block after separator"


def test_mask_span_covers_the_block_and_nothing_else(rows):
    for r in rows[:5000]:
        t, a, b = r["text"], r["mask_start"], r["mask_end"]
        span = t[a:b]
        assert span.startswith(HDR), f"{r['id']}: mask does not start at the header"
        assert SEP not in span, f"{r['id']}: mask swallows the separator"
        for name in r["facts"]:
            assert name in span, f"{r['id']}: fact {name} sits outside the mask"


def test_masked_fraction_is_in_the_design_band(rows):
    if not rows:
        pytest.skip("empty shard")
    fr = [(r["mask_end"] - r["mask_start"]) / max(len(r["text"]), 1) for r in rows]
    mean = sum(fr) / len(fr)
    assert 0.05 < mean < 0.60, (f"masked fraction {mean:.1%} is outside the 5–60% "
                                f"band; ~17–30% is the design target")


# ----------------------------------------------------------------- I6
def test_no_duplicate_examples(rows):
    ids = [r["id"] for r in rows]
    txt = [r["text"] for r in rows]
    dup_id = len(ids) - len(set(ids))
    dup_tx = len(txt) - len(set(txt))
    assert dup_id == 0, f"{dup_id} duplicate ids"
    assert dup_tx == 0, (f"{dup_tx} examples are byte-identical — the model would "
                         f"see them twice per epoch")


# ----------------------------------------------------------------- I7
def test_train_and_eval_do_not_overlap(rows):
    """Example-level leak, distinct from the held-out fact check in I2."""
    ev_path = SHARD.replace(".jsonl", "_eval.jsonl")
    if not os.path.exists(ev_path):
        pytest.skip("no eval file beside this shard")
    ev = load(ev_path)
    tr_txt = {r["text"] for r in rows}
    tr_thm = {r.get("theorem") for r in rows}
    same_txt = [r["id"] for r in ev if r["text"] in tr_txt]
    same_thm = [r["id"] for r in ev if r.get("theorem") in tr_thm]
    assert not same_txt, f"{len(same_txt)} eval examples appear verbatim in train"
    assert not same_thm, (f"{len(same_thm)} eval theorems are also proved in train — "
                          f"the same result reached another way still leaks")


# ----------------------------------------------------------------- I8
def test_text_is_clean(rows):
    bad_ctrl, bad_repl = [], []
    for r in rows[:50_000]:
        t = r["text"]
        if "\ufffd" in t:
            bad_repl.append(r["id"])
        if any(ord(c) < 9 or 13 < ord(c) < 32 for c in t):
            bad_ctrl.append(r["id"])
    assert not bad_repl, f"{len(bad_repl)} examples contain U+FFFD (bad decode)"
    assert not bad_ctrl, f"{len(bad_ctrl)} examples contain control characters"


def test_statements_are_not_truncated(rows):
    bad = [(r["id"], n) for r in rows for n, s in r["facts"].items()
           if len(s.strip()) < 3 or s.rstrip().endswith(("…", "..."))]
    assert not bad, (f"{len(bad)} statements look truncated — a clipped fact makes "
                     f"the block a bad oracle: {bad[:3]}")


# ----------------------------------------------------------------- I9
def test_goals_are_nondegenerate(rows):
    bad = [r["id"] for r in rows if len(r.get("goal", "").strip()) < 3]
    assert not bad, f"{len(bad)} examples have an empty or trivial goal: {bad[:3]}"


# ----------------------------------------------------------------- I10
def test_fact_block_order_does_not_leak_the_proof(rows):
    """If the block is listed in citation order, the model reads the step
    sequence straight off the prompt without deriving it."""
    multi = [r for r in rows if len(r["facts"]) >= 3]
    if len(multi) < 50:
        pytest.skip("too few multi-fact examples to judge ordering")
    same = sum(1 for r in multi if list(r["facts"]) == r["cited"])
    share = same / len(multi)
    assert share < 0.20, (f"{share:.0%} of multi-fact blocks are in citation order — "
                          f"the block leaks the derivation sequence. Shuffle it with "
                          f"a per-example deterministic seed.")


# ----------------------------------------------------------------- I11
def test_heldout_manifest_is_the_shared_one(heldout):
    """Every machine must mask the same 500 facts or the eval is contaminated."""
    expected = os.environ.get("HELDOUT_SHA256")
    if not expected:
        pytest.skip("set HELDOUT_SHA256 to pin the shared manifest")
    import hashlib
    got = hashlib.sha256(
        json.dumps(sorted(heldout)).encode()).hexdigest()
    assert got == expected, (f"held-out set does not match the shared manifest\n"
                             f"  expected {expected}\n  got      {got}")
```

### 10.2 `scripts/build_mizar_shard.py`

Reference implementation. Machines A, C and D follow this structure, changing only the
parser and the filters named in §5.

```python
"""Reference shard builder for Mizar — the pattern the other three jobs follow.

Produces JSONL where each line is one training example with the fact block, the
goal, the target, the rendered text, and explicit mask offsets. Also emits the
held-out fact set so every machine masks the same 500 facts.

Three decisions are baked in, per the plan:
  * article-local labels are resolved (`Th5` inside POLYRED -> `POLYRED:5`)
  * examples whose block cannot be fully filled are DROPPED, not rendered partial
  * the held-out 500 are drawn from facts cited once or twice, and both the proofs
    citing them and their own proofs are removed from training

Usage:
    python scripts/build_mizar_shard.py --out /tmp/dscount/shards
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

MIZAR = "/tmp/dscount/mizar/html2"
HDR = "I know these mathematical statements:"
SEP = "---"

THM = re.compile(r"^theorem(?:\s+(\w+))?\s*:\s*::\s*([A-Z_0-9]+:\d+)\s*$")
DEFT = re.compile(r"^::\s*deftheorem(?:\s+(\w+))?\s+defines\s+\S+\s+"
                  r"([A-Z_0-9]+:def_\d+)")
BLOCK = re.compile(r"^theorem(?: \w+)?: :: (\S+)\n(.*?)\nproof\n(.*?)\nend;",
                   re.S | re.M)
BY = re.compile(r"\b(?:by|from)\s+([^;]*?);")
GREF = re.compile(r"\b([A-Z][A-Z_0-9]*:(?:def_)?\d+)\b")
LREF = re.compile(r"\b(Th\d+|Def\d+|Lm\d+)\b")


def parse_article(path):
    """(global name -> statement, local label -> global name) for one article."""
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    stmt, local = {}, {}
    for i, ln in enumerate(lines):
        m = THM.match(ln)
        if m:
            lbl, g = m.group(1), m.group(2)
            body = []
            for nx in lines[i + 1:i + 10]:
                s = nx.strip()
                if not s or s.startswith("proof"):
                    break
                body.append(s)
            if body:
                stmt[g] = " ".join(body)
            if lbl:
                local[lbl] = g
            continue
        d = DEFT.match(ln)
        if d:
            lbl, g = d.group(1), d.group(2)
            body = []
            for nx in lines[i + 1:i + 12]:
                s = nx.strip()
                if not s:
                    break
                body.append(s)
                if s.endswith(";"):
                    break
            if body:
                stmt[g] = " ".join(body)
            if lbl:
                local[lbl] = g
    return stmt, local


def cited_names(body, local):
    """Global names cited in a proof body, with article-local labels resolved."""
    out = []
    for m in BY.finditer(body):
        seg = m.group(1)
        out.extend(GREF.findall(seg))
        out.extend(g for g in (local.get(x) for x in LREF.findall(seg)) if g)
    return list(dict.fromkeys(out))


def render(facts, goal, target):
    block = HDR + "\n" + "\n".join(f"{n} : {s}" for n, s in facts.items())
    text = f"{block}\n{SEP}\nGOAL {goal}\n{target}"
    return text, 0, len(block)


def shuffled(refs, key):
    """Order the block independently of citation order.

    Listing premises in the order the derivation uses them hands the model the
    step sequence for free — it can read the proof structure off the prompt
    instead of deriving it. The permutation is seeded per example so the corpus
    is reproducible.
    """
    r = list(refs)
    random.Random(key).shuffle(r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dscount/shards")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(MIZAR, "*.txt")))
    stmt, local_of = {}, {}
    for p in files:
        s, l = parse_article(p)
        stmt.update(s)
        local_of[p] = l

    # pass 1: citation counts, so the held-out set can come from the low tail
    counts = {}
    proofs = []
    for p in files:
        t = open(p, encoding="utf-8", errors="replace").read()
        loc = local_of[p]
        for thm, goal, body in BLOCK.findall(t):
            refs = cited_names(body, loc)
            if not refs:
                continue
            proofs.append((thm, " ".join(goal.split()), body, refs))
            for r in refs:
                counts[r] = counts.get(r, 0) + 1

    tail = sorted(n for n, c in counts.items() if c in (1, 2) and n in stmt)
    rng = random.Random(a.seed)
    held = set(rng.sample(tail, min(a.heldout, len(tail))))
    json.dump({"facts": sorted(held), "seed": a.seed,
               "policy": "cited 1-2 times; own proof and all citing proofs removed"},
              open(os.path.join(a.out, "heldout.json"), "w"), indent=1)

    kept = dropped_incomplete = dropped_heldout = dropped_dup = 0
    tb = 0
    seen_text = set()
    with open(os.path.join(a.out, "mizar.jsonl"), "w") as fh, \
            open(os.path.join(a.out, "mizar_eval.jsonl"), "w") as ev:
        for thm, goal, body, refs in proofs:
            is_eval = thm in held or bool(set(refs) & held)
            if not all(r in stmt for r in refs):
                if not is_eval:
                    dropped_incomplete += 1
                continue
            eid = hashlib.md5(f"{thm}|{goal}".encode()).hexdigest()[:12]
            facts = {r: stmt[r] for r in shuffled(refs, eid)}
            steps = [l.strip() for l in body.split("\n") if l.strip()]
            target = "\n".join(steps)
            text, ms, me = render(facts, goal, target)
            if text in seen_text:
                dropped_dup += 1
                continue
            seen_text.add(text)
            rec = {"id": eid,
                   "theorem": thm, "facts": facts, "cited": refs,
                   "goal": goal, "target": target, "text": text,
                   "mask_start": ms, "mask_end": me}
            if is_eval:
                ev.write(json.dumps(rec) + "\n")
                dropped_heldout += 1
            else:
                fh.write(json.dumps(rec) + "\n")
                kept += 1
                tb += len(text.encode())

    print(f"facts with statements : {len(stmt):,}")
    print(f"held-out facts        : {len(held):,} (from {len(tail):,} cited 1-2x)")
    print(f"train examples        : {kept:,}")
    print(f"eval examples         : {dropped_heldout:,}")
    print(f"dropped, incomplete   : {dropped_incomplete:,}")
    print(f"dropped, duplicate    : {dropped_dup:,}")
    print(f"train bytes           : {tb/1e6:.1f} MB  ~{tb/2.2/1e6:.0f}M GPT-2 tok")
    print(f"wrote {a.out}/mizar.jsonl, mizar_eval.jsonl, heldout.json")


if __name__ == "__main__":
    main()
```

### 10.3 `scripts/make_mutants.py`

```python
"""Corrupt a shard six ways, so the invariant suite can be proven to bite.

A green test run is meaningless until the test has been watched to fail. Each mutant
injects exactly one defect that a real extraction bug would produce, and the suite
must go red on every one.

Note on m3: mutate a fact that occurs in MANY examples. Altering a fact that appears
once creates no clash, the suite stays green, and it looks like a weak test when it
is really a weak mutation. That trap cost a debugging cycle during development.

Usage:
    python scripts/make_mutants.py --shard shards/mizar.jsonl --out shards/mutants
"""

import argparse
import json
import os
from collections import Counter


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--heldout", default=None,
                    help="heldout.json; defaults to a sibling of --shard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    with open(a.shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 100:
        raise SystemExit(f"shard too small to mutate: {len(rows)} rows")

    hp = a.heldout or os.path.join(os.path.dirname(a.shard), "heldout.json")
    if os.path.exists(hp):
        held = json.load(open(hp))["facts"]
    else:
        # A format-demo shard carries no held-out split. Mint a stub of SYNTHETIC
        # names that appear nowhere in the shard: the real shard must still pass
        # against this manifest, while m2 and m6 inject the probes and so fail I2.
        # Drawing the stub from real shard facts would make the real shard fail —
        # a false alarm on good data.
        held = ["__HELDOUT_PROBE_1__", "__HELDOUT_PROBE_2__"]
        json.dump({"facts": held, "seed": None,
                   "policy": "STUB from make_mutants — synthetic probes, not a real "
                             "held-out set"},
                  open(hp, "w"), indent=1)
        print(f"no held-out set at {hp}; wrote a synthetic 2-probe stub")

    freq = Counter(n for r in rows for n in r["facts"])
    common, ncommon = freq.most_common(1)[0]

    def clone():
        return [dict(r) for r in rows]

    # m1 — a fact with a name but no statement
    m = clone()
    m[10] = dict(m[10]); m[10]["facts"] = dict(m[10]["facts"])
    m[10]["facts"][list(m[10]["facts"])[0]] = ""
    write(f"{a.out}/m1_empty_stmt.jsonl", m)

    # m2 — a training example citing a held-out fact
    m = clone()
    m[20] = dict(m[20])
    m[20]["cited"] = list(m[20]["cited"]) + [held[0]]
    m[20]["facts"] = dict(m[20]["facts"]); m[20]["facts"][held[0]] = "leaked stmt"
    write(f"{a.out}/m2_heldout_cited.jsonl", m)

    # m3 — one name, two statements (mutate a FREQUENT fact)
    m = clone()
    hits = 0
    for i, r in enumerate(m):
        if common in r["facts"] and hits < 3:
            m[i] = dict(r); m[i]["facts"] = dict(r["facts"])
            m[i]["facts"][common] = f"variant {i}"
            hits += 1
    write(f"{a.out}/m3_name_clash.jsonl", m)

    # m4 — target identical to the goal
    m = clone()
    m[40] = dict(m[40]); m[40]["target"] = m[40]["goal"]
    write(f"{a.out}/m4_degenerate_target.jsonl", m)

    # m5 — mask truncated mid-block
    m = clone()
    m[50] = dict(m[50]); m[50]["mask_end"] = 30
    write(f"{a.out}/m5_bad_mask.jsonl", m)

    # m6 — the proof OF a held-out fact left in training
    m = clone()
    m[60] = dict(m[60]); m[60]["theorem"] = held[1]
    write(f"{a.out}/m6_heldout_proof.jsonl", m)

    print(f"wrote 6 mutants of {len(rows):,} rows to {a.out}")
    print(f"  m3 mutates {common} ({ncommon} occurrences) — frequent enough to clash")
    print("  every mutant MUST make tests/test_corpus_invariants.py fail")


if __name__ == "__main__":
    main()
```

### 10.4 `scripts/inspect_shard.py`

```python
"""The four data checks no invariant can express, run automatically.

The invariant suite catches structural defects. It cannot catch a shard that is
structurally perfect and semantically wrong — a fact block that is really a file
dump, a target that is mostly copied, a drop rate that means the completeness
filter never ran. Every one of those shipped undetected during the corpus survey.

Exits non-zero if a check trips, so it can gate a build.

Usage:
    python scripts/inspect_shard.py --shard out/mizar.jsonl --source-bytes 156300000
"""

import argparse
import json
import random
import sys
from collections import Counter

HDR = "I know these mathematical statements:"


def load(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_samples(rows, n, seed):
    """1. Print full examples for human reading. Never fails; always look."""
    print("=" * 74)
    print(f"CHECK 1 — {n} random examples, printed in full. READ THESE.")
    print("=" * 74)
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        print(f"\n--- id={r['id']}  theorem={r.get('theorem','?')} ---")
        print(r["text"][:1400])
        if len(r["text"]) > 1400:
            print(f"   … {len(r['text'])-1400} more chars …")
    return True


def check_volume(rows, source_bytes):
    """2. Rendered bytes against source bytes."""
    print("\n" + "=" * 74)
    print("CHECK 2 — rendered volume against source")
    print("=" * 74)
    b = sum(len(r["text"].encode()) for r in rows)
    print(f"  examples        {len(rows):>12,}")
    print(f"  rendered bytes  {b/1e6:>12,.1f} MB")
    print(f"  ~GPT-2 tokens   {b/2.2/1e6:>12,.0f} M")
    if not source_bytes:
        print("  (pass --source-bytes to enable the ratio check)")
        return True
    ratio = b / source_bytes
    print(f"  source bytes    {source_bytes/1e6:>12,.1f} MB")
    print(f"  ratio           {ratio:>12.2f}x")
    if ratio > 5.0:
        print("  FAIL: >5x the source means the fact block is being duplicated")
        print("        far more than the reference (Mizar step-level is 0.89x)")
        return False
    if ratio < 0.02:
        print("  FAIL: <0.02x the source means examples are being dropped silently")
        return False
    print("  ok")
    return True


def check_block_sizes(rows):
    """3. A spike in block size means a file dump, not premise selection."""
    print("\n" + "=" * 74)
    print("CHECK 3 — fact-block size distribution")
    print("=" * 74)
    sizes = [len(r["facts"]) for r in rows]
    c = Counter(sizes)
    mean = sum(sizes) / max(len(sizes), 1)
    top, n = c.most_common(1)[0]
    share = n / len(sizes)
    print(f"  mean facts per block {mean:>8.2f}   distinct sizes {len(c):>6,}")
    print(f"  {'size':>6}{'examples':>12}{'share':>9}")
    for s, k in sorted(c.items())[:8]:
        print(f"  {s:>6}{k:>12,}{100*k/len(sizes):>8.1f}%")
    if len(c) > 8:
        print(f"  … {len(c)-8} more sizes …")
    if share > 0.60:
        print(f"  FAIL: {share:.0%} of examples share block size {top} — this is a")
        print("        file dump, not per-example premise selection")
        return False
    if len(c) < 3:
        print("  FAIL: fewer than 3 distinct block sizes; blocks are not per-example")
        return False
    print("  ok")
    return True


def check_drop_rate(rows, candidates):
    """4. A near-zero drop rate means the completeness filter never ran."""
    print("\n" + "=" * 74)
    print("CHECK 4 — drop rate")
    print("=" * 74)
    if not candidates:
        print("  (pass --candidates <n> — the count of source proofs considered)")
        return True
    dropped = candidates - len(rows)
    rate = dropped / max(candidates, 1)
    print(f"  candidates {candidates:,}   kept {len(rows):,}   dropped {dropped:,}")
    print(f"  drop rate  {rate:.1%}")
    if rate < 0.01:
        print("  FAIL: <1% dropped. The completeness filter is probably not running —")
        print("        Mizar's reference job drops 40% for incomplete fact blocks")
        return False
    if rate > 0.90:
        print("  FAIL: >90% dropped. The extractor is rejecting almost everything")
        return False
    print("  ok")
    return True


def check_paste(rows):
    """Bonus: how much of the target is copied from the goal."""
    print("\n" + "=" * 74)
    print("CHECK 5 — paste share (target words already present in the goal)")
    print("=" * 74)
    vals = []
    for r in rows:
        g, t = set(r.get("goal", "").split()), set(r.get("target", "").split())
        if t:
            vals.append(len(t & g) / len(t))
    if not vals:
        return True
    vals.sort()
    mean = sum(vals) / len(vals)
    print(f"  mean {mean:.3f}   median {vals[len(vals)//2]:.3f}")
    print("  reference: Mizar 0.332, Metamath 0.503, LeanDojo 0.891 (dropped)")
    if mean > 0.75:
        print("  FAIL: the model can copy >75% of each answer; this is not a")
        print("        content target")
        return False
    print("  ok")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--source-bytes", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=0,
                    help="source proofs considered, for the drop-rate check")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load(a.shard, a.limit)
    if not rows:
        print("empty shard")
        sys.exit(1)

    ok = [
        check_samples(rows, a.samples, a.seed),
        check_volume(rows, a.source_bytes),
        check_block_sizes(rows),
        check_drop_rate(rows, a.candidates),
        check_paste(rows),
    ]
    print("\n" + "=" * 74)
    if all(ok):
        print("ALL AUTOMATED CHECKS PASSED — now read the examples from check 1.")
        sys.exit(0)
    print(f"{ok.count(False)} CHECK(S) FAILED — do not ship this shard.")
    sys.exit(1)


if __name__ == "__main__":
    main()
```

### 10.5 `scripts/build_metamath_sample.py`

```python
"""Build a small Metamath shard in the target format — a concrete spec artifact.

The architecture team needs a real file to shape their loader against, not a schema
description. This produces N examples in exactly the format the four production jobs
emit, including the verification step Machine A requires.

Differences from the full Machine A job: it caps at --limit examples and skips the
held-out split, since the point is the shape rather than the corpus.

Usage:
    python scripts/build_metamath_sample.py --out /tmp/dscount/shards --limit 500
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, "scripts")
from mm_expand import MM, expand  # noqa: E402

HDR = "I know these mathematical statements:"
SEP = "---"


def render_fact(label, kind, data):
    """`name : hypotheses => conclusion`, so inference rules are self-contained.

    Printing only the conclusion makes `syl` read `|- ( ph -> ch )`, which says
    nothing — 57.0% of cited Metamath facts carry $e hypotheses.
    """
    concl = " ".join(data[0])
    hyps = [" ".join(h[2]) for h in (data[1] if len(data) > 1 else [])
            if h[0] == "$e"]
    return f"{' & '.join(hyps)} => {concl}" if hyps else concl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/dscount/mm/set.mm")
    ap.add_argument("--out", default="/tmp/dscount/shards")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mm = MM().parse(a.db)
    logical = {l for l, (k, d) in mm.labels.items()
               if k in ("$a", "$p") and d and d[0] and d[0][0] == "|-"}
    prov = sorted(l for l, (k, _) in mm.labels.items() if k == "$p")

    kept = verified = failed = 0
    rows = []
    for lbl in prov:
        if kept >= a.limit:
            break
        try:
            expr, mand, refs, trace = expand(mm, lbl)
        except Exception:
            continue
        steps = [(l, " ".join(e)) for (l, e, _) in trace if e and e[0] == "|-"]
        if not (3 <= len(steps) <= 10):
            continue

        # Machine A's required check: the proof must reduce to its own statement
        if steps[-1][1] != " ".join(expr):
            failed += 1
            continue
        verified += 1

        used = [r for r in dict.fromkeys(refs) if r in logical]
        if not (2 <= len(used) <= 6):
            continue

        eid = hashlib.md5(lbl.encode()).hexdigest()[:12]
        order = list(used)
        random.Random(eid).shuffle(order)          # block order must not leak step order
        facts = {r: render_fact(r, *mm.labels[r]) for r in order}

        goal = " ".join(expr)
        target = "\n".join(f"{i+1:>3}  {l:<12} {e}"
                           for i, (l, e) in enumerate(steps))
        block = HDR + "\n" + "\n".join(f"{n} : {s}" for n, s in facts.items())
        text = f"{block}\n{SEP}\nGOAL {goal}\n{target}"

        rows.append({"id": eid, "theorem": lbl, "facts": facts, "cited": used,
                     "goal": goal, "target": target, "text": text,
                     "mask_start": 0, "mask_end": len(block)})
        kept += 1

    path = os.path.join(a.out, "metamath_sample.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    b = sum(len(r["text"].encode()) for r in rows)
    mf = sum((r["mask_end"] - r["mask_start"]) / len(r["text"]) for r in rows)
    print(f"wrote {path}")
    print(f"  examples        {len(rows):,}")
    print(f"  verified        {verified:,}   failed to reduce {failed:,}")
    print(f"  text bytes      {b/1e6:.2f} MB   file {os.path.getsize(path)/1e6:.2f} MB")
    print(f"  facts/example   {sum(len(r['facts']) for r in rows)/len(rows):.2f}")
    print(f"  masked fraction {mf/len(rows):.1%}")


if __name__ == "__main__":
    main()
```

### 10.6 `scripts/mm_expand.py`

The Metamath proof decoder. §10.5 imports `MM` and `expand` from this; Machine A's
production job needs the same. Stack machine over compressed proofs — decode the label
list between the parentheses after `$=`, replay the step encoding, and the final `|-`
entry is the theorem's own statement, which is what makes verification free.

```python
"""A Metamath proof expander: computes the formula at every proof step.

set.mm stores proofs as compressed label sequences; the intermediate formulas are
not in the file. This runs the RPN stack machine a verifier runs, and records the
expression produced at each step — which is what a model in the state-prediction
design would have to emit.

Enough of the Metamath spec is implemented to execute set.mm proofs: scoping,
mandatory hypotheses, compressed-proof decoding, and substitution. Disjoint
variable conditions are parsed but not enforced, which is fine for measuring
token volume and for display.
"""

from __future__ import annotations

import re
import sys
from typing import Optional

COMMENT = re.compile(r"\$\(.*?\$\)", re.DOTALL)


class Frame:
    __slots__ = ("v", "f", "e", "f_order")

    def __init__(self):
        self.v = set()
        self.f = {}        # var -> (label, typecode)
        self.e = []        # (label, expr)
        self.f_order = []  # (label, typecode, var) in appearance order


class MM:
    def __init__(self):
        self.constants = set()
        self.stack: list[Frame] = [Frame()]
        self.labels = {}   # label -> ('$f'|'$e'|'$a'|'$p', data)

    # ---------------------------------------------------------------- scope
    def push(self):
        self.stack.append(Frame())

    def pop(self):
        self.stack.pop()

    def lookup_f(self, var):
        for fr in reversed(self.stack):
            if var in fr.f:
                return fr.f[var]
        return None

    def active_e(self):
        out = []
        for fr in self.stack:
            out.extend(fr.e)
        return out

    def active_f(self):
        out = []
        for fr in self.stack:
            out.extend(fr.f_order)
        return out

    def is_var(self, tok):
        for fr in reversed(self.stack):
            if tok in fr.v:
                return True
        return False

    # ------------------------------------------------------------ mandatory
    def mandatory(self, expr):
        ess = self.active_e()
        used = set()
        for tok in expr:
            if self.is_var(tok):
                used.add(tok)
        for _, e in ess:
            for tok in e:
                if self.is_var(tok):
                    used.add(tok)
        flo = [(lbl, tc, v) for lbl, tc, v in self.active_f() if v in used]
        # Mandatory hypotheses are ordered by appearance: all $f for mandatory
        # variables first (set.mm declares them before use), then the $e.
        return [("$f", lbl, (tc, v)) for lbl, tc, v in flo] + \
               [("$e", lbl, e) for lbl, e in ess]

    # ---------------------------------------------------------------- parse
    def parse(self, path):
        text = COMMENT.sub(" ", open(path, encoding="utf-8", errors="replace").read())
        toks = text.split()
        i, n = 0, len(toks)
        label = None
        while i < n:
            t = toks[i]
            if t == "${":
                self.push(); i += 1
            elif t == "$}":
                self.pop(); i += 1
            elif t == "$c":
                j = toks.index("$.", i)
                self.constants.update(toks[i + 1:j]); i = j + 1
            elif t == "$v":
                j = toks.index("$.", i)
                self.stack[-1].v.update(toks[i + 1:j]); i = j + 1
            elif t == "$d":
                j = toks.index("$.", i); i = j + 1
            elif t == "$f":
                j = toks.index("$.", i)
                tc, var = toks[i + 1], toks[i + 2]
                self.stack[-1].f[var] = (label, tc)
                self.stack[-1].f_order.append((label, tc, var))
                self.labels[label] = ("$f", (tc, var))
                i = j + 1
            elif t == "$e":
                j = toks.index("$.", i)
                expr = toks[i + 1:j]
                self.stack[-1].e.append((label, expr))
                self.labels[label] = ("$e", expr)
                i = j + 1
            elif t == "$a":
                j = toks.index("$.", i)
                expr = toks[i + 1:j]
                self.labels[label] = ("$a", (expr, self.mandatory(expr)))
                i = j + 1
            elif t == "$p":
                j = toks.index("$=", i)
                expr = toks[i + 1:j]
                k = toks.index("$.", j)
                proof = toks[j + 1:k]
                self.labels[label] = ("$p", (expr, self.mandatory(expr), proof))
                i = k + 1
            else:
                label = t
                i += 1
        return self


def apply_subst(expr, subst):
    out = []
    for tok in expr:
        if tok in subst:
            out.extend(subst[tok])
        else:
            out.append(tok)
    return out


def decode_compressed(proof, mand_n, labels_n):
    """Yield step indices (0-based into mand + labels + saved) and save flags."""
    body = "".join(proof)
    out, num, started = [], 0, False
    for c in body:
        if "U" <= c <= "Y":
            num = num * 5 + (ord(c) - ord("U") + 1)
            started = True
        elif "A" <= c <= "T":
            num = num * 20 + (ord(c) - ord("A") + 1)
            out.append(("step", num - 1))
            num, started = 0, False
        elif c == "Z":
            out.append(("save", None))
        elif c == "?":
            out.append(("unknown", None))
    return out


class Incomplete(Exception):
    pass


def expand(mm: MM, label: str):
    kind, (expr, mand, proof) = mm.labels[label]
    if not proof or proof[0] != "(":
        raise Incomplete("uncompressed or empty proof")
    close = proof.index(")")
    ref_labels = proof[1:close]
    ops = decode_compressed(proof[close + 1:], len(mand), len(ref_labels))

    stack, saved, trace = [], [], []
    for kind_op, idx in ops:
        if kind_op == "unknown":
            raise Incomplete("proof contains ?")
        if kind_op == "save":
            saved.append(list(stack[-1]))
            continue
        if idx < len(mand):
            hk, hl, hd = mand[idx]
            e = [hd[0], hd[1]] if hk == "$f" else list(hd)
            stack.append(e)
            trace.append((hl, list(e), True))
            continue
        idx -= len(mand)
        if idx < len(ref_labels):
            lbl = ref_labels[idx]
            lk, ld = mm.labels[lbl]
            if lk == "$f":
                e = [ld[0], ld[1]]
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            if lk == "$e":
                e = list(ld)
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            sexpr, smand = ld[0], ld[1]
            k = len(smand)
            if k > len(stack):
                raise Incomplete("stack underflow")
            args = stack[-k:] if k else []
            del stack[len(stack) - k:]
            subst = {}
            for (hk, hl, hd), arg in zip(smand, args):
                if hk == "$f":
                    tc, v = hd
                    if not arg or arg[0] != tc:
                        raise Incomplete("typecode mismatch")
                    subst[v] = arg[1:]
            res = apply_subst(sexpr, subst)
            stack.append(res)
            trace.append((lbl, list(res), False))
            continue
        idx -= len(ref_labels)
        if idx >= len(saved):
            raise Incomplete("bad saved index")
        e = list(saved[idx])
        stack.append(e)
        trace.append(("(reuse)", e, True))
    if len(stack) != 1:
        raise Incomplete(f"final stack size {len(stack)}")
    return expr, mand, ref_labels, trace


if __name__ == "__main__":
    mm = MM().parse(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dscount/set.mm")
    print(f"parsed {len(mm.labels):,} labels")
    target = sys.argv[2] if len(sys.argv) > 2 else "mp2"
    expr, mand, refs, trace = expand(mm, target)
    print(f"\ntheorem {target} : {' '.join(expr)}")
    print(f"mandatory hyps: {[l for _, l, _ in mand]}")
    print(f"referenced    : {refs}")
    print("\nfull trace:")
    for i, (lbl, e, is_hyp) in enumerate(trace, 1):
        tag = "hyp " if is_hyp else "step"
        print(f"  {i:>3} {tag} {lbl:<12} {' '.join(e)}")
```

### 10.7 Environment

All five scripts are stdlib-only except where noted. Place them as:

```
scripts/mm_expand.py              # §10.6 — imported by build_metamath_sample.py
scripts/build_metamath_sample.py  # §10.5
scripts/build_mizar_shard.py      # §10.2
scripts/make_mutants.py           # §10.3
scripts/inspect_shard.py          # §10.4
tests/test_corpus_invariants.py   # §10.1
```

```bash
python3 -m pip install pytest ijson pyarrow      # ijson only for the Isabelle job
mkdir -p $SHARED/shards $OUT/mutants
```

Python 3.10+. No GPU needed for extraction. Machine D needs ~4 GB RAM for streaming;
the others run comfortably in 2 GB.
