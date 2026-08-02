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
    assert not bad, (
        f"{len(bad)} facts have a name but no statement — the block is "
        f"not an oracle, e.g. {bad[:3]}"
    )


def test_every_cited_name_appears_in_the_block(rows):
    """The derivation may only cite facts the block supplies."""
    bad = []
    for r in rows:
        missing = set(r.get("cited", [])) - set(r["facts"])
        if missing:
            bad.append((r["id"], sorted(missing)[:3]))
    assert not bad, f"{len(bad)} examples cite a fact absent from their block, " f"e.g. {bad[:3]}"


# ----------------------------------------------------------------- I2
def test_no_training_example_cites_a_heldout_fact(rows, heldout):
    bad = [
        (r["id"], sorted(set(r.get("cited", [])) & heldout)[:3])
        for r in rows
        if set(r.get("cited", [])) & heldout
    ]
    assert not bad, f"{len(bad)} training examples cite a held-out fact: {bad[:3]}"


def test_no_training_example_proves_a_heldout_fact(rows, heldout):
    """The goal-line leak: a fact's own proof exposes its statement."""
    bad = [r["id"] for r in rows if r.get("theorem") in heldout]
    assert not bad, (
        f"{len(bad)} training examples ARE the proof of a held-out "
        f"fact, leaking its statement as the goal: {bad[:3]}"
    )


def test_no_heldout_statement_appears_verbatim_in_training(rows, heldout):
    """Catches leaks through paths I1/I2 miss, e.g. a restated goal."""
    if not rows:
        pytest.skip("empty shard")
    stmts = {}
    for r in rows:
        for n, s in r["facts"].items():
            if n in heldout:
                stmts[n] = s
    assert not stmts, (
        f"{len(stmts)} held-out statements appear in training fact " f"blocks: {list(stmts)[:3]}"
    )


# ----------------------------------------------------------------- I3
def test_one_name_denotes_one_statement(rows):
    seen: dict = {}
    clashes: list = []
    for r in rows:
        for n, s in r["facts"].items():
            k = " ".join(s.split())
            if n in seen and seen[n] != k:
                clashes.append(n)
            seen.setdefault(n, k)
    uniq = sorted(set(clashes))
    assert not uniq, (
        f"{len(uniq)} names denote more than one statement — the store "
        f"is not persistent: {uniq[:5]}"
    )


# ----------------------------------------------------------------- I4
def test_target_is_nonempty_and_differs_from_the_goal(rows):
    bad = [
        r["id"]
        for r in rows
        if not r.get("target", "").strip()
        or " ".join(r["target"].split()) == " ".join(r.get("goal", "").split())
    ]
    assert not bad, f"{len(bad)} examples have an empty or unchanged target: {bad[:3]}"


def test_no_constant_target_dominates(rows):
    """41% of LeanDojo's targets were the literal string 'no goals'."""
    from collections import Counter

    c = Counter(" ".join(r.get("target", "").split()) for r in rows)
    if not c:
        pytest.skip("empty shard")
    top, n = c.most_common(1)[0]
    share = n / len(rows)
    assert share < 0.05, (
        f"target {top[:40]!r} accounts for {share:.1%} of the " f"shard — degenerate"
    )


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
    assert 0.05 < mean < 0.60, (
        f"masked fraction {mean:.1%} is outside the 5–60% " f"band; ~17–30% is the design target"
    )


# ----------------------------------------------------------------- I6
def test_no_duplicate_examples(rows):
    ids = [r["id"] for r in rows]
    txt = [r["text"] for r in rows]
    dup_id = len(ids) - len(set(ids))
    dup_tx = len(txt) - len(set(txt))
    assert dup_id == 0, f"{dup_id} duplicate ids"
    assert dup_tx == 0, (
        f"{dup_tx} examples are byte-identical — the model would " f"see them twice per epoch"
    )


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
    assert not same_thm, (
        f"{len(same_thm)} eval theorems are also proved in train — "
        f"the same result reached another way still leaks"
    )


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
    bad = [
        (r["id"], n)
        for r in rows
        for n, s in r["facts"].items()
        if len(s.strip()) < 3 or s.rstrip().endswith(("…", "..."))
    ]
    assert not bad, (
        f"{len(bad)} statements look truncated — a clipped fact makes "
        f"the block a bad oracle: {bad[:3]}"
    )


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
    assert share < 0.20, (
        f"{share:.0%} of multi-fact blocks are in citation order — "
        f"the block leaks the derivation sequence. Shuffle it with "
        f"a per-example deterministic seed."
    )


# ----------------------------------------------------------------- I11
def test_heldout_manifest_is_the_shared_one(heldout):
    """Every machine must mask the same 500 facts or the eval is contaminated."""
    expected = os.environ.get("HELDOUT_SHA256")
    if not expected:
        pytest.skip("set HELDOUT_SHA256 to pin the shared manifest")
    import hashlib

    got = hashlib.sha256(json.dumps(sorted(heldout)).encode()).hexdigest()
    assert got == expected, (
        f"held-out set does not match the shared manifest\n"
        f"  expected {expected}\n  got      {got}"
    )
