"""Family coverage and bounded-memory loss checks for the P3 evaluator."""

import hashlib
import json
import random
from types import SimpleNamespace

import pytest
import torch

from . import load_project_module


run_eval = load_project_module("run_eval")


def test_default_generation_budget_is_large_enough_for_secondary_proof_metrics():
    assert run_eval.DEFAULT_MAX_NEW_TOKENS == 8_192


def test_metamath_verifier_requires_the_corpus_snapshot(tmp_path):
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    files = {}
    for name in ("set.mm", "iset.mm", "nf.mm"):
        payload = name.encode()
        (mm_dir / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest()}
    manifest = tmp_path / "metamath_sources.json"
    manifest.write_text(json.dumps({"commit": "abc", "files": files}))

    assert run_eval.verify_metamath_sources(mm_dir, manifest)["commit"] == "abc"

    (mm_dir / "set.mm").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="does not match corpus snapshot"):
        run_eval.verify_metamath_sources(mm_dir, manifest)


def test_context_filter_matches_training_eos_inclusive_length_rule():
    class CharTokenizer:
        def __call__(self, texts, *, add_special_tokens):
            assert add_special_tokens is False
            if isinstance(texts, str):
                return {"input_ids": list(range(len(texts)))}
            return {"input_ids": [list(range(len(text))) for text in texts]}

    rows = [
        {"id": "fits", "text": "1234"},
        {"id": "too-long", "text": "12345"},
    ]
    kept, excluded = run_eval.partition_context_eligible(
        rows,
        CharTokenizer(),
        context_length=5,
    )

    assert [row["id"] for row in kept] == ["fits"]
    assert excluded == [{"id": "too-long", "tokens_with_eos": 6}]


def test_generation_budget_uses_remaining_context_instead_of_skipping_row():
    assert run_eval.generation_budgets(
        [8, 14, 16],
        context_length=16,
        max_new_tokens=6,
    ) == {0: 6, 1: 2}


def test_corrupted_condition_never_keeps_the_original_statement():
    row = {"facts": {"f": "A"}, "goal": "G"}
    prompt = run_eval.build_prompt(
        row,
        "facts_corrupted",
        random.Random(1),
        ["A", "B"],
    )
    assert "f : B" in prompt
    assert "f : A" not in prompt


def test_metamath_prompt_keeps_local_assumptions_before_the_separator():
    row = {
        "facts": {"f": "A"},
        "local_assumptions": {"th.1": "|- ph"},
        "goal": "|- ps",
    }

    present = run_eval.build_prompt(
        row,
        "facts_present",
        random.Random(1),
        ["A", "B"],
    )
    absent = run_eval.build_prompt(
        row,
        "facts_absent",
        random.Random(1),
        ["A", "B"],
    )

    expected_local = "Local assumptions:\nth.1 : |- ph"
    assert expected_local in present
    assert present.index(expected_local) < present.index("\n---\nGOAL ")
    assert expected_local in absent, "fact interventions must not remove theorem givens"
    assert "f : A" not in absent


def test_metamath_row_forwards_local_assumptions_to_the_verifier(monkeypatch):
    captured = {}

    def fake_verify(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(run_eval, "verify_proof", fake_verify)
    row = {
        "theorem": "set:th",
        "goal": "|- ph",
        "facts": {"ext": "|- ph"},
        "local_assumptions": {"th.1": "|- ph"},
        "target": "  1  ext  |- ph",
    }

    run_eval.verify_metamath_row({"set": object()}, row, row["target"])

    assert captured["kwargs"]["local_assumptions"] == row["local_assumptions"]


def test_discovers_all_six_eval_families_and_shared_manifests(tmp_path):
    eval_dir = tmp_path / "eval"
    heldout_dir = tmp_path / "heldout"
    eval_dir.mkdir()
    heldout_dir.mkdir()
    for family in run_eval.FAMILIES:
        (eval_dir / f"{family}.jsonl").write_text("{}\n")
    for manifest in set(run_eval.HELDOUT_MANIFEST.values()):
        (heldout_dir / f"{manifest}.json").write_text(json.dumps({"facts": []}))

    assert run_eval.discover_families(tmp_path) == list(run_eval.FAMILIES)
    assert run_eval.HELDOUT_MANIFEST["prf2"] == run_eval.HELDOUT_MANIFEST["enigma"]
    assert run_eval.HELDOUT_MANIFEST["mizar"] == run_eval.HELDOUT_MANIFEST["thproofs"]


def test_target_chunks_cover_each_target_once_with_bounded_context():
    chunks = list(
        run_eval.iter_target_chunks(
            total_tokens=101,
            target_start=19,
            context_length=32,
            chunk_size=7,
        )
    )

    covered = [i for _, start, end in chunks for i in range(start, end)]
    assert covered == list(range(19, 101))
    for context_start, score_start, score_end in chunks:
        assert score_end - context_start <= 32
        assert context_start < score_start
        assert score_end - score_start <= 7


def test_chunked_teacher_forcing_scores_the_same_tokens_as_full_bigram_loss():
    vocab = 16

    class BigramModel:
        def __call__(self, *, input_ids, logits_to_keep, **kwargs):
            del kwargs
            kept = input_ids[:, -logits_to_keep:]
            logits = torch.zeros((*kept.shape, vocab))
            logits.scatter_(-1, ((kept + 1) % vocab).unsqueeze(-1), 4.0)
            return SimpleNamespace(logits=logits)

    ids = torch.arange(13) % vocab
    got_nll, got_tokens, got_correct = run_eval.chunked_sequence_nll(
        BigramModel(),
        ids,
        target_start=4,
        context_length=8,
        chunk_size=3,
        device="cpu",
    )

    logits = torch.zeros((len(ids) - 1, vocab))
    logits.scatter_(-1, ((ids[:-1] + 1) % vocab).unsqueeze(-1), 4.0)
    expected = torch.nn.functional.cross_entropy(
        logits[3:],
        ids[4:],
        reduction="sum",
    )
    assert got_tokens == len(ids) - 4
    assert got_correct == got_tokens
    assert got_nll == pytest.approx(float(expected))


def test_metamath_gold_gate_rejects_unsupplied_assumptions_and_reuse():
    base = {
        "facts": {"ax": "|- ph"},
        "target": "  1  ax       |- ph",
    }
    assert run_eval.gold_trace_uses_only_supplied_labels(base)

    assumption = {**base, "target": "  1  theorem.1  |- ph"}
    reuse = {**base, "target": "  1  (reuse)    |- ph"}
    assert not run_eval.gold_trace_uses_only_supplied_labels(assumption)
    assert not run_eval.gold_trace_uses_only_supplied_labels(reuse)
