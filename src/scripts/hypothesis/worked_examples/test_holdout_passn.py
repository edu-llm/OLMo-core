"""Lightweight checks for Pass@N helpers (no GPU)."""

from __future__ import annotations

from holdout_passn import HoldoutItem, aggregate_pass_metrics, is_correct, score_generations


def test_normalize_and_correct():
    assert is_correct("Answer: 42\n", "42")
    assert is_correct(".... #### 1,234", "1234")
    assert not is_correct("41", "42")


def test_aggregate_pass_metrics():
    # item0: 1/2 correct → pass=1, ratio=0.5; item1: none → pass=0, ratio=0
    m = aggregate_pass_metrics([[True, False], [False, False]])
    assert abs(m["eval/pass_at_n"] - 0.5) < 1e-9
    assert abs(m["eval/pass_ratio_at_n"] - 0.25) < 1e-9


def test_score_generations():
    items = [
        HoldoutItem(prompt="Problem: x\n", final_answer="2"),
        HoldoutItem(prompt="Problem: y\n", final_answer="3"),
    ]
    gens = [["2", "9"], ["1", "3", "3"]]
    m = score_generations(items, gens)
    assert abs(m["eval/pass_at_n"] - 1.0) < 1e-9
    assert abs(m["eval/pass_ratio_at_n"] - (0.5 + 2 / 3) / 2) < 1e-9


if __name__ == "__main__":
    test_normalize_and_correct()
    test_aggregate_pass_metrics()
    test_score_generations()
    print("ok")
