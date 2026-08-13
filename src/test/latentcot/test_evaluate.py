"""Tests for the evaluation harness (PRD Phase 6.1/6.2)."""

import pytest
import torch

from olmo_core.latentcot import evaluate as E
from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.nn.transformer import TransformerConfig

D_MODEL = 128


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_model():
    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
    )
    return cfg.build(init_device="cpu")


def _examples(k=3):
    return [
        encode_example(
            generate(num_nodes=12, branching=2, depth=d, seed=d, reachable=bool(d % 2)), k
        )
        for d in (2, 3, 4)
    ]


@pytest.mark.parametrize("mode", ["codi", "no_cot", "explicit_cot"])
def test_predict_reachable_returns_bool(tok, tiny_model, mode):
    ex = _examples()[0]
    pred = E.predict_reachable(tiny_model, ex, mode, max_new_tokens=16)
    assert isinstance(pred, bool)


def test_solve_rate_by_depth_structure(tok, tiny_model):
    rates = E.solve_rate_by_depth(tiny_model, _examples(), "codi")
    assert set(rates) == {2, 3, 4}
    assert all(0.0 <= v <= 1.0 for v in rates.values())


def test_gate_a_curve_and_slope():
    cont = {2: 0.60, 3: 0.75, 4: 0.95}
    disc = {2: 0.55, 3: 0.60, 4: 0.65}
    curve = E.gate_a_curve(cont, disc)
    assert curve == {2: pytest.approx(0.05), 3: pytest.approx(0.15), 4: pytest.approx(0.30)}
    assert E.linear_slope(curve) > 0  # advantage grows with depth


def test_inference_token_cost_positive(tok):
    ex = _examples()[0]
    for mode in ("codi", "no_cot", "explicit_cot"):
        assert E.inference_token_cost(ex, mode) > 0


def test_run_eval_report_structure(tok, tiny_model):
    examples = _examples()
    # same tiny model for each arm — we only check the report structure here
    models = {"A0": tiny_model, "A2": tiny_model, "A3": tiny_model}
    report = E.run_eval(models, examples)
    assert set(report["per_arm"]) == {"A0", "A2", "A3"}
    assert "decodability" in report["per_arm"]["A2"]  # codi arm
    assert "decodability" not in report["per_arm"]["A0"]  # explicit_cot arm
    assert "slope" in report["gate_a"] and "curve" in report["gate_a"]
    assert set(report["gate_b"]) == {"A2", "A3"}
    assert report["gate_b"]["A2"]["decodability"] is not None


class _RecordingModel(torch.nn.Module):
    """Emits a fixed token forever and records the ids of every forward it is given."""

    def __init__(self, emit: int, vocab: int):
        super().__init__()
        self.emit = emit
        self.vocab = vocab
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, ids, **kwargs):
        self.calls.append(list(ids[0].tolist()))
        logits = torch.zeros(1, ids.shape[1], self.vocab)
        logits[0, -1, self.emit] = 10.0
        return logits


def test_explicit_cot_reads_the_answer_at_an_eot_distill_it_appends(tok):
    """
    The answer position must be one the eval constructs, not one the model has to emit.

    encode_example masks the EOT and DISTILL positions to False, so nothing trains A0 to produce
    them. Reading the answer wherever generation happened to stop scored A0 at 0.518 on
    run_019ff806 while its teacher CE was 0.001. The final forward must therefore end with exactly
    [EOT, DISTILL], rebuilding the layout the answer was supervised under.
    """
    from olmo_core.latentcot.tokens import DISTILL, EOT

    ex = _examples()[0]
    # A token that is not a stop token, so generation runs to the cap.
    filler = E.node_token_id(1)
    assert filler not in set(E.cot_stop_token_ids())
    model = _RecordingModel(emit=filler, vocab=T.PADDED_VOCAB_SIZE)

    stats: dict = {}
    E.answer_logits(model, ex, "explicit_cot", max_new_tokens=5, stats=stats)

    final = model.calls[-1]
    assert final[-2:] == [EOT, DISTILL], "answer must be read at an appended <eot><distill>"
    assert final.count(DISTILL) == 1, "DISTILL must not be duplicated"
    # The cap was hit, and that is exactly what the counter exists to surface.
    assert stats["explicit_cot_cap_hits"] == 1
    assert stats["explicit_cot_examples"] == 1


def test_greedy_generate_reports_whether_it_hit_the_cap(tok):
    """hit_cap distinguishes "the arm cannot answer" from "the eval stopped in the wrong place"."""
    stop = E.cot_stop_token_ids()[0]
    filler = E.node_token_id(2)

    stopped = _RecordingModel(emit=stop, vocab=T.PADDED_VOCAB_SIZE)
    ids, hit_cap = E.greedy_generate(stopped, [1, 2, 3], max_new_tokens=10, stop_token=stop)
    assert hit_cap is False and ids[-1] == stop

    runaway = _RecordingModel(emit=filler, vocab=T.PADDED_VOCAB_SIZE)
    ids, hit_cap = E.greedy_generate(runaway, [1, 2, 3], max_new_tokens=4, stop_token=stop)
    assert hit_cap is True and len(ids) == 3 + 4
