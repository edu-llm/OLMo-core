"""
Tests for the tool_call_sft_producer.py script.

The load-bearing property is that segments built by construction concatenate to exactly what
OLMo 3's chat template renders. If that holds, the loss mask is provably aligned; if it drifts,
every row is silently mis-masked. That test needs network and is skipped without it — the rest of
the suite is offline and covers the masking logic, the guardrails, and the on-disk format.
"""

import importlib.util
import json
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "tool_call_sft_producer", "src/scripts/data/tool_call_sft_producer.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load tool_call_sft_producer.py")
producer = importlib.util.module_from_spec(spec)
# Register before exec: @dataclass resolves cls.__module__ through sys.modules.
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)


SYS = 'You are a helpful function-calling AI assistant. <functions>[{"a":1}]</functions>'
CALL = '<function_calls>get_weather(city="Boston")</function_calls>'


def _conv(n_assistant: int = 1) -> list[dict[str, str]]:
    """Build a well-formed conversation with ``n_assistant`` assistant turns."""
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYS}]
    for i in range(n_assistant):
        msgs.append({"role": "user", "content": f"turn {i}?"})
        msgs.append({"role": "assistant", "content": CALL})
        if i < n_assistant - 1:
            msgs.append({"role": "environment", "content": '{"temp_f":54}'})
    return msgs


# ==================== normalise_messages ====================


def test_normalise_rewrites_tool_role_to_environment():
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "a"},
    ]
    assert [m["role"] for m in producer.normalise_messages(msgs)] == [
        "system",
        "user",
        "environment",
        "assistant",
    ]


@pytest.mark.parametrize(
    "bad,match",
    [
        ([], "empty"),
        ([{"role": "wizard", "content": "x"}], "expected one of"),
        ([{"role": "assistant"}], "non-empty string"),
        ([{"role": "assistant", "content": None}], "non-empty string"),
        ([{"role": "assistant", "content": ""}], "non-empty string"),
        ([{"role": "user", "content": "x"}], "must end on 'assistant'"),
    ],
)
def test_normalise_rejects_malformed(bad, match):
    with pytest.raises(ValueError, match=match):
        producer.normalise_messages(bad)


# ==================== segments and spans ====================


@pytest.mark.parametrize("n_assistant", [1, 2, 5, 10])
def test_one_trainable_span_per_assistant_turn(n_assistant):
    """The whole point of this producer: every assistant turn stays trainable."""
    msgs = producer.normalise_messages(_conv(n_assistant))
    _, spans = producer.trainable_char_spans(producer.build_segments(msgs))
    assert len(spans) == n_assistant


def test_only_the_final_turn_closes_with_eos():
    msgs = producer.normalise_messages(_conv(3))
    rendered, _ = producer.trainable_char_spans(producer.build_segments(msgs))
    assert rendered.count(producer.DEFAULT_EOS) == 1
    assert rendered.endswith(producer.DEFAULT_EOS)


def test_assistant_header_is_masked_but_content_is_not():
    msgs = producer.normalise_messages(_conv(1))
    segs = producer.build_segments(msgs)
    headers = [s for s in segs if s.text == "<|im_start|>assistant\n"]
    assert len(headers) == 1
    assert headers[0].trainable is False
    bodies = [s for s in segs if s.trainable]
    assert len(bodies) == 1
    assert bodies[0].text.startswith(CALL)


def test_spans_are_disjoint_and_ordered():
    msgs = producer.normalise_messages(_conv(4))
    _, spans = producer.trainable_char_spans(producer.build_segments(msgs))
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a0 < a1 <= b0 < b1


# ==================== mask_from_offsets ====================


def test_mask_marks_overlapping_tokens_only():
    offsets = [(0, 5), (5, 10), (10, 15), (15, 20)]
    mask = producer.mask_from_offsets(offsets, [(5, 15)])
    assert mask.tolist() == [False, True, True, False]


def test_mask_keeps_a_token_straddling_a_boundary():
    """Overlap, not containment — a boundary token stays trainable rather than being dropped."""
    mask = producer.mask_from_offsets([(0, 10)], [(8, 20)])
    assert mask.tolist() == [True]


def test_mask_ignores_zero_width_tokens():
    mask = producer.mask_from_offsets([(5, 5)], [(0, 10)])
    assert mask.tolist() == [False]


# ==================== end to end ====================


class _StubTokenizer:
    """Character-level tokenizer: one token per character, offsets are exact."""

    class _Enc:
        def __init__(self, text: str):
            self.ids = [ord(c) % 1000 for c in text]
            self.offsets = [(i, i + 1) for i in range(len(text))]

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        return self._Enc(text)


def test_encode_row_masks_exactly_the_trainable_characters():
    msgs = _conv(2)
    ids, mask = producer.encode_row(msgs, _StubTokenizer())
    rendered, spans = producer.trainable_char_spans(
        producer.build_segments(producer.normalise_messages(msgs))
    )
    assert ids.size == len(rendered) == mask.size
    expected = sum(b - a for a, b in spans)
    assert int(mask.sum()) == expected


def test_encode_row_rejects_a_row_with_no_eos():
    msgs = _conv(1)
    with pytest.raises(ValueError, match="exactly one"):
        producer.encode_row(msgs, _StubTokenizer(), eos="")


def test_build_writes_headerless_arrays_of_equal_length(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "train-00000.jsonl").write_text(
        "\n".join(json.dumps({"messages": _conv(k)}) for k in (1, 2, 3)) + "\n"
    )
    out = tmp_path / "out"
    stats = producer.build(src, out, _StubTokenizer())

    assert stats["rows"] == 3
    assert stats["skipped"] == 0
    assert stats["shards"] == 1

    ids = np.memmap(out / "token_ids_part_0000.npy", dtype=np.uint32, mode="r")
    mask = np.memmap(out / "labels_mask_part_0000.npy", dtype=np.bool_, mode="r")
    assert ids.size == mask.size == stats["tokens"]
    assert int(mask.sum()) == stats["trainable"]

    # Headerless: a real .npy would start with the numpy magic string.
    assert (out / "token_ids_part_0000.npy").read_bytes()[:6] != b"\x93NUMPY"


def test_build_skips_malformed_rows_without_failing(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    good = json.dumps({"messages": _conv(1)})
    bad = json.dumps({"messages": [{"role": "user", "content": "ends on user"}]})
    (src / "train-00000.jsonl").write_text(f"{good}\n{bad}\n")
    stats = producer.build(src, tmp_path / "out", _StubTokenizer())
    assert stats["rows"] == 1
    assert stats["skipped"] == 1


# ==================== the invariant that needs network ====================


@pytest.mark.parametrize("n_assistant", [1, 2, 5])
def test_segments_match_the_real_chat_template(n_assistant):
    """Segments built by construction must equal the shipped template's render, byte for byte."""
    try:
        msgs = producer.normalise_messages(_conv(n_assistant))
        theirs = producer.render_with_real_template(msgs, eos=producer.DEFAULT_EOS)
    except Exception as e:  # pragma: no cover - needs jinja2 + network
        pytest.skip(f"chat template unavailable: {e}")
    ours, _ = producer.trainable_char_spans(producer.build_segments(msgs))
    assert ours == theirs
