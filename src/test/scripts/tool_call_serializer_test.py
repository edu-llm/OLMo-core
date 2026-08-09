"""
Tests for the tool_call_serializer.py script.

The serializer exists so that no model ever writes the wire format. That only pays off if the
serializer is right, so the properties worth testing are round-tripping, the JSON-vs-Python
argument spellings, and the shapes we deliberately refuse.
"""

import importlib.util
import json
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "tool_call_serializer", "src/scripts/data/tool_call_serializer.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load tool_call_serializer.py")
ser = importlib.util.module_from_spec(spec)
# Register before exec: @dataclass resolves cls.__module__ through sys.modules.
sys.modules[spec.name] = ser
spec.loader.exec_module(ser)


WEATHER = ser.ToolSchema(
    name="weather.forecast_weather_api",
    description="Fetches weather forecast and alerts from a weather API.",
    parameters={
        "type": "object",
        "properties": {"q": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["q"],
    },
)
GRADE = ser.ToolSchema(
    name="post_score",
    description="Post one learner's score.",
    parameters={
        "type": "object",
        "properties": {"userId": {"type": "string"}, "excuse": {"type": "boolean"}},
        "required": ["userId"],
    },
)


# ==================== call serialisation ====================


def test_call_matches_the_template_form():
    call = ser.Call("get_weather", {"city": "Boston", "days": 5})
    assert ser.serialize_call(call) == 'get_weather(city="Boston", days=5)'


def test_dotted_function_names_survive():
    call = ser.Call("weather.forecast_weather_api", {"q": "Paris"})
    assert ser.serialize_call(call) == 'weather.forecast_weather_api(q="Paris")'
    assert ser.parse_call(ser.serialize_call(call)) == call


@pytest.mark.parametrize(
    "value,wire",
    [
        (True, "true"),
        (False, "false"),
        (None, "null"),
        ("Boston", '"Boston"'),
        (5, "5"),
        (1.5, "1.5"),
        ([1, 2], "[1, 2]"),
        ({"a": 1}, '{"a": 1}'),
    ],
)
def test_values_are_json_encoded_not_python_repr(value, wire):
    """The template uses ``| tojson``, so booleans are ``true``, never Python's ``True``."""
    assert ser.serialize_call(ser.Call("f", {"x": value})) == f"f(x={wire})"


@pytest.mark.parametrize(
    "value", [True, False, None, "s", 5, -3, 1.5, [1, {"a": None}], {"k": [True]}]
)
def test_every_json_value_round_trips(value):
    """The regression this module was written for: ``ast.literal_eval`` raises on ``true``."""
    call = ser.Call("f", {"x": value})
    assert ser.parse_call(ser.serialize_call(call)).arguments["x"] == value


def test_parse_rejects_positional_arguments():
    with pytest.raises(ValueError, match="positional"):
        ser.parse_call('f("Boston")')


def test_parse_rejects_a_non_literal_value():
    with pytest.raises(ValueError, match="not a JSON literal"):
        ser.parse_call("f(x=some_variable)")


def test_serialize_rejects_a_non_identifier_function_name():
    with pytest.raises(ValueError, match="identifier"):
        ser.serialize_call(ser.Call("Market Trends API", {}))


# ==================== parallel calls ====================


def test_parallel_calls_share_one_block_joined_by_newline():
    calls = [ser.Call("f", {"a": 1}), ser.Call("g", {"b": 2})]
    out = ser.serialize_calls(calls)
    assert out == "<function_calls>f(a=1)\ng(b=2)</function_calls>"
    assert out.count("<function_calls>") == 1


def test_parse_rejects_two_separate_call_blocks():
    row = ser.build_row(schemas=[WEATHER], user="q", calls=[ser.Call("f", {})])
    row["messages"][-1][
        "content"
    ] = "<function_calls>f()</function_calls><function_calls>g()</function_calls>"
    with pytest.raises(ValueError, match="ONE block"):
        ser.parse_row(row)


def test_parse_rejects_text_after_the_call_block():
    row = ser.build_row(schemas=[WEATHER], user="q", calls=[ser.Call("f", {})])
    row["messages"][-1]["content"] += " and then I will check."
    with pytest.raises(ValueError, match="nothing may follow"):
        ser.parse_row(row)


# ==================== the schema block ====================


def test_schema_block_is_one_line_and_keeps_the_leading_space():
    block = ser.serialize_schemas([WEATHER, GRADE])
    assert block.startswith(" <functions>")
    assert block.endswith("</functions>")
    assert "\n" not in block


def test_the_preamble_mentions_the_tag_and_that_must_not_confuse_the_parser():
    """Regression: the default preamble contains the literal ``<functions></functions>``.

    A naive ``find`` locates that decorative mention instead of the real block and fails on every
    single row.
    """
    assert "<functions></functions>" in ser.DEFAULT_PREAMBLE
    row = ser.build_row(schemas=[WEATHER], user="q", calls=[ser.Call("f", {})])
    assert [s.name for s in ser.parse_row(row).schemas] == [WEATHER.name]


def test_a_tool_description_containing_the_literal_tag_still_parses():
    sneaky = ser.ToolSchema(
        name="doc_tool",
        description="Explains that signatures live inside <functions></functions> tags.",
        parameters={"type": "object", "properties": {}},
    )
    row = ser.build_row(schemas=[sneaky, WEATHER], user="q", prose="none apply")
    assert [s.name for s in ser.parse_row(row).schemas] == ["doc_tool", WEATHER.name]


def test_schema_block_round_trips_through_the_row():
    row = ser.build_row(schemas=[WEATHER, GRADE], user="q", calls=[ser.Call("f", {})])
    parsed = ser.parse_row(row)
    assert [s.name for s in parsed.schemas] == [WEATHER.name, GRADE.name]
    assert parsed.schemas[0].parameters == WEATHER.parameters


# ==================== rows ====================


def test_row_has_the_expected_roles_and_ends_on_assistant():
    row = ser.build_row(schemas=[WEATHER], user="weather?", calls=[ser.Call("f", {})])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]


def test_abstention_row_carries_prose_and_no_call_block():
    row = ser.build_row(schemas=[WEATHER], user="who wrote Hamlet?", prose="Shakespeare.")
    parsed = ser.parse_row(row)
    assert parsed.calls == []
    assert parsed.prose == "Shakespeare."
    assert ser.CALL_OPEN not in row["messages"][-1]["content"]


def test_row_rejects_both_calls_and_prose():
    with pytest.raises(ValueError, match="exactly one"):
        ser.build_row(schemas=[WEATHER], user="q", calls=[ser.Call("f", {})], prose="hi")


def test_row_rejects_neither_calls_nor_prose():
    with pytest.raises(ValueError, match="exactly one"):
        ser.build_row(schemas=[WEATHER], user="q")


def test_abstention_prose_may_not_smuggle_in_a_call():
    with pytest.raises(ValueError, match="must not contain"):
        ser.build_row(
            schemas=[WEATHER], user="q", prose="sure: <function_calls>f()</function_calls>"
        )


def test_multi_turn_row_places_prior_turns_before_the_user_turn():
    row = ser.build_row(
        schemas=[WEATHER],
        user="and Madrid?",
        calls=[ser.Call("f", {})],
        prior_turns=[
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": '<function_calls>f(q="Paris")</function_calls>'},
            {"role": "environment", "content": '{"temp_c":21}'},
        ],
    )
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "environment",
        "user",
        "assistant",
    ]


def test_multi_turn_rejects_an_unknown_prior_role():
    with pytest.raises(ValueError, match="must be user, assistant or environment"):
        ser.build_row(
            schemas=[WEATHER],
            user="q",
            calls=[ser.Call("f", {})],
            prior_turns=[{"role": "tool", "content": "{}"}],
        )


def test_extra_fields_land_at_the_top_level():
    row = ser.build_row(
        schemas=[WEATHER],
        user="q",
        calls=[ser.Call("f", {})],
        extra={"domain": "general", "expected_result": 42},
    )
    assert row["domain"] == "general"
    assert row["expected_result"] == 42


def test_extra_may_not_clobber_messages():
    with pytest.raises(ValueError, match="may not override"):
        ser.build_row(schemas=[WEATHER], user="q", prose="x", extra={"messages": []})


# ==================== round trip + jsonl ====================


def test_assert_round_trip_accepts_what_we_build():
    row = ser.build_row(
        schemas=[WEATHER, GRADE],
        user="post it",
        calls=[ser.Call("post_score", {"userId": "u1", "excuse": False})],
    )
    parsed = ser.assert_round_trip(row)
    assert parsed.calls[0].arguments == {"userId": "u1", "excuse": False}


def test_iter_jsonl_emits_one_valid_line_per_row():
    rows = [
        ser.build_row(schemas=[WEATHER], user="a", calls=[ser.Call("f", {"x": True})]),
        ser.build_row(schemas=[WEATHER], user="b", prose="no tool needed"),
    ]
    lines = list(ser.iter_jsonl(rows))
    assert len(lines) == 2
    for line in lines:
        assert "\n" not in line
        assert json.loads(line)["messages"][0]["role"] == "system"


# ==================== it feeds the producer ====================


def test_serialized_rows_are_accepted_by_the_producer():
    """The two halves must agree: whatever the serializer writes, the producer must tokenize."""
    prod_spec = importlib.util.spec_from_file_location(
        "tool_call_sft_producer", "src/scripts/data/tool_call_sft_producer.py"
    )
    assert prod_spec is not None and prod_spec.loader is not None
    producer = importlib.util.module_from_spec(prod_spec)
    sys.modules[prod_spec.name] = producer
    prod_spec.loader.exec_module(producer)

    class _StubTokenizer:
        class _Enc:
            def __init__(self, text):
                self.ids = [ord(c) % 1000 for c in text]
                self.offsets = [(i, i + 1) for i in range(len(text))]

        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            return self._Enc(text)

    for row in [
        ser.build_row(schemas=[WEATHER], user="a", calls=[ser.Call("f", {"x": True})]),
        ser.build_row(schemas=[WEATHER], user="b", prose="no tool needed"),
        ser.build_row(
            schemas=[WEATHER],
            user="c",
            calls=[ser.Call("f", {})],
            prior_turns=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "<function_calls>g()</function_calls>"},
                {"role": "environment", "content": "{}"},
            ],
        ),
    ]:
        ids, mask = producer.encode_row(row["messages"], _StubTokenizer())
        assert ids.size == mask.size
        assert mask.any()
