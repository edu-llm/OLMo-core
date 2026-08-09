"""Probe the real sft-conversations/v1 checks with our tool-call record shapes.

Run with the venv that has edullm-data installed:
    <repo>/.venv/bin/python3 docs/tool-call/verify/verify_record_shape.py

Settles the one irreversible record-format decision: whether the tool call lives in ``content``
or in a sibling field. The profile's leakage check hashes message ``content`` only, so the answer
is not a matter of taste.

Format is OLMo 3's convention (``<functions>`` 100266/100267, ``<function_calls>`` 100268/100269,
results on role ``environment``), with all three payloads INLINED into ``content`` -- see
docs/tool-call/dataset-design.md sections 2, 3 and 7.
"""

from edullm_data.profiles.sft_conversations_v1 import _dedup_key, _messages_wellformed

SCHEMAS = (
    '[{"type":"function","function":{"name":"get_weather","description":"Current conditions'
    ' for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},'
    '"required":["city"]}}}]'
)
SYS = f"You are a helpful function-calling AI assistant. <functions>{SCHEMAS}</functions>"
USER = "weather in Boston?"

CALL_A = '<function_calls>get_weather(city="Boston")</function_calls>'
CALL_B = '<function_calls>get_forecast(city="Boston", days=7)</function_calls>'

# --- OURS: OLMo convention, call inlined into assistant content ---------------------
OURS = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_A},
    ]
}
# same conversation text, DIFFERENT call
OURS_ALT = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_B},
    ]
}

# --- AI2's Dolci row layout: call in a SIBLING field, content is null ---------------
# Verified against a real allenai/Dolci-Instruct-SFT-Tool-Use row.
THEIRS = {
    "messages": [
        {"role": "system", "content": SYS, "function_calls": None, "functions": SCHEMAS},
        {"role": "user", "content": USER, "function_calls": None, "functions": None},
        {
            "role": "assistant",
            "content": None,
            "function_calls": 'get_weather(city="Boston")',
            "functions": None,
        },
    ]
}
THEIRS_ALT = {
    "messages": [
        {"role": "system", "content": SYS, "function_calls": None, "functions": SCHEMAS},
        {"role": "user", "content": USER, "function_calls": None, "functions": None},
        {
            "role": "assistant",
            "content": None,
            "function_calls": 'get_forecast(city="Boston", days=7)',
            "functions": None,
        },
    ]
}

PARALLEL = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "weather in Paris and Madrid?"},
        {
            "role": "assistant",
            "content": '<function_calls>get_weather(city="Paris")\n'
            'get_weather(city="Madrid")</function_calls>',
        },
    ]
}
MULTITURN = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_A},
        {"role": "environment", "content": '{"temp_f":54}'},
        {"role": "assistant", "content": "It's 54F in Boston."},
    ]
}
ABSTAIN = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "who wrote Hamlet?"},
        {"role": "assistant", "content": "Shakespeare. No tool is needed for that."},
    ]
}
NO_CONTENT_KEY = {"messages": [{"role": "assistant", "function_calls": "f()"}]}

print("=== well-formedness (None == PASSES) ===")
for name, row in [
    ("ours: call inlined", OURS),
    ("ours: parallel, one block", PARALLEL),
    ("ours: multi-turn + environment", MULTITURN),
    ("ours: abstention", ABSTAIN),
    ("AI2 layout: content=None", THEIRS),
    ("no content KEY at all", NO_CONTENT_KEY),
]:
    print(f"  {name:32s} -> {_messages_wellformed(row['messages'])}")
print("  NOTE: the AI2 layout PASSES well-formedness -- null content is accepted.")
print("        Only the leakage test below shows why we cannot use it as-is.")

print()
print("=== leakage-key collision: the decision ===")
o, oa = _dedup_key(OURS, None), _dedup_key(OURS_ALT, None)
t, ta = _dedup_key(THEIRS, None), _dedup_key(THEIRS_ALT, None)
print("  AI2 layout  (call in sibling field, content=None):")
print(f"    {t[:16]}  {ta[:16]}   COLLIDE={t == ta}   <-- validator is BLIND to the call")
print("  Ours        (call inlined into content):")
print(f"    {o[:16]}  {oa[:16]}   COLLIDE={o == oa}   <-- validator sees the call")
print()
print("  With max_leakage=0, a train/heldout pair differing only in the call would")
print("  refuse the entire publish under the AI2 layout. Hence the inlining decision.")

print()
print("=== dedup_key override reads TOP-LEVEL row fields, not message fields ===")
row = {"messages": OURS["messages"], "tools": [{"name": "get_weather"}], "id": "x1"}
print(f"  dedup_key=['messages'] -> {_dedup_key(row, ['messages'])[:16]}")
print("  (docstring says 'message fields'; code does str(row.get(f,'')) -> top level only,")
print("   so it cannot reach inside messages and is NOT an escape hatch)")

print()
print("=== .jsonl extension / format honesty ===")
from edullm_data.manifest import (  # noqa: E402
    EXTENSION_FORMAT,
    Format,
    check_extension_matches_format,
)

print("  .jsonl    claims:", EXTENSION_FORMAT[".jsonl"])
print("  .jsonl.gz claims:", EXTENSION_FORMAT[".jsonl.gz"])
p = "conversations/general/single-call/train-00000.jsonl"
print(
    "  codec=none ->",
    check_extension_matches_format(p, Format(container="jsonl", codec="none")) or "OK",
)
print(
    "  codec=gzip ->",
    check_extension_matches_format(p, Format(container="jsonl", codec="gzip")) or "OK",
)
