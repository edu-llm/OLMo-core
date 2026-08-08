"""Probe the real sft-conversations/v1 checks with candidate tool-call record shapes.

Run with the venv that has edullm-data installed:
    <repo>/.venv/bin/python3 scratch/verify_record_shape.py

This exists to settle, empirically, the one irreversible record-format decision: whether a
tool call lives in ``content`` or in a sibling ``tool_calls`` field. The leakage check hashes
message ``content`` only, so the answer is not a matter of taste.
"""

from edullm_data.profiles.sft_conversations_v1 import _messages_wellformed, _dedup_key

SYS = "You have tools."
USER = "weather in Boston?"

# A: OpenAI style — call in a SIBLING field, content=None
A = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Boston"}}],
        },
    ]
}
# A2: same conversation text, DIFFERENT tool call
A2 = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"name": "get_forecast", "arguments": {"city": "Boston", "days": 7}}
            ],
        },
    ]
}
# B: call SERIALIZED INTO content
CALL_B = '<tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call>'
CALL_B2 = '<tool_call>{"name":"get_forecast","arguments":{"city":"Boston","days":7}}</tool_call>'
B = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_B},
    ]
}
B2 = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_B2},
    ]
}
# C: missing content KEY entirely
C = {"messages": [{"role": "assistant", "tool_calls": [{"name": "f", "arguments": {}}]}]}
# D: full multi-turn with a tool result fed back
D = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": CALL_B},
        {"role": "tool", "content": '{"temp_f":54}', "name": "get_weather"},
        {"role": "assistant", "content": "It's 54F in Boston."},
    ]
}
# E: irrelevance / abstention — no call at all
E = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "who wrote Hamlet?"},
        {"role": "assistant", "content": "Shakespeare. No tool is needed for that."},
    ]
}

print("=== well-formedness (None == PASSES the check) ===")
cases = [
    ("A  tool_calls + content:None", A),
    ("B  call-in-content", B),
    ("C  no content key", C),
    ("D  multi-turn + tool role", D),
    ("E  abstention / no call", E),
]
for name, row in cases:
    print(f"  {name:30s} -> {_messages_wellformed(row['messages'])}")

print()
print("=== leakage-key collision (default dedup key) ===")
a, a2 = _dedup_key(A, None), _dedup_key(A2, None)
b, b2 = _dedup_key(B, None), _dedup_key(B2, None)
print("  A vs A2  same text, DIFFERENT call in SIBLING field:")
print(f"    {a[:16]}  {a2[:16]}   COLLIDE={a == a2}")
print("  B vs B2  same text, DIFFERENT call INSIDE content:")
print(f"    {b[:16]}  {b2[:16]}   COLLIDE={b == b2}")

print()
print("=== dedup_key override reads TOP-LEVEL row fields, not message fields ===")
row = {"messages": A["messages"], "tools": [{"name": "get_weather"}], "id": "x1"}
print(f"  dedup_key=['messages'] -> {_dedup_key(row, ['messages'])[:16]}")
print("  (docstring says 'message fields'; code does str(row.get(f,'')) -> top level only)")

print()
print("=== .jsonl extension / format honesty ===")
from edullm_data.manifest import (  # noqa: E402
    EXTENSION_FORMAT,
    Format,
    check_extension_matches_format,
)

print("  .jsonl    claims:", EXTENSION_FORMAT[".jsonl"])
print("  .jsonl.gz claims:", EXTENSION_FORMAT[".jsonl.gz"])
good = Format(container="jsonl", codec="none")
mismatch = Format(container="jsonl", codec="gzip")
p = "conversations/train-00000.jsonl"
print("  .jsonl + codec=none ->", check_extension_matches_format(p, good) or "OK")
print("  .jsonl + codec=gzip ->", check_extension_matches_format(p, mismatch) or "OK")
