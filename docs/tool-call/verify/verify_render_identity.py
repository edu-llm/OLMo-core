"""Prove our inlined row renders BYTE-IDENTICALLY to AI2's sibling-field row.

This is the check docs/tool-call/dataset-design.md section 15 Q1 asks for. It settles the
leading-space question and validates the whole inlining decision in one run.

    python3 docs/tool-call/verify/verify_render_identity.py

Needs jinja2 and network. Fetches:
  - allenai/Olmo-3-7B-Instruct  chat_template.jinja
  - one real row of allenai/Dolci-Instruct-SFT-Tool-Use

Renders (a) AI2's row as published -- sibling ``functions`` / ``function_calls``, ``content``
null -- and (b) our row with all payloads inlined into ``content`` and NO sibling fields, then
byte-compares. Identical output means we inherit OLMo's exact token stream while keeping the
call inside ``content`` where sft-conversations/v1's leakage key can see it.

The jinja environment mirrors transformers' ``_compile_jinja_template``: an immutable sandbox
with trim_blocks/lstrip_blocks and transformers' own ``tojson`` (jinja's default tojson is
HTML-safe and would escape < > & ').
"""

from __future__ import annotations

import json
import urllib.request

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_URL = "https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/chat_template.jinja"
ROWS_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=allenai%2FDolci-Instruct-SFT-Tool-Use&config=default&split=train"
)
EOS = "<|endoftext|>"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def build_env(source: str):
    def tojson(eval_ctx, value, indent=None):
        # transformers' version: NOT html-safe, unlike jinja's builtin
        return json.dumps(value, ensure_ascii=False, indent=indent)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = jinja2.pass_eval_context(tojson)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


def render(tmpl, messages, tools=None) -> str:
    return tmpl.render(
        messages=messages, tools=tools, eos_token=EOS, add_generation_prompt=False
    )


def inline(messages: list[dict]) -> list[dict]:
    """AI2 row layout -> our layout. Move the two sibling fields into ``content``.

    The separators are dictated by the template, not chosen by us:
      system    : content + ' <functions>' + functions + '</functions>'   (LEADING SPACE)
      assistant : '<function_calls>' + function_calls + '</function_calls>'
    """
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content")
        fns = m.get("functions")
        calls = m.get("function_calls")
        if role == "system":
            body = content or ""
            if fns is not None:
                body = f"{body} <functions>{fns}</functions>"
            out.append({"role": "system", "content": body})
        elif role == "assistant":
            body = content or ""
            if calls is not None:
                body = f"{body}<function_calls>{calls}</function_calls>"
            out.append({"role": "assistant", "content": body})
        else:
            out.append({"role": role, "content": content or ""})
    return out


def show_diff(a: str, b: str) -> None:
    if a == b:
        return
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    print(f"    first difference at byte {i}")
    print(f"      theirs: ...{a[max(0, i - 60):i]!r} >>{a[i:i + 40]!r}")
    print(f"      ours:   ...{b[max(0, i - 60):i]!r} >>{b[i:i + 40]!r}")


print("fetching template and one real Dolci row ...")
tmpl = build_env(fetch(TEMPLATE_URL).decode("utf-8"))
rows = json.loads(fetch(ROWS_URL))["rows"]
print(f"  got {len(rows)} rows\n")

allsame = True
for idx, entry in enumerate(rows[:5]):
    row = entry["row"]
    theirs_msgs = row["messages"]
    ours_msgs = inline(theirs_msgs)

    theirs = render(tmpl, theirs_msgs)
    ours = render(tmpl, ours_msgs)
    same = theirs == ours

    roles = "/".join(m["role"][:4] for m in theirs_msgs)
    print(f"row {idx}  turns={len(theirs_msgs):2d} [{roles}]  bytes={len(theirs):6d}  "
          f"IDENTICAL={same}")
    if not same:
        allsame = False
        show_diff(theirs, ours)

print()
print("=== the two separators the template dictates ===")
print("  system   : content + ' <functions>' + functions + '</functions>'   <- LEADING SPACE")
print("             (row-level `tools` path uses NO space: '<functions>')")
print("  assistant: '<function_calls>' + function_calls + '</function_calls>'")

print()
print("=== sanity: our rows carry NO sibling fields ===")
ours0 = inline(rows[0]["row"]["messages"])
keys = sorted({k for m in ours0 for k in m})
print(f"  our message keys: {keys}")
print(f"  any content None? {any(m['content'] is None for m in ours0)}")

print()
print("=== VERDICT ===")
print("  byte-identical on every row checked:" if allsame else "  MISMATCH -- see diffs above:", allsame)
if allsame:
    print("  => we inherit OLMo's exact token stream, and the call sits inside `content`")
    print("     where sft-conversations/v1's leakage key can reach it.")
