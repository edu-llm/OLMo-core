"""Settle WHY multi-turn cannot be tokenized, and which fixes actually work.

    python3 docs/tool-call/verify/verify_multiturn_mask.py

The open-instruct converter builds the prompt loss mask by offset mapping. For each assistant
message i it renders the conversation twice:

    header  = apply_chat_template(messages[:i],   add_generation_prompt=True)
    through = apply_chat_template(messages[:i+1], add_generation_prompt=False)

and requires the FULL render to start with ``through`` -- otherwise the character offsets it
computed do not point at the span it thinks they do, and it raises rather than mask the wrong
tokens.

The suspicion this script tests: OLMo 3's template closes a NON-final assistant turn with
``<|im_end|>\\n`` but the FINAL one with ``eos_token``. In the sub-render, assistant i IS final
(``loop.last`` is true), so it emits ``eos_token``; in the full render the same position holds
``<|im_end|>``. If ``eos_token != "<|im_end|>"`` the prefix check fails on every conversation with
two or more assistant turns.

It also tests the candidate fixes so the choice is evidence-based rather than argued.
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
REAL_EOS = "<|endoftext|>"   # dolma2 / olmo-3-instruct eos, id 100257
IM_END = "<|im_end|>"        # id 100265


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def build(source: str):
    def tojson(eval_ctx, value, indent=None):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = jinja2.pass_eval_context(tojson)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(
        jinja2.exceptions.TemplateError(m)
    )
    return env.from_string(source)


TMPL = build(fetch(TEMPLATE_URL).decode("utf-8"))


def render(messages, *, eos, gen_prompt=False) -> str:
    return TMPL.render(
        messages=messages, tools=None, eos_token=eos, add_generation_prompt=gen_prompt
    )


def prefix_stable(messages, *, eos) -> tuple[bool, list[str]]:
    """Replicate the converter's check for every assistant turn. Returns (all_ok, notes)."""
    full = render(messages, eos=eos)
    notes = []
    ok = True
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        through = render(messages[: i + 1], eos=eos)
        good = full.startswith(through)
        last = i == len(messages) - 1
        if not good:
            ok = False
            n = min(len(full), len(through))
            k = next((x for x in range(n) if full[x] != through[x]), n)
            notes.append(
                f"    assistant[{i}] (final={last}) UNSTABLE at byte {k}: "
                f"full has {full[k:k+12]!r}, sub-render has {through[k:k+12]!r}"
            )
        else:
            notes.append(f"    assistant[{i}] (final={last}) stable")
    return ok, notes


SYS = "You are a helpful function-calling AI assistant. <functions>[]</functions>"
CALL = '<function_calls>get_weather(city="Boston")</function_calls>'

SINGLE = [
    {"role": "system", "content": SYS},
    {"role": "user", "content": "weather in Boston?"},
    {"role": "assistant", "content": CALL},
]
MULTI = SINGLE + [
    {"role": "environment", "content": '{"temp_f":54}'},
    {"role": "assistant", "content": "It's 54F in Boston."},
]

print("=" * 72)
print("1. THE MECHANISM")
print("=" * 72)
for name, msgs in [("single-turn (1 assistant, final)", SINGLE),
                   ("multi-turn (2 assistants)", MULTI)]:
    ok, notes = prefix_stable(msgs, eos=REAL_EOS)
    print(f"\n  {name}  eos={REAL_EOS!r}  ->  {'OK' if ok else 'RAISES'}")
    for n in notes:
        print(n)

print()
print("=" * 72)
print("2. CANDIDATE FIX A — set eos_token = '<|im_end|>' during tokenization")
print("=" * 72)
ok, notes = prefix_stable(MULTI, eos=IM_END)
print(f"\n  multi-turn  eos={IM_END!r}  ->  {'OK' if ok else 'RAISES'}")
for n in notes:
    print(n)
tail = render(MULTI, eos=IM_END)[-40:]
print(f"\n  but the rendered tail becomes: {tail!r}")
print(f"  contains real eos {REAL_EOS!r}? {REAL_EOS in render(MULTI, eos=IM_END)}")
print("  -> if the conversation no longer ends in the real EOS id (100257), OLMo-core's")
print("     packing cannot find the document boundary. Fix A trades one bug for another")
print("     unless the producer appends the real EOS itself.")

print()
print("=" * 72)
print("3. CANDIDATE FIX B — split into prefix rows. MEASURED: DOES NOT WORK")
print("=" * 72)
splits = []
for i, m in enumerate(MULTI):
    if m["role"] == "assistant":
        splits.append(MULTI[: i + 1])
print(f"\n  {len(MULTI)}-message conversation -> {len(splits)} prefix rows")
for s in splits:
    ok, _ = prefix_stable(s, eos=REAL_EOS)
    n_asst = sum(1 for m in s if m["role"] == "assistant")
    print(f"    {len(s)} msgs, {n_asst} assistant turn(s), roles="
          f"{'/'.join(m['role'][:4] for m in s)}  -> {'OK' if ok else 'RAISES'}")
print("\n  The longer prefix row STILL RAISES, because it still contains a NON-FINAL")
print("  assistant turn. Splitting at assistant boundaries does not remove the problem;")
print("  it only removes it from the shortest row. Rejected on evidence, not on taste.")
from edullm_data.profiles.sft_conversations_v1 import _dedup_key  # noqa: E402

keys = [_dedup_key({"messages": s}, None)[:16] for s in splits]
print(f"  (aside) dedup keys of prefix rows: {keys} distinct={len(set(keys)) == len(keys)}")
print("  If prefix-splitting were ever used, all rows from one source conversation must")
print("  land on the SAME side of the train/heldout carve or the shared prefix leaks.")

print()
print("=" * 72)
print("4. THE ACTUAL FIX — open-instruct already has a flag for this")
print("=" * 72)
print("""
  Read from open_instruct/dataset_transformation.py (fetched, not recalled):

    1212:  last_turn_only: bool = False,
    1248:  if last_turn_only and message_idx != last_assistant_idx:
    1249:      continue

  and two transform functions are exported:

    sft_tulu_tokenize_and_truncate_v1        -> last_turn_only=False  (default)
    last_turn_tulu_tokenize_and_truncate_v1  -> last_turn_only=True

  With last_turn_only=True the loop SKIPS every non-final assistant turn, so the
  only prefix-stability check performed is on the final turn -- which section 1
  shows is stable. Their own error text even names the cause:
  "the template appends eos_token only on the final turn".

  So multi-turn is NOT blocked. It is a transform-function choice, and the cost is
  that only the FINAL assistant turn is trainable; earlier tool calls become
  masked context. On a 21-message Dolci row with 10 assistant turns that trains
  1 of 10.
""")

print()
print("=" * 72)
print("5. REAL DOLCI ROWS — how many would fail under the DEFAULT transform")
print("=" * 72)
rows = json.loads(fetch(ROWS_URL))["rows"]
affected = 0
for idx, e in enumerate(rows[:8]):
    msgs = [{"role": m["role"],
             "content": (m.get("content") or "")
             + (f"<function_calls>{m['function_calls']}</function_calls>"
                if m.get("function_calls") else "")}
            for m in e["row"]["messages"]]
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    ok, _ = prefix_stable(msgs, eos=REAL_EOS)
    if not ok:
        affected += 1
    print(f"  row {idx}: {len(msgs):2d} msgs, {n_asst:2d} assistant turns -> "
          f"{'OK' if ok else 'RAISES'}")
print(f"\n  {affected}/8 sampled real rows fail under last_turn_only=False.")
print("  All 8 would PASS under last_turn_only=True, since only the final turn is checked.")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
single_ok, _ = prefix_stable(SINGLE, eos=REAL_EOS)
multi_ok, _ = prefix_stable(MULTI, eos=REAL_EOS)
print(f"  single-turn, real eos, default transform : {'OK' if single_ok else 'RAISES'}")
print(f"  multi-turn,  real eos, default transform : {'OK' if multi_ok else 'RAISES'}")
print(f"  multi-turn,  eos forced to <|im_end|>    : "
      f"{'OK' if prefix_stable(MULTI, eos=IM_END)[0] else 'RAISES'}  (but loses the 100257 doc boundary)")
print("  multi-turn,  last_turn_only=True         : OK (only the final turn is checked)")
print()
print("  Ranked fixes:")
print("   1. last_turn_only=True via last_turn_tulu_tokenize_and_truncate_v1 -- zero code,")
print("      costs training signal on all but the final assistant turn.")
print("   2. Our own producer, mask built BY CONSTRUCTION (render turn-by-turn, tokenize each")
print("      segment, concatenate). No offset mapping, so no prefix requirement, and every")
print("      assistant turn stays trainable. Safe here because all segment boundaries are")
print("      atomic added-tokens (<|im_start|>, <|im_end|>, <|endoftext|>), so no BPE merge")
print("      can straddle a boundary and change the tokenization.")
print("   3. Patch the template so non-final and final assistant turns close identically,")
print("      then append the real EOS once. Upstream-visible change; least attractive.")
