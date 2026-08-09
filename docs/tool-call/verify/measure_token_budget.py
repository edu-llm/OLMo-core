"""Measure the real token cost of a tool-call row, and project the corpus budget.

    python3 docs/tool-call/verify/measure_token_budget.py

Tokenizes real `allenai/Dolci-Instruct-SFT-Tool-Use` rows with the real
`allenai/olmo-3-tokenizer-instruct-dev`, inlined into our wire format, so the numbers reflect
actual schema blocks rather than toy examples. Reports the distribution split by turn count, since
our v1 is single-turn only, and projects the 40,000-row plan.
"""

import json
import statistics as stats
import urllib.request

from tokenizers import Tokenizer

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=allenai%2FDolci-Instruct-SFT-Tool-Use"
    "&config=default&split=train&offset=0&length=100"
)
TOKENIZER = "allenai/olmo-3-tokenizer-instruct-dev"
EOS = "<|endoftext|>"

# The 32-cell plan, from docs/tool-call/dataset-design.md section 6.
PLAN = {"general": 15_000, "arithmetic": 7_000, "web-search": 7_000, "pedagogy": 11_000}
TOTAL_ROWS = sum(PLAN.values())


def render(messages, eos=EOS) -> str:
    """Inline AI2's sibling fields into content and render as our producer does."""
    out = []
    last = len(messages) - 1
    for i, m in enumerate(messages):
        role = m["role"]
        body = m.get("content") or ""
        if role == "system" and m.get("functions"):
            body = f"{body} <functions>{m['functions']}</functions>"
        if role == "assistant" and m.get("function_calls"):
            body = f"{body}<function_calls>{m['function_calls']}</function_calls>"
        close = eos if (role == "assistant" and i == last) else "<|im_end|>\n"
        out.append(f"<|im_start|>{role}\n{body}{close}")
    return "".join(out)


def summarise(name, values):
    if not values:
        print(f"  {name:34s} (none)")
        return
    print(
        f"  {name:34s} n={len(values):3d}  "
        f"min={min(values):6,}  median={int(stats.median(values)):6,}  "
        f"mean={int(stats.mean(values)):6,}  p90={int(sorted(values)[int(len(values)*0.9)]):6,}  "
        f"max={max(values):6,}"
    )


print("fetching 100 real rows and the tokenizer ...")
rows = json.loads(urllib.request.urlopen(ROWS_URL, timeout=90).read())["rows"]
tok = Tokenizer.from_pretrained(TOKENIZER)

all_tokens, single_turn, multi_turn, sys_only = [], [], [], []
for e in rows:
    msgs = e["row"]["messages"]
    n = len(tok.encode(render(msgs), add_special_tokens=False).ids)
    all_tokens.append(n)
    (single_turn if len(msgs) == 3 else multi_turn).append(n)
    # cost of the schema block alone -- the fixed overhead every row pays
    sysm = [m for m in msgs if m["role"] == "system"]
    if sysm:
        sys_only.append(len(tok.encode(render(sysm[:1]), add_special_tokens=False).ids))

print()
print("=== tokens per row, real rows, our wire format ===")
summarise("all sampled rows", all_tokens)
summarise("3-message (our v1 scope)", single_turn)
summarise("multi-turn (v2)", multi_turn)
summarise("system/schema block alone", sys_only)

median_single = int(stats.median(single_turn)) if single_turn else 0
mean_single = int(stats.mean(single_turn)) if single_turn else 0

print()
print("=== corpus projection, 40,000 single-turn rows ===")
print(f"  {'domain':14s} {'rows':>7s} {'tokens @median':>16s} {'tokens @mean':>15s}")
for d, r in PLAN.items():
    print(f"  {d:14s} {r:7,} {r * median_single:16,} {r * mean_single:15,}")
print(f"  {'TOTAL':14s} {TOTAL_ROWS:7,} "
      f"{TOTAL_ROWS * median_single:16,} {TOTAL_ROWS * mean_single:15,}")

print()
print("  NOTE Dolci's schema blocks are general-purpose APIs. Our pedagogy tools nest deeper")
print("  (Canvas rubric_assessment maps, Perplexity's 13 params), and multi-tool-select rows")
print("  offer 3-20 schemas, so the real mean will sit ABOVE this. Treat as a floor.")

print()
print("=== trainable tokens (what loss is actually computed on) ===")
print("  The schema block is masked, so only the assistant turn trains. Measured share on our")
print("  own three-row end-to-end sample was ~15%; on these real rows the schema block alone is")
if sys_only and all_tokens:
    share = 100.0 * stats.median(sys_only) / stats.median(all_tokens)
    print(f"  a median {share:.0f}% of the row, and it is entirely masked.")

print()
print("=== how this lands against a sequence length ===")
for seq in (2048, 4096, 8192, 16384, 32768):
    over_all = sum(1 for n in all_tokens if n > seq)
    over_single = sum(1 for n in single_turn if n > seq)
    print(f"  seq_len={seq:6,}  rows over limit: {over_all:3d}/100 all, "
          f"{over_single}/{len(single_turn)} single-turn")
