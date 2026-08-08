"""Show which upstream datasets carry a TAGGED license vs prose-only.

    python3 docs/tool-call/verify/verify_licenses.py

A "tagged" license is a `license:` key in the dataset card's YAML frontmatter, drawn from HF's
controlled vocabulary. The Hub parses it, renders a badge, exposes it as a `license:*` entry in the
API `tags`, and makes it filterable. A license mentioned only in the README prose is none of those
things -- nothing parses it, so it is a weaker claim about the uploader's intent.

This is the check behind the "Dolci licence sign-off" open question.
"""

import json
import urllib.request

DATASETS = [
    "Team-ACE/ToolACE",
    "NousResearch/hermes-function-calling-v1",
    "allenai/Dolci-Instruct-SFT-Tool-Use",
    "allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA",
    "Salesforce/xlam-function-calling-60k",
]

for name in DATASETS:
    url = f"https://huggingface.co/api/datasets/{name}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        print(f"{name}\n    UNREACHABLE: {e}\n")
        continue
    card = d.get("cardData") or {}
    tags = [t for t in d.get("tags", []) if t.startswith("license")]
    print(name)
    print(f"    gated                 : {d.get('gated')}")
    print(f"    frontmatter `license:` : {card.get('license', '<ABSENT>')}")
    print(f"    license:* in API tags  : {tags or '<NONE>'}")
    verdict = "TAGGED" if tags or card.get("license") else "PROSE-ONLY or ABSENT"
    print(f"    -> {verdict}")
    print()
