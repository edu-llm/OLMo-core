"""Verify existence, access and licence tag for every candidate upstream source.

    python3 docs/tool-call/verify/verify_sources.py

Prints one line per dataset: access, the frontmatter `license:` tag (the only licence signal we
accept without a human read), and row count where the datasets-server will give one.
"""

import json
import urllib.parse
import urllib.request

CANDIDATES = [
    # general
    ("argilla/Synth-APIGen-v0.1", "general"),
    ("MadeAgents/xlam-irrelevance-7.5k", "general"),
    ("nvidia/When2Call", "general / answer-directly"),
    ("Agent-Ark/Toucan-1.5M", "general / nested-args"),
    ("Team-ACE/ToolACE", "general"),
    ("allenai/Dolci-Instruct-SFT-Tool-Use", "general (INCUMBENT — check the tag)"),
    # arithmetic
    ("MU-NLPC/Calc-gsm8k", "arithmetic"),
    ("openai/gsm8k", "arithmetic"),
    ("allenai/math_qa", "arithmetic"),
    ("ChilleD/SVAMP", "arithmetic"),
    ("nvidia/OpenMathInstruct-1", "arithmetic (license: other)"),
    # web-search
    ("aialt/RetrievalQA", "web-search / answer-directly"),
    ("xanhho/2WikiMultihopQA", "web-search"),
    ("ChilleD/StrategyQA", "web-search"),
    ("spacemanidol/orcas", "web-search (query prior)"),
    # pedagogy
    ("eth-nlped/mathdial", "pedagogy"),
    ("allenai/mathfish", "pedagogy (standards metadata)"),
    ("allenai/tutormoments-preview", "pedagogy"),
    ("Eedi/Question-Anchored-Tutoring-Dialogues-2k", "pedagogy (expect NC)"),
]

ACCEPTABLE = {"apache-2.0", "mit", "cc-by-4.0", "odc-by", "cc0-1.0", "bsd-3-clause"}


def api(path: str):
    with urllib.request.urlopen(f"https://huggingface.co/api/{path}", timeout=30) as r:
        return json.load(r)


def rows(ds: str):
    try:
        u = f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(ds, safe='')}"
        with urllib.request.urlopen(u, timeout=30) as r:
            d = json.load(r)
        return d["size"]["dataset"]["num_rows"]
    except Exception:
        return None


print(f"{'dataset':52s} {'access':10s} {'license tag':16s} {'rows':>10s}  verdict")
print("-" * 108)
for ds, _domain in CANDIDATES:
    try:
        d = api(f"datasets/{ds}")
    except Exception as e:
        code = getattr(e, "code", "ERR")
        print(f"{ds:52s} {'HTTP ' + str(code):10s} {'-':16s} {'-':>10s}  UNREACHABLE")
        continue
    gated = d.get("gated")
    private = d.get("private")
    access = "PRIVATE" if private else (f"gated:{gated}" if gated else "public")
    lic = (d.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = ",".join(lic)
    tag = lic or "<NO TAG>"
    n = rows(ds)
    if not lic:
        verdict = "REJECT (no frontmatter tag)"
    elif gated and gated != False:  # noqa: E712
        verdict = "REJECT (gated)"
    elif any(x in str(lic) for x in ("-nc", "-sa")):
        verdict = "REJECT (NC or share-alike)"
    elif str(lic) in ACCEPTABLE:
        verdict = "USABLE"
    else:
        verdict = "NEEDS HUMAN READ"
    print(f"{ds:52s} {access:10s} {tag:16s} {str(n or '?'):>10s}  {verdict}")

print()
print("=== Qwen3 generator: frontmatter tag vs the actual LICENSE file ===")
print("(Qwen2.5-72B is the cautionary case — tag and file disagreed)")
for model in ["Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen2.5-72B-Instruct",
              "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "allenai/Olmo-3-7B-Instruct"]:
    try:
        m = api(f"models/{model}")
        tag = (m.get("cardData") or {}).get("license", "<NO TAG>")
    except Exception as e:
        print(f"  {model:48s} API {getattr(e, 'code', 'ERR')}")
        continue
    head = "<unreadable>"
    for fname in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        try:
            u = f"https://huggingface.co/{model}/raw/main/{fname}"
            with urllib.request.urlopen(u, timeout=30) as r:
                body = r.read(400).decode("utf-8", "replace")
            head = " ".join(body.split())[:80]
            break
        except Exception:
            continue
    print(f"  {model}")
    print(f"      tag : {tag}")
    print(f"      file: {head}")
