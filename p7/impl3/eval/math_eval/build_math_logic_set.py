"""Build the prior-task retention probe: 250 GSM8K problems with verifiable integer answers.

Deterministic by construction — final-answer exact match, no judge and no subagent.

WHY GSM8K ONLY. Three other sources were tried and all three sit on the floor for a 1B model,
where an item cannot show forgetting because there is no accuracy to lose:

    MATH-500   dropped first: ``expr`` answers need symbolic/LLM verification.
    AIME-2024  base scores  0.0%.
    BBH ld-7   base scores  6.7% — BELOW the 14.3% chance rate of its 7-way multiple choice, and
               every item is the same "seven objects in a fixed order" template under five
               cosmetic skins, so they do not even fail independently.

GSM8K is the only probe with real headroom and it carries the whole observed effect: on the full
250 items the base scores 66.4% under the boxed-answer hint while a vanilla SFT checkpoint falls to
21.2%, deflecting into a counter-question on 47.6% of items (the base never does).
Splitting the budget with a floor-level source just buys noise.

SIZE. 45 -> 250. At base's ~60% accuracy, 45 items resolved only a ~22-point gap at 80% power
while the config-to-config gaps of interest are 5-15; 250 resolves ~12. The original set's GSM8K
ids are retained as a subset so everything already scored stays comparable.
"""
import json, os, random
from collections import Counter

from datasets import load_dataset

N_GSM8K = 250
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "math_logic_prompts.jsonl")

random.seed(7)


def previously_used(prefix):
    """Ids from the 45-item set, so the old set stays a strict subset of the new one."""
    if not os.path.exists(OUT):
        return []
    keep = []
    for line in open(OUT):
        r = json.loads(line)
        if r["id"].startswith(prefix):
            keep.append(int(r["id"].rsplit("_", 1)[-1]))
    return keep


out = []

gsm = load_dataset("openai/gsm8k", "main", split="test")
idx = previously_used("gsm8k_")
pool = [i for i in range(len(gsm)) if i not in set(idx)]
idx += random.sample(pool, N_GSM8K - len(idx))
for i in idx:
    a = gsm[i]["answer"].split("####")[-1].strip().replace(",", "")
    out.append({"id": f"gsm8k_{i}", "source": "GSM8K", "category": "math",
                "difficulty": "easy", "prompt": gsm[i]["question"],
                "gold": a, "answer_type": "int"})

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("total:", len(out))
print("by source:", dict(Counter(r["source"] for r in out)))
print("by category:", dict(Counter(r["category"] for r in out)))
