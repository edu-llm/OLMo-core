"""Pre-tokenize NIAH needles offline, so no run needs a tokenizer or a network.

    python experiments/linear-attn-vs-gdn/make_needle_assets.py \
        --out experiments/linear-attn-vs-gdn/needle_assets.json

WHY THIS IS A SEPARATE, OFFLINE SCRIPT. An English needle-in-a-haystack needs a tokenizer to turn
"The special magic number for Barcelona is 8241." into ids. `TokenizerConfig` in this repository
carries only metadata, so getting a real one means `AutoTokenizer.from_pretrained` reaching
HuggingFace -- a public-internet fetch from inside a run whose whole claim is that it read a
sealed, audited corpus, over a network path nothing establishes exists.

None of that is necessary. The needle text is a handful of fixed templates and a small set of
keys and values; tokenizing it is a one-time job whose entire output is a few kilobytes of
integers. So this runs on a workstation, WHERE a tokenizer and a network are ordinary, and commits
the ids. The eval then reads a JSON and needs neither.

It also makes the tokenization auditable. The committed file records the tokenizer identifier, its
vocabulary size and the exact surface string beside every id list, so a reader can check that the
ids mean what the eval claims rather than trusting that they were produced correctly once.

WHY THREE VALUE KINDS. RULER's S-NIAH variants differ in how much prior the value carries, and
that changes what a score means:

  * ``digits``  -- "8241". The model has strong priors for numerals after "is", so retrieval is
    easiest here and a good part of any score is template completion rather than memory. This is
    the high-resolution condition and the one whose control matters most.
  * ``uuid``    -- "a3f9c1d0-...". High entropy, essentially no prior, which is RULER's S-NIAH-3
    and the honest test of whether a value was actually stored.
  * ``words``   -- an uncommon noun. Between the two.

Reporting all three separates "can retrieve" from "can guess", which a single condition cannot.
"""

import argparse
import json
import random
import sys
from typing import Any, Dict, List

TOKENIZER = "allenai/dolma2-tokenizer"

# One template, because the eval splices KEY and VALUE into a haystack and needs the surrounding
# tokens to be identical between the planted needle and the trailing query -- otherwise the query
# differs from the needle in ways other than position and the comparison is not clean.
NEEDLE = "The special magic number for {key} is {value}."
QUERY = "The special magic number for {key} is"

# MANY KEYS, BECAUSE THE CONTROL NEEDS A SAME-LENGTH ALTERNATIVE. The control query differs from
# the real one only in the key, and swapping in a key that tokenizes to a different number of
# tokens would change the sequence length -- so the control would differ from the real condition
# in length as well as in key, and `gain` would stop being a clean contrast. The eval therefore
# draws the K planted keys AND the control key from one group of equal token length, so it needs
# at least K+1 keys per group. With twelve keys the largest group held exactly five, which is the
# bare minimum at K=4 and no margin at all; this list is long enough that several groups clear it.
KEYS = [
    "Barcelona",
    "Reykjavik",
    "Montevideo",
    "Kathmandu",
    "Casablanca",
    "Tallinn",
    "Valparaiso",
    "Bratislava",
    "Antananarivo",
    "Vientiane",
    "Asuncion",
    "Ljubljana",
    "Marrakesh",
    "Surabaya",
    "Chittagong",
    "Guayaquil",
    "Novosibirsk",
    "Zanzibar",
    "Trondheim",
    "Salzburg",
    "Nagasaki",
    "Adelaide",
    "Winnipeg",
    "Bergen",
    "Cordoba",
    "Palermo",
    "Nantes",
    "Utrecht",
    "Aarhus",
    "Porto",
    "Kaunas",
    "Odense",
    "Turku",
    "Gdansk",
    "Rijeka",
    "Varna",
    "Mombasa",
    "Kigali",
    "Lusaka",
    "Windhoek",
    "Maputo",
    "Gaborone",
]

WORDS = [
    "clarinet",
    "obsidian",
    "marzipan",
    "quasar",
    "trellis",
    "vellum",
    "zephyr",
    "pumice",
    "gantry",
    "lichen",
    "spinnaker",
    "kudzu",
]


def make_values(kind: str, n: int, rng: random.Random) -> List[str]:
    if kind == "digits":
        return [f"{rng.randint(1000, 9999)}" for _ in range(n)]
    if kind == "uuid":
        return [
            "-".join("".join(rng.choice("0123456789abcdef") for _ in range(k)) for k in (8, 4, 4))
            for _ in range(n)
        ]
    if kind == "words":
        # sampled with replacement so the value count is not capped by the word list; the key
        # varies independently, so items stay distinct.
        return [rng.choice(WORDS) for _ in range(n)]
    raise ValueError(kind)


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-tokenize NIAH needles.")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--per-kind", type=int, default=42)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(opts.tokenizer)
    rng = random.Random(opts.seed)

    def enc(s: str) -> List[int]:
        return list(tok(s, add_special_tokens=False)["input_ids"])

    assets: Dict[str, Any] = {
        "tokenizer": opts.tokenizer,
        "vocab_size": int(tok.vocab_size),
        "needle_template": NEEDLE,
        "query_template": QUERY,
        "items": {},
    }

    for kind in ("digits", "uuid", "words"):
        vals = make_values(kind, opts.per_kind, rng)
        items = []
        for i, val in enumerate(vals):
            key = KEYS[i % len(KEYS)]
            needle_s = NEEDLE.format(key=key, value=val)
            query_s = QUERY.format(key=key)
            needle_ids, query_ids = enc(needle_s), enc(query_s)
            # THE ANSWER IS DEFINED AS THE TAIL OF THE NEEDLE AFTER THE QUERY PREFIX, not as
            # enc(value) on its own. Tokenizing " 8241" alone can merge differently than the same
            # characters in context, and scoring against a differently-merged id sequence would
            # mark a correct continuation wrong. Deriving it by subtraction guarantees the answer
            # is exactly what follows the query inside the real needle.
            if needle_ids[: len(query_ids)] != query_ids:
                print(f"  SKIP {kind} {key}/{val}: query is not a token-prefix of the needle")
                continue
            answer_ids = needle_ids[len(query_ids) :]
            items.append(
                {
                    "key": key,
                    "value": val,
                    "needle_text": needle_s,
                    "query_text": query_s,
                    "needle_ids": needle_ids,
                    "query_ids": query_ids,
                    "answer_ids": answer_ids,
                    "answer_text": tok.decode(answer_ids),
                }
            )
        assets["items"][kind] = items
        lens = [len(it["answer_ids"]) for it in items]
        print(
            f"{kind:7} {len(items):3} items, answer length {min(lens)}-{max(lens)} tokens, "
            f"e.g. {items[0]['needle_text']!r} -> answer {items[0]['answer_text']!r}"
        )

    # Every id must be inside the model's vocabulary, or the embedding lookup is out of range.
    all_ids = [i for k in assets["items"] for it in assets["items"][k] for i in it["needle_ids"]]
    assert max(all_ids) < assets["vocab_size"], (max(all_ids), assets["vocab_size"])
    print(f"\nmax id {max(all_ids)} < vocab {assets['vocab_size']}  OK")

    with open(opts.out, "w") as f:
        json.dump(assets, f, indent=1)
    print(f"wrote {opts.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
