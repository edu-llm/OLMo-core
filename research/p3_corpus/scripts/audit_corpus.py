"""Flag low-value examples without removing them.

Writes `corpus/flags/<shard>.jsonl`, one line per flagged example:

    {"id": "9f2c1a4b7e03", "flags": ["tiny_target", "single_fact_short"], "tok": 214}

Nothing is deleted. The point is to make each class of problem countable, in
examples and in tokens, so the decision to drop a class is yours and reversible.
Use `scripts/filter_corpus.py` afterwards to act on a chosen set of flags.

The classes fall into three groups:

  SIZE — the example does not fit the window it will be trained in. These matter
  most for the ATP shards, whose targets run to thousands of tokens. If the block
  alone overruns the window the target is never seen at all, and the example is
  pure cost.

  DEGENERACY — the target does not require deriving anything: it is too short to
  carry a derivation, it is a list of premise names, or it is copied from the
  prompt. This is the class that would let the split arm win for the wrong
  reason, since the answer is sitting in the fact block.

  LEAKAGE — the goal appears verbatim in the block, so the example asks the model
  to restate something it was handed.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

WORD = re.compile(r"[A-Za-z_][\w.']*|\S")
ALNUM = re.compile(r"[A-Za-z0-9]")
# references rather than content: article-local Mizar labels and prover step ids
REFTOK = re.compile(r"\b(?:A\d+|Th\d+|Def\d+|Lm\d+|c_0_\d+|esk\d+_\d+)\b")
# tactic and proof-language scaffolding; present in every target, carries no
# mathematics, and would otherwise mask a target that is purely a citation
KEYWORD = re.compile(
    r"\b(?:by|from|using|unfolding|thus|hence|then|have|show|obtain|proof|end|"
    r"qed|with|also|finally|assume|let|simp|simp_all|auto|force|metis|meson|"
    r"blast|fastforce|arith|rule|intro|elim|dest|subst|induct|induction|cases|"
    r"of|add|del|symmetric|OF|THEN|where|and|is|be|st|holds|for|the)\b")


def norm(s):
    return " ".join(s.split())


def content_ratio(target, names):
    """Share of the target that is neither scaffolding nor a name from the block.

    A fact is listed in the block under its qualified name (`Regular_Set.concE`)
    but cited in the target by its short one (`concE`), so both forms have to go.
    """
    variants = set()
    for n in names:
        variants.add(n)
        variants.add(re.split(r"[.:]", n)[-1])
    rem = target
    for n in sorted(variants, key=len, reverse=True):
        if len(n) > 1:
            rem = rem.replace(n, " ")
    rem = KEYWORD.sub(" ", REFTOK.sub(" ", rem))
    a = len(ALNUM.findall(KEYWORD.sub(" ", target)))
    return (len(ALNUM.findall(rem)) / a) if a else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--big-window", type=int, default=8192)
    ap.add_argument("--tiny-target", type=int, default=12)
    ap.add_argument("--min-content", type=float, default=0.15)
    ap.add_argument("--copy-ngram", type=int, default=16)
    ap.add_argument("--copy-share", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--scan", default="shards",
                    help="which split to flag: shards or eval")
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    fd = os.path.join(a.corpus,
                      "flags" if a.scan == "shards" else f"flags_{a.scan}")
    os.makedirs(fd, exist_ok=True)

    grand = Counter()
    grand_tok = Counter()
    tot_n = tot_t = 0
    rows = []

    def chunks(path, size):
        """Stream records in batches — a 233M-token shard will not fit in 8 GB
        if its token lists are materialised whole."""
        buf = []
        with open(path) as f:
            for line in f:
                buf.append(json.loads(line))
                if len(buf) >= size:
                    yield buf
                    buf = []
        if buf:
            yield buf

    for s in sorted(glob.glob(os.path.join(a.corpus, a.scan, "*.jsonl"))):
        name = os.path.basename(s)[:-6]
        cnt = Counter()
        ctok = Counter()
        shard_tok = 0
        n_rec = 0
        with open(os.path.join(fd, f"{name}.jsonl"), "w") as fh:
          for recs in chunks(s, a.chunk):
            n_rec += len(recs)
            lens = []
            for key in ("text", "block", "target"):
                if key == "block":
                    docs = [r["text"][:r["mask_end"]] for r in recs]
                else:
                    docs = [r[key] for r in recs]
                lens.append([len(x) for x in
                             enc.encode_ordinary_batch(docs, num_threads=8)])
            for r, N, B, T in zip(recs, *lens):
                  shard_tok += N
                  f = []

                  if N > a.big_window:
                      f.append("over_8k")
                  elif N > a.window:
                      f.append("over_4k")
                  if B > a.window:
                      f.append("block_over_window")
                  if T < a.tiny_target:
                      f.append("tiny_target")
                  if N and B / N >= 0.9:
                      f.append("block_dominant")
                  if len(r["facts"]) == 1 and T < 30:
                      f.append("single_fact_short")

                  if content_ratio(r["target"], r["facts"]) < a.min_content:
                      f.append("name_echo")

                  g = norm(r["goal"])
                  if len(g) >= 40 and g in norm(r["text"][:r["mask_end"]]):
                      f.append("goal_in_block")

                  tw = WORD.findall(r["target"])
                  if len(tw) >= a.copy_ngram:
                      cw = WORD.findall(r["text"][:r["mask_end"]] + " " + r["goal"])
                      k = a.copy_ngram
                      cg = {tuple(cw[i:i + k]) for i in range(len(cw) - k + 1)}
                      tg = [tuple(tw[i:i + k]) for i in range(len(tw) - k + 1)]
                      if tg and sum(1 for x in tg if x in cg) / len(tg) >= a.copy_share:
                          f.append("high_copy")

                  if f:
                      fh.write(json.dumps({"id": r["id"], "flags": f,
                                           "tok": N}) + "\n")
                      for x in f:
                          cnt[x] += 1
                          ctok[x] += N

        flagged = sum(1 for _ in open(os.path.join(fd, f"{name}.jsonl")))
        rows.append((name, n_rec, shard_tok, flagged, cnt, ctok))
        grand.update(cnt)
        grand_tok.update(ctok)
        tot_n += n_rec
        tot_t += shard_tok
        print(f"  {name:<12}{n_rec:>8,} ex  {shard_tok/1e6:>6.1f}M tok   "
              f"{flagged:>7,} flagged ({100*flagged/max(n_rec,1):>4.1f}%)", flush=True)

    print(f"\n  {'TOTAL':<12}{tot_n:>8,} ex  {tot_t/1e6:>6.0f}M tok")

    print(f"\n{'flag':<20}{'examples':>10}{'% of ex':>9}{'tokens':>10}"
          f"{'% of tok':>10}  worst shards")
    for k, v in grand.most_common():
        top = sorted(((c[k], n) for n, _, _, _, c, _ in rows if c[k]),
                     reverse=True)[:2]
        where = ", ".join(f"{n} {c:,}" for c, n in top)
        print(f"{k:<20}{v:>10,}{100*v/tot_n:>8.1f}%{grand_tok[k]/1e6:>9.0f}M"
              f"{100*grand_tok[k]/tot_t:>9.1f}%  {where}")

    order = [k for k, _ in grand.most_common()]
    print(f"\nper shard\n{'':<12}" + "".join(f"{k[:9]:>10}" for k in order))
    for name, n, _, _, c, _ in rows:
        print(f"{name:<12}" + "".join(
            f"{(100*c[k]/max(n,1)):>9.0f}%" if c[k] else f"{'-':>10}"
            for k in order))

    uniq = 0
    utok = 0
    for name, _, _, _, _, _ in rows:
        for line in open(os.path.join(fd, f"{name}.jsonl")):
            uniq += 1
            utok += json.loads(line)["tok"]
    print(f"\n{'ANY FLAG':<20}{uniq:>10,}{100*uniq/tot_n:>8.1f}%{utok/1e6:>9.0f}M"
          f"{100*utok/tot_t:>9.1f}%")
    print(f"\nflags written to {fd}/  — nothing removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
