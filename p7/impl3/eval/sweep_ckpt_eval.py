#!/usr/bin/env python
"""Per-CHECKPOINT eval across a sweep — the data behind Figure 3 of RL's Razor.

The sweep's normal eval only scores each run's FINAL checkpoint, which gives one point per
(variant, T). Figure 3 instead treats every mid-training checkpoint as a point and fits a curve
per series, so it needs all three axes measured at every checkpoint:

    x  new-task KL        KL(pi_0 || pi) on held-out pedagogy prompts   -> kl_new_SI
    y1 prior-task score   math retention, two prompt conditions         -> math_bare, math_hint
    y2 new-task perf      held-out pedagogy NLL (lower = learned more)  -> ped_nll

The math probe runs in BOTH conditions at every checkpoint, neither of which carries a pedagogy
system instruction:

    bare   the question alone
    hint   the question plus "Put your final answer inside \\boxed{ }"  (the POC's "nosi" run)

They are not interchangeable. On the same items an SFT checkpoint scores like the base when asked
bare and collapses when asked with the hint, because the hint collides with the tutor persona and
the model deflects into a counter-question instead of answering. Reporting one number would hide
the whole effect, so both are recorded along with the deflection and commit rates that explain it.

Doing that naively is ~100x more expensive than it needs to be, so three things are hoisted out
of the per-checkpoint loop, all of which are checkpoint-INDEPENDENT:

  1. the base model is loaded once, not once per checkpoint;
  2. the base's KL continuations are generated once (KL(pi_0||pi) samples from the BASE policy,
     so the continuation never depends on the checkpoint) — see common.kl.base_continuations;
  3. the base's math/IFEval answers are generated once instead of re-derived for every point.

Generation is also batched (left-padded), which is the difference between ~8 min and ~30 s per
checkpoint on an H200.

Results are appended to a JSONL as each checkpoint finishes and already-scored checkpoints are
skipped on restart, so a preempted job resumes instead of starting over.

    python eval/sweep_ckpt_eval.py --runs 'out/*' --out out/ckpt_sweep_eval.jsonl
"""
import argparse
import gc
import glob
import hashlib
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "math_eval"))

import torch  # noqa: E402

from common.kl import base_continuations, mean_kl_cached, pedagogy_contexts  # noqa: E402
from common.modeling import load_for_inference  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--runs", default="out/*", help="Glob of run dirs (relative to project root).")
    p.add_argument("--out", default="out/ckpt_sweep_eval.jsonl")
    p.add_argument("--val_file", default="data/socrateach_sft_val.jsonl")
    p.add_argument("--math_prompts", default="eval/math_eval/math_logic_prompts.jsonl")
    p.add_argument("--extra_ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="Score a standalone adapter dir that is not out/<run>/checkpoint-N, "
                        "e.g. --extra_ckpt poc-c923=checkpoint-923. Repeatable.")
    p.add_argument("--n_kl", type=int, default=64, help="Held-out prompts to average KL over.")
    p.add_argument("--n_nll", type=int, default=128, help="Held-out dialogues for the pedagogy NLL.")
    p.add_argument("--batch", type=int, default=16, help="Generation batch size.")
    p.add_argument("--gen_max", type=int, default=512, help="Max new tokens for the probes.")
    p.add_argument("--kl_gen_max", type=int, default=200)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--require_epoch", type=float, default=0.99,
                   help="Skip runs whose final checkpoint reports fewer epochs than this (0 = keep all).")
    return p.parse_args()


def measurement_protocol(args, math_rows):
    """Identifier for how the numbers in a row were measured.

    Successive fixes changed what these columns MEAN without changing their names: the KL context
    became the dialogue truncated before the first tutor turn (it was the whole finished dialogue),
    math went from one condition to bare+hint, and the item set went 45 mixed -> 250 GSM8K.
    Stamping the protocol on each row lets the resume logic tell "already scored" apart from
    "scored under different rules", which is otherwise undetectable downstream.

    The item ids are hashed rather than just counted: a 250-item set that is half BBH and one that
    is all GSM8K are wildly different probes that a count alone would call identical.

    The trailing 'ifeval=off' is frozen text. The IFEval probe has been removed, but every row
    already scored carries that token, and the resume logic compares protocol strings verbatim —
    dropping it would mark all existing rows stale and force a full rescore for no gain.
    """
    digest = hashlib.sha1(";".join(sorted(r["id"] for r in math_rows)).encode()).hexdigest()[:8]
    return f"kl=ctx-first-turn;math=bare+hint@{len(math_rows)}/{digest};ifeval=off"


def with_boxed_hint(row):
    hint = ("Put ONLY the letter of the correct option inside \\boxed{ }, e.g. \\boxed{C}."
            if row.get("answer_type") == "mc" else "Put your final answer inside \\boxed{ }.")
    return row["prompt"] + "\n\n" + hint


# --------------------------------------------------------------------------------------
# batched greedy generation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_batched(model, tok, device, prompts, *, batch=16, gen_max=512):
    """Greedy-decode a list of user prompts. Left-padded: right padding would put the pad run
    between the prompt and the first generated token and corrupt every non-longest row."""
    prev_side, tok.padding_side = tok.padding_side, "left"
    outs = []
    try:
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=gen_max, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
    finally:
        tok.padding_side = prev_side
    return outs


# --------------------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------------------
def score_math(responses, metas):
    from math_scoring import score
    return sum(score(r, m) for r, m in zip(responses, metas)) / len(metas) if metas else None


def math_stats(responses, metas, prefix):
    """Accuracy plus the two rates that say WHY it moved.

    A drop in accuracy has two very different causes that the raw number cannot distinguish: the
    model tried and got it wrong (skill loss), or it never committed to an answer at all and asked
    the user a question back (Socratic refusal). ``commit`` is the fraction that produced a parsable
    answer and ``deflect`` the fraction that ended on a question mark; ``acc_given_commit`` is the
    skill measured only over the attempts, which is what stays flat when the loss is pure refusal.
    """
    from math_scoring import extract, score

    n = len(metas)
    if not n:
        return {}
    correct = [score(r, m) for r, m in zip(responses, metas)]
    commit = [extract(r, m["answer_type"]) is not None for r, m in zip(responses, metas)]
    deflect = [(r or "").rstrip().endswith("?") for r in responses]
    nc = sum(commit)
    return {
        prefix: sum(correct) / n,
        f"{prefix}_commit": nc / n,
        f"{prefix}_deflect": sum(deflect) / n,
        f"{prefix}_acc_given_commit": (sum(c and k for c, k in zip(correct, commit)) / nc) if nc else None,
    }


@torch.no_grad()
def pedagogy_nll(model, items):
    """Mean per-token NLL of the gold tutor turns — new-task performance without a judge.

    Pedagogy quality is really a judge-scored quantity, but judging every checkpoint would mean
    thousands of graded responses. NLL on the held-out gold turns is the standard cheap stand-in:
    continuous, forward-pass only, and monotone in how well the tutor behavior has been learned.
    """
    tot, ntok = 0.0, 0
    for ids, labels in items:
        logits = model(ids).logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        if not mask.any():
            continue
        lp = torch.log_softmax(logits, dim=-1)
        picked = lp.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        tot += float(-(picked[mask]).sum())
        ntok += int(mask.sum())
    return tot / ntok if ntok else None


# --------------------------------------------------------------------------------------
# checkpoint discovery
# --------------------------------------------------------------------------------------
def final_epoch(run_dir):
    cks = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not cks:
        return None
    latest = max(cks, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    try:
        return json.load(open(latest)).get("epoch")
    except Exception:
        return None


def discover(runs_glob, require_epoch):
    """[(run, step, path)] for every checkpoint of every run that finished training."""
    out = []
    for run_dir in sorted(glob.glob(runs_glob)):
        if not os.path.isdir(run_dir):
            continue
        run = os.path.basename(run_dir.rstrip("/"))
        ep = final_epoch(run_dir)
        if ep is None:
            continue  # not a run dir at all (out/figures, stray output) — nothing to say about it
        if require_epoch and ep < require_epoch:
            print(f"skip {run}: final checkpoint reports epoch={ep} (< {require_epoch})")
            continue
        for ck in glob.glob(os.path.join(run_dir, "checkpoint-*")):
            m = re.search(r"checkpoint-(\d+)$", ck)
            if m:
                out.append((run, int(m.group(1)), ck))
    return sorted(out, key=lambda t: (t[0], t[1]))


def parse_extras(specs):
    """--extra_ckpt LABEL=PATH -> [(label, step, path)].

    Adapters that live outside the out/<run>/checkpoint-N layout — chiefly the POC's
    checkpoint-923, which is the lineage every comparison is against — are not reachable by the run
    glob and have no trainer_state.json for the epoch filter, so they are named explicitly.
    """
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--extra_ckpt needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"--extra_ckpt {label}: no such directory {path!r}")
        step = int(m.group(1)) if (m := re.search(r"checkpoint-(\d+)", path)) else 0
        out.append((label, step, path))
    return out


def variant_and_temp(run):
    m = re.match(r"impl3-([ab])-T([\d.]+)$", run)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def main():
    args = parse_args()
    os.chdir(ROOT)

    math_rows = [json.loads(l) for l in open(args.math_prompts, encoding="utf-8") if l.strip()]
    bare_prompts = [r["prompt"] for r in math_rows]
    hint_prompts = [with_boxed_hint(r) for r in math_rows]
    print(f"math probe: {len(math_rows)} prompts x 2 conditions (bare, hint), no pedagogy SI")


    val = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
    kl_items = pedagogy_contexts(val, args.n_kl)
    print(f"KL probe: {len(kl_items)} pedagogy contexts (truncated before the first tutor turn)")

    protocol = measurement_protocol(args, math_rows)
    print(f"protocol: {protocol}")

    todo = discover(args.runs, args.require_epoch) + parse_extras(args.extra_ckpt)
    done = set()
    if os.path.exists(args.out):
        stale = set()
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("protocol") == protocol:
                    done.add((r["run"], r["step"]))
                else:
                    stale.add(r.get("protocol"))
        # Resume-by-skipping is only valid within one measurement protocol. Rows scored under a
        # different one (e.g. before the KL context and boxed-hint fixes) are not comparable, and
        # silently appending to them would produce a file that mixes two definitions of the same
        # column — which no downstream plot could detect.
        if stale:
            raise SystemExit(
                f"{args.out} holds rows from a different protocol ({', '.join(str(s) for s in sorted(stale, key=str))}).\n"
                f"Current protocol is '{protocol}'. Those numbers are not comparable, so this run\n"
                f"would corrupt the file. Write somewhere new instead, e.g.:\n"
                f"    OUT={os.path.splitext(args.out)[0]}_v2.jsonl sbatch clusters/orcd/ckpt_sweep_eval.sbatch"
            )
        print(f"resuming: {len(done)} checkpoints already scored in {args.out}")
    todo = [t for t in todo if (t[0], t[1]) not in done]
    print(f"{len(todo)} checkpoints to score across {len({t[0] for t in todo})} runs")
    if not todo:
        return

    # ---- load base once and precompute everything that does not depend on the checkpoint ----
    base, tok, device = load_for_inference(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    from common.chat import make_tokenize_fn
    tok_fn = make_tokenize_fn(tok, args.max_len)
    nll_items = []
    for r in val[:args.n_nll]:
        ex = tok_fn(r)
        if any(t != -100 for t in ex["labels"]):
            nll_items.append((torch.tensor([ex["input_ids"]], device=device),
                              torch.tensor([ex["labels"]], device=device)))
    print(f"pedagogy NLL over {len(nll_items)} held-out dialogues")

    print("precomputing base KL continuations (once for the whole sweep) ...")
    cached_si = base_continuations(base, tok, kl_items, True, gen_max=args.kl_gen_max)
    cached_no = base_continuations(base, tok, kl_items, False, gen_max=args.kl_gen_max)

    print("precomputing base answers on the retention probes (once) ...")
    gen = lambda m, p: generate_batched(m, tok, device, p, batch=args.batch, gen_max=args.gen_max)
    base_stats = {**math_stats(gen(base, bare_prompts), math_rows, "math_bare"),
                  **math_stats(gen(base, hint_prompts), math_rows, "math_hint"),
                  "ped_nll": pedagogy_nll(base, nll_items)}
    # "Prior task score" for the Figure-3 x-axis. The hinted condition is the POC-parity probe and
    # the only one where these models visibly forget, so it is the headline; math_bare is kept
    # alongside it because the gap between the two IS the refusal effect.
    base_stats["prior_score"] = base_stats["math_hint"]
    print(f"base: {base_stats}")

    # The base row is the origin of every curve (KL = 0 by definition), so record it once.
    if ("base", 0) not in done:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run": "base", "step": 0, "variant": None, "temperature": None,
                                "kl_new_SI": 0.0, "kl_ped_noSI": 0.0, "protocol": protocol,
                                "math_bare_forget": 0.0, "math_hint_forget": 0.0,
                                **base_stats}) + "\n")

    # ---- per-checkpoint loop ----------------------------------------------------------------
    for i, (run, step, path) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {run} @ step {step}")
        try:
            sft, _, _ = load_for_inference(args.base_model, adapter_dir=path, merge=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (load failed): {e}")
            continue
        v, T = variant_and_temp(run)
        rec = {"run": run, "step": step, "variant": v, "temperature": T, "protocol": protocol,
               "kl_new_SI": mean_kl_cached(base, sft, cached_si),
               "kl_ped_noSI": mean_kl_cached(base, sft, cached_no)}

        rec.update(math_stats(gen(sft, bare_prompts), math_rows, "math_bare"))
        rec.update(math_stats(gen(sft, hint_prompts), math_rows, "math_hint"))
        rec["ped_nll"] = pedagogy_nll(sft, nll_items)
        # Forgetting is the drop from the base measured on the identical prompts.
        for cond in ("math_bare", "math_hint"):
            rec[f"{cond}_forget"] = base_stats[cond] - rec[cond]
        rec["prior_score"] = rec["math_hint"]

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  KL={rec['kl_new_SI']:.3f} "
              f"math bare={rec['math_bare']:.3f} hint={rec['math_hint']:.3f} "
              f"(deflect {rec['math_bare_deflect']:.2f}->{rec['math_hint_deflect']:.2f}) "
              f"ped_nll={rec['ped_nll']:.4f}")

        del sft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
