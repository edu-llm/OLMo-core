#!/usr/bin/env python
"""Generate first tutor turns from SEVERAL trained arms on ONE problem set.

``ORCD-SFT/generate_test_results.py`` answers "does SFT beat prompting?" — it fixes one
adapter and varies the *condition* (2x2: raw/sft x noSI/SI). This answers a different
question, "which implementation teaches better?", so it fixes the condition (``--si``) and
varies the *arm*. Run it twice, once per condition, on the same problem set and the SI-gating
contrast becomes paired instead of a comparison across two samples. Both feed the same judge: ``judge_pedagogy.py`` discovers setups from the data.

    python generate_arms.py \\
        --arm impl2_A1=/path/runs/A1/ckpt-923 \\
        --arm impl4_A3=/path/runs/A3/ckpt-923 \\
        --arm impl5_D4=/path/D4/ckpt-923 \\
        --out test_results_arms.jsonl

**The problem set is read, not rebuilt.** ``--problems`` defaults to the committed
``test_results_instruct.jsonl``, which already carries ``problem``, ``context``,
``gold_tutor`` and ``answer`` per record. Re-deriving them from the test split would risk a
different sample and break comparability with the published A/B/C/D table.

**``CANONICAL_SI`` is imported, not copied**, from ``generate_test_results.py`` (by path —
the dash in ``ORCD-SFT`` makes it unimportable). A second copy of that string that drifted
would silently change what every arm is being asked to do.

**Greedy, like the 2x2 generator.** ``do_sample=False``, so the only run-to-run variation is
numerical. Every arm is generated in one process on one device, so between-arm differences
cannot come from decoding or hardware.

``B_raw_SI`` (base + SI, prompt-only) is emitted by default because the PRD's definition of
done is stated against it, and because it anchors the judge's scale: a rubric score is only
interpretable next to something.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
DEFAULT_PROBLEMS = POC_ROOT / "test_results_instruct.jsonl"
GEN_2X2 = POC_ROOT / "ORCD-SFT" / "generate_test_results.py"
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"


def canonical_si() -> str:
    """The one copy of the test-time SI, loaded from the 2x2 generator by path."""
    spec = importlib.util.spec_from_file_location("p7_gen_2x2", GEN_2X2)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {GEN_2X2}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CANONICAL_SI


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=ADAPTER_DIR",
                   help="Repeatable. NAME becomes the setup key the judge sees.")
    p.add_argument("--problems", default=str(DEFAULT_PROBLEMS))
    p.add_argument("--out", default=str(HERE / "test_results_arms.jsonl"))
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--gen_max_new", type=int, default=220, help="Matches the 2x2 generator.")
    p.add_argument("--limit", type=int, default=0, help="Cap #problems (0 = all).")
    p.add_argument("--no_base", action="store_true", help="Skip the base reference cell.")
    p.add_argument("--si", choices=("canonical", "none"), default="canonical",
                   help="Whether the canonical pedagogy SI goes in context. 'canonical' is "
                        "the POC's cell D and the condition every KL/NLL number is measured "
                        "in; 'none' is cell C. Mirrors gen_pedagogy.py --si. Running both on "
                        "ONE problem set is what makes an SI-gating contrast paired rather "
                        "than a comparison across two samples.")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    return p.parse_args()


def pick_device(name: str) -> tuple[str, torch.dtype]:
    """bfloat16 where it is real, float16 on MPS.

    The training runs were bf16 on an L40S. MPS advertises bfloat16 but falls back to
    float32 for several ops, which is slower than float16 for no accuracy that matters to a
    greedy short generation being read by a judge. The dtype is recorded in the output so a
    reader is not left guessing which one produced the text.
    """
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda":
        return name, torch.bfloat16
    if name == "mps":
        return name, torch.float16
    return name, torch.float32


def main():
    args = parse_args()
    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm needs NAME=ADAPTER_DIR, got {spec!r}")
        name, _, path = spec.partition("=")
        if not Path(path, "adapter_config.json").exists():
            raise SystemExit(f"{path} has no adapter_config.json — not a PEFT adapter dir.")
        arms.append((name, path))

    rows = [json.loads(line) for line in open(args.problems, encoding="utf-8") if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    device, dtype = pick_device(args.device)
    SI = canonical_si() if args.si == "canonical" else None
    # The base cell is named for its 2x2 identity so a merged file can never silently put
    # cell A and cell B under one key.
    base_key = "B_raw_SI" if SI else "A_raw_noSI"

    print(f"problems : {len(rows)} from {args.problems}")
    print(f"arms     : {', '.join(n for n, _ in arms)}"
          f"{'' if args.no_base else f' (+ {base_key} reference)'}")
    print(f"device   : {device} / {dtype}")
    print(f"SI       : {args.si}" + (f", {len(SI)} chars, ends {SI[-40:]!r}" if SI else
                                      "  (cell C / A — no system message in context)"), flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    @torch.no_grad()
    def generate_turn(m, messages):
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(m.device)
        out = m.generate(
            **enc, max_new_tokens=args.gen_max_new, do_sample=False,   # greedy = reproducible
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    # Records keyed by (dialogue_id, turn) so each arm's pass fills the same rows. The judge
    # pairs on the problem, so a row that is missing one arm would silently drop out of every
    # paired contrast rather than erroring.
    recs = {}
    for r in rows:
        key = (r.get("dialogue_id"), r.get("turn"))
        # Copy every field except the outputs map rather than an allow-list, so problem-set
        # metadata (build_multiturn_set.py's depth / n_assistant_turns / problem_id) reaches
        # the judge instead of being silently dropped on the way through.
        recs[key] = {k: v for k, v in r.items() if k != "outputs"}
        recs[key]["outputs"] = {}

    print("\nloading base ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    base.config.use_cache = True
    base.eval()

    passes = ([] if args.no_base else [(base_key, None)]) + [(n, p) for n, p in arms]
    for name, adapter in passes:
        t0 = time.time()
        # One adapter attached at a time, then discarded. PeftModel.from_pretrained wraps the
        # SAME base weights, so stacking arms without unloading would compose their LoRAs and
        # silently score a model nobody trained.
        model = base if adapter is None else PeftModel.from_pretrained(base, adapter).to(device)
        model.eval()
        for i, key in enumerate(recs, 1):
            r = recs[key]
            msgs = ([{"role": "system", "content": SI}] if SI else []) + r["context"]
            r["outputs"][name] = generate_turn(model, msgs)
            if i % 4 == 0 or i == len(recs):
                print(f"  {name}: {i}/{len(recs)}  ({time.time() - t0:.0f}s)", flush=True)
        if adapter is not None:
            model = model.unload()          # strip the LoRA, restore pristine base weights
        del model

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs.values():
            r["gen_meta"] = {"device": device, "dtype": str(dtype), "greedy": True,
                             "si": args.si,
                             "base_model": args.base_model, "gen_max_new": args.gen_max_new}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    setups = sorted({s for r in recs.values() for s in r["outputs"]})
    print(f"\nwrote {len(recs)} records x {len(setups)} setups -> {out}")
    print(f"setups: {setups}")

    r = next(iter(recs.values()))
    print("\n" + "=" * 74)
    print("PROBLEM:", r["problem"][:200])
    print("GOLD   :", r["gold_tutor"][:200])
    for s in setups:
        print(f"\n----- {s} -----\n{r['outputs'][s][:300]}")


if __name__ == "__main__":
    main()
