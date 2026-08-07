#!/usr/bin/env python
"""Generate the seven 7B Impl-3 arms on the multi-turn problem set, on the platform.

``generate_arms.py`` already does the generating and needs no changes to handle a 7B base —
``--base_model`` is a flag. This wrapper exists for the three things it cannot do:

1. **The adapters live in subfolders of one Hub repo**, and ``--arm NAME=DIR`` wants a local
   directory holding ``adapter_config.json``. So ``fetch`` snapshot-downloads each subfolder
   to its own directory first.
2. **The image has no AWS CLI.** An appended ``aws s3 cp`` is what killed the first impl3x5
   job, 40 minutes in. Output has to be written to ``$EDULLM_OUTPUT_PREFIX`` from inside
   Python, which is what ``put`` does.
3. **A run that only generates still has to satisfy the checkpoint contract.**
   ``olmo-core-train`` refuses any submission whose command text does not mention
   ``$EDULLM_CHECKPOINT_DIR``, so the flag is accepted and used — the results file is written
   there as well as to S3, because the two survive different failures.

    python run_gen7b.py --stages deps,fetch,gen,put \\
        --checkpoint_dir "$EDULLM_CHECKPOINT_DIR" --output_prefix "$EDULLM_OUTPUT_PREFIX"

**The arm keys are the contract with the judge**, not decoration: they become the ``outputs``
keys, which is what ``judge_pedagogy.py`` discovers setups from and what every contrast is
named after. They carry a ``_7b`` suffix so that merging the already-generated 1B outputs for
the same 300 problems into one judge batch cannot collide — the 1B family uses the bare names
(``SFT``, ``density_B_T2``, ...). Scores are only comparable *within* a judge batch, so that
merge is the only way to read the two scales against each other at all.

``B_raw_SI`` (7B base + SI, no adapter) is emitted by ``generate_arms.py`` by default and is
kept: it anchors the rubric scale, and it is the win-rate baseline the judge auto-detects.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The seven adapters, as ``(judge key, Hub subfolder)``. Order is the card's: control first,
#: then variant b coldest-to-warmest, then the two variant-a arms it calls non-candidates.
REPO = "meric533/socrateach-7b-impl3-adapters"
ARMS = [
    ("sft_control_7b", "sft-control"),
    ("b_T4_7b", "impl3-b-T4"),
    ("b_T2_7b", "impl3-b-T2"),
    ("b_T1_7b", "impl3-b-T1"),
    ("b_T05_7b", "impl3-b-T0.5"),
    ("a_T8_7b", "impl3-a-T8"),
    ("a_T4_7b", "impl3-a-T4"),
]

BASE_MODEL = "allenai/Olmo-3-7B-Instruct"

#: Same pins as ``impl3x5_klw/run_klw.py``. peft 0.20.0 in particular is the version the
#: adapters were written by (``peft_version`` in every ``adapter_config.json``).
PINS = ["transformers==5.14.1", "peft==0.20.0", "huggingface_hub==1.25.1",
        "accelerate==1.14.0", "numpy==2.4.6"]

ALL_STAGES = ("deps", "fetch", "smoke", "gen", "put")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="all", help=f"Comma-separated from {ALL_STAGES}.")
    p.add_argument("--problems", default=str(HERE / "multiturn_set.jsonl"),
                   help="The 300-item multi-turn set — the current eval's problem set.")
    p.add_argument("--adapters_root", default="/tmp/arms7b",
                   help="Where fetched subfolders land, one directory per arm.")
    p.add_argument("--out_name", default="arms_multiturn_7b.jsonl")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--si", choices=("canonical", "none"), default="canonical")
    p.add_argument("--limit", type=int, default=0, help="Cap #problems (0 = all). Smoke tests.")
    p.add_argument("--gen_max_new", type=int, default=220)
    p.add_argument("--checkpoint_dir", default=os.environ.get("EDULLM_CHECKPOINT_DIR") or None)
    p.add_argument("--output_prefix", default=os.environ.get("EDULLM_OUTPUT_PREFIX") or None)
    return p.parse_args()


def sh(cmd: str, cwd: Path | None = None) -> None:
    print(f"  $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd or HERE))


def s3_split(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key.rstrip("/")


def s3_put(local: Path, uri_prefix: str | None, name: str | None = None) -> None:
    """Upload, or do nothing off-platform. Mirrors ``run_klw.py``'s helper exactly."""
    if not uri_prefix or not Path(local).exists():
        return
    import boto3
    bucket, key = s3_split(uri_prefix)
    key = f"{key}/{name or Path(local).name}" if key else (name or Path(local).name)
    try:
        boto3.client("s3").upload_file(str(local), bucket, key)
        print(f"  s3 up   s3://{bucket}/{key} "
              f"({Path(local).stat().st_size / 2**20:.1f} MB)", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        print(f"  s3 FAIL s3://{bucket}/{key}: {exc}", flush=True)


def stage_deps() -> None:
    sh(f"{shlex.quote(sys.executable)} -m pip -q install "
       f"{' '.join(shlex.quote(p) for p in PINS)}")


def stage_fetch(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """One directory per arm, holding just that subfolder's two files.

    ``snapshot_download`` with ``allow_patterns`` rather than seven whole-repo clones: the
    repo is 2.1 GB and each arm needs 305 MB of it. ``local_dir`` is used so the paths handed
    to ``generate_arms.py`` are stable and readable in the log rather than blob hashes.
    """
    from huggingface_hub import snapshot_download

    root = Path(args.adapters_root)
    fetched = []
    for key, sub in ARMS:
        dest = root / key
        if (dest / "adapter_config.json").exists():
            print(f"  {key:<16} already on disk ({dest})", flush=True)
        else:
            snapshot_download(repo_id=REPO, allow_patterns=[f"{sub}/*"],
                              local_dir=str(root / "_repo"))
            src = root / "_repo" / sub
            dest.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    f.replace(dest / f.name)
            print(f"  {key:<16} <- {REPO}/{sub}", flush=True)
        if not (dest / "adapter_config.json").exists():
            raise SystemExit(f"{dest} has no adapter_config.json after fetch")
        fetched.append((key, dest))
    return fetched


def stage_gen(args: argparse.Namespace, arms: list[tuple[str, Path]], out: Path) -> None:
    """Hand off to ``generate_arms.py``, which is the thing that must stay shared.

    Invoked as a subprocess rather than imported so that the exact command lands in the log
    and can be re-run by hand off-platform. Every decode setting is that script's default;
    the only overrides here are the base model and the problem set.
    """
    arm_flags = " ".join(f"--arm {shlex.quote(f'{k}={d}')}" for k, d in arms)
    cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'generate_arms.py'))} "
           f"--base_model {shlex.quote(args.base_model)} "
           f"--problems {shlex.quote(args.problems)} "
           f"--out {shlex.quote(str(out))} "
           f"--si {args.si} --gen_max_new {args.gen_max_new} "
           f"{'--limit ' + str(args.limit) + ' ' if args.limit else ''}"
           f"{arm_flags}")
    sh(cmd)


def stage_smoke(args: argparse.Namespace, arms: list[tuple[str, Path]]) -> None:
    """Two problems through all eight passes, with assertions, before committing two GPU-hours.

    This exists because the 7B pipeline cannot be rehearsed off-platform — the base is 14 GB
    and the laptop that built this has 16 GB of RAM — so the platform run is the first real
    execution and a silent failure would be discovered only in the judge scores.

    The third assertion is the one that matters. If ``generate`` is handed a stop set that
    omits the token the chat template actually closes a turn with, nothing errors: every arm
    simply runs to ``--gen_max_new`` and keeps writing past its own turn, and the result reads
    as a badly-behaved tutor rather than as a broken harness. Catching it costs one minute
    here and is invisible two hours later. See ``generate_arms.stop_token_ids``.
    """
    import json

    from transformers import AutoTokenizer

    tmp = Path(args.adapters_root) / "_smoke.jsonl"
    smoke = argparse.Namespace(**{**vars(args), "limit": 2})
    stage_gen(smoke, arms, tmp)

    rows = [json.loads(ln) for ln in open(tmp, encoding="utf-8") if ln.strip()]
    if not rows:
        raise SystemExit("smoke: generator wrote no rows")
    expected = {k for k, _ in ARMS} | ({"B_raw_SI"} if args.si == "canonical" else
                                       {"A_raw_noSI"})
    tok = AutoTokenizer.from_pretrained(args.base_model)
    cap, at_cap, empty = args.gen_max_new, [], []

    for r in rows:
        missing = expected - set(r["outputs"])
        if missing:
            raise SystemExit(f"smoke: row {r.get('dialogue_id')} missing arms {sorted(missing)}")
        for arm, text in r["outputs"].items():
            if not text.strip():
                empty.append(arm)
            if len(tok(text, add_special_tokens=False)["input_ids"]) >= cap - 5:
                at_cap.append(arm)

    if empty:
        raise SystemExit(f"smoke: empty generations from {sorted(set(empty))}")
    if at_cap:
        raise SystemExit(
            f"smoke: {sorted(set(at_cap))} ran to the {cap}-token cap without stopping. "
            "That is the signature of a wrong stop set, not of a verbose model — check the "
            "'stop ids' line above against the base model's generation_config.")

    # All-identical outputs would mean the LoRAs never attached and every 'arm' is the base.
    for r in rows:
        if len({r["outputs"][k] for k, _ in ARMS}) == 1:
            raise SystemExit(f"smoke: all seven arms produced identical text on "
                             f"{r.get('dialogue_id')} — adapters are probably not applied")

    lens = {a: len(r["outputs"][a]) for r in rows[:1] for a in sorted(r["outputs"])}
    print(f"  smoke OK: {len(rows)} rows x {len(expected)} arms, none at cap, arms differ")
    print(f"  first-row output chars: {lens}", flush=True)
    tmp.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    stages = ALL_STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(ALL_STAGES)}")

    # Resolve before anything reads them. A relative path passes its own existence check
    # against the process cwd and then fails inside a subprocess running with cwd=HERE —
    # that trap cost impl3x5 a run that died in 3.8 seconds.
    args.problems = str(Path(args.problems).resolve())
    out_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / args.out_name

    print(f"stages   : {','.join(stages)}")
    print(f"base     : {args.base_model}")
    print(f"problems : {args.problems}")
    print(f"arms     : {', '.join(k for k, _ in ARMS)}  (+ B_raw_SI reference)")
    print(f"out      : {out}")
    print(f"s3       : {args.output_prefix or '(none — off platform)'}", flush=True)

    if not Path(args.problems).exists():
        raise SystemExit(f"problem set not found: {args.problems}")

    arms: list[tuple[str, Path]] = [(k, Path(args.adapters_root) / k) for k, _ in ARMS]
    for name in stages:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        if name == "deps":
            stage_deps()
        elif name == "fetch":
            arms = stage_fetch(args)
        elif name == "smoke":
            stage_smoke(args, arms)
        elif name == "gen":
            stage_gen(args, arms, out)
        elif name == "put":
            s3_put(out, args.output_prefix)
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
