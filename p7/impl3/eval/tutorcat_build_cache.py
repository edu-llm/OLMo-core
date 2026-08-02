#!/usr/bin/env python
"""Wire our pre-generated response shards into tutor_cat's CAT run, without touching their code.

tutor_cat's CAT is adaptive: it picks the next scenario from the current ability estimate, so it
expects to call a tutor live over an API. Every tutor it ships is an API client (`provider:
openai|anthropic|google`) — there is no offline provider, and our 18 models are local weights.

The seam is `CachedTutor` (tutor_cat/tutors.py), which every tutor is wrapped in:

    path = cache_dir / re.sub(r"[^A-Za-z0-9._-]", "_", model) / f"{scenario_id}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    ...  # only now does it hit the network

Pre-populating that cache for every (model, scenario) means the API client is constructed but
never called: the CAT engine reads our vLLM-generated text and drives item selection off it. No
fork of their repo, and their code stays updatable.

This writes the cache and a config.yaml whose `tutors:` block is our 18 models. Tutor `model`
strings are the exact ids recorded in the shards, because those are what determine the cache path.

    python eval/tutorcat_build_cache.py \
        --responses ~/tutorcat_runs/responses --benchmark TutorBench \
        --repo ~/olmo-eval-full/eduLLM-Evals --out-config ~/tutorcat_p7_config.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re


def safe_model(model_id: str) -> str:
    """Mirror CachedTutor's directory sanitiser exactly. If this drifts, every lookup misses and
    the run silently tries to call OpenAI with our local paths as model names."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


def short_label(model_id: str) -> str:
    """Readable tutor name for run dirs/plots. The full path is kept as `model`."""
    base = model_id.rstrip("/").split("/")[-1]
    return "base-instruct" if base.startswith("OLMo-2-") else base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", required=True, help="dir written by `tutor-cat generate`")
    p.add_argument("--benchmark", default="TutorBench",
                   help="which benchmark's shards to load; must match the scenarios/rubrics in the "
                        "config. TutorBench and Bridge use different skill spaces and cannot be mixed.")
    p.add_argument("--repo", required=True, help="eduLLM-Evals checkout (holds config.yaml)")
    p.add_argument("--cache-dir", default=None, help="default: <repo>/cache")
    p.add_argument("--out-config", required=True)
    p.add_argument("--judge-url", default="http://localhost:8000/v1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo = pathlib.Path(os.path.expanduser(args.repo))
    cache_dir = pathlib.Path(os.path.expanduser(args.cache_dir or (repo / "cache")))
    shards = sorted(glob.glob(os.path.expanduser(f"{args.responses}/{args.benchmark}/*.jsonl")))
    if not shards:
        raise SystemExit(f"no {args.benchmark} shards under {args.responses}")

    tutors, total, skipped = [], 0, 0
    for shard in shards:
        rows = [json.loads(line) for line in open(shard)]
        if not rows:
            print(f"[warn] empty shard {shard}")
            continue
        model_id = rows[0]["Model"]
        dest = cache_dir / safe_model(model_id)
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)

        written = 0
        for r in rows:
            text = r.get("Output") or ""
            # A blank response would be graded as a real (terrible) tutor turn rather than as a
            # missing datum, quietly dragging that model's ability down.
            if not text.strip():
                skipped += 1
                continue
            if not args.dry_run:
                (dest / f"{r['Scenario']}.json").write_text(
                    json.dumps({"model": model_id, "scenario_id": r["Scenario"], "response": text},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")
            written += 1
        total += written
        print(f"{short_label(model_id):16} -> {dest.name}  ({written} scenarios)")

        tutors.append({
            "name": short_label(model_id),
            "provider": "openai",       # never actually called; every scenario is a cache hit
            "model": model_id,          # determines the cache path, so it must match the shard
            "base_url": args.judge_url,
            "api_key_env": "JUDGE_API_KEY",
            "temperature": 0.0,
        })

    print(f"\n{len(tutors)} tutors, {total} cached responses"
          + (f", {skipped} blank responses skipped" if skipped else ""))

    import yaml
    cfg = yaml.safe_load(open(repo / "config.yaml"))
    cfg["tutors"] = tutors
    cfg["cache_dir"] = str(cache_dir)
    cfg["judge"]["base_url"] = args.judge_url
    cfg.setdefault("data", {})
    cfg["data"]["scenarios"] = f"data/{args.benchmark}/scenarios.jsonl"

    if args.dry_run:
        print("\n--- config (dry run, not written) ---")
        print(yaml.safe_dump({"tutors": tutors[:2], "data": cfg["data"]}, sort_keys=False))
        return

    out = pathlib.Path(os.path.expanduser(args.out_config))
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"config -> {out}")


if __name__ == "__main__":
    main()
