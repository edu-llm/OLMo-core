#!/usr/bin/env python3
"""Assert the pre-campaign smoke actually honoured the contract.

A green Slurm state means the process exited 0. It does not mean checkpoints
were retained, the durable marker advanced, the resume cycle ran, the
fingerprint carried what it must, or that nothing leaked into $HOME. Every
one of those has failed silently at least once in this project's history, so
each gets an explicit check here.

Runs on FarmShare against a finished run directory. Standard library only:
FarmShare's plain `python3` has no numpy.

Usage: assert_smoke.py RUN_DIR [--expect-steps N]
Exit status: 0 if every check passed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, detail))
    return bool(ok)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expect-steps", type=int, default=None)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 1

    # The arm subdirectory name is built by train_job.sbatch; there is exactly
    # one per run directory.
    runs_root = run_dir / "runs"
    arms = sorted(p for p in runs_root.iterdir() if p.is_dir()) if runs_root.is_dir() else []
    if not check(len(arms) == 1, "exactly one arm directory", f"found {[p.name for p in arms]}"):
        report()
        return 1
    arm = arms[0]
    save_folder = arm / "checkpoints"
    progress = arm / "progress"

    # --- code identity (plan 10e / finding 1f) -------------------------
    provenance = load_json(run_dir / "code_provenance.json")
    if check(isinstance(provenance, dict), "code_provenance.json is readable", str(provenance)):
        check(
            provenance.get("dirty") is False,
            "code was a clean commit",
            f"dirty={provenance.get('dirty')!r} -- a dirty run is not poolable",
        )
        commit = str(provenance.get("commit", ""))
        check(
            len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "commit is a full sha",
            f"commit={commit!r}",
        )

    # --- fingerprint contents (findings 1e, 1f; plan section 2) --------
    fingerprint = load_json(progress / "current_fingerprint" / "run_fingerprint.json")
    identity = {}
    if check(isinstance(fingerprint, dict), "run_fingerprint.json is readable", str(fingerprint)):
        check(fingerprint.get("schema_version") == 2, "fingerprint schema is 2")
        identity = fingerprint.get("identity") or {}
        numerics = identity.get("numerics") or {}
        missing = [
            key
            for key in ("dp_name", "reduce_dtype", "ep_degree", "prefetch_factor", "float8")
            if key not in numerics
        ]
        check(not missing, "fingerprint carries every numerics knob", f"missing {missing}")
        check(
            isinstance(identity.get("task_loss_nproc"), int),
            "fingerprint pins the eval world size",
            f"task_loss_nproc={identity.get('task_loss_nproc')!r}",
        )
        definition = identity.get("difficulty_metric_definition")
        if identity.get("difficulty_metric"):
            check(
                isinstance(definition, dict) and "formula" in definition,
                "fingerprint carries the ratified metric definition",
                f"got {definition!r}",
            )

    # --- ladder retention (plan 10c) -----------------------------------
    expected = identity.get("checkpoint_steps")
    if args.expect_steps is not None:
        expected = [s for s in range(0, args.expect_steps + 1, 125)]
    if check(
        isinstance(expected, list) and expected,
        "expected ladder is known",
        f"got {expected!r}; pass --expect-steps to override",
    ):
        present = sorted(
            int(p.name[4:]) for p in save_folder.glob("step*") if p.name[4:].isdigit()
        )
        check(
            present == sorted(int(s) for s in expected),
            "KEEP_CHECKPOINTS=all retained the whole ladder",
            f"expected {sorted(int(s) for s in expected)}, found {present}",
        )
        for step in present:
            check(
                (save_folder / f"step{step}" / "run_fingerprint.json").is_file(),
                f"step{step} carries a fingerprint copy",
            )

    # --- durability marker (plan 10a/10d) ------------------------------
    marker = load_json(progress / "last_durable_step.json")
    if check(isinstance(marker, dict), "last_durable_step.json is readable", str(marker)):
        final = max(int(s) for s in expected) if isinstance(expected, list) and expected else None
        check(
            final is None or int(marker.get("last_durable_step", -1)) == final,
            "durable marker reached the final step",
            f"marker={marker.get('last_durable_step')!r}, final={final!r}",
        )
        check(
            marker.get("durability") == "local_scratch",
            "durability claims scratch only",
            f"got {marker.get('durability')!r}; 'local_scratch+wandb' would claim "
            f"a W&B replica that is no longer created",
        )
        # The one runtime-observable consequence of the upload policy.
        check(
            "checkpoint_artifact" not in marker,
            "no model artifact reference was recorded",
            f"found {marker.get('checkpoint_artifact')!r}",
        )

    # --- task-loss suite completed at every ladder step ----------------
    if isinstance(expected, list):
        for step in sorted(int(s) for s in expected):
            result = load_json(progress / "task_loss_results" / f"step{step}_task_loss.json")
            if not check(
                isinstance(result, dict),
                f"step{step} task-loss result is readable",
                str(result),
            ):
                continue
            labels = result.get("labels") or result.get("task_loss_bpb") or {}
            numeric = [
                v
                for v in labels.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            check(
                len(numeric) == 20,
                f"step{step} has all 20 raw bpb labels",
                f"got {len(numeric)}",
            )

    # --- $HOME containment (plan 10b) ----------------------------------
    # This used to assert that none of ~/.cache/huggingface, ~/.cache/wandb,
    # ~/.triton, ~/.nv/ComputeCache or ~/.config/wandb exists. That check was
    # not run-attributable and could never pass here: all five already exist
    # from earlier campaigns (17G in $HOME), and the account runs other jobs
    # concurrently, so neither existence nor mtime says anything about what
    # THIS run did. It would have reported a failure the run could not cause
    # and could not fix.
    #
    # The authoritative guard is in common.sh, and it is attributable: it
    # readlink -f's every redirected variable and exits 1 if any resolves
    # under $HOME, so a leak kills the job before it trains. What is left to
    # verify here is the positive half -- that the redirects actually took
    # effect rather than merely being exported -- which shows up as the
    # targets existing and being populated on scratch.
    # Note the two distinct cache locations, which are easy to conflate:
    # launch.sh sources common.sh ONCE at the top, so every redirected
    # variable (HF_HOME, TRITON_CACHE_DIR, WANDB_DIR, ...) is scoped to the
    # RUN directory, while the entrypoint is separately handed
    # `--cache-dir <arm>/cache` for its own resolved-metric and loader state.
    check(
        (arm / "cache").is_dir(),
        "the arm wrote its own cache on scratch",
        f"{arm / 'cache'} missing; --cache-dir was not honoured",
    )

    # Existence alone proves little here: common.sh mkdir -p's all fourteen
    # redirect targets before anything runs. What proves the redirect took
    # EFFECT is content appearing in them, and the tokenizer is the one
    # download every run must make.
    hub = run_dir / "cache" / "huggingface" / "hub"
    models = sorted(p.name for p in hub.glob("models--*")) if hub.is_dir() else []
    check(
        bool(models),
        "the eval's HF downloads landed on scratch, not in $HOME",
        f"no models--* under {hub}; HF_HOME was exported but not used",
    )
    check(
        (run_dir / "wandb").is_dir(),
        "wandb wrote its run directory on scratch",
        f"{run_dir / 'wandb'} missing; WANDB_DIR did not take effect",
    )

    # --- the hard-stop/resume cycle actually ran (plan 6a machinery) ---
    logs = sorted((run_dir / "logs").glob("train-*.out")) if (run_dir / "logs").is_dir() else []
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    check(
        "checkpoint boundary reached" in text,
        "the hard-stop/resume cycle ran at least once",
        "no restart observed; an intermediate checkpoint should force one",
    )
    check(
        "training run finished" in text,
        "the supervisor loop saw training finish",
    )
    check(
        "exceeded MAX_RESTARTS" not in text,
        "did not hit the restart ceiling",
    )

    return report()


def report() -> int:
    width = max(len(name) for _, name, _ in RESULTS) if RESULTS else 0
    failures = 0
    for ok, name, detail in RESULTS:
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name.ljust(width)}  {detail}")
    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} checks passed")
    if failures:
        print("SMOKE FAILED -- do not freeze the fingerprint or start Phase 0")
    else:
        print("smoke contract holds")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
